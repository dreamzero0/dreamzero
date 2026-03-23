"""
WanTextEncoder — UMT5-XXL 风格的文本编码器模块。

在 DreamZero 管线中属于 **Encoder 部分**（冻结），负责将 tokenized 文本序列编码为语义嵌入，
供 CausalWanModel 的交叉注意力使用。

架构层次:
    WanTextEncoder
    ├── token_embedding: nn.Embedding(vocab=256384, dim=4096)
    ├── pos_embedding: T5RelativeEmbedding (可选共享)
    ├── blocks: 24 × T5SelfAttention
    │   ├── norm1 + T5Attention (自注意力, 无缩放)
    │   ├── norm2 + T5FeedForward (门控 GELU FFN)
    │   └── (可选) 独立 T5RelativeEmbedding
    └── norm: T5LayerNorm (RMSNorm 变体)

数据流位置:
    DefaultDataCollator.collate()  →  text: (B, L_text) token IDs
                                        text_attention_mask: (B, L_text)
        ↓
    WANPolicyHead.encode_prompt()
        ↓
    WanTextEncoder.forward(ids, mask)  →  (B, L_text, 4096)
        ↓
    text_projection  →  (B, L_text, 5120)  →  作为 CausalWanModel 交叉注意力 context

典型维度 (UMT5-XXL): dim=4096, dim_attn=4096, dim_ffn=10240, num_heads=64, num_layers=24
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):
    """
    Tanh 近似的 GELU 激活函数。

    公式: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    用于 T5FeedForward 的门控分支。
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    """
    T5 RMSNorm：仅使用 RMS（均方根）归一化，无 bias/mean 中心化。

    公式: y = weight * x / sqrt(mean(x²) + eps)

    与标准 LayerNorm 不同，不减去均值，只除以 RMS，计算更高效。

    Inputs:
        x: (*, dim) — 任意前导维度，最后一维为 dim。
    Output:
        (*, dim) — 归一化后的张量，与输入同形。
    """

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    """
    T5 多头注意力（无缩放）。

    与标准 Scaled Dot-Product Attention 不同，T5 **不** 对 QK 乘积做 1/sqrt(d_k) 缩放，
    而是通过初始化 std 控制注意力权重的量级。

    当 context=None 时为自注意力；否则为交叉注意力。

    Inputs:
        x:        (B, L1, dim) — 查询序列。
        context:  (B, L2, dim) 或 None — 键值序列（None 时退化为自注意力）。
        mask:     (B, L2) 或 (B, L1, L2) 或 None — attention mask，0 表示屏蔽。
        pos_bias: (1, num_heads, L1, L2) 或 None — T5 相对位置偏置。
    Output:
        (B, L1, dim) — 注意力输出。

    被调用: T5SelfAttention.forward()
    """

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    """
    T5 门控 FFN (Gated GELU Feed-Forward Network)。

    结构: fc1(x) * gate(x) → dropout → fc2 → dropout
    其中 gate = Linear + GELU，fc1 = Linear（无激活），两路逐元素相乘实现门控。

    Inputs:
        x: (B, L, dim) — 输入特征。
    Output:
        (B, L, dim) — 与输入同形。

    被调用: T5SelfAttention.forward()
    """

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    """
    T5 Encoder Block：Pre-Norm 自注意力 + 门控 FFN + 残差连接。

    结构: x → Norm1 → T5Attention(self-attn + pos_bias) → + residual
          → Norm2 → T5FeedForward(gated GELU)         → + residual

    Inputs:
        x:        (B, L, dim) — 输入 token 特征。
        mask:     (B, L) 或 None — attention mask，0=屏蔽 padding。
        pos_bias: (1, num_heads, L, L) 或 None — 共享相对位置偏置（shared_pos=True 时从外部传入）。
    Output:
        (B, L, dim) — 与输入同形。

    被调用: WanTextEncoder.forward() 循环调用 24 次。
    """

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    """
    T5 相对位置编码：将相对距离映射到 bucket → 查 embedding 表 → 位置偏置。

    原理: 将相对位置 (q_pos - k_pos) 量化到 num_buckets 个桶中
    （近距离用精确桶，远距离用对数桶），然后查表得到每个 head 的 attention bias。

    Inputs (forward):
        lq: int — query 序列长度。
        lk: int — key 序列长度。
    Output:
        (1, num_heads, lq, lk) — 相对位置偏置，加到注意力 logits 上。

    被调用: WanTextEncoder.forward()（shared_pos=True 时在外层计算一次传给所有 block）
           或 T5SelfAttention.forward()（shared_pos=False 时每层独立计算）。
    """

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class WanTextEncoder(torch.nn.Module):
    """
    UMT5-XXL 风格的文本编码器（Encoder-only Transformer）。

    作用:
        将 tokenized 文本序列编码为上下文相关的语义嵌入向量。
        在 DreamZero 中属于 **Encoder 部分**，始终冻结（requires_grad=False）。

    原理:
        1. Token Embedding: token IDs → (B, L, dim=4096)
        2. 可选共享的 T5RelativeEmbedding: 计算相对位置偏置 (1, num_heads, L, L)
        3. 24 层 T5SelfAttention block:
           - Pre-Norm (RMSNorm) → 自注意力 (T5 无缩放) + 相对位置偏置 → 残差
           - Pre-Norm → 门控 GELU FFN → 残差
        4. 最终 RMSNorm + Dropout

    Inputs (forward):
        ids:  (B, L_text) — token IDs（由 UMT5 tokenizer 产生）。
        mask: (B, L_text) 或 None — attention mask，1=有效 token，0=padding。
    Output:
        (B, L_text, dim=4096) — 每个 token 位置的上下文嵌入。
        在 WANPolicyHead.encode_prompt() 中，padding 位置会被置零，
        然后经 text_projection 投影到 (B, L_text, 5120) 作为 DiT 交叉注意力的 context。

    上游: DefaultDataCollator.collate() 产生 text (B, L_text) + text_attention_mask (B, L_text)
    下游: WANPolicyHead.encode_prompt() → text_projection → CausalWanModel 交叉注意力 context

    预训练权重: models_t5_umt5-xxl-enc-bf16.pth
    冻结状态: requires_grad=False，始终不训练

    典型参数:
        vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
        num_heads=64, num_layers=24, num_buckets=32
    """

    def __init__(self,
                 vocab: int | nn.Embedding = 256384,
                 dim=4096,
                 dim_attn=4096,
                 dim_ffn=10240,
                 num_heads=64,
                 num_layers=24,
                 num_buckets=32,
                 shared_pos=False,
                 dropout=0.1,
                 text_encoder_pretrained_path: str=None):
        super(WanTextEncoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.text_encoder_pretrained_path = text_encoder_pretrained_path

        # layers
        if isinstance(vocab, int):
            self.token_embedding = nn.Embedding(vocab, dim)
        else:
            self.token_embedding = vocab
        if shared_pos:
            self.pos_embedding = T5RelativeEmbedding(
                num_buckets, num_heads, bidirectional=True)
        else:
            self.pos_embedding = None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        """
        前向编码。

        Args:
            ids:  (B, L_text) — token IDs。
            mask: (B, L_text) 或 None — attention mask (1=有效, 0=padding)。

        Returns:
            (B, L_text, dim=4096) — 每个位置的上下文语义嵌入。

        被调用: WANPolicyHead.encode_prompt()
        """
        x = self.token_embedding(ids)
        x = self.dropout(x)
        if self.shared_pos:
            assert self.pos_embedding is not None
            e = self.pos_embedding(x.size(1), x.size(1))
        else:
            e = None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
    @staticmethod
    def state_dict_converter():
        return WanTextEncoderStateDictConverter()
    
    
class WanTextEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        return state_dict