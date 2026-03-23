"""
Causal DiT for Video + Action (VLA): CausalWanModel and related modules.

This module implements a causal Diffusion Transformer that jointly models video latent
and action tokens with blockwise causal attention. Used for VLA (Vision-Language-Action)
policy learning with flow matching.

Main components:
- CategorySpecificLinear / CategorySpecificMLP: Per-embodiment (category) linear/MLP.
- MultiEmbodimentActionEncoder: Encodes (action, timestep) with embodiment-specific weights.
- causal_rope_action_apply_*: RoPE for video + action/state tokens (polar/no_polar).
- CausalWanSelfAttention: Blockwise causal self-attention over [first_image | image_blocks | action_blocks | state_blocks].
- CausalWanAttentionBlock: One DiT block (causal self-attn + cross-attn + FFN).
- CausalHead: Output head for video latent prediction.
- CausalWanModel: Full causal DiT with state/action encoder/decoder and video generation.
"""
from typing import Any, TypeAlias

# ===== 内部模块导入 =====
# AttentionModule: Flash Attention 封装，支持 causal/non-causal
from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
# SinusoidalPositionalEncoding: 正弦位置编码，用于时间步嵌入
# swish: swish 激活函数 x * sigmoid(x)
from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
# WanRMSNorm: RMS 归一化; rope_action_apply: 视频+action/state 的 RoPE 应用
# WanLayerNorm: 标准 LayerNorm; WAN_CROSSATTENTION_CLASSES: 交叉注意力类注册表
# rope_params: RoPE 频率预计算; MLPProj: CLIP 特征投影; sinusoidal_embedding_1d: 1D 正弦嵌入
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    rope_action_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
# FlexAttention: PyTorch 的灵活注意力 mask API（用于构建 BlockMask 稀疏 mask）
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention.flex_attention import BlockMask
# diffusers: Hugging Face 扩散模型基类，提供配置管理和模型序列化
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist
import os

# 是否启用 TensorRT 推理优化（影响 RoPE 实现选择: polar vs no_polar）
ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"


class CategorySpecificLinear(nn.Module):
    """
    按类别（embodiment）选择不同权重/偏置的线性层：y = x @ W[cat_ids] + b[cat_ids]。

    Inputs:
        x: (B, T, input_dim) — batch, sequence, input feature dim.
        cat_ids: (B,) — 每个样本的类别 id，用于索引 W 和 b。
    Output:
        (B, T, hidden_dim) — 与 x 的 batch/seq 维一致，最后一维为 hidden_dim。
    """
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        # W: (num_categories, input_dim, hidden_dim), b: (num_categories, hidden_dim)
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # selected_W: (B, input_dim, hidden_dim) — 按 cat_ids 选出每样本的权重
        selected_W = self.W[cat_ids]
        # selected_b: (B, hidden_dim)
        selected_b = self.b[cat_ids]
        # bmm(x, W): (B, T, input_dim) @ (B, input_dim, hidden_dim) -> (B, T, hidden_dim); 再加 broadcast 的 bias
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """
    按类别（embodiment）选权重的两层 MLP。

    结构: Linear(input_dim→hidden_dim) → ReLU → Linear(hidden_dim→output_dim)
    每层使用 CategorySpecificLinear，按 cat_ids 选择 embodiment 特定的权重矩阵。

    在 CausalWanModel 中有两个实例，分别扮演 **Encoder** 和 **Decoder** 角色：

    1. **state_encoder** (Encoder 角色):
       - input_dim=max_state_dim(64), hidden_dim=hidden_size(5120), output_dim=dim(5120)
       - 将归一化状态映射到 DiT 维度
       - 可训练，从头初始化

    2. **action_decoder** (Decoder Head #2: 动作噪声预测头):
       - input_dim=dim(5120), hidden_dim=hidden_size(5120), output_dim=action_dim(64)
       - 从 Backbone 输出的动作 token 段解码出动作空间噪声预测
       - 可训练，从头初始化

    作为 action_decoder 时的损失函数 (action_loss):
        MSE(action_noise_pred, training_target_action) × action_mask × has_real_action × training_weight
        其中 training_target_action = noise_action - actions（Flow Matching 速度场目标）。
        双重 mask: action_mask（维度有效性）+ has_real_action（样本级有效性）。

    作为 action_decoder 时的评估指标:
        - action_loss_avg: 训练日志 10 步滑动平均
        - open-loop MSE: scripts/open_loop_yam.py 离线评估

    Inputs:
        x: (B, T, input_dim) 或 (B, input_dim)。
        cat_ids: (B,) 类别 id（embodiment_id）。
    Output:
        (B, T, output_dim) 或 (B, output_dim)，与 x 的 batch/seq 维一致。

    被调用:
        作为 state_encoder: CausalWanModel._forward_train / _forward_blocks
        作为 action_decoder: CausalWanModel._forward_train / _forward_blocks
    """
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        # hidden: (B, T, hidden_dim) — 第一层线性 + ReLU
        hidden = F.relu(self.layer1(x, cat_ids))
        # (B, T, output_dim)
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """
    将 (actions, timesteps, cat_ids) 编码为 action embedding，支持多 embodiment（类别）的权重。

    流程：W1(actions) -> a_emb; pos_enc(timesteps) -> tau_emb; concat -> W2 -> swish -> W3 -> out.

    Inputs:
        actions: (B, T, action_dim) — 动作序列。
        timesteps: (B,) — 扩散时间步（标量 per batch），用于正弦位置编码。
        cat_ids: (B,) — embodiment 类别 id。
    Output:
        (B, T, hidden_size) — 与 actions 的 B、T 一致，最后一维为 hidden_size。
    """
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape

        # a_emb: (B, T, hidden_size) — 动作线性映射
        a_emb = self.W1(actions, cat_ids)

        # tau_emb: (B, hidden_size) 广播到 (B, T, hidden_size) — 时间步正弦编码
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # x: (B, T, 2*hidden_size) — 拼接后经 W2+swish+W3 -> (B, T, hidden_size)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        x = self.W3(x, cat_ids)
        return x


def causal_rope_action_apply(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index):
    """
    对 Q/K 应用因果 RoPE：视频 token 用 3D freqs，action/state token 用 1D freqs（按块对齐）。

    Inputs:
        x: (B, seq_len, num_heads, head_dim) — 实部/虚部交错，最后一维为复数形式 (d/2, 2)。
        freqs: 视频 3D RoPE 频率，形状与 seq 中视频部分对应。
        freqs_action, freqs_state: 1D RoPE 频率，用于 action/state 段。
        action_register_length: 当前块内 action+state token 数；None 表示无 action/state。
        num_action_per_block, num_state_per_block: 每块 action/state 数。
        action_state_index: 当前块索引，用于切片 freqs_action/freqs_state。
    Output:
        与 x 同 shape 的旋转后张量 (B, seq_len, num_heads, head_dim)。
    """
    if ENABLE_TENSORRT:
        return causal_rope_action_apply_no_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)
    else:
        return causal_rope_action_apply_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)


def causal_rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    """
    用实数 cos/sin 形式应用 RoPE（兼容 TensorRT，无 torch.polar）。
    x 最后一维为 head_dim，按 (d/2, 2) 解释为实部/虚部；旋转: (real, imag) @ (cos, -sin; sin, cos).
    """
    # x: (B, seq_len, n, d) -> 拆成实部/虚部 (B, seq_len, n, d/2, 2)
    B, seq_len, n, d = x.shape
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    # freqs: (seq_len', d/2, 2) -> view 成 (1, seq_len', 1, d/2, 2)，再拆 cos/sin
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos = freqs[..., 0]
    freqs_sin = freqs[..., 1]

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
            action_register_length, 1, -1, 2
        )
        freqs_cos_1d = freqs_1d[..., 0]
        freqs_sin_1d = freqs_1d[..., 1]
        freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_sin_1d], dim=0).unsqueeze(0)

    # 复数旋转: (real', imag') = (real*cos - imag*sin, real*sin + imag*cos)
    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos
    x_rotated = torch.stack((x_real_rotated, x_imag_rotated), dim=-1)
    return x_rotated.flatten(3)

def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    """
    用复数乘法应用 RoPE: x_rot = x * freqs（freqs 为预计算的 e^{i*theta}）。
    若有 action/state 段，将 freqs_action/freqs_state 按块拼到 freqs 序列后，再逐位相乘。
    Input x: (B, seq_len, n, head_dim)，head_dim 为复数形式 (d/2, 2)。
    Output: (B, seq_len, n, head_dim)。
    """
    B, seq_len, n, _ = x.shape
    # 转为复数: (B, seq_len, n, d/2, 2) -> complex
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


class CausalWanSelfAttention(nn.Module):
    """
    块级因果自注意力：序列布局为 [首帧 | 视频块 | 动作块 | 状态块]，
    按块规则计算 attention（首帧仅自注意；图像块可看首帧+过去/当前图像+当前动作/状态；动作块可看首帧+图像+当前状态+自身；状态块仅自注意）。
    支持 local_attn_size（时间窗口）、KV cache。Q/K 可选 RMSNorm，再应用 RoPE。
    """

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        """
        Args:
            dim: 隐藏维度（如 5120）
            num_heads: 注意力头数（如 40）
            frame_seqlen: 每帧 patch token 数（如 220）
            local_attn_size: 时间局部注意力窗口帧数，-1 为全局
            sink_size: attention sink 帧数（KV cache 滚动时保留的首部帧）
            num_frame_per_block: 每个注意力块包含的帧数（如 2）
            qk_norm: 是否对 Q/K 做 RMSNorm
            num_action_per_block: 每块 action token 数（如 32）
            num_state_per_block: 每块 state token 数（如 1）
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每头维度，如 128
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        # max_attention_size: KV cache 的最大 token 数（全局时为 21 帧；局部时为窗口大小）
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        # Q/K/V/O 投影层: (B, L, dim) -> (B, L, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)  # 输出投影
        # Q/K 归一化: RMSNorm 稳定注意力分数
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # Flash Attention 模块: 非因果（用于切片后的局部注意力）
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)
        # 因果 Flash Attention: causal=True 使用三角形 mask
        self.causal_attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim, causal=True)

    def _visualize_attention_mask(self, total_len, first_image_len, image_blocks_len, 
                                   action_len, state_len, num_image_blocks, 
                                   num_action_blocks, num_state_blocks,
                                   num_frame_per_block, frame_seqlen,
                                   num_action_per_block, num_state_per_block):
        """
        构造并返回块级因果注意力 mask 的稠密表示（调试/可视化用）。

        返回 mask: (total_len, total_len) bool 张量，mask[i,j]=True 表示 token i 可 attend token j。
        按照与 _blockwise_causal_flash_attn 完全相同的注意力规则构造。
        """
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Create mask tensor
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # First image: self-attention only
        mask[first_image_start:first_image_end, first_image_start:first_image_end] = True
        
        # Image blocks
        for block_idx in range(num_image_blocks):
            block_start = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            
            # Attend to first image
            mask[block_start:block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[block_start:block_end, image_kv_start:block_end] = True
            
            # Attend to current action block
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            mask[block_start:block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[block_start:block_end, state_block_start:state_block_end] = True
        
        # Action blocks
        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            
            # Attend to first image
            mask[action_block_start:action_block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            image_block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[action_block_start:action_block_end, image_kv_start:image_block_end] = True
            
            # Self-attention
            mask[action_block_start:action_block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[action_block_start:action_block_end, state_block_start:state_block_end] = True
        
        # State blocks: self-attention only
        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[state_block_start:state_block_end, state_block_start:state_block_end] = True
        
        return mask

    def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, num_frame_per_block=1, 
                                       action_horizon=None, state_horizon=None, 
                                       num_action_per_block=None, num_state_per_block=None,
                                       visualize_mask=False):
        """
        块级因果 Flash Attention 实现。不构造显式 mask 矩阵，而是通过切片 Q/K/V
        并多次调用 Flash Attention 来实现块级因果模式。

        序列结构: [首帧 (conditioning)] [图像块×N] [动作块×N] [状态块×N]

        注意力规则:
        - 首帧: 仅自注意力
        - 图像块 i: 可看首帧 + 所有之前图像块 + 当前图像块 + 当前动作块 i + 当前状态块 i
        - 动作块 i: 可看首帧 + 所有之前图像块 + 当前图像块 + 自身动作块 + 当前状态块 i
        - 状态块: 仅自注意力（conditioning，不看外部信息）

        Args:
            q, k, v: (B, L, num_heads, head_dim) — 经 RoPE 后的 Q/K 和原始 V
            frame_seqlen: 每帧的 patch token 数（如 DROID: 220）
            num_frame_per_block: 每个注意力块包含几帧
            action_horizon: action token 总数（None 表示无 action/state）
            state_horizon: state token 总数
            num_action_per_block: 每块 action token 数（如 32）
            num_state_per_block: 每块 state token 数（如 1）
        
        Returns:
            output: (B, L, num_heads, head_dim) — 与输入同 shape
        """
        b, total_len, n, d = q.shape
        
        has_action_state = (action_horizon is not None and state_horizon is not None)
        
        if not has_action_state:
            # ===== 纯视频块级因果注意力（无 action/state）=====
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block  # 每块 token 数
            num_blocks = (num_frames - 1) // num_frame_per_block  # 首帧除外的块数
            
            if num_blocks <= 0:
                # 序列太短，只有首帧 -> 全局 attention
                return self.attn(q, k, v)
            
            if self.local_attn_size == -1:
                # 全局注意力优化：一次 causal flash_attention 处理所有 token
                # causal_attn 内部使用 causal=True 的三角 mask
                return self.causal_attn(q, k, v)
            
            # 局部注意力：需要逐块处理，每块只看最近 local_attn_size 帧的 KV
            output = torch.empty_like(q)
            
            # 预计算每块的 Q 范围和 KV 范围
            block_starts = [frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            # kv_start: 截断到最近 local_attn_size 帧的起始位置
            kv_starts = [max(0, end - self.local_attn_size * frame_seqlen) for end in block_ends]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]
                
                # Flash Attention: Q=当前块, K/V=窗口内的所有历史+当前块
                # Q: (B, block_size, n, d), K/V: (B, window_size, n, d)
                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end]
                )
            
            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        # ===== 多模态序列结构: [首帧] [图像块] [动作块] [状态块] =====
        # 计算各段长度
        first_image_len = frame_seqlen                                         # 首帧 token 数
        action_len = action_horizon                                            # 全部 action token 数
        state_len = state_horizon                                              # 全部 state token 数
        image_blocks_len = total_len - first_image_len - action_len - state_len  # 图像块总 token 数
        
        # 各模态的块数（必须对齐: 图像块数 == 动作块数 == 状态块数）
        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block

        assert num_image_blocks == num_action_blocks == num_state_blocks
        
        # 计算每种模态在序列中的起止索引
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Visualize attention mask if requested
        if visualize_mask:
            mask = self._visualize_attention_mask(
                total_len, first_image_len, image_blocks_len, 
                action_len, state_len, num_image_blocks,
                num_action_blocks, num_state_blocks,
                num_frame_per_block, frame_seqlen,
                num_action_per_block, num_state_per_block
            )
            
            print("\n" + "="*80)
            print("ATTENTION MASK VISUALIZATION")
            print("="*80)
            print(f"Total length: {total_len}")
            print(f"First image: [{first_image_start}:{first_image_end}] (len={first_image_len})")
            print(f"Image blocks: [{image_blocks_start}:{image_blocks_end}] (len={image_blocks_len}, num_blocks={num_image_blocks})")
            print(f"Action tokens: [{action_start}:{action_end}] (len={action_len}, num_blocks={num_action_blocks})")
            print(f"State tokens: [{state_start}:{state_end}] (len={state_len}, num_blocks={num_state_blocks})")
            print(f"Local attention size: {self.local_attn_size}")
            print("-"*80)
            
            # Print a downsampled version of the mask if it's too large
            if total_len <= 100:
                # Print full mask for small sequences
                print("Attention mask (1=can attend, 0=cannot attend):")
                print("Rows=Query tokens, Cols=Key tokens")
                for i in range(total_len):
                    row = "".join(["1" if mask[i, j] else "." for j in range(total_len)])
                    print(f"{i:4d}: {row}")
            else:
                # Print downsampled version for large sequences
                downsample = max(1, total_len // 100)
                print(f"Attention mask (downsampled by {downsample}x):")
                print("Rows=Query tokens, Cols=Key tokens (1=can attend, .=cannot attend)")
                for i in range(0, total_len, downsample):
                    row = "".join(["1" if mask[i, j] else "." for j in range(0, total_len, downsample)])
                    print(f"{i:4d}: {row}")
            
            # Save mask as image
            try:
                import cv2
                import numpy as np
                mask_np = mask.cpu().float().numpy()
                # Resize for visualization if needed
                if total_len > 1000:
                    mask_np = cv2.resize(mask_np, (1000, 1000), interpolation=cv2.INTER_NEAREST)
                mask_img = (mask_np * 255).astype(np.uint8)
                cv2.imwrite("attention_mask_blockwise_flash.png", mask_img)
                print(f"\nMask saved to: attention_mask_blockwise_flash.png")
            except Exception as e:
                print(f"Could not save mask image: {e}")
            
            print("="*80 + "\n")
        
        # 预分配输出张量，避免 list.append + cat 的开销
        output = torch.empty_like(q)
        
        # ===== 首帧: conditioning，仅自注意力 =====
        # Q/K/V 都只取首帧范围，FlashAttention 无额外 mask
        output[:, first_image_start:first_image_end] = self.attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end]
        )
        
        # 预计算所有块的起止索引，减少循环内开销
        image_block_starts = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        image_block_ends = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        if self.local_attn_size != -1:
            # 局部注意力：每块只看最近 local_attn_size 帧窗口
            image_kv_starts = [max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends]
        else:
            # 全局注意力：从图像块起始位置开始
            image_kv_starts = [image_blocks_start] * num_image_blocks
        
        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]
        
        # ===== 图像块 i: 可看首帧 + 之前图像块 + 当前图像块 + 当前动作块 + 当前状态块 =====
        for block_idx in range(num_image_blocks):
            block_start = image_block_starts[block_idx]
            block_end = image_block_ends[block_idx]
            image_kv_start = image_kv_starts[block_idx]
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # 拼接 KV context: [首帧 | 历史+当前图像块 | 当前动作块 | 当前状态块]
            # 通过 torch.cat 在 seq 维度拼接，FlashAttention 对整个 context 做全连接
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],       # 首帧 KV
                k[:, image_kv_start:block_end],                # 历史到当前的图像块 KV
                k[:, action_block_start:action_block_end],     # 当前动作块 KV
                k[:, state_block_start:state_block_end]        # 当前状态块 KV
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            # Q 只取当前图像块，K/V 为上面拼好的 context
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )
        
        # ===== 动作块 i: 可看首帧 + 历史图像块 + 当前图像块 + 自身动作 + 当前状态 =====
        for block_idx in range(num_action_blocks):
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            image_block_end = image_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            
            # 拼接 KV context: [首帧 | 历史+当前图像块 | 自身动作块 | 当前状态块]
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],
                k[:, image_kv_start:image_block_end],
                k[:, action_block_start:action_block_end],
                k[:, state_block_start:state_block_end]
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:image_block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )
        
        # ===== 状态块: conditioning，仅自注意力（不看外部信息）=====
        for block_idx in range(num_state_blocks):
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end]
            )
        
        return output

    def _process_clean_image_only(self, clean_image_q, clean_image_k, clean_image_v, clean_frames):
        """
        处理 Teacher Forcing 中 clean（干净）图像 token 的块级因果注意力。

        注意力规则:
        - 首帧: conditioning，仅自注意力
        - 块 i: 可看首帧 + 之前所有块 (0..i-1) + 当前块 i

        优化: 全局注意力时（local_attn_size==-1），用一次 causal flash_attention
        代替逐块循环，大幅提速。

        Args:
            clean_image_q/k/v: (B, clean_image_seq_len, n, d)，clean 图像的 Q/K/V
            clean_frames: 干净帧总数
        Returns:
            output: (B, clean_image_seq_len, n, d)
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            # Only first frame - single attention call
            return self.attn(
                clean_image_q[:, :self.frame_seqlen],
                clean_image_k[:, :self.frame_seqlen],
                clean_image_v[:, :self.frame_seqlen]
            )
        
        # Pre-allocate output tensor (avoids list append + cat overhead)
        b, total_len, n, d = clean_image_q.shape
        output = torch.empty_like(clean_image_q)
        
        # First frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            clean_image_q[:, :self.frame_seqlen],
            clean_image_k[:, :self.frame_seqlen],
            clean_image_v[:, :self.frame_seqlen]
        )
        
        # OPTIMIZATION: Process all blocks together with causal masking
        # For global attention (no local_attn_size), we can process all blocks in one call
        if self.local_attn_size == -1:
            # Single attention call for all blocks!
            # Each position can attend to first_frame + everything up to itself
            blocks_q = clean_image_q[:, self.frame_seqlen:]
            blocks_k = clean_image_k  # Can attend to everything including first frame
            blocks_v = clean_image_v
            
            # Use causal masking: each block token can see first frame + all previous tokens
            output[:, self.frame_seqlen:] = self.causal_attn(
                blocks_q, blocks_k, blocks_v
            )
        else:
            # With local attention, we still need to loop but with optimizations
            # Pre-compute all block boundaries to reduce overhead
            block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                
                q_block = clean_image_q[:, block_start:block_end]
                
                # Context: first frame + recent blocks within local_attn_size
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                k_context = torch.cat([
                    clean_image_k[:, :self.frame_seqlen],  # First frame
                    clean_image_k[:, image_kv_start:block_end]  # Recent blocks + current
                ], dim=1)
                v_context = torch.cat([
                    clean_image_v[:, :self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end]
                ], dim=1)
                
                output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
        """
        处理状态块: 每个状态块仅做自注意力（独立 conditioning，不看外部信息）。

        各状态块互相独立，理论上可并行处理。当前实现逐块循环。
        每块大小 = num_state_per_block（通常为 1），所以 attention 退化为恒等操作。

        Args:
            state_q/k/v: (B, state_horizon, n, d)，所有状态 token 的 Q/K/V
            state_horizon: 状态 token 总数
        Returns:
            output: (B, state_horizon, n, d)
        """
        num_blocks = state_horizon // self.num_state_per_block
        
        if num_blocks == 1:
            # Single block - one attention call
            return self.attn(state_q, state_k, state_v)
        
        # OPTIMIZATION: Since each state block only attends to itself (no cross-block attention),
        # we can process all blocks in a single batched call. Flash attention will handle this
        # efficiently. The blocks are independent, so this is safe.
        # Alternative: reshape and process as separate batch items
        
        # Pre-allocate output
        output = torch.empty_like(state_q)
        
        # Process all blocks (keeping loop for now due to block-diagonal pattern)
        # This could be further optimized with custom masking
        for block_idx in range(num_blocks):
            state_block_start = block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            
            output[:, state_block_start:state_block_end] = self.attn(
                state_q[:, state_block_start:state_block_end],
                state_k[:, state_block_start:state_block_end],
                state_v[:, state_block_start:state_block_end]
            )
        
        return output
    
    def _process_noisy_image_blocks(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                     clean_image_k, clean_image_v,
                                     noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                                     half_frames, action_horizon, state_horizon):
        """
        处理 Teacher Forcing 中 noisy（加噪）图像块的注意力。

        注意力规则:
        - noisy 首帧: 仅自注意力（conditioning）
        - noisy 块 i: 可看 clean 首帧 + clean 块[0:i] + 当前 noisy 块 + noisy action[i] + noisy state[i]

        这使得 noisy 图像块能利用 clean 图像作为条件上下文，
        同时也能看到同步的 action/state 信息。

        Args:
            noisy_image_q/k/v: (B, noisy_image_seq_len, n, d)
            clean_image_k/v: (B, clean_image_seq_len, n, d) — clean 半的 K/V（供 noisy 查看）
            noisy_action_k/v: (B, action_horizon, n, d) — noisy action 的 K/V
            noisy_state_k/v: (B, state_horizon, n, d) — noisy state 的 K/V
            half_frames: noisy 半的帧数
        Returns:
            output: (B, noisy_image_seq_len, n, d)
        """
        # block_size: 每个图像块的 token 数 = frame_seqlen × num_frame_per_block
        block_size = self.frame_seqlen * self.num_frame_per_block
        # num_blocks: 除首帧外的块数（首帧单独处理）
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        output = torch.empty_like(noisy_image_q)
        
        # noisy 首帧: conditioning，仅自注意力
        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            noisy_image_k[:, :self.frame_seqlen],
            noisy_image_v[:, :self.frame_seqlen]
        )
        
        if num_blocks == 0:
            return output
        
        # 预计算各块的索引范围
        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
        # clean_context_ends[i]: noisy 块 i 可看的 clean 图像范围 [0, clean_end)
        # 即 clean 首帧 + clean 块 [0, i-1]（不包含 clean 块 i，因为 i 对应 noisy 同一时间位置）
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            # Q: 当前 noisy 图像块 (B, block_size, n, d)
            q_block = noisy_image_q[:, noisy_start:noisy_end]
            
            # K/V context: [clean首帧+clean历史块 | 当前noisy图像块 | noisy action[i] | noisy state[i]]
            # clean_image_k[:, :clean_end]: clean 首帧 + clean 块 0..i-1 的 K
            # noisy_image_k[:, noisy_start:noisy_end]: 当前 noisy 图像块的 K（自注意力部分）
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            # FlashAttention: Q=noisy图像块, K/V=拼接的context (无mask，全连接)
            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_noisy_action_blocks(self, noisy_action_q, noisy_action_k, noisy_action_v,
                                      clean_image_k, clean_image_v,
                                      noisy_image_k, noisy_image_v,
                                      noisy_state_k, noisy_state_v,
                                      half_frames, action_horizon, state_horizon):
        """
        处理 Teacher Forcing 中 noisy 动作块的注意力。

        注意力规则:
        - 动作块 i: 可看 clean 首帧 + clean 块[0:i] + noisy 图像块[i] + 自身动作块[i] + state[i]

        这使得 noisy 动作块能利用 clean 图像上下文和同步的 noisy 图像/状态信息。

        Args:
            noisy_action_q/k/v: (B, action_horizon, n, d)
            clean_image_k/v: (B, clean_image_seq_len, n, d) — clean 图像的 K/V
            noisy_image_k/v: (B, noisy_image_seq_len, n, d) — noisy 图像的 K/V
            noisy_state_k/v: (B, state_horizon, n, d)
            half_frames: noisy 半的帧数
        Returns:
            output: (B, action_horizon, n, d)
        """
        # 除首帧外的块数（首帧对应的 action 不做额外处理）
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            return torch.empty_like(noisy_action_q)
        
        output = torch.empty_like(noisy_action_q)
        
        # 预计算各块的索引
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        # clean_context_ends[i]: 动作块 i 可看的 clean 图像范围 = clean首帧 + clean块[0:i-1]
        clean_context_ends = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        # noisy 图像块 i 的范围（动作块 i 可看对应的 noisy 图像块）
        noisy_image_block_starts = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_ends = [start + self.frame_seqlen * self.num_frame_per_block for start in noisy_image_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        for block_idx in range(num_blocks):
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            noisy_img_start = noisy_image_block_starts[block_idx]
            noisy_img_end = noisy_image_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            # Q: 当前 noisy 动作块 (B, num_action_per_block, n, d)
            q_block = noisy_action_q[:, action_start:action_end]
            
            # K/V context: [clean首帧+clean历史块 | noisy图像块[i] | 自身action[i] | state[i]]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],                   # clean 首帧 + clean 历史图像块
                noisy_image_k[:, noisy_img_start:noisy_img_end], # 对应的 noisy 图像块
                noisy_action_k[:, action_start:action_end],      # 自身 action 块（自注意力）
                noisy_state_k[:, state_start:state_end]          # 对应的 state 块
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_img_start:noisy_img_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)
        
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        因果自注意力前向。

        Args:
            x: (B, L, C) — 输入序列（视频+可选 action/state tokens），C=dim。
            freqs: 视频 3D RoPE 频率。
            freqs_action, freqs_state: action/state 的 1D RoPE 频率。
            action_register_length: action+state token 总数；None 表示无。
            kv_cache: 推理时每层的 KV cache；训练为 None。
            current_start_frame: 推理时当前起始帧（用于 RoPE 偏移）。
            is_tf: 是否 teacher forcing 模式（clean+noisy 两半）。
        Returns:
            output: (B, L, C) — 与 x 同 shape。
            updated_kv_cache: 更新后的 KV cache 或 None。
        """
        # b,s,n,d: batch, seq_len, num_heads, head_dim
        # b=batch, s=总序列长度, n=num_heads, d=head_dim=dim//num_heads
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            """
            将输入 x: (B, s, dim) 分别投影为 Q, K, V，并 reshape 为多头格式。
            Q, K 额外经过 RMSNorm（qk_norm=True 时）。
            返回 q, k, v 各为 (B, s, num_heads, head_dim)。
            """
            # self.q: nn.Linear(dim, dim); norm_q: WanRMSNorm 或 Identity
            # (B, s, dim) -> Linear -> (B, s, dim) -> Norm -> (B, s, dim) -> view -> (B, s, n, d)
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            # V 不做 norm，仅线性投影
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        updated_kv_cache: torch.Tensor | None = None

        # ===== 训练路径（无 KV cache）=====
        if kv_cache is None:
            if is_tf:
                # Teacher Forcing 训练模式：序列前半为 clean（干净），后半为 noisy（加噪）
                # 两半共享相同的 RoPE 频率（因为它们代表相同时间位置的不同噪声水平）
                if action_register_length is not None:
                    # 有 action/state 时：noisy 半包含 [视频token; action; state]
                    # 序列布局: [clean 视频] [noisy 视频 | action | state]
                    # 纯视频 token 数 = (总长 - action_register_length) / 2
                    q_context = q[:, :(s-action_register_length)//2]
                    k_context = k[:, :(s-action_register_length)//2]
                    q_noisy = q[:, (s-action_register_length)//2:]  
                    k_noisy = k[:, (s-action_register_length)//2:]
                else:
                    # 无 action：前半 clean，后半 noisy，对称切分
                    q_context = q[:, :s//2]
                    k_context = k[:, :s//2]
                    q_noisy = q[:, s//2:]
                    k_noisy = k[:, s//2:]
                roped_query = []
                roped_key = []

                # 对 clean 和 noisy 部分分别应用 RoPE（位置频率相同）
                # rope_action_apply: 对视频 token 用 3D RoPE (freqs)，
                #   对 action/state token 用 1D RoPE (freqs_action/freqs_state)
                rq_context = rope_action_apply(
                    x=q_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)
                rk_context = rope_action_apply(
                    x=k_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)

                rq_noisy = rope_action_apply(
                    x=q_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                rk_noisy = rope_action_apply(
                    x=k_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                # 将 [clean_context; noisy] 两部分 RoPE 后的 Q/K 拼回完整序列
                roped_query.append(rq_context)
                roped_key.append(rk_context)
                roped_query.append(rq_noisy)
                roped_key.append(rk_noisy)

                # roped_query/roped_key: (B, s, n, d)，与原始 q/k 同 shape
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                # half_seq_len: 纯视频 token 数的一半（clean 或 noisy 中视频部分的 token 数）
                half_seq_len = (s - (action_register_length if action_register_length is not None else 0)) // 2
                
                if action_register_length is not None:
                    # ===== 有 action/state 的 Teacher Forcing =====
                    # 序列布局:
                    #   前 half_seq_len: [clean 视频 token]（只有视频，无 action/state）
                    #   后: [noisy 视频 token (half_seq_len)] [action token] [state token]
                    # 因果性仅对图像块施加！
                    
                    # clean 半：纯视频 token
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen  # 干净帧数
                    
                    # noisy 半：视频 + action + state
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    # 图像块数 = (帧数 - 1) / num_frame_per_block，减 1 是因为首帧独立处理
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block
                    
                    # 切分 clean 部分的 Q/K/V —— 只有视频 token
                    clean_image_q = roped_query[:, :clean_image_seq_len]
                    clean_image_k = roped_key[:, :clean_image_seq_len]
                    clean_image_v = v[:, :clean_image_seq_len]

                    assert roped_query.shape[1] == half_seq_len + noisy_image_seq_len + action_horizon + state_horizon
                    
                    # 切分 noisy 部分的 Q/K/V —— [视频 | action | state]
                    noisy_image_q = roped_query[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_q = roped_query[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_q = roped_query[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    noisy_image_k = roped_key[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_k = roped_key[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_k = roped_key[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    noisy_image_v = v[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_v = v[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_v = v[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    # ========== 处理 CLEAN（上下文）图像 token ==========
                    # 干净图像：简单块级因果注意力（首帧自注意力 + 后续块因果看前面）
                    clean_image_outputs = self._process_clean_image_only(
                        clean_image_q, clean_image_k, clean_image_v, clean_frames)
                    
                    # ========== 处理 NOISY token ==========
                    # noisy 图像块：可看之前的 clean 图像块 + 当前 noisy 图像 + 当前 noisy action + 当前 noisy state
                    noisy_image_outputs = self._process_noisy_image_blocks(
                        noisy_image_q, noisy_image_k, noisy_image_v,
                        clean_image_k, clean_image_v,
                        noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # noisy 动作块：可看之前的 clean 图像块 + 当前 noisy 图像 + 自身 action + 当前 state
                    noisy_action_outputs = self._process_noisy_action_blocks(
                        noisy_action_q, noisy_action_k, noisy_action_v,
                        clean_image_k, clean_image_v, 
                        noisy_image_k, noisy_image_v,
                        noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # noisy 状态块：仅自注意力（conditioning，不看其他 token）
                    noisy_state_outputs = self._process_state_blocks(
                        noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)
                    
                    # 按原始顺序拼接回：[clean_img; noisy_img; noisy_action; noisy_state]
                    x = torch.cat([
                        clean_image_outputs,
                        noisy_image_outputs, noisy_action_outputs, noisy_state_outputs
                    ], dim=1)
                else:
                    # ===== 无 action/state 的纯视频 Teacher Forcing =====
                    half_frames = half_seq_len // self.frame_seqlen
                    # 切分为 clean 和 noisy 两半，各 (B, half_seq_len, n, d)
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]
                    
                    # clean 帧：块级因果 attention（每块只看首帧 + 之前的块 + 当前块）
                    x_clean = self._blockwise_causal_flash_attn(
                        clean_q, clean_k, clean_v, self.frame_seqlen, self.num_frame_per_block,
                        action_horizon=None, state_horizon=None,
                        num_action_per_block=None, num_state_per_block=None,
                        visualize_mask=False)
                    
                    # noisy 帧：可看所有 clean 帧 + 自身（非因果，全局 attention）
                    # full_k/v: (B, 2*half_seq_len, n, d) 拼接 clean+noisy
                    full_k = torch.cat([clean_k, noisy_k], dim=1)
                    full_v = torch.cat([clean_v, noisy_v], dim=1)
                    # self.attn: Flash Attention 无 mask（noisy query 对全体 clean+noisy key 做全连接）
                    x_noisy = self.attn(noisy_q, full_k, full_v)
                    
                    x = torch.cat([x_clean, x_noisy], dim=1)

            else:
                # ===== 非 Teacher Forcing 的训练路径（标准因果推理训练）=====
                roped_query = rope_action_apply(
                    x=q,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    x=k,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                # 从 action_register_length 反推 action/state 的 horizon
                if action_register_length is not None:
                    # chunk_size = 块数，action_register_length = chunk_size * (num_action + num_state)
                    chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None

                # 使用块级因果 Flash Attention：避免构造巨大的全量 mask
                visualize = False
                x = self._blockwise_causal_flash_attn(
                    roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=self.num_action_per_block if action_register_length else None,
                    num_state_per_block=self.num_state_per_block if action_register_length else None,
                    visualize_mask=visualize)

        else:
            # ===== 推理路径（有 KV cache）=====
            # 逐块/逐帧生成时，利用 KV cache 避免重算之前帧的 K/V

            # action_state_index: 当前生成块的索引，用于 RoPE 中定位 action/state 频率
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block

            # 对当前新 token 的 Q/K 应用 RoPE（使用 causal_rope_action_apply 而非 rope_action_apply）
            # causal_rope_action_apply 支持按 action_state_index 切片 freqs_action/freqs_state
            roped_query = causal_rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            # 将序列末尾的 action+state token 从 roped Q/K/V 中分离出来
            # 因为 action/state token 不参与 KV cache 的累积（每次重新提供）
            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                # 最后 action_register_length 个 token 是 action+state
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None

            # num_new_tokens: 当前步新增的视频 token 数
            num_new_tokens = roped_query.shape[1]
            assert roped_key.shape[1] == num_new_tokens
            assert v.shape[1] == num_new_tokens

            # 将新的 K/V 追加到 KV cache 中
            # kv_cache: (2, B, cache_len, n, d)，[0]=K，[1]=V
            updated_kv_cache = kv_cache
            updated_k = updated_kv_cache[0]  # (B, cache_len, n, d)
            updated_v = updated_kv_cache[1]
            # 拼接：(B, cache_len+num_new_tokens, n, d)
            new_k = torch.cat([updated_k, roped_key], dim=1)
            new_v = torch.cat([updated_v, v], dim=1)

            # 若启用 local attention，截断 KV cache 到最近 max_attention_size 个 token
            new_k = new_k[:, -self.max_attention_size:]
            new_v = new_v[:, -self.max_attention_size:]

            if action_register_length is not None:
                # 有 action/state 时：Q = [视频; action+state]，K/V = [cache; action+state]
                # 这样 action token 可以 attend 到所有历史视频帧 + 自身
                x = self.attn(
                    torch.cat([roped_query, roped_action_query], dim=1),
                    torch.cat([new_k, roped_action_key], dim=1),
                    torch.cat([new_v, action_v], dim=1),
                )
            else:
                # 纯视频：Q = 新 token，K/V = 历史 cache
                x = self.attn(
                    roped_query,
                    new_k,
                    new_v,
                )
            # 保存更新后的 KV cache: (2, B, new_cache_len, n, d)
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)


        # ===== 输出投影 =====
        # x: (B, s, n, d) -> flatten 多头 -> (B, s, n*d=dim) -> 输出线性层 o -> (B, s, dim)
        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class CausalWanAttentionBlock(nn.Module):
    """
    单层 DiT Block：AdaLN 调制的因果自注意力 + 文本/图像交叉注意力 + FFN。
    输入 x 与 context 做 self-attn(RoPE) 与 cross-attn，再 FFN，输出与 x 同 shape。
    """

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # ===== 子模块构建 =====
        # norm1: 自注意力前的 LayerNorm
        self.norm1 = WanLayerNorm(dim, eps)
        # self_attn: 块级因果自注意力（含 RoPE、QK-Norm、FlashAttention）
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        # norm3: 交叉注意力前的 LayerNorm（可选，cross_attn_norm=True 时启用）
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # cross_attn: 文本/图像条件的交叉注意力
        # cross_attn_type: 't2v_cross_attn' 或 'i2v_cross_attn'
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        # norm2: FFN 前的 LayerNorm
        self.norm2 = WanLayerNorm(dim, eps)
        # ffn: 两层 MLP + GELU 激活: dim -> ffn_dim -> dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation: AdaLN 可学习基准参数 (1, 6, dim)
        # 与时间步嵌入 e 相加后产生 6 个调制分量:
        # (shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn)
        # 初始化: N(0, 1/sqrt(dim))，保证初始调制幅度合理
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x: (B, L, C) 序列（视频+可选 action/state）。
            e: (B, F, 6, C) 时间调制向量，6 为 AdaLN 的 shift/scale/gate x2。
            freqs, freqs_action, freqs_state: RoPE 频率。
            action_register_length: action+state token 数。
            context: (B, L_ctx, C) 文本/图像条件。
            kv_cache, crossattn_cache: 推理用 cache。
        Returns:
            (B, L, C), updated_kv_cache。
        """
        # ===== AdaLN 调制参数生成 =====
        # self.modulation: 可学习参数 (1, 6, dim)，e: 时间步条件 (B, L, 6, dim)
        # 相加后沿 dim=2 切成 6 份，各 (B, L, 1, dim)
        # 6 个分量分别为: (shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn)
        #   - sa: self-attention 的 AdaLN 参数
        #   - ffn: feed-forward 的 AdaLN 参数
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # ===== 因果自注意力（带 AdaLN 调制）=====
        # AdaLN: x' = LayerNorm(x) * (1 + scale) + shift
        # norm1(x): (B, L, dim) -> LayerNorm
        # e[0]=shift_sa, e[1]=scale_sa: (B, L, 1, dim)，squeeze(2) -> (B, L, dim)
        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
        )
        # 残差连接 + gate: x = x + gate_sa * self_attn_output
        # e[2]=gate_sa: (B, L, 1, dim)，squeeze(2) -> (B, L, dim)
        x = x + (y * e[2].squeeze(2))

        # ===== 交叉注意力 + FFN（带 AdaLN 调制）=====
        def cross_attn_ffn(x, context, e):
            """
            交叉注意力: Q 来自 x (当前序列)，K/V 来自 context (文本/图像条件)
            FFN: 两层 MLP (dim -> ffn_dim -> dim) 带 GELU 激活
            """
            # norm3 + cross_attn: x 与 context 做交叉注意力，残差连接
            # cross_attn 内部: Q=norm3(x), K=V=context
            x = x + self.cross_attn(self.norm3(x), context)
            # AdaLN 调制后过 FFN: x' = norm2(x) * (1 + scale_ffn) + shift_ffn
            # e[3]=shift_ffn, e[4]=scale_ffn, e[5]=gate_ffn
            y = self.ffn(
                (self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            )
            # 残差 + gate: x = x + gate_ffn * ffn_output
            x = x + (y * e[5].squeeze(2))
            return x

        x = cross_attn_ffn(x, context, e)
        return x, updated_kv_cache


class CausalHead(nn.Module):
    """
    **Decoder Head #1: 视频潜空间噪声预测头**（Video Decoder Head）。

    作用:
        从 Backbone（CausalWanModel 40层 Transformer）输出的视频 token 段，
        解码出每个 patch 的潜空间噪声预测。经 unpatchify 后还原为
        (B, out_dim, T_lat, H_lat, W_lat) 的视频潜空间噪声，用于 Flow Matching 训练。

    原理:
        AdaLN（自适应 LayerNorm）调制 + 线性投影。
        使用时间步嵌入 e 对归一化后的特征做 scale/shift 调制，然后线性投影到 patch 空间。
        公式: output = Linear( LayerNorm(x) * (1 + scale) + shift )
        其中 (shift, scale) 由可学习的 modulation 参数 + 时间步嵌入 e 产生。

    Inputs (forward):
        x: (B, seq_len, dim=5120) — Backbone 输出的视频 token 段。
           seq_len = T_lat × (H_lat/ph) × (W_lat/pw)，如 DROID: 9×11×20=1980。
        e: (B, seq_len, 1, dim=5120) — 视频时间步调制向量（e_video.unsqueeze(2)）。
    Output:
        (B, seq_len, out_dim × pt × ph × pw) — 每个 patch 的噪声预测。
        典型: (B, 1980, 16×1×2×2=64)。
        后续经 unpatchify 还原为 (B, 16, 9, 22, 40) 与 VAE 潜变量同形。

    损失函数 (dynamics_loss):
        MSE(video_noise_pred, training_target) × training_weight(timestep)
        其中 training_target = noise - latents（Flow Matching 速度场目标）。
        无额外 mask，对所有帧和通道均计算。

    评估指标:
        - dynamics_loss_avg: 训练日志 10 步滑动平均
        - FVD/SSIM/PSNR: 需外部评估脚本

    上游: CausalWanModel._forward_train / _forward_blocks 中从 Backbone 输出取视频段
    下游: unpatchify → video_noise_pred → MSE loss (WANPolicyHead.forward)

    可训练: ✅ 始终可训练
    """

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """
        Args:
            dim: 输入隐藏维度（如 5120）
            out_dim: 输出通道数（如 16，VAE 潜变量通道）
            patch_size: 3D patch 尺寸 (pt, ph, pw)，如 (1, 2, 2)
            eps: LayerNorm epsilon
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        # 实际输出维度 = patch 体积 × 通道数，如 1*2*2*16=64
        out_dim = math.prod(patch_size) * out_dim
        # AdaLN: LayerNorm + shift/scale 调制
        self.norm = WanLayerNorm(dim, eps)
        # 线性投影: dim -> out_dim (patch 级输出)
        self.head = nn.Linear(dim, out_dim)
        # AdaLN 调制参数: (1, 2, dim)，2 个分量 = (shift, scale)
        # 初始化: N(0, 1/sqrt(dim))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        AdaLN 调制 + Linear 投影。

        Args:
            x: (B, seq_len, dim) — 视频 token 特征。
            e: (B, seq_len, 1, dim) — 时间步调制向量。

        Returns:
            (B, seq_len, out_dim × prod(patch_size)) — patch 级噪声预测。
            典型: (B, 1980, 64)。

        被调用: CausalWanModel._forward_train() / _forward_blocks()
        """
        # modulation (1, 2, dim) + e (B, seq_len, 1, dim) -> chunk(2) -> shift, scale 各 (B, seq_len, 1, dim)
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        # AdaLN 调制: x' = LayerNorm(x) * (1 + scale) + shift，然后线性投影
        # e[0]=shift, e[1]=scale; squeeze(2) 去掉多余维度 -> (B, seq_len, dim)
        # self.head: Linear(dim -> out_dim*pt*ph*pw)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    因果视频+动作 DiT 主干：支持 T2V/I2V，联合预测视频潜变量与动作噪声。
    序列为 [首帧 | 视频块 | 动作块 | 状态块]，块级因果注意力；含 state_encoder、action_encoder、action_decoder，
    patch/text/time 嵌入与多层 CausalWanAttentionBlock，CausalHead 出视频，action_decoder 出动作。
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 frame_seqlen=220,
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 max_chunk_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1, 
                 action_dim=32,
                 num_registers=8,
                 max_state_dim=64,
                 max_num_embodiments=32,
                 hidden_size=1024,
                 diffusion_model_pretrained_path=None,
                 num_action_per_block=32,
                 num_state_per_block=1):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type  # 't2v'=文本到视频, 'i2v'=图像到视频

        # ===== 保存配置参数 =====
        self.patch_size = patch_size        # 3D patch 尺寸 (pt, ph, pw)，典型 (1, 2, 2)
        self.frame_seqlen = frame_seqlen    # 每帧 patch 数 = (H_lat/ph) × (W_lat/pw)，如 DROID: 11×20=220
        self.text_len = text_len            # 文本 token 长度，固定 512
        self.in_dim = in_dim                # 输入视频通道数（VAE 潜变量 16ch）
        self.dim = dim                      # Transformer 隐藏维度（5120 大模型）
        self.ffn_dim = ffn_dim              # FFN 中间维度（13824 大模型）
        self.freq_dim = freq_dim            # 正弦时间嵌入维度（256）
        self.text_dim = text_dim            # 文本嵌入输入维度（4096 for T5-XXL）
        self.out_dim = out_dim              # 输出视频通道数（16）
        self.num_heads = num_heads          # 注意力头数（40 大模型）
        self.num_layers = num_layers        # Transformer 层数（40 大模型）
        # local_attn_size: 时间局部注意力窗口（帧数），-1 表示全局注意力
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block  # 每个注意力块包含的帧数
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim                    # 动作空间维度（64）
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim              # 状态空间维度（64）
        self.max_num_embodiments = max_num_embodiments  # 最大 embodiment 类别数
        self.hidden_size = hidden_size                  # encoder/decoder MLP 中间维度
        self.num_action_per_block = num_action_per_block  # 每个注意力块的 action token 数（32）
        self.num_state_per_block = num_state_per_block    # 每个注意力块的 state token 数（1）

        # 当前实现只用 1 个 embodiment
        max_num_embodiments = 1

        # ===== Encoder: 状态编码器 =====
        # state (B, T_state, 64) -> CategorySpecificMLP -> (B, T_state, dim)
        # 将归一化后的机器人状态映射到 DiT 隐藏维度
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        # ===== Encoder: 动作编码器 =====
        # (actions, timestep, cat_ids) -> (B, action_horizon, dim)
        # 将动作+扩散时间步编码为 token
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        # ===== Decoder Head #2: 动作噪声预测头 =====
        # 从 Backbone 输出的动作段 (B, action_horizon, dim) → (B, action_horizon, action_dim)
        # loss: MSE × action_mask × has_real_action × training_weight
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # ===== 嵌入层 =====
        # patch_embedding: Conv3d 将 3D 视频潜变量转为 patch token
        # (B, in_dim, F, H, W) -> Conv3d(k=patch_size, s=patch_size) -> (B, dim, F/pt, H/ph, W/pw)
        # 实际 patch_size=(1,2,2) 时: (B, 16, 9, 22, 40) -> (B, 5120, 9, 11, 20)
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        # text_embedding: Linear(text_dim→dim) + GELU + Linear(dim→dim)
        # (B, text_len, text_dim=4096) -> (B, text_len, dim=5120)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # time_embedding: 将正弦时间嵌入映射到 dim
        # sinusoidal(freq_dim) -> Linear(freq_dim→dim) -> SiLU -> Linear(dim→dim) -> e: (*, dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # time_projection: 从 time embedding 产生 AdaLN 的 6 个调制参数
        # e: (*, dim) -> SiLU -> Linear(dim→dim*6) -> (*, dim*6) -> reshape -> (*, 6, dim)
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # ===== Backbone: num_layers 层 CausalWanAttentionBlock =====
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                                    self.local_attn_size, sink_size, num_frame_per_block, qk_norm, cross_attn_norm, eps,
                                    num_action_per_block, num_state_per_block)
            for _ in range(num_layers)
        ])

        # ===== Decoder Head #1: 视频潜空间噪声预测头 =====
        # 从 Backbone 输出的视频段 (B, seq_len, dim) → AdaLN+Linear → (B, seq_len, out_dim*pt*ph*pw)
        # → unpatchify → (B, out_dim, T_lat, H_lat, W_lat) 视频噪声预测
        # loss: MSE × training_weight（无 mask）
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # ===== RoPE 频率预计算 =====
        # 不用 register_buffer 是为了避免 .to() 时 dtype 被自动转换
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads  # head_dim，如 5120/40=128
        
        # freqs_action: 1D RoPE 频率，供 action token 使用，最大位置 10240
        self.freqs_action = rope_params(1024*10, d)
        # freqs_state: 1D RoPE 频率，供 state token 使用，最大位置 1024
        self.freqs_state = rope_params(1024, d)
        # freqs: 3D RoPE 频率 [时间, 高度, 宽度] 三个分量
        # 时间维: head_dim - 4*(head_dim//6) 维; 高度/宽度各: 2*(head_dim//6) 维
        # 三个分量 concat 后恰好为 head_dim
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),  # 时间 RoPE
            rope_params(1024, 2 * (d // 6)),       # 高度 RoPE
            rope_params(1024, 2 * (d // 6)),       # 宽度 RoPE
        ]
        if model_type == 'i2v':
            # I2V 模式：CLIP 图像特征投影 (1280 → dim)
            self.img_emb = MLPProj(1280, dim)

        # Xavier 初始化所有参数
        self.init_weights()

        self.gradient_checkpointing = True
        # 首帧独立处理仅在 num_frame_per_block > 1 时启用
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1, action_horizon=1, state_horizon=1, num_action_per_block=30, num_state_per_block=1
    ) -> BlockMask:
        """
        使用 PyTorch FlexAttention API 构建块级因果注意力 mask。
        仅用于非 Teacher Forcing 的训练/评估路径（实际训练中主要用 _blockwise_causal_flash_attn）。

        序列布局: [首帧 (conditioning)] [图像块×N] [动作块×N] [状态块×N]

        注意力规则:
        - 首帧: 仅自注意力（self_attn，不看其他 token）
        - 图像块 i: 可看首帧 + 所有之前图像块 (0..i) + 当前动作块 i + 当前状态块 i
        - 动作块 i: 可看首帧 + 所有之前图像块 (0..i) + 自身动作块 i + 当前状态块 i
        - 状态块: 仅自注意力

        使用 torch.nn.attention.flex_attention.create_block_mask 创建
        BlockMask（稀疏表示），避免 O(n²) 的显式 mask 矩阵。

        Returns:
            BlockMask 对象，可直接传入 flex_attention 或用于调试可视化
        """
        # 计算各模态块数
        num_image_blocks = (num_frames - 1) // num_frame_per_block  # 首帧独立，剩余帧分块
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        
        # 块数必须对齐: 每个图像块配一个动作块和一个状态块
        assert num_image_blocks == num_action_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_action_blocks}"
        assert num_image_blocks == num_state_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_state_blocks}"
        
        # 各段 token 长度
        first_image_len = frame_seqlen  # 首帧 token 数
        image_blocks_len = num_image_blocks * num_frame_per_block * frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        total_length = first_image_len + image_blocks_len + action_len + state_len
        
        # FlexAttention 要求序列长度为 128 的倍数，右 padding 补齐
        padded_length = math.ceil((local_attn_size * frame_seqlen + (local_attn_size - 1) + 32 * (local_attn_size - 1))/128) * 128 - total_length
        total_padded_length = total_length + padded_length
        # print("total_padded_length", total_padded_length, total_length, padded_length)
        
        # Define token ranges for each modality
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # 预计算每个 token 的块号索引，用于 attention_mask 函数中判断因果关系
        # block_indices[i] = token i 所属的块号（图像/动作/状态块共享编号空间 0..N-1）
        block_indices = torch.zeros(total_padded_length, device=device, dtype=torch.long)
        
        # 首帧: 块号 -1（特殊标记，仅自注意力）
        block_indices[first_image_start:first_image_end] = -1
        
        # 图像块: 块号 0 ~ num_image_blocks-1
        for block_idx in range(num_image_blocks):
            start_idx = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            end_idx = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            block_indices[start_idx:end_idx] = block_idx
        
        # 动作块: 块号 0 ~ num_action_blocks-1（与图像块对齐）
        for block_idx in range(num_action_blocks):
            start_idx = action_start + block_idx * num_action_per_block
            end_idx = action_start + (block_idx + 1) * num_action_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # 状态块: 块号 0 ~ num_state_blocks-1（与图像块对齐）
        for block_idx in range(num_state_blocks):
            start_idx = state_start + block_idx * num_state_per_block
            end_idx = state_start + (block_idx + 1) * num_state_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # padding token: 块号 = num_image_blocks（不属于任何有效块，不会被 attend）
        block_indices[total_length:] = num_image_blocks
        
        def attention_mask(b, h, q_idx, kv_idx):
            """
            FlexAttention mask 函数: 给定 query 位置 q_idx 和 key 位置 kv_idx，
            返回是否允许注意力（True=可 attend）。
            b, h 参数用于 batch/head 级差异化（此处统一，不区分）。
            内部通过 block_indices[q_idx/kv_idx] 判断所属模态和块号。
            """
            # 自注意力: 任何 token 都可以 attend 自身
            self_attn = (q_idx == kv_idx)
            
            # 判断 query / key 所属模态
            q_is_first_image = (q_idx >= first_image_start) & (q_idx < first_image_end)
            q_is_image_block = (q_idx >= image_blocks_start) & (q_idx < image_blocks_end)
            q_is_action = (q_idx >= action_start) & (q_idx < action_end)
            q_is_state = (q_idx >= state_start) & (q_idx < state_end)
            
            kv_is_first_image = (kv_idx >= first_image_start) & (kv_idx < first_image_end)
            kv_is_image_block = (kv_idx >= image_blocks_start) & (kv_idx < image_blocks_end)
            kv_is_action = (kv_idx >= action_start) & (kv_idx < action_end)
            kv_is_state = (kv_idx >= state_start) & (kv_idx < state_end)
            
            # 获取 query/key 所属的块号
            q_block = block_indices[q_idx]
            kv_block = block_indices[kv_idx]
            
            # 首帧 query: 不看任何外部 token（仅靠 self_attn 维持自注意力）
            first_image_mask = q_is_first_image & False
            
            # 图像块 query 的注意力规则:
            image_to_first = q_is_image_block & kv_is_first_image  # → 首帧: 始终可看
            image_to_image = q_is_image_block & kv_is_image_block & (kv_block <= q_block)  # → 图像块: 因果（可看当前+之前块）
            image_to_action = q_is_image_block & kv_is_action & (kv_block == q_block)  # → 动作块: 仅当前同号块
            image_to_state = q_is_image_block & kv_is_state & (kv_block == q_block)  # → 状态块: 仅当前同号块
            image_block_mask = image_to_first | image_to_image | image_to_action | image_to_state
            
            # 动作块 query 的注意力规则:
            action_to_image = q_is_action & kv_is_image_block & (kv_block <= q_block)  # → 图像块: 因果（当前+之前）
            action_to_action = q_is_action & kv_is_action & (kv_block == q_block)  # → 动作块: 仅同块（自注意力）
            action_to_state = q_is_action & kv_is_state & (kv_block == q_block)  # → 状态块: 仅同块
            action_to_first = q_is_action & kv_is_first_image  # → 首帧: 始终可看
            action_mask = action_to_image | action_to_action | action_to_state | action_to_first
            
            # 状态块 query: 不看任何外部 token（仅靠 self_attn 维持自注意力）
            state_mask = q_is_state & False
            
            # 组合: 自注意力 | 首帧mask | 图像块mask | 动作块mask | 状态块mask
            return self_attn | first_image_mask | image_block_mask | action_mask | state_mask
        
        # create_block_mask: 将 attention_mask 函数编译为 BlockMask 稀疏结构
        # B=None, H=None: 不区分 batch/head（全局共享 mask）
        # Q_LEN=KV_LEN=total_padded_length: 含 padding 的总序列长度
        # _compile=False: 不预编译（节省启动时间）
        block_mask = create_block_mask(
            attention_mask, B=None, H=None, 
            Q_LEN=total_padded_length,
            KV_LEN=total_padded_length, 
            _compile=False, device=device
        )
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Created blockwise causal attention mask:")
            print(f"  first_image_tokens={first_image_len} (conditioning)")
            print(f"  num_image_blocks={num_image_blocks} (blocks of {num_frame_per_block * frame_seqlen})")
            print(f"  num_action_blocks={num_action_blocks} (blocks of {num_action_per_block})")
            print(f"  num_state_blocks={num_state_blocks} (blocks of {num_state_per_block})")
            print(f"  total_length={total_length}, padded_length={padded_length}")
            print(block_mask)

            # Debug: materialize a small slice of the mask into 0/1 strings
            try:
                dense_mask = create_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_padded_length,
                    KV_LEN=total_padded_length,
                    device=device,
                )[0, 0]  # [Q, K]
                preview_q = min(979, dense_mask.shape[0])
                preview_k = min(979, dense_mask.shape[1])
                print("Block mask (preview):")
                for qi in range(preview_q):
                    row = dense_mask[qi, :preview_k].to(torch.int8).tolist()
                    print(" ".join(str(int(v)) for v in row))
            except Exception as err:
                print("[warn] Failed to materialize block mask preview:", err)
        
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        构建纯视频（无 action/state）Teacher Forcing 训练的注意力 mask。

        序列结构: [clean 帧 ×N] [noisy 帧 ×N]，总长 = 2 × num_frames × frame_seqlen

        注意力规则:
        - clean 帧块 i: 可看之前所有 clean 块 [0..i]（块级因果）
        - noisy 帧块 i: 可看之前所有 clean 块 [0..i-1] + 自身 noisy 块 i（跨半）
        - 所有 token 可自注意力

        注意: 此方法使用 FlexAttention API，但实际训练中更常用
        _blockwise_causal_flash_attn 的切片实现（不构造显式 mask）。
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(self.local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        I2V（图像到视频）模式的块级因果注意力 mask。

        序列结构: [首帧 (1帧)] [块1 (N帧)] [块2 (N帧)] ... [块K (N帧)]
        首帧独立出来作为 I2V 的图像条件。

        注意力规则:
        - 首帧: 仅自注意力
        - 块 i: 可看首帧 + 之前所有块 + 当前块（因果）
        - 若 local_attn_size != -1: 块 i 仅看最近 local_attn_size 帧窗口

        Returns:
            BlockMask 稀疏注意力结构
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        在潜空间序列上执行多层 CausalWanAttentionBlock。
        x 为 [B, L, C]，L = seq_len + action_register_length；先 patch_embed 得到 x，拼 action/state 嵌入后过各 block，
        最后只取前 seq_len 维经 CausalHead 得到视频预测，动作段经 action_decoder 得到动作预测。

        Returns:
            x_video: (B, seq_len, C_out) 经 head 后 unpatchify 前的视频表示（实际在外部 unpatchify）。
            action_noise_pred: (B, action_length, action_dim) 或 None。
            updated_kv_caches: 每层更新后的 KV cache。
        """
        # ===== 将 5D 视频 patch 特征展平为 2D 序列 =====
        # x 输入: (B, dim, F_pat, H_pat, W_pat)，F_pat=帧数/pt, H_pat=高/ph, W_pat=宽/pw
        # flatten(2): (B, dim, F*H*W) -> transpose: (B, F*H*W, dim) 即 (B, seq_len, dim)
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]  # 帧数

        # ===== 动作/状态编码 =====
        if action is not None:
            # 当前只有 1 个 embodiment，全部用 id=0
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            # action_encoder: (actions, timestep_action, embodiment_id) -> (B, action_length, dim)
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            # state_encoder: (state, embodiment_id) -> (B, state_length, dim)
            state_features = self.state_encoder(state, embodiment_id)
            # 拼接为 action_register: (B, action_length+state_length, dim)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            # 拼到视频序列后: (B, seq_len+action_register_length, dim)
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # ===== 时间步嵌入 =====
        # timestep: (B, F) -> 每帧展开 seq_len//F 次 -> (B, seq_len)
        # 使得每个视频 token 都有对应的时间步值
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            # 状态步少于动作步，下采样得到时间步
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            # 拼接: (B, seq_len + action_horizon + state_horizon)
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        # sinusoidal_embedding_1d: (B*L,) -> (B*L, freq_dim) 正弦位置编码
        # time_embedding: (B*L, freq_dim) -> MLP -> (B*L, dim)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        # unflatten: (B*L, dim) -> (B, L, dim)
        e = e.unflatten(dim=0, sizes=(B, -1))
        # time_projection: (B, L, dim) -> (B, L, dim*6) -> unflatten -> (B, L, 6, dim)
        # 6 维用于 AdaLN 的 (shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn)
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # ===== 文本/图像条件嵌入 =====
        # text_embedding: (B, text_len, text_dim) -> MLP -> (B, text_len, dim)
        context = self.text_embedding(context)
        
        if clip_feature is not None:
            # I2V: CLIP 特征 (B, 257, 1280) -> MLPProj -> (B, 257, dim)
            clip_embedding = self.img_emb(clip_feature)
            # 拼到 text 前面: (B, 257+text_len, dim)
            context = torch.cat([clip_embedding, context], dim=1)

        # ===== 逐层过 Transformer Block =====
        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            # 每层: x (B, L, dim) -> CausalWanAttentionBlock -> (B, L, dim)
            # L = seq_len + action_register_length (若有 action)
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        # ===== 从 Backbone 输出中拆出动作段并解码 =====
        if action is not None:
            # 动作段位于 x[:, seq_len : seq_len+action_length]
            # action_decoder: (B, action_length, dim) -> (B, action_length, action_dim)
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # ===== 视频段经 CausalHead 解码 =====
        # x_video: (B, seq_len, dim) — 仅取视频 token
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        # head: AdaLN + Linear -> (B, seq_len, out_dim * pt * ph * pw)
        # e_video.unsqueeze(2): (B, seq_len, dim) -> (B, seq_len, 1, dim) 供 AdaLN 调制
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches


    def _forward_inference_trt(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        TensorRT 推理入口（非 DROID 数据集）。
        kv_cache_packed: (num_layers, 2, B, cache_seq_len, n, d) 打包的 KV cache。
        frame_seqlen=880 对应 2 帧/块 × 440 token/帧 的设定。
        """
        frame_seqlen = 880
        seq_len = 2*frame_seqlen  # 每次处理 2 帧
        # 从 cache 长度反推已生成到第几帧
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        # 将打包的 kv_cache 拆成逐层 list
        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        ) 

        return x_video, action_noise_pred

    def _forward_inference_trt_droid(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        TensorRT 推理入口（DROID 数据集）。
        frame_seqlen=440 对应 DROID 的 2 帧/块 × 220 token/帧。
        """
        frame_seqlen = 440
        seq_len = 2*frame_seqlen
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        ) 

        return x_video, action_noise_pred


    def _forward_inference(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
        embodiment_id=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        推理前向（自回归生成时调用，支持 KV cache）。
        逐块/逐帧生成视频，每次只处理新增的帧，利用 KV cache 避免重算历史。

        Args:
            x: (B, C_in, F_new, H, W) — 当前步的噪声潜变量（新帧）
            timestep: (B, F_new) — 当前步各帧的扩散时间步
            context: (B, text_len, text_dim) — 文本条件嵌入
            seq_len: 当前步视频 token 数 = F_new * (H/ph) * (W/pw)
            kv_cache: list[torch.Tensor]，每层的 KV cache (2, B, cache_len, n, d)
            current_start_frame: 当前生成到第几帧（用于 RoPE 偏移）
            y: (B, 20, F_new, H, W) — I2V 首帧条件
            clip_feature: (B, 257, 1280) — CLIP 图像特征
            action: (B, action_horizon, action_dim)
            timestep_action: (B, action_horizon)
            state: (B, state_horizon, state_dim)
        Returns:
            video_noise_pred: (B, C_out, F_new*pt, H, W) — 视频噪声预测
            action_noise_pred: (B, action_horizon, action_dim) 或 None
            updated_kv_caches: 更新后的各层 KV cache
        """      
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        if y is not None:
            # I2V: 通道维拼接首帧条件 y
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # Conv3d patch 嵌入: (B, C_in, F, H, W) -> (B, dim, F_pat, H_pat, W_pat)
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        # 创建 RoPE 频率，从 current_start_frame 开始偏移
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        # 调用 _forward_blocks: 过所有 Transformer 层 + head/decoder
        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        # clone 输出以断开计算图（推理时不需要梯度）
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()

        # unpatchify: (B, seq_len, out_dim*pt*ph*pw) + grid_size -> (B, C_out, F*pt, H*ph, W*pw)
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
    ):
        r"""
        训练前向：可选 teacher forcing（clean_x + noisy x）。patch 化后拼 action/state，
        时间嵌入按视频/动作/状态展开，过所有 block 后视频段进 CausalHead、动作段进 action_decoder。

        Args:
            x: (B, C_in, F, H, W) 噪声潜变量，C_in=16 或 36(i2v)。
            timestep: (B, F) 每帧时间步。
            context: (B, text_len, text_dim) 文本嵌入。
            seq_len: 视频 token 数 F*H*W。
            clean_x: 可选，(B, C_in, F, H, W) 干净潜变量，与 x 拼成 [clean; noisy] 做 TF。
            y: I2V 时首帧潜变量；(B, 20, F, H, W) 与 x 通道维 concat。
            clip_feature: (B, 257, 1280) 首帧 CLIP 特征。
            action: (B, action_horizon, action_dim)；state: (B, state_horizon, state_dim)。
        Returns:
            video_noise_pred: (B, C_out, F, H, W)；action_noise_pred: (B, action_horizon, action_dim) 或 None。
        """
        if self.model_type == 'i2v':
            # I2V 模式下必须提供首帧条件：clip_feature 与 y
            assert clip_feature is not None and y is not None

        if y is not None:
            # I2V：在通道维拼接首帧潜变量 y；x (B, 16, F, H, W) + y (B, 20, F, H, W) -> (B, 36, F, H, W)
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # ---------- 视频潜变量 patch 嵌入 ----------
        # patch_embedding: Conv3d(kernel=patch_size=(1,2,2))，将 (B, C_in, F, H, W) -> (B, dim, F, H/2, W/2)
        x = self.patch_embedding(x)

        # grid_size: (F, H/2, W/2) 即 patch 网格，用于 RoPE 和后续 unpatchify
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        # freqs: (F*H/2*W/2, 1, head_dim) 3D RoPE 频率，供所有视频 token 使用
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=0,
        )

        # 将 patch 序列展平为 (B, seq_len, dim)，seq_len = F * (H/2) * (W/2)
        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        F = timestep.shape[1]

        # ---------- 动作/状态嵌入并拼到序列后（有 action 时）----------
        if action is not None:
            # embodiment_id: (B,) 全 0，用于 CategorySpecific 线性层索引
            embodiment_id = torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
            # action_features: (B, action_horizon, dim) 动作+时间步编码
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            action_length = action_features.shape[1]
            # state_features: (B, state_horizon, dim) 状态编码
            state_features = self.state_encoder(state, embodiment_id)
            # action_register: (B, action_horizon+state_horizon, dim) 拼成一块
            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            # x: (B, seq_len+action_register_length, dim)
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            action_length = None
            state_features = None
            action_register = None
            action_register_length = None

        # ---------- 时间步嵌入：按视频帧/动作步/状态步展开 ----------
        # timestep (B, F) -> 每帧复制 seq_len//F 次 -> (B, seq_len)，表示每个视频 token 对应的时间步
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        timestep_original = timestep.clone()

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            # 状态步数少于动作步数，用 stride 下采样得到每状态对应的时间步
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            # timestep: (B, seq_len + action_horizon + state_horizon)，与 x 的序列维一一对应
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        # 正弦时间嵌入 -> time_embedding -> e; 再 time_projection -> e0 供 AdaLN 使用
        # sinusoidal_embedding_1d(freq_dim, timestep.flatten()): (B*L_total,) -> (B*L_total, freq_dim)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        # e: (B*L_total, dim) -> (B, L_total, dim)
        e = e.unflatten(dim=0, sizes=(B, -1))
        # e0: (B, L_total, dim*6) -> (B, L_total, 6, dim)，每个 token 的 6 维 AdaLN 调制
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # ---------- 文本/图像条件 context ----------
        assert context.shape[1] == self.text_len
        # context: (B, text_len, text_dim) -> (B, text_len, dim)
        context = self.text_embedding(context)

        if clip_feature is not None:
            # clip_embedding: (B, 257, 1280) -> (B, 257, dim)
            clip_embedding = self.img_emb(clip_feature)
            # context: (B, 257+text_len, dim)
            context = torch.cat([clip_embedding, context], dim=1)

        # ---------- Teacher Forcing：若有 clean_x，拼 [clean; noisy] 并扩展时间嵌入 ----------
        if clean_x is not None:
            if y is not None:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len

            # 序列变为 [clean 视频 token; noisy 视频+动作+状态 token]，总长 2*seq_len + action_register_length（若有 action）
            x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)
            assert aug_t is not None

            # 干净部分用 aug_t（通常为 0）做时间嵌入，与 noisy 部分拼成完整 e0
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean)
            e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
            # e0: (B, seq_len + L_noisy, 6, dim)，前半为 clean 的 e0_clean，后半为 noisy 的 e0
            e0 = torch.cat([e0_clean, e0], dim=1)

        # 传入各 block 的通用参数字典（freqs 与 context 等）
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
            is_tf=clean_x is not None,
        )

        # ===== 梯度检查点封装 =====
        # gradient checkpointing: 以时间换空间，训练时不缓存中间激活，
        # 反向传播时重新计算——大幅降低显存占用。
        # 训练时无 KV cache（kv_cache=None），create_custom_forward 丢弃 kv_cache 返回值。
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None  # 训练时不使用 KV cache
                return outputs
            return custom_forward

        # ===== 逐层过 Transformer Block =====
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # torch.utils.checkpoint.checkpoint: 前向只保存输入，反向时重计算
                # use_reentrant=False: 推荐的非重入模式（PyTorch 2.0+）
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # 非 checkpoint 模式: 正常前向
                # 每层: x (B, L_total, dim) -> CausalWanAttentionBlock -> (B, L_total, dim)
                x = block(x, **kwargs)

        # 若做了 TF，只保留 noisy 部分（去掉前半 clean 视频 token）
        if clean_x is not None:
            x = x[:, clean_x.shape[1]:]

        # ---------- 从 block 输出中拆出动作段并解码 ----------
        if action is not None:
            # 动作段: x[:, seq_len : seq_len+action_length]，shape (B, action_length, dim)
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            # action_decoder -> (B, action_length, action_dim)
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # ---------- 仅取视频段做 head 并 unpatchify ----------
        # x_video: (B, seq_len, dim)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # head: AdaLN + Linear，输出 (B, seq_len, out_dim*pt*ph*pw)
        x_video = self.head(x_video, e_video.unsqueeze(2))
        # unpatchify: (B, seq_len, ...) + grid_size -> (B, out_dim, F, H, W) 视频潜空间噪声预测
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    def forward(
        self,
        *args,
        **kwargs
    ):
        """有 kv_cache 走推理 _forward_inference，否则走训练 _forward_train。"""
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_size):
        r"""
        将 patch token 序列还原为 3D 视频潜变量张量。

        步骤:
        1. x: (B, L, C_out*pt*ph*pw) -> view -> (B, f, h, w, pt, ph, pw, C_out)
           其中 L = f*h*w，f/h/w 为 patch 网格维度
        2. einsum 'bfhwpqrc->bcfphqwr' 重排维度:
           b=batch, c=channel, f/h/w=网格, p/q/r=patch 内位置
           结果: (B, C_out, f, pt, h, ph, w, pw)
        3. reshape: (B, C_out, f*pt, h*ph, w*pw) = (B, C_out, F_lat, H_lat, W_lat)

        Args:
            x: (B, L, C_out * prod(patch_size))，L = prod(grid_size)
            grid_size: (f, h, w) patch 网格维度
        Returns:
            (B, C_out, f*pt, h*ph, w*pw) — 视频潜空间张量
        
        典型 DROID: grid_size=(9,11,20), patch_size=(1,2,2)
            x: (B, 1980, 64) -> (B, 16, 9, 22, 40)
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        # (B, f, h, w, pt, ph, pw, c) — 将每个 token 展开为 patch 内各位置的值
        x = x.view(B, *grid_size, *self.patch_size, c)
        # einsum 重排: 将 patch 内位置交织到空间维度中
        # 'bfhwpqrc' 中 f/h/w 为网格坐标, p/q/r 为 patch 内坐标, c 为通道
        # -> 'bcfphqwr' 使 f*p 构成时间维, h*q 构成高度维, w*r 构成宽度维
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        # reshape: (B, c, f*pt, h*ph, w*pw) — 最终视频潜变量
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        """
        根据 patch 网格尺寸和起始帧创建 3D RoPE 频率张量。

        3D RoPE 将 head_dim 拆为三部分 [时间, 高度, 宽度]，各自独立编码位置:
        - 时间维: head_dim - 4*(head_dim//6) 维
        - 高度维: 2*(head_dim//6) 维  
        - 宽度维: 2*(head_dim//6) 维

        对于每个视频 token (t, h, w)，其 RoPE 频率为 concat(freq_t[t], freq_h[h], freq_w[w])。

        Args:
            grid_size: (3,) = (f, h, w) 即 patch 网格维度
            start_frame: RoPE 时间维的起始偏移（推理时用于 KV cache 对齐）
        Returns:
            freqs: (f*h*w, 1, head_dim) — 每个视频 token 的 RoPE 频率
        """
        device = self.patch_embedding.weight.device
        # 确保频率张量在正确设备上
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        # 三个分量分别广播到 (f, h, w, dim_i) 然后在最后一维 concat
        # freqs[0][start_frame:start_frame+f]: 时间维 RoPE，(f, dim_t)
        #   -> view(f,1,1,-1) -> expand(f,h,w,dim_t)：每个 (h,w) 位置共享同一时间频率
        # freqs[1][:h]: 高度维 RoPE，同理
        # freqs[2][:w]: 宽度维 RoPE，同理
        freqs = torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1  # concat 后得到 (f, h, w, head_dim)
        ).reshape(f * h * w, 1, -1)  # -> (f*h*w, 1, head_dim)

        return freqs

    def init_weights(self):
        r"""
        初始化模型参数。

        策略:
        1. 所有 nn.Linear: Xavier 均匀初始化 + bias 置零
        2. patch_embedding (Conv3d): Xavier 均匀初始化（weight 展平为 2D）
        3. text_embedding / time_embedding 中的 Linear: 正态初始化 std=0.02
        4. head.head (输出投影层): 权重置零（确保训练初期输出接近零，稳定训练）
        """

        # 所有 Linear 层: Xavier 均匀 + bias 零初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 均匀: W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # patch 嵌入 Conv3d: 将 5D 权重展平为 2D 后做 Xavier 均匀
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        # text/time 嵌入: 正态分布 std=0.02（较小方差避免初期嵌入值过大）
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # 输出头权重置零: 训练初期 head 输出全零，等效于预测"不变"
        nn.init.zeros_(self.head.head.weight)
