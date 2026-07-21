# DreamZero 训练数据流性能建模

> 模型：Wan2.1-I2V-14B (~16.5B 参数) | LoRA rank=4 | FSDP full_shard × 8 NPU
> 配置：num_frames=33, action_horizon=24, num_views=3, resolution=176×320
> DiT：40 layers, dim=5120, num_heads=40, head_dim=128, ffn_dim=13824
> VAE：z_dim=16, spatial 8×↓, temporal 4×↓

---

## 总览：端到端数据流

> **关键架构说明**：3 个相机视角（256×480 each）在 DreamTransform 中被拼接为 2×2 网格（512×960），
> 然后在模型 forward 中 resize 到 176×320 再送入 VAE。DiT 看到的 token 来自这个包含 3 视角的合成图像。

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    DROID Dataset (parquet + MP4)                          │
│  Video ×3 views: (T_vid, ~256, ~480, 3) uint8, MP4 decoded              │
│  Action: (T_action, 8) float64      State: (T_state, 8) float64         │
│  Language: text strings (3 variants)                                     │
│  T_vid = 8n+1 (9~41), T_action = 24m (24~120), T_state = 1~5 anchors   │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ Per-sample: video ~5-15 MB raw (varies by T_vid)
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 0: DreamTransform — Grid Composition (CPU)                        │
│  • 3 views → 2×2 composite grid:                                         │
│    [ wrist(×2 wide) | wrist(×2 wide) ]  top row (480×2=960)             │
│    [ exterior_left  | exterior_right  ]  bottom row                      │
│  • Total grid: (T_vid, 512, 960, 3) uint8                               │
│  • Resize/Crop to (T_vid, 256, 480, 3) then compose                      │
│  • State padded to (T_state, 44) float64, Action padded to (T, 32)      │
│                                                                          │
│  Output: video (B, T, 512, 960, 3) uint8    State (B, T_s, 44) float64  │
│          action (B, T_a, 32) float64         Text token ids (B, 512)     │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Model Forward — Video Preprocessing (NPU)                      │
│  • rearrange: (B, T, 512, 960, 3) → (B, 3, T, 512, 960)                │
│  • uint8 → float32 / 255 → Normalize(0.5, 0.5) → bf16                   │
│  • Resize to target: interpolate((B, 3, T, 512, 960) → (B, 3, 33, 176, 320))│
│  • T is trimmed/padded to exactly 33 frames                              │
│                                                                          │
│  Output: video (B, 3, 33, 176, 320) bf16  ≈ 11.2 MB                     │
│          action (B, 24, 7) bf16             ≈ 336 B                      │
│          state  (B, 1, 7) bf16              ≈ 14 B                       │
│          text   (B, 512) int64              ≈ 4 KB                       │
│                                                                          │
│  注：B = batch_size=1（合成网格含 3 视角），effective 单步样本数 = 8     │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 2: VAE Encode (NPU, frozen, no_grad)                              │
│  • Spatial: 176×320 → 22×40  (8×↓)                                      │
│  • Temporal: 33 → 9          (4×↓, padded to 36 then /4)                │
│  • Channel: 3 → 16           (z_dim)                                     │
│                                                                          │
│  Input:  (B, 3,  33, 176, 320) bf16 ≈ 11.2 MB                           │
│  Output: (B, 16, 9,  22,  40)  bf16 ≈ 0.62 MB (18× compression)         │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 3: CLIP + T5 Encode (NPU, frozen, no_grad)                       │
│                                                                          │
│  CLIP (first frame → image condition):                                   │
│    Input:  first frame (B, 3, 176, 320) bf16                             │
│    Output: (B, 257, 1280) → MLPProj → (B, 257, 5120) ≈ 2.6 MB           │
│                                                                          │
│  T5 (language instruction):                                              │
│    Input:  token ids (B, 512) int64                                      │
│    Output: (B, 512, 4096) → text_emb → (B, 512, 5120) ≈ 5.2 MB          │
│                                                                          │
│  First Frame Condition y (via VAE):                                      │
│    VAE([first_frame, zeros(32frames)]) → (B, 16, 9, 22, 40)             │
│    + mask(4) → (B, 20, 9, 22, 40) bf16                       ≈ 0.76 MB  │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 4: Flow Matching Noise (NPU)                                      │
│  • σ ~ Beta(3,1) or Uniform(0,1000)                                     │
│  • z_noisy = (1-σ)·z_0 + σ·ε                                            │
│  • target = ε - z_0                                                      │
│                                                                          │
│  noise:          (B, 16, 9, 22, 40) bf16     ≈ 0.62 MB                  │
│  noisy_latents:  (B, 16, 9, 22, 40) bf16     ≈ 0.62 MB                  │
│  training_target:(B, 16, 9, 22, 40) bf16     ≈ 0.62 MB                  │
│  noise_action:   (B, 24, 7) bf16             ≈ 336 B                     │
│  noisy_actions:  (B, 24, 7) bf16             ≈ 336 B                     │
│  target_action:  (B, 24, 7) bf16             ≈ 336 B                     │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 5: Patch Embedding (NPU)                                          │
│  • Concat [x(16ch), y(20ch)] → 36ch input                               │
│  • Conv3d(36→5120, kernel=(1,2,2), stride=(1,2,2))                      │
│                                                                          │
│  Input concat:  (B, 36, 9, 22, 40) bf16      ≈ 1.36 MB                  │
│  After embed:   (B, 5120, 9, 11, 20) bf16    ≈ 19.3 MB                  │
│  Flatten:       (B, 1980, 5120) bf16          ≈ 19.3 MB                  │
│  (tokens_per_frame = 11×20 = 220, seq_len = 9×220 = 1980)               │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 6: Token Assembly — Teacher Forcing (NPU)                         │
│                                                                          │
│  clean_x tokens:   1980 tokens × 5120 dim  ≈ 19.3 MB (bf16)             │
│  noisy_x tokens:   1980 tokens × 5120 dim  ≈ 19.3 MB (bf16)             │
│  action_register:  24 tokens × 5120 dim     ≈ 0.24 MB (bf16)            │
│  state_register:   1 token × 5120 dim       ≈ 10 KB  (bf16)             │
│  ─────────────────────────────────────────────────                       │
│  Total sequence:   3985 tokens × 5120 dim   ≈ 38.9 MB (bf16)            │
│                                                                          │
│  Context (cross-attn):                                                   │
│  CLIP:  257 tokens × 5120 dim  ≈ 2.6 MB                                 │
│  T5:    512 tokens × 5120 dim  ≈ 5.2 MB                                 │
│  Total context: 769 tokens      ≈ 7.9 MB                                │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 7: DiT × 40 Layers (NPU, FSDP wrapped)                            │
│                                                                          │
│  Per layer operations (for each of 40 CausalWanAttentionBlocks):         │
│                                                                          │
│  ┌─ SelfAttention ─────────────────────────────────────────────────┐    │
│  │  Q/K/V: Linear(5120→5120) × 3 → (B, 3985, 40, 128) bf16      │    │
│  │  RoPE: 3D (video), 1D (action/state), complex→real on NPU       │    │
│  │  Causal mask: blockwise (2 frames/block)                        │    │
│  │  SDPA: Q×K^T → (B, 40, 3985, ≤3985) × V                     │    │
│  │  Output: Linear(5120→5120) → (B, 3985, 5120)                 │    │
│  │  Data accessed: ~122 MB read, ~41 MB write (bf16)              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─ CrossAttention (I2V, 2 branches) ───────────────────────────────┐   │
│  │  Branch 1: Q(video,3985) × K,V(CLIP,257) → (B, 3985, 5120)    │   │
│  │  Branch 2: Q(video,3985) × K,V(T5,512)   → (B, 3985, 5120)    │   │
│  │  Data accessed: ~49 MB read, ~41 MB write                        │   │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─ FFN + AdaLN ───────────────────────────────────────────────────┐    │
│  │  RMS Norm → Linear(5120→13824) → GELU → Linear(13824→5120)      │    │
│  │  AdaLN: time-modulated shift/scale/gate (6 params × 5120)       │    │
│  │  Data accessed: ~163 MB read, ~41 MB write                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Per-layer total data movement (read+write): ~457 MB                     │
│  40 layers total: ~18.3 GB (logical, before gradient checkpointing)      │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 8: Head + Unpatchify (NPU)                                        │
│  • Extract x[:, :1980] → (B, 1980, 5120)                              │
│  • CausalHead: Linear(5120→64) → (B, 1980, 64)                        │
│  • Unpatchify: (B, 1980, 64) → (B, 16, 9, 22, 40)                  │
│  • video_noise_pred: (B, 16, 9, 22, 40) bf16      ≈ 1.2 MB            │
│                                                                          │
│  • Extract x[:, 1980:2004] → (B, 24, 5120)                            │
│  • Action Decoder: MLP(5120→1024→64→7) → (B, 24, 7)                  │
│  • action_noise_pred: (B, 24, 7) bf16             ≈ 1.0 KB            │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 9: Loss Computation (NPU)                                         │
│  • L_dynamics = weighted_MSE(v_pred, target) per sample                 │
│  • L_action = weighted_MSE(a_pred, target_a) × action_mask              │
│  • L_total = L_dynamics + L_action                                      │
│  • Backward: gradient flows through DiT → LoRA params only              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 一、各阶段详细数据量

### 1.1 原始数据集 (per sample, before batching)

| 模态 | 形状 | 数据类型 | 字节数 | 备注 |
|------|------|----------|--------|------|
| Video ×3 views | (25, 176, 320, 3) × 3 | uint8 | ~12.7 MB | MP4 解码后原始帧 |
| Action | (24, 7) | float32 | 672 B | 7-DoF 绝对位置 |
| State | (1, 7) | float32 | 28 B | joint pos(6) + gripper(1) |
| Language | variable (≤512 tokens) | string | ~200 B | 任务指令文本 |

> **注**：数据集存储为 parquet + MP4，网络传输量取决于 chunk 大小和压缩率。MP4 压缩率约 30-50×（12.7 MB → ~300 KB）。

### 1.2 Data Transforms 后 (per sample on CPU, before moving to NPU)

| 模态 | 形状 | 数据类型 | 字节数 |
|------|------|----------|--------|
| Video (rearranged) | (3, 33, 176, 320) | bf16 | ~11.2 MB |
| Video ×3 views | (3, 3, 33, 176, 320) | bf16 | ~33.5 MB |
| Action (relative) | (24, 7) | float32→bf16 | 336 B |
| State | (4, 7) | float32→bf16 | 56 B |
| Text tokens | (512,) | int64 | 4 KB |

### 1.3 VAE Encode 后 (on NPU, per grid sample)

| 模态 | 形状 | 数据类型 | 字节数 |
|------|------|----------|--------|
| Latents z_0 | (B, 16, 9, 22, 40) | bf16 | 0.62 MB |
| First frame condition y | (B, 20, 9, 22, 40) | bf16 | 0.76 MB |

**压缩比计算：**
- VAE input (grid): 3 × 33 × 176 × 320 = 5,575,680 像素 × 2 bytes (bf16) = 11.15 MB
- VAE output: 16 × 9 × 22 × 40 = 126,720 元素 × 2 bytes = 0.25 MB → **45× 压缩**
- 含 context (CLIP+T5): ~11.2 + ~0.6 + ~5 = ~16.8 MB → ~2.0 MB latent+context → **~8.4× 总压缩**

### 1.4 CLIP + T5 Encode 后 (per grid sample)

| 模态 | 形状 | 数据类型 | 字节数 |
|------|------|----------|--------|
| CLIP features | (B, 257, 1280) | bf16 | 0.63 MB |
| After MLPProj | (B, 257, 5120) | bf16 | 2.63 MB |
| T5 embeddings (raw) | (B, 512, 4096) | bf16 | 4.00 MB |
| After text_emb | (B, 512, 5120) | bf16 | 5.00 MB |

### 1.5 Patch Embedding 后 (per grid sample)

| 步骤 | 形状 | 数据类型 | 字节数 |
|------|------|----------|--------|
| Concat [x(16), y(20)] | (B, 36, 9, 22, 40) | bf16 | 1.36 MB |
| After Conv3d | (B, 5120, 9, 11, 20) | bf16 | 19.34 MB |
| Flatten (per branch) | (B, 1980, 5120) | bf16 | 19.34 MB |

> tokens_per_frame = (176/8)/2 × (320/8)/2 = 11 × 20 = **220**  
> seq_len = 9 × 220 = **1980** (per branch: clean or noisy)

### 1.6 完整 DiT 输入 Token 序列

| 组成部分 | Token 数 | Dim | 字节数 (bf16) |
|----------|----------|-----|---------------|
| Clean video tokens | 1,980 | 5,120 | 19.34 MB |
| Noisy video tokens | 1,980 | 5,120 | 19.34 MB |
| Action register | 24 | 5,120 | 0.24 MB |
| State register | 1 | 5,120 | 0.01 MB |
| **主序列总计** | **3,985** | **5,120** | **38.92 MB** |
| | | | |
| CLIP context (cross-attn) | 257 | 5,120 | 2.63 MB |
| T5 context (cross-attn) | 512 | 5,120 | 5.00 MB |
| **Context 总计** | **769** | **5,120** | **7.63 MB** |

---

## 二、DiT 单层数据运动量

### 2.1 SelfAttention（主序列 3985 tokens）

| 操作 | 计算 | 数据量 (bf16) |
|------|------|---------------|
| Q = Linear(5120→5120)(x) | 权重: 5120×5120 | 50 MB param, 78 MB act |
| K = Linear(5120→5120)(x) | 同上 | 50 MB param, 78 MB act |
| V = Linear(5120→5120)(x) | 同上 | 50 MB param, 78 MB act |
| RoPE apply | Q, K in-place 变换 | — |
| SDPA: Q×K^T | (B, 40, 3985, max_attn) | ~2.6 GB FLOPs, ~6 MB attn matrix |
| SDPA: ×V | (B, 40, 3985, 128) | — |
| O = Linear(5120→5120) | 权重: 5120×5120 | 50 MB param, 78 MB act |

**SelfAttention 数据运动合计（per layer）：**
- 参数读取 (LoRA): 4 × (5120×4 + 4×5120) × 2 bytes ≈ 328 KB (LoRA only, base frozen)
- 参数读取 (base, FSDP all-gather): 4 × 5120×5120 × 2 bytes / 8 cards = **25 MB**（all-gather 到每卡）
- 激活值: Q/K/V/O 各 ~78 MB (rw) = **~312 MB**
- **总计 per layer**: ~337 MB

### 2.2 CrossAttention（2 branches: CLIP 257 + T5 512）

| 操作 | 数据量 |
|------|--------|
| Q from main seq (3985) | 78 MB |
| K,V from CLIP (257) | 2 × 2.63 MB = 5.26 MB |
| K,V from T5 (512) | 2 × 5.00 MB = 10.00 MB |
| Output projection ×2 | 78 MB × 2 = 156 MB |
| **总计 per layer** | ~249 MB |

### 2.3 FFN + AdaLN

| 操作 | 计算 | 数据量 |
|------|------|--------|
| RMS Norm | (B, 3985, 5120) | 78 MB |
| Linear 5120→13824 | 权重: 5120×13824 | 141 MB param, 211 MB act |
| GELU | element-wise | — |
| Linear 13824→5120 | 权重: 13824×5120 | 141 MB param, 78 MB act |
| AdaLN modulation | 6 × 5120 | 120 KB |
| **总计 per layer** | | ~649 MB |

> **注**：以上为逻辑数据量。由于 FSDP `full_shard`，基座参数分布在 8 张卡上，每卡只需 all-gather 1/8 的参数。LoRA 参数 (bf16) 完全复制在每张卡上。

### 2.4 单层 DiT Block 汇总

| 组件 | 逻辑数据运动 | FSDP 均摊后（per card） |
|------|-------------|------------------------|
| SelfAttention | ~337 MB | ~160 MB |
| CrossAttention | ~249 MB | ~200 MB (context 不 shard) |
| FFN + AdaLN | ~649 MB | ~375 MB |
| **单层合计** | **~1.24 GB** | **~735 MB** |
| **40 层合计** | **~49.6 GB** | **~29.4 GB** |

---

## 三、FSDP 通信模式

### 3.1 FSDP 单元配置

| 参数 | 值 |
|------|-----|
| 总模型参数 | ~16.5B (含冻结编码器，不含) |
| DiT 参数量 | ~16.2B（blocks 16.15B + embeddings/head ~0.3B） |
| FSDP 单元数 | 41 (1 root + 40 × CausalWanAttentionBlock) |
| Per-block 参数 | **~403.8M** |
| Per-block 大小 (bf16) | **~807.7 MB** |
| Per-block per-card 分片 (8 NPU) | **~101 MB** |
| Root unit 参数 | ~336M (patch_emb, text_emb, time_emb, time_proj, head, encoders/decoders) |
| Root unit 大小 (bf16) | ~672 MB |

### 3.2 Per-Block 参数明细

| 组件 | 操作 | 参数量 |
|------|------|--------|
| Self-attn (q,k,v,o) | 4 × Linear(5120,5120) | ~104.9M |
| Cross-attn (q,k,v,o,k_img,v_img) | 6 × Linear(5120,5120) | ~157.3M |
| FFN (ffn.0, ffn.2) | Linear(5120,13824) + Linear(13824,5120) | ~141.6M |
| RMS norms + modulation | 5×RMSNorm + nn.Parameter(6,5120) | ~0.07M |
| **Per block 合计** | | **~403.8M** |
| **40 blocks 合计** | | **~16.15B** |

### 3.3 All-Gather 通信（Forward）

每次 forward 经过一个 FSDP 单元时（per block, per card）：
1. All-gather: 7 张卡各送 ~101 MB → 每卡收 ~707 MB（凑齐完整 ~808 MB block）
2. 该卡用完整参数计算 forward
3. 释放 all-gathered 副本（或保留到 backward）

**Per step forward (40 blocks + root unit):**
- All-gather 发送: 40 × 707 + 588 = **28.9 GB** per card
- All-gather 接收: **28.9 GB** per card

### 3.4 Reduce-Scatter 通信（Backward）

Backward 计算完梯度后：
- All-gather（获取完整参数用于梯度计算）：**28.9 GB** send + **28.9 GB** recv per card
- Reduce-scatter（梯度均摊回各卡）：**28.9 GB** send + **~3.6 GB** recv per card

### 3.5 总 HCCL 通信量 (per training step, per card)

| 方向 | Forward AG (send/recv) | Backward AG (send/recv) | RS (send/recv) | **合计** |
|------|----------------------|------------------------|----------------|----------|
| Send | 28.9 GB | 28.9 GB | 28.9 GB | **86.7 GB** |
| Recv | 28.9 GB | 28.9 GB | 3.6 GB | **61.4 GB** |
| **双向总计** | | | | **~148 GB** |

> 注：这是逻辑数据量下限。Ring all-gather 实现引入的额外开销（分段传输、等待）可能使实际通信量更高。

---

## 四、梯度检查点 (Gradient Checkpointing)

### 4.1 激活值内存

| 场景 | 保存内容 | 激活内存 (per card) |
|------|----------|---------------------|
| 无 checkpoint | 全部 40 层中间激活 | ~15-20 GB |
| Checkpoint (use_reentrant=True) | 仅 checkpoint 边界 | ~3-5 GB |

**当前配置**: `use_reentrant=False`（非重入 autograd Function）。每层 DiT block 的 forward 中只保存输入 tensor `x`，所有中间激活（QKV 投影、attention scores、FFN 中间值、modulation 输出）在 backward 时重新计算。

### 4.2 Recompute 开销

- Forward pass: 1× (正常 forward，只保存 checkpoint 边界 tensor `x`)
- Backward pass: 每层重新 forward + backward: 1× + 1× = 2×
- **总 compute**: 3× forward + 1× backward = ~4× 纯 forward
- **Saved tensors**: 40 × [B, 3985, 5120] bf16 ≈ 40 × 38.9 MB ≈ **1.56 GB**（仅输入，不含 context/e0/freqs）

---

## 五、LoRA 可训练参数

### 5.1 LoRA 配置

| 参数 | 值 |
|------|-----|
| Rank | 4 |
| Alpha | 4 |
| 目标模块 | q, k, v, o, ffn.0, ffn.2 |
| 注入层数 | 40 blocks × 10 modules = 400 LoRA pairs |
| ⚠️ 注意 | PEFT 按后缀匹配 → q,k,v,o 同时注入 self_attn 和 cross_attn |

### 5.2 参数量计算

**每个 Linear(5120, 5120) 适配器（q, k, v, o）：**
- lora_A: 4 × 5120 = 20,480
- lora_B: 5120 × 4 = 20,480
- 合计: 40,960

**每个 ffn.0 适配器（Linear(5120, 13824)）：**
- lora_A: 4 × 5120 = 20,480
- lora_B: 13824 × 4 = 55,296
- 合计: 75,776

**每个 ffn.2 适配器（Linear(13824, 5120)）：**
- lora_A: 4 × 13824 = 55,296
- lora_B: 5120 × 4 = 20,480
- 合计: 75,776

**Per block：**
| 模块组 | 数量 | 每个 | 小计 |
|--------|------|------|------|
| q (self+cross) | 2 | 40,960 | 81,920 |
| k (self+cross) | 2 | 40,960 | 81,920 |
| v (self+cross) | 2 | 40,960 | 81,920 |
| o (self+cross) | 2 | 40,960 | 81,920 |
| ffn.0 | 1 | 75,776 | 75,776 |
| ffn.2 | 1 | 75,776 | 75,776 |
| **Per block 合计** | **10** | | **479,232** |

**40 blocks LoRA 合计**: 19,169,280 (~19.2M)

### 5.3 非 LoRA 可训练参数

| 模块 | 参数量 | 备注 |
|------|--------|------|
| state_encoder (CategorySpecificMLP) | ~10.6M | input=64, hidden=1024, output=5120 |
| action_encoder (MultiEmbodimentActionEncoder) | ~78.8M | CategorySpecificLinear ×3 |
| action_decoder (CategorySpecificMLP) | ~5.3M | input=5120, hidden=1024, output=7 |
| **非 LoRA 小计** | **~94.7M** | |

### 5.4 总计

| 类别 | 参数量 | bf16 大小 |
|------|--------|-----------|
| LoRA (40 blocks) | 19.2M | 38.4 MB |
| 非 LoRA 可训练 | 94.7M | 189.4 MB |
| **总可训练参数** | **~113.9M** | **~227.8 MB** |
| 基座冻结 (DiT + embeddings) | ~16.4B | ~32.7 GB |
| **模型总参数** | **~16.5B** | **~33.0 GB** |

**优化器状态 (AdamW, fp32 per card, 8 NPU FSDP sharded):**
- exp_avg (fp32): 227.8 × 2 = 455.6 MB → /8 = 57.0 MB
- exp_avg_sq (fp32): 227.8 × 2 = 455.6 MB → /8 = 57.0 MB
- 参数 (bf16): 227.8 MB → /8 = 28.5 MB
- **每卡优化器+参数总计**: ~142.5 MB
- 加上 frozen 参数分片 (16.4B/8 × 2 bytes): ~4.1 GB

---

## 六、GPU 显存占用分解 (per NPU card, 61.28 GiB total)

| 组件 | 大小 | 备注 |
|------|------|------|
| DiT 参数分片 (14B / 8) | ~3.5 GB | bf16, FSDP sharded |
| LoRA 参数 | ~25 MB | bf16, 完全复制 |
| LoRA 梯度 | ~50 MB | fp32 (bf16 params × 2) |
| LoRA 优化器状态 | ~150 MB | fp32 × 3 (AdamW momenta) |
| 其他可训练参数 | ~147 MB | bf16 |
| 其他可训练梯度 | ~294 MB | fp32 |
| 其他可训练优化器 | ~882 MB | fp32 × 3 |
| All-gather buffer (peak 2 layers) | ~1.4 GB | 2 × 700 MB, bf16 |
| 激活值 (gradient checkpoint) | ~4 GB | 含 clean+noisy token seq |
| CLIP/T5/VAE (CPU offloaded) | 0 | 编码后释放 |
| **总计** | **~10.5 GB** | |
| **利用率** | **~17%** | 61.28 GB / 10.5 GB |

> 实测 (v40 smoke test): Forward baseline 15.70 GB, backward peak ~19.46 GB。剩余空间可用于增大 batch size。

---

## 七、端到端吞吐量分析

### 7.1 每步数据流汇总

| 阶段 | 数据量 | 方向 | 位置 |
|------|--------|------|------|
| 数据加载 (per sample) | ~13.9 MB raw → ~35 MB transformed | CPU→CPU | CPU |
| CPU→NPU 传输 (video) | ~33.5 MB | PCIe | 跨总线 |
| VAE encode | 33.5 MB → 1.21 MB | NPU internal | HBM |
| CLIP encode | ~0.67 MB → ~0.63 MB | NPU internal | HBM |
| T5 encode | ~4 KB → ~4 MB | NPU internal | HBM |
| Patch embedding | 2.7 MB → 38.7 MB (×2) | NPU internal | HBM |
| FSDP all-gather (fwd) | 3.6 GB send, 25.1 GB recv | HCCL | 跨卡 |
| DiT forward (40 layers) | ~29.4 GB data moved | NPU internal | HBM |
| Loss + backward | compute + gradient flow | NPU+HCCL | HBM+跨卡 |
| FSDP reduce-scatter (bwd) | 25.1 GB send+recv | HCCL | 跨卡 |
| Optimizer step | ~172 MB params updated | NPU internal | HBM |

### 7.2 时间分解 (实测 ~32s/step, 8 NPU)

| 阶段 | 估算时间 | 瓶颈 |
|------|----------|------|
| 数据加载 + transforms | ~2-3s | CPU, disk I/O |
| VAE/CLIP/T5 encode | ~3-5s | NPU compute |
| DiT forward (40 layers) | ~8-10s | NPU compute + HCCL |
| DiT backward + gradient checkpoint | ~12-15s | NPU compute + HCCL |
| HCCL all-gather + reduce-scatter | ~2-3s | 跨卡带宽 |
| Optimizer step | ~0.5-1s | NPU compute |
| **总计** | **~32s** | |

### 7.3 HCCL 带宽利用率

- Per step 单向总发送: ~86.7 GB per card
- 假设 HCCS 有效带宽: ~50 GB/s (单向)
- 纯通信时间下限: 86.7 / 50 ≈ 1.7s
- 实际通信时间占比: ~1.7/32 ≈ 5.3%
- **结论：通信不是瓶颈，compute-bound**

---

### 7.4 NPU FSDP Cleanup（关键适配）

NPU 上 `_post_backward_final_callback` 不触发 → FSDP 状态残留 → Step 1+ OOM。
每个 training_step 执行三段手动清理（`base.py`）：

1. **重置执行顺序**：`_exec_order_data._iter = 0`, `handles_post_forward_order.clear()`
2. **删除残留 hook state**：`flat_param._post_backward_hook_state` → 重新注册
3. **重置训练状态**：`training_state → IDLE`, `_ran_pre_backward_hook = False`

额外 monkey-patch `_assert_in_training_states` 放宽 gradient checkpointing 时的状态检查。

---

## 八、关键指标卡 (Per Training Step, B=1, 8 NPU)

| 指标 | 数值 |
|------|------|
| 模型总参数 | ~16.5B (含冻结编码器) |
| DiT Block 参数 (40 blocks) | ~16.15B (~403.8M/block) |
| 可训练参数 | ~113.9M (0.69%) — LoRA 19.2M + Encoder/Decoder 94.7M |
| LoRA adapters | 400 (10/block: q,k,v,o×2 + ffn.0,ffn.2) |
| 视频 token 数 | 3,960 (clean 1980 + noisy 1980) |
| Action token 数 | 24 |
| State token 数 | 1 |
| 主序列总 token 数 | 3,985 |
| Context token 数 (cross-attn) | 769 |
| Per-token 维度 | 5,120 |
| 主序列大小 (bf16) | **~38.9 MB** |
| Context 大小 (bf16) | **~7.9 MB** |
| FSDP 单元数 | 41 (40 blocks + 1 root) |
| Per-block 参数 (bf16) | ~807.7 MB |
| Per-card per-block shard | ~101 MB |
| All-gather per step per card (fwd) | 28.9 GB send + 28.9 GB recv |
| All-gather per step per card (bwd) | 28.9 GB send + 28.9 GB recv |
| Reduce-scatter per step per card | 28.9 GB send + 3.6 GB recv |
| HCCL 总通信量 (per step, per card) | **~148 GB 双向** |
| 激活峰值内存 (per card) | ~19.5 GB |
| 优化器状态 (per card, sharded) | ~142 MB (仅可训练参数) |
| 训练时间 (per step) | ~32s |
| 有效吞吐量 | ~0.031 steps/s |
| 1000 steps 预计时间 | ~8.9 hours |

---

## 九、优化空间分析

### 9.1 增大 Batch Size

当前 per_device_batch_size=1, num_views=3。有效 batch = 3 samples/step。
- 显存余量: 61.28 - 19.5 ≈ 41.8 GB
- 增大到 batch_size=2: 激活 ~39 GB → 仍在余量内
- **预期加速**: ~1.5-1.7× (通信摊薄)

### 9.2 Gradient Checkpointing 调优

- 当前 `use_reentrant=True`：安全但保守
- 改为 `use_reentrant=False`：减少 recompute，但 NPU pre-backward hook 有已知问题
- **不建议在 NPU 上修改**（Fix 7 相关，NPU 的 queue_callback 不触发）

### 9.3 FSDP 参数调优

- `backward_prefetch=no_prefetch`（当前保守设置）
- 改为 `BACKWARD_PRE`：预取下一层参数，可能减少等待时间
- **风险**：NPU pre-backward hook 兼容性问题

### 9.4 混合精度

- 当前 bf16 训练 + fp32 优化器
- Pure bf16 优化器可省 ~340 MB per card（但精度损失未知）

### 9.5 CPU Offload

- T5/CLIP/VAE 已在首次编码后卸载到 CPU → **已优化**
- 编码结果（context）保留在 NPU → 用于 cross-attention

---

## 十、与原始 DreamZero 论文的差异

| 维度 | 论文 | 本实现 |
|------|------|--------|
| 硬件 | NVIDIA GPU (H100/A100) | Huawei Ascend 910 NPU |
| FSDP 后端 | NCCL | HCCL (HCCS 互联) |
| Flash Attention | FA2/3 | SDPA 降级 |
| RoPE | complex/polar | real-valued (NPU 不支持 complex) |
| 分布式框架 | DeepSpeed ZeRO-2 可选 | FSDP only |
| communication | NVLink/NVSwitch | HCCS |
| batch_size | 未明确 | 1 per device |
| 显存 | 80 GB (H100) | 61.28 GB (Ascend 910) |
| 可训练参数 | 全量 16.5B (?)| LoRA 114M (0.69%) |

---

## 十一、全量训练 (Full Fine-Tuning) 性能建模

> **场景假设**：16.5B DiT 参数全部可训练，LoRA 移除，其余配置不变。
> B=1, 8× Ascend 910 (61.28 GB), bf16 混合精度, FSDP full_shard, gradient checkpointing。

### 11.1 与 LoRA 训练的核心差异

| 维度 | LoRA 训练 | 全量训练 | 倍数 |
|------|----------|---------|------|
| 可训练参数 | 113.9M | 16,500M | **145×** |
| 梯度参数量 (per step) | 113.9M fp32 | 16,500M fp32 | **145×** |
| 优化器状态总量 | 113.9M × 8B = 0.91 GB | 16,500M × 8B = 132 GB | **145×** |
| 参数更新量 (per step) | 113.9M bf16 | 16,500M bf16 | **145×** |
| Forward 计算量 | 相同 | 相同 | 1× |
| Backward 计算量 | 仅 LoRA grad | 全量 grad | ~3-4× |
| 激活值内存 | 相同 | 相同 | 1× |
| 前向数据流 | 相同 | 相同 | 1× |

### 11.2 显存占用分解 (per NPU card, 8-way FSDP sharding)

| 组件 | LoRA 训练 | 全量训练 | 增量 |
|------|----------|---------|------|
| DiT 参数分片 (bf16) | ~4.1 GB | ~4.1 GB | — |
| 可训练参数副本 (bf16, 非 shard) | ~0.23 GB | — | -0.23 GB |
| Master 参数 (fp32, sharded) | — | ~8.25 GB | +8.25 GB |
| 梯度 (fp32, per-unit peak) | ~0.01 GB | ~0.81 GB | +0.80 GB |
| 优化器 exp_avg (fp32, sharded) | ~0.06 GB | ~8.25 GB | +8.19 GB |
| 优化器 exp_avg_sq (fp32, sharded) | ~0.06 GB | ~8.25 GB | +8.19 GB |
| All-gather buffer (peak 1 block) | ~0.81 GB | ~0.81 GB | — |
| Saved checkpoint inputs (40 blocks) | ~1.56 GB | ~1.56 GB | — |
| Persistent tensors (context, e0) | ~0.32 GB | ~0.32 GB | — |
| Recompute intermediates (peak) | ~0.21 GB | ~0.21 GB | — |
| 其他开销 (CUDA context, etc.) | ~2 GB | ~2 GB | — |
| **总计 per card** | **~9.4 GB** | **~34.6 GB** | **+25.2 GB** |
| **HBM 利用率 (61.28 GB)** | **15%** | **56%** | |

> **关键结论：全量训练在 8× Ascend 910 上可以跑，显存利用率 ~56%。余量约 26.7 GB，batch_size=1 安全，batch_size=2 需验证。**

### 11.3 优化器状态详细分解

全量训练下 AdamW 优化器追踪所有 16.5B 参数：

| 状态 | 精度 | 总大小 | Per-card (8-way shard) |
|------|------|--------|------------------------|
| Master params | fp32 | 66.0 GB | 8.25 GB |
| exp_avg (momentum) | fp32 | 66.0 GB | 8.25 GB |
| exp_avg_sq (variance) | fp32 | 66.0 GB | 8.25 GB |
| **优化器状态合计** | | **198.0 GB** | **24.75 GB** |

对比 LoRA 训练（仅可训练参数有优化器状态）：
- LoRA 优化器状态总量: 113.9M × 12 bytes = 1.37 GB → /8 = 0.17 GB per card
- **全量 / LoRA = 24.75 / 0.17 ≈ 145×**

### 11.4 HCCL 通信量变化

前向通信 (all-gather) 完全相同（参数相同）。反向通信显著增大：

**Reduce-Scatter（梯度汇聚）：**

以单个 403.8M-param block 为单位：
- Block fp32 梯度: 403.8M × 4 bytes = 1,615 MB
- 分为 8 shards: 每 shard ~202 MB
- Per card send: 7 shards × 202 MB = 1,414 MB
- Per card recv: 7 cards × 202 MB = 1,414 MB

40 blocks + root unit (~336M params, ~1,344 MB fp32 gradients):
- Root unit per card RS: ~1,176 MB send, ~1,176 MB recv

| 通信阶段 | LoRA | 全量 | 增量 |
|----------|------|------|------|
| Forward AG send | 28.9 GB | 28.9 GB | — |
| Forward AG recv | 28.9 GB | 28.9 GB | — |
| Backward AG send | 28.9 GB | 28.9 GB | — |
| Backward AG recv | 28.9 GB | 28.9 GB | — |
| Reduce-scatter send | ~0.5 GB | **~57.7 GB** | **115×** |
| Reduce-scatter recv | ~0.5 GB | **~57.7 GB** | **115×** |
| **Per card 单向发送总计** | **~58.3 GB** | **~115.5 GB** | **1.98×** |
| **Per card 单向接收总计** | **~58.3 GB** | **~115.5 GB** | **1.98×** |
| **Per card 双向总计** | **~116.6 GB** | **~231.0 GB** | **1.98×** |

> 注：LoRA 训练的 reduce-scatter 仅传输可训练参数的梯度。全量训练传输全部 16.5B 参数的梯度。

### 11.5 Per-Step 时间估算

| 阶段 | LoRA (32s) | 全量 (估算) | 变化原因 |
|------|-----------|------------|----------|
| 数据加载 + transforms | ~2-3s | ~2-3s | 相同 |
| VAE/CLIP/T5 encode | ~3-5s | ~3-5s | 相同 |
| DiT forward (40 layers) | ~8-10s | ~8-10s | 相同 |
| DiT backward + GC recompute | ~12-15s | **~18-25s** | 全量梯度计算 + 全量 reduce-scatter |
| HCCL reduce-scatter | ~0.2s | **~1.2-1.5s** | 57.7 GB vs 0.5 GB per card |
| Optimizer step | ~0.5-1s | **~5-8s** | 更新 16.5B vs 114M 参数 |
| **总计** | **~32s** | **~40-55s** | **1.25-1.7×** |

> HCCS 有效带宽按 50 GB/s 计算，纯 RS 通信时间下限: 57.7/50 ≈ 1.15s。  
> Optimizer step 时间取决于 NPU 内存带宽和 AdamW 实现的效率，8.25 GB 参数更新约需 5-8s。

### 11.6 16 NPU 场景

Ascend 910 服务器有 16 个物理 NPU。扩展到 16 卡：

| 指标 | 8 NPU | 16 NPU | 变化 |
|------|-------|--------|------|
| Per-card 参数分片 | 4.1 GB | 2.06 GB | ½× |
| Per-card 优化器状态 | 24.75 GB | 12.38 GB | ½× |
| Per-card 总显存 | ~34.6 GB | ~20.3 GB | 0.59× |
| HBM 利用率 | 56% | 33% | |
| Per-card HCCL send | 115.5 GB | ~65 GB | 0.56× |
| 有效 batch size | 8 | 16 | 2× |
| Per-step 时间 (估算) | ~40-55s | ~45-60s | 通信拓扑变化 |

> 16 NPU 场景显存更宽裕 (33%)。HCCL 通信拓扑从 8 卡变为 16 卡，ring all-gather 跳数增加，但每卡传输量减少。总通信时间可能相近或略增。

### 11.7 全量训练 vs LoRA 训练：决策矩阵

| 维度 | LoRA 训练 | 全量训练 |
|------|----------|---------|
| 显存需求 (per card) | ~9.4 GB ✅ | ~34.6 GB ✅ |
| 训练速度 | ~32s/step | ~40-55s/step (1.25-1.7× slower) |
| Checkpoint 大小 | ~228 MB | ~66 GB (fp32 master) / ~33 GB (bf16) |
| Checkpoint 保存时间 | ~1-2s | ~30-60s (I/O bound) |
| 灾难恢复 | 快速（小 checkpoint） | 慢（大 checkpoint） |
| 微调灵活性 | 可快速切换任务 | 需保存完整权重 |
| 过拟合风险 | 低（仅 0.69% 参数） | 高（需更强正则化） |
| 精度上限 | 受 LoRA rank 限制 | 理论上限更高 |
| 数据效率 | 小数据集友好 | 需大量数据 |

### 11.8 全量训练的 Checkpoint 策略

| 策略 | 大小 | 保存时间 | 恢复方式 |
|------|------|----------|----------|
| Full state dict (fp32) | ~66 GB | ~60-120s | 完整恢复 |
| Full state dict (bf16) | ~33 GB | ~30-60s | 完整恢复 |
| Sharded checkpoint (8 shards) | ~8.25 GB/shard | ~10-15s/shard | 需同数量 GPU 恢复 |
| FSDP checkpoint (distributed) | ~8.25 GB/卡同步写 | ~10-20s | `torch.distributed.checkpoint` |

> 推荐使用 sharded/distributed checkpoint 方式。每个 checkpoint 约 8.25 GB per card × 8 cards = 66 GB total。
> 1000 steps 保存一次 → 10 checkpoints → 660 GB 磁盘。需确保 `/checkpoints` 分区 (294 GB) 有足够空间或使用外部存储。

### 11.9 数据流不变部分

以下阶段在全量训练中**完全不变**（因为 forward pass 相同）：

```
Stage 0:  DreamTransform 网格拼接
Stage 1:  Video 预处理 (resize, normalize)
Stage 2:  VAE Encode
Stage 3:  CLIP + T5 Encode
Stage 4:  Flow Matching Noise
Stage 5:  Patch Embedding
Stage 6:  Token Assembly (Teacher Forcing)
Stage 7:  DiT × 40 Forward (compute-wise)
Stage 8:  Head + Unpatchify
Stage 9:  Loss Computation
```

**变化的阶段**：
- **Stage 7 反向**：梯度计算量 145×（所有参数，非仅 LoRA）
- **HCCL Reduce-Scatter**：通信量 ~115×（全量梯度 vs LoRA 梯度）
- **Optimizer Step**：更新量 145×（16.5B vs 114M 参数）
- **Checkpoint I/O**：保存量 ~145×（66 GB vs 0.46 GB）

---

*文档生成日期：2026-07-21*  
*基于 commit 1834044 的训练配置和实测数据*
