# DreamZero 训练数据流与模型管线（Markdown 总览）

本文档从**磁盘上的 LeRobot 标注/数据**出发，串联 **Dataset → 预处理 → Collator → `VLA` 前向 → `WANPolicyHead` / `CausalWanModel` → Loss → Trainer**，并说明 **冻结策略、训练阶段与优化配置**。路径均以仓库 `groot/vla/` 为根。

---

## 1. 磁盘上的「原始」数据由谁读、读什么

### 1.1 数据根目录（LeRobot v2 布局）

典型 DROID 转换后目录包含：

| 位置 | 内容 |
|------|------|
| `meta/info.json` | `data_path`、`video_path`、`chunks_size`、`features` 等 |
| `meta/modality.json` | state/action/video 子键与 `start`/`end`、`original_key` |
| `meta/episodes.jsonl` | 每行 `episode_index`、`length`、`tasks`、`success` |
| `meta/tasks.jsonl` | `task_index` → 自然语言 `task` 字符串 |
| `data/chunk-XXX/episode_YYYYYY.parquet` | 每 episode 一步一行：低维 state/action/timestamp/语言 task 索引等 |
| `videos/chunk-XXX/observation.images.<cam>/episode_YYYYYY.mp4` | 各相机视频 |

### 1.2 读取这些文件的类

| 类 | 文件 | 职责 |
|----|------|------|
| **`LeRobotSingleDataset`** | `groot/vla/data/dataset/lerobot.py` | 读 `meta/*`、按 `get_parquet_path` / `get_video_path` 打开 **parquet** 与 **mp4**；`__getitem__` 用 `ModalityConfig.delta_indices` 取窗口；`get_step_data` 组装多模态字典 |
| **`ShardedLeRobotSubLangSingleActionChunkDatasetDROID`** | `groot/vla/data/dataset/lerobot_sharded.py` | 继承上者；预计算 **shard**（按步数切块）、缓存 parquet/视频路径，供流式 `IterableDataset` 高效顺序读 |
| **`ShardedLeRobotMixtureDataset`** | 同上 | `IterableDataset`：按权重在多个子数据集间采样，委托对应 **single dataset** 的 `__getitem__` |

配置入口（DROID 相对动作示例）：`groot/vla/configs/data/dreamzero/droid_relative.yaml`  
- `train_dataset._target_`: `ShardedLeRobotMixtureDataset.from_mixture_spec`  
- `dataset_class`: `ShardedLeRobotSubLangSingleActionChunkDatasetDROID`  
- `dataset_path` 指向磁盘上的 LeRobot 根目录（如 `droid_data_root`）

### 1.3 语言「标注」如何从整数变字符串

Parquet 中 `annotation.language.*` 常为 **task_index（int）**。  
`lerobot.py` 中 `get_data_by_modality` 对数值型语言列会用 **`self.tasks.loc[indices]["task"]`** 转成自然语言字符串（见 `meta/tasks.jsonl`）。

---

## 2. 单条 Train Sample（Dataset `__getitem__` 经 Transform 后）

### 2.1 Dataset 输出（进入 `ComposedModalityTransform` 之前）

逻辑与 `LeRobotSingleDataset.__getitem__` 一致：

- 键为 **`video.*` / `state.*` / `action.*` / `annotation.language.*`** 等（由 modality 配置决定）。
- **Video**：`(T_video, V, H, W, C)`，`uint8`（经 Video 管线后）。
- **State / Action**：拼接后的 `(T_state, D_state)`、`(T_action, D_action)`（DROID 示例：`T_video=25`，`T_action=24`，`T_state=1`）。
- **Language**：字符串或可从 task 表解析出的文本。

若开启 **`relative_action`**（`droid_relative.yaml`），在 dataset 内对指定 action 子键做**相对动作**处理（见 `lerobot.py` 中 relative 逻辑）。

### 2.2 `ComposedModalityTransform`（以 `transform_oxe_droid` 为例）

定义于 `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`：

| 步骤 | 类 | 作用 |
|------|-----|------|
| 视频 | `VideoToTensor` → `VideoCrop` → `VideoResize` → `VideoColorJitter` → `VideoToNumpy` | 转张量、随机裁剪、缩放到配置分辨率、增强、回到 `uint8` numpy |
| State/Action | `StateActionToTensor` → `StateActionTransform` | 按 `normalization_modes`（如 `q99`）用 **stats** 归一化；通常映射到约 **[-1, 1]** 量级供 Flow Matching 使用 |
| 拼接 | `ConcatTransform` | 多 view → `(T, V, H, W, C)`；多 state/action 键沿最后一维 concat |

### 2.3 `DreamTransform`（`model_specific_transform`）

文件：`groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`

单样本 `apply_single` 产出字典（训练时）要点：

| 键 | 含义 | Shape（单样本，训练） |
|----|------|------------------------|
| `images` | DROID：三相机拼成 **1 路「2×2 拼图」**（腕部顶行加宽、双外参底行）；`uint8` | `(1, T, C, 2H, 2W)` |
| `text` | 选中的语言字符串（多 key 时 DROID 可随机选一）；**Collator 里再 tokenize** | 标量 str |
| `state` | 填充到 `max_state_dim` | `(state_horizon, max_state_dim)` |
| `state_mask` | 真实维度为 True | 同 state |
| `action` | 填充到 `max_action_dim` | `(action_horizon, max_action_dim)` |
| `action_mask` | 有效动作维为 True | 同 action |
| `has_real_action` | 是否对主 action 算 flow loss | 标量 `bool` |
| `embodiment_id` | 本体标签整数（映射表来自 config） | 标量 |
| `text_negative` | CFG 用负向提示（固定长句） | str |
| `is_cotrain_instance` 等 | 多任务分支标记 | 标量 |

`_apply_vlm_processing` 将 `images` 从 `(1,T,C,H,W)` 变为 **`(T*1, H, W, C)`** 即 `(T, H, W, C)` 的扁平视图供后续拼 batch（与 `num_views` 配置一致时扩展）。

---

## 3. Collator → Train Batch

### 3.1 类与入口

- **`DefaultDataCollator`**：`groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`  
- 在 `dreamzero_cotrain.yaml` 中作为 `data_collator` 挂到 **Hydra**；**`VLATrainer.compute_loss`** 收到的 `inputs` 即此输出。

### 3.2 `collate()` 行为摘要

- **`text`**：`HuggingfaceTokenizer`（如 `umt5-xxl`），`padding='max_length'`，`max_length=seq_len`（如 512）。  
  - 对 **OXE_DROID** 会拼接固定前缀描述多视角布局（见 `collate` 内分支）。  
  - 输出 **`text`**: `(B, L_text)`，`int64`；**`text_attention_mask`**: `(B, L_text)`，1=有效 token。
- **其它键**（`images`, `state`, `action`, mask, `embodiment_id` 等）：`np.stack` → `torch.from_numpy` → 形状前加 **`B`**。

### 3.3 Batch 张量含义与典型维度（需与当前 Hydra 覆盖一致）

以下与 `scripts/train/droid_training_full_finetune.sh` 中示例一致时：

| 键 | Shape | 维度含义 |
|----|--------|----------|
| `images` | `(B, T, H, W, C)` 或等价 | B=batch；T=时间帧；H,W=拼图后高宽；C=3 RGB |
| `text` | `(B, L_text)` | token id |
| `text_attention_mask` | `(B, L_text)` | padding mask |
| `state` | `(B, state_horizon, max_state_dim)` | 本体状态（已归一化） |
| `state_mask` | `(B, state_horizon, max_state_dim)` | 有效维 |
| `action` | `(B, action_horizon, max_action_dim)` | 目标动作序列（已归一化，须在 [-1,1] 内进 head） |
| `action_mask` | `(B, action_horizon, max_action_dim)` | 有效维 |
| `embodiment_id` | `(B,)` | 本体 ID |
| `has_real_action` | `(B,)` | 是否参与 action 分支 loss |

**说明**：`max_state_dim` / `max_action_dim` / `num_frames` / `action_horizon` 由 **data + model** 的 Hydra 配置共同约束，需与 `WANPolicyHead` 和 `CausalWanModel` 的块参数一致（如 `num_frame_per_block`、`num_action_per_block`）。

---

## 4. 模型入口：`VLA` 与输入划分

文件：`groot/vla/model/dreamzero/base_vla.py`

```text
inputs (dict)
  → prepare_input()
      → backbone_inputs = backbone.prepare_input(inputs)
      → action_inputs   = action_head.prepare_input(inputs)
  → backbone_outputs  = backbone(backbone_inputs)
  → action_head_outputs = action_head(backbone_outputs, action_inputs)
```

### 4.1 `IdentityBackbone`

文件：`groot/vla/model/dreamzero/backbone/identity.py`

- **`prepare_input`**：若有 `action`，返回 `BatchFeature({"action": batch["action"]})`；否则可用 `state`/`video` 仅推断 batch 维。
- **`forward`**：返回 **`backbone_features`**，形状 **`(B, 1, 0)`** 的空特征（占位）；**真实视觉与语言不经过独立 backbone，全部在 `WANPolicyHead` 内处理**。

### 4.2 `WANPolicyHead.prepare_input`

文件：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`  
- **`prepare_input(batch)`**：`BatchFeature(data=batch)`，即把 collator 的 dict 原样包成 `BatchFeature`（键名与 `forward` 中读取的一致：`images`, `text`, `text_attention_mask`, `state`, `action`, `action_mask`, `embodiment_id`, `has_real_action` 等）。

---

## 5. `WANPolicyHead.forward`：子模块、算子与中间张量

文件：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

### 5.1 冻结与模式

- **`set_frozen_modules_to_eval_mode()`**：训练时仍把冻结模块置于 eval（如 BN 行为）。
- **`text_encoder` / `image_encoder` / `vae`**：`requires_grad=False`（预训练加载权重，**不训练**）。
- **`train_architecture`**：  
  - **`"lora"`**（默认 `wan_flow_matching_action_tf.yaml`）：对 **`CausalWanModel`** 注入 **PEFT LoRA**，主干权重冻结，仅 LoRA + **`state_encoder` / `action_encoder` / `action_decoder`** 等可训。  
  - **`"full"`**（如 `droid_training_full_finetune.sh` 中 `train_architecture=full`）：不注入 LoRA，由 **`tune_projector` / `tune_diffusion_model`** 控制；全量微调时二者为 true 则 **整颗 DiT 可训**（仍不训 T5/CLIP/VAE）。

### 5.2 前向主要步骤（训练）

| 顺序 | 模块 | 文件 | 输入 → 输出 | 主要算子/操作 |
|------|------|------|-------------|----------------|
| 1 | 图像归一化 | 同文件 `forward` | `images` uint8 → float，`/255`，`Normalize(0.5,0.5,0.5)` | 除法、`torchvision.transforms.v2.Normalize` |
| 2 | **`WanTextEncoder`** | `groot/vla/model/dreamzero/modules/wan_video_text_encoder.py` | `text`, `text_attention_mask` → **`prompt_embs`** | Transformer 编码（bf16）；padding 位置置 0 |
| 3 | **`WanVideoVAE.encode`** | `groot/vla/model/dreamzero/modules/wan_video_vae.py` | `videos` `(B,C,T,H,W)` → **`latents`** | 卷积 VAE 编码，`torch.no_grad` |
| 4 | **`WanImageEncoder` + VAE** | `wan_video_image_encoder.py` + VAE | 首帧 → **`clip_feas`**, **`ys`**（I2V 条件潜变量+mask） | CLIP 图像编码；首帧潜变量与 mask 拼接 |
| 5 | **`FlowMatchScheduler`** | `groot/vla/model/dreamzero/modules/flow_match_scheduler.py` | 对 latents / actions 采样 **timestep**，**`add_noise`**，**`training_target`** | 流匹配噪声调度；可与 video/action **耦合或解耦** timestep（见 config） |
| 6 | **`CausalWanModel`**（训练分支 `_forward_train`） | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` | `noisy_latents`, `timestep`, `clip_feature`, `y`, `context`, `seq_len`, `state`, `action`, `timestep_action`, `clean_x` → **`video_noise_pred`**, **`action_noise_pred`** | Patch embed；**`action_encoder`/`state_encoder`**；多层 **`CausalWanAttentionBlock`**（自注意力+交叉注意力+FFN，FlexAttention/块因果 mask）；**`CausalHead`** 出视频支路；**`action_decoder`** 出动作支路；**`unpatchify`** 还原潜空间形状 |
| 7 | Loss | 同 `WANPolicyHead.forward` | 与 `training_target` 对齐 | **`F.mse_loss`**（按 scheduler **`training_weight(timestep)`** 加权）；视频：**`dynamics_loss`**；动作：乘 **`action_mask`** 与 **`has_real_action`** → **`action_loss`**；`loss = dynamics_loss + action_loss` |

### 5.3 `CausalWanModel` 训练输出形状（摘要）

- **`video_noise_pred`**：与 **VAE latent** 同空间，形如 **`(B, C_latent, T_latent, H_latent, W_latent)`**（与 `unpatchify` 及 `patch_size` 一致）。  
- **`action_noise_pred`**：**`(B, action_horizon, action_dim)`**，与 flow matching 的 action 目标同形。

### 5.4 `WANPolicyHead.forward` 返回值（训练）

`BatchFeature`：

- **`loss`**：标量，反传用。  
- **`dynamics_loss`**：视频/潜空间分支加权 MSE。  
- **`action_loss`**：动作分支加权 MSE（无 action 时为 0）。

---

## 6. 得到网络输出之后的流程

1. **`VLATrainer.compute_loss`**（`groot/vla/experiment/base.py`）：`outputs = model(inputs)`，取 **`outputs["loss"]`**；附加记录 `*_loss` 的移动平均日志。  
2. **反向传播**：标准 HuggingFace `Trainer` + 可选 **DeepSpeed**（如 `zero2_offload.json`）。  
3. **优化器**：`TrainingArguments` 中 `optim`（如 `adamw_torch`）、`learning_rate`、`weight_decay`、`lr_scheduler_type`（如 cosine）、`warmup_ratio`。  
4. **`create_optimizer`**：对 `requires_grad=True` 的参数分 **weight decay / 无 decay** 两组。  
5. **保存**：`save_pretrained`；若 `save_lora_only=true` 则只存可训练参数（LoRA 等）。

---

## 7. 「训练阶段」与 Freeze 策略（本仓库中的实际含义）

### 7.1 非分阶段 curriculum

当前 **DreamZero DROID 脚本**（`droid_training_full_finetune.sh`）是 **单一连续训练循环**：**没有**在代码里写死的「阶段 1/2/3」切换 schedule。  
若需多阶段，需自行改 config 或脚本（例如先 `train_architecture=lora` 再加载全量微调）。

### 7.2 可视为「策略维度」的配置

| 维度 | 选项 | 效果 |
|------|------|------|
| **`train_architecture`** | `lora` / `full` | LoRA 仅训低秩适配器 + action/state 头；full 可训整 DiT（仍冻结 T5/CLIP/VAE） |
| **`tune_projector` / `tune_diffusion_model`** | bool | 是否允许训练 DiT 主体（非 LoRA 模式下） |
| **冻结** | 代码写死 | **`text_encoder`、`image_encoder`、`vae`** 始终 `requires_grad=False` |

### 7.3 DeepSpeed ZeRO

`training_args.deepspeed` 指向的 JSON（如 **ZeRO-2 + offload**）决定 **优化器状态/参数分片**，与「是否冻结某层」正交。

---

## 8. Loss 与优化策略小结

| 项目 | 内容 |
|------|------|
| **主损失** | Flow Matching：**视频潜空间预测** + **动作噪声预测**，均为 **MSE**（reduction 后按 timestep **training_weight** 加权） |
| **掩码** | 动作：`action_mask` × `has_real_action` |
| **可选噪声** | `decouple_video_action_noise`、`use_high_noise_emphasis` 等（见 `WANPolicyHeadConfig`） |
| **优化器** | AdamW（默认分组 weight decay） |
| **学习率** | 如 `1e-5`（全量微调脚本）；scheduler：cosine + warmup |
| **精度** | bf16 + tf32（脚本中常见）；`torch.amp.autocast` 在 head 内对部分块使用 |

---

## 9. 端到端数据流简图（含每步 I/O 含义与 Shape）

### 9.1 总览简图

```text
磁盘: parquet + mp4 + meta/*.json(l)
    ↓
ShardedLeRobotMixtureDataset / ShardedLeRobotSubLangSingleActionChunkDatasetDROID
    ↓
ComposedModalityTransform (视频增广 + StateAction 归一化 + Concat)
    ↓
DreamTransform (拼图、语言、state/action pad、embodiment_id)
    ↓
DefaultDataCollator (tokenize text → text + attention_mask；stack → torch)
    ↓
VLA.forward
    IdentityBackbone → 占位 backbone_features (B,1,0)
    WANPolicyHead: T5 文本编码、VAE 编视频、CLIP+首帧、加噪、CausalWanModel、MSE loss
    ↓
Trainer.compute_loss → backward → AdamW + DeepSpeed(可选)
```

**符号约定**：`B`=batch；`T_v`/`T_a`/`T_s` 分别为配置里 video / action / state 的时间长度（如 DROID 常见 `T_v=25`，`T_a=24`，`T_s=1`）；`V`=原始相机路数（DROID 为 3）；`H,W` 为 resize 后单路分辨率；拼图后高宽为 `2H×2W`；`D_s^eff`/`D_a^eff` 为拼接后 state/action 有效维；`max_state_dim`/`max_action_dim` 为 pad 后维；`L_text` 为 tokenizer `max_length`；潜空间维度用 `C_lat,T_lat,H_lat,W_lat` 表示（由 VAE 与 `num_frames` 决定）。**具体数以当前 Hydra 为准。**

### 9.2 步骤 0：磁盘

| 项目 | 说明 |
|------|------|
| **输入** | 无（数据源） |
| **输出** | 文件系统上的 LeRobot 包：`parquet`（逐步 state/action/时间戳/语言 task 索引）、`mp4`（多相机）、`meta/*.json(l)`（模态、episode、任务表） |
| **Shape** | 不适用；parquet 每行一步，列宽由 `modality.json` / `info.json` 定义 |

### 9.3 步骤 1：`Dataset.__getitem__`（Shard 内采样）

| 项目 | 说明 |
|------|------|
| **输入** | 逻辑索引 → 对应 `(trajectory_id, base_index)`；各模态 `delta_indices` 生成步下标 |
| **输出** | **`dict`**：多键，如 `video` `(T_v, V, H_raw, W_raw, C)` uint8；各 `state.*`/`action.*` 经拼接前为多键；`annotation.language.*` 为字符串（或由 task 表解析） |
| **Shape（视频）** | `(T_v, V, H_raw, W_raw, C)`，`C=3` |
| **Shape（state/action）** | 进入 Compose 前常为单键张量，经 `ConcatTransform` 后为 `(T_s, D_s^eff)`、`(T_a, D_a^eff)` |

### 9.4 步骤 2：`ComposedModalityTransform`

| 项目 | 说明 |
|------|------|
| **输入** | 上一步 `dict`（多模态键） |
| **输出** | 同结构 `dict`，**`video`**：`(T_v, V, H, W, C)` uint8（已 crop/resize/增广）；**`state`/`action`**：float，**归一化**后约在 `[-1,1]` 量级 |
| **Shape** | `video`: `(T_v, V, H, W, C)`；`state`: `(T_s, D_s^eff)`；`action`: `(T_a, D_a^eff)` |

### 9.5 步骤 3：`DreamTransform.apply_single`

| 项目 | 说明 |
|------|------|
| **输入** | 上一步 `dict`（含 `video`、语言键、`state`、`action`） |
| **输出** | **`dict`**：`images`（DROID 1 路拼图）、`text`（字符串）、`state`/`state_mask`、`action`/`action_mask`、`embodiment_id`、`has_real_action`、`text_negative` 等 |
| **Shape** | `images`: `(1, T_v, 3, 2H, 2W)` uint8；`_apply_vlm_processing` 后并入 batch 流时展平为 **`(T_v, 2H, 2W, 3)`**（单样本）；`state`: `(T_s, max_state_dim)`；`action`: `(T_a, max_action_dim)`；mask 与之一致 |

### 9.6 步骤 4：`DefaultDataCollator`（`collate`）

| 项目 | 说明 |
|------|------|
| **输入** | `List[dict]`，长度 `B`（每元素为 DreamTransform 输出） |
| **输出** | **`dict[str, Tensor]`**（及 `numpy`→`torch`）：供 `VLA.forward` 使用 |
| **Shape** | `images`: `(B, T_v, H_p, W_p, 3)`，`H_p=2H, W_p=2W`；`text`: `(B, L_text)`；`text_attention_mask`: `(B, L_text)`；`state`: `(B, T_s, max_state_dim)`；`state_mask`: 同；`action`: `(B, T_a, max_action_dim)`；`action_mask`: 同；`embodiment_id`: `(B,)`；`has_real_action`: `(B,)` |

### 9.7 步骤 5：`VLA.prepare_input` → `IdentityBackbone`

| 项目 | 说明 |
|------|------|
| **输入** | Collator 输出的 **`inputs` dict** |
| **输出（backbone）** | `BatchFeature`：`backbone_features` **占位** |
| **Shape** | `backbone_features`: **`(B, 1, 0)`**（空特征维）；`action` 等仅用于推断 `B` |
| **输出（action_head 侧）** | `action_input = action_head.prepare_input(inputs)` → 等价于把原 batch 包进 `BatchFeature`，键名不变 |

### 9.8 步骤 6：`WANPolicyHead.forward`（训练，子步骤）

| 子步骤 | 输入含义与 Shape | 输出含义与 Shape |
|--------|------------------|------------------|
| **6a 图像归一化** | `images` `(B,T_v,H_p,W_p,3)` uint8 → 转 `(B,3,T_v,H_p,W_p)` float，归一化到 `[-1,1]` | `videos` `(B,3,T_v,H_p,W_p)` bf16/float |
| **6b 文本编码** | `text` `(B,L_text)`，`text_attention_mask` `(B,L_text)` | `prompt_embs` `(B, L_text, D_t5)`（padding 位置已置 0；`D_t5` 为 T5 隐藏维） |
| **6c VAE 编码视频** | `videos` `(B,3,T_v,H_p,W_p)` | `latents`：视频潜变量，形如 **`(B, C_lat, T_lat, H_lat, W_lat)`**（与 WAN VAE 下采样一致）；内部可能 `transpose` 与块划分对齐 |
| **6d CLIP + 首帧 I2V 条件** | 首帧 `videos[:,:,:1,...]` | `clip_feas`（图像全局/序列特征）；`ys` 等与 I2V mask+首帧潜变量拼接，供 DiT 条件 |
| **6e Flow 加噪** | `latents`，`actions` `(B,T_a,max_action_dim)`；采样 `timestep` / `timestep_action` | `noisy_latents`（同 latent shape）；`noisy_actions` `(B,T_a,max_action_dim)`；`training_target`（视频支路）、`training_target_action`（动作支路） |
| **6f `CausalWanModel`（训练）** | 上述张量 + `seq_len`（由 `frame_seqlen` 与 patch 网格推导）+ `state` `(B,T_s,max_state_dim)` + `embodiment_id` | `video_noise_pred`：与 **latent** 同空间 **`(B, C_lat, T_lat, H_lat, W_lat)`**；`action_noise_pred` **`(B, T_a, max_action_dim)`** |
| **6g Loss** | 预测 vs `training_target`（视频/动作） | 标量 **`loss`**；`dynamics_loss`；`action_loss`；`BatchFeature` 返回 |

### 9.9 步骤 7：`Trainer.compute_loss` 及之后

| 项目 | 说明 |
|------|------|
| **输入** | `model(inputs)` 返回的 `BatchFeature`，至少含 **`loss`** |
| **输出** | 反向传播后的参数更新；日志中的 `loss` / 各 `*_loss` 滑动平均 |
| **Shape** | `loss`: 标量 `()`；无额外张量返回给 DataLoader |

---

## 10. 核心模块详细分析：`WANPolicyHead` 与 `CausalWanModel`

本节深入剖析 DreamZero 最核心的两个模块：**WANPolicyHead**（策略头）与 **CausalWanModel**（因果 DiT 骨干）。这两个模块继承自 Wan2.1-I2V-14B（图像到视频生成模型），并扩展了机器人动作生成能力。

---

### 10.1 `WANPolicyHead` 总体架构

**文件**: `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

#### 10.1.1 模块组成与职责

| 子模块 | 类 | 职责 | 是否可训练 | 预训练权重 |
|--------|-----|------|-----------|-----------|
| **文本编码器** | `WanTextEncoder` (umt5-xxl) | 编码语言指令 → prompt_emb | ❌ 冻结 | ✅ 加载 `models_t5_umt5-xxl-enc-bf16.pth` |
| **图像编码器** | `WanImageEncoder` (CLIP) | 编码首帧 → clip_feas | ❌ 冻结 | ✅ 加载 `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |
| **视频 VAE** | `WanVideoVAE` | 视频 ↔ 潜空间 编解码 | ❌ 冻结 | ✅ 加载 `Wan2.1_VAE.pth` |
| **流匹配调度器** | `FlowMatchScheduler` | 管理加噪/去噪 timestep | - | - |
| **因果 DiT** | `CausalWanModel` | 核心去噪网络，预测视频+动作噪声 | ✅ 可训练（LoRA/全量） | ✅ 加载 `diffusion_pytorch_model.safetensors` |
| **状态编码器** | `CategorySpecificMLP` (state_encoder) | 编码当前状态 → DiT 维度 | ✅ 可训练 | ❌ 从头初始化 |
| **动作编码器** | `MultiEmbodimentActionEncoder` (action_encoder) | 编码带噪动作序列 | ✅ 可训练 | ❌ 从头初始化 |
| **动作解码器** | `MultiEmbodimentActionDecoder` (action_decoder) | 从 DiT 输出解码动作噪声 | ✅ 可训练 | ❌ 从头初始化 |

#### 10.1.2 训练时数据流（简化版）

```
输入 Batch (B=4, T_v=33, T_a=24, H=176, W=320)
    │
    ├─► images (B, T_v, H, W, 3) uint8
    │     ├─► Normalize → (B, 3, T_v, H, W) float ∈[-1,1]
    │     ├─► VAE.encode() → latents (B, 16, 9, 22, 40)  [C_lat=16, T_lat=⌈33/4⌉=9, 4x下采样]
    │     └─► 首帧 → CLIP.encode() → clip_feas (B, 1280)
    │
    ├─► text (B, 512) ──► T5.encode() → prompt_embs (B, 512, 2048)
    │
    ├─► state (B, 1, 64) ──► state_encoder ──► state_features (B, 1, 5120)
    │
    ├─► action (B, 24, 64) ∈[-1,1]
    │     ├─► 采样 timestep_action (B, 24)
    │     ├─► scheduler.add_noise() → noisy_actions (B, 24, 64)
    │     └─► action_encoder ──► action_features (B, 24, 5120)
    │
    └─► 采样 timestep_video (B, 9)
          ├─► scheduler.add_noise() → noisy_latents (B, 16, 9, 22, 40)
          └─► transpose → (B, 9, 16, 22, 40) → flatten → (B, 9, 16×22×40=14080)?
             
          [实际: patch_embed → (B, seq_len_video, dim=5120)]
          
DiT 输入: [patch_embed_video | action_features | state_features]
           seq:   (B, L_video + 24 + 1, 5120)
           条件:  prompt_embs (B, 512, 2048) → text_proj → (B, 512, 5120)
                 clip_feas → img_emb → (B, 257, 5120)
                 ys (I2V 条件) → (B, 17, 22, 40) [mask+首帧潜变量]
                 timestep_emb (视频+动作时间步)

CausalWanModel.forward()
    ├─► 40层 CausalWanAttentionBlock (自注意力 + 交叉注意力 + FFN)
    ├─► CausalHead ──► video_noise_pred (B, seq_len_video, 16×2×2=64) → unpatchify
    └─► action_decoder ──► action_noise_pred (B, 24, 64)

Loss: MSE(video_noise_pred, training_target) + MSE(action_noise_pred, training_target_action)
```

---

### 10.2 `CausalWanModel`（因果 DiT）详细结构

**文件**: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

#### 10.2.1 架构参数（DROID 全量微调配置示例）

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 5120 | DiT 隐藏维度（与 Wan2.1-I2V-14B 一致） |
| `num_layers` | 40 | Transformer 层数 |
| `num_heads` | 40 | 注意力头数（每头 dim_head=128） |
| `ffn_dim` | 13824 | FFN 中间维度 |
| `patch_size` | (1, 2, 2) | 3D Patch 尺寸（时间×高×宽） |
| `in_dim / out_dim` | 16 | VAE 潜空间通道数 |
| `num_frame_per_block` | 2 | 视频块粒度（每2帧一个时序块） |
| `num_action_per_block` | 24 | 动作块粒度（24步动作为一个块） |
| `num_state_per_block` | 1 | 状态块粒度（1步状态为一个块） |
| `frame_seqlen` | 880 | 每帧 patch 序列长度（由分辨率决定） |

#### 10.2.2 输入处理流程（逐组件分解）

**Step 1: 视频 Patch Embedding**

```python
# 输入: noisy_latents (B, 16, 9, 22, 40) - 已由 WANPolicyHead 转置为 (B, C, F, H, W)
# 注意: VAE输出的是 (B, 16, 9, 22, 40)，其中 9 = ⌈(33-1)/4⌉+1 = 9，22=176/8, 40=320/8

x = noisy_latents.transpose(1, 2)  # (B, 9, 16, 22, 40)
x = x.flatten(start_dim=2).transpose(1, 2)  # (B, 9×22×40=7920, 16)? 不对，实际是先过 patch_embed

# 实际: 通过 3D Conv patch_embed (kernel=(1,2,2), stride=(1,2,2))
# (B, 16, 9, 22, 40) → (B, 5120, 9, 11, 20) → flatten → (B, 9×11×20=1980, 5120)
```

实际计算：
- Patch 后空间尺寸：H/2=11, W/2=20（因为 patch_size=(1,2,2) 只有空间下采样）
- 时间维不变：9 帧
- seq_len_video = 9 × 11 × 20 = 1980
- 但配置中 `frame_seqlen=880`，说明有额外的下采样或配置调整

**Step 2: 动作与状态编码**

```python
# 输入: noisy_actions (B, 24, 64), state (B, 1, 64), embodiment_id (B,)

action_features = action_encoder(noisy_actions, timestep_action, embodiment_id)
# (B, 24, 64) + timestep_emb → (B, 24, 5120)

state_features = state_encoder(state, embodiment_id)
# (B, 1, 64) → (B, 1, 5120)

# 拼接: [video_patches | action_features | state_features]
x = torch.cat([video_patches, action_features, state_features], dim=1)
# (B, 1980 + 24 + 1 = 2005, 5120)
```

**Step 3: 条件注入（文本、图像、时间步）**

```python
# 文本条件
prompt_embs = text_encoder(text, mask)  # (B, 512, 2048)
context = text_projection(prompt_embs)    # (B, 512, 5120)

# 图像条件（I2V）
clip_feas = image_encoder.encode_image(first_frame)  # (B, 1280)
clip_emb = img_emb(clip_feas)                        # (B, 257, 5120) [CLS token + patch tokens]
context = torch.cat([clip_emb, context], dim=1)      # (B, 257+512=769, 5120)

# I2V 专用条件: ys (mask + 首帧潜变量)
# msk: (B, 4, 3, 22, 40) [ temporal压缩后 9帧 → 3个temporal块 ]
# new_image: (B, 16, 3, 22, 40) 首帧潜变量复制
# y = torch.cat([msk, new_image], dim=1)  # (B, 20, 3, 22, 40)? 实际更复杂

# 时间步嵌入
video_timestep_emb = sinusoidal_embedding(timestep_video)  # (B, 9, 256) → MLP → (B, 9, 5120)
action_timestep_emb = sinusoidal_embedding(timestep_action) # (B, 24, 5120)
state_timestep_emb = action_timestep_emb[:, ::24]  # (B, 1, 5120) 简化表示

e = torch.cat([video_timestep_emb, action_timestep_emb, state_timestep_emb], dim=1)
# (B, 1980+24+1=2005, 5120)
```

#### 10.2.3 CausalWanAttentionBlock（单层结构）

```python
class CausalWanAttentionBlock(nn.Module):
    def forward(x, e, freqs, context, action_register_length):
        # x: (B, L, C) = (B, 2005, 5120)
        # e: (B, L, 6, C) 时间调制参数 (AdaLN的6个参数: shift1, scale1, gate1, shift2, scale2, gate2)
        # freqs: (B, L, head_dim) RoPE 频率（视频/动作/状态可能有不同频率）
        # context: (B, L_ctx, C) = (B, 769, 5120) 文本+图像条件
        # action_register_length: 25 (24动作+1状态)
        
        # 1. AdaLN 调制（使用e对x进行scale/shift）
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        normed_x = self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        
        # 2. 因果自注意力（FlexAttention实现的块因果mask）
        # - 视频块内部全可见
        # - 动作块只可见对应视频块+之前所有块+自己
        # - 状态块只可见对应视频块+之前所有块+自己
        y, kv_cache = self.self_attn(
            normed_x, 
            freqs=freqs,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=True
        )
        x = x + y * e[2].squeeze(2)  # gate1调制
        
        # 3. 交叉注意力（文本+图像条件）
        x = x + self.cross_attn(self.norm3(x), context)
        
        # 4. FFN (GELU)
        normed_x = self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        y = self.ffn(normed_x)
        x = x + y * e[5].squeeze(2)  # gate2调制
        
        return x, kv_cache
```

#### 10.2.4 因果注意力 Mask 结构

FlexAttention 实现的块因果注意力（Blockwise Causal Attention）：

```
序列布局: [首帧 | 视频块1 | 视频块2 | ... | 视频块N | 动作块 | 状态块]
           ↓      ↓          ↓              ↓          ↓         ↓
          全局可见 块内可见   块内可见        块内可见    仅见对应视频块+之前块

具体 Mask 规则（伪代码）:
def attention_mask(b, h, q_idx, kv_idx):
    # q_idx: 查询位置, kv_idx: 被查询位置
    
    # 首帧（clean image）全局可见
    if q_idx < frame_seqlen:
        return kv_idx < frame_seqlen  # 只能看见首帧
    
    # 视频块：块内全可见 + 首帧可见
    if is_video_token(q_idx):
        block_start = get_block_start(q_idx)
        block_end = get_block_end(q_idx)
        return (kv_idx >= block_start and kv_idx < block_end) or kv_idx < frame_seqlen
    
    # 动作块：可见对应视频块 + 之前所有块 + 自己
    if is_action_token(q_idx):
        video_context_end = get_corresponding_video_end(q_idx)
        return kv_idx < video_context_end or kv_idx == q_idx
    
    # 状态块：同动作块
    if is_state_token(q_idx):
        video_context_end = get_corresponding_video_end(q_idx)
        return kv_idx < video_context_end or kv_idx == q_idx
```

#### 10.2.5 输出头

```python
# 经过40层 CausalWanAttentionBlock 后
x: (B, 2005, 5120)

# 视频预测头（CausalHead）
video_output = x[:, :seq_len_video]  # (B, 1980, 5120)
video_noise_pred = head(video_output, e_video)  # (B, 1980, 64) [64 = 16×2×2 = C_out×patch_h×patch_w]

# unpatchify: (B, 1980, 64) → (B, 16, 9, 22, 40) 潜空间噪声预测
video_noise_pred = unpatchify(video_noise_pred, grid_size=(9, 11, 20))
# (B, 16, 9, 22, 40) → transpose → (B, 16, 9, 22, 40) [与 latents 同形]

# 动作预测头（action_decoder）
action_output = x[:, seq_len_video:seq_len_video+action_length]  # (B, 24, 5120)
action_noise_pred = action_decoder(action_output, embodiment_id)  # (B, 24, 64)
```

---

### 10.3 关键维度变化总结表

以 `B=4, num_frames=33, H=176, W=320, action_horizon=24, max_state_dim=64, max_action_dim=64` 为例：

| 阶段 | 张量名 | Shape | 计算说明 |
|------|--------|-------|----------|
| **输入** | images | `(4, 33, 176, 320, 3)` | 原始视频（拼图后分辨率） |
| | text | `(4, 512)` | token ids |
| | state | `(4, 1, 64)` | 当前状态（已归一化） |
| | action | `(4, 24, 64)` | 目标动作（gt，已归一化） |
| **编码** | prompt_embs | `(4, 512, 2048)` | T5 文本编码（umt5-xxl） |
| | clip_feas | `(4, 1280)` | CLIP 图像编码 |
| | latents | `(4, 16, 9, 22, 40)` | VAE 编码（16通道，4x时间下采样，8x空间下采样） |
| | ys | `(4, 20, 3, 22, 40)` | I2V 条件（mask 4ch + 首帧潜变量 16ch → 拼接为 20ch，时间压缩后 3块） |
| **加噪** | noisy_latents | `(4, 16, 9, 22, 40)` | 与 latents 同形 |
| | noisy_actions | `(4, 24, 64)` | 与 actions 同形 |
| **DiT 输入** | video_patches | `(4, 1980, 5120)` | 9×11×20=1980 patches，每 patch 5120 dim |
| | action_features | `(4, 24, 5120)` | 24 步动作，action_encoder 输出 |
| | state_features | `(4, 1, 5120)` | 1 步状态，state_encoder 输出 |
| | x (拼接后) | `(4, 2005, 5120)` | 1980+24+1=2005 tokens |
| | context | `(4, 769, 5120)` | 257(CLIP)+512(T5)=769 条件 tokens |
| **DiT 输出** | video_noise_pred | `(4, 16, 9, 22, 40)` | 视频噪声预测（与 latents 同形） |
| | action_noise_pred | `(4, 24, 64)` | 动作噪声预测 |
| **Loss** | dynamics_loss | `scalar` | MSE(video_noise_pred, target) |
| | action_loss | `scalar` | MSE(action_noise_pred, target_action) × action_mask × has_real_action |
| | loss | `scalar` | dynamics_loss + action_loss |

---

### 10.4 训练 vs 推理差异

| 特性 | 训练 (`forward`) | 推理 (`get_action`) |
|------|------------------|---------------------|
| **输入动作** | gt action（用于计算 loss） | None（模型自回归生成或完全去噪） |
| **时间步** | 随机采样（Beta分布或均匀） | 调度器预设序列（如16步或4步） |
| **噪声状态** | 加噪到随机 timestep | 从纯噪声开始逐步去噪 |
| **DiT 调用** | 单次前向预测噪声 | 多轮迭代，每轮调用 DiT 预测噪声并去噪 |
| **输出** | loss | action_pred ∈[-1,1] |
| **CFG** | 不使用 | 使用（cond + scale×(cond-uncond)） |
| **KV Cache** | 不使用（is_tf=True） | 使用（缓存之前计算的 KV） |

推理时的迭代去噪（以16步为例）：

```python
latents = randn(B, 16, 9, 22, 40)  # 纯噪声
for t in [1000, 937, ..., 0]:  # 16个timestep
    timestep = torch.full((B, 9), t)
    noise_pred, action_pred = model(latents, timestep, ...)
    latents = scheduler.step(noise_pred, t, latents).prev_sample  # 去噪
    
# 最终: VAE.decode(latents) → 视频; action_decoder输出 → 动作
```

---

## 11. 架构分解：Backbone / Encoder / Decoder

本节将 DreamZero 的模型管线明确划分为 **Encoder（编码器）**、**Backbone（骨干网络）** 和 **Decoder（解码器/预测头）** 三大部分，详细说明每个组件的角色、I/O shape、损失函数与评估指标。

---

### 11.1 总体架构图

```
┌─────────────────────────────────────── Encoders（全部冻结，除 state/action encoder 可训练） ───────────────────────────────────────┐
│                                                                                                                                    │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐ │
│  │ WanTextEncoder (T5) │  │ WanImageEncoder(CLIP)│  │ WanVideoVAE.encode       │  │ state_encoder      │  │ action_encoder       │ │
│  │ text→prompt_embs    │  │ 首帧→clip_feas       │  │ video→latents            │  │ state→state_feat   │  │ noisy_act+t→act_feat │ │
│  │ (B,512,4096)        │  │ (B,257,1280)         │  │ (B,16,T_lat,H_lat,W_lat) │  │ (B,1,5120)         │  │ (B,24,5120)          │ │
│  └──────┬──────────────┘  └──────┬───────────────┘  └──────────┬───────────────┘  └──────┬─────────────┘  └──────┬───────────────┘ │
└─────────┼────────────────────────┼──────────────────────────────┼─────────────────────────┼───────────────────────┼─────────────────┘
          │                        │                              │                         │                       │
          │  text_proj             │  img_emb                     │  patch_embed             │ concat                │ concat
          │  (B,512,5120)          │  (B,257,5120)                │  (B,seq_len,5120)        │                       │
          ▼                        ▼                              ▼                         ▼                       ▼
┌─────────────────────────────────────── Backbone（DiT Transformer，LoRA/全量可训练） ──────────────────────────────────────────────────┐
│                                                                                                                                      │
│  context = [clip_emb | prompt_embs_proj]  ←── 交叉注意力条件: (B, 769, 5120)                                                         │
│  x = [video_patches | action_features | state_features]  ←── 自注意力序列: (B, 2005, 5120)                                           │
│  e = [video_time_emb | action_time_emb | state_time_emb]  ←── AdaLN 时间调制                                                        │
│                                                                                                                                      │
│  ┌──────────────────────────────────────────────────────────┐                                                                        │
│  │  40 × CausalWanAttentionBlock                           │                                                                        │
│  │   → AdaLN 调制 → 因果自注意力(FlexAttention) → 交叉注意力 → FFN                                                                  │
│  │   → RoPE 位置编码, 块因果 Mask                            │                                                                        │
│  └────────────────────┬─────────────────┬──────────────────┘                                                                        │
│                       │                 │                                                                                            │
└───────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┘
                        │                 │
                        ▼                 ▼
┌──────────────────────────────────────── Decoders（两个预测头） ─────────────────────────────────────────────────────────────────────────┐
│                                                                                                                                       │
│  ┌────────────────────────────────────────────┐    ┌────────────────────────────────────────────────┐                                 │
│  │ Video Head (CausalHead)                    │    │ Action Head (action_decoder, CategorySpecificMLP)│                                │
│  │ video_tokens → AdaLN+Linear → unpatchify   │    │ action_tokens → MLP → action_noise_pred         │                                │
│  │ (B,seq_len,5120) → (B,16,T_lat,H_lat,W_lat)│   │ (B,24,5120) → (B,24,action_dim)                │                                │
│  │ Loss: dynamics_loss (MSE + training_weight) │    │ Loss: action_loss (MSE × mask × training_weight)│                                │
│  └────────────────────────────────────────────┘    └────────────────────────────────────────────────┘                                 │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 11.2 Encoder 部分（编码器）

Encoder 负责将原始模态数据（文本、图像、视频、状态、动作）映射到统一的特征空间，供 Backbone 使用。

#### 11.2.1 编码器总览

| # | 编码器名称 | 类 | 文件 | 作用 | I/O Shape | 冻结 | 可训练 |
|---|-----------|-----|------|------|-----------|------|--------|
| 1 | **文本编码器** | `WanTextEncoder` | `modules/wan_video_text_encoder.py` | 将 tokenized 文本编码为语义嵌入 | `(B,L)` → `(B,L,4096)` | ✅ | ❌ |
| 2 | **图像编码器** | `WanImageEncoder` | `modules/wan_video_image_encoder.py` | 将首帧编码为 CLIP 视觉特征 | `(B,3,H,W)` → `(B,257,1280)` | ✅ | ❌ |
| 3 | **视频 VAE** | `WanVideoVAE` | `modules/wan_video_vae.py` | 将视频像素空间映射到潜空间 | `(B,3,T,H,W)` → `(B,16,T_lat,H_lat,W_lat)` | ✅ | ❌ |
| 4 | **状态编码器** | `CategorySpecificMLP` | `modules/wan_video_dit_action_casual_chunk.py` | 将归一化状态映射到 DiT 维度 | `(B,T_s,max_state_dim)` → `(B,T_s,dim)` | ❌ | ✅ |
| 5 | **动作编码器** | `MultiEmbodimentActionEncoder` | `modules/wan_video_dit_action_casual_chunk.py` | 将带噪动作+时间步编码到 DiT 维度 | `(B,T_a,action_dim)` + `(B,)` → `(B,T_a,dim)` | ❌ | ✅ |

#### 11.2.2 各编码器详细说明

**1. WanTextEncoder（文本编码器）**

| 属性 | 说明 |
|------|------|
| **文件** | `groot/vla/model/dreamzero/modules/wan_video_text_encoder.py` |
| **架构** | UMT5-XXL 风格 Transformer encoder，24层，`dim=4096`，`num_heads=64`，`ffn_dim=10240` |
| **输入** | `ids`: `(B, L_text)` token IDs; `mask`: `(B, L_text)` attention mask |
| **输出** | `(B, L_text, 4096)` — 文本语义嵌入（padding 位置在 `WANPolicyHead` 中被置零） |
| **下游** | `WANPolicyHead.encode_prompt()` → `text_projection` 投影到 `(B, L_text, 5120)` → 作为 DiT 交叉注意力的 `context` |
| **冻结** | `requires_grad=False`，始终冻结 |
| **预训练** | 加载 `models_t5_umt5-xxl-enc-bf16.pth` |

**2. WanImageEncoder（图像编码器）**

| 属性 | 说明 |
|------|------|
| **文件** | `groot/vla/model/dreamzero/modules/wan_video_image_encoder.py` |
| **架构** | CLIP ViT-H/14（`clip_xlm_roberta_vit_h_14`），`vision_dim=1280`，`image_size=224`，`patch_size=14`，32层 ViT |
| **输入** | 首帧图像 `(B, 3, 224, 224)` — 经 CLIP 归一化 |
| **输出** | `(B, 257, 1280)` — 1 个 CLS token + 256 个 patch tokens（`use_31_block=True` 取倒数第二层输出） |
| **下游** | `WANPolicyHead.encode_image()` → `img_emb` 投影到 `(B, 257, 5120)` → 拼接到 `context` 中参与交叉注意力 |
| **冻结** | `requires_grad=False`，始终冻结 |
| **预训练** | 加载 `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |

**3. WanVideoVAE（视频 VAE 编码器）**

| 属性 | 说明 |
|------|------|
| **文件** | `groot/vla/model/dreamzero/modules/wan_video_vae.py` |
| **架构** | 3D 因果卷积 VAE，`z_dim=16`，空间 8× 下采样，时间 4× 下采样 |
| **输入（encode）** | `videos`: `(B, 3, T, H, W)` — 归一化到 `[-1,1]` 的视频 |
| **输出（encode）** | `latents`: `(B, 16, T_lat, H_lat, W_lat)` — 其中 `T_lat = 1 + (T-1)//4`, `H_lat = H//8`, `W_lat = W//8` |
| **输入（decode）** | `z`: `(B, 16, T_lat, H_lat, W_lat)` — 潜变量 |
| **输出（decode）** | `(B, 3, T, H, W)` — 重建视频（推理时使用） |
| **下游** | `latents` → FlowMatchScheduler 加噪 → `noisy_latents` → CausalWanModel patch embed |
| **冻结** | `requires_grad=False`，始终冻结；`torch.no_grad()` 下执行 |
| **预训练** | 加载 `Wan2.1_VAE.pth` |

**4. state_encoder（状态编码器）**

| 属性 | 说明 |
|------|------|
| **文件** | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| **类** | `CategorySpecificMLP(num_categories=1, input_dim=max_state_dim, hidden_dim=hidden_size, output_dim=dim)` |
| **架构** | 两层 MLP：`Linear(max_state_dim→hidden_size)` → ReLU → `Linear(hidden_size→dim)`，按 `embodiment_id` 选择权重 |
| **输入** | `state`: `(B, T_s, max_state_dim)` 归一化状态; `embodiment_id`: `(B,)` |
| **输出** | `(B, T_s, dim=5120)` — 状态特征，拼接到 DiT 序列末尾 |
| **下游** | 拼接到 `[video_patches | action_features | state_features]` 参与自注意力 |
| **可训练** | ✅ 从头初始化，始终可训练 |

**5. action_encoder（动作编码器）**

| 属性 | 说明 |
|------|------|
| **文件** | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| **类** | `MultiEmbodimentActionEncoder(action_dim, hidden_size=dim, num_embodiments=1)` |
| **架构** | `W1(action)` → `a_emb`; `SinusoidalPosEnc(timestep)` → `tau_emb`; `concat → W2 → swish → W3` |
| **输入** | `actions`: `(B, T_a, action_dim)` 带噪动作; `timesteps`: `(B,)` 扩散时间步; `cat_ids`: `(B,)` |
| **输出** | `(B, T_a, dim=5120)` — 动作特征，拼接到 DiT 序列中 |
| **下游** | 拼接到 `[video_patches | action_features | state_features]` 参与自注意力 |
| **可训练** | ✅ 从头初始化，始终可训练 |

---

### 11.3 Backbone 部分（DiT Transformer 骨干）

Backbone 是 `CausalWanModel` 的核心——由 Patch Embedding 和 40 层 `CausalWanAttentionBlock` 组成的因果 Diffusion Transformer。

| 属性 | 说明 |
|------|------|
| **文件** | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| **类** | `CausalWanModel`（继承自 Wan2.1-I2V-14B DiT） |
| **参数量级** | ~14B（与 Wan2.1 一致），`dim=5120`, `num_layers=40`, `num_heads=40` |

#### 11.3.1 输入拼接

Backbone 将编码器输出拼接为统一序列：

```
x = [ video_patches (B, L_video, 5120) | action_features (B, 24, 5120) | state_features (B, 1, 5120) ]
     └──────── seq_len ─────────────┘   └──── action_length ────────┘   └── state_length ──┘
                                                                                     
总序列: (B, L_video + 24 + 1, 5120) = (B, 2005, 5120) [以 DROID 典型配置为例]
```

#### 11.3.2 条件注入

| 条件类型 | 来源 | 注入方式 | Shape |
|---------|------|---------|-------|
| **文本+图像** | `prompt_embs` + `clip_emb` | 交叉注意力 `context` | `(B, 769, 5120)` |
| **时间步** | `timestep` / `timestep_action` | AdaLN 调制 `e` | `(B, L_total, 6, 5120)` |
| **I2V 条件** | `ys`（mask + 首帧潜变量拼接） | 与 `noisy_latents` 通道拼接后过 `patch_embed` | 随 VAE 潜空间维度变化 |
| **RoPE** | 视频/动作/状态各有独立频率 | 自注意力 Q/K 旋转位置编码 | `(L_total, head_dim)` |

#### 11.3.3 注意力 Mask（块因果）

```
序列布局: [首帧 patches | 视频块1 | 视频块2 | ... | 视频块N | 动作块 | 状态块]

Mask 规则:
- 首帧 patches: 仅看见自身（全局锚点）
- 视频块 k:    看见 首帧 + 视频块1..k（因果，块内全可见）
- 动作块:      看见 首帧 + 所有视频块 + 自身动作块
- 状态块:      看见 首帧 + 所有视频块 + 动作块 + 自身状态块
```

#### 11.3.4 训练策略

| 策略 | LoRA 模式 | Full 模式 |
|------|----------|----------|
| 40层 Transformer blocks | 冻结主权重 + LoRA 适配器可训 | 全量可训练 |
| patch_embed / text_proj / img_emb | 由 `tune_projector` 控制 | 通常可训 |
| 预训练权重 | `diffusion_pytorch_model.safetensors` | 同 |

---

### 11.4 Decoder 部分（两个预测头）— 重点

Decoder 从 Backbone 的输出序列中分别提取 **视频 tokens** 和 **动作 tokens**，解码为各自的预测目标。

#### 11.4.1 Decoder 总览

| # | Head 名称 | 类 | 输入来源 | 输出目标 | 损失函数 | 评估指标 |
|---|----------|-----|---------|---------|---------|---------|
| 1 | **Video Head** | `CausalHead` | DiT 输出的前 `seq_len` 个视频 tokens | 视频潜空间噪声预测 | `dynamics_loss` (加权 MSE) | `dynamics_loss_avg` |
| 2 | **Action Head** | `CategorySpecificMLP` (action_decoder) | DiT 输出的动作段 tokens | 动作噪声预测 | `action_loss` (masked 加权 MSE) | `action_loss_avg` + open-loop MSE |

---

#### 11.4.2 Video Head（`CausalHead`）— 视频潜空间噪声预测

**文件**: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

**（a）架构**

```python
class CausalHead(nn.Module):
    # AdaLN 调制 + 线性投影
    norm: WanLayerNorm(dim)           # LayerNorm
    head: nn.Linear(dim, out_dim * prod(patch_size))  # 5120 → 16×1×2×2 = 64
    modulation: nn.Parameter(1, 2, dim)  # AdaLN shift/scale 参数
```

**（b）输入/输出**

| 方向 | 张量 | Shape | 含义 |
|------|------|-------|------|
| **输入 x** | DiT 视频段输出 | `(B, seq_len, 5120)` | 40层 Transformer 处理后的视频 token 特征；`seq_len = T_lat × (H_lat/ph) × (W_lat/pw)` |
| **输入 e** | 时间步调制 | `(B, seq_len, 1, 5120)` | 由 `timestep_emb` 产生的 AdaLN 调制向量（`e_video.unsqueeze(2)`） |
| **输出** | 噪声预测（patch空间） | `(B, seq_len, 64)` | 每个 patch 的 `out_dim × pt × ph × pw = 16×1×2×2 = 64` 维预测 |

**（c）后处理：unpatchify**

```
(B, seq_len, 64) → unpatchify(grid_size=(T_lat, H_lat/ph, W_lat/pw))
                 → (B, out_dim=16, T_lat, H_lat, W_lat)
                 = (B, 16, 9, 22, 40) [DROID 典型值]
```

`unpatchify` 将扁平化的 patch 预测还原为 VAE 潜空间的体积张量。

**（d）损失函数：`dynamics_loss`**

```
1. 计算 Flow Matching target:
   training_target = noise - latents   # 流匹配速度场目标
   Shape: (B, 16, T_lat, H_lat, W_lat)

2. 逐像素 MSE:
   mse = F.mse_loss(video_noise_pred, training_target, reduction='none')
   Shape: (B, C_lat, T_lat, H_lat, W_lat)

3. 对空间维求均值:
   mse_reduced = mse.mean(dim=(1, 3, 4))   # → (B, T_lat)

4. 乘以时间步权重:
   weight = scheduler.training_weight(timestep)  # → (B, T_lat) 
   # training_weight: 线性插值权重表，高噪声时间步权重更大
   weighted = mse_reduced * weight               # → (B, T_lat)

5. 全局平均:
   dynamics_loss = weighted.mean()               # → scalar
```

**关键点**: `dynamics_loss` **没有** mask，对所有视频帧和潜空间通道均计算 loss。

**（e）评估指标**

| 指标 | 位置 | 说明 |
|------|------|------|
| `dynamics_loss_avg` | `BaseTrainer.compute_loss` → `LossLoggerCallback` | 10 步滑动平均，写入 `loss_log.jsonl` |
| FVD / SSIM / PSNR | 仓库外部 | 未在仓库内实现；需外部 sim_evals 或自定义脚本 |

---

#### 11.4.3 Action Head（`action_decoder`，`CategorySpecificMLP`）— 动作噪声预测

**文件**: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

**（a）架构**

```python
self.action_decoder = CategorySpecificMLP(
    num_categories=1,           # 当前固定为单 embodiment
    input_dim=dim,              # 5120
    hidden_dim=hidden_size,     # 5120
    output_dim=action_dim,      # 64 (max_action_dim)
)
# 等价于: Linear(5120→5120) → ReLU → Linear(5120→64)，按 embodiment_id 索引权重
```

**（b）输入/输出**

| 方向 | 张量 | Shape | 含义 |
|------|------|-------|------|
| **输入 x** | DiT 动作段输出 | `(B, action_horizon, 5120)` | 40层 Transformer 处理后的动作 token 特征；`action_horizon=24` |
| **输入 cat_ids** | embodiment ID | `(B,)` | 用于选择 embodiment 特定的 MLP 权重 |
| **输出** | 动作噪声预测 | `(B, action_horizon, action_dim)` = `(B, 24, 64)` | 每步的动作空间噪声预测 |

**（c）损失函数：`action_loss`**

```
1. 计算 Flow Matching target:
   training_target_action = noise_action - actions   # 流匹配速度场目标
   Shape: (B, action_horizon, action_dim) = (B, 24, 64)

2. 逐元素 MSE:
   mse = F.mse_loss(action_noise_pred, training_target_action, reduction='none')
   Shape: (B, 24, 64)

3. 应用 action_mask（维度有效性 mask）:
   masked_mse = mse * action_mask     # (B, 24, 64) × (B, 24, 64)
   # action_mask: 仅保留有效动作维度（非 padding），pad 位置为 0

4. 应用 has_real_action（样本级 mask）:
   masked_mse = has_real_action[:, None].float() * masked_mse   # (B, 1) × (B, 24, 64)
   # has_real_action: 整个样本是否有真实动作标签（如 cotrain 视频数据无动作时为 False）

5. 对 action_dim 维求均值:
   mse_per_step = masked_mse.mean(dim=2)   # → (B, 24)

6. 乘以时间步权重:
   weight = scheduler.training_weight(timestep_action)  # → (B, 24)
   weighted = mse_per_step * weight                     # → (B, 24)

7. 全局平均:
   action_loss = weighted.mean()                        # → scalar
```

**关键点**: Action Head 有 **双重 mask 机制**:
- `action_mask`: 维度级 mask，处理不同 embodiment 的动作维度差异（padding 到 `max_action_dim`）
- `has_real_action`: 样本级 mask，在 cotrain/视频生成等无动作数据场景中屏蔽整个样本

**（d）评估指标**

| 指标 | 位置 | 说明 |
|------|------|------|
| `action_loss_avg` | `BaseTrainer.compute_loss` → `LossLoggerCallback` | 10 步滑动平均，写入 `loss_log.jsonl` |
| Open-loop MSE | `scripts/open_loop_yam.py` | 离线评估：加载模型推理 → 与 GT 动作对比，计算总体/逐键/逐维 MSE，输出 `mse.txt` + 可视化图 |
| Closed-loop 成功率 | 仓库外部 `sim_evals` | 仿真环境中闭环执行策略，统计任务成功率（不在本仓库实现） |

---

#### 11.4.4 总损失函数

```python
loss = dynamics_loss + action_loss
```

两个 head 的损失 **等权相加**（无额外系数），由 `WANPolicyHead.forward()` 返回，经 `BaseTrainer.compute_loss()` 用于反向传播。

当没有动作数据时（`actions.numel() == 0`），`action_loss = 0`，仅计算 `dynamics_loss`。

---

#### 11.4.5 Decoder 部分代码文件清单

| 组件 | 类 | 文件 |
|------|-----|------|
| Video Head | `CausalHead` | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` (L1269-1292) |
| Action Decoder | `CategorySpecificMLP` | 同上 (L74-94, 实例化于 L1417-1421) |
| 底层线性层 | `CategorySpecificLinear` | 同上 (L47-71) |
| unpatchify | `CausalWanModel.unpatchify` | 同上（CausalWanModel 方法） |
| Loss 计算 | `WANPolicyHead.forward` | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (L856-876) |
| Flow target/weight | `FlowMatchScheduler.training_target / training_weight` | `groot/vla/model/dreamzero/modules/flow_match_scheduler.py` |
| 训练日志 | `LossLoggerCallback` | `groot/vla/experiment/base.py` |
| Open-loop 评估 | `evaluate()` | `scripts/open_loop_yam.py` |

---

### 11.5 三部分可训练性总结

| 部分 | 组件 | LoRA 模式 | Full 模式 |
|------|------|----------|----------|
| **Encoder** | WanTextEncoder | ❌ 冻结 | ❌ 冻结 |
| | WanImageEncoder | ❌ 冻结 | ❌ 冻结 |
| | WanVideoVAE | ❌ 冻结 | ❌ 冻结 |
| | state_encoder | ✅ 可训 | ✅ 可训 |
| | action_encoder | ✅ 可训 | ✅ 可训 |
| **Backbone** | 40×CausalWanAttentionBlock | LoRA 适配器可训 | ✅ 全量可训 |
| | patch_embed / text_proj / img_emb | 由 `tune_projector` 控制 | 通常 ✅ |
| **Decoder** | CausalHead (video) | ✅ 可训 | ✅ 可训 |
| | action_decoder | ✅ 可训 | ✅ 可训 |

---

## 12. 参考文件清单（按阅读顺序）

1. `groot/vla/configs/data/dreamzero/droid_relative.yaml` — 数据集与 shard 类  
2. `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` — modality 窗口与 transform 链  
3. `groot/vla/data/dataset/lerobot.py` — parquet/视频读取与步索引  
4. `groot/vla/data/dataset/lerobot_sharded.py` — DROID shard 与 Iterable 采样  
5. `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` — DreamTransform + Collator  
6. `groot/vla/model/dreamzero/base_vla.py` — `VLA` 组装与前向  
7. `groot/vla/model/dreamzero/backbone/identity.py` — Identity backbone  
8. `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` — 训练主循环与 loss  
9. `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` — `CausalWanModel`  
10. `groot/vla/model/dreamzero/modules/wan_video_text_encoder.py` — `WanTextEncoder` 文本编码器  
11. `groot/vla/model/dreamzero/modules/wan_video_image_encoder.py` — `WanImageEncoder` 图像编码器  
12. `groot/vla/model/dreamzero/modules/wan_video_vae.py` — `WanVideoVAE` 视频 VAE  
13. `groot/vla/model/dreamzero/modules/flow_match_scheduler.py` — `FlowMatchScheduler` 流匹配调度器  
14. `groot/vla/experiment/base.py` / `experiment.py` — `VLATrainer`、`compute_loss`  
15. `groot/vla/configs/conf.yaml` — 默认 TrainingArguments 与 trainer 入口  
16. `scripts/train/droid_training_full_finetune.sh` — 一键示例超参  
17. `scripts/open_loop_yam.py` — 离线 open-loop 评估脚本  

---

*文档生成自代码阅读；具体 shape 以你当前 Hydra 覆盖的 `num_frames`、`action_horizon`、`image_resolution_*`、`frame_seqlen`、`num_frame_per_block`、`num_action_per_block` 为准。*
