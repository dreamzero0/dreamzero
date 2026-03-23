# LeRobot Dataset 构造与 Sample 结构分析

本文档说明 `groot/vla/data/dataset/lerobot.py` 中 Dataset 的构造方式、每个 sample 包含的数据、长度与 mask 的设置，以及 video/action 与不同频率原始数据的对齐方式。

---

## 一、Dataset 构造概览

### 1.1 核心类

- **LeRobotSingleDataset**：单数据源（单 embodiment），按「步」索引，每步对应一条轨迹上的一个 **base_index**。
- **LeRobotMixtureDataset**：多数据源混合，按权重采样 (dataset, trajectory_id, step_index)，再委托给对应 dataset 的 `__getitem__`。

### 1.2 索引：`all_steps` 与 `step_filter`

- **`all_steps`**：`list[(trajectory_id, base_index)]`，由 `_get_all_steps()` 生成。
  - 遍历每条轨迹 `trajectory_id`，再遍历该轨迹允许的 `base_index`（来自 `step_filter[trajectory_id]`）。
  - 若存在 `meta/step_filter.jsonl`，则每个 episode 的 `step_indices` 表示要**排除**的步，允许的步为 `np.setdiff1d(all_indices, indices_to_filter)`。
  - 若不存在 step_filter，则允许的步为 `np.arange(trajectory_length)`，即整条轨迹每一步都可作为 base_index。
- **`__len__`**：`len(self.all_steps)`，即「可采样的 (trajectory_id, base_index) 对」总数。

因此，**每个 sample 由一对 (trajectory_id, base_index) 唯一确定**；base_index 必须在 `step_filter` 允许的范围内，且要满足后续 delta 不越界（在 Mixture 采样时用 `max_delta_index` 约束）。

---

## 二、每个 Sample 包含的数据

### 2.1 `__getitem__(index)` 流程

1. `trajectory_id, base_index = self.all_steps[index]`
2. 对每个 modality 的每个 key，计算 **步下标**：`indices[key] = base_index + delta_indices[key]`（均为 numpy 一维数组）。
3. `get_step_data(trajectory_id, indices)` 按 modality 调用 `get_data_by_modality(...)`，得到**未做 Concat/Transform 前的多 key 字典**。
4. 再经过 `self.transforms(data)`（含 ConcatTransform 等），得到**最终 batch 里的一条 sample 结构**（见下）。

### 2.2 各 Modality 的「长度」由谁决定

- 每个 modality 的每个 key 对应一个 **`ModalityConfig`**，其中有：
  - **`delta_indices`**：相对 base_index 的偏移，例如 `[0,1,...,24]`。
- 对某个 key，**步下标** = `base_index + np.array(delta_indices)`，长度 = **len(delta_indices)**。
- 因此：
  - **video**：长度 T_video = len(video_config.delta_indices)，例如 25（OXE Droid 配置）。
  - **state**：长度 T_state = len(state_config.delta_indices)，例如 1。
  - **action**：长度 T_action = len(action_config.delta_indices)，例如 24。
  - **language**：长度 T_lang = len(language_config.delta_indices)，通常为 1，即一条指令字符串。

所以 **video / state / action 的「时间」长度可以不一致**（例如 25 帧视频、24 步 action），由各自的 `delta_indices` 决定；**每个 sample 内，三者都对齐到同一组「控制步」**（见第四节）。

### 2.3 原始 `get_step_data` 返回结构（Transform 前）

- **video**：每个 key 如 `video.exterior_image_1_left`，shape 为 **(T_video, H, W, C)**，即按步取出的 T_video 帧。
- **state**：每个 key 如 `state.joint_position`，shape **(T_state, D_state)**。
- **action**：每个 key 如 `action.joint_position`，shape **(T_action, D_action)**。
- **language**：每个 key 返回 **list of str**，长度为 T_lang（通常为 1）。

越界步会通过 `retrieve_data_and_pad` 做 padding（见下）。

### 2.4 经 ConcatTransform 后的单条 sample（Transform 后、Collator 前）

- **video**：多 view 沿新轴拼接，shape **(T_video, V, H, W, C)**，例如 T_video=25, V=3。
- **state**：多 key 沿最后一维 concat，shape **(T_state, D_state_total)**。
- **action**：多 key 沿最后一维 concat，shape **(T_action, D_action_total)**。
- **language**：仍为 list[str]（或保留多条 key），长度 T_lang；后续在 **Collator** 里被 tokenize 成 **text** 与 **text_attention_mask**。

### 2.5 Collator 之后（送入模型的 batch）

- **video**：`(B, T_video, V, H, W, C)` 或按 data pipeline 的约定。
- **state**：`(B, T_state, D_state)`。
- **action**：`(B, T_action, D_action)`。
- **text**：tokenizer 输出 **input_ids**，shape **(B, text_len)**，例如 512。
- **text_attention_mask**：**(B, text_len)**，1=有效 token，0=padding。

**结论**：每个 sample 包含 **video（按 T_video 帧）、state（按 T_state 步）、action（按 T_action 步）、以及一条语言指令**；**不是「每个 timestamp 都有一一对应的 video 和 action」**，而是 **每个 modality 有自己的长度（由 delta_indices 决定）**，且通过**同一组控制步（base_index + delta）** 对齐。

---

## 三、长度与 Mask 的设置

### 3.1 各模态长度

| 数据       | 长度来源 | 说明 |
|------------|----------|------|
| video      | T_video = len(video_config.delta_indices) | 例如 [0..24] → 25 帧 |
| state      | T_state = len(state_config.delta_indices) | 例如 [0] → 1 |
| action     | T_action = len(action_config.delta_indices) | 例如 [0..23] → 24 |
| language   | T_lang = len(language_config.delta_indices) | 通常 1，一条指令 |

配置示例（OXE Droid）：

- video: `delta_indices: [0,1,...,24]` → 25 帧。
- action: `delta_indices: [0,1,...,23]` → 24 步 action。
- state: `delta_indices: [0]` → 1 个 state 向量。

### 3.2 Padding 与越界

- **state / action**（及 lapa_action、dream_actions 等）：通过 **`retrieve_data_and_pad`** 处理越界步：
  - `step_indices < 0` 或 `>= max_length` 的位置视为 padding。
  - 非 padding 位置从 `array[step_indices[~padding_positions]]` 取值；padding 位置按策略填：
    - **absolute**：用首/末步数据（first_last）。
    - 非 absolute：用 0（zero）。
- **video**：在 `get_video` 中先把 `step_indices` 裁剪到 `[0, trajectory_length-1]`，再取 `timestamp[step_indices]`，用 **`get_frames_by_timestamps`** 按时间戳取帧，因此**不会出现「负索引」的帧**，而是用边界时间戳取边界帧，等价于首/末帧 padding。

### 3.3 Mask

- **Dataset 内部**：lerobot 的 `get_step_data` / `get_data_by_modality` **不**返回 attention_mask；只做步下标与 padding，输出 raw 数组或 list。
- **Text 的 mask**：在 **Collator**（如 `DefaultDataCollator` / `collate`）中，对语言做 tokenize 时使用 `padding='max_length'`, `max_length=max_length`，得到：
  - **input_ids**：**(B, text_len)**，例如 512。
  - **text_attention_mask**：**(B, text_len)**，1=真实 token，0=padding。
- **Video / state / action**：当前 pipeline 中**没有**单独的 validity mask；长度由 delta_indices 和 padding 策略隐式确定。若需要「有效步」mask，需在后续 transform 或 model 侧根据约定自行构造。

---

## 四、Action、Video 是否「每个 timestamp 都有」？对齐方式

### 4.1 是否每个 timestamp 都有 video 和 action？

- **不是**「每条轨迹的每个物理 timestamp 都有一帧 video 和一个 action」。
- 而是：**每个 sample 对应一个 base_index（一个控制步）**，然后：
  - **video**：取 base_index + video_delta_indices 对应的 **T_video 个控制步**的时间戳，用这些时间戳去视频里取 **T_video 帧**。
  - **action**：取 base_index + action_delta_indices 对应的 **T_action 个控制步**的 action 向量。
- 因此，**同一个 sample 内**：video 的 T_video 个「步」与 action 的 T_action 个「步」都来自**同一套控制步时间轴**；若 delta 重叠（例如 video [0..24]、action [0..23]），则前 24 个控制步上既有 video 也有 action，第 25 步只有 video。

### 4.2 原始数据中 video 和 action 频率不一样怎么办？

- **存储与索引单位**：LeRobot 的 parquet 按**控制步**存储：每一行是一个控制步，包含该步的 state、action 以及一个 **timestamp**（和可选的 frame_index）。也就是说，**时间轴是按控制步离散的**，而不是按视频帧或按 action 的原始频率。
- **Video**：  
  - 视频文件是连续时间的；取帧时**不按「第几帧」**，而是按 **timestamp**。  
  - 代码路径：`get_video(trajectory_id, key, step_indices)` → `timestamp = self.curr_traj_data["timestamp"].to_numpy()`，`video_timestamp = timestamp[step_indices]` → **`get_frames_by_timestamps(video_path, video_timestamp, ...)`**。  
  - 即：用**该控制步的 timestamp** 去视频里找**最接近该时刻的一帧**。因此无论视频是 30fps 还是 10fps，**每个控制步只对应一帧**，由该步的 timestamp 决定。
- **Action / State**：  
  - 直接从 parquet 的对应行（控制步）读取，**每个控制步一行**，与视频通过 **同一列 timestamp** 对齐到同一时间轴。
- **结论**：  
  - **对齐维度是「控制步」+「该步的 timestamp」**：video 用 timestamp 取帧，action/state 用步索引取行。  
  - 若原始数据里 video 是 30fps、控制是 10Hz，则每个控制步会通过 timestamp 映射到 30fps 视频的某一帧（最近邻），从而**在「控制步」粒度上对齐**，不会出现「每个物理 timestamp 都有一一对应的 video 和 action」的逐帧对齐，而是**每个 sample 一段控制步窗口内的 T_video 帧 + T_action 步 action**。

---

## 五、简要对照表

| 项目           | 说明 |
|----------------|------|
| 样本单位       | 一个 (trajectory_id, base_index)，base_index 来自 step_filter |
| 长度           | video/state/action 长度分别由各自 ModalityConfig.delta_indices 决定，可不同 |
| 对齐           | 同一 base_index + 各自 delta_indices → 同一批控制步；video 用该步 timestamp 取帧 |
| 越界           | state/action 用 retrieve_data_and_pad（first_last 或 zero）；video 用边界 timestamp 取边界帧 |
| Mask           | 仅 text 在 Collator 中产生 text_attention_mask（1/0）；video/state/action 无显式 mask |
| 频率不一致     | 以控制步+timestamp 为准：video 按 timestamp 取帧，action/state 按步索引取行，在控制步维度对齐 |

以上即为 `lerobot.py` 中 dataset 的构造方式、每个 sample 的数据与长度、以及 video/action 对齐与频率差异的处理方式。
