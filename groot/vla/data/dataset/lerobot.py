from collections import defaultdict
import copy
import glob
import hashlib
import importlib
import json
from pathlib import Path
import time
from typing import Sequence, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml
import torch

from groot.vla.common.utils import get_all_frames, get_frames_by_timestamps
from groot.vla.data.conversion.gr1.get_initial_actions import load_initial_actions
from groot.vla.data.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    EmbodimentTag,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from groot.vla.data.transform import ComposedModalityTransform

T_LeRobotMixtureDataset = TypeVar("T_LeRobotMixtureDataset", bound="LeRobotMixtureDataset")

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_TASK_EMBEDDINGS_FILENAME = "meta/task_embeddings.pt"
LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME = "meta/episodes_detail_global_instruction.jsonl"
INITIAL_ACTIONS_FILENAME = "meta/initial_actions.npz"
METADATA_DIR = Path(importlib.import_module("groot.vla.data").__file__).parent / "metadata"  # type: ignore
STEP_FILTER_FILENAME = "meta/step_filter.jsonl"
LEROBOT_RELATIVE_STATS_FILE_NAME = "meta/relative_stats_dreamzero.json"
LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME = "meta/relative_horizon_stats_dreamzero.json"

# Special language keys that load from metadata files instead of parquet columns
METADATA_LANG_KEYS = ["detailed_global_instruction_medium", "detailed_global_instruction_concise"]


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, DatasetStatisticalValues]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = DatasetStatisticalValues(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality defining how data should be sampled and loaded.

    This class specifies which indices to sample relative to a base index and which
    keys to load for a particular modality (e.g., video, state, action).
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    eval_delta_indices: list[int] | None = None
    """Delta indices to sample relative to the current index for evaluation. If None, uses the same indices as delta_indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

    def model_post_init(self, *args, **kwargs):
        """Initialize eval_delta_indices to delta_indices if not provided."""
        super().model_post_init(*args, **kwargs)
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


class LeRobotSingleDataset(Dataset):
    """
    LeRobot单数据集类：从LeRobot格式（parquet+视频）加载多模态机器人数据。

    【作用与原理】
    该类是DreamZero数据管道的核心入口，负责：
    1. 读取LeRobot格式的meta信息（info.json/modality.json/episodes.jsonl等）
    2. 根据ModalityConfig定义的delta_indices采样时间窗口
    3. 从parquet加载state/action/语言等低维数据
    4. 从mp4视频文件按时间戳采样图像帧
    5. 支持相对动作归一化（relative_action）和分块读取（chunk）

    【数据流位置】
    上游：磁盘LeRobot格式数据集（由convert_droid.py等生成）
    下游：ComposedModalityTransform → DreamTransform → Collator → VLA模型

    【关键概念】
    - delta_indices: 相对base_index的时间偏移，决定每个sample的时间窗口长度
    - step_filter: 过滤某些episode的特定step（用于过滤静止帧等）
    - modality: 数据模态（video/state/action/language/lapa_action/dream_actions等）

    【典型配置示例】（DROID）
    - video.delta_indices: [0..24] → T_video=25帧
    - action.delta_indices: [0..23] → T_action=24步
    - state.delta_indices: [0] → T_state=1步
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        use_global_metadata: bool = True,
        metadata_version: str | None = None,
        video_backend: str = "ffmpeg",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        discard_bad_trajectories: bool = True,
        fps: float = None,
        max_chunk_size: int = None,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
        relative_action_per_horizon: bool = False,
    ):
        """
        初始化LeRobot数据集。

        Args:
            dataset_path (Path | str): LeRobot数据集根目录路径（包含meta/和data/）。
            modality_configs (dict[str, ModalityConfig]): 各模态的配置字典。
                key: 模态名（"video"/"state"/"action"/"language"等）。
                value: ModalityConfig，定义delta_indices和modality_keys。
            embodiment_tag (str | EmbodimentTag): 本体标识（如"oxe_droid"/"agibot"）。
            use_global_metadata (bool): 是否使用全局预计算统计信息（而非每个数据集独立计算）。
            metadata_version (str | None): 全局元数据版本（如"0221"）。
            video_backend (str): 视频读取后端（"ffmpeg"/"decord"/"torchvision_av"）。
            video_backend_kwargs (dict | None): 视频后端参数。
            transforms (ComposedModalityTransform | None): 数据变换管道（ToTensor/Resize/Normalize等）。
            discard_bad_trajectories (bool): 是否丢弃meta中标记为失败的episode。
            fps (float | None): 视频帧率（覆盖dataset默认）。
            max_chunk_size (int | None): 每个chunk的episode数量（用于分块读取）。
            relative_action (bool): 是否使用相对动作（当前step与初始step的差）。
            relative_action_keys (list[str] | None): 应用relative action的键列表。
            relative_action_per_horizon (bool): 是否为每个horizon位置单独计算relative stats。

        【初始化流程】
        1. 加载meta/modality.json → lerobot_modality_meta
        2. 加载meta/info.json → lerobot_info_meta（含data_path/video_path/chunk_size等）
        3. 加载meta/stats.json → lerobot_stats_meta（归一化统计）
        4. 加载meta/episodes.jsonl → trajectory_ids/trajectory_lengths
        5. 加载step_filter（如有）→ 确定可用的base_index
        6. 生成all_steps列表（所有可采样的(trajectory_id, base_index)对）
        7. 加载relative stats（如启用relative_action）
        8. 设置transforms的metadata
        """"
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.use_global_metadata = use_global_metadata
        self.metadata_version = metadata_version
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.fps = fps
        self.max_chunk_size = max_chunk_size
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )
        self.discard_bad_trajectories = discard_bad_trajectories
        self.relative_action = relative_action
        self.relative_action_per_horizon = relative_action_per_horizon
        # Determine which action keys should use relative action
        if relative_action_keys is not None:
            self.relative_action_keys = relative_action_keys
        else:
            # Default: apply to all action keys except those containing 'gripper'
            self.relative_action_keys = None  # Will be set after modality_configs is available
        self._relative_action_keys_input = relative_action_keys  # Store original input
        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        self.tag = EmbodimentTag(embodiment_tag)
        # For dream and lapa, we use the global metadata since the lapa_actions and dream_actions are already normalized
        if self.tag == EmbodimentTag.DREAM or self.tag == EmbodimentTag.LAPA:
            self.use_global_metadata = True
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        # Notice: We also include discarded trajectories in stats for larger state coverage, for questions please ask @Fengyuan Hu @Yuqi Xie
        self._lerobot_stats_meta = self._get_lerobot_stats_meta()
        
        # Initialize trajectory info and chunk size early (needed for relative stats calculation)
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._data_path_pattern = self._get_data_path_pattern()
        self._chunk_size = self._get_chunk_size()
        
        # Set default relative_action_keys if not provided
        if self.relative_action and self._relative_action_keys_input is None:
            # Default: apply to all action keys except those containing 'gripper'
            action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[0], modality_keys=[])).modality_keys
            self.relative_action_keys = [
                k.replace("action.", "") for k in action_keys 
                if "gripper" not in k.lower()
            ]
            print(f"Relative action will be applied to keys: {self.relative_action_keys}")
        # Load relative action stats if relative_action is enabled
        print("relative_action", self.relative_action)
        self._lerobot_relative_stats_meta = self._get_lerobot_relative_stats_meta() if self.relative_action else {}
        # Load per-horizon relative action stats if relative_action_per_horizon is enabled
        print("relative_action_per_horizon", self.relative_action_per_horizon)
        self._lerobot_relative_horizon_stats_meta = self._get_lerobot_relative_horizon_stats_meta() if self.relative_action_per_horizon else {}
        self._metadata = self._get_metadata()
        self._step_filter = self._get_step_filter()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._max_delta_index = self._get_max_delta_index()
        self._dataset_name = self._dataset_path.name

        # NOTE(YL): method to predict the task progress
        if "action.task_progress" in self._modality_keys["action"]:
            from groot.vla.data.schema import StateActionMetadata

            print("we will add task progress to the action modality")
            self._modality_keys["action"].append("action.task_progress")
            self._metadata.modalities.action["task_progress"] = StateActionMetadata(
                absolute=True, rotation_type=None, shape=(1,), continuous=True
            )
            # assume the task progress is uniformly distributed between 0 and 1
            self._metadata.statistics.action["task_progress"] = DatasetStatisticalValues(
                max=[1.0], min=[0.0], mean=[0.5], std=[0.2887], q01=[0.01], q99=[0.99]
            )

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # LeRobot-specific config (some already initialized above for relative stats)
        self._video_path_pattern = self._get_video_path_pattern()
        self._tasks = self._get_tasks()
        self._detailed_global_instructions = self._get_detailed_global_instructions()
        self.curr_traj_data = None
        self.curr_traj_id = None

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    def _get_max_delta_index(self) -> int:
        """Calculate the maximum delta index across all modalities.

        Returns:
            int: The maximum delta index value.
        """
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        """The maximum delta index across all modalities."""
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_stats_meta

    @property
    def lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """The relative action stats metadata for the LeRobot dataset."""
        return self._lerobot_relative_stats_meta

    @property
    def lerobot_relative_horizon_stats_meta(self) -> dict[str, dict[str, list]]:
        """The per-horizon relative action stats metadata for the LeRobot dataset.
        
        Format: {action_key: {stat_name: [[h0_vals], [h1_vals], ...]}}
        """
        return self._lerobot_relative_horizon_stats_meta

    @property
    def step_filter(self) -> dict[int, np.ndarray]:
        """The step filter for the dataset."""
        return self._step_filter

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            modality_meta_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LE_ROBOT_MODALITY_FILENAME).name
            )
            assert (
                modality_meta_path.exists()
            ), f"Please provide a {Path(LE_ROBOT_MODALITY_FILENAME).name} file in {METADATA_DIR / self.tag.value / self.metadata_version}"
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return modality_meta
        else:
            modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
            assert (
                modality_meta_path.exists()
            ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Get the metadata for the LeRobot dataset."""
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LE_ROBOT_STATS_FILENAME).name
            )
        else:
            stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name in ["num_trajectories", "total_trajectory_length"]:
                stats.pop(name, None)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats
        except (FileNotFoundError, ValidationError) as e:
            if self.use_global_metadata:
                raise ValueError(
                    f"{e}: Please provide a {Path(LE_ROBOT_STATS_FILENAME).name} file in {stats_path}"
                    " and ensure the metadata format is correct."
                )
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            lowdim_features = []
            le_features = self.lerobot_info_meta["features"]
            for feature in le_features:
                if "float" in le_features[feature]["dtype"]:
                    lowdim_features.append(feature)

            stats = calculate_dataset_statistics(parquet_files, lowdim_features)
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            with open(stats_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)
            return stats

    def _get_lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Get the relative action stats metadata for the LeRobot dataset.
        
        Returns:
            dict[str, DatasetStatisticalValues]: Dictionary mapping action keys to their relative stats.
        """
        # Determine the path for relative stats file
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LEROBOT_RELATIVE_STATS_FILE_NAME).name
            )
            assert (
                stats_path.exists()
            ), f"Please provide a {Path(LEROBOT_RELATIVE_STATS_FILE_NAME).name} file in {METADATA_DIR / self.tag.value / self.metadata_version}"
        else:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME
        
        # Try to load existing relative stats
        if stats_path.exists():
            print(f"Loading relative action stats from {stats_path}")
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats
        
        # Calculate relative stats if file doesn't exist
        print(f"Relative stats file not found at {stats_path}")
        print(f"Calculating relative action stats for {self.dataset_name}")
        
        # Get action keys from modality configs, filtered by relative_action_keys
        all_action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[0], modality_keys=[])).modality_keys
        if not all_action_keys:
            print("No action keys found in modality configs, skipping relative stats calculation")
            return {}
        
        # Filter to only the keys that should use relative action
        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)
        
        if not action_keys_to_process:
            print("No action keys to process for relative stats")
            return {}
        
        print(f"Will calculate relative stats for: {action_keys_to_process}")
        
        stats = {}
        for action_key in action_keys_to_process:
            print(f"Calculating relative stats for action key: {action_key}")
            try:
                relative_stats = self._calculate_relative_stats_for_key(action_key)
                stats[action_key] = relative_stats
            except Exception as e:
                print(f"Failed to calculate relative stats for {action_key}: {e}")
                continue
        
        if stats:
            # Save the calculated stats
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            # Only save to dataset path (not global metadata path)
            save_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME
            print(f"Saving relative action stats to {save_path}")
            with open(save_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)
        
        return stats

    def _get_lerobot_relative_horizon_stats_meta(self) -> dict[str, dict[str, list]]:
        """Get the per-horizon relative action stats metadata for the LeRobot dataset.
        
        Similar to _get_lerobot_relative_stats_meta but calculates separate stats for each
        action horizon index. Will load from file if exists, otherwise calculate and save.
        
        Returns:
            dict[str, dict[str, list]]: Nested dictionary where:
                - Outer key is the action key (e.g., 'joint_position')
                - Inner key is the stat name (e.g., 'max', 'min', 'mean', 'std', 'q01', 'q99')
                - Value is a list of stat values per horizon index
        """
        # Determine the path for per-horizon relative stats file
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME).name
            )
            assert (
                stats_path.exists()
            ), f"Please provide a {Path(LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME).name} file in {METADATA_DIR / self.tag.value / self.metadata_version}"
        else:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
        
        # Try to load existing per-horizon relative stats
        if stats_path.exists():
            print(f"Loading per-horizon relative action stats from {stats_path}")
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            return stats
        
        # Calculate per-horizon relative stats if file doesn't exist
        print(f"Per-horizon relative stats file not found at {stats_path}")
        print(f"Calculating per-horizon relative action stats for {self.dataset_name}")
        
        # Get action keys from modality configs, filtered by relative_action_keys
        all_action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[0], modality_keys=[])).modality_keys
        if not all_action_keys:
            print("No action keys found in modality configs, skipping per-horizon relative stats calculation")
            return {}
        
        # Filter to only the keys that should use relative action
        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)
        
        if not action_keys_to_process:
            print("No action keys to process for per-horizon relative stats")
            return {}
        
        print(f"Will calculate per-horizon relative stats for: {action_keys_to_process}")
        
        stats = {}
        for action_key in action_keys_to_process:
            print(f"Calculating per-horizon relative stats for action key: {action_key}")
            try:
                relative_stats = self._calculate_relative_stats_for_key_per_horizon(action_key)
                stats[action_key] = relative_stats
            except Exception as e:
                print(f"Failed to calculate per-horizon relative stats for {action_key}: {e}")
                continue
        
        if stats:
            # Only save to dataset path (not global metadata path)
            save_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
            print(f"Saving per-horizon relative action stats to {save_path}")
            with open(save_path, "w") as f:
                json.dump(stats, f, indent=4)
        
        return stats

    def _calculate_relative_stats_for_key(self, action_key: str) -> DatasetStatisticalValues:
        """Calculate relative action statistics for a specific action key.
        
        Args:
            action_key: The action key to calculate stats for (e.g., 'joint_position')
            
        Returns:
            DatasetStatisticalValues: The calculated statistics for the relative action.
        """
        # Get state and action metadata from lerobot modality config
        state_key = action_key  # Assume state key matches action key
        
        # Get the modality metadata to find original column names and indices
        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)
        
        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(f"Action key '{action_key}' not found in modality metadata")
        
        # Get the original column names (e.g., 'observation.state', 'action')
        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key
        
        # Get the indices to slice from the concatenated vectors
        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end
        
        state_delta_indices = self.modality_configs.get("state", ModalityConfig(delta_indices=[0], modality_keys=[])).delta_indices
        action_delta_indices = self.modality_configs["action"].delta_indices
        
        print(f"Calculating relative stats for {action_key}:")
        print(f"  State: column='{state_original_key}', indices=[{state_start}:{state_end}]")
        print(f"  Action: column='{action_original_key}', indices=[{action_start}:{action_end}]")
        
        # # Calculate relative actions for all trajectories
        all_relative_actions = []
        
        # for traj_id in tqdm(self.trajectory_ids, desc=f"Calculating relative stats for {action_key}"):
        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            # Randomly sample 500 trajectories
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(traj_ids_to_process), size=max_trajs_for_stats, replace=False)
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            print(f"Sampling {max_trajs_for_stats} trajectories out of {len(self.trajectory_ids)} for stats calculation")
        
        # Calculate relative actions for sampled trajectories
        all_relative_actions = []
        
        for traj_id in tqdm(traj_ids_to_process, desc=f"Calculating relative stats for {action_key}"):
            try:
                # Load trajectory data
                traj_data = self._load_trajectory_data(traj_id)
                if traj_data is None:
                    continue
                
                # Check if columns exist
                if state_original_key not in traj_data.columns or action_original_key not in traj_data.columns:
                    print(f"Missing columns: state='{state_original_key}' exists={state_original_key in traj_data.columns}, "
                          f"action='{action_original_key}' exists={action_original_key in traj_data.columns}")
                    continue
                
                # Load full state and action arrays, then slice to get the specific component
                full_state_data = np.stack(traj_data[state_original_key].values)
                full_action_data = np.stack(traj_data[action_original_key].values)
                
                # Slice to get just the component we care about (e.g., joint_position)
                state_data = full_state_data[:, state_start:state_end]
                action_data = full_action_data[:, action_start:action_end]
                
                # Calculate usable length based on action delta indices
                usable_length = len(traj_data) - max(action_delta_indices)
                
                for i in range(usable_length):
                    # Get reference state (last state before action chunk)
                    ref_state_idx = state_delta_indices[-1] + i
                    if ref_state_idx >= len(state_data):
                        continue
                    ref_state = state_data[ref_state_idx]
                    
                    # Get action chunk
                    action_indices = [idx + i for idx in action_delta_indices]
                    if max(action_indices) >= len(action_data):
                        continue
                    actions = action_data[action_indices]

                    # print("actions shape", actions.shape, "ref_state shape", ref_state.shape)
                    
                    # Calculate relative actions (action - reference state)
                    relative_actions = actions - ref_state
                    all_relative_actions.extend(relative_actions)
                    
            except Exception as e:
                print(f"Error processing trajectory {traj_id}: {e}")
                continue
        
        if not all_relative_actions:
            raise ValueError(f"No relative actions calculated for {action_key}")
        
        all_relative_actions = np.array(all_relative_actions)
        print(f"Collected {len(all_relative_actions)} relative action samples for {action_key}")
        
        return DatasetStatisticalValues(
            max=np.max(all_relative_actions, axis=0).tolist(),
            min=np.min(all_relative_actions, axis=0).tolist(),
            mean=np.mean(all_relative_actions, axis=0).tolist(),
            std=np.std(all_relative_actions, axis=0).tolist(),
            q01=np.quantile(all_relative_actions, 0.01, axis=0).tolist(),
            q99=np.quantile(all_relative_actions, 0.99, axis=0).tolist(),
        )

    def _calculate_relative_stats_for_key_per_horizon(
        self, action_key: str
    ) -> dict[str, list]:
        """Calculate relative action statistics for each delta index (horizon step) separately.
        
        Unlike `_calculate_relative_stats_for_key` which pools all horizon steps together,
        this method calculates separate statistics for each action horizon index.
        
        Args:
            action_key: The action key to calculate stats for (e.g., 'joint_position')
            
        Returns:
            dict[str, list]: Dictionary where keys are stat names (max, min, mean, std, q01, q99)
                and values are lists of stat values per horizon index.
                Format: {"max": [[h0_vals], [h1_vals], ...], "min": [...], ...}
        """
        # Get state and action metadata from lerobot modality config
        state_key = action_key  # Assume state key matches action key
        
        # Get the modality metadata to find original column names and indices
        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)
        
        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(f"Action key '{action_key}' not found in modality metadata")
        
        # Get the original column names (e.g., 'observation.state', 'action')
        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key
        
        # Get the indices to slice from the concatenated vectors
        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end
        
        state_delta_indices = self.modality_configs.get("state", ModalityConfig(delta_indices=[0], modality_keys=[])).delta_indices
        action_delta_indices = self.modality_configs["action"].delta_indices
        
        print(f"Calculating per-horizon relative stats for {action_key}:")
        print(f"  State: column='{state_original_key}', indices=[{state_start}:{state_end}]")
        print(f"  Action: column='{action_original_key}', indices=[{action_start}:{action_end}]")
        print(f"  Action delta indices: {action_delta_indices}")
        
        # Initialize separate lists for each horizon index
        all_relative_actions_per_horizon: dict[int, list] = {
            delta_idx: [] for delta_idx in action_delta_indices
        }
        
        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            # Randomly sample trajectories
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(traj_ids_to_process), size=max_trajs_for_stats, replace=False)
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            print(f"Sampling {max_trajs_for_stats} trajectories out of {len(self.trajectory_ids)} for stats calculation")
        
        for traj_id in tqdm(traj_ids_to_process, desc=f"Calculating per-horizon relative stats for {action_key}"):
            try:
                # Load trajectory data
                traj_data = self._load_trajectory_data(traj_id)
                if traj_data is None:
                    continue
                
                # Check if columns exist
                if state_original_key not in traj_data.columns or action_original_key not in traj_data.columns:
                    continue
                
                # Load full state and action arrays, then slice to get the specific component
                full_state_data = np.stack(traj_data[state_original_key].values)
                full_action_data = np.stack(traj_data[action_original_key].values)
                
                # Slice to get just the component we care about (e.g., joint_position)
                state_data = full_state_data[:, state_start:state_end]
                action_data = full_action_data[:, action_start:action_end]
                
                # Calculate usable length based on action delta indices
                usable_length = len(traj_data) - max(action_delta_indices)
                
                for i in range(usable_length):
                    # Get reference state (last state before action chunk)
                    ref_state_idx = state_delta_indices[-1] + i
                    if ref_state_idx >= len(state_data):
                        continue
                    ref_state = state_data[ref_state_idx]
                    
                    # Get action for each horizon index separately
                    for delta_idx in action_delta_indices:
                        action_idx = delta_idx + i
                        if action_idx >= len(action_data):
                            continue
                        action = action_data[action_idx]
                        
                        # Calculate relative action (action - reference state)
                        relative_action = action - ref_state
                        all_relative_actions_per_horizon[delta_idx].append(relative_action)
                        
            except Exception as e:
                print(f"Error processing trajectory {traj_id}: {e}")
                continue
        
        # Calculate stats for each horizon index and organize by stat name
        stat_names = ["max", "min", "mean", "std", "q01", "q99"]
        stats_by_name: dict[str, list] = {name: [] for name in stat_names}
        
        for delta_idx in action_delta_indices:
            relative_actions = all_relative_actions_per_horizon[delta_idx]
            if not relative_actions:
                print(f"Warning: No relative actions calculated for {action_key} at horizon index {delta_idx}")
                # Add empty/placeholder values
                for name in stat_names:
                    stats_by_name[name].append([])
                continue
            
            relative_actions_array = np.array(relative_actions)
            print(f"Collected {len(relative_actions_array)} relative action samples for {action_key} at horizon {delta_idx}")
            
            stats_by_name["max"].append(np.max(relative_actions_array, axis=0).tolist())
            stats_by_name["min"].append(np.min(relative_actions_array, axis=0).tolist())
            stats_by_name["mean"].append(np.mean(relative_actions_array, axis=0).tolist())
            stats_by_name["std"].append(np.std(relative_actions_array, axis=0).tolist())
            stats_by_name["q01"].append(np.quantile(relative_actions_array, 0.01, axis=0).tolist())
            stats_by_name["q99"].append(np.quantile(relative_actions_array, 0.99, axis=0).tolist())
        
        return stats_by_name

    def get_relative_stats_per_horizon(
        self, 
        action_keys: list[str] | None = None,
        save_to_file: bool = True,
    ) -> dict[str, dict[str, list]]:
        """Get relative action stats calculated separately for each horizon index.
        
        This is useful when you want different normalization for different action horizon
        steps, e.g., near-future actions vs far-future actions might have different distributions.
        
        Args:
            action_keys: List of action keys to calculate stats for. If None, uses 
                relative_action_keys (all action keys except gripper by default).
            save_to_file: Whether to save the calculated stats to a file.
            
        Returns:
            dict[str, dict[str, list]]: Nested dictionary where:
                - Outer key is the action key (e.g., 'joint_position')
                - Inner key is the stat name (e.g., 'max', 'min', 'mean', 'std', 'q01', 'q99')
                - Value is a list of stat values per horizon index
                
        Example output format:
            {
                "joint_position": {
                    "max": [[h0_vals], [h1_vals], ...],
                    "min": [[h0_vals], [h1_vals], ...],
                    ...
                }
            }
        """
        # Determine which action keys to process
        if action_keys is None:
            all_action_keys = self.modality_configs.get(
                "action", ModalityConfig(delta_indices=[0], modality_keys=[])
            ).modality_keys
            if not all_action_keys:
                print("No action keys found in modality configs")
                return {}
            # Default: apply to all action keys except those containing 'gripper'
            action_keys = [
                k.replace("action.", "") for k in all_action_keys 
                if "gripper" not in k.lower()
            ]
        
        if not action_keys:
            print("No action keys to process for per-horizon relative stats")
            return {}
        
        print(f"Calculating per-horizon relative stats for: {action_keys}")
        
        all_stats: dict[str, dict[str, list]] = {}
        
        for action_key in action_keys:
            print(f"Processing action key: {action_key}")
            try:
                stats_per_horizon = self._calculate_relative_stats_for_key_per_horizon(action_key)
                all_stats[action_key] = stats_per_horizon
            except Exception as e:
                print(f"Failed to calculate per-horizon relative stats for {action_key}: {e}")
                continue
        
        if save_to_file and all_stats:
            # Save to the designated file
            save_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
            print(f"Saving per-horizon relative action stats to {save_path}")
            
            with open(save_path, "w") as f:
                json.dump(all_stats, f, indent=4)
        
        return all_stats

    def load_relative_stats_per_horizon(
        self,
        stats_path: Path | str | None = None,
    ) -> dict[str, dict[str, list]]:
        """Load pre-computed per-horizon relative stats from a file.
        
        Args:
            stats_path: Path to the stats file. If None, uses the default path in the dataset.
            
        Returns:
            dict[str, dict[str, list]]: Nested dictionary of stats per horizon.
        """
        if stats_path is None:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
        else:
            stats_path = Path(stats_path)
        
        if not stats_path.exists():
            print(f"Per-horizon relative stats file not found at {stats_path}")
            return {}
        
        print(f"Loading per-horizon relative stats from {stats_path}")
        with open(stats_path, "r") as f:
            all_stats = json.load(f)
        
        return all_stats

    def _load_trajectory_data(self, traj_id: int) -> pd.DataFrame | None:
        """Load trajectory data from parquet file.
        
        Args:
            traj_id: The trajectory ID to load.
            
        Returns:
            pd.DataFrame or None if loading fails.
        """
        try:
            chunk_index = traj_id // self.chunk_size
            parquet_path = self.dataset_path / f"data/chunk-{chunk_index:03d}/episode_{traj_id:06d}.parquet"
            if not parquet_path.exists():
                # Try alternative pattern
                parquet_files = list(self.dataset_path.glob(f"data/*/episode_{traj_id:06d}.parquet"))
                if parquet_files:
                    parquet_path = parquet_files[0]
                else:
                    return None
            return pd.read_parquet(parquet_path)
        except Exception:
            return None

    def _get_step_filter(self) -> dict[int, np.ndarray]:
        """Get the step filter for the dataset."""
        step_filter_path = self.dataset_path / STEP_FILTER_FILENAME
        step_filter = {}
        if step_filter_path.exists():
            with open(step_filter_path, "r") as f:
                for line in f:
                    episode_step_filter = json.loads(line)
                    trajectory_id = episode_step_filter["episode_index"]
                    all_indices = np.arange(self.trajectory_lengths[trajectory_id].item())
                    indices_to_filter = np.array(episode_step_filter["step_indices"])
                    step_filter[trajectory_id] = np.setdiff1d(all_indices, indices_to_filter)
        else:
            for trajectory_id in self.trajectory_ids:
                step_filter[trajectory_id] = np.arange(
                    self.trajectory_lengths[trajectory_id].item()
                )
        return step_filter

    def _get_metadata(self) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                self.lerobot_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in self.lerobot_modality_meta.video:
            original_key = self.lerobot_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        dataset_statistics = {}
        le_statistics = {k: v.model_dump() for k, v in self.lerobot_stats_meta.items()}
        # Prepare relative stats if available
        relative_stats = {}
        if self.relative_action and hasattr(self, '_lerobot_relative_stats_meta'):
            relative_stats = {k: v.model_dump() for k, v in self._lerobot_relative_stats_meta.items()}
        
        # Prepare per-horizon relative stats if available
        per_horizon_stats = {}
        if self.relative_action_per_horizon and hasattr(self, '_lerobot_relative_horizon_stats_meta'):
            per_horizon_stats = self._lerobot_relative_horizon_stats_meta
        
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = self.lerobot_modality_meta.get_key_meta(
                    f"{our_modality}.{subkey}"
                )
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                
                # Check if we should use per-horizon relative stats for this action key
                should_use_per_horizon = (
                    our_modality == "action"
                    and self.relative_action_per_horizon
                    and subkey in per_horizon_stats
                    and (self.relative_action_keys is None or subkey in self.relative_action_keys)
                )
                
                # Use relative stats for action modality if relative_action is enabled and stats are available
                # Also check if this subkey is in the list of keys that should use relative action
                should_use_relative = (
                    our_modality == "action" 
                    and self.relative_action 
                    and subkey in relative_stats
                    and (self.relative_action_keys is None or subkey in self.relative_action_keys)
                )
                
                if should_use_per_horizon:
                    # Use per-horizon relative action stats (format: {stat_name: [[h0_vals], [h1_vals], ...]})
                    for stat_name in per_horizon_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = per_horizon_stats[subkey][stat_name]
                    print(f"Using per-horizon relative stats for {subkey}")
                elif should_use_relative:
                    # Use relative action stats directly
                    for stat_name in relative_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = relative_stats[subkey][stat_name]
                    print(f"Using relative stats for {subkey}: {dataset_statistics[our_modality][subkey]}")
                else:
                    # Use original absolute stats
                    le_modality = state_action_meta.original_key
                    for stat_name in le_statistics[le_modality]:
                        indices = np.arange(
                            state_action_meta.start,
                            state_action_meta.end,
                        )
                        stat = np.array(le_statistics[le_modality][stat_name])
                        dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=self.tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[int, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.step_filter: {
                0: [0, 1, 2],
                1: [0, 1],
                2: [0, 2, 3]
            }
            return: [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1),
                (2, 0), (2, 2), (2, 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        # All steps is used in single dataset, so we need to discard bad trajectories
        # Mixture dataset directly use trajectory_ids, so we handle it by changing the sampling weights
        discarded_episode_indices = []
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get("discarded_episode_indices", [])

        for trajectory_id in self.trajectory_ids:
            if trajectory_id in discarded_episode_indices:
                continue
            for base_index in self.step_filter[trajectory_id]:
                all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.

        Returns:
            dict: Dictionary mapping modality names to their keys.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")
    
    def _get_task_embeddings(self) -> dict:
        """Get the task embeddings for the dataset."""
        task_embeddings_path = self.dataset_path / LE_ROBOT_TASK_EMBEDDINGS_FILENAME
        return torch.load(task_embeddings_path)

    def _get_detailed_global_instructions(self) -> dict[int, dict]:
        """Get the detailed global instructions for the dataset.
        
        Loads from episodes_detail_global_instruction.jsonl if it exists.
        
        Returns:
            dict[int, dict]: Mapping from episode_index to detailed instruction dict.
        """
        detailed_instruction_path = self.dataset_path / LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME
        if not detailed_instruction_path.exists():
            return {}
        with open(detailed_instruction_path, "r") as f:
            instructions_list = [json.loads(line) for line in f]
        return {entry["episode_index"]: entry for entry in instructions_list}

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality, modality_config in self.modality_configs.items():
            if modality in ["lapa_action", "dream_actions", "rl_info", "task_embedding"]:
                continue
            for key in modality_config.modality_keys:

                if key == "action.task_progress":
                    continue
                # Skip metadata-based language keys (they don't need modality metadata)
                if modality == "language" and key.startswith("annotation."):
                    lang_subkey = key.replace("annotation.", "")
                    if lang_subkey in METADATA_LANG_KEYS:
                        continue
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)
        # Also set per-horizon statistics if available
        if self.relative_action_per_horizon and hasattr(self, '_lerobot_relative_horizon_stats_meta'):
            if hasattr(self.transforms, 'set_per_horizon_statistics'):
                self.transforms.set_per_horizon_statistics(self._lerobot_relative_horizon_stats_meta)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """
        获取数据集的总样本数。

        【原理】
        返回self.all_steps的长度，即所有可采样的(trajectory_id, base_index)对的数量。
        每个样本对应一个控制步（base_index），根据各模态的delta_indices可提取时间窗口。

        【计算方式】
        all_steps = sum_{trajectory} len(step_filter[trajectory])
        若启用discard_bad_trajectories，则排除discarded_episode_indices中的episode。

        Returns:
            int: 数据集总样本数（所有可采样的控制步）。
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """
        获取单个训练样本（经过transforms的完整数据）。

        【作用】
        这是Dataset的核心接口，被DataLoader调用。流程：
        1. 从all_steps[index]获取(trajectory_id, base_index)
        2. 为每个modality_key计算采样索引：indices[key] = base_index + delta_indices[key]
        3. 调用get_step_data加载原始数据（video/state/action/language）
        4. 应用transforms（ComposedModalityTransform → DreamTransform）

        【输入输出】
        - 输入: index (int) - 样本在all_steps中的索引，范围[0, len(self))
        - 输出: dict - 经过transform后的数据字典，键包括：
          * "video": (T_v, V, H, W, C) uint8 或经变换后的张量
          * "state": (T_s, D_state) float
          * "action": (T_a, D_action) float
          * "language": str 或 list[str]
          * 经DreamTransform后还有"images", "text", "state_mask", "action_mask"等

        【调用关系】
        - 被: DataLoader（通过PyTorch Dataset接口）
        - 调用: get_step_data() → transforms()

        【时间窗口对齐】
        假设base_index=100，video.delta_indices=[0,1,2]，action.delta_indices=[0,1]：
        - video采样索引: [100, 101, 102] → 3帧
        - action采样索引: [100, 101] → 2步
        - 两者在base_index=100处对齐，但长度可能不同

        Args:
            index (int): 样本索引。

        Returns:
            dict: 经过transforms处理后的完整样本数据。
        """
        trajectory_id, base_index = self.all_steps[index]
        indices = {
            key: delta_indices + base_index for key, delta_indices in self.delta_indices.items()
        }
        return self.transforms(self.get_step_data(trajectory_id, indices))

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        """
        获取单个控制步的原始数据（未应用transforms）。

        【作用】
        从磁盘加载指定trajectory和indices的多模态原始数据。是__getitem__的核心实现。
        自动加载并缓存当前trajectory的parquet数据（通过curr_traj_data）。

        【原理】
        1. 检查并加载trajectory的parquet数据到self.curr_traj_data（带缓存机制）
        2. 遍历所有modality_keys，为每个key调用get_data_by_modality()
        3. video模态：通过时间戳从mp4采样帧
        4. state/action模态：从parquet列切片，根据modality.json的start/end提取子向量
        5. language模态：从parquet的task_index列通过tasks表解析为字符串

        【输入输出】
        - 输入:
          * trajectory_id (int): Episode索引（如0,1,2...）
          * indices (dict[str, np.ndarray]): 每个modality_key的采样索引数组
            例如: {"video.exterior_image_1_left": array([100,101,102]), ...}
        - 输出: dict - 原始数据字典，结构：
          * "video": {"video.xxx": (T_v, H, W, C) uint8, ...}
          * "state": {"state.xxx": (T_s, D) float, ...}
          * "action": {"action.xxx": (T_a, D) float, ...}
          * "language": str 或 list[str]
          * "timestamp": (T,) float
          * "frame_index": (T,) int

        【Shape细节】
        - video每个key: (len(indices[key]), H, W, C)，C=3，H/W为原始视频分辨率
        - state/action每个key: (len(indices[key]), D)，D为该键的维度（由modality.json定义）
        - 若indices包含负值或越界，通过retrieve_data_and_pad做padding

        【调用关系】
        - 被: __getitem__()
        - 调用: get_trajectory_data()加载parquet，get_data_by_modality()分模态处理

        Args:
            trajectory_id (int): Episode索引。
            indices (dict[str, np.ndarray]): 各modality_key的采样索引。

        Returns:
            dict: 原始多模态数据（未transform）。

        Example return:
            {
                "video": {
                    "video.image_side_0": (T, H, W, C) uint8 array,
                    "video.image_side_1": (T, H, W, C) uint8 array,
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                # Only load the data if the key is in the indices
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
        return data

    def get_parquet_path(self, trajectory_id: int) -> Path:
        """Get the parquet path for a trajectory."""
        chunk_index = self.get_episode_chunk(trajectory_id)
        return self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            parquet_path = self.get_parquet_path(trajectory_id)
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.

        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the trajectory.
            padding_strategy (str): The padding strategy, either "first_last" or "zero".
                "first_last" uses first/last step data for padding, "zero" uses zero padding.

        Returns:
            np.ndarray: The retrieved and padded data.
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        """Get the video file path for a specific trajectory and video key.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The video key (without 'video.' prefix).

        Returns:
            Path: Path to the video file.
        """
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        # try:
        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        # except:
            # self.video_backend = "torchvision_av"
            # return get_frames_by_timestamps(
            #     video_path.as_posix(),
            #     video_timestamp,
            #     video_backend=self.video_backend,
            #     video_backend_kwargs=self.video_backend_kwargs,
            # )


    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        subkey = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_lapa_action(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Get LAPA action data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the LAPA action data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray | None: The LAPA action data, or None if the key is not found.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Check key in the current trajectory data
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        if (
            key not in self.curr_traj_data.columns
        ):  # this ensures that we can still load data w/o lapa actions. will store values that are None.
            return None
        # assert key in self.curr_traj_data.columns, f"{key} not found in {trajectory_id=}"
        # Get the data array, shape: (T, D)
        data_array: np.ndarray = np.stack(self.curr_traj_data[key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last",
        )

    def get_dream_actions(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Get DREAM action data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the DREAM action data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray | None: The DREAM action data, or None if the key is not found.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Check key in the current trajectory data
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        if (
            key not in self.curr_traj_data.columns
        ):  # this ensures that we can still load data w/o lapa actions. will store values that are None.
            return None
        # assert key in self.curr_traj_data.columns, f"{key} not found in {trajectory_id=}"
        # Get the data array, shape: (T, D)
        data_array: np.ndarray = np.stack(self.curr_traj_data[key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            list[str]: The annotation data for the trajectory and step indices.
                If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        # print("subkey", subkey)
        
        # Check if this is a metadata-based language key (detailed_global_instruction_medium/concise)
        if subkey in METADATA_LANG_KEYS:
            # print("return metadata language")
            return self._get_language_from_metadata(trajectory_id, subkey, len(step_indices))
        
        # Otherwise, load from parquet columns (original behavior)
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        if pd.api.types.is_numeric_dtype(self.curr_traj_data[original_key]):
            # Stored as list of integers
            task_indices: list[int] = self.curr_traj_data[original_key].iloc[step_indices].tolist()
            return self.tasks.loc[task_indices]["task"].tolist()
        else:
            # Stored as list of strings
            return self.curr_traj_data[original_key].iloc[step_indices].astype(str).tolist()

    def _get_language_from_metadata(
        self,
        trajectory_id: int,
        lang_key: str,
        nframes: int,
    ) -> list[str]:
        """Get language instruction from metadata files for special language keys.
        
        Supports:
        - detailed_global_instruction_medium: Longer, detailed description
        - detailed_global_instruction_concise: Short summary
        
        Args:
            trajectory_id (int): The ID of the trajectory (episode_index).
            lang_key (str): The language key (e.g., "detailed_global_instruction_medium").
            nframes (int): Number of frames to return the instruction for.
            
        Returns:
            list[str]: The instruction repeated for each frame (empty string if not found).
        """
        if trajectory_id in self._detailed_global_instructions:
            instruction = self._detailed_global_instructions[trajectory_id].get(lang_key, "")
            # print("instruction", instruction)
        else:
            instruction = ""
        return [instruction] * nframes

    def get_rl_info(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the reward data for a trajectory by step indices.

        If the step indices are out of range, pad with first/last step data.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the reward data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray: The reward data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        data_array: np.ndarray = np.stack(self.curr_traj_data[key])  # type: ignore

        if key == "rl_info.next.reward":
            padding_strategy = "zero"
        else:
            padding_strategy = "first_last"

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str] | None:
        """Get the data corresponding to the modality for a trajectory by step indices.

        This method dispatches to the appropriate specialized method based on the modality.
        For the language modality, empty strings are returned if no matching data is found.

        Args:
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data (video, state, action, language, etc.).
            key (str): The key of the data.
            step_indices (np.ndarray): The step indices of the trajectory.

        Returns:
            np.ndarray | list[str] | None: The data for the specified modality.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        elif modality == "lapa_action":
            return self.get_lapa_action(trajectory_id, key, step_indices)
        elif modality == "dream_actions":
            return self.get_dream_actions(trajectory_id, key, step_indices)
        elif modality == "rl_info":
            return self.get_rl_info(trajectory_id, key, step_indices)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_initial_actions(self):
        """Load initial actions from the dataset if available.

        Returns:
            list: List containing initial actions if the file exists, empty list otherwise.
        """
        initial_actions_path = self.dataset_path / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []



class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    """A cached version of LeRobotSingleDataset that preloads all video frames into memory.

    This class caches video frames for each trajectory and key to improve access speed
    when video frames need to be accessed multiple times. Recommended for small datasets
    or when memory usage is not a concern.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the cached dataset and preload all video frames.

        Args:
            *args: Arguments passed to parent LeRobotSingleDataset.
            **kwargs: Keyword arguments passed to parent LeRobotSingleDataset.
        """
        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames, _ = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                assert (
                    frames.shape[0] == trajectory_length
                ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, step_indices: np.ndarray) -> np.ndarray:
        """Get video frames from the cached data.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The video key (with 'video.' prefix).
            step_indices (np.ndarray): The step indices to retrieve frames for.

        Returns:
            np.ndarray: The video frames with shape (T, H, W, C).
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]


def safe_hash(input_tuple):
    """Generate a safe hash from an input tuple.

    Creates a deterministic hash using SHA256 and returns the lower 128 bits.
    This is used for deterministic random seed generation.

    Args:
        input_tuple: The tuple to hash.

    Returns:
        int: A 128-bit hash value.
    """
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    """Specification element for a dataset mixture defining paths and weights.

    This class validates dataset paths by embodiment tag and handles weight distribution
    across multiple dataset paths if requested.
    """

    dataset_path: dict[str, list[Path] | Path] = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )

    @field_validator("dataset_path", mode="after")
    def validate_dataset_path_keys(cls, v: dict[str, list[Path] | Path]) -> dict[str, list[Path]]:
        """Validate dataset paths and expand glob patterns.

        Args:
            v (dict[str, list[Path] | Path]): Dictionary mapping embodiment tags to paths.

        Returns:
            dict[str, list[Path]]: Validated and expanded paths.

        Raises:
            ValueError: If an invalid embodiment tag is provided.
        """
        all_globbed_paths: dict[str, list[Path]] = {}
        for embodiment_tag, paths in v.items():
            try:
                _ = EmbodimentTag(embodiment_tag)
            except ValueError:
                raise ValueError(f"Invalid embodiment tag: {embodiment_tag}")
            if isinstance(paths, Path):
                paths = [paths]
            globbed_paths = []
            for path in paths:
                globbed_paths.extend(glob.glob(str(path)))
            all_globbed_paths[embodiment_tag] = globbed_paths
        return all_globbed_paths


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        allow_padding_at_end: bool = False,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            training (bool): If True, __getitem__ will return different samples every epoch; if False, __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            allow_padding_at_end (bool): If True, allow padding at the end of the dataset.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.training = training
        self.allow_padding_at_end = allow_padding_at_end

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_ids))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= np.array(
                    [
                        len(dataset.step_filter[trajectory_id])
                        for trajectory_id in dataset.trajectory_ids
                    ]
                )

            if dataset.discard_bad_trajectories:
                bad_trajectory_indices = dataset.lerobot_info_meta.get(
                    "discarded_episode_indices", []
                )
                trajectory_sampling_weights[bad_trajectory_indices] = 0.0

            if trajectory_sampling_weights.sum() == 0:
                raise ValueError(f"No valid trajectories found for dataset {dataset}")

            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        # Create a merged metadata for the mixture dataset (we don't need this in the future as eval will directly use `get_metadata`)
        self.update_metadata(metadata_config)

        # Set the transforms to training or evaluation mode
        if self.training:
            for dataset in self.datasets:
                dataset.transforms.train()
        else:
            for dataset in self.datasets:
                dataset.transforms.eval()

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        """Return a string representation of the mixture dataset with weights."""
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump({"Mixture dataset": dataset_descriptions})

    @classmethod
    def from_mixture_spec(
        cls: type[T_LeRobotMixtureDataset],
        mixture_spec: Sequence[MixtureSpecElement | dict],
        dataset_class: type[LeRobotSingleDataset] | str,
        all_modality_configs: dict[str, dict[str, ModalityConfig]],
        all_transforms: dict[str, ComposedModalityTransform],
        metadata_versions: dict[str, str],
        fps: float = None,
        dataset_kwargs: dict | None = None,
        mixture_kwargs: dict | None = None,
    ) -> T_LeRobotMixtureDataset:
        """Initialize the mixture dataset from a specification.

        Args:
            mixture_spec (Sequence[MixtureSpecElement | dict]): The specification for the mixture dataset.
            dataset_class (type[LeRobotSingleDataset] | str): The dataset class or its string path.
            all_modality_configs (dict[str, dict[str, ModalityConfig]]): The modality configs for each embodiment.
            all_transforms (dict[str, ComposedModalityTransform]): The transforms for each embodiment.
            metadata_versions (dict[str, str]): The metadata versions for each embodiment.
            dataset_kwargs (dict | None): Additional keyword arguments for the dataset classes.
            mixture_kwargs (dict | None): Additional keyword arguments for the mixture dataset.

        Returns:
            LeRobotMixtureDataset: The initialized mixture dataset.
        """
        if isinstance(dataset_class, str):
            module_name, class_name = dataset_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_class = getattr(module, class_name)
        assert not isinstance(dataset_class, str), f"{dataset_class} is a string"
        assert issubclass(
            dataset_class, LeRobotSingleDataset
        ), f"{dataset_class} is not a subclass of LeRobotSingleDataset"
        data_mixture = []

        for dataset_spec in tqdm(
            mixture_spec,
            total=len(mixture_spec),
            desc="Initializing datasets",
        ):
            start_time = time.time()
            if isinstance(dataset_spec, dict):
                dataset_spec = MixtureSpecElement.model_validate(dataset_spec)
            datasets = []
            for embodiment_tag, paths in dataset_spec.dataset_path.items():
                if isinstance(paths, Path):
                    paths = [paths]
                for dataset_path in paths:
                    if '.sh' in dataset_path or '.json' in dataset_path:
                        continue
                    assert (
                        embodiment_tag in all_modality_configs
                    ), f"{embodiment_tag} not in modality_configs: {all_modality_configs.keys()}"
                    assert (
                        embodiment_tag in all_transforms
                    ), f"{embodiment_tag} not in transforms: {all_transforms.keys()}"
                    dataset = dataset_class(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=copy.copy(all_modality_configs[embodiment_tag]),
                        transforms=copy.copy(all_transforms[embodiment_tag]),
                        metadata_version=metadata_versions[embodiment_tag],
                        fps=fps[embodiment_tag] if embodiment_tag in fps else None,
                        **(dataset_kwargs if dataset_kwargs is not None else {}),
                    )
                    datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                if dataset_spec.distribute_weights:
                    weight = relative_length * dataset_spec.dataset_weight
                else:
                    weight = dataset_spec.dataset_weight
                data_mixture.append((dataset, weight))

            print(
                f"Time taken to initialize {len(datasets)} datasets: {time.time() - start_time:.2f} seconds"
            )

        return cls(
            data_mixture=data_mixture,
            **(mixture_kwargs if mixture_kwargs is not None else {}),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the mixture dataset.

        Args:
            index (int): The index to sample (used for deterministic sampling).

        Returns:
            tuple[LeRobotSingleDataset, int, int]: A tuple of (dataset, trajectory_id, step_index).
        """
        # return self.sampled_steps[index]

        # Set seed
        if self.training:
            seed = safe_hash((self.epoch, index, self.seed))
            rng = np.random.default_rng(seed)

            # Sample dataset
            dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
            dataset = self.datasets[dataset_index]

            if self.allow_padding_at_end:
                # Sample trajectory
                trajectory_index = rng.choice(
                    len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
                )
                trajectory_id = dataset.trajectory_ids[trajectory_index]

                allowed_length = dataset.trajectory_lengths[trajectory_index]
            else:
                # Avoid padding at the end of the trajectory
                max_delta_index = dataset.max_delta_index
                trajectory_length = 0
                trajectory_id = None
                while trajectory_length < max_delta_index + 1:
                    # Sample trajectory
                    trajectory_index = rng.choice(
                        len(dataset.trajectory_ids),
                        p=self.trajectory_sampling_weights[dataset_index],
                    )
                    trajectory_id = dataset.trajectory_ids[trajectory_index]
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                assert trajectory_id is not None

                # Sample step
                assert (
                    trajectory_length >= max_delta_index + 1
                ), f"{trajectory_length=}, {max_delta_index=}"
                allowed_length = trajectory_length - max_delta_index
            # Get the allowed indices from the step filter
            allowed_indices = dataset.step_filter[trajectory_id]
            # Remove indices that are too large
            allowed_indices = allowed_indices[allowed_indices <= allowed_length]
            step_index = rng.choice(allowed_indices)
            return dataset, trajectory_id, step_index
        else:
            length_cumsum = np.cumsum(self.dataset_lengths)
            dataset_index = np.searchsorted(length_cumsum, index)
            dataset = self.datasets[dataset_index]
            assert (
                len(dataset._lerobot_info_meta.get("discarded_episode_indices", [])) == 0
            ), f"Find discarded episode indices in evaluation dataset {dataset.dataset_path}"
            trajectory_id, step_index = dataset.all_steps[index - length_cumsum[dataset_index]]
            return dataset, trajectory_id, step_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        dataset, trajectory_id, step_index = self.sample_step(index)
        indices = {
            key: delta_indices + step_index for key, delta_indices in dataset.delta_indices.items()
        }
        return dataset.transforms(dataset.get_step_data(trajectory_id, indices))

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if self.training:
            return int((self.dataset_lengths * self.dataset_sampling_weights).sum())
        else:
            return int(self.dataset_lengths.sum())

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Check if stats are per-horizon (2D) by examining the first task's mean
            first_mean = np.array(per_task_stats[0][modality]["mean"])
            is_per_horizon = first_mean.ndim == 2  # Shape (horizon_len, action_dim)
            
            if is_per_horizon:
                # Handle per-horizon stats (2D arrays)
                stats_shape = first_mean.shape  # (horizon_len, action_dim)
                
                # Initialize accumulators for means and variances
                weighted_means = np.zeros(stats_shape)
                weighted_squares = np.zeros(stats_shape)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(np.array(stats["min"]))
                    max_list.append(np.array(stats["max"]))
                    q01_list.append(np.array(stats["q01"]))
                    q99_list.append(np.array(stats["q99"]))

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                # Stack along new axis: (num_tasks, horizon_len, action_dim)
                overall_min = np.min(np.stack(min_list, axis=0), axis=0).tolist()
                overall_max = np.max(np.stack(max_list, axis=0), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                q01_array = np.stack(q01_list, axis=0)  # (num_tasks, horizon_len, action_dim)
                q99_array = np.stack(q99_list, axis=0)
                if percentile_mixing_method == "weighted_average":
                    # Weighted average along task axis
                    weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                    weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")
            else:
                # Handle regular stats (1D arrays)
                num_dims = len(first_mean)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(num_dims)
                weighted_squares = np.zeros(num_dims)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(stats["min"])
                    max_list.append(stats["max"])
                    q01_list.append(stats["q01"])
                    q99_list.append(stats["q99"])

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                overall_min = np.min(np.array(min_list), axis=0).tolist()
                overall_max = np.max(np.array(max_list), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                # Use weighted average of per-task quantiles
                q01_array = np.array(q01_list)
                q99_array = np.array(q99_list)
                if percentile_mixing_method == "weighted_average":
                    weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                    weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag.value not in all_metadatas:
                all_metadatas[dataset.tag.value] = []
            all_metadatas[dataset.tag.value].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag.value])

    def get_initial_actions(self):
        initial_actions = []
        for dataset in self.datasets:
            if hasattr(dataset, "get_initial_actions"):
                initial_actions.extend(dataset.get_initial_actions())
        return initial_actions
