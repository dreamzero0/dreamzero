from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import yaml

from groot.vla.common.utils import get_frames_by_timestamps

from .lerobot import LE_ROBOT_EPISODE_FILENAME, LeRobotMixtureDataset, LeRobotSingleDataset


class ShardedLeRobotSingleDataset(LeRobotSingleDataset):
    """
    A single dataset with shards.
    """

    def __init__(
        self,
        *args,
        num_steps_per_shard: int = int(1e4),
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.num_steps_per_shard = num_steps_per_shard
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()
        self.frames_to_load = self.get_all_frames_to_load()

        # Set shard caching properties
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self.frame_indices_map: dict[int, dict[str, np.ndarray]] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.sharded_trajectories)

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """Get the video paths for all trajectories and all views.

        Returns:
            dict[int, dict[str, Path]]: The video paths for all trajectories.
        """
        video_paths = {}
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        """Get the parquet paths for all trajectories.

        Returns:
            dict[int, Path]: The parquet paths for all trajectories.
        """
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self.trajectory_ids
        }

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        """Generate shards of trajectories. We recommend num_steps_per_shard >> average trajectory length.

        Args:
            num_steps_per_shard (int): The number of steps per shard.

        Returns:
            list[list[str]]: The shards of trajectories.
        """
        sharded_trajectories = [[]]
        curr_num_steps = 0
        curr_shard_index = 0
        discarded_episode_indices = []
        trajectory_ids = self.trajectory_ids
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get("discarded_episode_indices", [])
            trajectory_ids = [
                trajectory_id
                for trajectory_id in trajectory_ids
                if trajectory_id not in discarded_episode_indices
            ]

        assert (
            len(trajectory_ids) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"
        total_steps = np.sum(
            [len(self.step_filter[trajectory_id]) for trajectory_id in trajectory_ids]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.num_steps_per_shard).astype(int)
        cutoffs = np.linspace(0, total_steps, num_shards + 1)[1:]  # Exclude the first cutoff (0)
        shard_lengths = []
        last_num_steps = 0
        for trajectory_id in trajectory_ids:
            sharded_trajectories[-1].append(trajectory_id)
            curr_num_steps += len(self.step_filter[trajectory_id])
            if curr_num_steps > cutoffs[curr_shard_index]:
                sharded_trajectories.append([])
                curr_shard_index += 1
                shard_lengths.append(curr_num_steps - last_num_steps)
                last_num_steps = curr_num_steps
        shard_lengths.append(curr_num_steps - last_num_steps)
        assert (
            curr_num_steps == total_steps
        ), "Total steps not equal to the sum of trajectory lengths"
        assert (
            len(shard_lengths) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        assert (
            len(sharded_trajectories) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        print(f"Generated {len(sharded_trajectories)} shards for dataset {self.dataset_path}")
        return sharded_trajectories, np.array(shard_lengths)

    def get_all_frames_to_load(self):
        """Generate a map of video frame indices to trajectory indices."""
        all_frames_to_load = {}
        for trajectory_id in self.trajectory_ids:
            all_frames_to_load[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                filtered_indices = self.step_filter[trajectory_id]
                if len(filtered_indices) > 0:
                    frames_to_load = np.unique(
                        np.concatenate(
                            [i + np.array(self.delta_indices[key]) for i in filtered_indices]
                        )
                    )
                    # Cap within the length of the trajectory and >= 0
                    frames_to_load = frames_to_load[
                        (frames_to_load < self.trajectory_lengths[trajectory_id])
                        & (frames_to_load >= 0)
                    ]
                else:
                    frames_to_load = np.array([])
                all_frames_to_load[trajectory_id][key] = frames_to_load
        return all_frames_to_load

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        frames_to_load: dict[int, dict[str, np.ndarray]],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
    ) -> tuple[
        dict[str, np.ndarray], dict[int, int], pd.DataFrame, dict[int, dict[str, np.ndarray]]
    ]:
        print("Caching shard")
        start_time = time.time()
        assert "video" in modality_keys, "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        frame_indices_map = {}
        curr_step_index = 0
        cached_df = None
        curr_frame_index = {key: 0 for key in modality_keys["video"]}
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            # Check timestamps are in sync
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            frame_indices_map[trajectory_id] = {}
            for key in modality_keys["video"]:
                # Only load the frames that are needed
                this_frames_to_load = frames_to_load[trajectory_id][key]
                if len(this_frames_to_load) == 0:
                    continue
                load_timestamps = parquet_timestamps[this_frames_to_load]
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                # Store a mapping that frame_indices_map[trajectory_id][key][frame_index] = index_in_concat_video_frames
                frame_indices_map[trajectory_id][key] = (
                    np.ones(len(parquet_timestamps), dtype=np.int32) * -1
                )
                frame_indices_map[trajectory_id][key][this_frames_to_load] = np.arange(
                    curr_frame_index[key],
                    curr_frame_index[key] + len(this_frames_to_load),
                    dtype=np.int32,
                )
                curr_frame_index[key] += len(this_frames_to_load)
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                video_paths[trajectory_id][key].as_posix(),
                    timestamps=load_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs or {},
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        # Concatenate the frames
        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        print(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        # Add global "index" column if missing (some dataset formats omit it)
        if "index" not in cached_df.columns:
            cached_df = cached_df.reset_index(drop=True)
            cached_df["index"] = cached_df.index
        return cached_frames, trajectory_start_indices, cached_df, frame_indices_map

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.frames_to_load,
            self.video_backend,
            self.video_backend_kwargs,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df, self.frame_indices_map = (
            self._cache_job.result()
        )
        self._cache_job = None  # Clear the future to allow memory to be freed

    def delete_cached_shard(self):
        """Delete the cached shard."""
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_video(self, trajectory_id: int, key: str, step_indices: np.ndarray) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by a base index.

        Args:
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        # Calculate the absolute indices
        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
            and self.frame_indices_map is not None
            and trajectory_id in self.frame_indices_map
            and key in self.frame_indices_map[trajectory_id]
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.frame_indices_map[trajectory_id][key][step_indices]
        assert np.all(
            indices_in_shard != -1
        ), f"Indices in shard are not loaded for {trajectory_id=}, {key=}, {step_indices=}"
        return self.cached_shard[key][indices_in_shard]

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the trajectory data."""
        assert self.cached_df is not None, "Cached dataframe is None"
        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} {self.args} {self.kwargs}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        return traj_data


class ShardedLeRobotSubLangSingleActionChunkDatasetDROID(LeRobotSingleDataset):
    """
    DROID专用分块单数据集类：支持shard缓存和预加载，优化大规模数据集训练效率。

    【作用与原理】
    继承LeRobotSingleDataset，增加了分片（shard）机制，用于高效处理大规模DROID数据集：
    1. 将trajectory按步数切分为多个shard（默认约10000步/shard）
    2. 支持shard级缓存：预加载整个shard的parquet和视频数据到内存
    3. 使用ThreadPoolExecutor预加载下一个shard，实现I/O与计算重叠
    4. 专为IterableDataset设计，配合ShardedLeRobotMixtureDataset使用

    【数据流位置】
    上游：磁盘LeRobot数据集
    下游：ShardedLeRobotMixtureDataset（多数据集混合采样）→ DreamTransform → Collator

    【核心机制】
    - Shard生成：按num_steps_per_shard将trajectory分组，确保每个shard总步数≈目标值
    - 缓存策略：维护当前shard（cached_shard/cached_df）和预加载任务（_cache_job）
    - 路径预计算：初始化时预计算所有video/parquet路径，避免运行时反复查找

    【适用场景】
    大规模DROID数据集（数十万episode），需要：
    - 流式读取避免全量加载内存
    - 顺序访问模式下通过shard缓存提高I/O效率
    - 多数据集混合训练（通过ShardedLeRobotMixtureDataset）
    """

    def __init__(
        self,
        *args,
        num_steps_per_shard: int = int(1e4),
        **kwargs,
    ):
        """
        初始化DROID分块数据集。

        Args:
            *args: 传递给LeRobotSingleDataset的位置参数。
            num_steps_per_shard (int): 每个shard的目标步数（默认10000）。
                实际shard大小可能略有偏差，以完整trajectory为单位。
            **kwargs: 传递给LeRobotSingleDataset的关键字参数。

        【初始化流程】
        1. 调用父类LeRobotSingleDataset.__init__()完成基础加载
        2. 预计算all_video_paths和all_parquet_paths（所有trajectory的路径映射）
        3. 调用generate_shards()生成分片方案
        4. 初始化shard缓存相关属性（shard_start_indices, cached_shard, cached_df等）
        5. 创建ThreadPoolExecutor用于异步预加载
        """
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.num_steps_per_shard = num_steps_per_shard
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()

        # Set shard caching properties
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None
        # self._traj_cache: dict[int, pd.DataFrame] = {}
        # # Precompute language key once to avoid repeated scans
        # self.language_key: str | None = next(
        #     (
        #         modality_key
        #         for modality_name in self.modality_keys
        #         for modality_key in self.modality_keys[modality_name]
        #         if modality_key.startswith("annotation.")
        #     ),
        #     None,
        # )
        # # Track current sample's chunk count for alignment across modalities
        # self._current_num_chunks: dict[int, int] = {}

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.sharded_trajectories)

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """Get the video paths for all trajectories and all views.

        Returns:
            dict[int, dict[str, Path]]: The video paths for all trajectories.
        """
        video_paths = {}
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        """Get the parquet paths for all trajectories.

        Returns:
            dict[int, Path]: The parquet paths for all trajectories.
        """
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self.trajectory_ids
        }

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        """Generate shards of trajectories. We recommend num_steps_per_shard >> average trajectory length.

        Args:
            num_steps_per_shard (int): The number of steps per shard.

        Returns:
            list[list[str]]: The shards of trajectories.
        """
        sharded_trajectories = [[]]
        curr_num_steps = 0
        curr_shard_index = 0
        discarded_episode_indices = []
        trajectory_ids = self.trajectory_ids
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get("discarded_episode_indices", [])
            trajectory_ids = [
                trajectory_id
                for trajectory_id in trajectory_ids
                if trajectory_id not in discarded_episode_indices
            ]

        assert len(trajectory_ids) > 0, "No valid trajectories found for dataset"
        total_steps = np.sum(
            [len(self.step_filter[trajectory_id]) for trajectory_id in trajectory_ids]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.num_steps_per_shard).astype(int)
        cutoffs = np.linspace(0, total_steps, num_shards + 1)[1:]  # Exclude the first cutoff (0)
        shard_lengths = []
        last_num_steps = 0
        for trajectory_id in trajectory_ids:
            sharded_trajectories[-1].append(trajectory_id)
            curr_num_steps += len(self.step_filter[trajectory_id])
            if curr_num_steps > cutoffs[curr_shard_index]:
                sharded_trajectories.append([])
                curr_shard_index += 1
                shard_lengths.append(curr_num_steps - last_num_steps)
                last_num_steps = curr_num_steps
        shard_lengths.append(curr_num_steps - last_num_steps)
        assert (
            curr_num_steps == total_steps
        ), "Total steps not equal to the sum of trajectory lengths"
        assert (
            len(shard_lengths) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        assert (
            len(sharded_trajectories) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        print(f"Generated {len(sharded_trajectories)} shards for dataset {self.dataset_path}")
        return sharded_trajectories, np.array(shard_lengths)

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
        fps: float = None,
    ) -> tuple[dict[str, np.ndarray], dict[int, int], pd.DataFrame]:
        # Optional logging to avoid stdout overhead during tight loops
        # (controlled by instance-level verbose flag)
        # Using a staticmethod, we cannot read self.verbose; defer to caller to control prints
        print("Caching shard")
        start_time = time.time()
        assert "video" in modality_keys, "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        curr_step_index = 0
        cached_df = None
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            # Check timestamps are in sync
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            for key in modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                    video_paths[trajectory_id][key].as_posix(),
                    timestamps=parquet_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs,
                    fps=fps,
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        # Concatenate the frames
        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        print(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        # Add global "index" column if missing (some dataset formats omit it)
        if "index" not in cached_df.columns:
            cached_df = cached_df.reset_index(drop=True)
            cached_df["index"] = cached_df.index
        return cached_frames, trajectory_start_indices, cached_df

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.video_backend,
            self.video_backend_kwargs,
            self.fps,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df = self._cache_job.result()
        self._cache_job = None  # Clear the future to allow memory to be freed

    def delete_cached_shard(self):
        """Delete the cached shard."""
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df
        # self._traj_cache.clear()

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            indices (dict[str, np.ndarray]): The indices for each modality.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
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
                    # Skip this sample if state or action data is empty
                    if data[key] is not None and hasattr(data[key], '__len__') and len(data[key]) == 0:
                        return None
        return data

    def get_video(self, trajectory_id: int, key: str, step_indices: np.ndarray) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by uniformly sampling from language-consistent ranges.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the video.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        
        # Get trajectory data to access language annotations (reuse if already loaded)
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        traj_data = self.get_trajectory_data(trajectory_id)
        # print("trajectory id", trajectory_id, step_indices, trajectory_index)
        
        # Get language annotations for all steps in the trajectory
        # language_key = self.language_key
        for modality in self.modality_keys:
            for modality_key in self.modality_keys[modality]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    subkey_meta = annotation_meta[subkey]
                    language_key = subkey_meta.original_key
                    break
        assert language_key is not None, "Language key not found"
        if language_key in traj_data.columns:
            language_annotations = traj_data[language_key].values
        else:
            # Fallback to original behavior if language annotations are not available
            step_indices = np.maximum(step_indices, 0)
            step_indices = np.minimum(step_indices, trajectory_length - 1)
            assert (
                self.shard_start_indices is not None
                and self.cached_shard is not None
                and trajectory_id in self.shard_start_indices
            ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
            indices_in_shard = self.shard_start_indices[trajectory_id] + step_indices
            return self.cached_shard[key][indices_in_shard]
        
        # Find language-consistent ranges and uniformly sample from them
        sampled_indices = self._uniform_sample_from_language_ranges(
            step_indices, language_annotations, trajectory_length
        )
        
        # Ensure the sampled indices are within the valid range
        sampled_indices = np.maximum(sampled_indices, 0)
        sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)
        # print("sampled indices", sampled_indices)
        
        # Calculate the absolute indices
        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.shard_start_indices[trajectory_id] + sampled_indices
        return self.cached_shard[key][indices_in_shard]

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
        elif modality == "state":
            return self.get_state(trajectory_id, modality, key, step_indices)
        elif modality == "action":
            return self.get_action(trajectory_id, modality, key, step_indices)
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


    def get_state(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the state data for a trajectory by a base index.
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

        # Build sampled indices for state aligned with language and video sampling
        # For state, select only the anchor index per 30-frame chunk (stride 30):
        # [..., first_idx-30, first_idx, first_idx+30, ...]
        # Stop on language change at the step anchor, bounds, or when reaching 16 anchors (to match 16 chunks).
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        # language_key = self.language_key
        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = None
        for modality_name in self.modality_keys:
            for modality_key in self.modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    subkey_meta = annotation_meta[subkey]
                    language_key = subkey_meta.original_key
                    break
        if language_key is not None and language_key in traj_data.columns and len(step_indices) > 0:
            language_annotations = traj_data[language_key].values
            first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
            target_language = language_annotations[first_idx]
            
            # Get the number of chunks from video sampling to ensure alignment
            target_num_chunks = None
            # if first_idx in self._current_num_chunks:
            if hasattr(self, '_current_num_chunks') and first_idx in self._current_num_chunks:
                target_num_chunks = self._current_num_chunks[first_idx]
                # print(f"State: Using target_num_chunks from video: {target_num_chunks}")
            
            max_frames = self.max_chunk_size  # 16 anchors to align with 16 chunks as video/action
            sampled_list: list[int] = []

            def add_anchor(anchor_index: int) -> None:
                nonlocal sampled_list
                if len(sampled_list) >= max_frames:
                    return
                # If we have a target number of chunks, stop when we reach it
                if target_num_chunks is not None and len(sampled_list) >= target_num_chunks:
                    return
                # Require full 32-length window to exist for alignment with action/video
                if 0 <= anchor_index and anchor_index + 24 < trajectory_length:
                    sampled_list.append(int(anchor_index))

            # Always include first_idx anchor
            add_anchor(first_idx)

            # Expand outward in 32-frame steps
            step = 1
            back_done = False
            fwd_done = False
            while len(sampled_list) < max_frames and (not back_done or not fwd_done):
                # Stop if we've reached the target number of chunks
                if target_num_chunks is not None and len(sampled_list) >= target_num_chunks:
                    break
                    
                if not back_done:
                    back_anchor = first_idx - 24 * step
                    if back_anchor < 0:
                        back_done = True
                    elif language_annotations[back_anchor] != target_language:
                        back_done = True
                    else:
                        add_anchor(back_anchor)
                if len(sampled_list) >= max_frames:
                    break
                if not fwd_done:
                    fwd_anchor = first_idx + 24 * step
                    if fwd_anchor >= trajectory_length:
                        fwd_done = True
                    elif language_annotations[fwd_anchor] != target_language:
                        fwd_done = True
                    else:
                        add_anchor(fwd_anchor)
                step += 1

            if len(sampled_list) > 0:
                sampled_indices = np.array(sorted(set(sampled_list)), dtype=int)
            else:
                sampled_indices = np.array([], dtype=int)
        else:
            # Fallback: use provided indices with bounds
            sampled_indices = np.maximum(step_indices, 0)
            sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)
        
        # print("sampled indices for state", sampled_indices)

        # Pad the data using the computed sampled indices
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the action data for a trajectory by a base index.
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


        # Build sampled indices for action aligned with language and video sampling
        # Action runs at 30fps, so for each ±30-frame step around first_idx,
        # collect a 30-length chunk with stride 1: [anchor ... anchor+29].
        # Stop on language change at the step anchor, bounds, or when reaching 480 frames (16 chunks * 30).
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        # language_key = self.language_key
        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = None
        for modality_name in self.modality_keys:
            for modality_key in self.modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    subkey_meta = annotation_meta[subkey]
                    language_key = subkey_meta.original_key
                    break
        if language_key is not None and language_key in traj_data.columns and len(step_indices) > 0:
            language_annotations = traj_data[language_key].values
            first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
            target_language = language_annotations[first_idx]
            
            # Get the number of chunks from video sampling to ensure alignment
            target_num_chunks = None
            # if first_idx in self._current_num_chunks:
            if hasattr(self, '_current_num_chunks') and first_idx in self._current_num_chunks:
                target_num_chunks = self._current_num_chunks[first_idx]
                # print(f"Using target_num_chunks from video: {target_num_chunks}")
            
            max_frames = 24 * self.max_chunk_size
            per_step_offsets = list(range(24))  # 0..23
            sampled_list: list[int] = []

            def add_step_set(anchor_index: int) -> None:
                nonlocal sampled_list
                # Ensure the whole 32-length chunk fits within bounds
                if anchor_index < 0 or anchor_index + 24 >= trajectory_length:
                    return
                # Ensure we don't overrun the max_frames cap with a partial chunk
                if len(sampled_list) + 24 > max_frames:
                    return
                # If we have a target number of chunks, stop when we reach it
                if target_num_chunks is not None and len(sampled_list) // 24 >= target_num_chunks:
                    return
                for offset in per_step_offsets:
                    idx = anchor_index + offset
                    sampled_list.append(int(idx))

            # Always include first_idx chunk
            add_step_set(first_idx)

                # Expand outward in 32-frame steps
            step = 1
            back_done = False
            fwd_done = False
            while len(sampled_list) < max_frames and (not back_done or not fwd_done):
                # Stop if we've reached the target number of chunks
                if target_num_chunks is not None and len(sampled_list) // 24 >= target_num_chunks:
                    break
                    
                if not back_done:
                    back_anchor = first_idx - 24 * step
                    if back_anchor < 0:
                        back_done = True
                    elif language_annotations[back_anchor] != target_language:
                        back_done = True
                    else:
                        add_step_set(back_anchor)
                if len(sampled_list) >= max_frames:
                    break
                if not fwd_done:
                    fwd_anchor = first_idx + 24 * step
                    if fwd_anchor >= trajectory_length:
                        fwd_done = True
                    elif language_annotations[fwd_anchor] != target_language:
                        fwd_done = True
                    else:
                        add_step_set(fwd_anchor)
                step += 1

            if len(sampled_list) > 0:
                unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
                # Enforce divisibility by 30 and the 480 cap
                capped_size = min(unique_sorted.size, max_frames)
                divisible_size = (capped_size // 24) * 24
                sampled_indices = unique_sorted[:divisible_size]
            else:
                sampled_indices = np.array([], dtype=int)
        else:
            # Fallback: use provided indices with bounds
            sampled_indices = np.maximum(step_indices, 0)
            sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)
        
        # print("sampled indices for action", first_idx, sampled_indices, trajectory_length)

        # Pad the data using the computed sampled indices
        action_data = self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )
        # print("action data before convert", key)
        # Calculate relative action on the fly if relative_action is enabled
        # Only apply to keys that are in relative_action_keys
        subkey = key.replace("action.", "")
        should_convert_to_relative = (
            (self.relative_action or self.relative_action_per_horizon)  
            and len(sampled_indices) > 0
            and (self.relative_action_keys is None or subkey in self.relative_action_keys)
        )
        if should_convert_to_relative:
            # print("action data before convert", action_data[0], action_data[-1], key)
            action_data = self._convert_to_relative_action(
                action_data=action_data,
                action_key=key,
                sampled_indices=sampled_indices,
                trajectory_id=trajectory_id,
                chunk_size=24,
            )
            # print("action data after convert", action_data[0], action_data[-1], key)
        
        return action_data
    
    def _convert_to_relative_action(
        self,
        action_data: np.ndarray,
        action_key: str,
        sampled_indices: np.ndarray,
        trajectory_id: int,
        chunk_size: int = 24,
    ) -> np.ndarray:
        """Convert absolute action to relative action by subtracting reference state.
        
        Args:
            action_data: Absolute action data, shape (T, D)
            action_key: The action key (e.g., 'action.left_arm_joints')
            sampled_indices: The sampled indices for the action
            trajectory_id: The trajectory ID
            chunk_size: Size of each action chunk (default 24)
            
        Returns:
            np.ndarray: Relative action data, shape (T, D)
        """
        # Get corresponding state key (assume state key matches action key)
        state_key = action_key.replace("action.", "state.")
        subkey = action_key.replace("action.", "")
        
        # Get state data from trajectory
        traj_data = self.get_trajectory_data(trajectory_id)
        le_state_cfg = getattr(self.lerobot_modality_meta, "state", None)
        
        if le_state_cfg is None or subkey not in le_state_cfg:
            # If no corresponding state key, return original action data
            return action_data
        
        le_state_key = le_state_cfg[subkey].original_key
        if le_state_key is None:
            le_state_key = subkey
        
        if le_state_key not in traj_data.columns:
            # If state column doesn't exist, return original action data
            return action_data
        
        # Get state data array
        state_array: np.ndarray = np.stack(traj_data[le_state_key])
        if state_array.ndim == 1:
            state_array = state_array.reshape(-1, 1)
        
        # Apply same indices as action
        le_indices = np.arange(
            le_state_cfg[subkey].start,
            le_state_cfg[subkey].end,
        )
        state_array = state_array[:, le_indices]
        
        # Calculate relative action for each chunk
        relative_action_data = action_data.copy()
        num_chunks = len(sampled_indices) // chunk_size
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size
            
            # Get anchor index (first index of the chunk)
            anchor_idx = sampled_indices[chunk_start]
            
            # Get reference state at anchor index
            if anchor_idx < len(state_array):
                reference_state = state_array[anchor_idx]
                
                # Subtract reference state from all actions in this chunk
                relative_action_data[chunk_start:chunk_end] = (
                    action_data[chunk_start:chunk_end] - reference_state
                )
        
        return relative_action_data
    
    def _uniform_sample_from_language_ranges(
        self, 
        step_indices: np.ndarray, 
        language_annotations: np.ndarray, 
        trajectory_length: int
    ) -> np.ndarray:
        """Uniformly sample from language-consistent ranges based on the first index's language.
        
        Args:
            step_indices (np.ndarray): Original step indices to sample.
            language_annotations (np.ndarray): Language annotations for each step in the trajectory.
            trajectory_length (int): Total length of the trajectory.
            
        Returns:
            np.ndarray: New indices sampled uniformly from the language-consistent range of the first index.
        """
        if len(step_indices) == 0:
            return np.array([])
        
        # Use only the first index to determine the target language
        first_idx = max(0, min(step_indices[0], trajectory_length - 1))
        target_language = language_annotations[first_idx]
        
        # Build sampled indices by moving in ±32-frame steps from first_idx
        # and adding 4 frames at 8-frame strides for each step, while:
        # - staying within trajectory bounds,
        # - keeping language consistent with target_language at the anchor step,
        # - and limiting the total collected frames to 81.
        max_frames = 8 * self.max_chunk_size + 1
        per_step_offsets = [0, 3, 6, 9, 12, 15, 18, 21]
        sampled_list: list[int] = []
        
        def add_step_set(anchor_index: int) -> None:
            # Only add a complete 4-frame set if it fully fits and capacity allows
            # Require full 32-frame window to exist for alignment with action/state
            nonlocal sampled_list
            if anchor_index < 0 or anchor_index + 23 >= trajectory_length:
                return
            if len(sampled_list) + len(per_step_offsets) > max_frames:
                return
            for offset in per_step_offsets:
                idx = anchor_index + offset
                sampled_list.append(int(idx))
        
        # Always include the set at the first_idx
        add_step_set(first_idx)
        
        # Expand outward in both directions in 32-frame steps
        step = 1
        back_done = False
        fwd_done = False
        while len(sampled_list) < max_frames and (not back_done or not fwd_done):
            # Backward step
            if not back_done:
                back_anchor = first_idx - 24 * step
                if back_anchor < 0:
                    back_done = True
                elif language_annotations[back_anchor] != target_language:
                    back_done = True
                else:
                    add_step_set(back_anchor)
            # Forward step
            if len(sampled_list) >= max_frames:
                break
            if not fwd_done:
                fwd_anchor = first_idx + 24 * step
                if fwd_anchor >= trajectory_length:
                    fwd_done = True
                elif language_annotations[fwd_anchor] != target_language:
                    fwd_done = True
                else:
                    add_step_set(fwd_anchor)
            step += 1
        
        # De-duplicate and sort ascending for stable ordering
        if len(sampled_list) == 0:
            return np.array([])
        unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
        # Ensure we return at most 81 frames
        if unique_sorted.size > max_frames:
            unique_sorted = unique_sorted[:max_frames]
        
        # Convert to 4n+1 format by adding one more frame at the end with 8-frame stride
        if unique_sorted.size > 0:
            # Get the last index and add one more frame with 8-frame stride
            last_idx = unique_sorted[-1]
            additional_idx = last_idx + 3
            
            # Only add if it doesn't exceed trajectory bounds and max_frames
            if additional_idx < trajectory_length and unique_sorted.size < max_frames:
                unique_sorted = np.append(unique_sorted, additional_idx)
            else: 
                # print("additional_idx", additional_idx, trajectory_length, unique_sorted.size, max_frames)
                unique_sorted = unique_sorted[:-7]
        
        # ensure that unique_sorted has 4n+1 frames
        assert unique_sorted.size % 8 == 1, f"unique_sorted size {unique_sorted.size} is not 4n+1"
        
        # Store the number of chunks for alignment with action/state
        num_video_chunks = (unique_sorted.size - 1) // 8
        if not hasattr(self, '_current_num_chunks'):
            self._current_num_chunks = {}
        # Use first_idx as a key to track the current sample's chunk count
        self._current_num_chunks[first_idx] = num_video_chunks
        
        # print("unique_sorted size", unique_sorted.size, "num_video_chunks", num_video_chunks)
        return unique_sorted


    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the trajectory data."""
        assert self.cached_df is not None, "Cached dataframe is None"

            # Quick verification
        if self.cached_df.empty:
            raise ValueError("cached_df is completely empty!")

        # # Fast path: return cached slice if available
        # if trajectory_id in self._traj_cache:
        #     return self._traj_cache[trajectory_id]

        available_episodes = self.cached_df["episode_index"].unique()
        if trajectory_id not in available_episodes:
            raise ValueError(
                f"trajectory_id {trajectory_id} not found in cached_df. "
                f"Available episodes: {sorted(available_episodes)}"
            )

        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} {self.args} {self.kwargs}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        # Store in cache to avoid repeated filtering on subsequent calls within a batch
        # self._traj_cache[trajectory_id] = traj_data
        return traj_data



class ShardedLeRobotMixtureDataset(LeRobotMixtureDataset, IterableDataset):
    """
    多数据集混合采样器：基于权重在多个ShardedLeRobotSingleDataset间采样，支持shard级流式迭代。

    【作用与原理】
    该类是DreamZero支持多本体（multi-embodiment）混合训练的核心组件：
    1. 维护多个子数据集（如DROID、AgiBot、YAM等）及其采样权重
    2. 在shard级别采样（而非单个样本），提高I/O效率
    3. 支持两种权重平衡策略：
       - balance_dataset_weights: 按数据集总步数加权
       - balance_trajectory_weights: 在数据集内按trajectory长度加权
    4. 作为IterableDataset，通过__iter__生成无限流式样本（配合Trainer多epoch训练）

    【数据流位置】
    上游：多个ShardedLeRobotSingleDataset（或子类）
    下游：DataLoader → DreamTransform → DefaultDataCollator → VLA模型

    【采样流程】
    1. 计算全局shard权重：合并所有数据集的shard，按(shard_length × dataset_weight)加权
    2. __iter__迭代时：
       a. 按权重随机选一个shard（含dataset_id和shard_idx）
       b. 让对应数据集加载并缓存该shard
       c. 从该shard中按shard_sampling_rate采样若干step
       d. 对每个sample调用dataset.__getitem__()获取完整数据
    3. 预加载下一个shard实现I/O overlap

    【适用场景】
    - 多本体联合训练（DROID + AgiBot + YAM等）
    - 超大规模数据集（无法全量加载内存）
    - 需要灵活调整各数据源采样比例

    【关键概念】
    - shard_sampling_rate: 每个shard实际采样的比例（0-1），用于控制数据复用程度
    - num_shards_to_sample: 每个epoch最多采样的shard数（防止无限迭代）
    - seed: 控制shard采样顺序的随机性（resume时可重置）
    """

    def __init__(
        self,
        data_mixture: list[tuple[LeRobotSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
        allow_padding_at_end: bool = False,
    ):
        """
        初始化混合数据集。

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): 
                数据集列表，每个元素为(dataset, weight)元组。
                dataset: ShardedLeRobotSingleDataset（或其子类）实例。
                weight: 该数据集的采样权重（原始权重，会被balance策略调整）。
            training (bool): 是否训练模式。训练时随机采样shard；验证时确定性采样。
            balance_dataset_weights (bool): 
                是否按数据集总步数平衡权重。若为True，dataset_weight会乘以该数据集总步数。
                这样大数据集（如DROID）会比小数据集（如YAM）采样更频繁。
            balance_trajectory_weights (bool): 
                是否在数据集内按trajectory长度加权。长trajectory被采样概率更高。
            seed (int): 随机种子，控制shard采样顺序。resume training时可重置。
            shard_sampling_rate (float): 
                每个shard中实际采样的步数比例（0-1）。例如0.1表示每个shard只采10%的步。
                用于控制数据复用和epoch概念（shard级partial sampling）。
            num_shards_to_sample (int): 
                每个epoch最多采样的shard数量（默认2^20，实际近似无限）。
                控制虚拟epoch长度，方便wandb logging。
            allow_padding_at_end (bool): 
                当shard采样导致样本不足时，是否允许用最后一个样本padding。

        【初始化流程】
        1. 调用父类LeRobotMixtureDataset.__init__()处理data_mixture和基础权重
        2. 收集所有数据集的shard信息，计算全局shard采样权重：
           weight_shard = shard_length / sum(shard_lengths) × dataset_weight_normalized
        3. 保存shard_sampling_rate和num_shards_to_sample
        4. 初始化迭代器状态（_shard_iterator）
        """
        super().__init__(
            data_mixture=data_mixture,
            training=training,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
            allow_padding_at_end=allow_padding_at_end,
        )
        # Add type hint
        self.datasets: list[ShardedLeRobotSingleDataset] = self.datasets
        # Set properties
        self.shard_sampling_rate = shard_sampling_rate
        self.num_shards_to_sample = num_shards_to_sample

        # Calculate shard sampling weights
        all_shard_sampling_weights = []
        all_shards = []
        for dataset_id, (dataset, weight) in enumerate(
            zip(self.datasets, self._dataset_sampling_weights)
        ):
            shard_sampling_weights = dataset.shard_lengths / dataset.shard_lengths.sum()
            all_shard_sampling_weights.append(shard_sampling_weights * weight)
            all_shards.extend(
                [(dataset_id, shard_idx) for shard_idx in range(shard_sampling_weights.shape[0])]
            )
        all_shard_sampling_weights = np.concatenate(all_shard_sampling_weights)
        all_shard_sampling_weights /= all_shard_sampling_weights.sum()
        self._shard_sampling_weights = all_shard_sampling_weights
        self._all_shards = all_shards

        # Generate shards sample schedule for all ranks and workers
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

        # Check shard sampling rate
        assert 0 <= shard_sampling_rate <= 1, "Shard sampling rate must be between 0 and 1"

        # Set properties for distributed training
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The dataset sampling weights."""
        return self._dataset_sampling_weights

    @property
    def shard_sampling_weights(self) -> list[np.ndarray]:
        """The weights of each shard."""
        return self._shard_sampling_weights

    @property
    def all_shards(self) -> list[tuple[int, int]]:
        """The shards to sample."""
        return self._all_shards

    @property
    def shards_sample_schedule(self) -> list[tuple[int, int]]:
        """The shards sample schedule.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        assert self._shards_sample_schedule is not None, "Shards sample schedule not set."
        return self._shards_sample_schedule

    @property
    def trajectory_sampling_weights(self):
        """The trajectory sampling weights."""
        raise ValueError("ShardedRobotMixtureDataset does not support trajectory sampling weights.")

    @property
    def primary_dataset_indices(self):
        """The primary dataset indices."""
        raise ValueError("ShardedRobotMixtureDataset does not support primary dataset indices.")

    def reset_seed(self, seed: int):
        self.seed = seed
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

    def generate_shards_sample_schedule(self):
        if self.training:
            rng = np.random.default_rng(self.seed)
            sampled_shard_ids = rng.choice(
                len(self.all_shards), size=self.num_shards_to_sample, p=self.shard_sampling_weights
            )
            shards_sample_schedule = [self.all_shards[i] for i in sampled_shard_ids]
            rng.shuffle(shards_sample_schedule)
        else:
            shards_sample_schedule = [
                self.all_shards[i % len(self.all_shards)] for i in range(self.num_shards_to_sample)
            ]
        return shards_sample_schedule

    def filter_shards_sample_schedule(self):
        """Filter the shards sample schedule for each worker.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        # Filter shards for each worker
        filtered_schedule = []
        worker_info = get_worker_info()
        # If we have multiple workers, further split shards among them
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert (
                self.worker_id == worker_id and self.num_workers == num_workers
            ), "Worker ID or number of workers has been changed since it was set. This is not allowed."

        for i, shard in enumerate(self.shards_sample_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        # print(f"Filtered shards for rank {self.rank}, worker {worker_id}: {filtered_schedule}")
        return filtered_schedule

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            shard_lengths = dataset.shard_lengths
            assert len(shard_lengths.shape) == 1, "Shard lengths must be a 1D array"
            num_shards = shard_lengths.shape[0]
            max_shard_length = int(shard_lengths.max())
            min_shard_length = int(shard_lengths.min())
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
                "Num shards": num_shards,
                "Max shard length": max_shard_length,
                "Min shard length": min_shard_length,
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump(
            {
                "Mixture dataset": dataset_descriptions,
                "Rank": self.rank,
                "World size": self.world_size,
            }
        )

    def __iter__(self):
        """Iterate over the dataset."""

        # Not supported: balance_trajectory_weights=False
        if not self.balance_trajectory_weights:
            raise NotImplementedError(
                "balance_trajectory_weights=False is not supported. Please use balance_dataset_weights=True instead."
            )

        self._shards_sample_schedule = self.filter_shards_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed)
        for i, (dataset_index, shard_index) in enumerate(self.shards_sample_schedule):
            self.curr_shard_index += 1
            assert (
                i == self.curr_shard_index
            ), f"Shard index mismatch: {i} != {self.curr_shard_index}"
            dataset = self.datasets[dataset_index]
            wait_start = time.time()
            dataset.finish_cache_shard()
            wait_end = time.time()
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )
            # Start caching the next shard immediately
            self.cache_next_shard()
            all_steps: list[tuple[int, int]] = []
            for trajectory_id in dataset.get_trajectories_in_shard():
                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                if self.allow_padding_at_end:
                    allowed_length = dataset.trajectory_lengths[trajectory_index]
                else:
                    max_delta_index = dataset.max_delta_index
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                    allowed_length = trajectory_length - max_delta_index
                # Get the allowed indices from the step filter
                allowed_indices = dataset.step_filter[trajectory_id]
                # Remove indices that are too large
                allowed_indices = allowed_indices[allowed_indices <= allowed_length]
                for i in allowed_indices:
                    all_steps.append((trajectory_id, i))
            if self.training:
                rng.shuffle(all_steps)
            sampled_steps = all_steps[: int(dataset.num_steps_per_shard * self.shard_sampling_rate)]
            for trajectory_id, step_index in sampled_steps:
                # print(
                #     f"Loading step data from rank {self.rank}, worker {self.worker_id}: {dataset_index} {trajectory_id}, {step_index}"
                # )
                indices = {
                    key: delta_indices + step_index
                    for key, delta_indices in dataset.delta_indices.items()
                }
                step_data = dataset.get_step_data(trajectory_id, indices)
                # Skip samples where state or action would be empty
                if step_data is not None:
                    yield dataset.transforms(step_data)

            # Delete the cached shard and shard start indices to free up memory
            dataset.delete_cached_shard()

    def cache_next_shard(self):
        """Cache the next shard in a background thread."""
        next_dataset_idx, next_shard_idx = self.shards_sample_schedule[self.curr_shard_index + 1]
        self.datasets[next_dataset_idx].start_cache_shard(next_shard_idx)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "__getitem__ is not supported for CachedRobotMixtureDataset. Please use __iter__ instead."
        )

    def __len__(self) -> int:
        """The length of the dataset."""
        total_length = 0
        for dataset_idx, _ in self.shards_sample_schedule:
            dataset = self.datasets[dataset_idx]
            total_length += int(dataset.num_steps_per_shard * self.shard_sampling_rate)
        return total_length
