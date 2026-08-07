#!/usr/bin/env python3
r"""Convert a dual-arm G2 LeRobot v3 dataset into DreamZero/GEAR datasets.

The converter is intentionally strict.  It validates the 16-D joint layout,
requires exactly one ``subtask_index`` per source episode, resolves that index
through ``meta/subtasks.parquet``, and writes the resolved subtask text to
``annotation.language.action_text``.  It materializes one parquet and one video
per existing source episode, forces all three camera streams to the same
CFR/resolution, writes GEAR metadata, and can physically separate a held-out
test set from the training set.

Important: the source episodes are already subtask clips.  This converter does
not split frames or videos again; it fixes the language supervision used by
DreamZero.

Expected packed state/action layout (from create_g2_dataset_using_lerobot.py):
    [left_joint_1..7, left_gripper, right_joint_1..7, right_gripper]

Example (120 episodes -> 110 train + 10 held-out test):
    python convert_lerobot_g2_to_gear.py \
      --source /data/.../g2_mock_light_module_joint_streaming \
      --output /data/.../g2_mock_light_module_gear \
      --test-episodes 10 --split-mode tail \
      --video-width 320 --video-height 176 \
      --video-codec libx264 --workers 8

Output:
    <output>/train/   # pass this directory to DreamZero training
    <output>/test/    # never read by the training job
    <output>/split_manifest.json

For a quick smoke test, add ``--max-source-episodes 2 --test-episodes 0``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq


LOG = logging.getLogger("g2-v3-to-gear")

SOURCE_CAMERAS = {
    "observation.images.head_color": "observation.images.top_head",
    "observation.images.hand_left_color": "observation.images.hand_left",
    "observation.images.hand_right_color": "observation.images.hand_right",
}
VIDEO_MODALITY_ORDER = ("top_head", "hand_left", "hand_right")

JOINT_SLICES: dict[str, tuple[int, int]] = {
    "left_joint_position": (0, 7),
    "left_gripper_position": (7, 8),
    "right_joint_position": (8, 15),
    "right_gripper_position": (15, 16),
}
EXPECTED_NAMES = [
    *(f"l.joint{i}.pos" for i in range(1, 8)),
    "l.gripper.pos",
    *(f"r.joint{i}.pos" for i in range(1, 8)),
    "r.gripper.pos",
]


@dataclass(frozen=True)
class SourceEpisode:
    source_index: int
    length: int
    task: str
    source_task: str
    subtask_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OutputEpisode:
    source: SourceEpisode
    output_index: int


def nonempty_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path cannot be empty")
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=nonempty_path, required=True)
    parser.add_argument("--output", type=nonempty_path, required=True)
    parser.add_argument("--embodiment-tag", default="g2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=176)
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "pad", "crop"),
        default="stretch",
        help="How to make all camera streams the same size.",
    )
    parser.add_argument(
        "--video-codec",
        choices=("libx264", "h264_nvenc"),
        default="libx264",
    )
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=10,
        help="Number of physically isolated held-out episodes.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("tail", "random", "stratified"),
        default="tail",
        help=(
            "tail reserves the final N source episodes; random samples globally; "
            "stratified samples evenly from every subtask."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--max-source-episodes",
        type=int,
        default=None,
        help="Use only the first N source episodes (smoke tests only).",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=None,
        help="Default is action_horizon + 1; shorter episodes fail validation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_tools(video_codec: str) -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable not found: {executable}")
    if video_codec == "h264_nvenc":
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
        if "h264_nvenc" not in completed.stdout:
            raise RuntimeError("ffmpeg does not provide the h264_nvenc encoder")


def validate_paths(source: Path, output: Path, overwrite: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or output in source.parents or source in output.parents:
        raise ValueError(f"Source and output must not overlap: {source} / {output}")
    if output == Path(output.anchor) or output == Path.home().resolve():
        raise ValueError(f"Refusing dangerous output path: {output}")
    if (output / ".git").exists():
        raise ValueError(f"Refusing to overwrite a Git repository: {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to rebuild: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files below {path}")
    return files


def read_many_parquets(files: list[Path]) -> pa.Table:
    return pads.dataset([str(path) for path in files], format="parquet").to_table()


def indexed_text_map(
    path: Path,
    index_column: str,
    text_columns: tuple[str, ...],
) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    if index_column not in table.column_names:
        raise ValueError(
            f"{path} is missing {index_column!r}; columns are {table.column_names}"
        )
    text_column = next(
        (name for name in text_columns if name in table.column_names),
        None,
    )
    if text_column is None:
        raise ValueError(
            f"{path} has no text column among {text_columns}; "
            f"columns are {table.column_names}"
        )

    result: dict[int, str] = {}
    for row in table.select([index_column, text_column]).to_pylist():
        index = int(row[index_column])
        text = str(row[text_column]).strip()
        if not text:
            raise ValueError(f"{path}: empty text for {index_column}={index}")
        previous = result.get(index)
        if previous is not None and previous != text:
            raise ValueError(
                f"{path}: conflicting text for {index_column}={index}: "
                f"{previous!r} / {text!r}"
            )
        result[index] = text
    if not result:
        raise ValueError(f"{path} contains no mappings")
    return result


def episode_subtask_indices(data_dataset: pads.Dataset) -> dict[int, int]:
    table = data_dataset.to_table(columns=["episode_index", "subtask_index"])
    episode_values = table["episode_index"].to_numpy(zero_copy_only=False)
    subtask_values = table["subtask_index"].to_numpy(zero_copy_only=False)
    seen: dict[int, int] = {}
    conflicts: dict[int, set[int]] = {}

    for episode_value, subtask_value in zip(
        episode_values, subtask_values, strict=True
    ):
        episode_index = int(episode_value)
        subtask_index = int(subtask_value)
        previous = seen.setdefault(episode_index, subtask_index)
        if previous != subtask_index:
            conflicts.setdefault(episode_index, {previous}).add(subtask_index)

    if conflicts:
        examples = {
            episode: sorted(indices)
            for episode, indices in list(sorted(conflicts.items()))[:20]
        }
        raise ValueError(
            "Every source episode must contain exactly one subtask_index; "
            f"conflicts: {examples}"
        )
    return seen


def validate_source_info(info: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}"
        )
    features = info.get("features", {})
    for key in ("observation.state", "action", *SOURCE_CAMERAS):
        if key not in features:
            raise ValueError(f"Missing source feature: {key}")
    for key in ("observation.state", "action"):
        shape = list(features[key].get("shape", []))
        if shape != [16]:
            raise ValueError(f"{key} must be 16-D joint data, got shape {shape}")
        names = features[key].get("names")
        if names and list(names) != EXPECTED_NAMES:
            raise ValueError(
                f"{key} names do not match the G2 joint layout.\n"
                f"Expected: {EXPECTED_NAMES}\nGot: {names}"
            )
    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"Invalid source fps: {fps}")


def load_source_episodes(
    source: Path, info: dict[str, Any], max_source_episodes: int | None
) -> tuple[list[SourceEpisode], list[Path], pads.Dataset]:
    data_files = parquet_files(source / "data")
    episode_files = parquet_files(source / "meta/episodes")
    tasks_path = source / "meta/tasks.parquet"
    subtasks_path = source / "meta/subtasks.parquet"

    data_dataset = pads.dataset([str(path) for path in data_files], format="parquet")
    required_columns = {
        "observation.state",
        "action",
        "episode_index",
        "subtask_index",
    }
    missing = required_columns - set(data_dataset.schema.names)
    if missing:
        raise ValueError(f"Source data is missing columns: {sorted(missing)}")

    episode_rows = read_many_parquets(episode_files).to_pylist()
    task_by_index = indexed_text_map(tasks_path, "task_index", ("task",))
    subtask_by_index = indexed_text_map(
        subtasks_path,
        "subtask_index",
        ("subtask", "task", "text"),
    )
    subtask_index_by_episode = episode_subtask_indices(data_dataset)

    episodes: list[SourceEpisode] = []
    for row in sorted(episode_rows, key=lambda item: int(item["episode_index"])):
        source_index = int(row["episode_index"])
        declared_tasks = [
            str(task).strip()
            for task in row.get("tasks", [])
            if str(task).strip()
        ]
        if len(set(declared_tasks)) > 1:
            raise ValueError(
                f"Episode {source_index} declares multiple high-level tasks: "
                f"{declared_tasks}"
            )
        source_task = declared_tasks[0] if declared_tasks else ""
        if not source_task and "task_index" in row:
            source_task = task_by_index.get(int(row["task_index"]), "")
        if not source_task:
            raise ValueError(f"Episode {source_index} has no high-level task")

        if source_index not in subtask_index_by_episode:
            raise ValueError(f"Episode {source_index} has no frame-level subtask_index")
        subtask_index = subtask_index_by_episode[source_index]
        task = subtask_by_index.get(subtask_index, "")
        if not task:
            raise ValueError(
                f"Episode {source_index}: subtask_index={subtask_index} "
                f"is absent from {subtasks_path}"
            )

        episodes.append(
            SourceEpisode(
                source_index=source_index,
                length=int(row["length"]),
                task=task,
                source_task=source_task,
                subtask_index=subtask_index,
                metadata=row,
            )
        )

    total = int(info["total_episodes"])
    if len(episodes) != total:
        raise ValueError(f"Episode metadata has {len(episodes)} rows; info.json says {total}")
    if max_source_episodes is not None:
        if max_source_episodes < 1 or max_source_episodes > len(episodes):
            raise ValueError(
                f"--max-source-episodes must be in [1, {len(episodes)}]"
            )
        episodes = episodes[:max_source_episodes]

    used_subtask_indices = {item.subtask_index for item in episodes}
    LOG.info(
        "Resolved %d source episodes to %d subtask labels "
        "(%d high-level task labels retained for provenance)",
        len(episodes),
        len(used_subtask_indices),
        len({item.source_task for item in episodes}),
    )
    return episodes, data_files, data_dataset


def split_episodes(
    episodes: list[SourceEpisode], test_count: int, mode: str, seed: int
) -> tuple[list[SourceEpisode], list[SourceEpisode]]:
    if test_count < 0 or test_count >= len(episodes):
        if test_count == 0:
            return episodes, []
        raise ValueError(f"--test-episodes must be in [0, {len(episodes) - 1}]")
    if test_count == 0:
        return episodes, []
    if mode == "tail":
        return episodes[:-test_count], episodes[-test_count:]
    rng = random.Random(seed)
    if mode == "stratified":
        by_subtask: dict[int, list[SourceEpisode]] = {}
        for item in episodes:
            by_subtask.setdefault(item.subtask_index, []).append(item)
        subtask_ids = sorted(by_subtask)
        if test_count < len(subtask_ids):
            raise ValueError(
                "Stratified split needs at least one test episode per subtask: "
                f"test_count={test_count}, subtasks={len(subtask_ids)}"
            )
        base, remainder = divmod(test_count, len(subtask_ids))
        test_ids: set[int] = set()
        for position, subtask_id in enumerate(subtask_ids):
            count = base + int(position < remainder)
            group = by_subtask[subtask_id]
            if count >= len(group):
                raise ValueError(
                    f"Subtask {subtask_id} has {len(group)} episodes; "
                    f"cannot reserve {count} and retain training data"
                )
            test_ids.update(
                item.source_index for item in rng.sample(group, count)
            )
    else:
        test_ids = set(rng.sample([item.source_index for item in episodes], test_count))
    train = [item for item in episodes if item.source_index not in test_ids]
    test = [item for item in episodes if item.source_index in test_ids]
    return train, test


def validate_episode_lengths(
    episodes: list[SourceEpisode], min_frames: int, split_name: str
) -> None:
    short = [(item.source_index, item.length) for item in episodes if item.length < min_frames]
    if short:
        raise ValueError(
            f"{split_name} has episodes shorter than {min_frames} frames: {short[:20]}"
        )


def table_for_episode(data_dataset: pads.Dataset, episode: SourceEpisode) -> pa.Table:
    columns = ["observation.state", "action", "episode_index"]
    for optional in (
        "timestamp",
        "frame_index",
        "index",
        "task_index",
        "subtask_index",
    ):
        if optional in data_dataset.schema.names:
            columns.append(optional)
    table = data_dataset.to_table(
        filter=pads.field("episode_index") == episode.source_index,
        columns=columns,
    )
    if table.num_rows != episode.length:
        raise ValueError(
            f"Episode {episode.source_index}: data has {table.num_rows} rows; "
            f"metadata says {episode.length}"
        )
    actual_subtasks = {
        int(value)
        for value in table["subtask_index"].to_pylist()
        if value is not None
    }
    if actual_subtasks != {episode.subtask_index}:
        raise ValueError(
            f"Episode {episode.source_index}: expected subtask_index "
            f"{episode.subtask_index}, got {sorted(actual_subtasks)}"
        )
    return table


def fixed_size_vectors(column: pa.ChunkedArray, name: str) -> np.ndarray:
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"{name} must have shape [N, 16], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return values


def output_table(
    source_table: pa.Table,
    episode: OutputEpisode,
    fps: float,
    task_index: int,
    global_start: int,
) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    state = fixed_size_vectors(source_table["observation.state"], "observation.state")
    action = fixed_size_vectors(source_table["action"], "action")
    length = episode.source.length
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 16)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 16)),
            "annotation.language.action_text": pa.array(
                [episode.source.task] * length, type=pa.large_string()
            ),
            "timestamp": pa.array(
                np.arange(length, dtype=np.float32) / np.float32(fps)
            ),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(
                np.full(length, episode.output_index, dtype=np.int64)
            ),
            "index": pa.array(
                np.arange(global_start, global_start + length, dtype=np.int64)
            ),
            "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
        }
    )
    return table, state, action


class StatsAccumulator:
    def __init__(self) -> None:
        self.state: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.relative: dict[str, list[np.ndarray]] = {key: [] for key in JOINT_SLICES}

    def add(self, state: np.ndarray, action: np.ndarray, horizon: int) -> None:
        self.state.append(state)
        self.action.append(action)
        usable = len(state) - horizon + 1
        if usable <= 0:
            return
        for key, (start, end) in JOINT_SLICES.items():
            reference = state[:usable, start:end]
            chunks = np.stack(
                [action[offset : offset + usable, start:end] - reference for offset in range(horizon)],
                axis=1,
            )
            self.relative[key].append(chunks.reshape(-1, end - start))


def numeric_stats(values: np.ndarray) -> dict[str, list[float]]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values64, axis=0).tolist(),
        "max": np.max(values64, axis=0).tolist(),
        "mean": np.mean(values64, axis=0).tolist(),
        "std": np.std(values64, axis=0).tolist(),
        "q01": np.quantile(values64, 0.01, axis=0).tolist(),
        "q99": np.quantile(values64, 0.99, axis=0).tolist(),
    }


def finish_stats(accumulator: StatsAccumulator) -> tuple[dict[str, Any], dict[str, Any]]:
    if not accumulator.state or not accumulator.action:
        raise ValueError("Cannot compute statistics for an empty split")
    stats = {
        "observation.state": numeric_stats(np.concatenate(accumulator.state, axis=0)),
        "action": numeric_stats(np.concatenate(accumulator.action, axis=0)),
    }
    relative: dict[str, Any] = {}
    for key, arrays in accumulator.relative.items():
        if not arrays:
            raise ValueError(f"No relative-action samples for {key}")
        relative[key] = numeric_stats(np.concatenate(arrays, axis=0))
    return stats, relative


def output_video_path(root: Path, episode_index: int, output_key: str) -> Path:
    return (
        root
        / f"videos/chunk-{episode_index // 1000:03d}"
        / output_key
        / f"episode_{episode_index:06d}.mp4"
    )


def probe_video(path: Path) -> tuple[int, int, int, float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    rate = numerator / denominator if denominator else 0.0
    return int(stream["nb_read_frames"]), int(stream["width"]), int(stream["height"]), rate


def resize_filter(width: int, height: int, mode: str) -> str:
    if mode == "stretch":
        return f"scale={width}:{height}:flags=lanczos,setsar=1"
    if mode == "pad":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1"
    )


def source_video_path(
    source: Path,
    source_info: dict[str, Any],
    metadata: dict[str, Any],
    source_key: str,
) -> tuple[Path, float]:
    prefix = f"videos/{source_key}"
    required = (
        f"{prefix}/file_index",
        f"{prefix}/chunk_index",
        f"{prefix}/from_timestamp",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Episode video metadata is missing: {missing}")
    path = source / source_info["video_path"].format(
        video_key=source_key,
        chunk_index=int(metadata[f"{prefix}/chunk_index"]),
        file_index=int(metadata[f"{prefix}/file_index"]),
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, float(metadata[f"{prefix}/from_timestamp"])


def convert_video(
    source: Path,
    destination_root: Path,
    source_info: dict[str, Any],
    episode: OutputEpisode,
    source_key: str,
    output_key: str,
    width: int,
    height: int,
    mode: str,
    codec: str,
    preset: str,
    crf: int,
) -> dict[str, Any]:
    source_path, start = source_video_path(
        source, source_info, episode.source.metadata, source_key
    )
    destination = output_video_path(destination_root, episode.output_index, output_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = float(source_info["fps"])
    frame_count = episode.source.length
    filters = (
        f"fps={fps:g},{resize_filter(width, height, mode)},"
        f"tpad=stop_mode=clone:stop_duration=2,trim=end_frame={frame_count},"
        "setpts=N/FRAME_RATE/TB"
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(source_path),
        "-an",
        "-vf",
        filters,
        "-frames:v",
        str(frame_count),
        "-r",
        f"{fps:g}",
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        command += ["-preset", preset, "-crf", str(crf)]
    else:
        command += ["-preset", "p4", "-cq", str(crf), "-b:v", "0"]
    command += [
        "-pix_fmt",
        "yuv420p",
        "-g",
        "2",
        "-keyint_min",
        "2",
        "-sc_threshold",
        "0",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    actual_frames, actual_width, actual_height, actual_fps = probe_video(destination)
    if actual_frames != frame_count:
        raise ValueError(
            f"{destination}: {actual_frames} frames; expected {frame_count}"
        )
    if (actual_width, actual_height) != (width, height):
        raise ValueError(
            f"{destination}: resolution {(actual_width, actual_height)}; "
            f"expected {(width, height)}"
        )
    if abs(actual_fps - fps) > 1e-3:
        raise ValueError(f"{destination}: fps {actual_fps}; expected {fps}")
    return {
        "source_episode_index": episode.source.source_index,
        "output_episode_index": episode.output_index,
        "camera": output_key,
        "frames": actual_frames,
        "width": actual_width,
        "height": actual_height,
        "fps": actual_fps,
        "source": str(source_path),
        "output": str(destination),
    }


def build_modality() -> dict[str, Any]:
    def field(original_key: str, start: int, end: int) -> dict[str, Any]:
        return {
            "original_key": original_key,
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }

    return {
        "state": {
            key: field("observation.state", start, end)
            for key, (start, end) in JOINT_SLICES.items()
        },
        "action": {
            key: field("action", start, end)
            for key, (start, end) in JOINT_SLICES.items()
        },
        "video": {
            "top_head": {"original_key": "observation.images.top_head"},
            "hand_left": {"original_key": "observation.images.hand_left"},
            "hand_right": {"original_key": "observation.images.hand_right"},
        },
        "annotation": {
            "language.action_text": {
                "original_key": "annotation.language.action_text"
            }
        },
    }


def build_info(
    source_info: dict[str, Any],
    episodes: list[OutputEpisode],
    tasks: list[str],
    width: int,
    height: int,
    split_name: str,
) -> dict[str, Any]:
    fps = float(source_info["fps"])
    vector_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [16],
            "names": EXPECTED_NAMES,
        },
        "action": {"dtype": "float32", "shape": [16], "names": EXPECTED_NAMES},
    }
    features: dict[str, Any] = {
        **vector_features,
        "annotation.language.action_text": {"dtype": "string", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for output_key in SOURCE_CAMERAS.values():
        features[output_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return {
        "codebase_version": "v2.1",
        "robot_type": "g2",
        "total_episodes": len(episodes),
        "total_frames": sum(item.source.length for item in episodes),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {split_name: f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def write_split_metadata(
    root: Path,
    source_info: dict[str, Any],
    split_name: str,
    episodes: list[OutputEpisode],
    tasks: list[str],
    stats: dict[str, Any],
    relative_stats: dict[str, Any],
    embodiment_tag: str,
    width: int,
    height: int,
    action_horizon: int,
    video_reports: list[dict[str, Any]],
    stats_source: str,
) -> None:
    meta = root / "meta"
    task_to_index = {task: index for index, task in enumerate(tasks)}
    write_json(
        meta / "info.json",
        build_info(source_info, episodes, tasks, width, height, split_name),
    )
    write_json(meta / "modality.json", build_modality())
    write_json(meta / "embodiment.json", {"embodiment_tag": embodiment_tag})
    write_json(meta / "stats.json", stats)
    write_json(meta / "relative_stats_dreamzero.json", relative_stats)
    write_jsonl(
        meta / "tasks.jsonl",
        ({"task_index": index, "task": task} for index, task in enumerate(tasks)),
    )
    write_jsonl(
        meta / "episodes.jsonl",
        (
            {
                "episode_index": item.output_index,
                "tasks": [item.source.task],
                "length": item.source.length,
            }
            for item in episodes
        ),
    )
    write_json(
        meta / "conversion_report.json",
        {
            "split": split_name,
            "episode_count": len(episodes),
            "frame_count": sum(item.source.length for item in episodes),
            "source_episode_indices": [item.source.source_index for item in episodes],
            "task_indices": {
                str(item.output_index): task_to_index[item.source.task] for item in episodes
            },
            "source_subtask_indices": {
                str(item.output_index): item.source.subtask_index for item in episodes
            },
            "source_high_level_tasks": {
                str(item.output_index): item.source.source_task for item in episodes
            },
            "language_supervision": "subtask_index -> meta/subtasks.parquet",
            "language_task_count": len(tasks),
            "language_tasks": tasks,
            "state_dim": 16,
            "action_dim": 16,
            "action_horizon": action_horizon,
            "camera_order": list(VIDEO_MODALITY_ORDER),
            "video_resolution": [width, height],
            "video_count": len(video_reports),
            "stats_source": stats_source,
        },
    )


def convert_split_data(
    root: Path,
    source_info: dict[str, Any],
    data_dataset: pads.Dataset,
    source_episodes: list[SourceEpisode],
    horizon: int,
) -> tuple[list[OutputEpisode], list[str], StatsAccumulator]:
    episodes = [OutputEpisode(source=item, output_index=index) for index, item in enumerate(source_episodes)]
    tasks = list(dict.fromkeys(item.source.task for item in episodes))
    task_to_index = {task: index for index, task in enumerate(tasks)}
    accumulator = StatsAccumulator()
    global_index = 0
    for count, item in enumerate(episodes, start=1):
        source_table = table_for_episode(data_dataset, item.source)
        table, state, action = output_table(
            source_table,
            item,
            float(source_info["fps"]),
            task_to_index[item.source.task],
            global_index,
        )
        destination = (
            root
            / f"data/chunk-{item.output_index // 1000:03d}"
            / f"episode_{item.output_index:06d}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd", row_group_size=item.source.length)
        accumulator.add(state, action, horizon)
        global_index += item.source.length
        if count % 20 == 0 or count == len(episodes):
            LOG.info("%s parquet: %d/%d", root.name, count, len(episodes))
    return episodes, tasks, accumulator


def convert_split_videos(
    source: Path,
    root: Path,
    source_info: dict[str, Any],
    episodes: list[OutputEpisode],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jobs = [
        (episode, source_key, output_key)
        for episode in episodes
        for source_key, output_key in SOURCE_CAMERAS.items()
    ]
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                convert_video,
                source,
                root,
                source_info,
                episode,
                source_key,
                output_key,
                args.video_width,
                args.video_height,
                args.resize_mode,
                args.video_codec,
                args.video_preset,
                args.video_crf,
            )
            for episode, source_key, output_key in jobs
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            reports.append(future.result())
            if count % 30 == 0 or count == len(jobs):
                LOG.info("%s videos: %d/%d", root.name, count, len(jobs))
    return sorted(
        reports,
        key=lambda item: (item["output_episode_index"], item["camera"]),
    )


def validate_output(root: Path, split_name: str) -> None:
    meta = root / "meta"
    required = (
        "info.json",
        "modality.json",
        "embodiment.json",
        "stats.json",
        "relative_stats_dreamzero.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "conversion_report.json",
    )
    missing = [name for name in required if not (meta / name).is_file()]
    if missing:
        raise ValueError(f"{split_name}: missing metadata files: {missing}")
    info = read_json(meta / "info.json")
    episode_count = int(info["total_episodes"])
    data_count = len(list((root / "data").glob("chunk-*/*.parquet")))
    video_count = len(list((root / "videos").glob("chunk-*/*/*.mp4")))
    if data_count != episode_count:
        raise ValueError(f"{split_name}: {data_count} parquets for {episode_count} episodes")
    if video_count != episode_count * 3:
        raise ValueError(f"{split_name}: {video_count} videos; expected {episode_count * 3}")

    output_data = pads.dataset(
        [str(path) for path in parquet_files(root / "data")],
        format="parquet",
    ).to_table(
        columns=[
            "episode_index",
            "annotation.language.action_text",
            "task_index",
        ]
    )
    output_episode_values = output_data["episode_index"].to_numpy(
        zero_copy_only=False
    )
    output_task_values = output_data["task_index"].to_numpy(zero_copy_only=False)
    output_text_values = output_data[
        "annotation.language.action_text"
    ].to_pylist()

    episode_labels: dict[int, tuple[int, str]] = {}
    for episode_value, task_value, text_value in zip(
        output_episode_values,
        output_task_values,
        output_text_values,
        strict=True,
    ):
        episode_index = int(episode_value)
        label = (int(task_value), str(text_value).strip())
        previous = episode_labels.setdefault(episode_index, label)
        if previous != label:
            raise ValueError(
                f"{split_name}: episode {episode_index} has multiple language labels: "
                f"{previous!r} / {label!r}"
            )

    if len(episode_labels) != episode_count:
        raise ValueError(
            f"{split_name}: found language labels for {len(episode_labels)} "
            f"of {episode_count} episodes"
        )

    task_rows: list[dict[str, Any]] = []
    with (meta / "tasks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                task_rows.append(json.loads(line))
    metadata_tasks = {
        int(row["task_index"]): str(row["task"]).strip() for row in task_rows
    }
    observed_tasks: dict[int, str] = {}
    for task_index, text in episode_labels.values():
        previous = observed_tasks.setdefault(task_index, text)
        if previous != text:
            raise ValueError(
                f"{split_name}: task_index {task_index} maps to multiple texts: "
                f"{previous!r} / {text!r}"
            )
    if observed_tasks != metadata_tasks:
        raise ValueError(
            f"{split_name}: parquet language labels do not match tasks.jsonl"
        )
    if len(metadata_tasks) != int(info["total_tasks"]):
        raise ValueError(
            f"{split_name}: tasks.jsonl has {len(metadata_tasks)} tasks; "
            f"info.json says {info['total_tasks']}"
        )
    LOG.info(
        "%s language audit: %d episodes, %d unique subtask texts",
        split_name,
        episode_count,
        len(metadata_tasks),
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    if min(args.video_width, args.video_height) < 2 or args.video_width % 2 or args.video_height % 2:
        raise ValueError("Video width and height must be positive even integers")

    source = args.source.resolve()
    output = args.output.resolve()
    require_tools(args.video_codec)
    validate_paths(source, output, args.overwrite)

    source_info = read_json(source / "meta/info.json")
    validate_source_info(source_info)
    source_episodes, data_files, data_dataset = load_source_episodes(
        source, source_info, args.max_source_episodes
    )
    train_source, test_source = split_episodes(
        source_episodes, args.test_episodes, args.split_mode, args.split_seed
    )
    min_frames = args.min_episode_frames or (args.action_horizon + 1)
    validate_episode_lengths(train_source, min_frames, "train")
    if test_source:
        validate_episode_lengths(test_source, min_frames, "test")

    LOG.info(
        "Source: %d episodes, %d parquet shards; split into %d train / %d test",
        len(source_episodes),
        len(data_files),
        len(train_source),
        len(test_source),
    )
    write_json(
        output / "split_manifest.json",
        {
            "source": str(source),
            "split_mode": args.split_mode,
            "split_seed": args.split_seed if args.split_mode != "tail" else None,
            "language_supervision": "subtask_index -> meta/subtasks.parquet",
            "train_source_episode_indices": [item.source_index for item in train_source],
            "test_source_episode_indices": [item.source_index for item in test_source],
            "train_subtask_indices": sorted(
                {item.subtask_index for item in train_source}
            ),
            "test_subtask_indices": sorted(
                {item.subtask_index for item in test_source}
            ),
        },
    )

    train_root = output / "train"
    train_episodes, train_tasks, train_acc = convert_split_data(
        train_root,
        source_info,
        data_dataset,
        train_source,
        args.action_horizon,
    )
    train_stats, train_relative_stats = finish_stats(train_acc)
    train_video_reports = convert_split_videos(
        source, train_root, source_info, train_episodes, args
    )
    write_split_metadata(
        train_root,
        source_info,
        "train",
        train_episodes,
        train_tasks,
        train_stats,
        train_relative_stats,
        args.embodiment_tag,
        args.video_width,
        args.video_height,
        args.action_horizon,
        train_video_reports,
        "train",
    )
    validate_output(train_root, "train")

    if test_source:
        test_root = output / "test"
        test_episodes, test_tasks, _ = convert_split_data(
            test_root,
            source_info,
            data_dataset,
            test_source,
            args.action_horizon,
        )
        test_video_reports = convert_split_videos(
            source, test_root, source_info, test_episodes, args
        )
        # Test data deliberately uses training-set normalization statistics.
        write_split_metadata(
            test_root,
            source_info,
            "test",
            test_episodes,
            test_tasks,
            train_stats,
            train_relative_stats,
            args.embodiment_tag,
            args.video_width,
            args.video_height,
            args.action_horizon,
            test_video_reports,
            "train",
        )
        validate_output(test_root, "test")

    LOG.info("Conversion complete. DreamZero train root: %s", train_root)
    if test_source:
        LOG.info("Held-out test root: %s", output / "test")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Conversion failed")
        sys.exit(1)
