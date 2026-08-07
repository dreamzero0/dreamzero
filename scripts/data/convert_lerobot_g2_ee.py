#!/usr/bin/env python3
"""Convert the FruitPackaging LeRobot v3 dataset to DreamZero's LeRobot v2 layout.

The source dataset stores many episodes in shared parquet and video shards. DreamZero's
current loader expects one parquet and one video per episode, plus GEAR metadata files.
This converter preserves all samples, converts source xyzw quaternions to rotation-6d,
materializes the task text, splits videos on episode boundaries, and computes stats.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


LOG = logging.getLogger("fruit-v3-converter")

SOURCE_VECTOR_NAMES = [
    "l.ee.x", "l.ee.y", "l.ee.z",
    "l.ee.qx", "l.ee.qy", "l.ee.qz", "l.ee.qw",
    "l.ee.gripper.pos",
    "r.ee.x", "r.ee.y", "r.ee.z",
    "r.ee.qx", "r.ee.qy", "r.ee.qz", "r.ee.qw",
    "r.ee.gripper.pos",
]

# PyTorch3D-compatible rotation-6d convention: flatten the first two matrix rows.
# Conversion is implemented in NumPy so DreamZero training does not need PyTorch3D.
OUTPUT_VECTOR_NAMES = [
    "l.ee.x", "l.ee.y", "l.ee.z",
    "l.ee.rot6d.r0c0", "l.ee.rot6d.r0c1", "l.ee.rot6d.r0c2",
    "l.ee.rot6d.r1c0", "l.ee.rot6d.r1c1", "l.ee.rot6d.r1c2",
    "l.ee.gripper.pos",
    "r.ee.x", "r.ee.y", "r.ee.z",
    "r.ee.rot6d.r0c0", "r.ee.rot6d.r0c1", "r.ee.rot6d.r0c2",
    "r.ee.rot6d.r1c0", "r.ee.rot6d.r1c1", "r.ee.rot6d.r1c2",
    "r.ee.gripper.pos",
]

OUTPUT_VECTOR_DIM = 20
ROTATION_OUTPUT_SLICES = ((3, 9), (13, 19))

CAMERAS = {
    "observation.images.head_color": "observation.images.top_head",
    "observation.images.hand_left_color": "observation.images.hand_left",
    "observation.images.hand_right_color": "observation.images.hand_right",
}

STATE_ACTION_FIELDS = {
    "left_effector_position": {
        "bounds": (0, 3),
        "rotation_type": None,
    },
    "left_effector_rotation": {
        "bounds": (3, 9),
        "rotation_type": "rotation_6d",
    },
    "left_gripper_position": {
        "bounds": (9, 10),
        "rotation_type": None,
    },
    "right_effector_position": {
        "bounds": (10, 13),
        "rotation_type": None,
    },
    "right_effector_rotation": {
        "bounds": (13, 19),
        "rotation_type": "rotation_6d",
    },
    "right_gripper_position": {
        "bounds": (19, 20),
        "rotation_type": None,
    },
}

RELATIVE_ACTION_KEYS = (
    "left_effector_position",
    "left_gripper_position",
    "right_effector_position",
    "right_gripper_position",
)


def nonempty_path(value: str) -> Path:
    """Reject an unset shell variable such as --output "$TEST_DATA"."""
    if not value.strip():
        raise argparse.ArgumentTypeError("path cannot be empty")
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=nonempty_path, required=True)
    parser.add_argument("--output", type=nonempty_path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--video-width",
        type=int,
        default=None,
        help="Resize every output camera to this width (use with --video-height).",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=None,
        help="Resize every output camera to this height (use with --video-width).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes (intended for converter tests).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def check_tools() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable not found: {executable}")


def validate_output_path(source: Path, output: Path) -> None:
    """Refuse paths where --overwrite could delete source code or source data."""
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    filesystem_root = Path(output.anchor).resolve()

    if output in {filesystem_root, home}:
        raise ValueError(f"Refusing dangerous output path: {output}")

    if output == cwd or output in cwd.parents:
        raise ValueError(
            f"Refusing output path {output}: it is the current directory or its parent"
        )

    if output == source or output in source.parents or source in output.parents:
        raise ValueError(
            f"Refusing output path {output}: it overlaps source dataset {source}"
        )

    if (output / ".git").exists():
        raise ValueError(f"Refusing to overwrite Git repository: {output}")


def source_table_paths(source: Path) -> tuple[Path, Path, Path]:
    data_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    episode_files = sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))
    tasks_path = source / "meta/tasks.parquet"
    if len(data_files) != 1:
        raise ValueError(f"Expected one source data parquet, found {len(data_files)}")
    if len(episode_files) != 1:
        raise ValueError(f"Expected one source episode parquet, found {len(episode_files)}")
    if not tasks_path.is_file():
        raise FileNotFoundError(tasks_path)
    return data_files[0], episode_files[0], tasks_path


def validate_source(
    info: dict[str, Any], data_file: Path, episode_rows: list[dict[str, Any]], max_episodes: int | None
) -> int:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    for key in ("observation.state", "action", *CAMERAS):
        if key not in info["features"]:
            raise ValueError(f"Missing source feature: {key}")
    if info["features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Fruit converter expects a 16-dimensional state")
    if info["features"]["action"]["shape"] != [16]:
        raise ValueError("Fruit converter expects a 16-dimensional action")
    for key in ("observation.state", "action"):
        names = info["features"][key].get("names")
        if names != SOURCE_VECTOR_NAMES:
            raise ValueError(
                f"{key} names do not match the expected G2 EE xyzw layout: "
                f"expected={SOURCE_VECTOR_NAMES}, actual={names}"
            )

    total = int(info["total_episodes"])
    if len(episode_rows) != total:
        raise ValueError(f"Episode metadata has {len(episode_rows)} rows, expected {total}")
    parquet = pq.ParquetFile(data_file)
    if parquet.metadata.num_rows != int(info["total_frames"]):
        raise ValueError("Data parquet frame count does not match info.json")
    metadata_frames = sum(int(row["length"]) for row in episode_rows)
    if metadata_frames != parquet.metadata.num_rows:
        raise ValueError(
            "Episode metadata lengths do not add up to the data parquet row count: "
            f"{metadata_frames} != {parquet.metadata.num_rows}"
        )

    episode_index_table = pq.read_table(data_file, columns=["episode_index"])
    episode_ids = np.asarray(
        episode_index_table["episode_index"].combine_chunks().to_numpy(),
        dtype=np.int64,
    )
    expected_ids = np.asarray(
        [int(row["episode_index"]) for row in episode_rows],
        dtype=np.int64,
    )
    actual_ids, actual_counts = np.unique(episode_ids, return_counts=True)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(
            "Episode ids in the data parquet do not match meta/episodes: "
            f"data={actual_ids.tolist()}, metadata={expected_ids.tolist()}"
        )
    expected_counts = np.asarray(
        [int(row["length"]) for row in episode_rows],
        dtype=np.int64,
    )
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError(
            "Per-episode frame counts in the data parquet do not match meta/episodes: "
            f"data={actual_counts.tolist()}, metadata={expected_counts.tolist()}"
        )
    if max_episodes is not None:
        if max_episodes < 1 or max_episodes > total:
            raise ValueError(f"--max-episodes must be in [1, {total}]")
        return max_episodes
    return total


def feature_video_info(
    source_feature: dict[str, Any], video_width: int | None, video_height: int | None
) -> dict[str, Any]:
    shape = (
        [video_height, video_width, 3]
        if video_width is not None and video_height is not None
        else list(source_feature["shape"])
    )
    source_info = source_feature.get("info", source_feature.get("video_info", {}))
    return {
        "dtype": "video",
        "shape": shape,
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(source_info.get("video.fps", 30.0)),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def build_info(
    source_info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    episode_count: int,
    video_width: int | None,
    video_height: int | None,
) -> dict[str, Any]:
    total_frames = sum(int(row["length"]) for row in episode_rows[:episode_count])
    task_ids = {
        int(row["episode_index"]): tuple(row["tasks"])
        for row in episode_rows[:episode_count]
    }
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [OUTPUT_VECTOR_DIM],
            "names": OUTPUT_VECTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [OUTPUT_VECTOR_DIM],
            "names": OUTPUT_VECTOR_NAMES,
        },
        "annotation.task": {"dtype": "string", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for source_key, output_key in CAMERAS.items():
        features[output_key] = feature_video_info(
            source_info["features"][source_key], video_width, video_height
        )

    return {
        "codebase_version": "v2.1",
        "robot_type": "A2D_FRUIT_AGIBOT",
        "total_episodes": episode_count,
        "total_frames": total_frames,
        "total_tasks": len({tasks for tasks in task_ids.values()}),
        "chunks_size": 1000,
        "fps": float(source_info["fps"]),
        "splits": {"train": f"0:{episode_count}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }


def build_modality() -> dict[str, Any]:
    def packed_field(
        bounds: tuple[int, int], original_key: str, rotation_type: str | None
    ) -> dict[str, Any]:
        return {
            "original_key": original_key,
            "start": bounds[0],
            "end": bounds[1],
            "rotation_type": rotation_type,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }

    return {
        "state": {
            key: packed_field(field["bounds"], "observation.state", field["rotation_type"])
            for key, field in STATE_ACTION_FIELDS.items()
        },
        "action": {
            key: packed_field(field["bounds"], "action", field["rotation_type"])
            for key, field in STATE_ACTION_FIELDS.items()
        },
        "video": {
            "top_head": {"original_key": "observation.images.top_head"},
            "hand_left": {"original_key": "observation.images.hand_left"},
            "hand_right": {"original_key": "observation.images.hand_right"},
        },
        "annotation": {
            "language.action_text": {"original_key": "annotation.task"},
        },
    }


def compute_output_stats(
    output_parts: dict[str, list[np.ndarray]], source_stats: dict[str, Any]
) -> dict[str, Any]:
    required = ("min", "max", "mean", "std", "q01", "q99")

    result: dict[str, Any] = {}
    for key in ("observation.state", "action"):
        if not output_parts[key]:
            raise ValueError(f"No converted values available for {key} stats")
        values = np.concatenate(output_parts[key], axis=0).astype(np.float64)
        result[key] = {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
        # rotation-6d is analytically bounded by [-1, 1]. Using fixed bounds
        # avoids unstable min-max scaling when a small smoke subset has little
        # rotational variation.
        for start, end in ROTATION_OUTPUT_SLICES:
            result[key]["min"][start:end] = [-1.0] * (end - start)
            result[key]["max"][start:end] = [1.0] * (end - start)
    # LeRobot v3 datasets do not always store timestamp statistics. DreamZero
    # normalizes state/action, so retain timestamp stats only when supplied.
    if "timestamp" in source_stats:
        result["timestamp"] = {
            name: source_stats["timestamp"][name] for name in required
        }
    return result


def quaternion_xyzw_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to PyTorch3D-compatible matrix rotation-6d."""
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 1e-6):
        raise ValueError("Zero quaternion found")
    max_error = float(np.max(np.abs(norms - 1.0)))
    if max_error > 5e-2:
        raise ValueError(f"Quaternion max |norm-1| is too large: {max_error:.6f}")
    normalized = quaternion / norms[:, None]
    x, y, z, w = normalized.T
    matrix = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    return matrix[:, :2, :].reshape(-1, 6).astype(np.float32, copy=False)


def convert_xyzw_to_rotation_6d(values: np.ndarray) -> np.ndarray:
    """Convert both G2 EE quaternions and repack 16 source values as 20."""
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"Expected [N, 16] G2 vector, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("State/action contains NaN or Inf")
    output = np.empty((values.shape[0], OUTPUT_VECTOR_DIM), dtype=np.float32)
    output[:, 0:3] = values[:, 0:3]
    output[:, 3:9] = quaternion_xyzw_to_rotation_6d(values[:, 3:7])
    output[:, 9] = values[:, 7]
    output[:, 10:13] = values[:, 8:11]
    output[:, 13:19] = quaternion_xyzw_to_rotation_6d(values[:, 11:15])
    output[:, 19] = values[:, 15]
    return output


def replace_packed_column(
    table: pa.Table, column_name: str, values: np.ndarray
) -> pa.Table:
    index = table.schema.get_field_index(column_name)
    if index < 0:
        raise KeyError(column_name)
    packed = pa.array(values.tolist(), type=pa.list_(pa.float32(), OUTPUT_VECTOR_DIM))
    return table.set_column(index, column_name, packed)


def write_parquets_and_collect_relative(
    data_file: Path,
    output: Path,
    episode_rows: list[dict[str, Any]],
    task_by_index: dict[int, str],
    episode_count: int,
    action_horizon: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    relative_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in RELATIVE_ACTION_KEYS
    }
    output_parts: dict[str, list[np.ndarray]] = {
        "observation.state": [],
        "action": [],
    }
    columns = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    source_table = pq.read_table(data_file, columns=columns)

    for converted_index, metadata in enumerate(episode_rows[:episode_count], start=1):
        episode_index = int(metadata["episode_index"])
        table = source_table.filter(
            pc.equal(source_table["episode_index"], episode_index)
        )
        expected = int(metadata["length"])
        if table.num_rows != expected:
            raise ValueError(
                f"Episode {episode_index}: parquet has {table.num_rows} rows, expected {expected}"
            )
        episode_ids = np.asarray(
            table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64
        )
        if not np.all(episode_ids == episode_index):
            raise ValueError(f"Episode {episode_index}: row group contains another episode id")
        task_indices = np.asarray(
            table["task_index"].combine_chunks().to_numpy(), dtype=np.int64
        )
        if len(np.unique(task_indices)) != 1:
            raise ValueError(f"Episode {episode_index}: multiple task indices in one episode")
        task = task_by_index[int(task_indices[0])]
        declared_tasks = list(metadata["tasks"])
        if declared_tasks != [task]:
            raise ValueError(
                f"Episode {episode_index}: task mismatch: metadata={declared_tasks}, parquet={task!r}"
            )

        source_state = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        source_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = convert_xyzw_to_rotation_6d(source_state)
        action = convert_xyzw_to_rotation_6d(source_action)
        output_parts["observation.state"].append(state)
        output_parts["action"].append(action)
        table = replace_packed_column(table, "observation.state", state)
        table = replace_packed_column(table, "action", action)

        annotation = pa.array([task] * expected, type=pa.large_string())
        table = table.append_column("annotation.task", annotation)
        table = table.select(
            [
                "observation.state",
                "action",
                "annotation.task",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ]
        )
        chunk = episode_index // 1000
        destination = (
            output / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd", row_group_size=expected)

        usable = expected - action_horizon
        if usable > 0:
            for key in RELATIVE_ACTION_KEYS:
                start, end = STATE_ACTION_FIELDS[key]["bounds"]
                episode_relative = np.empty(
                    (usable, action_horizon, end - start), dtype=np.float32
                )
                reference = state[:usable, None, start:end]
                for horizon_index in range(action_horizon):
                    episode_relative[:, horizon_index] = (
                        action[horizon_index : horizon_index + usable, start:end] - reference[:, 0]
                    )
                relative_parts[key].append(episode_relative.reshape(-1, end - start))

        if converted_index % 50 == 0 or converted_index == episode_count:
            LOG.info("Wrote parquet episodes: %d/%d", converted_index, episode_count)

    return relative_parts, output_parts


def compute_relative_stats(parts: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, arrays in parts.items():
        if not arrays:
            raise ValueError(f"No relative-action samples for {key}")
        values = np.concatenate(arrays, axis=0).astype(np.float64)
        result[key] = {
            "max": np.max(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return result


def output_video_path(output: Path, episode_index: int, output_key: str) -> Path:
    return (
        output
        / f"videos/chunk-{episode_index // 1000:03d}"
        / output_key
        / f"episode_{episode_index:06d}.mp4"
    )


def probe_frames(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def convert_video_job(
    source: Path,
    output: Path,
    source_info: dict[str, Any],
    metadata: dict[str, Any],
    episode_index: int,
    source_key: str,
    output_key: str,
    fps: float,
    preset: str,
    crf: int,
    video_width: int | None,
    video_height: int | None,
) -> dict[str, Any]:
    prefix = f"videos/{source_key}"
    file_index = int(metadata[f"{prefix}/file_index"])
    chunk_index = int(metadata[f"{prefix}/chunk_index"])
    start = float(metadata[f"{prefix}/from_timestamp"])
    end = float(metadata[f"{prefix}/to_timestamp"])
    expected_frames = int(metadata["length"])
    available_frames = int(round((end - start) * fps))
    if available_frames < 1 or available_frames > expected_frames:
        raise ValueError(
            f"Episode {episode_index} {source_key}: invalid available frame count "
            f"{available_frames}, expected {expected_frames}"
        )
    pad_frames = expected_frames - available_frames
    source_path = source / source_info["video_path"].format(
        video_key=source_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    destination = output_video_path(output, episode_index, output_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    video_filter = f"trim=end_frame={available_frames}"
    if pad_frames:
        video_filter += f",tpad=stop_mode=clone:stop={pad_frames}"
    video_filter += ",setpts=PTS-STARTPTS"
    if video_width is not None and video_height is not None:
        video_filter += f",scale={video_width}:{video_height}:flags=lanczos,setsar=1"
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(source_path),
        "-an",
        "-vf",
        video_filter,
        "-frames:v",
        str(expected_frames),
        "-r",
        f"{fps:g}",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
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
    actual_frames = probe_frames(destination)
    if actual_frames != expected_frames:
        raise ValueError(
            f"Episode {episode_index} {output_key}: encoded {actual_frames} frames, "
            f"expected {expected_frames}"
        )
    return {
        "episode_index": episode_index,
        "camera": output_key,
        "source_file": str(source_path),
        "source_start": start,
        "source_end": end,
        "expected_frames": expected_frames,
        "source_interval_frames": available_frames,
        "padded_frames": pad_frames,
        "output": str(destination),
    }


def convert_videos(
    source: Path,
    output: Path,
    source_info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    episode_count: int,
    workers: int,
    preset: str,
    crf: int,
    video_width: int | None,
    video_height: int | None,
) -> list[dict[str, Any]]:
    fps = float(source_info["fps"])
    jobs = []
    for metadata in episode_rows[:episode_count]:
        episode_index = int(metadata["episode_index"])
        for source_key, output_key in CAMERAS.items():
            jobs.append((episode_index, metadata, source_key, output_key))

    reports = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                convert_video_job,
                source,
                output,
                source_info,
                metadata,
                episode_index,
                source_key,
                output_key,
                fps,
                preset,
                crf,
                video_width,
                video_height,
            )
            for episode_index, metadata, source_key, output_key in jobs
        ]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            reports.append(future.result())
            if completed_count % 50 == 0 or completed_count == len(jobs):
                LOG.info("Wrote episode videos: %d/%d", completed_count, len(jobs))
    return sorted(reports, key=lambda item: (item["episode_index"], item["camera"]))


def write_metadata_files(
    output: Path,
    source_info: dict[str, Any],
    output_stats: dict[str, Any],
    tasks: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    episode_count: int,
    relative_stats: dict[str, Any],
    video_report: list[dict[str, Any]],
    action_horizon: int,
    video_width: int | None,
    video_height: int | None,
) -> None:
    meta = output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    write_json(
        meta / "info.json",
        build_info(
            source_info, episode_rows, episode_count, video_width, video_height
        ),
    )
    write_json(meta / "modality.json", build_modality())
    write_json(meta / "embodiment.json", {"embodiment_tag": "agibot"})
    write_json(meta / "stats.json", output_stats)
    write_json(meta / "relative_stats_dreamzero.json", relative_stats)

    used_task_indices = {
        int(row["task_index"])
        for row in tasks
        if any(row["task"] in episode["tasks"] for episode in episode_rows[:episode_count])
    }
    with (meta / "tasks.jsonl").open("w") as handle:
        for row in tasks:
            if int(row["task_index"]) in used_task_indices:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    with (meta / "episodes.jsonl").open("w") as handle:
        for row in episode_rows[:episode_count]:
            handle.write(
                json.dumps(
                    {
                        "episode_index": int(row["episode_index"]),
                        "tasks": list(row["tasks"]),
                        "length": int(row["length"]),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    repairs = [item for item in video_report if item["padded_frames"]]
    write_json(
        meta / "conversion_report.json",
        {
            "source": str(source_info.get("source_path", "")),
            "episode_count": episode_count,
            "video_count": len(video_report),
            "action_horizon": action_horizon,
            "state_dim": OUTPUT_VECTOR_DIM,
            "action_dim": OUTPUT_VECTOR_DIM,
            "source_quaternion_order": "xyzw",
            "output_rotation_representation": "rotation_6d",
            "relative_action_keys": list(RELATIVE_ACTION_KEYS),
            "repairs": repairs,
        },
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = args.source.resolve()
    output = args.output.resolve()
    if (args.video_width is None) != (args.video_height is None):
        raise ValueError("--video-width and --video-height must be provided together")
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer")
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be a positive integer")
    if args.video_width is not None and (
        args.video_width < 2
        or args.video_height < 2
        or args.video_width % 2
        or args.video_height % 2
    ):
        raise ValueError("Output video width and height must be positive even integers")
    if not source.is_dir():
        raise FileNotFoundError(source)
    validate_output_path(source, output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    check_tools()

    source_info = load_json(source / "meta/info.json")
    source_info["source_path"] = str(source)
    source_stats = load_json(source / "meta/stats.json")
    data_file, episodes_file, tasks_file = source_table_paths(source)
    episode_rows = pq.read_table(episodes_file).to_pylist()
    tasks = pq.read_table(tasks_file).to_pylist()
    episode_count = validate_source(source_info, data_file, episode_rows, args.max_episodes)
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}

    LOG.info("Converting %d episodes from %s", episode_count, source)
    relative_parts, output_parts = write_parquets_and_collect_relative(
        data_file,
        output,
        episode_rows,
        task_by_index,
        episode_count,
        args.action_horizon,
    )
    relative_stats = compute_relative_stats(relative_parts)
    output_stats = compute_output_stats(output_parts, source_stats)
    video_report = convert_videos(
        source,
        output,
        source_info,
        episode_rows,
        episode_count,
        args.workers,
        args.video_preset,
        args.video_crf,
        args.video_width,
        args.video_height,
    )
    write_metadata_files(
        output,
        source_info,
        output_stats,
        tasks,
        episode_rows,
        episode_count,
        relative_stats,
        video_report,
        args.action_horizon,
        args.video_width,
        args.video_height,
    )
    LOG.info("Conversion completed: %s", output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Conversion failed")
        sys.exit(1)
