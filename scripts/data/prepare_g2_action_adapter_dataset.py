#!/usr/bin/env python3
"""Prepare a non-destructive G2 dataset for action-adapter-only training.

The source GEAR dataset stores G2 grippers in SDK space [0, -0.785]. This
script rewrites state/action gripper dimensions 7 and 15 into policy space
0=closed, 1=open, recomputes normalization metadata, and records active/hold
24-step windows. Videos are symlinked; source data is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.convert_lerobot_to_gear import (
    compute_relative_stats,
    compute_stats,
)


GRIPPER_DIMS = (7, 15)
ARM_DIMS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
G2_OPEN_POSITION = -0.785


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def sdk_to_policy(vector: object) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32).copy()
    if result.shape != (16,):
        raise ValueError(f"Expected 16D G2 vector, got {result.shape}")
    result[list(GRIPPER_DIMS)] = np.clip(
        result[list(GRIPPER_DIMS)] / G2_OPEN_POSITION,
        0.0,
        1.0,
    )
    return result


def prepare_split(source: Path, output: Path, action_horizon: int) -> None:
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite derived dataset: {output}"
        )
    (output / "data").mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    write_json(
        output / "meta" / "embodiment.json",
        {"robot_type": "g2", "embodiment_tag": "g2"},
    )

    source_videos = source / "videos"
    if source_videos.exists():
        (output / "videos").symlink_to(
            source_videos.resolve(),
            target_is_directory=True,
        )

    output_parquets: list[Path] = []
    window_metrics: list[tuple[int, int, float, bool]] = []
    for source_parquet in sorted(source.glob("data/chunk-*/*.parquet")):
        relative = source_parquet.relative_to(source)
        output_parquet = output / relative
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(source_parquet)
        for column in ("observation.state", "action"):
            if column not in frame:
                raise KeyError(f"{source_parquet}: missing {column}")
            frame[column] = [sdk_to_policy(value) for value in frame[column]]
        frame.to_parquet(output_parquet, index=False)
        output_parquets.append(output_parquet)

        episode = int(frame["episode_index"].iloc[0])
        states = np.stack(frame["observation.state"]).astype(np.float32)
        actions = np.stack(frame["action"]).astype(np.float32)
        usable = max(0, len(frame) - action_horizon + 1)
        for step in range(usable):
            future = actions[step : step + action_horizon]
            arm_delta = future[:, ARM_DIMS] - states[step, ARM_DIMS]
            arm_motion = float(np.linalg.norm(arm_delta, axis=1).max())
            grip_transition = bool(
                np.any(
                    np.abs(
                        future[:, GRIPPER_DIMS]
                        - states[step, GRIPPER_DIMS]
                    )
                    > 0.25
                )
            )
            window_metrics.append(
                (episode, step, arm_motion, grip_transition)
            )

    if not output_parquets:
        raise FileNotFoundError(f"No GEAR parquet files below {source}")

    modality = read_json(output / "meta" / "modality.json")
    info = read_json(output / "meta" / "info.json")
    numeric_columns = list(read_json(source / "meta" / "stats.json"))
    write_json(
        output / "meta" / "stats.json",
        compute_stats(output_parquets, numeric_columns),
    )
    write_json(
        output / "meta" / "relative_stats_dreamzero.json",
        compute_relative_stats(
            output_parquets,
            modality,
            ["left_joint_position", "right_joint_position"],
            action_horizon=action_horizon,
        ),
    )

    nonzero_motion = np.asarray(
        [metric[2] for metric in window_metrics if metric[2] > 1e-8],
        dtype=np.float64,
    )
    if nonzero_motion.size == 0:
        raise RuntimeError(f"{source}: no nonzero arm-motion windows")
    threshold = float(np.quantile(nonzero_motion, 0.30))
    active: dict[str, list[int]] = {}
    hold: dict[str, list[int]] = {}
    for episode, step, arm_motion, grip_transition in window_metrics:
        target = active if arm_motion >= threshold or grip_transition else hold
        target.setdefault(str(episode), []).append(step)
    write_json(
        output / "meta" / "g2_active_hold_windows.json",
        {
            "schema_version": 1,
            "action_horizon": action_horizon,
            "arm_motion_threshold_rule": "p30_nonzero_24_step_l2",
            "arm_motion_threshold": threshold,
            "gripper_transition_threshold": 0.25,
            "active_count": sum(map(len, active.values())),
            "hold_count": sum(map(len, hold.values())),
            "active": active,
            "hold": hold,
        },
    )
    print(
        f"[G2 DATASET] {source} -> {output} "
        f"episodes={info['total_episodes']} threshold={threshold:.8f} "
        f"active={sum(map(len, active.values()))} "
        f"hold={sum(map(len, hold.values()))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=24)
    args = parser.parse_args()
    for split in ("train", "test"):
        prepare_split(
            args.source_root / split,
            args.output_root / split,
            args.action_horizon,
        )


if __name__ == "__main__":
    main()
