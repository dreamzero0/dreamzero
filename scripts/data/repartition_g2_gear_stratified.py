#!/usr/bin/env python3
"""Repartition existing GEAR train/test splits into a task-stratified dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.convert_lerobot_to_gear import compute_relative_stats, compute_stats


@dataclass(frozen=True)
class Episode:
    root: Path
    source_split: str
    source_index: int
    task: str
    length: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_episodes(source: Path) -> tuple[list[Episode], list[str]]:
    episodes: list[Episode] = []
    canonical_tasks: list[str] = []
    for split in ("train", "test"):
        root = source / split
        task_rows = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
        tasks = {int(row["task_index"]): row["task"] for row in task_rows}
        for task in tasks.values():
            if task not in canonical_tasks:
                canonical_tasks.append(task)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            task = row["tasks"][0]
            episodes.append(
                Episode(root, split, int(row["episode_index"]), task, int(row["length"]))
            )
    return episodes, canonical_tasks


def source_parquet(item: Episode) -> Path:
    return item.root / f"data/chunk-{item.source_index // 1000:03d}/episode_{item.source_index:06d}.parquet"


def materialize_split(
    output: Path,
    split: str,
    episodes: list[Episode],
    canonical_tasks: list[str],
    info_template: dict,
    modality: dict,
    embodiment: dict,
    invert_policy_gripper: bool,
) -> tuple[list[Path], list[dict], list[str]]:
    root = output / split
    (root / "meta").mkdir(parents=True)
    used_tasks = [task for task in canonical_tasks if any(e.task == task for e in episodes)]
    task_ids = {task: index for index, task in enumerate(used_tasks)}
    parquet_paths: list[Path] = []
    episode_rows: list[dict] = []
    global_index = 0
    video_keys = tuple(modality["video"])

    for output_index, item in enumerate(episodes):
        chunk = output_index // 1000
        destination = root / f"data/chunk-{chunk:03d}/episode_{output_index:06d}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(source_parquet(item))
        if len(frame) != item.length:
            raise RuntimeError(f"Length mismatch for {source_parquet(item)}")
        if invert_policy_gripper:
            for column in ("observation.state", "action"):
                values = np.stack(frame[column]).astype(np.float32)
                if values.shape[1:] != (16,):
                    raise ValueError(f"{source_parquet(item)}: {column} is not 16D")
                if np.any(values[:, (7, 15)] < 0) or np.any(values[:, (7, 15)] > 1):
                    raise ValueError(
                        f"{source_parquet(item)}: {column} gripper is outside [0,1]"
                    )
                values[:, (7, 15)] = 1.0 - values[:, (7, 15)]
                frame[column] = list(values)
        frame["episode_index"] = output_index
        frame["task_index"] = task_ids[item.task]
        frame["index"] = range(global_index, global_index + len(frame))
        frame.to_parquet(destination, index=False)
        parquet_paths.append(destination)
        global_index += len(frame)
        episode_rows.append(
            {"episode_index": output_index, "tasks": [item.task], "length": len(frame)}
        )

        source_chunk = item.source_index // 1000
        for key in video_keys:
            source_video = item.root / f"videos/chunk-{source_chunk:03d}/observation.images.{key}/episode_{item.source_index:06d}.mp4"
            destination_video = root / f"videos/chunk-{chunk:03d}/observation.images.{key}/episode_{output_index:06d}.mp4"
            destination_video.parent.mkdir(parents=True, exist_ok=True)
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            os.link(source_video, destination_video)

    info = dict(info_template)
    info["total_episodes"] = len(episodes)
    info["total_frames"] = global_index
    info["total_tasks"] = len(used_tasks)
    info["splits"] = {split: f"0:{len(episodes)}"}
    write_json(root / "meta/info.json", info)
    write_json(root / "meta/modality.json", modality)
    write_json(root / "meta/embodiment.json", embodiment)
    (root / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": task}, ensure_ascii=False) + "\n"
            for index, task in enumerate(used_tasks)
        ),
        encoding="utf-8",
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    return parquet_paths, episode_rows, used_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--invert-policy-gripper",
        action="store_true",
        help="Convert source 0=open,1=closed to canonical 0=closed,1=open.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    episodes, canonical_tasks = load_episodes(args.source)
    by_task: dict[str, list[Episode]] = {task: [] for task in canonical_tasks}
    for item in episodes:
        by_task[item.task].append(item)
    rng = random.Random(args.seed)
    test_set: set[Episode] = set()
    for task, group in by_task.items():
        if len(group) <= args.test_per_task:
            raise RuntimeError(f"Task {task!r} has only {len(group)} episodes")
        test_set.update(rng.sample(group, args.test_per_task))
    train = [item for item in episodes if item not in test_set]
    test = [item for item in episodes if item in test_set]

    template_root = args.source / "train/meta"
    info = read_json(template_root / "info.json")
    modality = read_json(template_root / "modality.json")
    embodiment = read_json(template_root / "embodiment.json")
    try:
        train_paths, train_rows, train_tasks = materialize_split(
            args.output, "train", train, canonical_tasks, info, modality,
            embodiment, args.invert_policy_gripper,
        )
        test_paths, test_rows, test_tasks = materialize_split(
            args.output, "test", test, canonical_tasks, info, modality,
            embodiment, args.invert_policy_gripper,
        )
        numeric_columns = list(read_json(template_root / "stats.json"))
        train_stats = compute_stats(train_paths, numeric_columns)
        relative_stats = compute_relative_stats(
            train_paths,
            modality,
            ["left_joint_position", "right_joint_position"],
            action_horizon=24,
        )
        for split in ("train", "test"):
            write_json(args.output / split / "meta/stats.json", train_stats)
            write_json(
                args.output / split / "meta/relative_stats_dreamzero.json",
                relative_stats,
            )
            write_json(
                args.output / split / "meta/conversion_report.json",
                {
                    "source": str(args.source),
                    "split_mode": "stratified_by_subtask",
                    "test_per_task": args.test_per_task,
                    "seed": args.seed,
                    "invert_policy_gripper": args.invert_policy_gripper,
                },
            )
        write_json(
            args.output / "split_manifest.json",
            {
                "source": str(args.source),
                "split_mode": "stratified_by_subtask",
                "seed": args.seed,
                "test_per_task": args.test_per_task,
                "invert_policy_gripper": args.invert_policy_gripper,
                "train_episodes": len(train_rows),
                "test_episodes": len(test_rows),
                "train_tasks": train_tasks,
                "test_tasks": test_tasks,
                "test_source_episodes": [
                    {"split": item.source_split, "episode_index": item.source_index, "task": item.task}
                    for item in test
                ],
            },
        )
    except Exception:
        shutil.rmtree(args.output, ignore_errors=True)
        raise

    print(
        f"Created {args.output}: train={len(train)} test={len(test)} "
        f"tasks={len(canonical_tasks)} test_per_task={args.test_per_task}"
    )


if __name__ == "__main__":
    main()
