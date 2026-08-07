"""Evaluate production causal actions on every G2 episode start."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from eval_utils.policy_client import WebsocketClientPolicy


CAMERAS = ("top_head", "hand_left", "hand_right")


def _read_history(path: Path, end_row: int = 3) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) <= end_row:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {len(frames)}: {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames[-4:], axis=0).astype(np.uint8)


def _arm_metrics(
    action: np.ndarray, state: np.ndarray, ground_truth: np.ndarray
) -> dict[str, object]:
    pred_motion = [
        float(np.max(np.abs(action[:, 0:7] - state[None, 0:7]))),
        float(np.max(np.abs(action[:, 8:15] - state[None, 8:15]))),
    ]
    gt_motion = [
        float(np.max(np.abs(ground_truth[:, 0:7] - state[None, 0:7]))),
        float(np.max(np.abs(ground_truth[:, 8:15] - state[None, 8:15]))),
    ]
    pred_dominant = int(np.argmax(pred_motion))
    gt_dominant = int(np.argmax(gt_motion))
    gt_peak = max(gt_motion)
    pred_on_gt_arm = pred_motion[gt_dominant]
    return {
        "pred_motion_left_rad": pred_motion[0],
        "pred_motion_right_rad": pred_motion[1],
        "gt_motion_left_rad": gt_motion[0],
        "gt_motion_right_rad": gt_motion[1],
        "pred_dominant_arm": ("left", "right")[pred_dominant],
        "gt_dominant_arm": ("left", "right")[gt_dominant],
        "dominant_arm_correct": pred_dominant == gt_dominant,
        "motion_ratio_on_gt_arm": (
            pred_on_gt_arm / gt_peak if gt_peak > 1e-8 else None
        ),
        "mae_left_rad": float(
            np.mean(np.abs(action[:, 0:7] - ground_truth[:, 0:7]))
        ),
        "mae_right_rad": float(
            np.mean(np.abs(action[:, 8:15] - ground_truth[:, 8:15]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    with (args.dataset_root / "meta/info.json").open(
        "r", encoding="utf-8"
    ) as stream:
        info = json.load(stream)
    total = int(info["total_episodes"])
    if args.max_episodes > 0:
        total = min(total, args.max_episodes)

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    records: list[dict[str, object]] = []
    row = 3
    for episode in range(total):
        parquet_path = (
            args.dataset_root
            / "data"
            / f"chunk-{episode // 1000:03d}"
            / f"episode_{episode:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        states = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if len(actions) < row + 24:
            logging.warning("Skipping short episode %d", episode)
            continue
        prompt = str(table["annotation.language.action_text"][row].as_py())
        observation: dict[str, object] = {
            "observation/left_joint_position": states[row, 0:7],
            "observation/left_gripper_position": states[row, 7:8],
            "observation/right_joint_position": states[row, 8:15],
            "observation/right_gripper_position": states[row, 15:16],
            "prompt": prompt,
            "session_id": f"initial-action-audit-episode-{episode:06d}",
        }
        for camera in CAMERAS:
            video_path = (
                args.dataset_root
                / "videos"
                / f"chunk-{episode // 1000:03d}"
                / f"observation.images.{camera}"
                / f"episode_{episode:06d}.mp4"
            )
            observation[f"observation/{camera}"] = _read_history(video_path)

        predicted = np.asarray(
            client.infer(observation), dtype=np.float32
        )
        if predicted.shape != (24, 16):
            raise ValueError(
                f"Episode {episode}: expected (24,16), got {predicted.shape}"
            )
        record = {
            "episode": episode,
            **_arm_metrics(
                predicted,
                states[row],
                actions[row : row + 24],
            ),
        }
        records.append(record)
        logging.info(
            "episode=%d dominant=%s/%s correct=%s pred_LR=(%.4f,%.4f) "
            "gt_LR=(%.4f,%.4f)",
            episode,
            record["pred_dominant_arm"],
            record["gt_dominant_arm"],
            record["dominant_arm_correct"],
            record["pred_motion_left_rad"],
            record["pred_motion_right_rad"],
            record["gt_motion_left_rad"],
            record["gt_motion_right_rad"],
        )

    if not records:
        raise RuntimeError("No episodes were evaluated")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_episode.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(records, stream, ensure_ascii=False, indent=2)

    ratios = [
        float(record["motion_ratio_on_gt_arm"])
        for record in records
        if record["motion_ratio_on_gt_arm"] is not None
    ]
    summary = {
        "checkpoint_server": f"{args.host}:{args.port}",
        "dataset_root": str(args.dataset_root),
        "episodes": len(records),
        "dominant_arm_accuracy": float(
            np.mean([record["dominant_arm_correct"] for record in records])
        ),
        "median_motion_ratio_on_gt_arm": float(np.median(ratios)),
        "mean_motion_ratio_on_gt_arm": float(np.mean(ratios)),
        "mean_mae_left_rad": float(
            np.mean([record["mae_left_rad"] for record in records])
        ),
        "mean_mae_right_rad": float(
            np.mean([record["mae_right_rad"] for record in records])
        ),
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
