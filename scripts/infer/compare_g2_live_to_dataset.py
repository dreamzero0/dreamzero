"""Rank G2 Gear episode starts by similarity to one live camera snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("top_head", "hand_left", "hand_right")


def _first_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode first frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _image_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        candidate = cv2.resize(
            candidate,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    # Low-frequency comparison emphasizes camera framing and object layout over
    # JPEG noise and small illumination changes.
    reference_small = cv2.resize(reference, (80, 44), interpolation=cv2.INTER_AREA)
    candidate_small = cv2.resize(candidate, (80, 44), interpolation=cv2.INTER_AREA)
    return float(
        np.mean(
            np.abs(
                reference_small.astype(np.float32)
                - candidate_small.astype(np.float32)
            )
        )
        / 255.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    snapshot = np.load(args.snapshot, allow_pickle=False)
    live = {
        camera: np.asarray(
            snapshot[f"{camera}_jpeg_decoded"][-1], dtype=np.uint8
        )
        for camera in CAMERAS
    }
    live_state = np.asarray(snapshot["state_16"], dtype=np.float32)

    with (args.dataset_root / "meta/info.json").open(
        "r", encoding="utf-8"
    ) as stream:
        info = json.load(stream)
    total = int(info["total_episodes"])

    results: list[dict[str, object]] = []
    cached_frames: dict[int, dict[str, np.ndarray]] = {}
    for episode in range(total):
        frames: dict[str, np.ndarray] = {}
        view_scores: dict[str, float] = {}
        for camera in CAMERAS:
            path = (
                args.dataset_root
                / "videos"
                / f"chunk-{episode // 1000:03d}"
                / f"observation.images.{camera}"
                / f"episode_{episode:06d}.mp4"
            )
            frames[camera] = _first_rgb(path)
            view_scores[camera] = _image_score(live[camera], frames[camera])

        parquet = (
            args.dataset_root
            / "data"
            / f"chunk-{episode // 1000:03d}"
            / f"episode_{episode:06d}.parquet"
        )
        state0 = np.asarray(
            pq.read_table(
                parquet, columns=["observation.state"]
            )["observation.state"][0].as_py(),
            dtype=np.float32,
        )
        arm_indices = np.r_[0:7, 8:15]
        state_arm_mae = float(
            np.mean(np.abs(state0[arm_indices] - live_state[arm_indices]))
        )
        results.append(
            {
                "episode": episode,
                "image_score_mean": float(np.mean(list(view_scores.values()))),
                "view_scores": view_scores,
                "state_arm_mae_rad": state_arm_mae,
                "dataset_grippers": state0[[7, 15]].tolist(),
                "live_grippers": live_state[[7, 15]].tolist(),
            }
        )
        cached_frames[episode] = frames

    results.sort(key=lambda item: float(item["image_score_mean"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ranking.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)

    # One readable sheet per top match: dataset row followed by live row.
    for result in results[: args.top_k]:
        episode = int(result["episode"])
        rows = [
            np.concatenate([cached_frames[episode][camera] for camera in CAMERAS], axis=1),
            np.concatenate([live[camera] for camera in CAMERAS], axis=1),
        ]
        rgb = np.concatenate(rows, axis=0)
        cv2.imwrite(
            str(args.output_dir / f"episode_{episode:06d}_dataset_vs_live.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

    print(json.dumps(results[: args.top_k], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
