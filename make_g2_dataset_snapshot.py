"""Build a live-replay-compatible snapshot from a G2 Gear episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("top_head", "hand_left", "hand_right")


def _read_video_prefix(path: Path, end_row: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) <= end_row:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read frame {len(frames)} from {path}"
                )
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    start = max(0, end_row - 3)
    history = frames[start : end_row + 1]
    while len(history) < 4:
        history.insert(0, history[0])
    return np.stack(history, axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunk = args.episode // 1000
    parquet = (
        args.root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{args.episode:06d}.parquet"
    )
    table = pq.read_table(parquet)
    states = np.asarray(
        table["observation.state"].to_pylist(), dtype=np.float32
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if not 0 <= args.row < len(states):
        raise IndexError(f"row {args.row} is outside episode length {len(states)}")
    if args.row + 24 > len(actions):
        raise IndexError("Not enough future actions for a 24-step horizon")

    videos: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        video_path = (
            args.root
            / "videos"
            / f"chunk-{chunk:03d}"
            / f"observation.images.{camera}"
            / f"episode_{args.episode:06d}.mp4"
        )
        videos[camera] = _read_video_prefix(video_path, args.row)

    prompt = str(
        table["annotation.language.action_text"][args.row].as_py()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        top_head_jpeg_decoded=videos["top_head"],
        hand_left_jpeg_decoded=videos["hand_left"],
        hand_right_jpeg_decoded=videos["hand_right"],
        state_16=states[args.row],
        raw_server_actions=np.zeros((24, 16), dtype=np.float32),
        ground_truth_actions=actions[args.row : args.row + 24],
        prompt=np.asarray(prompt),
        episode=np.asarray(args.episode, dtype=np.int32),
        row=np.asarray(args.row, dtype=np.int32),
    )
    print(args.output)


if __name__ == "__main__":
    main()
