#!/usr/bin/env python3
"""Teacher-forced full-episode rollout against one persistent G2 server.

The model server is deliberately not started here.  This client connects to
the already-loaded checkpoint, sends one 4-frame observation packet every
``stride`` frames, and records every returned 24x16 action chunk against the
dataset action horizon at the same anchor.  At the end it sends one reset
message so the server writes its accumulated predicted video.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

from eval_utils.policy_client import WebsocketClientPolicy


ACTION_NAMES = (
    [f"left_joint_{i}" for i in range(7)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(7)]
    + ["right_gripper"]
)


def _episode_path(root: Path, template: str, episode: int, chunks_size: int, **kwargs: object) -> Path:
    return root / template.format(
        episode_chunk=episode // chunks_size,
        episode_index=episode,
        **kwargs,
    )


def _read_video(path: Path, expected_rows: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open G2 video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # The online G2 client sends camera arrays in this BGR convention;
            # keep it identical to the existing dataset evaluator.
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if len(frames) < expected_rows:
        raise RuntimeError(
            f"{path} contains {len(frames)} frames, expected at least {expected_rows}"
        )
    return np.stack(frames[:expected_rows], axis=0)


def _grid_rgb(top_bgr: np.ndarray, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    top = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    black = np.zeros_like(top)
    return np.concatenate(
        [np.concatenate([top, left], axis=1), np.concatenate([right, black], axis=1)],
        axis=0,
    )


def _label(frame_rgb: np.ndarray, text: str) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame_bgr, (0, 0), (360, 34), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        raise RuntimeError(f"No frames to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", macro_block_size=None)


def _metric(error: np.ndarray) -> dict[str, float | int]:
    arm = error[..., [*range(7), *range(8, 15)]]
    grip = error[..., [7, 15]]
    return {
        "overall_mae": float(error.mean()),
        "arm_mae": float(arm.mean()),
        "gripper_mae": float(grip.mean()),
        "left_arm_mae": float(error[..., :7].mean()),
        "right_arm_mae": float(error[..., 8:15].mean()),
        "num_action_rows": int(np.prod(error.shape[:-1])),
    }


def _write_action_reports(
    output_dir: Path,
    episode: int,
    anchors: list[int],
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, object]:
    error = np.abs(predicted - ground_truth)
    detail_path = output_dir / f"episode_{episode:06d}_full_action_pred_vs_gt.csv"
    fields = ["block_index", "anchor_frame", "horizon_step"]
    fields += [f"gt_{name}" for name in ACTION_NAMES]
    fields += [f"pred_{name}" for name in ACTION_NAMES]
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for block_index, anchor in enumerate(anchors):
            for horizon in range(predicted.shape[1]):
                gt_row = ground_truth[block_index, horizon]
                pred_row = predicted[block_index, horizon]
                row: dict[str, object] = {
                    "block_index": block_index,
                    "anchor_frame": anchor,
                    "horizon_step": horizon + 1,
                }
                row.update({f"gt_{name}": float(v) for name, v in zip(ACTION_NAMES, gt_row)})
                row.update({f"pred_{name}": float(v) for name, v in zip(ACTION_NAMES, pred_row)})
                writer.writerow(row)

    metrics_path = output_dir / f"episode_{episode:06d}_full_action_metrics.csv"
    metric_fields = [
        "scope",
        "block_index",
        "anchor_frame",
        "overall_mae",
        "arm_mae",
        "gripper_mae",
        "left_arm_mae",
        "right_arm_mae",
        "num_action_rows",
    ]
    rows: list[dict[str, object]] = []
    for block_index, anchor in enumerate(anchors):
        row: dict[str, object] = _metric(error[block_index][None, ...])
        row.update({
            "scope": f"block_{block_index:04d}",
            "block_index": block_index,
            "anchor_frame": anchor,
        })
        rows.append(row)
    overall: dict[str, object] = _metric(error)
    overall.update({"scope": "overall", "block_index": "", "anchor_frame": ""})
    rows.append(overall)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / f"episode_{episode:06d}_full_action_pred_vs_gt.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(predicted.shape[0])
        first_pred = predicted[:, 0]
        first_gt = ground_truth[:, 0]
        first_error = np.abs(first_pred - first_gt)
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        axes[0].plot(x, first_gt[:, :7].mean(axis=1), label="GT left arm", color="tab:blue")
        axes[0].plot(x, first_pred[:, :7].mean(axis=1), label="PRED left arm", color="tab:orange")
        axes[0].set_ylabel("left arm mean")
        axes[0].legend()
        axes[1].plot(x, first_gt[:, 8:15].mean(axis=1), label="GT right arm", color="tab:green")
        axes[1].plot(x, first_pred[:, 8:15].mean(axis=1), label="PRED right arm", color="tab:red")
        axes[1].set_ylabel("right arm mean")
        axes[1].legend()
        axes[2].plot(x, first_error[:, [*range(7), *range(8, 15)]].mean(axis=1), label="arm MAE")
        axes[2].plot(x, first_error[:, [7, 15]].mean(axis=1), label="gripper MAE")
        axes[2].set_xlabel("teacher-forced inference block (4 frames per block)")
        axes[2].set_ylabel("absolute error")
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle(f"G2 full-task first-action comparison, episode {episode}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logging.warning("Could not save action plot: %s", exc)
        plot_path = None

    return {
        "action_detail_csv": str(detail_path),
        "action_metrics_csv": str(metrics_path),
        "action_plot": str(plot_path) if plot_path else None,
        "overall_action_metrics": overall,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--test-data-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--server-video-dir",
        type=Path,
        required=True,
        help="Persistent server's VIDEO_SAVE_MODE=full output directory.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--block-stride", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.block_stride < 1:
        raise ValueError("--block-stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.test_data_root.resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    episode = int(args.episode_index)
    parquet_path = _episode_path(
        root,
        info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
        episode,
        chunks_size,
    )
    table = pq.read_table(parquet_path)
    # PyArrow returns an object array for the nested fixed-size columns; use
    # Python rows before converting so the 16-D vectors are not treated as
    # scalar objects.
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = state.reshape(table.num_rows, 16)
    action = action.reshape(table.num_rows, 16)
    num_rows = int(table.num_rows)
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    videos = {}
    for name, key in (("top_head", "observation.images.top_head"), ("hand_left", "observation.images.hand_left"), ("hand_right", "observation.images.hand_right")):
        path = _episode_path(root, video_template, episode, chunks_size, video_key=key)
        videos[name] = _read_video(path, num_rows)

    first_block = max(0, int(args.start_frame))
    last_anchor = num_rows - 24
    if args.end_frame >= 0:
        last_anchor = min(last_anchor, int(args.end_frame))
    # A packet starts at frame s and is anchored at s+3.  Keep a complete
    # 24-step GT horizon for every request.
    starts = list(range(first_block, max(first_block, last_anchor - 3) + 1, args.block_stride))
    if args.max_blocks is not None:
        starts = starts[: max(0, int(args.max_blocks))]
    if not starts:
        raise RuntimeError("No valid full-task inference blocks")

    prompt_key = "annotation.language.action_text"
    default_prompt = ""
    if prompt_key in table.column_names:
        default_prompt = str(table[prompt_key][0].as_py())
    prompt = args.prompt if args.prompt is not None else default_prompt
    session_id = f"g2-full-task-episode-{episode:06d}-{uuid.uuid4()}"

    logging.info(
        "Connecting to persistent server %s:%s; episode=%d rows=%d blocks=%d start=%d end=%d stride=%d",
        args.host,
        args.port,
        episode,
        num_rows,
        len(starts),
        starts[0],
        starts[-1],
        args.block_stride,
    )
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())

    predicted_chunks: list[np.ndarray] = []
    anchors: list[int] = []
    started = time.time()
    try:
        for block_index, packet_start in enumerate(starts):
            anchor = packet_start + 3
            observation = {
                "observation/top_head": videos["top_head"][packet_start : packet_start + 4],
                "observation/hand_left": videos["hand_left"][packet_start : packet_start + 4],
                "observation/hand_right": videos["hand_right"][packet_start : packet_start + 4],
                "observation/state": state[anchor],
                "prompt": prompt,
                "session_id": session_id,
            }
            result = np.asarray(client.infer(observation), dtype=np.float32)
            if result.shape != (24, 16):
                raise RuntimeError(
                    f"server block {block_index} returned {result.shape}, expected (24,16)"
                )
            predicted_chunks.append(result)
            anchors.append(anchor)
            if block_index == 0 or (block_index + 1) % 10 == 0 or block_index + 1 == len(starts):
                logging.info(
                    "full-task block %d/%d anchor=%d elapsed=%.1fs first_arm=[%.4f,%.4f,%.4f]",
                    block_index + 1,
                    len(starts),
                    anchor,
                    time.time() - started,
                    result[0, 0],
                    result[0, 1],
                    result[0, 2],
                )
    finally:
        # reset makes the persistent server flush its accumulated predicted
        # video; it does not unload the checkpoint or terminate the server.
        try:
            client.reset({"session_id": session_id})
        except Exception:
            logging.exception("Server reset/video flush failed")

    predicted = np.stack(predicted_chunks, axis=0)
    ground_truth = np.stack(
        [action[anchor : anchor + 24] for anchor in anchors],
        axis=0,
    )
    reports = _write_action_reports(
        args.output_dir,
        episode,
        anchors,
        predicted,
        ground_truth,
    )
    np.savez_compressed(
        args.output_dir / f"episode_{episode:06d}_full_action_arrays.npz",
        predicted=predicted,
        ground_truth=ground_truth,
        anchors=np.asarray(anchors, dtype=np.int32),
    )

    # The policy is queried every four real frames.  Use exactly those four
    # frames per block for the comparison timeline; comparing the server's
    # raw concatenated latent frames 1:1 against the 30-Hz GT makes the
    # prediction look artificially fast (one block decodes 9/12 frames while
    # the real observation stream advances only four frames).
    gt_rollout_frames: list[np.ndarray] = []
    for anchor in anchors:
        gt_rollout_frames.extend(
            _grid_rgb(
                videos["top_head"][index],
                videos["hand_left"][index],
                videos["hand_right"][index],
            )
            for index in range(anchor, anchor + 4)
        )
    gt_path = args.output_dir / f"episode_{episode:06d}_ground_truth_rollout_aligned.mp4"
    _save_video(gt_path, gt_rollout_frames)

    # The server's reset flushes its generated video under its configured
    # output directory.  Copy the latest file into this report directory and
    # build the side-by-side video over the common frame range.
    server_video_dir = args.server_video_dir.resolve()
    server_videos = sorted(server_video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    predicted_video_path = server_videos[-1] if server_videos else None
    comparison_path = None
    aligned_predicted_frames = 0
    raw_predicted_frames = 0
    if predicted_video_path is not None:
        predicted_reader = imageio.get_reader(predicted_video_path)
        raw_predicted = [np.asarray(frame) for frame in predicted_reader]
        predicted_reader.close()
        raw_predicted_frames = len(raw_predicted)
        if not raw_predicted:
            raise RuntimeError(f"Server video has no decoded frames: {predicted_video_path}")
        target_frames = len(gt_rollout_frames)
        if raw_predicted_frames == target_frames:
            aligned_predicted = raw_predicted
        else:
            # Preserve the complete predicted trajectory but put it on the
            # real 4-frames-per-policy-block clock.
            indices = np.rint(
                np.linspace(0, raw_predicted_frames - 1, target_frames)
            ).astype(np.int64)
            aligned_predicted = [raw_predicted[int(index)] for index in indices]
        aligned_predicted_frames = len(aligned_predicted)
        comparison_path = args.output_dir / f"episode_{episode:06d}_predicted_vs_ground_truth_full.mp4"
        writer = imageio.get_writer(comparison_path, fps=30, codec="libx264", macro_block_size=None)
        try:
            for pred_rgb, gt_rgb in zip(aligned_predicted, gt_rollout_frames):
                writer.append_data(
                    np.concatenate(
                        [_label(pred_rgb, "PREDICTED"), _label(gt_rgb, "G2 GROUND TRUTH")],
                        axis=1,
                    )
                )
        finally:
            writer.close()
        logging.info(
            "Saved time-aligned comparison video raw_pred=%d aligned=%d gt=%d: %s",
            raw_predicted_frames,
            aligned_predicted_frames,
            len(gt_rollout_frames),
            comparison_path,
        )

    report = {
        "episode_index": episode,
        "num_rows": num_rows,
        "num_blocks": len(starts),
        "block_stride": args.block_stride,
        "anchors": anchors,
        "session_id": session_id,
        "server_video": str(predicted_video_path) if predicted_video_path else None,
        "server_video_raw_frames": raw_predicted_frames,
        "comparison_video_aligned_frames": aligned_predicted_frames,
        "video_alignment": "4 real frames per teacher-forced policy block; predicted raw latent video uniformly resampled",
        "ground_truth_video": str(gt_path),
        "comparison_video": str(comparison_path) if comparison_path else None,
        **reports,
    }
    report_path = args.output_dir / f"episode_{episode:06d}_full_task_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved full-task report: %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
