#!/usr/bin/env python3
"""Teacher-forced DreamZero G2 video/action window evaluation.

The persistent server must be started with ``--no-reset-cache-each-request``.
Each evaluation window sends four consecutive 4-frame observations through the
same causal cache.  DreamZero then decodes one 33-frame video window.  The
matching GT segment is taken from the same episode timeline and is stitched
side-by-side at 30 Hz; no global frame-rate resampling is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import uuid
from pathlib import Path

# Keep evaluation code isolated under scripts/eval while importing the shared
# websocket client from the repository root.  This file never touches the
# robot deployment client.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
ARM_DIMS = [*range(7), *range(8, 15)]
GRIPPER_DIMS = [7, 15]


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
        while len(frames) < expected_rows:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if len(frames) < expected_rows:
        raise RuntimeError(
            f"{path} contains {len(frames)} frames, expected {expected_rows}"
        )
    return np.stack(frames, axis=0)


def _encode_video_observation(frames: np.ndarray, quality: int = 80) -> dict[str, object]:
    """Encode BGR frames exactly like the live G2 JPEG transport.

    The live client captures BGR, JPEG-encodes it, and the G2 server decodes
    then converts BGR to RGB before DreamZero sees it.  Sending cv2's raw BGR
    array would silently create a different offline evaluation condition.
    """
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected video frames with shape (T,H,W,3), got {array.shape}")
    encoded: list[bytes] = []
    jpeg_quality = int(np.clip(quality, 1, 100))
    for frame in array:
        ok, payload = cv2.imencode(
            ".jpg", np.ascontiguousarray(frame),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode evaluation frame")
        encoded.append(payload.tobytes())
    return {
        "__dreamzero_image_encoding__": "jpeg_sequence",
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "quality": jpeg_quality,
        "frames": encoded,
    }


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
    cv2.rectangle(frame_bgr, (0, 0), (370, 34), (0, 0, 0), -1)
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


def _find_new_server_video(server_dir: Path, before: set[str], timeout: float = 60.0) -> Path:
    deadline = time.time() + timeout
    while time.time() < deadline:
        candidates = [
            p for p in server_dir.glob("*.mp4")
            if p.name not in before and p.stat().st_size > 0
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime_ns)
        time.sleep(0.25)
    raise RuntimeError(
        f"Server reset did not create a new video in {server_dir}; "
        f"existing={sorted(before)[-3:]}"
    )


def _read_server_video(path: Path, expected_frames: int) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"Expected exactly {expected_frames} decoded prediction frames from "
            f"the causal window, got {len(frames)} ({path.name}). "
            "Do not resample this diagnostic: check server cache mode and VAE decode."
        )
    return [np.ascontiguousarray(frame) for frame in frames]


def _write_action_reports(
    output_dir: Path,
    episode: int,
    window_starts: list[int],
    anchors: list[int],
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, object]:
    # Shapes: [window, causal_block, horizon, action_dim].
    if predicted.shape != ground_truth.shape or predicted.ndim != 4:
        raise ValueError(f"Action arrays must match 4-D shape, got {predicted.shape} and {ground_truth.shape}")
    error = np.abs(predicted - ground_truth)
    detail_path = output_dir / f"episode_{episode:06d}_windowed_action_pred_vs_gt.csv"
    fields = ["window_index", "window_start", "block_index", "anchor_frame", "horizon_step"]
    fields += [f"gt_{name}" for name in ACTION_NAMES]
    fields += [f"pred_{name}" for name in ACTION_NAMES]
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for wi, start in enumerate(window_starts):
            for bi in range(predicted.shape[1]):
                anchor = anchors[wi * predicted.shape[1] + bi]
                for hi in range(predicted.shape[2]):
                    row: dict[str, object] = {
                        "window_index": wi,
                        "window_start": start,
                        "block_index": bi,
                        "anchor_frame": anchor,
                        "horizon_step": hi + 1,
                    }
                    row.update({f"gt_{name}": float(v) for name, v in zip(ACTION_NAMES, ground_truth[wi, bi, hi])})
                    row.update({f"pred_{name}": float(v) for name, v in zip(ACTION_NAMES, predicted[wi, bi, hi])})
                    writer.writerow(row)

    metrics_path = output_dir / f"episode_{episode:06d}_windowed_action_metrics.csv"
    metric_fields = ["scope", "window_index", "window_start", "block_index", "anchor_frame", "overall_mae", "arm_mae", "gripper_mae", "left_arm_mae", "right_arm_mae", "num_action_rows"]

    def metric_row(scope: str, e: np.ndarray, **extra: object) -> dict[str, object]:
        row: dict[str, object] = {
            "scope": scope,
            "overall_mae": float(e.mean()),
            "arm_mae": float(e[..., ARM_DIMS].mean()),
            "gripper_mae": float(e[..., GRIPPER_DIMS].mean()),
            "left_arm_mae": float(e[..., :7].mean()),
            "right_arm_mae": float(e[..., 8:15].mean()),
            "num_action_rows": int(np.prod(e.shape[:-1])),
        }
        row.update(extra)
        return row

    metric_rows: list[dict[str, object]] = []
    num_blocks = predicted.shape[1]
    for wi, start in enumerate(window_starts):
        for bi in range(num_blocks):
            metric_rows.append(
                metric_row(
                    f"window_{wi:04d}_block_{bi:02d}",
                    error[wi, bi],
                    window_index=wi,
                    window_start=start,
                    block_index=bi,
                    anchor_frame=anchors[wi * num_blocks + bi],
                )
            )
    overall = metric_row("overall", error, window_index="", window_start="", block_index="", anchor_frame="")
    metric_rows.append(overall)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    # One 16-dimension plot over the complete sampled task.  Each action row
    # is a predicted 24-step horizon at one causal observation anchor.
    pred_flat = predicted.reshape(-1, predicted.shape[-1])
    gt_flat = ground_truth.reshape(-1, ground_truth.shape[-1])
    plot_path = output_dir / f"episode_{episode:06d}_windowed_action_pred_vs_gt.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 4, figsize=(22, 13), sharex=True)
        x = np.arange(len(pred_flat))
        for dim, axis in enumerate(axes.flat):
            axis.plot(x, gt_flat[:, dim], color="tab:blue", linewidth=0.8, label="GT")
            axis.plot(x, pred_flat[:, dim], color="tab:orange", linewidth=0.8, alpha=0.9, label="PRED")
            axis.set_title(f"{dim}: {ACTION_NAMES[dim]}")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(loc="upper right")
        axes[-1, 0].set_xlabel("window/block/horizon action row")
        axes[-1, 1].set_xlabel("GT and predicted action")
        fig.suptitle(f"G2 checkpoint-3000: full windowed action vs GT, episode {episode}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

        mae_path = output_dir / f"episode_{episode:06d}_windowed_action_mae_by_horizon.png"
        horizon_mae = error.mean(axis=(0, 1, 3))
        arm_mae = error[..., ARM_DIMS].mean(axis=(0, 1, 3))
        grip_mae = error[..., GRIPPER_DIMS].mean(axis=(0, 1, 3))
        fig, axis = plt.subplots(figsize=(12, 5))
        h = np.arange(1, error.shape[2] + 1)
        axis.plot(h, horizon_mae, label="all 16 dims", linewidth=2)
        axis.plot(h, arm_mae, label="arm 14 dims", linewidth=2)
        axis.plot(h, grip_mae, label="gripper 2 dims", linewidth=2)
        axis.set_xlabel("action horizon step")
        axis.set_ylabel("mean absolute error")
        axis.set_title("Predicted action vs training GT: error by horizon step")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(mae_path, dpi=140)
        plt.close(fig)
    except Exception as exc:  # diagnostic output should not hide CSV results
        logging.warning("Could not create action plots: %s", exc)
        plot_path = None
        mae_path = None

    return {
        "action_detail_csv": str(detail_path),
        "action_metrics_csv": str(metrics_path),
        "action_plot": str(plot_path) if plot_path else None,
        "action_horizon_mae_plot": str(mae_path) if mae_path else None,
        "overall_action_metrics": overall,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--test-data-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-video-dir", type=Path, required=True)
    parser.add_argument("--window-future-frames", type=int, default=33)
    parser.add_argument("--window-history", type=int, default=4)
    parser.add_argument("--causal-blocks", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=33)
    parser.add_argument("--image-jpeg-quality", type=int, default=80)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()
    if args.window_history != 4 or args.causal_blocks != 4:
        raise ValueError("This protocol requires four consecutive 4-frame causal packets")
    if args.window_future_frames != 33:
        raise ValueError("This protocol requires exactly a 33-frame decoded video window")
    if args.window_stride < 1:
        raise ValueError("--window-stride must be positive")

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
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32).reshape(table.num_rows, 16)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32).reshape(table.num_rows, 16)
    num_rows = int(table.num_rows)
    video_template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    videos: dict[str, np.ndarray] = {}
    for name, key in (("top_head", "observation.images.top_head"), ("hand_left", "observation.images.hand_left"), ("hand_right", "observation.images.hand_right")):
        videos[name] = _read_video(_episode_path(root, video_template, episode, chunks_size, video_key=key), num_rows)

    # A window starts with the four-frame packet [start:start+4].  The video
    # timeline used for comparison begins at the first packet anchor
    # (start+3) and contains 33 frames, i.e. rows start+3..start+35.  The
    # four causal packets are context for that same request; they are not
    # appended to the future-video duration.  The old bound added the 16
    # context frames to the 33 future frames and silently dropped the tail of
    # every episode.
    max_start = num_rows - (args.window_history + args.window_future_frames - 1)
    starts = list(range(0, max_start + 1, args.window_stride))
    if starts[-1] < max_start:
        starts.append(max_start)
    if args.max_windows is not None:
        starts = starts[: max(0, int(args.max_windows))]
    if not starts:
        raise RuntimeError(f"No valid 33-frame windows in episode with {num_rows} rows")
    prompt_key = "annotation.language.action_text"
    prompt = args.prompt if args.prompt is not None else (str(table[prompt_key][0].as_py()) if prompt_key in table.column_names else "")
    session_id = f"g2-windowed-episode-{episode:06d}-{uuid.uuid4()}"
    server_dir = args.server_video_dir.resolve()
    server_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Windowed rollout: episode=%d rows=%d windows=%d stride=%d history=%d causal_blocks=%d future=%d",
        episode, num_rows, len(starts), args.window_stride, args.window_history, args.causal_blocks, args.window_future_frames,
    )
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())
    # Clear any stale server-side cache/video from an earlier client session.
    try:
        client.reset({"session_id": session_id})
    except Exception:
        logging.exception("Initial server reset failed")

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_anchors: list[int] = []
    all_comparison_frames: list[np.ndarray] = []
    window_reports: list[dict[str, object]] = []
    action_window_starts: list[int] = []
    started = time.time()
    for wi, window_start in enumerate(starts):
        before = {p.name for p in server_dir.glob("*.mp4")}
        pred_chunks: list[np.ndarray] = []
        gt_chunks: list[np.ndarray] = []
        anchors_window: list[int] = []
        for block_index in range(args.causal_blocks):
            packet_start = window_start + block_index * args.window_history
            indices = list(range(packet_start, packet_start + args.window_history))
            anchor = indices[-1]
            result = np.asarray(
                client.infer(
                    {
                        "observation/top_head": _encode_video_observation(
                            videos["top_head"][indices], args.image_jpeg_quality
                        ),
                        "observation/hand_left": _encode_video_observation(
                            videos["hand_left"][indices], args.image_jpeg_quality
                        ),
                        "observation/hand_right": _encode_video_observation(
                            videos["hand_right"][indices], args.image_jpeg_quality
                        ),
                        "observation/state": state[anchor],
                        "prompt": prompt,
                        "session_id": session_id,
                    }
                ),
                dtype=np.float32,
            )
            if result.shape != (24, 16):
                raise RuntimeError(f"window {wi} block {block_index}: server returned {result.shape}, expected (24,16)")
            pred_chunks.append(result)
            gt_chunks.append(action[anchor : anchor + 24])
            anchors_window.append(anchor)
        # Reset flushes exactly this four-forward causal video window; it does
        # not unload the server/checkpoint.
        client.reset({"session_id": session_id})
        predicted_video_path = _find_new_server_video(server_dir, before)
        predicted_frames = _read_server_video(predicted_video_path, args.window_future_frames)
        # The server has to save this temporary causal decode so that the
        # client can read it, but it is not an evaluation artifact.  Keep only
        # the single stitched episode video under args.output_dir instead of
        # leaving one 33-frame mp4 per causal window in the server directory.
        try:
            predicted_video_path.unlink()
        except OSError as exc:
            logging.warning("Could not remove temporary server video %s: %s", predicted_video_path, exc)
        gt_indices = range(window_start + 3, window_start + 3 + args.window_future_frames)
        gt_frames = [
            _grid_rgb(videos["top_head"][idx], videos["hand_left"][idx], videos["hand_right"][idx])
            for idx in gt_indices
        ]
        comparison_frames = [
            np.concatenate([_label(pred, "PREDICTED"), _label(gt, "G2 GROUND TRUTH")], axis=1)
            for pred, gt in zip(predicted_frames, gt_frames)
        ]
        # Avoid duplicate frames when a final tail window was appended to
        # cover an episode whose length is not an exact multiple of the
        # requested stride.  The GT index is the authoritative timeline.
        gt_start = window_start + 3
        trim = max(0, len(all_comparison_frames) + 3 - gt_start)
        all_comparison_frames.extend(comparison_frames[trim:])
        # A video tail can be valid while its last causal packet has fewer
        # than 24 GT action rows remaining.  Keep that video window, but do
        # not mix a short action slice into the fixed [24,16] action tensor;
        # the dedicated full-action pass below evaluates every valid anchor.
        action_valid = all(chunk.shape == (24, 16) for chunk in gt_chunks)
        action_mae: float | None = None
        if action_valid:
            pred_array = np.stack(pred_chunks)
            gt_array = np.stack(gt_chunks)
            all_pred.append(pred_array)
            all_gt.append(gt_array)
            all_anchors.extend(anchors_window)
            action_window_starts.append(window_start)
            action_mae = float(np.abs(pred_array - gt_array).mean())
        window_reports.append(
            {
                "window_index": wi,
                "window_start": window_start,
                "packet_starts": [window_start + args.window_history * i for i in range(args.causal_blocks)],
                "anchor_indices": anchors_window,
                "gt_video_indices": [window_start + 3, window_start + 3 + args.window_future_frames - 1],
                "predicted_video": str(predicted_video_path),
                "predicted_frames": len(predicted_frames),
                "ground_truth_frames": len(gt_frames),
                "action_mae": action_mae,
            }
        )
        logging.info(
            "window %d/%d start=%d video=%d frames action_mae=%s elapsed=%.1fs",
            wi + 1, len(starts), window_start, len(predicted_frames),
            f"{action_mae:.6f}" if action_mae is not None else "n/a",
            time.time() - started,
        )

    if not all_pred:
        raise RuntimeError("No complete 24-step action windows remained for action reporting")
    predicted = np.stack(all_pred)
    ground_truth = np.stack(all_gt)
    np.savez_compressed(
        args.output_dir / f"episode_{episode:06d}_windowed_action_arrays.npz",
        predicted=predicted,
        ground_truth=ground_truth,
        anchors=np.asarray(all_anchors, dtype=np.int32),
        window_starts=np.asarray(action_window_starts, dtype=np.int32),
    )
    comparison_path = args.output_dir / f"episode_{episode:06d}_windowed_predicted_vs_gt_f{len(all_comparison_frames)}.mp4"
    _save_video(comparison_path, all_comparison_frames, fps=30)
    reports = _write_action_reports(args.output_dir, episode, action_window_starts, all_anchors, predicted, ground_truth)
    report = {
        "episode_index": episode,
        "checkpoint_server_protocol": "four 4-frame history packets -> one causal 33-frame decoded window; reset only between windows",
        "server_cache_required": "DREAMZERO_RESET_AR_EACH_REQUEST=false",
        "num_rows": num_rows,
        "video_window_starts": starts,
        "action_window_starts": action_window_starts,
        "window_stride": args.window_stride,
        "window_history": args.window_history,
        "causal_blocks": args.causal_blocks,
        "future_frames": args.window_future_frames,
        "comparison_fps": 30,
        "comparison_video": str(comparison_path),
        "window_reports": window_reports,
        **reports,
    }
    report_path = args.output_dir / f"episode_{episode:06d}_windowed_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved windowed comparison: %s", comparison_path)
    logging.info("Saved windowed report: %s", report_path)


if __name__ == "__main__":
    main()
