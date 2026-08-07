#!/usr/bin/env python3
"""Full-episode G2 action-only evaluation against every 4-frame anchor.

This is deliberately separate from robot deployment.  It sends the same
4-frame teacher-forced observation packets used by the live client, keeps four
causal packets per server cache window, and records every returned 24x16
action chunk against the training episode GT.  Video files are ignored; the
result is the complete 16-D action table/curve requested for the task.
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

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from scripts.eval.rollout_g2_server_windowed import (  # noqa: E402
    ACTION_NAMES,
    ARM_DIMS,
    GRIPPER_DIMS,
    _episode_path,
    _encode_video_observation,
    _read_video,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30002)
    p.add_argument("--test-data-root", type=Path, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--server-video-dir",
        type=Path,
        default=None,
        help="Temporary server video directory; remove files flushed by this action-only run.",
    )
    p.add_argument("--window-blocks", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--image-jpeg-quality", type=int, default=80)
    return p.parse_args()


def _remove_new_server_videos(server_dir: Path | None, before: set[str]) -> None:
    """Drop per-reset video files; this pass evaluates actions only."""
    if server_dir is None:
        return
    try:
        for path in server_dir.glob("*.mp4"):
            if path.name not in before:
                try:
                    path.unlink()
                except OSError as exc:
                    logging.warning("Could not remove temporary server video %s: %s", path, exc)
    except OSError as exc:
        logging.warning("Could not scan temporary server video directory %s: %s", server_dir, exc)


def _write_reports(out: Path, episode: int, starts: list[int], anchors: list[int], pred: np.ndarray, gt: np.ndarray) -> None:
    err = np.abs(pred - gt)
    np.savez_compressed(out / f"episode_{episode:06d}_full_action_arrays.npz", predicted=pred, ground_truth=gt, packet_starts=np.asarray(starts), anchors=np.asarray(anchors))
    detail = out / f"episode_{episode:06d}_full_action_pred_vs_gt.csv"
    fields = ["block_index", "packet_start", "anchor_frame", "horizon_step"] + [f"gt_{x}" for x in ACTION_NAMES] + [f"pred_{x}" for x in ACTION_NAMES]
    with detail.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for bi, (start, anchor) in enumerate(zip(starts, anchors)):
            for hi in range(24):
                row = {"block_index": bi, "packet_start": start, "anchor_frame": anchor, "horizon_step": hi + 1}
                row.update({f"gt_{n}": float(v) for n, v in zip(ACTION_NAMES, gt[bi, hi])})
                row.update({f"pred_{n}": float(v) for n, v in zip(ACTION_NAMES, pred[bi, hi])})
                w.writerow(row)

    def metric(scope: str, e: np.ndarray) -> dict[str, object]:
        return {"scope": scope, "overall_mae": float(e.mean()), "arm_mae": float(e[..., ARM_DIMS].mean()), "gripper_mae": float(e[..., GRIPPER_DIMS].mean()), "left_arm_mae": float(e[..., :7].mean()), "right_arm_mae": float(e[..., 8:15].mean()), "num_action_rows": int(np.prod(e.shape[:-1]))}

    metrics = out / f"episode_{episode:06d}_full_action_metrics.csv"
    rows = [metric(f"block_{i:04d}", err[i : i + 1]) | {"block_index": i, "packet_start": starts[i], "anchor_frame": anchors[i]} for i in range(len(pred))]
    overall = metric("overall", err) | {"block_index": "", "packet_start": "", "anchor_frame": ""}
    rows.append(overall)
    mf = ["scope", "block_index", "packet_start", "anchor_frame", "overall_mae", "arm_mae", "gripper_mae", "left_arm_mae", "right_arm_mae", "num_action_rows"]
    with metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mf); w.writeheader(); w.writerows(rows)

    plot = out / f"episode_{episode:06d}_full_action_pred_vs_gt.png"
    mae_plot = out / f"episode_{episode:06d}_full_action_mae_by_horizon.png"
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(pred) * 24); pf = pred.reshape(-1, 16); gf = gt.reshape(-1, 16)
        fig, axes = plt.subplots(4, 4, figsize=(22, 13), sharex=True)
        for d, ax in enumerate(axes.flat):
            ax.plot(x, gf[:, d], color="tab:blue", lw=.8, label="GT")
            ax.plot(x, pf[:, d], color="tab:orange", lw=.8, label="PRED")
            ax.set_title(f"{d}: {ACTION_NAMES[d]}"); ax.grid(alpha=.2)
        axes[0, 0].legend(); fig.suptitle(f"G2 full 4-frame-anchor action vs GT, episode {episode}"); fig.tight_layout(); fig.savefig(plot, dpi=140); plt.close(fig)
        h = np.arange(1, 25); fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(h, err.mean((0, 2)), label="all 16"); ax.plot(h, err[..., ARM_DIMS].mean((0, 2)), label="arm 14"); ax.plot(h, err[..., GRIPPER_DIMS].mean((0, 2)), label="gripper 2"); ax.set_xlabel("horizon step"); ax.set_ylabel("MAE"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(mae_plot, dpi=140); plt.close(fig)
    except Exception as exc:
        logging.warning("plotting failed: %s", exc); plot = None; mae_plot = None
    report = {"episode_index": episode, "num_action_blocks": len(pred), "packet_stride": 4, "window_blocks": 4, "video_not_evaluated": True, "action_detail_csv": str(detail), "action_metrics_csv": str(metrics), "action_plot": str(plot) if plot else None, "action_mae_by_horizon": str(mae_plot) if mae_plot else None, "overall_action_metrics": overall}
    (out / f"episode_{episode:06d}_full_action_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved full action report: %s", report)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    a = _parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.window_blocks < 1: raise ValueError("--window-blocks must be positive")
    root = a.test_data_root.resolve(); info = json.loads((root / "meta/info.json").read_text(encoding="utf-8")); episode = int(a.episode_index); chunks = int(info.get("chunks_size", 1000))
    parquet = _episode_path(root, info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"), episode, chunks)
    table = pq.read_table(parquet); n = table.num_rows
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32).reshape(n, 16); action = np.asarray(table["action"].to_pylist(), dtype=np.float32).reshape(n, 16)
    tmpl = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    videos = {name: _read_video(_episode_path(root, tmpl, episode, chunks, video_key=key), n) for name, key in (("top_head", "observation.images.top_head"), ("hand_left", "observation.images.hand_left"), ("hand_right", "observation.images.hand_right"))}
    max_start = n - 27; starts = list(range(0, max_start + 1, 4));
    if a.max_blocks is not None: starts = starts[: max(0, int(a.max_blocks))]
    if not starts: raise RuntimeError("No valid 4-frame action anchors")
    prompt_key = "annotation.language.action_text"; prompt = a.prompt if a.prompt is not None else (str(table[prompt_key][0].as_py()) if prompt_key in table.column_names else "")
    sid = f"g2-full-action-episode-{episode:06d}-{uuid.uuid4()}"; client = WebsocketClientPolicy(host=a.host, port=a.port); logging.info("Full action pass episode=%d rows=%d blocks=%d server=%s:%d", episode, n, len(starts), a.host, a.port)
    pred: list[np.ndarray] = []; gt: list[np.ndarray] = []; anchors: list[int] = []; started = time.time()
    try:
        before_videos = {p.name for p in a.server_video_dir.glob("*.mp4")} if a.server_video_dir else set()
        client.reset({"session_id": sid})
        _remove_new_server_videos(a.server_video_dir, before_videos)
        for i, start in enumerate(starts):
            idx = list(range(start, start + 4)); anchor = start + 3
            result = np.asarray(client.infer({
                "observation/top_head": _encode_video_observation(videos["top_head"][idx], a.image_jpeg_quality),
                "observation/hand_left": _encode_video_observation(videos["hand_left"][idx], a.image_jpeg_quality),
                "observation/hand_right": _encode_video_observation(videos["hand_right"][idx], a.image_jpeg_quality),
                "observation/state": state[anchor], "prompt": prompt, "session_id": sid,
            }), dtype=np.float32)
            if result.shape != (24, 16): raise RuntimeError(f"block {i} returned {result.shape}")
            pred.append(result); gt.append(action[anchor : anchor + 24]); anchors.append(anchor)
            if (i + 1) % a.window_blocks == 0 or i + 1 == len(starts):
                before_videos = {p.name for p in a.server_video_dir.glob("*.mp4")} if a.server_video_dir else set()
                client.reset({"session_id": sid})
                _remove_new_server_videos(a.server_video_dir, before_videos)
            if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(starts): logging.info("action block %d/%d anchor=%d elapsed=%.1fs", i + 1, len(starts), anchor, time.time() - started)
    finally:
        try:
            before_videos = {p.name for p in a.server_video_dir.glob("*.mp4")} if a.server_video_dir else set()
            client.reset({"session_id": sid})
            _remove_new_server_videos(a.server_video_dir, before_videos)
        except Exception: logging.exception("final reset failed")
    _write_reports(a.output_dir, episode, starts, anchors, np.stack(pred), np.stack(gt))


if __name__ == "__main__":
    main()
