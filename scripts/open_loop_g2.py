#!/usr/bin/env python3
"""Simple offline open-loop evaluation for DreamZero G2.

This is the G2 counterpart of ``scripts/open_loop_yam.py``.  It evaluates
the action head directly from a checkpoint; no websocket server, server-side
cache, temporary MP4 files, or causal-window stitching is involved.

At each sampled anchor ``t`` the evaluator feeds the four ground-truth
frames ``[t-3, t]``, the state at ``t``, and the dataset's task prompt.  The
returned 24-step action chunk is compared with ``action[t:t+24]``.  Anchors
are independent: the action-head inference cache is reset before every
anchor, so a result cannot accidentally depend on the previous sample.

Example (on the training server):

    python scripts/open_loop_g2.py \
        --model-path /data/wangk/checkpoints/.../checkpoint-500 \
        --dataset-path /data/training_data/teleop/g2/g2_mock_light_module_joint_gear_policy_gripper/train \
        --episodes 0,1,2 \
        --max-anchors 8 \
        --output-dir /tmp/g2_open_loop_checkpoint_500

The script intentionally reports all 16 action dimensions: 14 arm joints
and two grippers.  Long generated-video comparison remains a separate visual
spot check because ``open_loop_yam.py`` itself is an action-fit evaluator,
not a video-rollout evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch._dynamo

torch._dynamo.config.disable = True

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


VIDEO_KEYS = {
    "top_head": "observation.images.top_head",
    "hand_left": "observation.images.hand_left",
    "hand_right": "observation.images.hand_right",
}

STATE_KEYS = {
    "state.left_joint_position": (0, 7),
    "state.left_gripper_position": (7, 8),
    "state.right_joint_position": (8, 15),
    "state.right_gripper_position": (15, 16),
}

ACTION_KEYS = (
    "action.left_joint_position",
    "action.left_gripper_position",
    "action.right_joint_position",
    "action.right_gripper_position",
)

ACTION_NAMES = (
    *(f"left_joint_{i}" for i in range(7)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(7)),
    "right_gripper",
)

ARM_DIMS = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
GRIPPER_DIMS = np.asarray([7, 15], dtype=np.int64)


@dataclass
class Episode:
    index: int
    state: np.ndarray
    action: np.ndarray
    videos: dict[str, np.ndarray]
    prompt: str


def _episode_path(
    root: Path,
    template: str,
    episode: int,
    chunks_size: int,
    **kwargs: object,
) -> Path:
    return root / template.format(
        episode_chunk=episode // chunks_size,
        episode_index=episode,
        **kwargs,
    )


def _read_video(path: Path, expected_rows: int) -> np.ndarray:
    """Read an MP4 as RGB frames with shape ``[T,H,W,3]``."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open G2 video: {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < expected_rows:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < expected_rows:
        raise RuntimeError(
            f"{path} contains {len(frames)} frames, expected {expected_rows}"
        )
    return np.stack(frames, axis=0)


def _scalar(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return _scalar(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return _scalar(value[0]) if value else ""
    return str(value)


def load_episode(root: Path, episode: int) -> Episode:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    parquet = _episode_path(
        root,
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        ),
        episode,
        chunks_size,
    )
    table = pq.read_table(parquet)
    n = int(table.num_rows)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32).reshape(n, 16)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32).reshape(n, 16)

    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    videos = {
        name: _read_video(
            _episode_path(
                root,
                video_template,
                episode,
                chunks_size,
                video_key=key,
            ),
            n,
        )
        for name, key in VIDEO_KEYS.items()
    }

    prompt_key = "annotation.language.action_text"
    if prompt_key in table.column_names:
        prompt = _scalar(table[prompt_key][0].as_py())
    else:
        prompt = ""
    return Episode(index=episode, state=state, action=action, videos=videos, prompt=prompt)


def _build_obs(episode: Episode, anchor: int, prompt: str) -> dict[str, object]:
    """Build the raw G2 observation expected by the G2 transform pipeline."""

    if anchor < 3:
        raise ValueError(f"G2 requires anchor >= 3 for four-frame history, got {anchor}")
    obs: dict[str, object] = {
        f"video.{name}": frames[anchor - 3 : anchor + 1]
        for name, frames in episode.videos.items()
    }
    state = episode.state[anchor]
    for key, (start, end) in STATE_KEYS.items():
        # The leading singleton is the state-time dimension.  GrootSimPolicy
        # adds the batch dimension itself, just as open_loop_yam.py does.
        obs[key] = state[start:end].reshape(1, -1).astype(np.float32)
    obs["annotation.language.action_text"] = prompt
    return obs


def _as_2d(value: object, width: int) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        if width == 1:
            array = array[:, None]
        else:
            array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise RuntimeError(f"Expected action array [24,{width}], got {array.shape}")
    return array.astype(np.float32, copy=False)


def _extract_action(result: Batch) -> np.ndarray:
    if not hasattr(result, "act"):
        raise RuntimeError("Policy result has no .act field")
    action = result.act
    pieces: list[np.ndarray] = []
    widths = (7, 1, 7, 1)
    for key, width in zip(ACTION_KEYS, widths):
        if key not in action:
            break
        pieces.append(_as_2d(action[key], width))
    if len(pieces) == len(ACTION_KEYS):
        output = np.concatenate(pieces, axis=-1)
    elif "action" in action:
        output = _as_2d(action["action"], 16)
    else:
        raise RuntimeError(f"Unexpected policy action keys: {list(action.keys())}")
    if output.shape != (24, 16):
        raise RuntimeError(f"Expected complete G2 action chunk [24,16], got {output.shape}")
    return output


def _reset_action_cache(policy: GrootSimPolicy) -> None:
    head = getattr(policy.trained_model, "action_head", None)
    reset = getattr(head, "reset_inference_cache", None)
    if reset is None:
        raise RuntimeError(
            "The loaded action head has no reset_inference_cache(); "
            "independent open-loop anchors would not be safe."
        )
    reset()


def _select_anchors(num_rows: int, stride: int, max_anchors: int | None) -> list[int]:
    last = num_rows - 24
    if last < 3:
        return []
    anchors = list(range(3, last + 1, max(1, stride)))
    if max_anchors is not None:
        anchors = anchors[: max(0, max_anchors)]
    return anchors


def _plot_action_chunk(path: Path, predicted: np.ndarray, ground_truth: np.ndarray, anchor: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharex=True)
    groups = (("left arm (7)", range(0, 7)), ("right arm (7)", range(8, 15)), ("grippers (2)", (7, 15)))
    horizon = np.arange(predicted.shape[0])
    for ax, (title, dims) in zip(axes, groups):
        for dim in dims:
            ax.plot(horizon, ground_truth[:, dim], "--", linewidth=1.4, alpha=0.75, label=f"GT {ACTION_NAMES[dim]}")
            ax.plot(horizon, predicted[:, dim], linewidth=1.4, alpha=0.85, label=f"Pred {ACTION_NAMES[dim]}")
        ax.set_title(title)
        ax.set_xlabel("chunk horizon (24 steps)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("absolute action")
    axes[-1].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"G2 open-loop action chunk | anchor={anchor} | solid=Pred, dashed=GT")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_anchor_first_step(path: Path, predicted: np.ndarray, ground_truth: np.ndarray, anchors: list[int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharex=True)
    groups = (("left arm (7)", range(0, 7)), ("right arm (7)", range(8, 15)), ("grippers (2)", (7, 15)))
    x = np.asarray(anchors)
    for ax, (title, dims) in zip(axes, groups):
        for dim in dims:
            ax.plot(x, ground_truth[:, 0, dim], "--", linewidth=1.2, alpha=0.75, label=f"GT {ACTION_NAMES[dim]}")
            ax.plot(x, predicted[:, 0, dim], linewidth=1.2, alpha=0.85, label=f"Pred {ACTION_NAMES[dim]}")
        ax.set_title(title)
        ax.set_xlabel("GT timeline anchor")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("first action in predicted chunk")
    axes[-1].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("G2 open-loop first-action timeline | solid=Pred, dashed=GT")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_horizon_error(path: Path, error: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.mean(error, axis=(0, 2)), label="all 16", linewidth=2)
    ax.plot(np.mean(error[:, :, ARM_DIMS], axis=(0, 2)), label="arm 14", linewidth=2)
    ax.plot(np.mean(error[:, :, GRIPPER_DIMS], axis=(0, 2)), label="gripper 2", linewidth=2)
    ax.set_xlabel("action horizon step")
    ax.set_ylabel("mean absolute error")
    ax.set_title("G2 open-loop error by action horizon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_episode_report(
    output_dir: Path,
    episode: Episode,
    anchors: list[int],
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    inference_seconds: list[float],
) -> dict[str, object]:
    error = np.abs(predicted - ground_truth)
    metrics = {
        "episode": episode.index,
        "prompt": episode.prompt,
        "num_rows": int(episode.action.shape[0]),
        "num_anchors": len(anchors),
        "anchors": anchors,
        "action_shape": list(predicted.shape),
        "mae_all_16": float(error.mean()),
        "mae_arm_14": float(error[:, :, ARM_DIMS].mean()),
        "mae_gripper_2": float(error[:, :, GRIPPER_DIMS].mean()),
        "mae_first_action_all_16": float(error[:, 0].mean()),
        "mae_by_horizon_all_16": error.mean(axis=(0, 2)).tolist(),
        "mae_by_dim": error.mean(axis=(0, 1)).tolist(),
        "avg_inference_seconds": float(np.mean(inference_seconds)),
        "median_inference_seconds": float(np.median(inference_seconds)),
    }

    np.savez_compressed(
        output_dir / f"episode_{episode.index:06d}_open_loop_actions.npz",
        anchors=np.asarray(anchors, dtype=np.int32),
        predicted=predicted,
        ground_truth=ground_truth,
    )
    with (output_dir / f"episode_{episode.index:06d}_anchor_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("anchor", "mae_all_16", "mae_arm_14", "mae_gripper_2", "inference_seconds"))
        for i, anchor in enumerate(anchors):
            row_error = error[i]
            writer.writerow((
                anchor,
                float(row_error.mean()),
                float(row_error[:, ARM_DIMS].mean()),
                float(row_error[:, GRIPPER_DIMS].mean()),
                inference_seconds[i],
            ))

    _plot_action_chunk(
        output_dir / f"episode_{episode.index:06d}_action_chunk_anchor_{anchors[0]:06d}.png",
        predicted[0],
        ground_truth[0],
        anchors[0],
    )
    _plot_anchor_first_step(
        output_dir / f"episode_{episode.index:06d}_action_first_step_timeline.png",
        predicted,
        ground_truth,
        anchors,
    )
    _plot_horizon_error(
        output_dir / f"episode_{episode.index:06d}_action_horizon_mae.png",
        error,
    )
    report_path = output_dir / f"episode_{episode.index:06d}_open_loop_report.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _init_single_process_dist(port: int) -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _set_cuda_device(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    if ":" in device:
        torch.cuda.set_device(int(device.rsplit(":", 1)[1]))


def evaluate(args: argparse.Namespace) -> None:
    _init_single_process_dist(args.dist_port)
    _set_cuda_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading G2 checkpoint directly from {args.model_path} ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.G2,
        model_path=str(args.model_path),
        device=args.device,
    )
    print("Model loaded.")

    episode_indices = [int(item) for item in args.episodes.split(",") if item.strip()]
    all_reports: list[dict[str, object]] = []
    for episode_index in episode_indices:
        episode = load_episode(args.dataset_path, episode_index)
        prompt = args.prompt if args.prompt is not None else episode.prompt
        if not prompt.strip():
            raise RuntimeError(
                f"Episode {episode_index} has an empty prompt. Pass --prompt explicitly "
                "or fix annotation.language.action_text in the dataset."
            )
        anchors = _select_anchors(len(episode.action), args.anchor_stride, args.max_anchors)
        if not anchors:
            raise RuntimeError(
                f"Episode {episode_index} has no valid anchors: rows={len(episode.action)}"
            )
        predicted_chunks: list[np.ndarray] = []
        ground_truth_chunks: list[np.ndarray] = []
        inference_seconds: list[float] = []
        print(
            f"\nEpisode {episode_index}: rows={len(episode.action)} "
            f"anchors={len(anchors)} prompt={prompt!r}"
        )
        for i, anchor in enumerate(anchors):
            _reset_action_cache(policy)
            obs = _build_obs(episode, anchor, prompt)
            started = time.perf_counter()
            with torch.inference_mode():
                result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
            elapsed = time.perf_counter() - started
            pred = _extract_action(result)
            gt = episode.action[anchor : anchor + 24].astype(np.float32, copy=False)
            if gt.shape != (24, 16):
                raise RuntimeError(f"Episode {episode_index} anchor {anchor}: GT shape={gt.shape}")
            predicted_chunks.append(pred)
            ground_truth_chunks.append(gt)
            inference_seconds.append(elapsed)
            if i == 0 or (i + 1) % args.log_every == 0 or i + 1 == len(anchors):
                chunk_mae = float(np.abs(pred - gt).mean())
                print(
                    f"  [{i + 1:>3d}/{len(anchors)}] anchor={anchor} "
                    f"mae={chunk_mae:.6f} infer={elapsed:.3f}s"
                )

        predicted = np.stack(predicted_chunks)
        ground_truth = np.stack(ground_truth_chunks)
        metrics = _write_episode_report(
            args.output_dir,
            episode,
            anchors,
            predicted,
            ground_truth,
            inference_seconds,
        )
        metrics["prompt_used"] = prompt
        all_reports.append(metrics)
        print(
            f"  episode metrics: all16={metrics['mae_all_16']:.6f} "
            f"arm14={metrics['mae_arm_14']:.6f} "
            f"gripper2={metrics['mae_gripper_2']:.6f}"
        )

    summary = {
        "protocol": "direct checkpoint open-loop; 4 GT history frames; independent cache reset per anchor",
        "checkpoint": str(args.model_path),
        "dataset": str(args.dataset_path),
        "episodes": episode_indices,
        "anchor_stride": args.anchor_stride,
        "max_anchors": args.max_anchors,
        "reports": all_reports,
        "mean_mae_all_16": float(np.mean([item["mae_all_16"] for item in all_reports])),
        "mean_mae_arm_14": float(np.mean([item["mae_arm_14"] for item in all_reports])),
        "mean_mae_gripper_2": float(np.mean([item["mae_gripper_2"] for item in all_reports])),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nSummary: all16={summary['mean_mae_all_16']:.6f} "
        f"arm14={summary['mean_mae_arm_14']:.6f} "
        f"gripper2={summary['mean_mae_gripper_2']:.6f}"
    )
    print(f"Results saved to {args.output_dir.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", default="0,1,2", help="Comma-separated episode indices")
    parser.add_argument("--prompt", default=None, help="Override the dataset prompt")
    parser.add_argument("--anchor-stride", type=int, default=24)
    parser.add_argument("--max-anchors", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results_g2_open_loop"))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dist-port", type=int, default=29501)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
