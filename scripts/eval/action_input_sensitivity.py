#!/usr/bin/env python3
"""Counterfactual G2 action/video sensitivity evaluation against a live server.

For every target training episode this script keeps the target video and the
target action trajectory fixed, then compares:

* ``baseline``: target state + chronological four-frame history;
* ``state_swap``: a time-aligned state trajectory and four-frame history from
  another episode, both aligned to the target timeline;
* ``history_reverse``: target state + the same four target frames in reverse
  order;
* ``state_swap_history_reverse``: the same swapped state/history input as
  ``state_swap``, but with those four swapped frames in reverse order.
* ``temporal_shift``: target state and chronological four-frame history from
  another target-episode time, while the GT/action remains at the baseline
  time.

The server still receives exactly the production G2 websocket observation
contract.  Action chunks are recorded as (24, 16) arrays and compared with
the target episode's GT action horizon.  Every decoded frame from the
server's raw generated video is retained and written at a configurable slow
display FPS; GT frames are aligned to each continuous inference round for an
intuitive prediction-vs-GT video.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402


ACTION_NAMES = tuple(
    [f"left_joint_{i}" for i in range(7)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(7)]
    + ["right_gripper"]
)
ARM_DIMS = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
GRIPPER_DIMS = np.asarray([7, 15], dtype=np.int64)
ACTION_HORIZON = 24
HISTORY_LEN = 4
STAGE_NAMES = (
    "right_grasp_ethernet_dock",
    "left_grasp_cable_1",
    "insert_cable_1_into_dock",
    "left_grasp_cable_2",
    "insert_cable_2_into_dock",
    "insert_dock_hold_5s",
    "unplug_ethernet_dock",
    "return_ethernet_dock",
)

CONDITION_ORDER = (
    "baseline",
    "state_only",
    "history_only",
    "state_history_swap",
    "state_swap",
    "history_reverse",
    "state_swap_history_reverse",
    "temporal_shift",
)
CONDITION_COLORS = {
    "baseline": "tab:blue",
    "state_only": "tab:red",
    "history_only": "tab:green",
    "state_history_swap": "tab:purple",
    "state_swap": "tab:red",
    "history_reverse": "tab:orange",
    "state_swap_history_reverse": "tab:brown",
    "temporal_shift": "tab:cyan",
}


def _stage_info(anchor: int, num_rows: int) -> dict[str, object]:
    """Return the same eight-bin trajectory-stage approximation used for selection."""
    stage_index = min(len(STAGE_NAMES) - 1, int(anchor * len(STAGE_NAMES) / num_rows))
    return {
        "index": stage_index,
        "name": STAGE_NAMES[stage_index],
        "source": "normalized episode progress split into eight bins",
    }


def _condition_label(
    condition: str,
    target_stage: dict[str, object],
    donor_stage: dict[str, object],
) -> str:
    target_name = str(target_stage["name"])
    donor_name = str(donor_stage["name"])
    return {
        "baseline": f"BASELINE target={target_name}",
        "state_only": f"STATE ONLY swap-source={donor_name}",
        "history_only": f"HISTORY ONLY swap-source={donor_name}",
        "state_history_swap": f"STATE+HISTORY swap-source={donor_name}",
    }.get(condition, condition.upper())


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
    """Read a G2 training video as BGR, matching the live client capture path."""
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


def _encode_video_observation(
    frames: np.ndarray,
    quality: int,
) -> dict[str, object]:
    """Encode BGR frames exactly as the production G2 JPEG transport does."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3) video frames, got {array.shape}")
    encoded: list[bytes] = []
    jpeg_quality = int(np.clip(quality, 1, 100))
    for frame in array:
        ok, payload = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(frame),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode an evaluation frame")
        encoded.append(payload.tobytes())
    return {
        "__dreamzero_image_encoding__": "jpeg_sequence",
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "quality": jpeg_quality,
        "frames": encoded,
    }


def _grid_rgb(
    top_bgr: np.ndarray,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
) -> np.ndarray:
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
    cv2.rectangle(frame_bgr, (0, 0), (max(430, frame_bgr.shape[1] // 2), 32), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError(f"No frames to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(fps), codec="libx264", macro_block_size=None)


def _parse_pairs(spec: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pair {token!r}; expected target_episode:donor_episode"
            )
        target, donor = (int(part.strip()) for part in parts)
        if target < 0 or donor < 0:
            raise ValueError(f"Episode indices must be non-negative: {token!r}")
        pairs.append((target, donor))
    if not pairs:
        raise ValueError("--pairs must contain at least one target:donor pair")
    return pairs


def _parse_anchor_pairs(spec: str) -> list[tuple[int, int, int, int]]:
    """Parse target:donor:target_anchor:donor_anchor specifications."""
    pairs: list[tuple[int, int, int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid anchor pair {token!r}; expected "
                "target_episode:donor_episode:target_anchor:donor_anchor"
            )
        target, donor, target_anchor, donor_anchor = (
            int(part.strip()) for part in parts
        )
        if min(target, donor, target_anchor, donor_anchor) < 0:
            raise ValueError(f"Anchor pair values must be non-negative: {token!r}")
        pairs.append((target, donor, target_anchor, donor_anchor))
    if not pairs:
        raise ValueError("--anchor-pairs must contain at least one item")
    return pairs


def _parse_anchor_offsets(spec: str) -> list[int]:
    offsets = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not offsets:
        raise ValueError("--anchor-offsets must contain at least one integer")
    if min(offsets) < 0:
        raise ValueError("--anchor-offsets values must be non-negative")
    return offsets


def _parse_conditions(spec: str) -> list[str]:
    conditions = [token.strip() for token in spec.split(",") if token.strip()]
    unknown = [token for token in conditions if token not in CONDITION_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown condition(s) {unknown}; choose from {CONDITION_ORDER}"
        )
    if "baseline" not in conditions:
        conditions.insert(0, "baseline")
    # Stable order makes all plots and summary files directly comparable.
    return [condition for condition in CONDITION_ORDER if condition in conditions]


def _parse_start_frames(spec: str | None) -> dict[int, int]:
    if not spec:
        return {}
    mapping: dict[int, int] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --start-frames item {token!r}; expected episode:start_frame"
            )
        episode, start_frame = (int(part.strip()) for part in parts)
        if episode < 0 or start_frame < 0:
            raise ValueError(f"Episode and start frame must be non-negative: {token!r}")
        mapping[episode] = start_frame
    return mapping


def _load_episode(
    root: Path,
    info: dict[str, object],
    episode: int,
) -> dict[str, object]:
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    parquet_path = _episode_path(root, data_template, episode, chunks_size)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode {episode} parquet not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    num_rows = int(table.num_rows)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32).reshape(num_rows, 16)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32).reshape(num_rows, 16)

    video_template = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    videos = {
        name: _read_video(
            _episode_path(root, video_template, episode, chunks_size, video_key=key),
            num_rows,
        )
        for name, key in (
            ("top_head", "observation.images.top_head"),
            ("hand_left", "observation.images.hand_left"),
            ("hand_right", "observation.images.hand_right"),
        )
    }
    prompt_key = "annotation.language.action_text"
    prompt = str(table[prompt_key][0].as_py()) if prompt_key in table.column_names else ""
    return {
        "episode": episode,
        "num_rows": num_rows,
        "state": state,
        "action": action,
        "videos": videos,
        "prompt": prompt,
    }


def _choose_starts(
    num_rows: int,
    block_stride: int,
    max_blocks: int | None,
    start_frame: int = 0,
    contiguous: bool = False,
) -> list[int]:
    # Packet [s:s+4] is anchored at s+3.  The GT horizon is action[s+3:s+27].
    max_start = num_rows - (HISTORY_LEN + ACTION_HORIZON)
    if max_start < 0:
        return []
    if start_frame < 0 or start_frame > max_start:
        return []
    starts = list(range(start_frame, max_start + 1, block_stride))
    if max_blocks is not None and len(starts) > max_blocks:
        if contiguous:
            # Presentation videos should be a continuous piece of one
            # trajectory.  The old evenly-spaced selection is useful for a
            # broad metric sweep but makes a visually confusing jump-cut
            # video, so it is now opt-in through the default=False path.
            starts = starts[:max_blocks]
        else:
            # Cover the whole trajectory for the metric-oriented report.
            positions = np.rint(np.linspace(0, len(starts) - 1, max_blocks)).astype(int)
            starts = [starts[int(pos)] for pos in positions]
    return starts


def _state_alignment_indices(target_rows: int, donor_rows: int) -> np.ndarray:
    if target_rows <= 1 or donor_rows <= 1:
        return np.zeros(target_rows, dtype=np.int64)
    return np.rint(np.linspace(0, donor_rows - 1, target_rows)).astype(np.int64)


def _list_video_names(server_video_dir: Path) -> set[str]:
    try:
        return {path.name for path in server_video_dir.glob("*.mp4")}
    except OSError as exc:
        raise RuntimeError(f"Cannot scan server video directory {server_video_dir}: {exc}") from exc


def _wait_for_new_video(
    server_video_dir: Path,
    before: set[str],
    timeout_seconds: float,
    min_frames: int = 1,
) -> Path:
    deadline = time.time() + timeout_seconds
    observed: dict[str, tuple[int, float]] = {}
    while time.time() < deadline:
        candidates = [
            path
            for path in server_video_dir.glob("*.mp4")
            if path.name not in before and path.stat().st_size > 0
        ]
        for candidate in sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True):
            try:
                size = int(candidate.stat().st_size)
            except OSError:
                continue
            now = time.time()
            previous = observed.get(candidate.name)
            if previous is None or previous[0] != size:
                observed[candidate.name] = (size, now)
                continue
            if now - previous[1] < 1.0:
                continue
            try:
                frame_count = len(_read_mp4(candidate))
            except Exception:
                continue
            if frame_count < min_frames:
                # A newly-created but incomplete MP4 can be readable while
                # only containing the first inference chunk. Keep waiting
                # for the same file to grow instead of silently copying it.
                continue
            return candidate
        time.sleep(0.25)
    raise RuntimeError(
        f"Server reset did not produce a new video in {server_video_dir}; "
        f"existing tail={sorted(before)[-3:]}; "
        f"expected at least {min_frames} decoded frames"
    )


def _read_mp4(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        frames = [np.ascontiguousarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"Server video contains no frames: {path}")
    return frames


def _resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    if target_count <= 0:
        return []
    if len(frames) == target_count:
        return frames
    indices = np.rint(np.linspace(0, len(frames) - 1, target_count)).astype(np.int64)
    return [frames[int(index)] for index in indices]


def _metric(error: np.ndarray) -> dict[str, float | int]:
    return {
        "overall_mae": float(error.mean()),
        "arm_mae": float(error[..., ARM_DIMS].mean()),
        "gripper_mae": float(error[..., GRIPPER_DIMS].mean()),
        "left_arm_mae": float(error[..., :7].mean()),
        "right_arm_mae": float(error[..., 8:15].mean()),
        "num_action_rows": int(np.prod(error.shape[:-1])),
    }


def _write_action_tables(
    output_dir: Path,
    target_episode: int,
    donor_episode: int,
    starts: list[int],
    anchors: list[int],
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    baseline = predictions["baseline"]
    detail_path = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_action_detail.csv"
    )
    fields = [
        "condition",
        "block_index",
        "packet_start",
        "anchor_frame",
        "horizon_step",
        "action_dim",
        "action_name",
        "gt",
        "pred",
        "abs_error_vs_gt",
        "pred_delta_vs_baseline",
        "abs_pred_delta_vs_baseline",
    ]
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, predicted in predictions.items():
            for block_index, (start, anchor) in enumerate(zip(starts, anchors)):
                for horizon_step in range(ACTION_HORIZON):
                    for dim, name in enumerate(ACTION_NAMES):
                        pred_value = float(predicted[block_index, horizon_step, dim])
                        base_value = float(baseline[block_index, horizon_step, dim])
                        gt_value = float(gt[block_index, horizon_step, dim])
                        signed_delta = pred_value - base_value
                        writer.writerow(
                            {
                                "condition": condition,
                                "block_index": block_index,
                                "packet_start": start,
                                "anchor_frame": anchor,
                                "horizon_step": horizon_step + 1,
                                "action_dim": dim,
                                "action_name": name,
                                "gt": gt_value,
                                "pred": pred_value,
                                "abs_error_vs_gt": abs(pred_value - gt_value),
                                "pred_delta_vs_baseline": signed_delta,
                                "abs_pred_delta_vs_baseline": abs(signed_delta),
                            }
                        )

    metrics_path = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_metrics.csv"
    )
    metric_fields = [
        "condition",
        "scope",
        "block_index",
        "anchor_frame",
        "horizon_step",
        "overall_mae",
        "arm_mae",
        "gripper_mae",
        "left_arm_mae",
        "right_arm_mae",
        "pred_delta_vs_baseline_mae",
        "state_input_delta_mae",
        "num_action_rows",
    ]
    metric_rows: list[dict[str, object]] = []
    for condition, predicted in predictions.items():
        error = np.abs(predicted - gt)
        delta = np.abs(predicted - baseline)
        overall = _metric(error)
        metric_rows.append(
            {
                "condition": condition,
                "scope": "overall",
                "block_index": "",
                "anchor_frame": "",
                "horizon_step": "",
                **overall,
                "pred_delta_vs_baseline_mae": float(delta.mean()),
                "state_input_delta_mae": "",
            }
        )
        for block_index, anchor in enumerate(anchors):
            block_metric = _metric(error[block_index : block_index + 1])
            metric_rows.append(
                {
                    "condition": condition,
                    "scope": "block",
                    "block_index": block_index,
                    "anchor_frame": anchor,
                    "horizon_step": "",
                    **block_metric,
                    "pred_delta_vs_baseline_mae": float(delta[block_index].mean()),
                    "state_input_delta_mae": "",
                }
            )
        horizon_mae = error.mean(axis=(0, 2))
        horizon_delta = delta.mean(axis=(0, 2))
        for horizon_step, (mae, delta_mae) in enumerate(zip(horizon_mae, horizon_delta), start=1):
            metric_rows.append(
                {
                    "condition": condition,
                    "scope": "horizon",
                    "block_index": "",
                    "anchor_frame": "",
                    "horizon_step": horizon_step,
                    "overall_mae": float(mae),
                    "arm_mae": float(error[:, horizon_step - 1, ARM_DIMS].mean()),
                    "gripper_mae": float(error[:, horizon_step - 1, GRIPPER_DIMS].mean()),
                    "left_arm_mae": float(error[:, horizon_step - 1, :7].mean()),
                    "right_arm_mae": float(error[:, horizon_step - 1, 8:15].mean()),
                    "pred_delta_vs_baseline_mae": float(delta_mae),
                    "state_input_delta_mae": "",
                    "num_action_rows": int(error.shape[0]),
                }
            )
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    arrays_path = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_action_arrays.npz"
    )
    np.savez_compressed(
        arrays_path,
        ground_truth=gt,
        starts=np.asarray(starts, dtype=np.int32),
        anchors=np.asarray(anchors, dtype=np.int32),
        **{f"pred_{condition}": predicted for condition, predicted in predictions.items()},
    )
    return {
        "action_detail_csv": str(detail_path),
        "metrics_csv": str(metrics_path),
        "action_arrays_npz": str(arrays_path),
        "overall_metrics": {
            condition: {
                **_metric(np.abs(predicted - gt)),
                "pred_delta_vs_baseline_mae": float(np.abs(predicted - baseline).mean()),
            }
            for condition, predicted in predictions.items()
        },
    }


def _write_action_plots(
    output_dir: Path,
    target_episode: int,
    donor_episode: int,
    anchors: list[int],
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - report still has CSV/NPZ
        logging.warning("Could not import matplotlib: %s", exc)
        return {
            "first_action_plot": None,
            "all16_first_action_plot": None,
            "horizon_mae_plot": None,
            "all16_horizon_plots": [],
        }

    x = np.arange(1, len(anchors) + 1, dtype=np.int32)
    representative_dims = [0, 7, 8, 15]
    first_plot = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_first_action_curves.png"
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for axis, dim in zip(axes.flat[:4], representative_dims):
        axis.plot(x, gt[:, 0, dim], color="black", linewidth=2.0, label="GT")
        for condition, predicted in predictions.items():
            axis.plot(
                x,
                predicted[:, 0, dim],
                color=CONDITION_COLORS[condition],
                linewidth=1.5,
                marker="o",
                markersize=3.5,
                label=condition,
            )
        axis.set_title(f"{dim}: {ACTION_NAMES[dim]}")
        axis.grid(alpha=0.22)
        axis.set_ylabel("joint position")
    axes[0, 0].legend(fontsize=8, loc="best")
    round_error = {
        condition: np.abs(predicted - gt).mean(axis=(1, 2))
        for condition, predicted in predictions.items()
    }
    for condition, values in round_error.items():
        axes[1, 1].plot(
            x,
            values,
            color=CONDITION_COLORS[condition],
            linewidth=1.8,
            marker="o",
            label=condition,
        )
    axes[1, 1].set_title("action-chunk MAE vs GT")
    axes[1, 1].set_ylabel("MAE")
    axes[1, 1].grid(alpha=0.22)
    baseline = predictions["baseline"]
    for condition, predicted in predictions.items():
        axes[1, 2].plot(
            x,
            np.abs(predicted - baseline).mean(axis=(1, 2)),
            color=CONDITION_COLORS[condition],
            linewidth=1.8,
            marker="o",
            label=condition,
        )
    axes[1, 2].set_title("change from baseline")
    axes[1, 2].set_ylabel("MAE")
    axes[1, 2].grid(alpha=0.22)
    for axis in axes[-1, :]:
        axis.set_xlabel("continuous inference round")
    fig.suptitle(
        f"G2 action curves: target episode {target_episode}, state donor {donor_episode}"
    )
    fig.tight_layout()
    fig.savefig(first_plot, dpi=140)
    plt.close(fig)

    all16_first_plot = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_all16_first_action_curves.png"
    )
    fig, axes = plt.subplots(4, 4, figsize=(22, 16), sharex=True)
    for axis, dim in zip(axes.flat, range(16)):
        axis.plot(x, gt[:, 0, dim], color="black", linewidth=2.0, label="GT")
        for condition, predicted in predictions.items():
            axis.plot(
                x,
                predicted[:, 0, dim],
                color=CONDITION_COLORS[condition],
                linewidth=1.5,
                marker="o",
                markersize=3.5,
                label=condition,
            )
        axis.set_title(f"{dim}: {ACTION_NAMES[dim]}")
        axis.set_xlabel("continuous inference round")
        axis.set_ylabel("position")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"G2 all 16 action dimensions: target episode {target_episode}, "
        f"donor episode {donor_episode}",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(all16_first_plot, dpi=140)
    plt.close(fig)

    mae_plot = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_horizon_mae.png"
    )
    selected_rounds = sorted(set([0, len(anchors) // 2, len(anchors) - 1]))
    fig, axes = plt.subplots(
        len(selected_rounds), len(representative_dims),
        figsize=(18, 4.2 * len(selected_rounds)),
        squeeze=False,
        sharex=True,
    )
    horizon = np.arange(1, ACTION_HORIZON + 1)
    for row, round_index in enumerate(selected_rounds):
        for col, dim in enumerate(representative_dims):
            axis = axes[row, col]
            axis.plot(horizon, gt[round_index, :, dim], color="black", linewidth=2.0, label="GT")
            for condition, predicted in predictions.items():
                axis.plot(
                    horizon,
                    predicted[round_index, :, dim],
                    color=CONDITION_COLORS[condition],
                    linewidth=1.3,
                    label=condition,
                )
            axis.set_title(
                f"round {round_index + 1}, {ACTION_NAMES[dim]} "
                f"(anchor {anchors[round_index]})"
            )
            axis.grid(alpha=0.22)
            axis.set_xlabel("action chunk horizon step")
            axis.set_ylabel("position")
            if row == 0 and col == 0:
                axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(mae_plot, dpi=140)
    plt.close(fig)

    all16_horizon_plots: list[str] = []
    for round_index, anchor in enumerate(anchors):
        full_horizon_plot = output_dir / (
            f"episode_{target_episode:06d}_donor_{donor_episode:06d}_"
            f"all16_horizon_round_{round_index + 1:02d}.png"
        )
        fig, axes = plt.subplots(4, 4, figsize=(22, 16), sharex=True)
        for axis, dim in zip(axes.flat, range(16)):
            axis.plot(
                horizon,
                gt[round_index, :, dim],
                color="black",
                linewidth=2.0,
                label="GT",
            )
            for condition, predicted in predictions.items():
                axis.plot(
                    horizon,
                    predicted[round_index, :, dim],
                    color=CONDITION_COLORS[condition],
                    linewidth=1.3,
                    label=condition,
                )
            axis.set_title(f"{dim}: {ACTION_NAMES[dim]}")
            axis.set_xlabel("action horizon step")
            axis.set_ylabel("position")
            axis.grid(alpha=0.22)
        axes[0, 0].legend(fontsize=8, loc="best")
        fig.suptitle(
            f"G2 all 16 action dimensions, round {round_index + 1}, "
            f"target anchor {anchor}: target episode {target_episode}, "
            f"donor episode {donor_episode}",
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(full_horizon_plot, dpi=140)
        plt.close(fig)
        all16_horizon_plots.append(str(full_horizon_plot))
    return {
        "first_action_plot": str(first_plot),
        "all16_first_action_plot": str(all16_first_plot),
        "horizon_mae_plot": str(mae_plot),
        "all16_horizon_plots": all16_horizon_plots,
    }


def _round_boundaries(total_frames: int, num_rounds: int) -> np.ndarray:
    if total_frames < 1 or num_rounds < 1:
        return np.zeros(num_rounds + 1, dtype=np.int64)
    boundaries = np.rint(np.linspace(0, total_frames, num_rounds + 1)).astype(np.int64)
    boundaries[0] = 0
    boundaries[-1] = total_frames
    return boundaries


def _make_aligned_gt_video_frames(
    target: dict[str, object],
    starts: list[int],
    total_frames: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Build a GT sequence with the same number of frames as server output.

    Each server inference round contributes one contiguous video chunk.  The
    matching GT chunk starts immediately after the four-frame observation
    packet (the predicted future starts at anchor+1).  We preserve every
    decoded server frame and only sample the GT timeline to that same length.
    """
    videos = target["videos"]
    assert isinstance(videos, dict)
    frames: list[np.ndarray] = []
    gt_indices: list[int] = []
    boundaries = _round_boundaries(total_frames, len(starts))
    num_rows = int(target["num_rows"])
    for block_index, start in enumerate(starts):
        anchor = start + HISTORY_LEN - 1
        chunk_count = max(1, int(boundaries[block_index + 1] - boundaries[block_index]))
        gt_start = min(anchor + 1, num_rows - 1)
        gt_end = min(gt_start + chunk_count - 1, num_rows - 1)
        indices = np.rint(np.linspace(gt_start, gt_end, chunk_count)).astype(np.int64)
        for frame_index in indices:
            frame = _grid_rgb(
                videos["top_head"][frame_index],
                videos["hand_left"][frame_index],
                videos["hand_right"][frame_index],
            )
            frames.append(_label(frame, f"GROUND TRUTH  round={block_index + 1} anchor={anchor}"))
            gt_indices.append(int(frame_index))
    return frames, gt_indices


def _make_condition_video(
    output_dir: Path,
    target_episode: int,
    donor_episode: int,
    condition: str,
    raw_video: Path,
    target: dict[str, object],
    starts: list[int],
    video_fps: int,
    condition_label: str | None = None,
) -> tuple[Path, list[np.ndarray], list[np.ndarray], list[int], int]:
    raw_frames = _read_mp4(raw_video)
    gt_frames, gt_indices = _make_aligned_gt_video_frames(target, starts, len(raw_frames))
    side_by_side: list[np.ndarray] = []
    for block_frame, (predicted, gt) in enumerate(zip(raw_frames, gt_frames)):
        if predicted.shape != gt.shape:
            predicted = cv2.resize(predicted, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        predicted = _label(
            predicted,
            f"{condition_label or condition.upper()}  server-frame={block_frame + 1}/{len(raw_frames)}",
        )
        side_by_side.append(np.concatenate([predicted, gt], axis=1))
    output_path = output_dir / (
        f"episode_{target_episode:06d}_donor_{donor_episode:06d}_{condition}_slow.mp4"
    )
    _save_video(output_path, side_by_side, video_fps)
    return output_path, raw_frames, gt_frames, gt_indices, len(raw_frames)


def _make_montage_video(
    output_path: Path,
    predictions: dict[str, list[np.ndarray]],
    gt_frames: list[np.ndarray],
    video_fps: int,
    condition_labels: dict[str, str] | None = None,
) -> None:
    conditions = list(predictions)
    if not conditions:
        raise RuntimeError("Cannot make a montage without predictions")
    common_count = min([len(gt_frames)] + [len(predictions[name]) for name in conditions])
    if common_count < 1:
        raise RuntimeError("Cannot make an empty montage")
    normalized_predictions = {
        condition: _resample_frames(predictions[condition], common_count)
        for condition in conditions
    }
    normalized_gt = _resample_frames(gt_frames, common_count)
    panels: list[np.ndarray] = []
    # The strong comparison uses four counterfactual conditions plus GT.  A
    # fixed 2x2 layout (used by the old three-condition report) silently
    # dropped the fourth condition, so build a compact grid for any number of
    # conditions and keep GT as the final panel.
    panel_count = len(conditions) + 1
    columns = 3 if panel_count > 4 else 2
    rows = int(np.ceil(panel_count / columns))
    for index, gt in enumerate(normalized_gt):
        predicted_panels: list[np.ndarray] = []
        for condition in conditions:
            frame = normalized_predictions[condition][index]
            if frame.shape != gt.shape:
                frame = cv2.resize(frame, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
            predicted_panels.append(
                _label(
                    frame,
                    (condition_labels or {}).get(condition, condition.upper()),
                )
            )
        predicted_panels.append(_label(gt, "GROUND TRUTH"))
        while len(predicted_panels) < rows * columns:
            predicted_panels.append(np.zeros_like(gt))
        row_images = [
            np.concatenate(predicted_panels[row * columns : (row + 1) * columns], axis=1)
            for row in range(rows)
        ]
        panels.append(np.concatenate(row_images, axis=0))
    _save_video(output_path, panels, video_fps)


def _write_round_contact_sheet(
    output_path: Path,
    target: dict[str, object],
    starts: list[int],
    predictions: dict[str, list[np.ndarray]],
    gt_frames: list[np.ndarray],
    condition_labels: dict[str, str] | None = None,
) -> None:
    """Save three representative continuous rounds as a simple image sheet."""
    conditions = list(predictions)
    common_count = min([len(gt_frames)] + [len(predictions[name]) for name in conditions])
    normalized_predictions = {
        condition: _resample_frames(predictions[condition], common_count)
        for condition in conditions
    }
    normalized_gt = _resample_frames(gt_frames, common_count)
    selected_rounds = sorted(set([0, len(starts) // 2, len(starts) - 1]))
    boundaries = _round_boundaries(common_count, len(starts))
    rows: list[np.ndarray] = []
    for round_index in selected_rounds:
        center = int((boundaries[round_index] + boundaries[round_index + 1] - 1) // 2)
        gt_panel = _label(
            normalized_gt[center],
            f"GT  round={round_index + 1} anchor={starts[round_index] + HISTORY_LEN - 1}",
        )
        row_panels = [gt_panel]
        for condition in conditions:
            frame = normalized_predictions[condition][center]
            if frame.shape != gt_panel.shape:
                frame = cv2.resize(frame, (gt_panel.shape[1], gt_panel.shape[0]), interpolation=cv2.INTER_AREA)
            row_panels.append(
                _label(
                    frame,
                    (condition_labels or {}).get(condition, condition.upper()),
                )
            )
        rows.append(np.concatenate(row_panels, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, sheet)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument("--test-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--server-video-dir",
        type=Path,
        required=True,
        help="The live server's VIDEO_SAVE_MODE=full output directory.",
    )
    parser.add_argument(
        "--pairs",
        default="0:935,769:1129,1044:1221",
        help="Comma-separated target_episode:donor_episode pairs.",
    )
    parser.add_argument(
        "--anchor-pairs",
        default=None,
        help=(
            "Explicit cross-stage anchors, comma-separated as "
            "target_episode:donor_episode:target_anchor:donor_anchor. "
            "When supplied, these replace --pairs and use --anchor-offsets."
        ),
    )
    parser.add_argument(
        "--anchor-offsets",
        default="0,4,8,12",
        help="Offsets added to each explicit target/donor anchor (default: 0,4,8,12).",
    )
    parser.add_argument(
        "--conditions",
        default="baseline,state_only,history_only,state_history_swap",
        help="Conditions to run; combined condition is optional.",
    )
    parser.add_argument("--block-stride", type=int, default=4)
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=12,
        help="Maximum action anchors per target episode.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First target frame used when selecting the presentation window.",
    )
    parser.add_argument(
        "--start-frames",
        default=None,
        help="Optional per-target overrides, e.g. 0:740,30:1332.",
    )
    parser.add_argument(
        "--temporal-offset",
        type=int,
        default=4,
        help="For temporal_shift, use the target episode packet at start+offset while evaluating GT at start.",
    )
    parser.add_argument(
        "--contiguous-blocks",
        action="store_true",
        help="Keep selected inference rounds contiguous for readable videos.",
    )
    parser.add_argument("--image-jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--video-fps",
        type=int,
        default=5,
        help="Display FPS for slow diagnostic videos; source server videos remain 30 FPS.",
    )
    parser.add_argument("--video-wait-seconds", type=float, default=120.0)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()
    if args.block_stride < 1:
        raise ValueError("--block-stride must be positive")
    if args.max_blocks is not None and args.max_blocks < 1:
        raise ValueError("--max-blocks must be positive")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be positive")

    root = args.test_data_root.resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    pairs = _parse_pairs(args.pairs)
    explicit_anchor_pairs = _parse_anchor_pairs(args.anchor_pairs) if args.anchor_pairs else None
    anchor_offsets = _parse_anchor_offsets(args.anchor_offsets)
    if explicit_anchor_pairs is not None:
        pairs = [(target, donor) for target, donor, _, _ in explicit_anchor_pairs]
    conditions = _parse_conditions(args.conditions)
    start_frame_overrides = _parse_start_frames(args.start_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_video_dir = args.server_video_dir.resolve()
    server_video_dir.mkdir(parents=True, exist_ok=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    logging.info("Connected to DreamZero server %s:%s metadata=%s", args.host, args.port, metadata)
    all_pair_reports: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    started = time.time()

    try:
        for pair_index, (target_episode, donor_episode) in enumerate(pairs):
            target = _load_episode(root, info, target_episode)
            donor = _load_episode(root, info, donor_episode)
            target_rows = int(target["num_rows"])
            donor_rows = int(donor["num_rows"])
            target_state = np.asarray(target["state"], dtype=np.float32)
            target_action = np.asarray(target["action"], dtype=np.float32)
            donor_state = np.asarray(donor["state"], dtype=np.float32)
            donor_indices = _state_alignment_indices(target_rows, donor_rows)
            explicit_anchor = (
                explicit_anchor_pairs[pair_index]
                if explicit_anchor_pairs is not None
                else None
            )
            if explicit_anchor is not None:
                _, _, target_anchor_base, donor_anchor_base = explicit_anchor
                starts = [
                    target_anchor_base - HISTORY_LEN + 1 + offset
                    for offset in anchor_offsets
                ]
                donor_starts = [
                    donor_anchor_base - HISTORY_LEN + 1 + offset
                    for offset in anchor_offsets
                ]
                if any(
                    start < 0 or start + HISTORY_LEN + ACTION_HORIZON > target_rows
                    for start in starts
                ):
                    raise RuntimeError(
                        f"Explicit target anchors {starts} do not leave a complete "
                        f"{HISTORY_LEN}-frame packet and {ACTION_HORIZON}-step horizon "
                        f"in episode {target_episode} ({target_rows} rows)"
                    )
                if any(
                    start < 0 or start + HISTORY_LEN > donor_rows
                    for start in donor_starts
                ):
                    raise RuntimeError(
                        f"Explicit donor anchors {donor_starts} do not leave a complete "
                        f"{HISTORY_LEN}-frame packet in episode {donor_episode} "
                        f"({donor_rows} rows)"
                    )
                pair_start_frame = starts[0]
                anchors = [start + HISTORY_LEN - 1 for start in starts]
                donor_anchors = [start + HISTORY_LEN - 1 for start in donor_starts]
            else:
                pair_start_frame = start_frame_overrides.get(target_episode, args.start_frame)
                starts = _choose_starts(
                    target_rows,
                    args.block_stride,
                    args.max_blocks,
                    start_frame=pair_start_frame,
                    contiguous=args.contiguous_blocks,
                )
                if not starts:
                    raise RuntimeError(
                        f"Target episode {target_episode} has only {target_rows} rows; "
                        f"no complete {ACTION_HORIZON}-step horizons"
                    )
                anchors = [start + HISTORY_LEN - 1 for start in starts]
                # Legacy state_swap alignment maps the target timeline onto the
                # donor timeline.  Explicit strong-comparison runs instead use
                # donor anchors supplied by --anchor-pairs above.
                donor_starts = [
                    int(donor_indices[start]) - HISTORY_LEN + 1 for start in starts
                ]
                donor_anchors = [int(donor_indices[anchor]) for anchor in anchors]
            target_stage = _stage_info(anchors[0], target_rows)
            donor_stage = _stage_info(donor_anchors[0], donor_rows)
            condition_labels = {
                condition: _condition_label(condition, target_stage, donor_stage)
                for condition in conditions
            }
            gt = np.stack([target_action[anchor : anchor + ACTION_HORIZON] for anchor in anchors])
            pair_output = args.output_dir / f"target_{target_episode:06d}_donor_{donor_episode:06d}"
            pair_output.mkdir(parents=True, exist_ok=True)
            predictions: dict[str, np.ndarray] = {}
            condition_video_frames: dict[str, list[np.ndarray]] = {}
            condition_gt_video_frames: dict[str, list[np.ndarray]] = {}
            condition_video_reports: dict[str, dict[str, object]] = {}
            session_ids: dict[str, str] = {}

            logging.info(
                "Pair %d/%d target=%d (%d rows, task=%s) donor=%d (%d rows, task=%s) blocks=%d anchors=%s",
                pair_index + 1,
                len(pairs),
                target_episode,
                target_rows,
                target["prompt"],
                donor_episode,
                donor_rows,
                donor["prompt"],
                len(starts),
                anchors,
            )

            for condition in conditions:
                session_id = (
                    f"action-sensitivity-{target_episode:06d}-{donor_episode:06d}-"
                    f"{condition}-{uuid.uuid4()}"
                )
                session_ids[condition] = session_id
                # Flush any previous condition and establish an empty server
                # video buffer before this condition's first inference.
                client.reset({"session_id": session_id})
                before_inferences = _list_video_names(server_video_dir)
                condition_predictions: list[np.ndarray] = []
                condition_started = time.time()
                for block_index, (start, anchor) in enumerate(zip(starts, anchors)):
                    reverse_history = condition in {
                        "history_reverse",
                        "state_swap_history_reverse",
                    }
                    if condition == "temporal_shift":
                        source_start = start + int(args.temporal_offset)
                        if source_start < 0 or source_start + HISTORY_LEN > target_rows:
                            raise RuntimeError(
                                f"temporal_shift source window [{source_start}, "
                                f"{source_start + HISTORY_LEN}) is outside target episode {target_episode}"
                            )
                        source_anchor = source_start + HISTORY_LEN - 1
                        input_state = target_state[source_anchor]
                        history_indices = np.arange(
                            source_start,
                            source_start + HISTORY_LEN,
                            dtype=np.int64,
                        )
                        source_videos = target["videos"]
                    else:
                        use_donor_state = condition in {
                            "state_only",
                            "state_history_swap",
                            "state_swap",
                            "state_swap_history_reverse",
                        }
                        use_donor_history = condition in {
                            "history_only",
                            "state_history_swap",
                            "state_swap",
                            "state_swap_history_reverse",
                        }
                        if use_donor_state:
                            state_index = donor_anchors[block_index]
                            input_state = donor_state[state_index]
                        else:
                            input_state = target_state[anchor]
                        if use_donor_history:
                            if explicit_anchor is not None:
                                history_indices = np.arange(
                                    donor_starts[block_index],
                                    donor_starts[block_index] + HISTORY_LEN,
                                    dtype=np.int64,
                                )
                            else:
                                # Preserve the legacy linearly time-aligned
                                # behavior when --anchor-pairs is not used.
                                history_indices = donor_indices[start : start + HISTORY_LEN]
                            source_videos = donor["videos"]
                        else:
                            history_indices = np.arange(start, start + HISTORY_LEN, dtype=np.int64)
                            source_videos = target["videos"]
                    camera_frames = {
                        key: source_videos[key][history_indices]
                        for key in ("top_head", "hand_left", "hand_right")
                    }
                    if reverse_history:
                        camera_frames = {key: frames[::-1] for key, frames in camera_frames.items()}
                    prompt = args.prompt if args.prompt is not None else str(target["prompt"])
                    observation = {
                        "observation/top_head": _encode_video_observation(camera_frames["top_head"], args.image_jpeg_quality),
                        "observation/hand_left": _encode_video_observation(camera_frames["hand_left"], args.image_jpeg_quality),
                        "observation/hand_right": _encode_video_observation(camera_frames["hand_right"], args.image_jpeg_quality),
                        "observation/state": input_state,
                        "prompt": prompt,
                        # Keep both aliases explicit.  The live G2 adapter
                        # accepts either one, and this makes the language
                        # payload auditable independently of adapter defaults.
                        "annotation.language.action_text": prompt,
                        "session_id": session_id,
                    }
                    result = np.asarray(client.infer(observation), dtype=np.float32)
                    if result.shape != (ACTION_HORIZON, 16):
                        raise RuntimeError(
                            f"target={target_episode} condition={condition} block={block_index} "
                            f"returned {result.shape}, expected {(ACTION_HORIZON, 16)}"
                        )
                    condition_predictions.append(result)
                    if block_index == 0 or block_index + 1 == len(starts):
                        logging.info(
                            "target=%d condition=%s block=%d/%d anchor=%d elapsed=%.1fs",
                            target_episode,
                            condition,
                            block_index + 1,
                            len(starts),
                            anchor,
                            time.time() - condition_started,
                        )

                # reset flushes the accumulated generated video for exactly
                # this condition while keeping the loaded checkpoint alive.
                before_flush = _list_video_names(server_video_dir)
                client.reset({"session_id": session_id})
                raw_video = _wait_for_new_video(
                    server_video_dir,
                    before_flush,
                    args.video_wait_seconds,
                    min_frames=max(1, len(starts) * 5),
                )
                # If a stale reset-created file appeared before the first
                # inference, it is intentionally not treated as this run's
                # video.  Keep the variable for a useful audit trail.
                stale_reset_files = sorted(_list_video_names(server_video_dir) - before_inferences)
                raw_copy = pair_output / f"{condition}_server_raw.mp4"
                shutil.copy2(raw_video, raw_copy)
                predicted = np.stack(condition_predictions)
                predictions[condition] = predicted
                condition_video_path, predicted_video_frames, aligned_gt_frames, gt_indices, raw_frame_count = _make_condition_video(
                    pair_output,
                    target_episode,
                    donor_episode,
                    condition,
                    raw_video,
                    target,
                    starts,
                    args.video_fps,
                    condition_labels[condition],
                )
                condition_video_frames[condition] = predicted_video_frames
                condition_gt_video_frames[condition] = aligned_gt_frames
                condition_video_reports[condition] = {
                    "server_raw_video": str(raw_copy),
                    "server_raw_frames": raw_frame_count,
                    "slow_video": str(condition_video_path),
                    "slow_video_fps": args.video_fps,
                    "display_frames": raw_frame_count,
                    "source_server_fps": 30,
                    "gt_frame_indices": gt_indices,
                    "video_frames_preserved": True,
                    "stale_reset_files_seen": stale_reset_files,
                }

            table_report = _write_action_tables(
                pair_output,
                target_episode,
                donor_episode,
                starts,
                anchors,
                gt,
                predictions,
            )
            plot_report = _write_action_plots(
                pair_output,
                target_episode,
                donor_episode,
                anchors,
                gt,
                predictions,
            )
            montage_path = pair_output / (
                f"episode_{target_episode:06d}_donor_{donor_episode:06d}_conditions_montage_slow.mp4"
            )
            reference_condition = conditions[0]
            montage_gt_frames = condition_gt_video_frames[reference_condition]
            _make_montage_video(
                montage_path,
                condition_video_frames,
                montage_gt_frames,
                args.video_fps,
                condition_labels,
            )
            contact_sheet_path = pair_output / (
                f"episode_{target_episode:06d}_donor_{donor_episode:06d}_rounds_contact_sheet.png"
            )
            _write_round_contact_sheet(
                contact_sheet_path,
                target,
                starts,
                condition_video_frames,
                montage_gt_frames,
                condition_labels,
            )

            state_delta = np.stack(
                [
                    np.abs(target_state[anchor] - donor_state[donor_anchor])
                    for anchor, donor_anchor in zip(anchors, donor_anchors)
                ]
            )
            pair_report = {
                "target_episode": target_episode,
                "donor_episode": donor_episode,
                "target_rows": target_rows,
                "donor_rows": donor_rows,
                "target_prompt": target["prompt"],
                "donor_prompt": donor["prompt"],
                "target_stage": target_stage,
                "donor_stage": donor_stage,
                "server": {"host": args.host, "port": args.port, "metadata": metadata},
                "protocol": {
                    "history_length": HISTORY_LEN,
                    "history_normal": "[t-3,t-2,t-1,t]",
                    "history_reverse": "[t,t-1,t-2,t-3]",
                    "state_only": "donor state at donor_anchor + target chronological four-frame history",
                    "history_only": "target state at target_anchor + donor chronological four-frame history",
                    "state_history_swap": "donor state and donor chronological four-frame history",
                    "state_swap_alignment": (
                        "explicit target/donor anchors and shared offsets"
                        if explicit_anchor is not None
                        else "donor state and donor four-frame history linearly time-aligned to target episode row count"
                    ),
                    "history_reorder_control": "state_swap and state_swap_history_reverse share the same donor state and four frames; only frame order differs",
                    "temporal_shift": "input state and chronological four-frame history are read from target start+temporal_offset; GT/action stays at target start",
                    "temporal_offset": args.temporal_offset,
                    "gt_action": "target episode action[anchor:anchor+24]",
                    "action_shape": [ACTION_HORIZON, 16],
                    "prompt_sent": target["prompt"],
                    "prompt_sent_length": len(str(target["prompt"])),
                    "prompt_input_aliases": ["prompt", "annotation.language.action_text"],
                },
                "starts": starts,
                "anchors": anchors,
                "donor_starts": donor_starts,
                "donor_anchors": donor_anchors,
                "anchor_offsets": anchor_offsets if explicit_anchor is not None else None,
                "start_frame": pair_start_frame,
                "state_input_delta": {
                    "mean_abs": float(state_delta.mean()),
                    "max_abs": float(state_delta.max()),
                    "by_dim_mean_abs": state_delta.mean(axis=0).tolist(),
                },
                "conditions": conditions,
                "session_ids": session_ids,
                "video_montage_slow": str(montage_path),
                "rounds_contact_sheet": str(contact_sheet_path),
                "display_video_fps": args.video_fps,
                "actual_target_fps": 30,
                "display_slowdown_factor": 30.0 / args.video_fps,
                "video_note": "Every decoded server output frame is kept; GT is aligned per continuous inference round.",
                **table_report,
                **plot_report,
                "condition_videos": condition_video_reports,
            }
            report_path = pair_output / (
                f"episode_{target_episode:06d}_donor_{donor_episode:06d}_report.json"
            )
            report_path.write_text(json.dumps(pair_report, ensure_ascii=False, indent=2), encoding="utf-8")
            pair_report["report_json"] = str(report_path)
            all_pair_reports.append(pair_report)

            for condition, metrics in table_report["overall_metrics"].items():
                all_summary_rows.append(
                    {
                        "target_episode": target_episode,
                        "donor_episode": donor_episode,
                        "target_stage_index": target_stage["index"],
                        "target_stage": target_stage["name"],
                        "donor_stage_index": donor_stage["index"],
                        "donor_stage": donor_stage["name"],
                        "condition": condition,
                        **metrics,
                        "state_input_delta_mae": float(state_delta.mean()),
                    }
                )

    finally:
        try:
            client.reset({"session_id": f"action-sensitivity-final-{uuid.uuid4()}"})
        except Exception:
            logging.exception("Final server reset failed")

    summary_csv = args.output_dir / "action_sensitivity_summary.csv"
    summary_fields = [
        "target_episode",
        "donor_episode",
        "target_stage_index",
        "target_stage",
        "donor_stage_index",
        "donor_stage",
        "condition",
        "overall_mae",
        "arm_mae",
        "gripper_mae",
        "left_arm_mae",
        "right_arm_mae",
        "num_action_rows",
        "pred_delta_vs_baseline_mae",
        "state_input_delta_mae",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_summary_rows)

    aggregate: dict[str, dict[str, float | int]] = {}
    for condition in conditions:
        rows = [row for row in all_summary_rows if row["condition"] == condition]
        if not rows:
            continue
        aggregate[condition] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in (
                "overall_mae",
                "arm_mae",
                "gripper_mae",
                "left_arm_mae",
                "right_arm_mae",
                "pred_delta_vs_baseline_mae",
                "state_input_delta_mae",
            )
        }
        aggregate[condition]["num_action_rows"] = int(
            sum(int(row["num_action_rows"]) for row in rows)
        )
    summary_json = args.output_dir / "action_sensitivity_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "server": {"host": args.host, "port": args.port, "metadata": metadata},
                "dataset_root": str(root),
                "pairs": pairs,
                "anchor_pairs": explicit_anchor_pairs,
                "anchor_offsets": anchor_offsets,
                "conditions": conditions,
                "block_stride": args.block_stride,
                "max_blocks": args.max_blocks,
                "start_frame": args.start_frame,
                "start_frame_overrides": start_frame_overrides,
                "contiguous_blocks": args.contiguous_blocks,
                "display_video_fps": args.video_fps,
                "pair_reports": all_pair_reports,
                "aggregate_mean_over_pairs": aggregate,
                "elapsed_seconds": time.time() - started,
                "summary_csv": str(summary_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Saved sensitivity summary CSV: %s", summary_csv)
    logging.info("Saved sensitivity summary JSON: %s", summary_json)


if __name__ == "__main__":
    main()
