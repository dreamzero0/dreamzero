"""Offline G2 checkpoint video diagnostic on the held-out G2 test split.

This module reuses the dataset/video plumbing from
``eval_agibot_checkpoint_on_g2_testset.py`` but constructs a native G2 policy.
Predicted actions are deliberately ignored: the purpose of this evaluator is
to compare world-model video quality across the base, joint-LoRA, and
video-only-LoRA checkpoints without connecting to a robot.
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import datetime
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import tyro

import eval_agibot_checkpoint_on_g2_testset as shared
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


@dataclasses.dataclass
class Args(shared.Args):
    model_path: str = (
        "/data/wangk/checkpoints/"
        "dreamzero_g2_video_only_lora_1k/checkpoint-1000"
    )
    output_dir: str = (
        "/data/wangk/dreamzero/g2_testset_video_eval/"
        "video_only_checkpoint-1000"
    )
    embodiment_tag: str = "g2"
    diagnostic_fps: int = 3
    full_episode: bool = False
    rollout_stride: int = 24
    action_eval: bool = False
    # Windowed teacher-forced diagnostic.  Each window is made of four
    # consecutive 4-frame observations; four causal forwards are concatenated
    # to the 33 decoded frames used for the video comparison.
    windowed: bool = False
    window_history: int = 4
    window_stride: int = 4
    window_starts: str = "0,4,8,12,16,20,24,28"
    rollout_future_frames: int = 33
    rollout_blocks: int = 4
    # Keep the default report compact: one stitched comparison MP4 and two
    # action CSV tables.  Per-window input/prediction artifacts are opt-in.
    save_window_artifacts: bool = False


class G2VideoDiagnosticPolicy(shared.G2RoboarenaPolicy):
    """Run the G2 video branch while making actions non-observable outputs."""

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        del action_dict
        return np.zeros((24, 16), dtype=np.float32)


def _checkpoint_g2_video_resolutions(
    model_path: Path,
) -> dict[str, tuple[int, int]]:
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    shared._require_file(metadata_path, "G2 checkpoint metadata")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    try:
        video_meta = metadata["g2"]["modalities"]["video"]
    except KeyError as exc:
        raise KeyError(
            f"{metadata_path} lacks g2.modalities.video"
        ) from exc

    result: dict[str, tuple[int, int]] = {}
    for name in ("top_head", "hand_left", "hand_right"):
        try:
            width, height = [
                int(value)
                for value in video_meta[name]["resolution"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid G2 raw resolution metadata for {name!r}"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Non-positive G2 raw resolution for {name}: "
                f"{width}x{height}"
            )
        result[name] = (height, width)
    return result


def _save_readable_comparison_views(
    summary: dict,
    diagnostic_fps: int,
) -> None:
    """Save a slow MP4 and a 3x3 sheet for a short predicted chunk."""
    comparison_value = summary.get("comparison_video")
    if not comparison_value:
        return
    comparison_path = Path(comparison_value)
    frames = shared._read_mp4_rgb(comparison_path)
    if not frames:
        return
    if diagnostic_fps <= 0:
        raise ValueError(
            f"diagnostic_fps must be positive, got {diagnostic_fps}"
        )

    g2_comparison_path = comparison_path.with_name(
        "g2_predicted_vs_g2_gt_f9.mp4"
    )
    shutil.copy2(comparison_path, g2_comparison_path)
    for position in ("first", "last"):
        legacy_path = comparison_path.with_name(
            f"agibot_predicted_vs_g2_gt_{position}.png"
        )
        if legacy_path.is_file():
            shutil.copy2(
                legacy_path,
                comparison_path.with_name(
                    f"g2_predicted_vs_g2_gt_{position}.png"
                ),
            )

    slow_path = comparison_path.with_name(
        "g2_predicted_vs_gt_f9_slow_3fps.mp4"
    )
    shared._save_rgb_video(
        slow_path,
        frames,
        fps=diagnostic_fps,
    )

    target_width = 640
    resized = [
        shared.cv2.resize(
            frame,
            (
                target_width,
                max(
                    1,
                    round(frame.shape[0] * target_width / frame.shape[1]),
                ),
            ),
            interpolation=shared.cv2.INTER_AREA,
        )
        for frame in frames
    ]
    frame_height, frame_width = resized[0].shape[:2]
    black = np.zeros(
        (frame_height, frame_width, 3),
        dtype=np.uint8,
    )
    cells = resized[:9] + [black] * max(0, 9 - len(resized))
    rows = [
        np.concatenate(cells[index:index + 3], axis=1)
        for index in range(0, 9, 3)
    ]
    sheet_path = comparison_path.with_name(
        "g2_predicted_vs_gt_f9_contact_sheet.png"
    )
    shared.imageio.imwrite(
        sheet_path,
        np.concatenate(rows, axis=0),
    )
    summary["comparison_video"] = str(g2_comparison_path)
    summary["slow_comparison_video"] = str(slow_path)
    summary["comparison_contact_sheet"] = str(sheet_path)


def _parse_window_starts(value: str) -> list[int]:
    starts: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            start = int(token)
        except ValueError as exc:
            raise ValueError(
                f"window_starts must be comma-separated integers, got {value!r}"
            ) from exc
        if start < 0:
            raise ValueError(f"window start must be non-negative, got {start}")
        starts.append(start)
    if not starts:
        raise ValueError("window_starts must contain at least one start index")
    return starts


def _decode_window_latents(
    wrapper: shared.G2RoboarenaPolicy,
    latent_chunks: list[torch.Tensor],
    expected_frames: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Decode the concatenated causal chunks and return RGB frames plus shape."""
    if not latent_chunks:
        raise RuntimeError("The window produced no video latent chunks")

    latent = torch.cat(latent_chunks, dim=2)
    action_head = wrapper._policy.trained_model.action_head
    device = getattr(action_head, "_device", None)
    if device is None:
        device = next(action_head.parameters()).device
    latent = latent.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        decoded = action_head.vae.decode(
            latent,
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )
    decoded_shape = [int(value) for value in decoded.shape]
    frames = decoded[0].permute(1, 2, 3, 0)
    frames = (
        # Wan VAE returns RGB in [-1, 1].  Convert to uint8 exactly as the
        # online G2 server does; omitting the 127.5 scale makes every
        # predicted frame appear black (values collapse to 0 or 1).
        (frames.float() + 1.0)
        * 127.5
    ).clamp(0.0, 255.0)
    frames = (
        frames
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    if frames.shape[0] < expected_frames:
        raise RuntimeError(
            "Causal rollout decoded fewer frames than requested: "
            f"decoded={frames.shape[0]} requested={expected_frames} "
            f"latent_shape={tuple(latent.shape)}"
        )
    return [np.ascontiguousarray(frame) for frame in frames[:expected_frames]], decoded_shape


def _save_window_action_plot(
    path: Path,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    """Save a compact action plot without making plotting a hard dependency."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - diagnostic-only fallback
        logging.warning("Could not create action plot %s: %s", path, exc)
        return

    # Plot the first causal block; all four blocks are retained in the npy
    # files and the JSON metrics below.
    pred0 = predicted[0]
    gt0 = ground_truth[0]
    horizon = np.arange(1, pred0.shape[0] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(
        horizon,
        np.abs(pred0[:, :7] - gt0[:, :7]).mean(axis=1),
        label="arm MAE",
        color="tab:blue",
        linewidth=2,
    )
    axes[0].plot(
        horizon,
        np.abs(pred0[:, [7, 15]] - gt0[:, [7, 15]]).mean(axis=1),
        label="gripper MAE",
        color="tab:orange",
        linewidth=2,
    )
    axes[0].set_ylabel("absolute error")
    axes[0].set_title("Window 0 action error (same forward as video prediction)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        horizon,
        gt0[:, :7].mean(axis=1),
        label="GT arm mean",
        color="tab:green",
    )
    axes[1].plot(
        horizon,
        pred0[:, :7].mean(axis=1),
        label="PRED arm mean",
        color="tab:red",
    )
    axes[1].set_xlabel("action horizon step")
    axes[1].set_ylabel("joint mean")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _run_windowed_episode_comparison(
    wrapper: shared.G2RoboarenaPolicy,
    args: Args,
    info: dict,
    episode_index: int,
    raw_resolutions: dict[str, tuple[int, int]],
) -> dict:
    """Run contiguous 4-frame packets and compare a 33-frame causal rollout.

    One window is intentionally teacher-forced at the observation-packet level,
    matching deployment: packets ``0:4``, ``4:8``, ``8:12`` and ``12:16`` are
    sent through the *same* policy cache.  The first packet's latent output has
    three frames and the next three have two each; VAE decoding therefore gives
    exactly 33 frames.  The next requested window resets the cache and starts
    at the next four-frame packet (by default 4:8).
    """
    if args.window_history != 4:
        raise ValueError(
            "The causal G2 evaluator currently requires window_history=4; "
            f"got {args.window_history}"
        )
    if args.rollout_blocks != 4:
        raise ValueError(
            "The 33-frame causal rollout requires rollout_blocks=4; "
            f"got {args.rollout_blocks}"
        )
    if args.rollout_future_frames != 33:
        raise ValueError(
            "This diagnostic is defined for rollout_future_frames=33; "
            f"got {args.rollout_future_frames}"
        )

    root = Path(args.test_data_root).resolve()
    chunks_size = int(info.get("chunks_size", 1000))
    parquet_path = shared._episode_file_from_template(
        root,
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        ),
        episode_index,
        chunks_size,
    )
    table = pq.read_table(parquet_path)
    state = shared._column_to_numpy(table, "observation.state", np.float32)
    actions = shared._column_to_numpy(table, "action", np.float32)
    num_rows = int(len(state))
    if state.shape != (num_rows, 16) or actions.shape != (num_rows, 16):
        raise ValueError(
            "Windowed G2 diagnostic expects state/action shapes "
            f"({num_rows},16), got state={state.shape} action={actions.shape}"
        )

    starts = _parse_window_starts(args.window_starts)
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4",
    )
    camera_features = {
        "top_head": "observation.images.top_head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    }
    video_paths = {
        short_name: str(
            shared._episode_file_from_template(
                root,
                video_template,
                episode_index,
                chunks_size,
                video_key=feature_key,
            )
        )
        for short_name, feature_key in camera_features.items()
    }

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_comparison_frames: list[np.ndarray] = []
    all_predicted_actions: list[np.ndarray] = []
    all_ground_truth_actions: list[np.ndarray] = []
    language_key = "annotation.language.action_text"
    window_summaries: list[dict] = []
    for window_start in starts:
        first_packet = window_start
        last_packet = window_start + args.window_history * args.rollout_blocks - 1
        anchor = window_start + args.window_history - 1
        gt_video_last = anchor + args.rollout_future_frames - 1
        gt_action_last = max(
            window_start + (args.rollout_blocks - 1) * args.window_history
            + args.window_history - 1
            + 24 - 1,
            anchor + 24 - 1,
        )
        last_needed = max(last_packet, gt_video_last, gt_action_last)
        if first_packet < 0 or last_needed >= num_rows:
            raise IndexError(
                f"Window start={window_start} needs rows through {last_needed}, "
                f"but episode {episode_index} has {num_rows} rows"
            )

        decoded: dict[str, dict[int, np.ndarray]] = {}
        for short_name, path_value in video_paths.items():
            decoded[short_name] = shared._decode_video_range_bgr(
                Path(path_value),
                first_packet,
                last_needed,
            )

        window_dir = (
            output_root / f"episode_{episode_index:06d}_window_{window_start:06d}"
        )
        if args.save_window_artifacts:
            window_dir.mkdir(parents=True, exist_ok=True)

        # Keep each four-frame packet visible so there is no ambiguity about
        # what was sent to the model for each causal block.
        packet_indices: list[list[int]] = []
        for block_index in range(args.rollout_blocks):
            packet_start = window_start + block_index * args.window_history
            indices = list(
                range(packet_start, packet_start + args.window_history)
            )
            packet_indices.append(indices)
            if args.save_window_artifacts:
                input_grids = [
                    shared._grid_rgb(
                        decoded["top_head"][index],
                        decoded["hand_left"][index],
                        decoded["hand_right"][index],
                    )
                    for index in indices
                ]
                shared._save_rgb_video(
                    window_dir
                    / f"input_context_block_{block_index:02d}_f{len(input_grids)}.mp4",
                    input_grids,
                    fps=30,
                )

        gt_indices = list(
            range(anchor, anchor + args.rollout_future_frames)
        )
        gt_grids = [
            shared._grid_rgb(
                decoded["top_head"][index],
                decoded["hand_left"][index],
                decoded["hand_right"][index],
            )
            for index in gt_indices
        ]
        gt_path = window_dir / f"ground_truth_anchor_{anchor:06d}_f33.mp4"
        if args.save_window_artifacts:
            shared._save_rgb_video(gt_path, gt_grids, fps=30)

        # Force the actual four-frame history for this diagnostic even when a
        # checkpoint metadata file advertises eval_delta_indices=[0].  The
        # online G2 client sends four frames; this makes the comparison match
        # that packet contract while retaining the checkpoint for the model.
        wrapper._expected_video_frames = args.window_history
        wrapper._output_dir = str(window_dir) if args.save_window_artifacts else None
        wrapper._msg_index = 0
        wrapper._reset_state(save_video=False)

        predicted_actions: list[np.ndarray] = []
        latent_chunks: list[torch.Tensor] = []
        block_prompts: list[str] = []
        window_prompt: str | None = None
        session_id = f"g2-windowed-episode-{episode_index:06d}"
        for block_index, indices in enumerate(packet_indices):
            block_anchor = indices[-1]
            if args.prompt_override is not None:
                raw_prompt = args.prompt_override
            elif language_key in table.column_names:
                raw_prompt = shared._unwrap_text(
                    table[language_key][block_anchor].as_py()
                )
            else:
                raw_prompt = ""
            if window_prompt is None:
                window_prompt = raw_prompt
            # Keep language fixed across the four causal blocks.  A language
            # change resets WAN's current_start_frame and would invalidate the
            # intended 3+2+2+2 latent concatenation.
            prompt = window_prompt
            block_prompts.append(prompt)
            obs = {
                "observation/top_head": np.stack(
                    [decoded["top_head"][index] for index in indices]
                ),
                "observation/hand_left": np.stack(
                    [decoded["hand_left"][index] for index in indices]
                ),
                "observation/hand_right": np.stack(
                    [decoded["hand_right"][index] for index in indices]
                ),
                "observation/state": state[block_anchor],
                "prompt": prompt,
                "session_id": session_id,
            }
            predicted_actions.append(
                np.asarray(wrapper.infer(obs), dtype=np.float32)
            )

        latent_chunks = [
            chunk.detach().cpu().clone()
            for chunk in wrapper.video_across_time
        ]
        predicted_frames, decoded_shape = _decode_window_latents(
            wrapper,
            latent_chunks,
            args.rollout_future_frames,
        )
        predicted_path = window_dir / "predicted_rollout_f33.mp4"
        if args.save_window_artifacts:
            shared._save_rgb_video(predicted_path, predicted_frames, fps=30)

        comparison_frames = [
            np.concatenate(
                [
                    shared._add_label_rgb(predicted, "PREDICTED"),
                    shared._add_label_rgb(ground_truth, "G2 GROUND TRUTH"),
                ],
                axis=1,
            )
            for predicted, ground_truth in zip(predicted_frames, gt_grids)
        ]
        comparison_path = window_dir / "predicted_vs_ground_truth_f33.mp4"
        if args.save_window_artifacts:
            shared._save_rgb_video(comparison_path, comparison_frames, fps=30)
            shared.imageio.imwrite(
                window_dir / "predicted_vs_ground_truth_first.png",
                comparison_frames[0],
            )
            shared.imageio.imwrite(
                window_dir / "predicted_vs_ground_truth_last.png",
                comparison_frames[-1],
            )

        # Keep one chronological report video instead of exposing four tiny
        # request videos per window.  The windows are deliberately
        # teacher-forced, so each appended segment has a matching GT segment.
        full_comparison_frames.extend(comparison_frames)

        predicted_action_array = np.stack(predicted_actions)
        gt_action_array = np.stack(
            [
                actions[indices[-1] : indices[-1] + 24]
                for indices in packet_indices
            ]
        )
        if predicted_action_array.shape != gt_action_array.shape:
            raise RuntimeError(
                "Predicted/GT action shape mismatch: "
                f"pred={predicted_action_array.shape} gt={gt_action_array.shape}"
            )
        if args.save_window_artifacts:
            np.save(window_dir / "predicted_actions_4x24x16.npy", predicted_action_array)
            np.save(window_dir / "ground_truth_actions_4x24x16.npy", gt_action_array)
        action_abs_error = np.abs(predicted_action_array - gt_action_array)
        if args.save_window_artifacts:
            np.save(window_dir / "action_abs_error_4x24x16.npy", action_abs_error)
            _save_window_action_plot(
                window_dir / "action_pred_vs_gt.png",
                predicted_action_array,
                gt_action_array,
            )

        all_predicted_actions.append(predicted_action_array)
        all_ground_truth_actions.append(gt_action_array)

        action_metrics = {
            "overall_mae": float(action_abs_error.mean()),
            "arm_mae": float(
                action_abs_error[:, :, [*range(7), *range(8, 15)]].mean()
            ),
            "gripper_mae": float(
                action_abs_error[:, :, [7, 15]].mean()
            ),
            "block_mae": action_abs_error.mean(axis=(1, 2)).tolist(),
            "horizon_mae": action_abs_error.mean(axis=(0, 2)).tolist(),
        }
        summary = {
            "checkpoint": str(Path(args.model_path).resolve()),
            "test_data_root": str(root),
            "episode_index": episode_index,
            "window_start": window_start,
            "packet_indices": packet_indices,
            "anchor_indices": [indices[-1] for indices in packet_indices],
            "gt_video_indices": gt_indices,
            "prompt_per_block": block_prompts,
            "checkpoint_eval_offsets": _checkpoint_eval_offsets(
                Path(args.model_path).resolve(), "g2"
            ),
            "forced_model_context_frames": args.window_history,
            "rollout_blocks": args.rollout_blocks,
            "rollout_future_frames": args.rollout_future_frames,
            "latent_chunks": [list(chunk.shape) for chunk in latent_chunks],
            "decoded_shape_bcthw": decoded_shape,
            "action_metrics": action_metrics,
            "same_forward_video_action": True,
            "teacher_forced_observation_packets": True,
        }
        if args.save_window_artifacts:
            summary.update(
                {
                    "predicted_video": str(predicted_path),
                    "ground_truth_video": str(gt_path),
                    "comparison_video": str(comparison_path),
                    "predicted_actions": str(
                        window_dir / "predicted_actions_4x24x16.npy"
                    ),
                    "ground_truth_actions": str(
                        window_dir / "ground_truth_actions_4x24x16.npy"
                    ),
                }
            )
            with (window_dir / "summary.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2)
        window_summaries.append(summary)

        # Reset the causal cache and frame buffer before the next independent
        # window, without writing the wrapper's generic concatenated MP4.  The
        # reset signal must reach rank 1 as well; otherwise the next window
        # mixes rank-0 and rank-1 KV caches and can stall or abort.
        wrapper._broadcast_signal_to_workers(shared.SIGNAL_RESET_CACHE)
        wrapper._reset_state(save_video=False)

    if not all_predicted_actions:
        raise RuntimeError("Windowed rollout produced no action predictions")

    predicted_sequence = np.stack(all_predicted_actions, axis=0)
    ground_truth_sequence = np.stack(all_ground_truth_actions, axis=0)
    sequence_error = np.abs(predicted_sequence - ground_truth_sequence)
    action_names = (
        [f"left_joint_{index}" for index in range(7)]
        + ["left_gripper"]
        + [f"right_joint_{index}" for index in range(7)]
        + ["right_gripper"]
    )

    # One detailed, machine-readable table for the complete selected rollout.
    detail_csv = output_root / (
        f"episode_{episode_index:06d}_action_pred_vs_gt.csv"
    )
    detail_fields = [
        "window_start",
        "block_index",
        "anchor_frame",
        "horizon_step",
    ] + [f"gt_{name}" for name in action_names] + [
        f"pred_{name}" for name in action_names
    ]
    with detail_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields)
        writer.writeheader()
        for window_index, window_start in enumerate(starts):
            for block_index in range(args.rollout_blocks):
                anchor_index = (
                    window_start + block_index * args.window_history
                    + args.window_history - 1
                )
                for horizon_index in range(24):
                    gt_row = ground_truth_sequence[
                        window_index, block_index, horizon_index
                    ]
                    pred_row = predicted_sequence[
                        window_index, block_index, horizon_index
                    ]
                    row = {
                        "window_start": window_start,
                        "block_index": block_index,
                        "anchor_frame": anchor_index,
                        "horizon_step": horizon_index + 1,
                    }
                    row.update(
                        {
                            f"gt_{name}": float(value)
                            for name, value in zip(action_names, gt_row)
                        }
                    )
                    row.update(
                        {
                            f"pred_{name}": float(value)
                            for name, value in zip(action_names, pred_row)
                        }
                    )
                    writer.writerow(row)

    def _metric_row(label: str, error: np.ndarray) -> dict[str, object]:
        return {
            "scope": label,
            "overall_mae": float(error.mean()),
            "arm_mae": float(
                error[..., [*range(7), *range(8, 15)]].mean()
            ),
            "gripper_mae": float(error[..., [7, 15]].mean()),
            "left_arm_mae": float(error[..., :7].mean()),
            "right_arm_mae": float(error[..., 8:15].mean()),
            "num_action_rows": int(np.prod(error.shape[:-1])),
        }

    metrics_csv = output_root / (
        f"episode_{episode_index:06d}_action_metrics.csv"
    )
    metric_fields = [
        "scope",
        "window_start",
        "block_index",
        "anchor_frame",
        "overall_mae",
        "arm_mae",
        "gripper_mae",
        "left_arm_mae",
        "right_arm_mae",
        "num_action_rows",
    ]
    metric_rows: list[dict[str, object]] = []
    for window_index, window_summary in enumerate(window_summaries):
        window_error = sequence_error[window_index]
        for block_index in range(args.rollout_blocks):
            block_error = window_error[block_index]
            metric = _metric_row(
                f"window_{window_summary['window_start']:06d}_block_{block_index:02d}",
                block_error[None, ...],
            )
            metric.update(
                {
                    "window_start": window_summary["window_start"],
                    "block_index": block_index,
                    "anchor_frame": window_summary["anchor_indices"][block_index],
                }
            )
            metric_rows.append(metric)
    overall_metric = _metric_row("overall", sequence_error)
    overall_metric.update(
        {"window_start": "", "block_index": "", "anchor_frame": ""}
    )
    metric_rows.append(overall_metric)
    with metrics_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    # Compact visual summary: means for each arm plus the absolute error.  It
    # is intentionally one figure for the whole selected rollout, not one
    # figure per 24-step request.
    action_plot = output_root / (
        f"episode_{episode_index:06d}_action_pred_vs_gt.png"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        flat_pred = predicted_sequence.reshape(-1, 16)
        flat_gt = ground_truth_sequence.reshape(-1, 16)
        flat_error = sequence_error.reshape(-1, 16)
        x = np.arange(1, len(flat_pred) + 1)
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        axes[0].plot(x, flat_gt[:, :7].mean(axis=1), label="GT left arm", color="tab:blue")
        axes[0].plot(x, flat_pred[:, :7].mean(axis=1), label="PRED left arm", color="tab:orange")
        axes[0].set_ylabel("left arm mean")
        axes[0].legend(loc="upper right")
        axes[1].plot(x, flat_gt[:, 8:15].mean(axis=1), label="GT right arm", color="tab:green")
        axes[1].plot(x, flat_pred[:, 8:15].mean(axis=1), label="PRED right arm", color="tab:red")
        axes[1].set_ylabel("right arm mean")
        axes[1].legend(loc="upper right")
        axes[2].plot(x, flat_error[:, [*range(7), *range(8, 15)]].mean(axis=1), label="arm MAE")
        axes[2].plot(x, flat_error[:, [7, 15]].mean(axis=1), label="gripper MAE")
        axes[2].set_ylabel("absolute error")
        axes[2].set_xlabel("concatenated action row (window/block/horizon)")
        axes[2].legend(loc="upper right")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle(
            f"G2 checkpoint action: predicted vs ground truth, episode {episode_index}"
        )
        fig.tight_layout()
        fig.savefig(action_plot, dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - diagnostic-only fallback
        logging.warning("Could not create sequence action plot %s: %s", action_plot, exc)
        action_plot = None

    full_comparison_path = output_root / (
        f"episode_{episode_index:06d}_full_predicted_vs_ground_truth_"
        f"f{len(full_comparison_frames)}.mp4"
    )
    shared._save_rgb_video(full_comparison_path, full_comparison_frames, fps=30)

    report = {
        "checkpoint": str(Path(args.model_path).resolve()),
        "test_data_root": str(root),
        "episode_index": episode_index,
        "window_starts": starts,
        "window_history": args.window_history,
        "window_stride": args.window_stride,
        "rollout_blocks": args.rollout_blocks,
        "rollout_future_frames": args.rollout_future_frames,
        "video_paths": video_paths,
        "full_comparison_video": str(full_comparison_path),
        "action_detail_csv": str(detail_csv),
        "action_metrics_csv": str(metrics_csv),
        "action_plot": str(action_plot) if action_plot else None,
        "num_comparison_frames": len(full_comparison_frames),
        "save_window_artifacts": bool(args.save_window_artifacts),
        "overall_action_metrics": overall_metric,
        "windows": window_summaries,
    }
    report_path = (
        Path(args.output_dir).resolve()
        / f"episode_{episode_index:06d}_windowed_report.json"
    )
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def _run_full_episode_comparison(
    wrapper: shared.G2RoboarenaPolicy,
    args: Args,
    test_info: dict,
    episode_index: int,
    eval_offsets: list[int],
    raw_resolutions: dict[str, tuple[int, int]],
) -> dict:
    """Roll through one complete held-out episode and keep one final MP4."""
    root = Path(args.test_data_root).resolve()
    chunks_size = int(test_info.get("chunks_size", 1000))
    parquet_path = shared._episode_file_from_template(
        root,
        test_info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        ),
        episode_index,
        chunks_size,
    )
    table = pq.read_table(parquet_path)
    num_rows = table.num_rows
    ground_truth_actions = shared._column_to_numpy(
        table, "action", np.float32
    )
    if ground_truth_actions.shape != (num_rows, 16):
        raise ValueError(
            f"Expected G2 ground-truth action shape {(num_rows, 16)}, "
            f"got {ground_truth_actions.shape}"
        )
    first_row = max(0, -min(eval_offsets))
    last_row = max(first_row, num_rows - max(args.future_frames, 9))
    stride = max(1, int(args.rollout_stride))

    output_root = Path(args.output_dir).resolve()
    temp_root = output_root / f".full_rollout_episode_{episode_index:06d}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    predicted_horizons: list[np.ndarray] = []
    ground_truth_horizons: list[np.ndarray] = []
    rollout_rows: list[int] = []
    chunks = 0
    session_id = f"g2-full-rollout-episode-{episode_index:06d}"
    for row_index in range(first_row, last_row + 1, stride):
        chunk_args = dataclasses.replace(
            args,
            frame_index=row_index,
            output_dir=str(temp_root),
            full_episode=False,
            session_id_override=session_id,
        )
        summary = shared._run_one_test_sample(
            wrapper,
            chunk_args,
            test_info,
            episode_index,
            eval_offsets,
            raw_resolutions,
        )
        action_path = (
            temp_root
            / f"episode_{episode_index:06d}_frame_{row_index:06d}"
            / "ignored_hold_actions.npy"
        )
        predicted = np.load(action_path)
        if predicted.shape != (24, 16):
            raise ValueError(
                f"Expected predicted action shape (24, 16), got {predicted.shape}"
            )
        ground_truth_horizon = ground_truth_actions[
            row_index:row_index + 24
        ]
        if ground_truth_horizon.shape != (24, 16):
            raise ValueError(
                "Expected ground-truth action horizon shape (24, 16), "
                f"got {ground_truth_horizon.shape} at row {row_index}"
            )
        predicted_horizons.append(predicted.astype(np.float32))
        ground_truth_horizons.append(
            ground_truth_horizon.astype(np.float32)
        )
        rollout_rows.append(row_index)
        comparison = summary.get("comparison_video")
        if comparison and not args.action_eval:
            frames.extend(shared._read_mp4_rgb(Path(comparison)))
        chunks += 1

    if args.action_eval:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred = np.stack(predicted_horizons)
        gt = np.stack(ground_truth_horizons)
        rows = np.asarray(rollout_rows)
        checkpoint_name = Path(args.model_path).name.replace("checkpoint-", "")
        plot_specs = [
            ("left_arm", list(range(7))),
            ("right_arm", list(range(8, 15))),
            ("grippers", [7, 15]),
        ]
        plot_paths: list[str] = []
        for label, indices in plot_specs:
            motion_score = np.ptp(
                gt[:, :, indices], axis=1
            ).mean(axis=1)
            window_index = int(np.argmax(motion_score))
            selected_row = int(rows[window_index])
            horizon_x = np.arange(1, 25)
            fig, axes = plt.subplots(
                len(indices), 1, figsize=(14, max(4, 2.2 * len(indices))),
                sharex=True,
            )
            axes = np.atleast_1d(axes)
            for axis, dim in zip(axes, indices):
                axis.plot(
                    horizon_x,
                    gt[window_index, :, dim],
                    color="tab:blue",
                    linewidth=2.0,
                    label="GT",
                )
                axis.plot(
                    horizon_x,
                    pred[window_index, :, dim],
                    color="tab:orange",
                    linewidth=1.8,
                    label="PRED",
                )
                axis.set_ylabel(f"dim {dim}")
                axis.grid(alpha=0.25)
            axes[0].legend()
            axes[-1].set_xticks([1, 4, 8, 12, 16, 20, 24])
            axes[-1].set_xlabel("future action step")
            fig.suptitle(
                f"G2 checkpoint-{checkpoint_name}: {label}, "
                f"episode row {selected_row}"
            )
            fig.tight_layout()
            plot_path = output_root / (
                f"g2_checkpoint{checkpoint_name}_action_horizon_"
                f"{label}_pred_vs_gt.png"
            )
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(plot_path))

        absolute_error = np.abs(pred - gt)
        arm_indices = list(range(7)) + list(range(8, 15))
        fig, axis = plt.subplots(figsize=(10, 5))
        horizon_steps = np.arange(1, 25)
        axis.plot(
            horizon_steps,
            absolute_error[:, :, arm_indices].mean(axis=(0, 2)),
            linewidth=2.0,
            label="arm MAE",
        )
        axis.plot(
            horizon_steps,
            absolute_error[:, :, [7, 15]].mean(axis=(0, 2)),
            linewidth=2.0,
            label="gripper MAE",
        )
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
        axis.set_xlabel("predicted horizon step")
        axis.set_ylabel("mean absolute error")
        axis.set_title(
            f"G2 checkpoint-{checkpoint_name}: 24-step action horizon error"
        )
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        error_path = output_root / (
            f"g2_checkpoint{checkpoint_name}_action_horizon_mae.png"
        )
        fig.savefig(error_path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(error_path))
        shutil.rmtree(temp_root)
        return {
            "episode_index": episode_index,
            "rollout_chunks": chunks,
            "action_plots": plot_paths,
        }

    if not frames:
        raise RuntimeError(
            f"Full rollout produced no comparison frames for episode {episode_index}"
        )
    checkpoint_name = Path(args.model_path).name.replace("checkpoint-", "")
    final_path = output_root / (
        f"g2_checkpoint{checkpoint_name}_full_pred_vs_gt.mp4"
    )
    shared._save_rgb_video(final_path, frames, fps=args.diagnostic_fps)
    shutil.rmtree(temp_root)
    return {
        "episode_index": episode_index,
        "rollout_chunks": chunks,
        "comparison_video": str(final_path),
    }


def main(args: Args) -> None:
    if args.embodiment_tag.lower() != "g2":
        raise ValueError("This evaluator requires --embodiment-tag g2")
    if args.future_frames <= 0:
        raise ValueError("--future-frames must be positive")

    os.environ["ENABLE_DIT_CACHE"] = (
        "true" if args.enable_dit_cache else "false"
    )
    if args.num_dit_steps is not None:
        os.environ["NUM_DIT_STEPS"] = str(args.num_dit_steps)
    elif args.enable_dit_cache:
        os.environ.setdefault("NUM_DIT_STEPS", "8")
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch._dynamo.config.recompile_limit = 800

    model_path = Path(args.model_path).resolve()
    shared._require_dir(model_path, "G2 checkpoint")
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    conf_path = model_path / "experiment_cfg" / "conf.yaml"
    shared._require_file(metadata_path, "G2 checkpoint metadata")
    shared._require_file(conf_path, "G2 checkpoint config")
    with metadata_path.open("r", encoding="utf-8") as stream:
        checkpoint_metadata = json.load(stream)
    if "g2" not in checkpoint_metadata:
        raise KeyError(
            f"Checkpoint does not contain G2 metadata: "
            f"{metadata_path}; keys={list(checkpoint_metadata)}"
        )

    test_info = shared._validate_test_dataset(
        Path(args.test_data_root).resolve()
    )
    episode_indices = shared._parse_episode_indices(
        args.episode_indices,
        int(test_info["total_episodes"]),
    )
    eval_offsets = shared._checkpoint_eval_offsets(model_path, "g2")
    raw_resolutions = _checkpoint_g2_video_resolutions(model_path)
    model_config_overrides, train_config_overrides = (
        shared._build_path_overrides(args)
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(
        "[PREFLIGHT OK] checkpoint=%s checkpoint_embodiment=g2 "
        "visual_source=g2 test=%s episodes=%s eval_offsets=%s "
        "raw_resolutions=%s output=%s",
        model_path,
        Path(args.test_data_root).resolve(),
        episode_indices,
        eval_offsets,
        {
            key: [value[1], value[0]]
            for key, value in raw_resolutions.items()
        },
        Path(args.output_dir).resolve(),
    )
    if args.preflight_only:
        logging.info(
            "Preflight-only validation completed; model was not loaded."
        )
        return

    device_mesh = shared.init_mesh()
    rank = dist.get_rank()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    signal_group = dist.new_group(
        backend="gloo",
        timeout=datetime.timedelta(seconds=args.timeout_seconds),
    )
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("g2"),
        model_path=str(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        model_config_overrides=model_config_overrides,
        train_config_overrides=train_config_overrides,
    )
    action_head = policy.trained_model.action_head
    if args.num_inference_timesteps < 0:
        raise ValueError(
            "--num-inference-timesteps must be non-negative"
        )
    if args.num_inference_timesteps > 0:
        action_head.num_inference_steps = int(
            args.num_inference_timesteps
        )
        action_head.num_inference_timesteps = int(
            args.num_inference_timesteps
        )
        if hasattr(action_head, "config"):
            action_head.config.num_inference_timesteps = int(
                args.num_inference_timesteps
            )

    if rank == 0:
        if args.windowed:
            # The windowed diagnostic intentionally keeps video and action in
            # one policy object and one sequence of causal forwards.
            wrapper = shared.G2RoboarenaPolicy(
                groot_policy=policy,
                signal_group=signal_group,
                output_dir=str(Path(args.output_dir).resolve()),
                video_save_mode="full",
            )
            try:
                reports = [
                    _run_windowed_episode_comparison(
                        wrapper,
                        args,
                        test_info,
                        episode_index,
                        raw_resolutions,
                    )
                    for episode_index in episode_indices
                ]
            finally:
                wrapper._broadcast_signal_to_workers(shared.SIGNAL_SHUTDOWN)

            report_path = (
                Path(args.output_dir).resolve() / "windowed_report.json"
            )
            with report_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "checkpoint": str(model_path),
                        "checkpoint_embodiment": "g2",
                        "visual_source_embodiment": "g2",
                        "test_data_root": str(
                            Path(args.test_data_root).resolve()
                        ),
                        "episode_indices": episode_indices,
                        "reports": reports,
                    },
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
            logging.info("Saved windowed report: %s", report_path)
            dist.barrier()
            dist.destroy_process_group()
            return

        wrapper_class = (
            shared.G2RoboarenaPolicy
            if args.action_eval
            else G2VideoDiagnosticPolicy
        )
        wrapper = wrapper_class(
            groot_policy=policy,
            signal_group=signal_group,
            output_dir=str(Path(args.output_dir).resolve()),
            video_save_mode="none",
        )
        summaries: list[dict] = []
        try:
            for episode_index in episode_indices:
                if args.full_episode:
                    summary = _run_full_episode_comparison(
                        wrapper,
                        args,
                        test_info,
                        episode_index,
                        eval_offsets,
                        raw_resolutions,
                    )
                else:
                    summary = shared._run_one_test_sample(
                        wrapper,
                        args,
                        test_info,
                        episode_index,
                        eval_offsets,
                        raw_resolutions,
                    )
                summary["checkpoint_embodiment"] = "g2"
                summary["g2_state_used_for_model_conditioning"] = True
                summary["action_note"] = (
                    "Actions were deliberately ignored; this run evaluates "
                    "only the predicted video."
                )
                if not args.full_episode:
                    _save_readable_comparison_views(
                        summary,
                        args.diagnostic_fps,
                    )
                summaries.append(summary)
        finally:
            wrapper._broadcast_signal_to_workers(shared.SIGNAL_SHUTDOWN)

        report_path = (
            Path(args.output_dir).resolve() / "testset_report.json"
        )
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "checkpoint": str(model_path),
                    "checkpoint_embodiment": "g2",
                    "visual_source_embodiment": "g2",
                    "test_data_root": str(
                        Path(args.test_data_root).resolve()
                    ),
                    "episode_indices": episode_indices,
                    "eval_offsets": eval_offsets,
                    "checkpoint_raw_resolutions": {
                        key: [value[1], value[0]]
                        for key, value in raw_resolutions.items()
                    },
                    "samples": summaries,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
        logging.info("Saved test-set report: %s", report_path)
        dist.barrier()
    else:
        worker = shared.WebsocketPolicyServer(
            policy=policy,
            host="127.0.0.1",
            port=0,
            metadata={},
            output_dir=None,
            signal_group=signal_group,
        )
        asyncio.run(worker._worker_loop())
        dist.barrier()

    dist.destroy_process_group()


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
