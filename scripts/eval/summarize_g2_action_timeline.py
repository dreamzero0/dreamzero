#!/usr/bin/env python3
"""Turn overlapping 24-step action chunks into a chronological task curve.

The full-anchor evaluator intentionally preserves every [anchor, horizon]
comparison.  This helper also selects the latest prediction available for
each episode frame, which is the useful receding-horizon view for deployment;
it avoids the misleading plot where overlapping chunks are simply flattened
end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ACTION_NAMES = (
    [f"left_joint_{i}" for i in range(7)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(7)]
    + ["right_gripper"]
)
ARM_DIMS = [*range(7), *range(8, 15)]
GRIPPER_DIMS = [7, 15]


def _metric(name: str, error: np.ndarray) -> dict[str, object]:
    return {
        "scope": name,
        "overall_mae": float(error.mean()),
        "arm_mae": float(error[..., ARM_DIMS].mean()),
        "gripper_mae": float(error[..., GRIPPER_DIMS].mean()),
        "left_arm_mae": float(error[..., :7].mean()),
        "right_arm_mae": float(error[..., 8:15].mean()),
        "num_action_rows": int(error.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out = (args.output_dir or args.input_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = out / f"episode_{args.episode_index:06d}_full_action_arrays.npz"
    if not source.is_file():
        source = args.input_dir.resolve() / f"episode_{args.episode_index:06d}_full_action_arrays.npz"
    data = np.load(source)
    predicted = np.asarray(data["predicted"], dtype=np.float32)
    ground_truth = np.asarray(data["ground_truth"], dtype=np.float32)
    anchors = np.asarray(data["anchors"], dtype=np.int64)
    if predicted.ndim != 3 or predicted.shape != ground_truth.shape or predicted.shape[-1] != 16:
        raise ValueError(f"Expected matching [blocks,24,16] arrays, got {predicted.shape} and {ground_truth.shape}")

    # Later anchors overwrite earlier ones.  This is the receding-horizon
    # action that would be current at each frame when replanning every packet.
    rows: dict[int, tuple[int, int, np.ndarray, np.ndarray]] = {}
    coverage: dict[int, int] = {}
    for block, anchor in enumerate(anchors.tolist()):
        for horizon in range(predicted.shape[1]):
            frame = int(anchor) + horizon
            rows[frame] = (block, horizon, predicted[block, horizon], ground_truth[block, horizon])
            coverage[frame] = coverage.get(frame, 0) + 1
    frames = np.asarray(sorted(rows), dtype=np.int64)
    block_used = np.asarray([rows[int(frame)][0] for frame in frames], dtype=np.int64)
    horizon_used = np.asarray([rows[int(frame)][1] for frame in frames], dtype=np.int64)
    pred = np.stack([rows[int(frame)][2] for frame in frames]).astype(np.float32)
    gt = np.stack([rows[int(frame)][3] for frame in frames]).astype(np.float32)
    counts = np.asarray([coverage[int(frame)] for frame in frames], dtype=np.int64)
    err = np.abs(pred - gt)

    np.savez_compressed(
        out / f"episode_{args.episode_index:06d}_full_action_timeline_arrays.npz",
        frame_indices=frames,
        predicted_latest=pred,
        ground_truth=gt,
        block_used=block_used,
        horizon_used=horizon_used,
        coverage_count=counts,
    )
    csv_path = out / f"episode_{args.episode_index:06d}_full_action_timeline_pred_vs_gt.csv"
    fields = ["frame_index", "block_used", "horizon_step", "coverage_count"]
    fields += [f"gt_{name}" for name in ACTION_NAMES]
    fields += [f"pred_{name}" for name in ACTION_NAMES]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, frame in enumerate(frames):
            row: dict[str, object] = {
                "frame_index": int(frame),
                "block_used": int(block_used[i]),
                "horizon_step": int(horizon_used[i]) + 1,
                "coverage_count": int(counts[i]),
            }
            row.update({f"gt_{name}": float(value) for name, value in zip(ACTION_NAMES, gt[i])})
            row.update({f"pred_{name}": float(value) for name, value in zip(ACTION_NAMES, pred[i])})
            writer.writerow(row)

    plot_path = out / f"episode_{args.episode_index:06d}_full_action_timeline_pred_vs_gt.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 4, figsize=(22, 13), sharex=True)
        for dim, axis in enumerate(axes.flat):
            axis.plot(frames, gt[:, dim], color="tab:blue", lw=0.8, label="GT")
            axis.plot(frames, pred[:, dim], color="tab:orange", lw=0.8, label="PRED latest")
            axis.set_title(f"{dim}: {ACTION_NAMES[dim]}")
            axis.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(f"G2 chronological latest-anchor action vs GT, episode {args.episode_index}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    except Exception:
        plot_path = None

    report = {
        "episode_index": args.episode_index,
        "source_arrays": str(source),
        "timeline_frames": int(len(frames)),
        "frame_start": int(frames[0]) if len(frames) else None,
        "frame_end": int(frames[-1]) if len(frames) else None,
        "selection": "latest prediction for each frame among overlapping 24-step chunks",
        "timeline_metrics": _metric("latest_timeline", err),
        "timeline_csv": str(csv_path),
        "timeline_plot": str(plot_path) if plot_path else None,
    }
    report_path = out / f"episode_{args.episode_index:06d}_full_action_timeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
