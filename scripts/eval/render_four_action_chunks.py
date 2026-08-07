#!/usr/bin/env python3
"""Render one compact all-16D plot containing every saved action chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ACTION_NAMES = (
    *[f"left_joint_{index}" for index in range(7)],
    "left_gripper",
    *[f"right_joint_{index}" for index in range(7)],
    "right_gripper",
)
COLORS = {
    "baseline": "tab:blue",
    "state_only": "tab:red",
    "history_only": "tab:green",
    "state_history_swap": "tab:purple",
}
LINESTYLES = {
    "baseline": "-",
    "state_only": "--",
    "history_only": ":",
    "state_history_swap": "-.",
}


def render(pair_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays_path = next(pair_dir.glob("*_action_arrays.npz"))
    report_path = next(pair_dir.glob("*_report.json"))
    arrays = np.load(arrays_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ground_truth = arrays["ground_truth"]
    conditions = [
        condition
        for condition in COLORS
        if f"pred_{condition}" in arrays.files
    ]
    chunks, horizon, dimensions = ground_truth.shape
    if dimensions != 16:
        raise RuntimeError(f"Expected 16 action dimensions, got {ground_truth.shape}")
    total_rows = chunks * horizon
    x = np.arange(1, total_rows + 1)
    gt_unrolled = ground_truth.reshape(total_rows, dimensions)
    predictions = {
        condition: arrays[f"pred_{condition}"].reshape(total_rows, dimensions)
        for condition in conditions
    }

    target_episode = int(report["target_episode"])
    source_episode = int(report["donor_episode"])
    output_path = pair_dir / (
        f"episode_{target_episode:06d}_source_{source_episode:06d}_"
        f"all16_{chunks}chunks_x_{horizon}steps.png"
    )
    fig, axes = plt.subplots(4, 4, figsize=(24, 16), sharex=True)
    for axis, dim in zip(axes.flat, range(dimensions)):
        axis.plot(x, gt_unrolled[:, dim], color="black", linewidth=1.8, label="GT")
        for condition in conditions:
            axis.plot(
                x,
                predictions[condition][:, dim],
                color=COLORS[condition],
                linestyle=LINESTYLES[condition],
                linewidth=1.65,
                alpha=0.82,
                label=condition,
            )
        for chunk_index in range(1, chunks):
            axis.axvline(
                chunk_index * horizon + 0.5,
                color="0.35",
                linestyle="--",
                linewidth=0.9,
            )
        axis.set_title(f"{dim}: {ACTION_NAMES[dim]}")
        axis.set_xlabel(f"all action rows ({chunks} chunks x {horizon} steps)")
        axis.set_ylabel("joint position")
        axis.grid(alpha=0.20)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"G2 all 16 action dimensions | target episode {target_episode} | "
        f"swap-source episode {source_episode} | {chunks} chunks x {horizon} steps",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for pair_dir in sorted(args.root.glob("target_*_donor_*")):
        if pair_dir.is_dir():
            print(render(pair_dir))


if __name__ == "__main__":
    main()
