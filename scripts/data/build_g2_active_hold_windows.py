#!/usr/bin/env python3
"""Build the active/hold sampling index for an existing policy-space G2 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARM_DIMS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
GRIP_DIMS = (7, 15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.split / "meta/g2_active_hold_windows.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")

    metrics: list[tuple[int, int, float, bool]] = []
    for path in sorted(args.split.glob("data/chunk-*/*.parquet")):
        frame = pd.read_parquet(path, columns=["observation.state", "action", "episode_index"])
        state = np.stack(frame["observation.state"]).astype(np.float32)
        action = np.stack(frame["action"]).astype(np.float32)
        if state.shape[1:] != (16,) or action.shape[1:] != (16,):
            raise ValueError(f"{path}: expected 16D state/action")
        if np.any(state[:, GRIP_DIMS] < 0) or np.any(state[:, GRIP_DIMS] > 1):
            raise ValueError(f"{path}: state gripper is not in policy space [0,1]")
        if np.any(action[:, GRIP_DIMS] < 0) or np.any(action[:, GRIP_DIMS] > 1):
            raise ValueError(f"{path}: action gripper is not in policy space [0,1]")
        episode = int(frame["episode_index"].iloc[0])
        for step in range(max(0, len(frame) - args.action_horizon + 1)):
            future = action[step : step + args.action_horizon]
            arm_motion = float(
                np.linalg.norm(future[:, ARM_DIMS] - state[step, ARM_DIMS], axis=1).max()
            )
            grip_transition = bool(
                np.any(np.abs(future[:, GRIP_DIMS] - state[step, GRIP_DIMS]) > 0.25)
            )
            metrics.append((episode, step, arm_motion, grip_transition))

    nonzero = np.asarray([motion for _, _, motion, _ in metrics if motion > 1e-8])
    if not nonzero.size:
        raise RuntimeError("No nonzero G2 arm motion found")
    threshold = float(np.quantile(nonzero, 0.30))
    active: dict[str, list[int]] = {}
    hold: dict[str, list[int]] = {}
    for episode, step, motion, grip_transition in metrics:
        target = active if motion >= threshold or grip_transition else hold
        target.setdefault(str(episode), []).append(step)

    payload = {
        "schema_version": 1,
        "action_horizon": args.action_horizon,
        "arm_motion_threshold_rule": "p30_nonzero_24_step_l2",
        "arm_motion_threshold": threshold,
        "gripper_transition_threshold": 0.25,
        "active_count": sum(map(len, active.values())),
        "hold_count": sum(map(len, hold.values())),
        "active": active,
        "hold": hold,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {output}: threshold={threshold:.8f} "
        f"active={payload['active_count']} hold={payload['hold_count']}"
    )


if __name__ == "__main__":
    main()
