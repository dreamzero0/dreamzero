"""Replay one captured G2 observation against a websocket policy server.

This diagnostic never talks to the robot SDK and never executes actions.  It
can repeat the exact same observation under one session ID to expose how the
server's causal cache changes consecutive action chunks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import uuid

import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Fresh UUID by default so the server resets its causal cache.",
    )
    parser.add_argument(
        "--left-gripper",
        type=float,
        default=None,
        help="Optional counterfactual override for state_16[7].",
    )
    return parser.parse_args()


def _build_observation(
    snapshot: np.lib.npyio.NpzFile,
    *,
    session_id: str,
    left_gripper: float | None,
) -> tuple[dict[str, object], np.ndarray]:
    state = np.asarray(snapshot["state_16"], dtype=np.float32)
    if state.shape != (16,):
        raise ValueError(f"Expected state_16 shape (16,), got {state.shape}")
    state = state.copy()
    if left_gripper is not None:
        state[7] = np.float32(left_gripper)
    return {
        "observation/top_head": np.asarray(
            snapshot["top_head_jpeg_decoded"], dtype=np.uint8
        ),
        "observation/hand_left": np.asarray(
            snapshot["hand_left_jpeg_decoded"], dtype=np.uint8
        ),
        "observation/hand_right": np.asarray(
            snapshot["hand_right_jpeg_decoded"], dtype=np.uint8
        ),
        "observation/left_joint_position": state[0:7],
        "observation/left_gripper_position": state[7:8],
        "observation/right_joint_position": state[8:15],
        "observation/right_gripper_position": state[15:16],
        "prompt": str(snapshot["prompt"].item()),
        "session_id": session_id,
    }, state


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    snapshot = np.load(args.snapshot, allow_pickle=False)
    session_id = args.session_id or f"snapshot-replay-{uuid.uuid4()}"
    observation, state = _build_observation(
        snapshot,
        session_id=session_id,
        left_gripper=args.left_gripper,
    )
    client = WebsocketClientPolicy(host=args.host, port=args.port)

    chunks: list[np.ndarray] = []
    for index in range(args.repeats):
        action = np.asarray(client.infer(dict(observation)), dtype=np.float32)
        if action.shape != (24, 16):
            raise ValueError(
                f"Replay {index} returned {action.shape}, expected (24,16)"
            )
        chunks.append(action)
        arm_delta = np.concatenate(
            (action[:, 0:7] - state[None, 0:7],
             action[:, 8:15] - state[None, 8:15]),
            axis=1,
        )
        logging.info(
            "chunk=%d first_arm_max_delta=%.6f "
            "full_arm_max_delta=%.6f left_grip=[%.6f,%.6f] "
            "right_grip=[%.6f,%.6f]",
            index,
            float(np.max(np.abs(arm_delta[0]))),
            float(np.max(np.abs(arm_delta))),
            float(action[:, 7].min()),
            float(action[:, 7].max()),
            float(action[:, 15].min()),
            float(action[:, 15].max()),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        replay_actions=np.stack(chunks),
        production_actions=np.asarray(
            snapshot["raw_server_actions"], dtype=np.float32
        ),
        state_16=state,
        prompt=np.asarray(str(snapshot["prompt"].item())),
        session_id=np.asarray(session_id),
    )
    logging.info("Saved replay comparison to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
