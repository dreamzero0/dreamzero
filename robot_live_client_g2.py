#!/usr/bin/env python3
"""DreamZero live-policy client for AgiBot G2.

The policy/inference loop is shared with :mod:`robot_live_client`.  This file
adapts the current ``agibot_gdk`` G2 API to the small camera/robot interface
used by that loop.  Run after sourcing ``~/.cache/agibot/app/env.sh``.

G2 has a three-axis head while the trained policy has two head dimensions.
They are mapped to yaw (idx11) and pitch (idx13); roll (idx12) is preserved.
G2's five-motor parallel waist needs an inverse-kinematics API which is not
exposed by the supplied examples, so waist commands are deliberately ignored.

python robot_live_client_g2.py \
    --host 111.0.22.33 \
    --port 30001 \
    --prompt "机器人右臂先从网口扩展坞物料区抓取网口扩展坞，左臂随后从网线物料区依次抓取两根网线并逐一插入" \
    --sdk-arm-order left_right \
    --observation-fps 5 \
    --observation-history 1 \
    --image-transport jpeg \
    --arm-execution-mode direct-48 \
    --direct-control-hz 15 \
    --arm-delta-limit 0.04 \
    --arm-velocity-limit 0.35 \
    --arm-acceleration-limit 0.80 \
    --arm-close-timeout 0.15 \
    --apply-actions


"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

try:
    import agibot_gdk
except Exception as exc:  # pragma: no cover - only available in G2 runtime
    raise RuntimeError(
        "Failed to import agibot_gdk. Source ~/.cache/agibot/app/env.sh and "
        "use the Python version shipped with the G2 GDK."
    ) from exc

import robot_live_client as live

_shared_execute_arm_trajectory = live._execute_arm_trajectory
_shared_execute_arm_direct48 = live._execute_arm_direct48
_shared_maybe_smooth_actions = live._maybe_smooth_actions
_SharedWebsocketClientPolicy = live.WebsocketClientPolicy


ARM_JOINT_NAMES = [
    *(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)),
    *(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)),
]
HEAD_JOINT_NAMES = ["idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3"]
BODY_JOINT_NAMES = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
G2_ACTION_DIM = 16


def _decode_g2_relative_action(
    actions: np.ndarray,
    current_arm: np.ndarray,
    current_gripper: np.ndarray,
) -> np.ndarray:
    """
    Convert DreamZero relative action into absolute G2 joint targets.

    Training:
        action = target_joint - current_joint

    Deployment:
        target_joint = predicted_delta + current_joint

    G2 action layout:
        [left_arm7,
         left_gripper,
         right_arm7,
         right_gripper]
    """

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != G2_ACTION_DIM:
        raise ValueError(
            f"Expected G2 relative action shape (T, {G2_ACTION_DIM}), "
            f"got {actions.shape}"
        )

    current_arm = np.asarray(current_arm, dtype=np.float32).reshape(14)
    current_gripper = np.asarray(current_gripper, dtype=np.float32).reshape(2)
    decoded = actions.copy()

    decoded[:, 0:7] += current_arm[0:7]
    decoded[:, 7] += current_gripper[0]
    decoded[:, 8:15] += current_arm[7:14]
    decoded[:, 15] += current_gripper[1]
    return decoded


G2_GRIPPER_OPEN_POSITION = -0.785
G2_GRIPPER_CLOSED_POSITION = 0.0
G2_ARM_GRIPPER_MIN_INTERVAL_S = 0.050
# Source: ~/.cache/agibot/app/gdk/config/mc_impl_config.json.
# Keep targets just inside the GDK boundary to avoid float32 round-off at the
# exact limit. Values are ordered like ARM_JOINT_NAMES (left 7, then right 7).
G2_ARM_JOINT_LIMIT_EPSILON = 1e-3
G2_ARM_JOINT_MIN = np.asarray(
    [
        -3.071796,
        -2.059505,
        -3.071796,
        -2.495838,
        -3.071796,
        -1.012308,
        -1.535907,
        -3.071796,
        -2.059505,
        -3.071796,
        -2.495838,
        -3.071796,
        -1.012308,
        -1.535907,
    ],
    dtype=np.float32,
)
G2_ARM_JOINT_MAX = np.asarray(
    [
        3.071796,
        2.059505,
        3.071796,
        1.012308,
        3.071796,
        1.012308,
        1.535907,
        3.071796,
        2.059505,
        3.071796,
        1.012308,
        3.071796,
        1.012308,
        1.535907,
    ],
    dtype=np.float32,
)


class G2WebsocketClientPolicy(_SharedWebsocketClientPolicy):
    """WebSocket policy client used by the G2 live client."""


def _position_by_name(robot: Any) -> tuple[dict[str, float], int]:
    response = robot.get_joint_states()
    states = response.get("states", [])
    positions = {
        str(state["name"]): float(state.get("motor_position", state.get("position", 0.0)))
        for state in states
    }
    timestamp = int(response.get("timestamp", time.time_ns()))
    return positions, timestamp


class G2Camera:
    """Present the G2 Camera API as the three named streams used by DreamZero."""

    _TYPES: dict[str, Any] = {
        "head": agibot_gdk.CameraType.kHeadColor,
        "hand_left": agibot_gdk.CameraType.kHandLeftColor,
        "hand_right": agibot_gdk.CameraType.kHandRightColor,
    }

    def __init__(self, names: list[str]) -> None:
        unknown = set(names) - self._TYPES.keys()
        if unknown:
            raise ValueError(f"Unsupported G2 camera names: {sorted(unknown)}")
        self._camera = agibot_gdk.Camera()

    @staticmethod
    def _decode(image: Any) -> np.ndarray | None:
        if image is None or not hasattr(image, "data"):
            return None
        data = image.data
        if data is None or np.asarray(data).size == 0:
            return None
        if image.encoding in (agibot_gdk.Encoding.JPEG, agibot_gdk.Encoding.PNG):
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image.encoding != agibot_gdk.Encoding.UNCOMPRESSED:
            raise RuntimeError(f"Unsupported G2 image encoding: {image.encoding}")

        raw = np.frombuffer(data, dtype=np.uint8)
        if image.color_format == agibot_gdk.ColorFormat.GRAY8:
            gray = raw.reshape((image.height, image.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frame = raw.reshape((image.height, image.width, 3))
        if image.color_format == agibot_gdk.ColorFormat.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif image.color_format != agibot_gdk.ColorFormat.BGR:
            raise RuntimeError(f"Unsupported G2 color format: {image.color_format}")
        return frame

    def get_latest_image(self, name: str) -> tuple[np.ndarray | None, int]:
        image = self._camera.get_latest_image(self._TYPES[name], 1000.0)
        if image is None:
            return None, 0
        timestamp = int(getattr(image, "timestamp_ns", getattr(image, "timestamp", 0)))
        return self._decode(image), timestamp

    def close(self) -> None:
        self._camera.close_camera()


class G2Robot:
    """Compatibility adapter around ``agibot_gdk.Robot`` and ``Pnc``."""

    def __init__(self) -> None:
        self._robot = agibot_gdk.Robot()
        self._pnc = None
        self._head_roll = 0.0
        self._waist_warning_emitted = False
        self._last_arm_gripper_command_at: float | None = None
        self._last_arm_gripper_command_kind: str | None = None
        # Arm and gripper commands share one G2 control channel. Serialize both
        # paths so commands cannot overlap across client worker threads.
        self._arm_gripper_lock = threading.Lock()

    def _wait_arm_gripper_switch(self, command_kind: str) -> None:
        if (
            self._last_arm_gripper_command_at is not None
            and self._last_arm_gripper_command_kind != command_kind
        ):
            remaining = (
                G2_ARM_GRIPPER_MIN_INTERVAL_S
                - (time.monotonic() - self._last_arm_gripper_command_at)
            )
            if remaining > 0.0:
                time.sleep(remaining)

    def _mark_arm_gripper_command(self, command_kind: str) -> None:
        self._last_arm_gripper_command_at = time.monotonic()
        self._last_arm_gripper_command_kind = command_kind

    def arm_joint_states(self) -> tuple[list[float], int]:
        positions, timestamp = _position_by_name(self._robot)
        missing = [name for name in ARM_JOINT_NAMES if name not in positions]
        if missing:
            raise RuntimeError(f"G2 joint-state response is missing arm joints: {missing}")
        return [positions[name] for name in ARM_JOINT_NAMES], timestamp

    def head_joint_states(self) -> tuple[list[float], int]:
        positions, timestamp = _position_by_name(self._robot)
        missing = [name for name in HEAD_JOINT_NAMES if name not in positions]
        if missing:
            raise RuntimeError(f"G2 joint-state response is missing head joints: {missing}")
        self._head_roll = positions[HEAD_JOINT_NAMES[1]]
        return [positions[HEAD_JOINT_NAMES[0]], positions[HEAD_JOINT_NAMES[2]]], timestamp

    def waist_joint_states(self) -> tuple[list[float], int]:
        # Raw body motor angles are not equivalent to policy [pitch, lift].
        _, timestamp = _position_by_name(self._robot)
        return [0.0, 0.0], timestamp

    def gripper_states(self) -> tuple[list[float], int]:
        response = self._robot.get_end_state()
        values: list[float] = []
        for side in ("left", "right"):
            end = response.get(f"{side}_end_state", {})
            states = end.get("end_states", [])
            position = float(states[0].get("position", 0.0)) if states else 0.0
            # G2 training data, policy output, and the omnipicker SDK all use
            # the same physical range: -0.785 is open and 0 is closed.
            values.append(
                float(
                    np.clip(
                        position,
                        G2_GRIPPER_OPEN_POSITION,
                        G2_GRIPPER_CLOSED_POSITION,
                    )
                )
            )
        return values, int(time.time_ns())

    def _joint_request(self, names: list[str], positions: list[float], speed: float = 0.3) -> None:
        request = agibot_gdk.JointControlReq()
        request.life_time = 1.0
        request.joint_names = names
        logging.info(
            "FINAL SDK ARM CMD=%s",
            positions
        )
        request.joint_positions = [float(value) for value in positions]
        request.joint_velocities = [float(speed)] * len(names)
        result = self._robot.joint_control_request(request)
        if result not in (None, 0):
            raise RuntimeError(f"G2 joint_control_request failed with result {result}")

    def move_arm(self, positions: list[float]) -> None:
        if len(positions) != 14:
            raise ValueError(f"G2 arm command must contain 14 positions, got {len(positions)}")

        # The shared executor applies interpolation and delta limits after the
        # policy action was clipped. Clamp the final values again immediately
        # before the SDK call so a boundary joint cannot drift out of range.
        values = np.asarray(positions, dtype=np.float64)
        safe_min = (
            G2_ARM_JOINT_MIN.astype(np.float64)
            + G2_ARM_JOINT_LIMIT_EPSILON
        )
        safe_max = (
            G2_ARM_JOINT_MAX.astype(np.float64)
            - G2_ARM_JOINT_LIMIT_EPSILON
        )
        clipped = np.clip(values, safe_min, safe_max)
        clipped_mask = clipped != values
        if np.any(clipped_mask):
            affected = [
                ARM_JOINT_NAMES[index]
                for index in np.flatnonzero(clipped_mask)
            ]
            logging.warning(
                "Clipped final G2 SDK arm command inside joint limits; affected joints=%s",
                affected,
            )

        with self._arm_gripper_lock:
            self._wait_arm_gripper_switch("arm")
            try:
                self._joint_request(ARM_JOINT_NAMES, clipped.tolist())
            finally:
                self._mark_arm_gripper_command("arm")

    def move_head(self, positions: list[float]) -> None:
        if len(positions) != 2:
            raise ValueError(f"G2 policy head command must contain 2 positions, got {len(positions)}")
        # G2 order is yaw, roll, pitch. Preserve the unmodelled roll axis.
        self._robot.move_head_joint(
            [float(positions[0]), self._head_roll, float(positions[1])],
            [0.3, 0.3, 0.3],
        )

    def move_waist(self, positions: list[float]) -> None:
        if not self._waist_warning_emitted:
            logging.warning(
                "Ignoring waist action: G2 uses five coupled body motors and the supplied SDK "
                "examples do not expose a safe pitch/lift command API."
            )
            self._waist_warning_emitted = True

    def move_gripper(self, positions: list[float]) -> None:
        if len(positions) != 2:
            raise ValueError(f"G2 gripper command must contain 2 positions, got {len(positions)}")
        command = agibot_gdk.JointStates()
        command.group = "dual_tool"
        command.target_type = "omnipicker"
        states: list[Any] = []
        for value in positions:
            state = agibot_gdk.JointState()
            state.position = float(
                np.clip(
                    value,
                    G2_GRIPPER_OPEN_POSITION,
                    G2_GRIPPER_CLOSED_POSITION,
                )
            )
            states.append(state)
        command.states = states
        command.nums = len(states)
        with self._arm_gripper_lock:
            self._wait_arm_gripper_switch("gripper")
            try:
                result = self._robot.move_ee_pos(command)
            finally:
                self._mark_arm_gripper_command("gripper")
        if result != 0:
            raise RuntimeError(f"G2 dual gripper move_ee_pos failed with result {result}")

    def move_wheel(self, linear: float, angular: float) -> None:
        if self._pnc is None:
            self._pnc = agibot_gdk.Pnc()
            self._pnc.request_chassis_control(0)
            time.sleep(0.5)
        twist = agibot_gdk.Twist()
        twist.linear = agibot_gdk.Vector3()
        twist.angular = agibot_gdk.Vector3()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self._pnc.move_chassis(twist)

    def shutdown(self) -> None:
        # G2 objects have no shutdown method in the supplied SDK examples.
        if self._pnc is not None:
            self.move_wheel(0.0, 0.0)


def _build_g2_obs(
    head_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    arm_pos: list[float],
    head_pos: list[float],
    waist_pos: list[float],
    gripper_pos: list[float],
    prompt: str,
    session_id: str,
    sdk_arm_order: live.ArmOrder,
    obs_flip_config: live.ObsFlipConfig,
    image_transport: live.ImageTransportMode,
    image_jpeg_quality: int,
) -> dict[str, object]:
    """Build the observation keys declared by modality_config_g2."""
    del head_pos, waist_pos
    policy_arm = live._sdk_to_policy_arm(arm_pos, sdk_arm_order)
    policy_arm = live._apply_obs_joint_sign_flips(policy_arm, obs_flip_config)
    gripper = np.asarray(gripper_pos, dtype=np.float32)
    if gripper.shape != (2,):
        raise ValueError(f"G2 gripper state must contain 2 values, got {gripper.shape}")
    return {
        "observation/top_head": live._encode_video_observation(
            head_img, image_transport=image_transport, jpeg_quality=image_jpeg_quality
        ),
        "observation/hand_left": live._encode_video_observation(
            left_img, image_transport=image_transport, jpeg_quality=image_jpeg_quality
        ),
        "observation/hand_right": live._encode_video_observation(
            right_img, image_transport=image_transport, jpeg_quality=image_jpeg_quality
        ),
        "observation/left_joint_position": policy_arm[:7],
        "observation/left_gripper_position": gripper[:1],
        "observation/right_joint_position": policy_arm[7:],
        "observation/right_gripper_position": gripper[1:],
        "prompt": prompt,
        "session_id": session_id,
    }


def _parse_g2_action_row(row: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Parse [left arm 7, left grip 1, right arm 7, right grip 1]."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape[0] == 22:
        # Internal representation used only by the shared 22-D executor.
        return {
            "left_arm": row[0:7],
            "right_arm": row[7:14],
            "gripper": row[14:16],
            "head": row[16:18],
            "waist": row[18:20],
            "wheel": row[20:22],
            "horizon": np.asarray([horizon], dtype=np.int32),
        }
    if row.shape[0] != G2_ACTION_DIM:
        raise ValueError(f"Expected G2 action dimension 16, got {row.shape[0]}")
    return {
        "left_arm": row[0:7],
        "right_arm": row[8:15],
        "gripper": np.asarray([row[7], row[15]], dtype=np.float32),
        # NaN marks modalities which do not exist in the G2 policy output, so
        # the shared executor skips their command paths.
        "head": np.full(2, np.nan, dtype=np.float32),
        "waist": np.full(2, np.nan, dtype=np.float32),
        "wheel": np.full(2, np.nan, dtype=np.float32),
        "horizon": np.asarray([horizon], dtype=np.int32),
    }


def _parse_g2_action_first(actions: np.ndarray) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != G2_ACTION_DIM:
        raise ValueError(f"Expected G2 action shape (T, 16), got {actions.shape}")
    return _parse_g2_action_row(actions[0], actions.shape[0])


def _select_g2_gripper_command(
    actions: np.ndarray, gripper_config: live.GripperConfig
) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != G2_ACTION_DIM:
        raise ValueError(f"Expected G2 action shape (T, 16), got {actions.shape}")
    first = actions[0, [7, 15]].astype(np.float32)
    last = actions[-1, [7, 15]].astype(np.float32)
    command = live._gripper_policy_to_command(last, gripper_config)
    return {"first_policy": first, "last_policy": last, "command_policy": command}


def _g2_gripper_policy_to_command(
    policy_values: np.ndarray, gripper_config: live.GripperConfig
) -> np.ndarray:
    """Pass G2 omnipicker positions through in the native SDK scale.

    G2 inference returns physical positions in [-0.785, 0], so thresholding
    them into the shared client's normalized open/closed values would reverse
    or discard the command. ``gripper_config`` is intentionally unused on G2.
    """
    del gripper_config
    pair = np.asarray(policy_values, dtype=np.float32).reshape(-1)
    if pair.shape[0] != 2:
        raise ValueError(f"Expected G2 gripper pair with 2 dims, got {pair.shape[0]}")
    return np.clip(
        pair,
        G2_GRIPPER_OPEN_POSITION,
        G2_GRIPPER_CLOSED_POSITION,
    ).astype(np.float32)


def _g2_gripper_state_for_log(values: list | np.ndarray) -> np.ndarray:
    """Keep G2 diagnostic state in the native range used on the wire."""
    pair = np.asarray(values, dtype=np.float32).reshape(-1)
    if pair.shape[0] != 2:
        raise ValueError(f"Expected G2 gripper state with 2 dims, got {pair.shape[0]}")
    return np.clip(
        pair,
        G2_GRIPPER_OPEN_POSITION,
        G2_GRIPPER_CLOSED_POSITION,
    ).astype(np.float32)


def _clip_g2_arm_joint_limits(actions: np.ndarray) -> np.ndarray:
    """Clamp G2 arm targets to the absolute limits enforced by the GDK."""
    arm_targets = np.concatenate((actions[:, 0:7], actions[:, 8:15]), axis=1)
    safe_min = G2_ARM_JOINT_MIN + G2_ARM_JOINT_LIMIT_EPSILON
    safe_max = G2_ARM_JOINT_MAX - G2_ARM_JOINT_LIMIT_EPSILON
    clipped_targets = np.clip(arm_targets, safe_min, safe_max).astype(np.float32)
    clipped_mask = clipped_targets != arm_targets
    if np.any(clipped_mask):
        affected = [
            ARM_JOINT_NAMES[index]
            for index in range(len(ARM_JOINT_NAMES))
            if np.any(clipped_mask[:, index])
        ]
        logging.warning(
            "Clipped %s G2 arm target value(s) to GDK absolute joint limits; affected joints=%s",
            int(np.count_nonzero(clipped_mask)),
            affected,
        )
    return clipped_targets


def _g2_to_shared_actions(actions: np.ndarray) -> np.ndarray:
    """Reorder external G2 actions into the shared executor's 22-D layout."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != G2_ACTION_DIM:
        raise ValueError(f"Expected G2 action shape (T, 16), got {actions.shape}")
    clipped_arm_targets = _clip_g2_arm_joint_limits(actions)
    shared = np.full((actions.shape[0], 22), np.nan, dtype=np.float32)
    shared[:, 0:14] = clipped_arm_targets
    shared[:, 14] = actions[:, 7]
    shared[:, 15] = actions[:, 15]
    return shared


def _shared_to_g2_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    g2 = np.empty((actions.shape[0], G2_ACTION_DIM), dtype=np.float32)
    g2[:, 0:7] = actions[:, 0:7]
    g2[:, 7] = actions[:, 14]
    g2[:, 8:15] = actions[:, 7:14]
    g2[:, 15] = actions[:, 15]
    return g2


def _decode_g2_actions_for_execution(
    *,
    robot: G2Robot,
    actions: np.ndarray,
    sdk_arm_order: live.ArmOrder,
) -> np.ndarray:
    """Decode checkpoint-relative actions against the latest robot state."""
    current_arm_sdk, _ = robot.arm_joint_states()
    current_gripper, _ = robot.gripper_states()
    current_arm_policy = live._sdk_to_policy_arm(
        np.asarray(current_arm_sdk, dtype=np.float32),
        sdk_arm_order,
    )
    decoded = _decode_g2_relative_action(
        actions,
        current_arm=current_arm_policy,
        current_gripper=np.asarray(current_gripper, dtype=np.float32),
    )
    logging.info(
        "Decoded G2 relative action against execution-boundary state | "
        "delta_first_left[:3]=%s delta_first_right[:3]=%s "
        "target_first_left[:3]=%s target_first_right[:3]=%s",
        np.round(np.asarray(actions, dtype=np.float32).reshape(-1, G2_ACTION_DIM)[0, 0:3], 4).tolist(),
        np.round(np.asarray(actions, dtype=np.float32).reshape(-1, G2_ACTION_DIM)[0, 8:11], 4).tolist(),
        np.round(decoded[0, 0:3], 4).tolist(),
        np.round(decoded[0, 8:11], 4).tolist(),
    )
    return decoded


def _execute_g2_trajectory(
    *,
    robot: G2Robot,
    actions: np.ndarray,
    sdk_arm_order: live.ArmOrder,
    **kwargs: Any,
) -> dict[str, Any]:
    decoded = _decode_g2_actions_for_execution(
        robot=robot,
        actions=actions,
        sdk_arm_order=sdk_arm_order,
    )
    return _shared_execute_arm_trajectory(
        robot=robot,
        actions=_g2_to_shared_actions(decoded),
        sdk_arm_order=sdk_arm_order,
        **kwargs,
    )


def _execute_g2_direct48(
    *,
    robot: G2Robot,
    actions: np.ndarray,
    sdk_arm_order: live.ArmOrder,
    **kwargs: Any,
) -> dict[str, Any]:
    decoded = _decode_g2_actions_for_execution(
        robot=robot,
        actions=actions,
        sdk_arm_order=sdk_arm_order,
    )
    return _shared_execute_arm_direct48(
        robot=robot,
        actions=_g2_to_shared_actions(decoded),
        sdk_arm_order=sdk_arm_order,
        **kwargs,
    )


def _smooth_g2_actions(
    actions: np.ndarray, config: live.ActionSmoothingConfig
) -> tuple[np.ndarray, float]:
    shared, duration = _shared_maybe_smooth_actions(_g2_to_shared_actions(actions), config)
    return _shared_to_g2_actions(shared), duration


def main() -> None:
    result = agibot_gdk.gdk_init()
    success = getattr(getattr(agibot_gdk, "GDKRes", object), "kSuccess", None)
    if success is not None and result != success:
        raise RuntimeError(f"agibot_gdk.gdk_init() failed: {result}")

    live.Camera = G2Camera
    live.Robot = G2Robot
    live.WebsocketClientPolicy = G2WebsocketClientPolicy
    live._build_obs = _build_g2_obs
    live._parse_action_row = _parse_g2_action_row
    live._parse_action_first = _parse_g2_action_first
    live._select_gripper_command = _select_g2_gripper_command
    live._gripper_policy_to_command = _g2_gripper_policy_to_command
    live._sdk_gripper_to_policy_obs = _g2_gripper_state_for_log
    live._execute_arm_trajectory = _execute_g2_trajectory
    live._execute_arm_direct48 = _execute_g2_direct48
    live._maybe_smooth_actions = _smooth_g2_actions

    # modality_config_g2.eval_delta_indices is [0], unlike the four-frame G1
    # evaluation history. Respect an explicit CLI override when one is given.
    if "--observation-history" not in sys.argv:
        sys.argv.extend(["--observation-history", "1"])
    live.main()


if __name__ == "__main__":
    main()
