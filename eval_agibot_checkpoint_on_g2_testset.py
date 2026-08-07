import dataclasses
import json
import logging
import socket
import asyncio
import os
import http
import logging
import time
import traceback
from pathlib import Path
import glob
import uuid

import pyarrow.parquet as pq
from omegaconf import OmegaConf
import torch
import tyro
from einops import rearrange
import datetime
import cv2

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig

logger = logging.getLogger(__name__)

SIGNAL_INFER = 0
SIGNAL_SHUTDOWN = 1
SIGNAL_IDLE = 2
SIGNAL_RESET_CACHE = 3


def _reset_policy_inference_cache(policy: object, reason: str) -> None:
    trained_model = getattr(policy, "trained_model", None)
    action_head = getattr(trained_model, "action_head", None)
    reset_fn = getattr(action_head, "reset_inference_cache", None)
    if callable(reset_fn):
        reset_fn()
        logger.info("Reset action-head inference cache on rank %s (%s)", dist.get_rank() if dist.is_initialized() else "?", reason)
    else:
        logger.warning("Policy action head does not expose reset_inference_cache(); cache reset skipped (%s)", reason)

@dataclasses.dataclass
class Args:
    # Model paths.
    model_path: str = "/data/wangk/checkpoints/dreamzero_agibot_fruit_lora_20k/checkpoint-12000"
    wan_ckpt_dir: str = "/data/wangk/checkpoints/Wan2.1-I2V-14B-480P"
    tokenizer_path: str = "/data/wangk/checkpoints/umt5-xxl"

    # G2 held-out split used only as a fixed three-camera visual source.
    test_data_root: str = "/data/training_data/teleop/g2/g2_tasks_g1_g7_joint_gear_subtask_v2/test"

    # Output and sample selection.
    output_dir: str = "/data/wangk/dreamzero/video_compare_agibot_vs_g1ft_vs_g2ft/g1_finetuned"
    episode_indices: str = "0"  # Comma/range syntax: 0,3,7-9
    frame_index: int = 30       # Row/frame within each episode; -1 chooses the midpoint.
    future_frames: int = 33
    prompt_override: str | None = None
    session_id_override: str | None = None

    # Inference.
    embodiment_tag: str = "agibot"
    num_inference_timesteps: int = 4
    enable_dit_cache: bool = False
    num_dit_steps: int | None = None
    timeout_seconds: int = 50000
    seed: int = 42
    preflight_only: bool = False

    # Kept for compatibility with shared model/path helpers.
    port: int = 0
    index: int = 0
    max_chunk_size: int | None = None
    video_save_mode: str = "none"



class DistributedRoboarenaPolicyBase:
    """Shared distributed inference plumbing for websocket policy wrappers."""

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        video_save_mode: str = "first",
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir
        self._video_save_mode = video_save_mode
        self._frame_buffers = self._init_frame_buffers()
        self._current_session_id: str | None = None
        self.video_across_time = []
        self._msg_index = 0

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _init_frame_buffers(self) -> dict[str, list[np.ndarray]]:
        return {
            "video.top_head": [],
            "video.hand_left": [],
            "video.hand_right": [],
        }

    def _reset_custom_state(self) -> None:
        pass

    def _after_infer(self) -> None:
        pass

    def _prepare_video_chunk(self, video_pred: torch.Tensor) -> torch.Tensor | None:
        if self._video_save_mode == "none":
            return None
        return video_pred

    def _video_save_fps(self) -> int:
        return 5

    def _convert_observation(self, obs: dict) -> dict:
        raise NotImplementedError

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        raise NotImplementedError

    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        import pickle

        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).clone().cuda()
        dist.broadcast(data_tensor, src=0)

    def _extract_action_dict(self, action_chunk_dict: object) -> dict[str, object]:
        action_dict: dict[str, object] = {}
        for key in dir(action_chunk_dict):
            if key.startswith('action.'):
                action_dict[key] = getattr(action_chunk_dict, key)
        return action_dict

    def _broadcast_signal_to_workers(self, signal: int) -> None:
        signal_tensor = torch.tensor([signal], dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

    def _diagnostic_dir(self) -> str | None:
        if not self._output_dir:
            return None
        path = os.path.join(self._output_dir, "diagnostics")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_predicted_video_chunk(self, video_pred: torch.Tensor) -> None:
        """Decode and save exactly one predicted latent chunk.

        This deliberately avoids concatenating independent 33-frame chunks
        before VAE decoding, so chunk-boundary artifacts cannot hide the
        quality of the current world-model prediction.
        """
        diagnostic_dir = self._diagnostic_dir()
        if diagnostic_dir is None:
            return

        try:
            if video_pred.ndim != 5:
                raise ValueError(
                    "video_pred must be (B,C,T,H,W), got "
                    f"{tuple(video_pred.shape)}"
                )
            finite = bool(torch.isfinite(video_pred).all().item())
            latent_float = video_pred.detach().float()
            stats = {
                "request_index": int(self._msg_index),
                "shape": list(video_pred.shape),
                "dtype": str(video_pred.dtype),
                "finite": finite,
                "min": float(latent_float.min().item()),
                "max": float(latent_float.max().item()),
                "mean": float(latent_float.mean().item()),
                "std": float(latent_float.std().item()),
            }
            with open(
                os.path.join(diagnostic_dir, "video_pred_stats.jsonl"),
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(json.dumps(stats, ensure_ascii=False) + "\n")

            if not finite:
                raise ValueError(f"video_pred contains NaN/Inf: {stats}")

            action_head = self._policy.trained_model.action_head
            device = getattr(action_head, "_device", None)
            if device is None:
                device = next(self._policy.trained_model.parameters()).device

            latent = video_pred.detach().to(
                device=device,
                dtype=torch.bfloat16,
            )
            frames = action_head.vae.decode(
                latent,
                tiled=action_head.tiled,
                tile_size=(
                    action_head.tile_size_height,
                    action_head.tile_size_width,
                ),
                tile_stride=(
                    action_head.tile_stride_height,
                    action_head.tile_stride_width,
                ),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = (
                (frames.float() + 1.0) * 127.5
            ).clamp(0, 255).cpu().numpy().astype(np.uint8)

            output_path = os.path.join(
                diagnostic_dir,
                f"pred_chunk_request_{self._msg_index:06d}_f{len(frames)}.mp4",
            )
            imageio.mimsave(
                output_path,
                list(frames),
                fps=self._video_save_fps(),
                codec="libx264",
                macro_block_size=None,
            )
            imageio.imwrite(
                os.path.join(
                    diagnostic_dir,
                    f"pred_chunk_request_{self._msg_index:06d}_first.png",
                ),
                frames[0],
            )
            imageio.imwrite(
                os.path.join(
                    diagnostic_dir,
                    f"pred_chunk_request_{self._msg_index:06d}_last.png",
                ),
                frames[-1],
            )

            # DreamZero packs three 320x176 views into a 2x2 RGB canvas in
            # ConcatTransform order: top_head, hand_left, hand_right, padding.
            # Decode the full canvas once, then split decoded RGB only for
            # diagnostics. Never split the latent before VAE decoding.
            frame_h, frame_w = int(frames.shape[1]), int(frames.shape[2])
            if frame_h % 2 == 0 and frame_w % 2 == 0:
                half_h, half_w = frame_h // 2, frame_w // 2
                view_frames = {
                    "top_head": frames[:, :half_h, :half_w],
                    "hand_left": frames[:, :half_h, half_w:],
                    "hand_right": frames[:, half_h:, :half_w],
                    "padding": frames[:, half_h:, half_w:],
                }
                view_stats = {}
                for view_name, view_video in view_frames.items():
                    view_path = os.path.join(
                        diagnostic_dir,
                        f"pred_chunk_request_{self._msg_index:06d}_{view_name}_f{len(view_video)}.mp4",
                    )
                    imageio.mimsave(
                        view_path,
                        list(view_video),
                        fps=self._video_save_fps(),
                        codec="libx264",
                        macro_block_size=None,
                    )
                    imageio.imwrite(
                        os.path.join(
                            diagnostic_dir,
                            f"pred_chunk_request_{self._msg_index:06d}_{view_name}_first.png",
                        ),
                        view_video[0],
                    )
                    imageio.imwrite(
                        os.path.join(
                            diagnostic_dir,
                            f"pred_chunk_request_{self._msg_index:06d}_{view_name}_last.png",
                        ),
                        view_video[-1],
                    )
                    view_float = view_video.astype(np.float32)
                    view_stats[view_name] = {
                        "shape": list(view_video.shape),
                        "mean_rgb": [
                            float(x)
                            for x in view_float.mean(axis=(0, 1, 2)).tolist()
                        ],
                        "std": float(view_float.std()),
                    }
                stats["decoded_shape"] = list(frames.shape)
                stats["decoded_views"] = view_stats
                with open(
                    os.path.join(diagnostic_dir, "decoded_view_stats.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as stream:
                    stream.write(json.dumps(stats, ensure_ascii=False) + "\n")
            else:
                logger.warning(
                    "Decoded video is not divisible into a 2x2 view grid: %s",
                    tuple(frames.shape),
                )

            logger.info(
                "Saved single predicted video chunk to %s | stats=%s",
                output_path,
                stats,
            )
        except Exception as exc:
            logger.exception(
                "Failed to save single predicted video chunk: %s", exc
            )

    def infer(self, obs: dict) -> np.ndarray:
        session_id = obs.get('session_id')
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed from '%s' to '%s', resetting state", self._current_session_id, session_id)
                self._broadcast_signal_to_workers(SIGNAL_RESET_CACHE)
                self._reset_state()
            else:
                logger.info("New session started: '%s'", session_id)
            self._current_session_id = session_id

        self._msg_index += 1
        converted_obs = self._convert_observation(obs)

        self._broadcast_signal_to_workers(SIGNAL_INFER)
        self._broadcast_batch_to_workers(converted_obs)

        batch = Batch(obs=converted_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        self._save_predicted_video_chunk(video_pred)

        video_chunk = self._prepare_video_chunk(video_pred)
        if video_chunk is not None:
            self.video_across_time.append(video_chunk.detach().cpu())
        action = self._convert_action(self._extract_action_dict(result_batch.act))
        self._after_infer()
        return action

    def _reset_state(self, save_video: bool = True) -> None:
        if save_video and len(self.video_across_time) > 0 and self._output_dir:
            try:
                frame_list = []
                action_head = self._policy.trained_model.action_head
                device = getattr(action_head, "_device", None)
                if device is None:
                    device = next(self._policy.trained_model.parameters()).device
                video_across_time_cat = torch.cat(self.video_across_time, dim=2).to(device=device, dtype=torch.bfloat16)
                frames = action_head.vae.decode(
                    video_across_time_cat,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
                frames = rearrange(frames, 'B C T H W -> B T H W C')
                frames = frames[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                for frame in frames:
                    frame_list.append(frame)

                if frame_list:
                    sample_frame = frame_list[0]
                    if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                        save_dir = self._output_dir
                        os.makedirs(save_dir, exist_ok=True)
                        all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        num_frames = len(frame_list)
                        output_path = os.path.join(save_dir, f'{timestamp}_{len(all_mp4_files):06}_f{num_frames}.mp4')
                        imageio.mimsave(output_path, frame_list, fps=self._video_save_fps(), codec='libx264')
                        logger.info('Saved video on reset to: %s', output_path)
            except Exception as exc:
                logger.warning('Failed to save video on reset: %s', exc)

        for key in self._frame_buffers:
            self._frame_buffers[key] = []

        self.video_across_time = []
        _reset_policy_inference_cache(self._policy, "wrapper reset_state")
        self._reset_custom_state()

    def reset(self, reset_info: dict) -> None:
        self._broadcast_signal_to_workers(SIGNAL_RESET_CACHE)
        self._reset_state(save_video=True)


class ARDroidRoboarenaPolicy(DistributedRoboarenaPolicyBase):
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid."""

    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        video_save_mode: str = "first",
    ) -> None:
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
        self._reset_custom_state()

    def _init_frame_buffers(self) -> dict[str, list[np.ndarray]]:
        return {
            'video.exterior_image_1_left': [],
            'video.exterior_image_2_left': [],
            'video.wrist_image_left': [],
        }

    def _reset_custom_state(self) -> None:
        self._is_first_call = True

    def _after_infer(self) -> None:
        self._is_first_call = False

    def _convert_observation(self, obs: dict) -> dict:
        converted = {}
        image_key_mapping = {
            'observation/exterior_image_0_left': 'video.exterior_image_1_left',
            'observation/exterior_image_1_left': 'video.exterior_image_2_left',
            'observation/wrist_image_left': 'video.wrist_image_left',
        }

        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        self._frame_buffers[droid_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                converted[droid_key] = np.stack(frames_to_use, axis=0)

        joint_pos = obs.get('observation/joint_position', np.zeros(7, dtype=np.float32))
        if joint_pos.ndim == 1:
            joint_pos = joint_pos.reshape(1, -1)
        converted['state.joint_position'] = joint_pos.astype(np.float64)

        gripper_pos = obs.get('observation/gripper_position', np.zeros(1, dtype=np.float32))
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos.reshape(1, -1)
        converted['state.gripper_position'] = gripper_pos.astype(np.float64)
        converted['annotation.language.action_text'] = obs.get('prompt', '')
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        joint_action = None
        gripper_action = None
        for key, value in action_dict.items():
            if 'joint_position' in key:
                joint_action = value
            elif 'gripper_position' in key or 'gripper' in key:
                gripper_action = value

        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)

        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)

        num_steps = joint_action.shape[0]
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((num_steps, 1), dtype=np.float32)

        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)



def _load_agibot_fixed_state_from_checkpoint(
    model_path: str,
    required_state_keys: list[str],
) -> dict[str, np.ndarray]:
    """Load nominal AgiBot state values from checkpoint metadata."""
    metadata_path = os.path.join(
        os.path.abspath(model_path),
        "experiment_cfg",
        "metadata.json",
    )
    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    try:
        stats = metadata["agibot"]["statistics"]["state"]
    except KeyError as exc:
        raise KeyError(
            f"{metadata_path} does not contain agibot.statistics.state"
        ) from exc

    fixed: dict[str, np.ndarray] = {}
    for target_key in required_state_keys:
        if not target_key.startswith("state."):
            raise ValueError(
                f"Unexpected AgiBot state modality key: {target_key!r}"
            )
        source_key = target_key.split(".", 1)[1]
        if source_key not in stats or "mean" not in stats[source_key]:
            raise KeyError(
                f"Missing checkpoint-mean state for {source_key!r} "
                f"in {metadata_path}"
            )
        fixed[target_key] = np.asarray(
            stats[source_key]["mean"],
            dtype=np.float64,
        ).reshape(1, -1)
    return fixed


class AgiBotRoboarenaPolicy(DistributedRoboarenaPolicyBase):
    """Run an AgiBot checkpoint on fixed G2 test-set camera frames.

    This is intentionally video-only. G2 proprioception is not used for
    conditioning; the checkpoint's own mean AgiBot state fills its required
    state fields. The native AgiBot action output is discarded.
    """

    VIDEO_KEY_MAPPING = {
        "observation/top_head": "video.top_head",
        "observation/hand_left": "video.hand_left",
        "observation/hand_right": "video.hand_right",
    }

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        model_path: str,
        output_dir: str | None = None,
        video_save_mode: str = "first",
    ) -> None:
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
        self._action_keys = list(
            self._policy.modality_configs.action.modality_keys
        )
        self._state_keys = list(
            self._policy.modality_configs.state.modality_keys
        )
        self._language_keys = list(
            self._policy.modality_configs.language.modality_keys
        )
        eval_indices = list(
            getattr(
                self._policy.modality_configs.video,
                "eval_delta_indices",
                [0],
            )
            or [0]
        )
        self._expected_video_frames = max(1, len(eval_indices))
        self._fixed_state = _load_agibot_fixed_state_from_checkpoint(
            model_path,
            self._state_keys,
        )
        self._safe_g2_hold_action = np.zeros(16, dtype=np.float32)
        logger.info(
            "[AGIBOT-ON-G2 VIDEO TEST] video_frames=%s "
            "state_keys=%s language_keys=%s",
            self._expected_video_frames,
            self._state_keys,
            self._language_keys,
        )

    @staticmethod
    def _normalize_video(
        value: object,
        target_key: str,
    ) -> np.ndarray:
        array_bgr = np.asarray(value)
        if array_bgr.ndim == 3:
            array_bgr = np.expand_dims(array_bgr, axis=0)
        elif array_bgr.ndim != 4:
            raise ValueError(
                f"AgiBot video input for {target_key} must have "
                f"shape (H,W,C) or (T,H,W,C), got {array_bgr.shape}"
            )
        if array_bgr.shape[-1] != 3:
            raise ValueError(
                f"AgiBot video input for {target_key} must have 3 channels, "
                f"got {array_bgr.shape}"
            )
        # OpenCV decodes the G2 test MP4 files as BGR. DreamZero expects RGB.
        return np.ascontiguousarray(array_bgr[..., ::-1])

    def _prepare_video_chunk(
        self,
        video_pred: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._video_save_mode == "none":
            return None
        if video_pred.ndim != 5:
            raise ValueError(
                "AgiBot video prediction must be 5D "
                f"(B,C,T,H,W), got {tuple(video_pred.shape)}"
            )
        if self._video_save_mode == "first":
            return video_pred[:, :, :1].contiguous()
        if self._video_save_mode == "full":
            return video_pred.contiguous()
        raise ValueError(
            f"Unsupported video_save_mode: {self._video_save_mode!r}"
        )

    def _video_save_fps(self) -> int:
        return 30

    def _save_input_grid(
        self,
        converted: dict[str, object],
    ) -> None:
        diagnostic_dir = self._diagnostic_dir()
        if diagnostic_dir is None:
            return

        camera_keys = (
            "video.top_head",
            "video.hand_left",
            "video.hand_right",
        )
        arrays = [np.asarray(converted[key]) for key in camera_keys]
        frame_count = min(array.shape[0] for array in arrays)
        grids: list[np.ndarray] = []

        # The checkpoint may require different raw resolutions per camera.
        # Resize only for this human-readable diagnostic grid; the arrays sent
        # to the model remain at their native checkpoint-required sizes.
        display_size = (320, 176)
        for index in range(frame_count):
            display_views = [
                cv2.resize(
                    np.asarray(array[index]),
                    display_size,
                    interpolation=cv2.INTER_AREA,
                )
                for array in arrays
            ]
            top_head, hand_left, hand_right = display_views
            black = np.zeros_like(top_head)
            grids.append(
                np.concatenate(
                    [
                        np.concatenate([top_head, hand_left], axis=1),
                        np.concatenate([hand_right, black], axis=1),
                    ],
                    axis=0,
                )
            )

        grid_path = os.path.join(
            diagnostic_dir,
            f"agibot_from_g2_input_request_{self._msg_index:06d}"
            f"_f{len(grids)}.mp4",
        )
        imageio.mimsave(
            grid_path,
            grids,
            fps=30,
            codec="libx264",
            macro_block_size=None,
        )
        imageio.imwrite(
            os.path.join(
                diagnostic_dir,
                f"agibot_from_g2_input_request_{self._msg_index:06d}"
                "_last.png",
            ),
            grids[-1],
        )
        logger.info(
            "Saved AgiBot model-input display grid to %s | native_shapes=%s",
            grid_path,
            {
                key: list(array.shape)
                for key, array in zip(camera_keys, arrays)
            },
        )

    def _convert_observation(
        self,
        obs: dict,
    ) -> dict:
        converted: dict[str, object] = {}
        missing_video: list[str] = []

        for source_key, target_key in self.VIDEO_KEY_MAPPING.items():
            value = obs.get(source_key, obs.get(target_key))
            if value is None:
                missing_video.append(source_key)
                continue
            frames = self._normalize_video(value, target_key)
            self._frame_buffers[target_key].extend(list(frames))
            history = list(
                self._frame_buffers[target_key][
                    -self._expected_video_frames:
                ]
            )
            while len(history) < self._expected_video_frames:
                history.insert(0, history[0])
            converted[target_key] = np.stack(history, axis=0)

        if missing_video:
            raise ValueError(
                "Missing G2 camera inputs for AgiBot video test: "
                + ", ".join(sorted(missing_video))
            )

        packed = np.asarray(
            obs.get("observation/state", np.zeros(16, dtype=np.float32)),
            dtype=np.float32,
        )
        if packed.ndim == 2:
            packed = packed[-1]
        if packed.ndim == 1 and packed.shape[0] == 16:
            self._safe_g2_hold_action = packed.copy()

        for key, value in self._fixed_state.items():
            converted[key] = value.copy()

        prompt = str(obs.get("prompt", ""))
        for language_key in self._language_keys:
            converted[language_key] = prompt

        self._save_input_grid(converted)
        logger.info(
            "[AGIBOT-ON-G2 VIDEO TEST] request=%s "
            "frames_per_view=%s language_keys=%s prompt=%r",
            self._msg_index,
            {
                key: int(np.asarray(converted[key]).shape[0])
                for key in self.VIDEO_KEY_MAPPING.values()
            },
            self._language_keys,
            prompt,
        )
        return converted

    def _convert_action(
        self,
        action_dict: dict,
    ) -> np.ndarray:
        # Native AgiBot actions are irrelevant for this diagnostic. Returning
        # a G2 hold-shaped tensor keeps the rest of the offline evaluator
        # simple and prevents accidental dimension comparisons.
        return np.repeat(
            self._safe_g2_hold_action.reshape(1, 16),
            48,
            axis=0,
        ).astype(np.float32)


class G2RoboarenaPolicy(DistributedRoboarenaPolicyBase):
    """Adapter for the G2 dual-arm joint-space policy."""
    FRAMES_PER_CHUNK = 4
    VIDEO_KEY_MAPPING = {
        'observation/top_head': 'video.top_head',
        'observation/hand_left': 'video.hand_left',
        'observation/hand_right': 'video.hand_right',
    }

    STATE_KEY_MAPPING = {
        'observation/left_joint_position': 'state.left_joint_position',
        'observation/left_gripper_position': 'state.left_gripper_position',
        'observation/right_joint_position': 'state.right_joint_position',
        'observation/right_gripper_position': 'state.right_gripper_position',
    }

    PACKED_STATE_KEYS = (
        'observation/state',
        'observation.state',
        'state',
    )

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        video_save_mode: str = "first",
    ) -> None:
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
        self._action_keys = list(
            self._policy.modality_configs.action.modality_keys
        )
        eval_indices = list(
            getattr(
                self._policy.modality_configs.video,
                "eval_delta_indices",
                [-3, -2, -1, 0],
            )
            or [0]
        )
        self._expected_video_frames = max(1, len(eval_indices))
        logger.info(
            "G2 wrapper expects %s evaluation frame(s): %s",
            self._expected_video_frames,
            eval_indices,
        )

    @staticmethod
    def _lookup_obs_value(
        obs: dict,
        source_key: str,
        target_key: str,
    ) -> object:
        if source_key in obs:
            return obs[source_key]
        return obs.get(target_key)

    @staticmethod
    def _normalize_video(
        value: object,
        target_key: str,
    ) -> np.ndarray:
        if (
            isinstance(value, dict)
            and value.get("__dreamzero_image_encoding__")
            == "jpeg_sequence"
        ):
            frames = []
            expected_shape = tuple(value.get("shape", ()))
            expected_dtype = np.dtype(
                value.get("dtype", "uint8")
            )
            for index, frame_bytes in enumerate(
                value.get("frames", [])
            ):
                encoded = np.frombuffer(
                    frame_bytes,
                    dtype=np.uint8,
                )
                frame_bgr = cv2.imdecode(
                    encoded,
                    cv2.IMREAD_COLOR,
                )
                if frame_bgr is None:
                    raise ValueError(
                        f"Failed to decode JPEG frame {index} "
                        f"for {target_key}"
                    )
                # The G2 client normalizes every camera frame to OpenCV BGR.
                # DreamZero's training video path is RGB, so convert exactly
                # once at the server/model boundary.
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
                frames.append(
                    frame_rgb.astype(
                        expected_dtype,
                        copy=False,
                    )
                )
            if not frames:
                raise ValueError(
                    f"No JPEG frames were provided for {target_key}"
                )
            array = np.stack(frames, axis=0)
            if expected_shape and tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Decoded JPEG video for {target_key} has "
                    f"shape {array.shape}, expected {expected_shape}"
                )
            return array

        array_bgr = np.asarray(value)
        if array_bgr.ndim == 3:
            array_bgr = np.expand_dims(array_bgr, axis=0)
        elif array_bgr.ndim != 4:
            raise ValueError(
                f"G2 video input for {target_key} must have "
                f"shape (H,W,C) or (T,H,W,C), got {array_bgr.shape}"
            )
        if array_bgr.shape[-1] != 3:
            raise ValueError(
                f"G2 video input for {target_key} must have 3 channels, "
                f"got {array_bgr.shape}"
            )
        # The non-JPEG G2 client path is also BGR because G2Camera._decode()
        # converts RGB camera buffers to BGR and leaves BGR buffers unchanged.
        # Convert it to the same RGB convention as the JPEG branch.
        return np.ascontiguousarray(array_bgr[..., ::-1])

    @staticmethod
    def _normalize_state(
        value: object,
        target_key: str,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim != 2:
            raise ValueError(
                f"G2 state input for {target_key} must be "
                f"1D or 2D, got {array.shape}"
            )
        return array

    @staticmethod
    def _split_packed_state(
        value: object,
    ) -> dict[str, np.ndarray]:
        packed = np.asarray(value, dtype=np.float64)
        if packed.ndim == 1:
            packed = packed.reshape(1, -1)
        elif packed.ndim != 2:
            raise ValueError(
                "Packed G2 state must have shape (16,) or "
                f"(T,16), got {packed.shape}"
            )
        if packed.shape[-1] != 16:
            raise ValueError(
                f"Packed G2 state must contain 16 values, "
                f"got {packed.shape}"
            )
        return {
            'state.left_joint_position': packed[:, 0:7],
            'state.left_gripper_position': packed[:, 7:8],
            'state.right_joint_position': packed[:, 8:15],
            'state.right_gripper_position': packed[:, 15:16],
        }

    def _prepare_video_chunk(
        self,
        video_pred: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._video_save_mode == "none":
            return None
        if video_pred.ndim != 5:
            raise ValueError(
                "G2 video prediction must be 5D "
                f"(B,C,T,H,W), got {tuple(video_pred.shape)}"
            )
        if self._video_save_mode == "first":
            return video_pred[:, :, :1].contiguous()
        if self._video_save_mode == "full":
            return video_pred.contiguous()
        raise ValueError(
            f"Unsupported video_save_mode: "
            f"{self._video_save_mode!r}"
        )

    def _video_save_fps(self) -> int:
        return 30

    def _save_input_grid(self, converted: dict[str, object]) -> None:
        """Save the exact four-frame, three-camera history sent to the model."""
        diagnostic_dir = self._diagnostic_dir()
        if diagnostic_dir is None:
            return

        camera_keys = (
            "video.top_head",
            "video.hand_left",
            "video.hand_right",
        )
        if any(key not in converted for key in camera_keys):
            return

        arrays = [np.asarray(converted[key]) for key in camera_keys]
        if any(array.ndim != 4 for array in arrays):
            raise ValueError(
                "Diagnostic input videos must be (T,H,W,C): "
                + ", ".join(
                    f"{key}={array.shape}"
                    for key, array in zip(camera_keys, arrays)
                )
            )
        frame_count = min(array.shape[0] for array in arrays)
        grids: list[np.ndarray] = []
        for index in range(frame_count):
            top_head, hand_left, hand_right = [
                np.asarray(array[index]) for array in arrays
            ]
            height, width = top_head.shape[:2]
            black = np.zeros_like(top_head)
            grid_rgb = np.concatenate(
                [
                    np.concatenate([top_head, hand_left], axis=1),
                    np.concatenate([hand_right, black], axis=1),
                ],
                axis=0,
            )
            # _normalize_video() already returns RGB for both JPEG and array
            # transport. Keep the diagnostic image byte-for-byte in model
            # color order instead of swapping it a second time.
            grids.append(np.ascontiguousarray(grid_rgb))

        output_path = os.path.join(
            diagnostic_dir,
            f"input_grid_request_{self._msg_index:06d}_f{len(grids)}.mp4",
        )
        imageio.mimsave(
            output_path,
            grids,
            fps=5,
            codec="libx264",
            macro_block_size=None,
        )
        imageio.imwrite(
            os.path.join(
                diagnostic_dir,
                f"input_grid_request_{self._msg_index:06d}_last.png",
            ),
            grids[-1],
        )
        channel_means = {
            key: [
                float(x)
                for x in np.asarray(array, dtype=np.float32).mean(
                    axis=(0, 1, 2)
                ).tolist()
            ]
            for key, array in zip(camera_keys, arrays)
        }
        logger.info(
            "Saved exact RGB model input grid to %s | shapes=%s | mean_rgb=%s",
            output_path,
            {key: list(array.shape) for key, array in zip(camera_keys, arrays)},
            channel_means,
        )

    def _convert_observation(self, obs: dict) -> dict:
        converted: dict[str, object] = {}
        missing_video: list[str] = []

        for source_key, target_key in self.VIDEO_KEY_MAPPING.items():
            value = self._lookup_obs_value(
                obs,
                source_key,
                target_key,
            )
            if value is None:
                missing_video.append(source_key)
                continue
            frames = self._normalize_video(
                value,
                target_key,
            )

            self._frame_buffers[target_key].extend(
                list(frames)
            )

            history = self._frame_buffers[target_key][
                -self._expected_video_frames:
            ]

            while len(history) < self._expected_video_frames:
                history.insert(0, history[0])

            converted[target_key] = np.stack(history, axis=0)

        if missing_video:
            raise ValueError(
                "G2 inference requires video keys: "
                + ", ".join(sorted(missing_video))
            )

        packed_state = None
        for key in self.PACKED_STATE_KEYS:
            if key in obs:
                packed_state = obs[key]
                break

        if packed_state is not None:
            converted.update(
                self._split_packed_state(packed_state)
            )
        else:
            missing_state: list[str] = []
            for source_key, target_key in self.STATE_KEY_MAPPING.items():
                value = self._lookup_obs_value(
                    obs,
                    source_key,
                    target_key,
                )
                if value is None:
                    missing_state.append(source_key)
                    continue
                converted[target_key] = self._normalize_state(
                    value,
                    target_key,
                )
            if missing_state:
                raise ValueError(
                    "G2 inference requires a packed 16-D state "
                    "under observation/state, observation.state, "
                    "or state; otherwise all split state keys are "
                    "required: "
                    + ", ".join(sorted(missing_state))
                )

        expected_dims = {
            'state.left_joint_position': 7,
            'state.left_gripper_position': 1,
            'state.right_joint_position': 7,
            'state.right_gripper_position': 1,
        }
        for key, expected_dim in expected_dims.items():
            array = np.asarray(converted[key])
            if array.shape[-1] != expected_dim:
                raise ValueError(
                    f"{key} must have last dimension "
                    f"{expected_dim}, got {array.shape}"
                )

        converted['annotation.language.action_text'] = obs.get(
            'prompt',
            obs.get(
                'annotation.language.action_text',
                '',
            ),
        )
        self._save_input_grid(converted)
        return converted

    def _convert_action(
        self,
        action_dict: dict,
    ) -> np.ndarray:
        missing = [
            key
            for key in self._action_keys
            if key not in action_dict
        ]
        if missing:
            raise RuntimeError(
                "Missing G2 action outputs: "
                + ", ".join(missing)
            )

        arrays: list[np.ndarray] = []
        horizon: int | None = None
        for key in self._action_keys:
            value = action_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim == 0:
                array = array.reshape(1, 1)
            elif array.ndim == 1:
                array = array.reshape(-1, 1)
            else:
                array = array.reshape(array.shape[0], -1)

            if horizon is None:
                horizon = array.shape[0]
            elif array.shape[0] != horizon:
                raise RuntimeError(
                    f"Inconsistent G2 action horizon for {key}: "
                    f"expected {horizon}, got {array.shape[0]}"
                )
            arrays.append(array.astype(np.float32))

        action = np.concatenate(arrays, axis=-1)
        if action.shape[-1] != 16:
            raise RuntimeError(
                "G2 action must have 16 dimensions ordered as "
                "[left_joint(7), left_gripper(1), "
                "right_joint(7), right_gripper(1)], "
                f"got {action.shape}"
            )
        return action.astype(np.float32)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []
        self._msg_index = 0
        self._signal_group = signal_group
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)

    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            await self._worker_loop()

    async def _worker_loop(self):
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == SIGNAL_SHUTDOWN:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break
                elif signal == SIGNAL_IDLE:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    continue
                elif signal == SIGNAL_RESET_CACHE:
                    logger.info(f"Rank {dist.get_rank()} received inference cache reset signal")
                    _reset_policy_inference_cache(self._policy, "worker signal")
                    continue

                batch = self._receive_batch_from_rank0()
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        import pickle

        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    def _broadcast_batch_to_workers(self, obs):
        import pickle

        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).clone().cuda()
        dist.broadcast(data_tensor, src=0)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')

        try:
            while True:
                try:
                    data = await websocket.recv()
                    obs = msgpack_numpy.unpackb(data)
                    self._msg_index += 1

                    signal_tensor.zero_()
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    dist.barrier()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()

                    action_chunk_dict = result_batch.act

                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out

                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    self.video_across_time = []
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            logger.info("Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)


def init_mesh() -> DeviceMesh:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    _ = torch.cuda.is_available()
    _ = torch.cuda.device_count()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (1, 2):
        raise ValueError(
            f"This DreamZero inference path only supports 1 or 2 GPUs, got world_size={world_size}. "
            "The action head parallelization code explicitly supports ip_size 1 or 2 only. "
            "Please launch with --nproc_per_node=2 (or 1)."
        )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to local_rank={local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _create_wrapper_policy(
    embodiment_tag: str,
    groot_policy: GrootSimPolicy,
    signal_group: dist.ProcessGroup,
    output_dir: str | None,
    video_save_mode: str,
    model_path: str | None = None,
) -> DistributedRoboarenaPolicyBase:
    if embodiment_tag == 'oxe_droid':
        return ARDroidRoboarenaPolicy(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
    if embodiment_tag == 'agibot':
        if model_path is None:
            raise ValueError('model_path is required for AgiBot wrapper')
        return AgiBotRoboarenaPolicy(
            groot_policy=groot_policy,
            signal_group=signal_group,
            model_path=model_path,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
    if embodiment_tag == 'g2':
        return G2RoboarenaPolicy(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
    raise ValueError(f'Unsupported embodiment_tag: {embodiment_tag}')


def _create_server_config(embodiment_tag: str) -> PolicyServerConfig:
    if embodiment_tag == 'oxe_droid':
        return PolicyServerConfig(
            image_resolution=(180, 320),
            needs_wrist_camera=True,
            n_external_cameras=2,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space='joint_position',
        )
    if embodiment_tag == 'agibot':
        return PolicyServerConfig(
            image_resolution=(640, 480),
            needs_wrist_camera=False,
            n_external_cameras=3,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space='agibot_flattened',
        )
    if embodiment_tag == 'g2':
        return PolicyServerConfig(
            image_resolution=(176, 320),
            needs_wrist_camera=False,
            n_external_cameras=3,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space='joint_position',
        )
    raise ValueError(f'Unsupported embodiment_tag: {embodiment_tag}')


def _build_path_overrides(args: Args) -> tuple[list[str], list[str]]:
    model_config_overrides: list[str] = []
    train_config_overrides: list[str] = []

    if args.wan_ckpt_dir:
        wan_ckpt_dir = os.path.abspath(args.wan_ckpt_dir)
        required_files = [
            os.path.join(wan_ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(wan_ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            os.path.join(wan_ckpt_dir, "Wan2.1_VAE.pth"),
        ]
        missing = [path for path in required_files if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "Missing Wan checkpoint component(s): " + ", ".join(missing)
            )
        model_config_overrides.extend(
            [
                f"action_head_cfg.config.diffusion_model_cfg.diffusion_model_pretrained_path={wan_ckpt_dir}",
                f"action_head_cfg.config.text_encoder_cfg.text_encoder_pretrained_path={wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                f"action_head_cfg.config.image_encoder_cfg.image_encoder_pretrained_path={wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                f"action_head_cfg.config.vae_cfg.vae_pretrained_path={wan_ckpt_dir}/Wan2.1_VAE.pth",
            ]
        )

    if args.tokenizer_path:
        tokenizer_path = os.path.abspath(args.tokenizer_path)
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
        if args.embodiment_tag.lower() == "agibot":
            train_config_overrides.append(
                f"transforms.agibot.transforms.10.tokenizer_path={tokenizer_path}"
            )
        elif args.embodiment_tag.lower() == "oxe_droid":
            train_config_overrides.append(
                f"transforms.oxe_droid.transforms.10.tokenizer_path={tokenizer_path}"
            )
        elif args.embodiment_tag.lower() == "g2":
            train_config_overrides.append(
                f"transforms.g2.transforms.10.tokenizer_path={tokenizer_path}"
            )

    return model_config_overrides, train_config_overrides



def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _parse_episode_indices(spec: str, total_episodes: int) -> list[int]:
    values: list[int] = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            left, right = token.split('-', 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid episode range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("--episode-indices cannot be empty")
    unique: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value < 0 or value >= total_episodes:
            raise IndexError(
                f"Episode {value} is outside [0, {total_episodes - 1}]"
            )
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _episode_file_from_template(
    root: Path,
    template: str,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    path = root / template.format(**values)
    if path.is_file():
        return path

    # Fallback for datasets whose chunk size metadata was rewritten without
    # moving the actual files.
    filename = f"episode_{episode_index:06d}" + path.suffix
    if video_key is None:
        candidates = sorted(root.glob(f"data/chunk-*/{filename}"))
    else:
        candidates = sorted(root.glob(f"videos/chunk-*/{video_key}/{filename}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not resolve episode file for episode={episode_index}, "
            f"video_key={video_key!r}; expected {path}, candidates={candidates}"
        )
    return candidates[0]


def _column_to_numpy(table: object, name: str, dtype: np.dtype) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"Parquet column missing: {name}")
    values = table[name].combine_chunks().to_pylist()
    return np.asarray(values, dtype=dtype)


def _unwrap_text(value: object) -> str:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if value is None:
        return ""
    return str(value)


def _decode_video_range_bgr(
    video_path: Path,
    first_index: int,
    last_index: int,
) -> dict[int, np.ndarray]:
    if first_index < 0 or last_index < first_index:
        raise ValueError(
            f"Invalid decode range [{first_index}, {last_index}] for {video_path}"
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(first_index))
        decoded: dict[int, np.ndarray] = {}
        for index in range(first_index, last_index + 1):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"Failed to decode frame {index} from {video_path}"
                )
            decoded[index] = np.ascontiguousarray(frame_bgr)
        return decoded
    finally:
        capture.release()


def _grid_rgb(
    top_bgr: np.ndarray,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
) -> np.ndarray:
    top = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    if top.shape != left.shape or top.shape != right.shape:
        raise ValueError(
            f"G2 test views must share one shape, got "
            f"top={top.shape}, left={left.shape}, right={right.shape}"
        )
    black = np.zeros_like(top)
    return np.concatenate(
        [
            np.concatenate([top, left], axis=1),
            np.concatenate([right, black], axis=1),
        ],
        axis=0,
    )


def _save_rgb_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        raise ValueError(f"No frames to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
    )


def _read_mp4_rgb(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open generated video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Generated video has no frames: {path}")
    return frames


def _add_label_rgb(frame_rgb: np.ndarray, label: str) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame_bgr, (0, 0), (360, 34), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)



def _checkpoint_eval_offsets(
    model_path: Path,
    embodiment_tag: str,
) -> list[int]:
    conf_path = model_path / "experiment_cfg" / "conf.yaml"
    _require_file(conf_path, "checkpoint experiment config")
    cfg = OmegaConf.to_container(OmegaConf.load(conf_path), resolve=True)
    config_key = f"modality_config_{embodiment_tag}"
    try:
        offsets = list(
            cfg[config_key]["video"]["eval_delta_indices"]
        )
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Could not read {config_key}.video.eval_delta_indices "
            f"from {conf_path}"
        ) from exc
    if not offsets:
        offsets = [0]
    return [int(value) for value in offsets]


def _checkpoint_agibot_video_resolutions(
    model_path: Path,
) -> dict[str, tuple[int, int]]:
    """Return required raw input resolutions as camera -> (height, width)."""
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    _require_file(metadata_path, "AgiBot checkpoint metadata")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    try:
        video_meta = metadata["agibot"]["modalities"]["video"]
    except KeyError as exc:
        raise KeyError(
            f"{metadata_path} lacks agibot.modalities.video"
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
                f"Invalid AgiBot raw resolution metadata for {name!r}"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Non-positive AgiBot raw resolution for {name}: "
                f"{width}x{height}"
            )
        result[name] = (height, width)
    return result


def _validate_test_dataset(root: Path) -> dict:
    _require_dir(root, "G2 test split")
    info_path = root / "meta" / "info.json"
    modality_path = root / "meta" / "modality.json"
    embodiment_path = root / "meta" / "embodiment.json"
    for path, description in (
        (info_path, "test info.json"),
        (modality_path, "test modality.json"),
        (embodiment_path, "test embodiment.json"),
    ):
        _require_file(path, description)

    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    with embodiment_path.open("r", encoding="utf-8") as stream:
        embodiment = json.load(stream)

    if embodiment.get("embodiment_tag") != "g2":
        raise ValueError(
            f"Expected test embodiment_tag='g2', got {embodiment}"
        )
    for key in ("observation.state", "action"):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [16]:
            raise ValueError(f"{key} must be 16-D, got {shape}")
    expected_videos = {
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    }
    actual_videos = {
        key
        for key, value in info.get("features", {}).items()
        if value.get("dtype") == "video"
    }
    if actual_videos != expected_videos:
        raise ValueError(
            f"Unexpected G2 test camera keys: {sorted(actual_videos)}"
        )
    return info



def _run_one_test_sample(
    wrapper: AgiBotRoboarenaPolicy,
    args: Args,
    info: dict,
    episode_index: int,
    eval_offsets: list[int],
    raw_resolutions: dict[str, tuple[int, int]],
) -> dict:
    root = Path(args.test_data_root).resolve()
    chunks_size = int(info.get("chunks_size", 1000))
    parquet_path = _episode_file_from_template(
        root,
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        ),
        episode_index,
        chunks_size,
    )
    table = pq.read_table(parquet_path)
    state = _column_to_numpy(table, "observation.state", np.float32)
    num_rows = int(len(state))
    if state.shape != (num_rows, 16):
        raise ValueError(
            f"Unexpected G2 test state shape for episode {episode_index}: "
            f"{state.shape}"
        )

    row_index = args.frame_index
    if row_index < 0:
        row_index = num_rows // 2
    minimum = -min(eval_offsets)
    maximum_future = max(args.future_frames, 9)
    if row_index < minimum:
        raise IndexError(
            f"frame_index={row_index} needs at least {minimum} previous "
            f"frames for offsets {eval_offsets}"
        )
    if row_index + maximum_future > num_rows:
        row_index = num_rows - maximum_future
    if row_index < minimum:
        raise IndexError(
            f"Episode {episode_index} is too short ({num_rows} rows) for "
            f"offsets={eval_offsets}, future_frames={args.future_frames}"
        )

    language_key = "annotation.language.action_text"
    if args.prompt_override is not None:
        prompt = args.prompt_override
    elif language_key in table.column_names:
        prompt = _unwrap_text(table[language_key][row_index].as_py())
    else:
        prompt = ""

    camera_features = {
        "top_head": "observation.images.top_head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    }
    context_indices = [row_index + offset for offset in eval_offsets]
    first_decode = min(context_indices)
    last_decode = min(
        num_rows - 1,
        row_index + args.future_frames - 1,
    )
    decoded: dict[str, dict[int, np.ndarray]] = {}
    video_paths: dict[str, str] = {}
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4",
    )
    for short_name, feature_key in camera_features.items():
        path = _episode_file_from_template(
            root,
            video_template,
            episode_index,
            chunks_size,
            video_key=feature_key,
        )
        video_paths[short_name] = str(path)
        decoded[short_name] = _decode_video_range_bgr(
            path,
            first_decode,
            last_decode,
        )

    sample_name = f"episode_{episode_index:06d}_frame_{row_index:06d}"
    sample_dir = Path(args.output_dir).resolve() / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable source context and source ground truth remain at the
    # original G2 test-set resolution.
    context_grids = [
        _grid_rgb(
            decoded["top_head"][index],
            decoded["hand_left"][index],
            decoded["hand_right"][index],
        )
        for index in context_indices
    ]
    _save_rgb_video(
        sample_dir / f"g2_source_context_f{len(context_grids)}.mp4",
        context_grids,
        fps=30,
    )
    imageio.imwrite(
        sample_dir / "g2_source_context_last.png",
        context_grids[-1],
    )

    gt_indices = list(range(row_index, last_decode + 1))
    gt_grids = [
        _grid_rgb(
            decoded["top_head"][index],
            decoded["hand_left"][index],
            decoded["hand_right"][index],
        )
        for index in gt_indices
    ]
    _save_rgb_video(
        sample_dir / f"g2_ground_truth_future_f{len(gt_grids)}.mp4",
        gt_grids,
        fps=30,
    )
    imageio.imwrite(
        sample_dir / "g2_ground_truth_first.png",
        gt_grids[0],
    )
    imageio.imwrite(
        sample_dir / "g2_ground_truth_last.png",
        gt_grids[-1],
    )

    # Resize the same G2 source images to the exact raw resolution contract
    # stored in the AgiBot checkpoint metadata.
    context_bgr: dict[str, np.ndarray] = {}
    for short_name in camera_features:
        target_h, target_w = raw_resolutions[short_name]
        frames = [
            cv2.resize(
                decoded[short_name][index],
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            for index in context_indices
        ]
        context_bgr[short_name] = np.stack(frames, axis=0)

    current_state = state[row_index]
    obs = {
        "observation/top_head": context_bgr["top_head"],
        "observation/hand_left": context_bgr["hand_left"],
        "observation/hand_right": context_bgr["hand_right"],
        "observation/state": current_state,
        "prompt": prompt,
        "session_id": (
            args.session_id_override
            if args.session_id_override is not None
            else f"offline-{sample_name}-{uuid.uuid4()}"
        ),
    }

    wrapper._output_dir = str(sample_dir)
    wrapper._msg_index = 0
    logger.info(
        "[AGIBOT-ON-G2 TESTSET] episode=%s row=%s context=%s "
        "prompt=%r raw_resolutions=%s parquet=%s",
        episode_index,
        row_index,
        context_indices,
        prompt,
        {
            key: [value[1], value[0]]
            for key, value in raw_resolutions.items()
        },
        parquet_path,
    )
    ignored_actions = wrapper.infer(obs)
    np.save(
        sample_dir / "ignored_hold_actions.npy",
        ignored_actions,
    )

    predicted_candidates = sorted(
        (sample_dir / "diagnostics").glob(
            "pred_chunk_request_000001_f*.mp4"
        )
    )
    comparison_path: str | None = None
    predicted_path: str | None = None
    if predicted_candidates:
        predicted_file = predicted_candidates[-1]
        predicted_path = str(predicted_file)
        predicted_frames = _read_mp4_rgb(predicted_file)
        compare_count = min(len(predicted_frames), len(gt_grids))
        comparison_frames: list[np.ndarray] = []
        for index in range(compare_count):
            pred = predicted_frames[index]
            gt = gt_grids[index]
            if pred.shape[:2] != gt.shape[:2]:
                pred = cv2.resize(
                    pred,
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            comparison_frames.append(
                np.concatenate(
                    [
                        _add_label_rgb(pred, "PREDICTED"),
                        _add_label_rgb(
                            gt,
                            "G2 TEST GROUND TRUTH",
                        ),
                    ],
                    axis=1,
                )
            )
        comparison_file = (
            sample_dir
            / f"agibot_predicted_vs_g2_gt_f{compare_count}.mp4"
        )
        _save_rgb_video(
            comparison_file,
            comparison_frames,
            fps=30,
        )
        imageio.imwrite(
            sample_dir / "agibot_predicted_vs_g2_gt_first.png",
            comparison_frames[0],
        )
        imageio.imwrite(
            sample_dir / "agibot_predicted_vs_g2_gt_last.png",
            comparison_frames[-1],
        )
        comparison_path = str(comparison_file)

    summary = {
        "checkpoint": str(Path(args.model_path).resolve()),
        "checkpoint_embodiment": "agibot",
        "visual_source_embodiment": "g2",
        "test_data_root": str(root),
        "episode_index": episode_index,
        "row_index": row_index,
        "parquet_path": str(parquet_path),
        "video_paths": video_paths,
        "context_offsets": eval_offsets,
        "context_indices": context_indices,
        "checkpoint_raw_resolutions": {
            key: [value[1], value[0]]
            for key, value in raw_resolutions.items()
        },
        "prompt": prompt,
        "g2_state_used_for_model_conditioning": False,
        "predicted_video": predicted_path,
        "comparison_video": comparison_path,
        "action_note": (
            "Native AgiBot actions were intentionally discarded; "
            "this is a video-only diagnostic."
        ),
    }
    with (sample_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary



def main(args: Args) -> None:
    if args.embodiment_tag.lower() != "agibot":
        raise ValueError(
            "This evaluator runs AgiBot checkpoints on G2 test-set images; "
            "use --embodiment-tag agibot"
        )
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
    _require_dir(model_path, "AgiBot checkpoint")
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    conf_path = model_path / "experiment_cfg" / "conf.yaml"
    _require_file(metadata_path, "AgiBot checkpoint metadata")
    _require_file(conf_path, "AgiBot checkpoint config")
    with metadata_path.open("r", encoding="utf-8") as stream:
        checkpoint_metadata = json.load(stream)
    if "agibot" not in checkpoint_metadata:
        raise KeyError(
            f"Checkpoint does not contain AgiBot metadata: "
            f"{metadata_path}; keys={list(checkpoint_metadata)}"
        )

    test_info = _validate_test_dataset(
        Path(args.test_data_root).resolve()
    )
    episode_indices = _parse_episode_indices(
        args.episode_indices,
        int(test_info["total_episodes"]),
    )
    eval_offsets = _checkpoint_eval_offsets(
        model_path,
        "agibot",
    )
    raw_resolutions = _checkpoint_agibot_video_resolutions(
        model_path
    )
    model_config_overrides, train_config_overrides = (
        _build_path_overrides(args)
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "[PREFLIGHT OK] checkpoint=%s checkpoint_embodiment=agibot "
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
        logger.info(
            "Preflight-only validation completed; model was not loaded."
        )
        return

    device_mesh = init_mesh()
    rank = dist.get_rank()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timeout_delta = datetime.timedelta(
        seconds=args.timeout_seconds
    )
    signal_group = dist.new_group(
        backend="gloo",
        timeout=timeout_delta,
    )
    logger.info(
        "Rank %s initialized signal_group (gloo)",
        rank,
    )

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("agibot"),
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
    logger.info(
        "[CONFIG CHECK] rank=%s diffusion_steps=%s "
        "frame_per_block=%s ENABLE_DIT_CACHE=%s",
        rank,
        getattr(action_head, "num_inference_steps", None),
        getattr(action_head, "num_frame_per_block", None),
        os.getenv("ENABLE_DIT_CACHE"),
    )

    if rank == 0:
        wrapper = AgiBotRoboarenaPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            model_path=str(model_path),
            output_dir=str(Path(args.output_dir).resolve()),
            video_save_mode="none",
        )
        summaries: list[dict] = []
        try:
            for episode_index in episode_indices:
                summaries.append(
                    _run_one_test_sample(
                        wrapper,
                        args,
                        test_info,
                        episode_index,
                        eval_offsets,
                        raw_resolutions,
                    )
                )
        finally:
            wrapper._broadcast_signal_to_workers(
                SIGNAL_SHUTDOWN
            )

        report_path = (
            Path(args.output_dir).resolve()
            / "testset_report.json"
        )
        with report_path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                {
                    "checkpoint": str(model_path),
                    "checkpoint_embodiment": "agibot",
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
        logger.info(
            "Saved test-set report: %s",
            report_path,
        )
        dist.barrier()
    else:
        worker = WebsocketPolicyServer(
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
