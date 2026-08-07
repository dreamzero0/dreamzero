import dataclasses
import logging
import socket
import asyncio
import os
import http
import logging
import time
import traceback
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
    port: int = 8000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "./checkpoints/dreamzero"
    wan_ckpt_dir: str | None = None  # Override Wan2.1 component paths stored in config.json.
    tokenizer_path: str | None = None  # Override tokenizer_path stored in experiment_cfg/conf.yaml.
    enable_dit_cache: bool = False  # Backward-compatible alias for num_dit_steps=8.
    num_dit_steps: int | None = None  # Actual DiT compute steps. Supported fast masks: 5, 6, 7, 8. None keeps model default.
    num_inference_timesteps: int | None = None  # Positive values override diffusion steps; 0 keeps checkpoint default.
    index: int = 0
    embodiment_tag: str = "oxe_droid"
    max_chunk_size: int | None = None  # If None, use config value. Otherwise override max_chunk_size for inference.
    video_save_mode: str = "first"  # Generated model video: none, first, or full.
    input_video_save_mode: str = "top_head"  # Real camera input: none, top_head, or all.
    input_video_fps: int = 30
    output_dir: str = "/data/wangk/dreamzero/video_rollout"


class DistributedRoboarenaPolicyBase:
    """Shared distributed inference plumbing for websocket policy wrappers."""

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        video_save_mode: str = "first",
        input_video_save_mode: str = "top_head",
        input_video_fps: int = 30,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir
        self._video_save_mode = video_save_mode
        self._input_video_save_mode = input_video_save_mode
        self._input_video_fps = int(input_video_fps)
        self._frame_buffers = self._init_frame_buffers()
        self._current_session_id: str | None = None
        self.video_across_time = []
        self._input_video_buffers: dict[str, list[np.ndarray]] = {}
        self._msg_index = 0

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _init_frame_buffers(self) -> dict[str, list[np.ndarray]]:
        return {}

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

    @staticmethod
    def _sanitize_filename(value: object) -> str:
        text = str(value or "session")
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in text
        )
        return safe[:80] or "session"

    def _record_input_video(self, modality_key: str, frames: np.ndarray) -> None:
        if self._input_video_save_mode == "none":
            return
        if self._input_video_save_mode == "top_head" and modality_key != "video.top_head":
            return

        array = np.asarray(frames)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            logger.warning(
                "Skipping input video %s with invalid shape %s",
                modality_key,
                array.shape,
            )
            return

        buffer = self._input_video_buffers.setdefault(modality_key, [])
        for frame in array:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            buffer.append(frame.copy())

    def _save_input_videos(self) -> None:
        if not self._output_dir or not self._input_video_buffers:
            return

        input_dir = os.path.join(self._output_dir, "inputs")
        os.makedirs(input_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = self._sanitize_filename(self._current_session_id)

        for modality_key, frames in self._input_video_buffers.items():
            if not frames:
                continue

            camera_name = modality_key.replace("video.", "")
            output_path = os.path.join(
                input_dir,
                f"{timestamp}_{session_name}_{camera_name}_f{len(frames)}.mp4",
            )

            try:
                imageio.mimsave(
                    output_path,
                    frames,
                    fps=self._input_video_fps,
                    codec="libx264",
                    macro_block_size=None,
                )
                logger.info(
                    "Saved real input video %s (%s frames) to %s",
                    modality_key,
                    len(frames),
                    output_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to save real input video %s: %s",
                    modality_key,
                    exc,
                )

        self._input_video_buffers = {}

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

        video_chunk = self._prepare_video_chunk(video_pred)
        if video_chunk is not None:
            self.video_across_time.append(video_chunk.detach().cpu())
        action = self._convert_action(self._extract_action_dict(result_batch.act))
        self._after_infer()
        return action

    def _reset_state(self, save_video: bool = True) -> None:
        if save_video:
            self._save_input_videos()

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


class AgiBotRoboarenaPolicy(DistributedRoboarenaPolicyBase):
    """Adapter that converts websocket observations into AgiBot modality keys."""

    VIDEO_KEY_MAPPING = {
        'observation/top_head': 'video.top_head',
        'observation/hand_left': 'video.hand_left',
        'observation/hand_right': 'video.hand_right',
    }
    STATE_KEY_MAPPING = {
        'observation/left_arm_joint_position': 'state.left_arm_joint_position',
        'observation/right_arm_joint_position': 'state.right_arm_joint_position',
        'observation/left_effector_position': 'state.left_effector_position',
        'observation/right_effector_position': 'state.right_effector_position',
        'observation/head_position': 'state.head_position',
        'observation/waist_pitch': 'state.waist_pitch',
        'observation/waist_lift': 'state.waist_lift',
    }

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        video_save_mode: str = "first",
        input_video_save_mode: str = "top_head",
        input_video_fps: int = 30,
    ) -> None:
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
            input_video_save_mode=input_video_save_mode,
            input_video_fps=input_video_fps,
        )
        self._action_keys = list(self._policy.modality_configs.action.modality_keys)

    def _lookup_obs_value(self, obs: dict, source_key: str, target_key: str) -> object:
        if source_key in obs:
            return obs[source_key]
        return obs.get(target_key)

    def _normalize_video(self, value: object, target_key: str) -> np.ndarray:
        if isinstance(value, dict) and value.get("__dreamzero_image_encoding__") == "jpeg_sequence":
            frames = []
            expected_shape = tuple(value.get("shape", ()))
            expected_dtype = np.dtype(value.get("dtype", "uint8"))
            for index, frame_bytes in enumerate(value.get("frames", [])):
                encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Failed to decode JPEG frame {index} for {target_key}")
                frames.append(frame.astype(expected_dtype, copy=False))
            array = np.stack(frames, axis=0)
            if expected_shape and tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Decoded JPEG video for {target_key} has shape {array.shape}, expected {expected_shape}"
                )
            return array

        array = np.asarray(value)
        if array.ndim == 3:
            return np.expand_dims(array, axis=0)
        if array.ndim == 4:
            return array
        raise ValueError(f'AgiBot video input for {target_key} must have shape (H, W, C) or (T, H, W, C), got {array.shape}')

    def _normalize_state(self, value: object, target_key: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 0:
            return array.reshape(1, 1).astype(np.float64)
        if array.ndim == 1:
            return array.reshape(1, -1).astype(np.float64)
        if array.ndim == 2:
            return array.astype(np.float64)
        raise ValueError(f'AgiBot state input for {target_key} must be 1D or 2D, got {array.shape}')

    def _prepare_video_chunk(self, video_pred: torch.Tensor) -> torch.Tensor | None:
        if self._video_save_mode == "none":
            return None
        if video_pred.ndim != 5:
            raise ValueError(f'AgiBot video prediction must be 5D (B, C, T, H, W), got {tuple(video_pred.shape)}')
        if self._video_save_mode == "first":
            return video_pred[:, :, :1].contiguous()
        if self._video_save_mode == "full":
            return video_pred.contiguous()
        raise ValueError(f"Unsupported video_save_mode: {self._video_save_mode!r}; expected none, first, or full")

    def _video_save_fps(self) -> int:
        return 20

    def _convert_observation(self, obs: dict) -> dict:
        converted = {}
        missing_keys: list[str] = []

        for source_key, target_key in self.VIDEO_KEY_MAPPING.items():
            value = self._lookup_obs_value(obs, source_key, target_key)
            if value is None:
                missing_keys.append(source_key)
                continue
            normalized_video = self._normalize_video(value, target_key)
            converted[target_key] = normalized_video
            self._record_input_video(target_key, normalized_video)

        for source_key, target_key in self.STATE_KEY_MAPPING.items():
            value = self._lookup_obs_value(obs, source_key, target_key)
            if value is None:
                missing_keys.append(source_key)
                continue
            converted[target_key] = self._normalize_state(value, target_key)

        if missing_keys:
            raise ValueError(
                'AgiBot inference requires the following observation keys: '
                + ', '.join(sorted(missing_keys))
            )

        converted['annotation.language.action_text'] = obs.get('prompt', obs.get('annotation.language.action_text', ''))
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        flattened_chunks: list[np.ndarray] = []
        expected_horizon: int | None = None
        missing_keys = [key for key in self._action_keys if key not in action_dict]
        if missing_keys:
            raise RuntimeError('Missing AgiBot action outputs: ' + ', '.join(missing_keys))

        for action_key in self._action_keys:
            value = action_dict[action_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim == 0:
                array = array.reshape(1, 1)
            elif array.ndim == 1:
                array = array.reshape(-1, 1)
            else:
                array = array.reshape(array.shape[0], -1)

            if expected_horizon is None:
                expected_horizon = array.shape[0]
            elif array.shape[0] != expected_horizon:
                raise RuntimeError(
                    f'Inconsistent AgiBot action horizon for {action_key}: expected {expected_horizon}, got {array.shape[0]}'
                )
            flattened_chunks.append(array.astype(np.float32))

        return np.concatenate(flattened_chunks, axis=-1).astype(np.float32)


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
    input_video_save_mode: str,
    input_video_fps: int,
) -> DistributedRoboarenaPolicyBase:
    if embodiment_tag == 'oxe_droid':
        return ARDroidRoboarenaPolicy(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
        )
    if embodiment_tag == 'agibot':
        return AgiBotRoboarenaPolicy(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
            video_save_mode=video_save_mode,
            input_video_save_mode=input_video_save_mode,
            input_video_fps=input_video_fps,
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

    return model_config_overrides, train_config_overrides


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    if args.num_dit_steps is not None:
        os.environ["NUM_DIT_STEPS"] = str(args.num_dit_steps)
    elif args.enable_dit_cache:
        os.environ.setdefault("NUM_DIT_STEPS", "8")
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    if args.video_save_mode not in {"none", "first", "full"}:
        raise ValueError(
            f"--video-save-mode must be one of none, first, full; "
            f"got {args.video_save_mode!r}"
        )
    if args.input_video_save_mode not in {"none", "top_head", "all"}:
        raise ValueError(
            f"--input-video-save-mode must be one of none, top_head, all; "
            f"got {args.input_video_save_mode!r}"
        )
    if args.input_video_fps <= 0:
        raise ValueError(
            f"--input-video-fps must be positive, got {args.input_video_fps}"
        )
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = args.embodiment_tag.lower()
    if embodiment_tag not in {'oxe_droid', 'agibot'}:
        raise ValueError(f'Unsupported embodiment_tag: {args.embodiment_tag}')
    model_path = args.model_path
    model_config_overrides, train_config_overrides = _build_path_overrides(args)
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
        "wan_ckpt_dir": args.wan_ckpt_dir,
        "tokenizer_path": args.tokenizer_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        model_config_overrides=model_config_overrides,
        train_config_overrides=train_config_overrides,
    )
    action_head = policy.trained_model.action_head
    if args.num_inference_timesteps is not None:
        if args.num_inference_timesteps == 0:
            logging.info("Keeping checkpoint diffusion inference steps because --num-inference-timesteps=0")
        elif args.num_inference_timesteps < 0:
            raise ValueError(
                f"--num-inference-timesteps must be non-negative, got {args.num_inference_timesteps}"
            )
        else:
            action_head.num_inference_steps = int(args.num_inference_timesteps)
            action_head.num_inference_timesteps = int(args.num_inference_timesteps)
            if hasattr(action_head, "config"):
                action_head.config.num_inference_timesteps = int(args.num_inference_timesteps)
            logging.info(
                "Overrode action_head diffusion inference steps to %s on rank %s",
                args.num_inference_timesteps,
                rank,
            )
    logging.info(
        "[CONFIG CHECK] rank=%s action_head.num_inference_steps=%s "
        "action_head.num_inference_timesteps=%s action_head.num_frame_per_block=%s "
        "action_head.model.num_frame_per_block=%s NUM_DIT_STEPS=%s ENABLE_DIT_CACHE=%s",
        rank,
        getattr(action_head, "num_inference_steps", None),
        getattr(action_head, "num_inference_timesteps", None),
        getattr(action_head, "num_frame_per_block", None),
        getattr(getattr(action_head, "model", None), "num_frame_per_block", None),
        os.getenv("NUM_DIT_STEPS"),
        os.getenv("ENABLE_DIT_CACHE"),
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        output_dir = None if args.video_save_mode == "none" else args.output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            logging.info("Generated videos will be saved to: %s", output_dir)
            logging.info(
                "Real input videos will be saved to: %s",
                os.path.join(output_dir, "inputs"),
            )
        else:
            logging.info("Video saving disabled; no output directory will be created.")
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")

    wrapper_policy = _create_wrapper_policy(
        embodiment_tag=embodiment_tag,
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
        video_save_mode=args.video_save_mode,
        input_video_save_mode=args.input_video_save_mode,
        input_video_fps=args.input_video_fps,
    )

    server_config = _create_server_config(embodiment_tag)

    if rank == 0:
        logging.info("Using roboarena policy server interface for %s", embodiment_tag)
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
