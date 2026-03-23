from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field, PrivateAttr
import torch

from groot.vla.data.schema import DatasetMetadata, StateActionMetadata
from groot.vla.data.transform.base import InvertibleModalityTransform, ModalityTransform


class ConcatTransform(InvertibleModalityTransform):
    """
    多模态拼接变换：将同一模态下的多个子键沿指定维度拼接，形成统一的tensor。

    【作用与原理】
    该类是ComposedModalityTransform管道中的关键一步，负责：
    1. **Video拼接**：将多个相机视角（如wrist、left_exterior、right_exterior）沿新轴拼接
    2. **State拼接**：将多个state子键（如joint_position、gripper_position）沿最后一维拼接
    3. **Action拼接**：将多个action子键沿最后一维拼接

    【数据流位置】
    上游：StateActionToTensor、VideoResize等预处理变换（已将各子键转为numpy/torch张量）
    当前：ConcatTransform
    下游：DreamTransform（拼图、padding等模型特定处理）

    【拼接规则】
    - Video: 沿axis=-4（view轴）拼接，输出[..., V, H, W, C]
      例如：3个view各为(T,H,W,C) → 拼接为(T,3,H,W,C)
    - State: 沿axis=-1（特征维）用torch.cat拼接，输出[T, D_state_total]
      例如：joint_position(7) + gripper_position(1) → [T, 8]
    - Action: 同state，沿axis=-1拼接

    【配置示例】（DROID）
    ```yaml
    video_concat_order:
      - video.exterior_image_1_left
      - video.exterior_image_2_left
      - video.wrist_image_left
    state_concat_order:
      - state.joint_position      # 7维
      - state.gripper_position    # 1维 → 总8维
    action_concat_order:
      - action.joint_position     # 7维
      - action.gripper_position   # 1维 → 总8维
    ```

    【输入输出】
    - 输入: dict，键为"video.xxx"、"state.xxx"、"action.xxx"等
    - 输出: dict，原多键被替换为统一键"video"、"state"、"action"
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="此transform不使用apply_to，保留用于兼容性。"
    )

    video_concat_order: list[str] = Field(
        ...,
        description="Video子键的拼接顺序，决定view轴顺序。"
        "格式: ['video.wrist', 'video.left', 'video.right']",
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="State子键拼接顺序。格式: ['state.joint_position', 'state.gripper_position']",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Action子键拼接顺序。格式: ['action.joint_position', 'action.gripper_position']",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="各action子键的维度（用于形状校验）。",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="各state子键的维度（用于形状校验）。",
    )

    action_dims_post_transform: dict[str, int] = Field(
        default_factory=dict,
        description="拼接后各action子键在新tensor中的维度范围（由apply计算填充）。",
    )
    state_dims_post_transform: dict[str, int] = Field(
        default_factory=dict,
        description="拼接后各state子键在新tensor中的维度范围（由apply计算填充）。",
    )
    # 存储transform管道引用，用于检查维度变换
    _transform_pipeline: List[ModalityTransform] = PrivateAttr(default_factory=list)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def set_transform_pipeline(self, transforms: List[ModalityTransform]):
        """Set the transform pipeline so this transform can examine it for dimension changes."""
        self._transform_pipeline = transforms

    def _get_target_rotations_from_pipeline(self) -> Dict[str, str]:
        """Extract target_rotations from StateActionTransform instances in the pipeline."""
        target_rotations = {}
        for transform in self._transform_pipeline:
            if hasattr(transform, "target_rotations"):
                transform_target_rotations = getattr(transform, "target_rotations", {})
                if transform_target_rotations:
                    target_rotations.update(transform_target_rotations)
        return target_rotations

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多模态拼接变换。

        【输入】
        - data (dict): 预处理后的数据字典，键格式为"modality.subkey"。
          例如：{
            "video.wrist_image_left": (T, H, W, C) ndarray,
            "video.exterior_image_1_left": (T, H, W, C) ndarray,
            "state.joint_position": (T, 7) tensor,
            "state.gripper_position": (T, 1) tensor,
            "action.joint_position": (T_a, 7) tensor,
            ...
          }

        【处理流程】
        1. 按键名分组：将"video.xxx"、"state.xxx"、"action.xxx"等分组到各自模态
        2. Video拼接：
           - 对每个video_key，np.expand_dims(..., axis=-4)插入view维度
           - np.concatenate沿axis=-4拼接所有view
           - 输出shape: (T, V, H, W, C)，其中V=len(video_concat_order)
        3. State拼接：
           - 按state_concat_order顺序pop各子键
           - torch.cat(..., dim=-1)沿特征维拼接
           - 同时填充state_dims_post_transform记录各子键在新tensor中的位置
           - 输出shape: (T, D_state_total)
        4. Action拼接：同state

        【输出】
        - data (dict): 拼接后的字典，原多键被替换为统一键。
          例如：{
            "video": (T, V, H, W, C) ndarray,
            "state": (T, D_state_total) tensor,
            "action": (T_a, D_action_total) tensor,
            ...（保留未处理的键如language等）
          }

        【Shape示例】（DROID）
        - 输入video: 3个key各为(25, 256, 320, 3) → 输出(25, 3, 256, 320, 3)
        - 输入state: joint_position(1,7) + gripper_position(1,1) → 输出(1, 8)
        - 输入action: joint_position(24,7) + gripper_position(24,1) → 输出(24, 8)

        【校验】
        - 断言所有concat_order中的键必须存在于data
        - 断言state/action各子键的最后维必须与配置维度一致（考虑旋转表示的特殊情况）

        Returns:
            Dict[str, Any]: 拼接后的数据字典。
        """
        # Step 1: 按键名前缀分组（video/state/action/language/others）
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        # Step 2: Video拼接 → 沿axis=-4（view轴）
        if "video" in grouped_keys:
            # 校验concat_order配置正确性
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None, f"{self.video_concat_order=}, {video_keys=}"
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            # Process each video view
            unsqueezed_videos = []
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)
                # [..., H, W, C] -> [..., 1, H, W, C]
                unsqueezed_video = np.expand_dims(video_data, axis=-4)
                unsqueezed_videos.append(unsqueezed_video)
            # Concatenate along the new axis → [..., V, H, W, C]
            unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)

            data["video"] = unsqueezed_video

        # Step 3: State拼接 → 沿axis=-1（特征维）
        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            assert all(
                item in state_keys for item in self.state_concat_order
            ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
            # 校验各子键维度（考虑旋转表示可能有多种形状）
            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.extend([3, 4, 6])  # axis_angle/quaternion/rotation_6d
                target_shapes.append(self.state_dims[key] * 2)  # sin-cos transform
                assert data[key].shape[-1] in target_shapes, \
                    f"State dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
            # Concatenate → [T, D_state_total]
            data["state"] = torch.cat(
                [data.pop(key) for key in self.state_concat_order], dim=-1
            )

        # Step 4: Action拼接 → 同state
        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            assert self.action_concat_order is not None, f"{self.action_concat_order=}"
            assert set(self.action_concat_order) == set(
                action_keys
            ), f"{set(self.action_concat_order)=}, {set(action_keys)=}"
            # Record the action dims
            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.extend(
                        [3, 4, 6]
                    )  # 3 -> axis_angle, 4 -> quaternion, 6 -> rotation_6d
                assert (
                    data[key].shape[-1] in target_shapes
                ), f"Action dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
            # Concatenate the action keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["action"] = torch.cat(
                [data.pop(key) for key in self.action_concat_order], dim=-1
            )  # [T, D_action]

        return data

    def unapply(self, data: dict) -> dict:
        start_dim = 0
        assert "action" in data, f"{data.keys()=}"
        # For those dataset without actions (LAPA), we'll never run unapply
        assert self.action_concat_order is not None, f"{self.action_concat_order=}"
        action_tensor = data.pop("action")
        for key in self.action_concat_order:
            if key not in self.action_dims:
                raise ValueError(f"Action dim {key} not found in action_dims.")
            end_dim = start_dim + self.get_state_action_dims_post_transform(key)
            data[key] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim
        if "state" in data:
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                end_dim = start_dim + self.get_state_action_dims_post_transform(key)
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
        assert isinstance(
            modality_config[subkey], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {subkey=}, got {type(modality_config[subkey])=}"
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def get_state_action_dims_post_transform(self, key: str) -> int:
        """
        This function is used to get the dims of the state/action keys after transform is applied.
        It is different from the `get_state_action_dims` function, because this function accounts for
        the case where we apply transforms and the # of dims is change eg. after applying axis_angle transform on
        quaternion, the dims change from 4D to 3D.
        """
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"

        if self.is_rotation_key(key):
            target_rotations = self._get_target_rotations_from_pipeline()
            if key in target_rotations:
                target_rotation = target_rotations[key]
                if target_rotation == "axis_angle":
                    return 3
                elif target_rotation == "quaternion":
                    return 4
                elif target_rotation == "rotation_6d":
                    return 6
                elif target_rotation == "euler_angles":
                    return 3
                else:
                    raise ValueError(f"Unknown target rotation type: {target_rotation}")
            else:
                # No target rotation specified, return original dimension
                return shape[0]
        else:
            return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        # Pre-compute the dimensions of the state and action keys
        if self.action_concat_order is not None:
            for key in self.action_concat_order:
                self.action_dims[key] = self.get_state_action_dims(key)
        if self.state_concat_order is not None:
            for key in self.state_concat_order:
                self.state_dims[key] = self.get_state_action_dims(key)
