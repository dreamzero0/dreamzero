from dataclasses import dataclass, field
from typing import Tuple

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class VLA(PreTrainedModel):
    """
    DreamZero Vision-Language-Action模型主类：组合backbone与action_head的统一接口。

    【作用与原理】
    VLA类是DreamZero模型的统一入口，采用模块化设计：
    1. **Backbone**: 视觉-语言编码器（如IdentityBackbone、Qwen2-VL等），提取多模态特征
    2. **ActionHead**: 动作生成头（如WANPolicyHead），基于backbone特征预测动作序列
    3. **输入划分**: prepare_input()将统一输入字典划分为backbone_inputs和action_inputs
    4. **前向传播**: forward()按序调用backbone→action_head，返回包含loss或action_pred的BatchFeature

    【数据流位置】
    上游：DefaultDataCollator输出的batch字典（images, text, state, action等）
    当前：VLA
    下游：Trainer.compute_loss()处理返回的BatchFeature中的loss

    【设计模式】
    - 松耦合：backbone和action_head通过配置文件动态实例化，可独立替换
    - 验证机制：validate_inputs()检查输入格式，validate_data()检查输出格式
    - 多模式支持：forward()用于训练（返回loss），get_action()用于推理（返回action_pred）

    【关键概念】
    - BACKBONE_FEATURE_KEY: "backbone_features"，backbone输出中特征张量的键名
    - ACTION_KEY: "action_pred"，action_head推理输出的动作预测键名
    - LOSS_KEY: "loss"，action_head训练输出的损失键名
    """

    supports_gradient_checkpointing = True
    config_class = VLAConfig

    def __init__(
        self,
        config: VLAConfig,
    ):
        """
        初始化VLA模型。

        Args:
            config (VLAConfig): VLA配置，包含：
                - backbone_cfg: Backbone配置字典（如IdentityBackbone）
                - action_head_cfg: ActionHead配置字典（如WANPolicyHead）
                - action_horizon: 动作序列长度（如24）
                - action_dim: 动作维度（如64）
                - compute_dtype: 计算精度（如"float32"/"bfloat16"）

        【初始化流程】
        1. 验证config类型
        2. 调用父类PreTrainedModel.__init__()
        3. 通过hydra instantiate()动态创建backbone和action_head
        4. 保存action_horizon、action_dim、compute_dtype
        """
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] % self.action_horizon == 0
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        """
        训练模式前向传播：执行backbone编码→action_head预测→返回loss。

        【输入】
        - inputs (dict): DataCollator输出的batch字典，包含：
          * "images": (B, T, H, W, 3) uint8，拼图后的视频帧
          * "text": (B, L_text) int64，语言token ids
          * "text_attention_mask": (B, L_text) int64，语言mask
          * "state": (B, T_s, max_state_dim) float，状态
          * "action": (B, T_a, max_action_dim) float，目标动作（用于计算loss）
          * "action_mask": (B, T_a, max_action_dim) bool，动作有效mask
          * "embodiment_id": (B,) int，本体标签
          * "has_real_action": (B,) bool，是否计算action loss

        【处理流程】
        1. prepare_input(): 将inputs划分为backbone_inputs和action_inputs
        2. backbone.forward(): 提取视觉-语言特征（如IdentityBackbone返回占位符）
        3. action_head.forward(): 执行Flow Matching训练（VAE编码、加噪、DiT预测、计算MSE loss）
        4. 返回包含loss的BatchFeature

        【输出】
        - BatchFeature: 包含训练结果的字典，必须包含：
          * "loss": scalar tensor，总损失（反向传播用）
          * "dynamics_loss": scalar tensor，视频/潜空间分支loss
          * "action_loss": scalar tensor，动作分支loss

        【Shape变化】
        - inputs["images"]: (B, T, H, W, 3) → VAE编码 → latents: (B, C_lat, T_lat, H_lat, W_lat)
        - inputs["action"]: (B, T_a, D_a) → 加噪 → noisy_actions: 同shape
        - action_head输出loss: scalar

        【调用关系】
        - 被: VLATrainer.compute_loss()（通过model(inputs)调用）
        - 调用: prepare_input(), backbone.forward(), action_head.forward()

        Args:
            inputs (dict): Batch输入字典。

        Returns:
            BatchFeature: 包含loss的训练输出。
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal_gt_cond(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        action_head_outputs = self.action_head.lazy_joint_video_action_causal_gt_cond(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def lazy_joint_video_action_efficient(
        self,
        inputs: dict,
        prompt_embs: torch.Tensor | None = None,
        prompt_emb_nega: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action_efficient(backbone_outputs, action_inputs, prompt_embs=prompt_embs, prompt_emb_nega=prompt_emb_nega)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def gt_video_action_pred(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.gt_video_action_pred(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def get_language(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone.generate(backbone_inputs)
        return backbone_outputs

    def get_video(
        self,
        inputs: dict,
    ) -> BatchFeature:
        _, video_inputs = self.prepare_input(inputs)
        video_outputs = self.action_head.get_video(video_inputs)
        return video_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        """
        输入准备：划分backbone与action_head的输入，并处理设备/精度。

        【输入】
        - inputs (dict): DataCollator输出的batch字典（见forward()的输入说明）。

        【处理流程】
        1. validate_inputs(): 校验输入格式（action shape、video dtype等）
        2. backbone.prepare_input(): 提取backbone需要的子集（如IdentityBackbone用action推断batch）
        3. action_head.prepare_input(): 提取action_head需要的子集（通常为整个batch）
        4. 设备/精度转换：
           - 浮点张量：转到self.device并转换为action_head.dtype（如bfloat16）
           - 非浮点（如images uint8）：仅转设备，保持dtype

        【输出】
        - Tuple[BatchFeature, BatchFeature]: (backbone_inputs, action_inputs)
          * backbone_inputs: backbone专用输入（如IdentityBackbone只需要action推断B）
          * action_inputs: action_head完整输入（images, text, state, action等）

        【Shape变化】
        - 输入张量：可能在CPU → 输出：在self.device（如cuda）
        - 浮点精度：float32 → bfloat16（若action_head.dtype为bf16）
        - 图像保持uint8（非浮点）

        【调用关系】
        - 被: forward(), get_action(), joint_video_action()等所有前向方法
        - 调用: validate_inputs(), backbone.prepare_input(), action_head.prepare_input()

        Returns:
            Tuple[BatchFeature, BatchFeature]: (backbone_inputs, action_inputs)
        """
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            """将张量转移到设备，浮点张量额外转换dtype。"""
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs


    @classmethod
    def from_pretrained_for_tuning(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None,  # This config will now be USED
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        offload_state_dict: bool = True,
        lora_weights_path: str | None = None,
    ):
        if config is None:
            raise ValueError(
                "A `config` object must be provided to build the model structure."
            )

        import os
        import json
        import gc
        from safetensors.torch import load_file

        model = cls(config)

        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        if os.path.exists(safetensors_index_path):
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            missing_keys_accum = set()
            unexpected_keys_accum = set()
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                missing_keys, unexpected_keys = model.load_state_dict(shard_state_dict, strict=False)
                if missing_keys:
                    missing_keys_accum.update(missing_keys)
                if unexpected_keys:
                    unexpected_keys_accum.update(unexpected_keys)
                # Free shard immediately
                del shard_state_dict
                gc.collect()
            if missing_keys_accum:
                print(f"Missing keys when loading sharded pretrained weights: {sorted(missing_keys_accum)} ... total={len(missing_keys_accum)}")
            if unexpected_keys_accum:
                print(f"Unexpected keys when loading sharded pretrained weights: {sorted(unexpected_keys_accum)} ... total={len(unexpected_keys_accum)}")
            if not missing_keys_accum and not unexpected_keys_accum:
                print("Successfully loaded pretrained base weights (sharded)")
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict = load_file(safetensors_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                print("Successfully loaded pretrained base weights")
        else:
            raise FileNotFoundError(
                f"No weights found at '{pretrained_model_name_or_path}'. "
                "Expected 'model.safetensors' or 'model.safetensors.index.json'."
            )

        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}")
            model.load_lora_weight(lora_weights_path)
        else:
            if hasattr(model, 'action_head') and hasattr(model.action_head, 'inject_lora_after_loading') and model.action_head.config.defer_lora_injection:
                print("Injecting LoRA adapters into action_head after loading pretrained weights")
                model.action_head.inject_lora_after_loading()
        
        print(f"{cls}\n")
        return model

    @classmethod
    def load_lora(
        cls, 
        pretrained_model_name_or_path: str
    ): 
        from safetensors.torch import load_file
        import os
        import json
        print("loading lora@@@@@")

        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
        
        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        
        # Load config
        print("loading config@@")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        print("loading model")

        # Disable defer_lora_injection so LoRA layers are created during init,
        # matching the PEFT key hierarchy (base_model.model.*) in the checkpoint.
        ah_cfg = config.action_head_cfg
        inner = ah_cfg.get('config', ah_cfg) if isinstance(ah_cfg.get('config'), dict) else ah_cfg
        if 'defer_lora_injection' in inner:
            inner['defer_lora_injection'] = False
            print("defer_lora_injection disabled for load_lora")
        # Enable component loading so DiT base weights are loaded from pretrained
        if 'skip_component_loading' in inner:
            inner['skip_component_loading'] = False
            print("skip_component_loading disabled for load_lora")

        # Instantiate model (LoRA layers now exist from init)
        model = cls(config)

        # Remove .base_layer from keys if present
        has_base_layer = any(".base_layer." in key for key in state_dict.keys())
        if has_base_layer:
            print("Removing '.base_layer' from state dict keys")
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        print("Successfully loaded pretrained weights")

        print(f"{cls}\n")
        return model

    def load_lora_weight(self, pretrained_model_name_or_path: str):
        """Load only LoRA weights from a pretrained model without loading config."""
        from safetensors.torch import load_file
        import os
        import json
        
        print(f"Loading LoRA weights from {pretrained_model_name_or_path}")
        
        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        else:
            raise FileNotFoundError(f"No valid checkpoint found at {pretrained_model_name_or_path}")
        
        print("Loading LoRA weights into existing model")

        def rewrite_lora_state_dict_keys(state_dict, pattern, repl):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(pattern, repl)
                new_state_dict[new_k] = v
            return new_state_dict

        has_target_pattern = any("action_head.model.base_model.model" in key for key in state_dict.keys())
        
        if not has_target_pattern:
            print("Rewriting LoRA state dict keys from 'action_head.model' to 'action_head.model.base_model.model'")
            state_dict = rewrite_lora_state_dict_keys(
                state_dict,
                pattern="action_head.model",
                repl="action_head.model.base_model.model",
            )
        else:
            print("State dict already has 'action_head.model.base_model.model' pattern, skipping key rewrite")
        
        # Load only the weights into the existing model
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        print("Successfully loaded LoRA state dict")
            
        if missing_keys:
            print(f"Missing keys when loading LoRA weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading LoRA weights: {unexpected_keys}")
        
        print("Successfully loaded LoRA weights")

    @classmethod
    def from_config_with_lora_weights(
        cls,
        config: VLAConfig,
        pretrained_model_path: str,
    ):
        """Create VLA model from config and then load LoRA weights from pretrained model."""
        print(f"Creating VLA model from config and loading LoRA weights from {pretrained_model_path}")
        
        # 1. Create model from config (similar to vla.yaml)
        model = cls(config)
        print("Model created from config")
        
        # 2. Load LoRA weights into the created model
        model.load_lora_weight(pretrained_model_path)
        
        return model

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None
    ):
        del config

        from safetensors.torch import load_file
        import os
        import json
        print("loading pretrained@@@@@")
        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        
        # Load config
        print("loading config@@")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        print("loading model")
        print("config.action_head_cfg", config.action_head_cfg)
        # Always disable defer_lora_injection
        # config.action_head_cfg is a dict, and defer_lora_injection is nested in config.action_head_cfg['config']
        if 'config' in config.action_head_cfg and isinstance(config.action_head_cfg['config'], dict):
            if 'defer_lora_injection' in config.action_head_cfg['config']:
                config.action_head_cfg['config']['defer_lora_injection'] = False
                print("config.action_head_cfg['config']['defer_lora_injection'] disabled (set to False)")
        elif 'defer_lora_injection' in config.action_head_cfg:
            config.action_head_cfg['defer_lora_injection'] = False
            print("config.action_head_cfg['defer_lora_injection'] disabled (set to False)")

        # Instantiate model
        model = cls(config)
        print("model", model)
        # Remove .base_layer from keys (e.g., 'action_head.model.base_model.model.blocks.19.self_attn.v.base_layer.bias' -> 'action_head.model.base_model.model.blocks.19.self_attn.v.bias')
        has_base_layer = any(".base_layer." in key for key in state_dict.keys())
        if has_base_layer:
            print("Removing '.base_layer' from state dict keys")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(".base_layer.", ".")
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        print("Successfully loaded pretrained weights")

        print(f"{cls}\n")
        return model

    def post_initialize(self):
        self.action_head.post_initialize()

    def parallelize(self, device_mesh: DeviceMesh):
        self.action_head.parallelize(device_mesh=device_mesh)


class CotrainVLA(VLA):

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        if "cotrain" in inputs and inputs["cotrain"]:
            return self.backbone.cotrain(inputs)
        return super().forward(inputs)


def create_vla_with_pretrained_action_head(pretrained_vla_path: str, config: VLAConfig):
    # 1. Instantiate a new VLAModel
    vla = VLA(config)

    # 2. Load the pretrained VLAModel
    pretrained_vla = VLA.from_pretrained(pretrained_vla_path)

    # 3. Replace the action head in the new VLAModel with the pretrained action head
    vla.action_head = pretrained_vla.action_head

    # 4. Replace the action head config in the new VLAModel with the pretrained action head config
    vla.config.action_head_cfg = pretrained_vla.config.action_head_cfg

    # 5. Return the new VLAModel
    return vla


# register
AutoConfig.register("vla", VLAConfig)
AutoModel.register(VLAConfig, VLA)
