import random
from typing import Any, Dict, List, Optional

from einops import rearrange
import numpy as np
from pydantic import Field, PrivateAttr
import torch
from transformers import AutoProcessor, ProcessorMixin, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
import tree
import re
import ftfy
import html
import regex as re
import ast

from groot.vla.data.schema import (
    EmbodimentTag,
    DatasetMetadata,
)
from groot.vla.data.transform.base import InvertibleModalityTransform
from groot.vla.model.dreamzero.transform.common import formalize_language


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)


        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        # elif self.clean == 'lower':
        #     text = whitespace_clean(basic_clean(text)).lower()
        # elif self.clean == 'canonicalize':
        #     text = canonicalize(basic_clean(text))
        return text


def collate(features: List[dict], tokenizer: AutoTokenizer, num_views=3, embodiment_tag_mapping=None) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                item = elem[key]
                try:
                    parsed_item = ast.literal_eval(item)
                    # Handle different return types from ast.literal_eval
                    if isinstance(parsed_item, (list, tuple)):
                        processed_item = str(parsed_item[0])
                    else:
                        # If it's already a scalar (string, float, int, etc.), convert to string
                        processed_item = str(parsed_item)
                    
                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:
                        processed_item = "A multi-view video shows that a robot " + processed_item.lower() + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GR1_UNIFIED.value]:
                        processed_item = "A single view video shows that a human " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.MECKA_HANDS.value]:
                        processed_item = "A single view video shows that a human " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.XDOF.value]:
                        processed_item = "A multi-view video shows that a robot " + processed_item.lower() + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.YAM.value]:
                        processed_item = "A multi-view video shows that a robot " + processed_item.lower() + " The video is split into four views: The top-left view shows the top camera, the top-right view shows the right camera, the bottom-left view shows the left camera, and the bottom-right view is a black screen. The robot " + processed_item.lower()
                    else:
                        raise ValueError(f"Embodiment ID {elem['embodiment_id']} not supported.") 
                    output_values.append(processed_item)  
                except (ValueError, SyntaxError, TypeError):
                    # If parsing fails or item is already a string, use it directly
                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:
                        item = "A multi-view video shows that a robot " + str(item).lower() + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot " + str(item).lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GR1_UNIFIED.value]:
                        item = "A single view video shows that a human " + str(item).lower() 
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.MECKA_HANDS.value]:
                        item = "A single view video shows that a human " + str(item).lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.XDOF.value]:
                        item = "A multi-view video shows that a robot " + str(item).lower() + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot " + str(item).lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.YAM.value]:
                        item = "A multi-view video shows that a robot " + str(item).lower() + " The video is split into four views: The top-left view shows the top camera, the top-right view shows the right camera, the bottom-left view shows the left camera, and the bottom-right view is a black screen. The robot " + str(item).lower()
                    else:
                        raise ValueError(f"Embodiment ID {elem['embodiment_id']} not supported.")   
                    output_values.append(item)
            # print("output_values", output_values)
            ids, mask = tokenizer(output_values, return_mask=True, add_special_tokens=True)
            batch[key] = ids 
            batch['text_attention_mask'] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids 
            batch['text_attention_mask_negative'] = mask
        else:
            values = [elem[key] for elem in features]
            batch[key] = torch.from_numpy(np.stack(values))
    return batch



class DefaultDataCollator(DataCollatorMixin):
    """
    DreamZero默认数据整理器：将DreamTransform输出的单样本列表整理为训练batch。

    【作用与原理】
    该类是DataLoader的collator，负责：
    1. 调用collate()函数将样本列表整理为batch tensor
    2. 管理tokenizer（HuggingfaceTokenizer，基于umt5-xxl）
    3. 处理语言tokenization：将字符串指令转为input_ids和attention_mask
    4. 处理多视角图像拼接（根据num_views和embodiment_tag）

    【数据流位置】
    上游：DreamTransform.apply()输出的单样本字典列表（长度=batch_size）
    当前：DefaultDataCollator
    下游：VLA模型（WANPolicyHead等）

    【关键操作】
    - 语言处理：通过HuggingfaceTokenizer进行tokenize，padding到max_length
    - 张量堆叠：numpy数组→torch.stack→batch tensor
    - 本体特定处理：不同embodiment（DROID/AgiBot/GR1等）有不同的语言前缀模板

    【配置参数】
    - tokenizer_path: "google/umt5-xxl"（WAN2.1使用的文本编码器）
    - max_length: 512（最大token长度）
    - num_views: 视角数量（DROID=3，AgiBot=4等）
    """

    def __init__(self, tokenizer_path: str="google/umt5-xxl", max_length: int=512, num_views: int=1, embodiment_tag_mapping=None):
        """
        初始化DataCollator。

        Args:
            tokenizer_path (str): HuggingFace tokenizer路径（默认umt5-xxl）。
            max_length (int): 语言序列最大长度（默认512）。
            num_views (int): 相机视角数量（决定图像处理逻辑）。
            embodiment_tag_mapping (dict): 本体标签到整数的映射表。
        """
        super().__init__()
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=max_length, clean='whitespace')
        self.num_views = num_views
        self.embodiment_tag_mapping = embodiment_tag_mapping

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        整理样本列表为batch。

        【输入】
        - features (List[dict]): DreamTransform输出的样本列表，长度=B。
          每个元素为包含"images", "text", "state", "action"等的字典。

        【输出】
        - dict: Batch字典，所有值都是torch.Tensor（除text已tokenize）。
          例如：{
            "images": (B, T, H, W, 3) uint8,
            "text": (B, L_text) int64,
            "text_attention_mask": (B, L_text) int64,
            "state": (B, T_s, max_state_dim) float,
            "action": (B, T_a, max_action_dim) float,
            ...
          }

        【调用关系】
        - 被: DataLoader（通过PyTorch DataLoader接口）
        - 调用: collate()函数（本文件内定义）

        Returns:
            Dict[str, Any]: Batch化的数据字典。
        """
        return collate(features, self.tokenizer, self.num_views, self.embodiment_tag_mapping)


class DreamTransform(InvertibleModalityTransform):
    """
    DreamZero模型专用变换：将LeRobot格式的多模态数据转换为模型输入格式。

    【作用与原理】
    该类是数据管道的最后一步变换，负责：
    1. **Video拼图处理**：将多相机图像拼接为2×2网格（DROID：腕部顶行全宽，双外参底行；其他：左上/左下/右上/右下）
    2. **语言处理**：从多语言键中选择指令，添加本体特定前缀，支持语言dropout
    3. **State/Action Padding**：将state/action填充到统一维度（max_state_dim/max_action_dim）
    4. **特殊模式支持**：LAPA/DREAM/COTRAIN等联合训练模式的数据路由
    5. **生成mask**：state_mask/action_mask标记有效维度

    【数据流位置】
    上游：ConcatTransform（已拼接多view、多键的state/action）
    当前：DreamTransform
    下游：DefaultDataCollator → VLA模型

    【核心处理】
    - DROID拼图：3个view → 1路2×2网格（腕部顶行横向拉伸，左右外参底行并排）
    - 语言前缀：根据embodiment_tag添加描述多视角布局的固定前缀
    - Padding：不足max_dim则补0，生成对应mask
    - 特殊标签：is_lapa_instance/is_dream_instance控制后续loss路由

    【配置参数】
    - max_state_dim/max_action_dim: 统一维度（如64）
    - state_horizon/action_horizon: 时间步数（如1/24）
    - default_instruction: 语言dropout时的默认指令
    - language_dropout_prob: 语言dropout概率（训练时随机替换为默认指令）
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="此transform不使用apply_to，保留用于兼容性。"
    )
    training: bool = Field(
        default=True, description="是否训练模式（影响语言dropout和拼图随机性）。"
    )

    formalize_language: bool = Field(default=False, description="是否规范化语言格式。")

    embodiment_tag_mapping: dict[str, int] = Field(
        default_factory=dict,
        description="本体标签到projector索引的映射（如{'oxe_droid': 26, 'agibot': 17}）。",
    )

    language_dropout_prob: float = Field(
        default=0.0,
        description="语言dropout概率（训练时随机概率替换为默认指令）。",
    )
    always_use_default_instruction: bool = Field(
        default=False,
        description="是否总是使用默认指令（用于ablation研究语言作用）。",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[str] = PrivateAttr(default=None)  # 当前使用的语言键
    _language_keys: Optional[list[str]] = PrivateAttr(default=None)  # 所有可用语言键

    # XEmbDiT arguments
    default_instruction: str  # 默认语言指令
    max_state_dim: int  # state最大维度（如64）
    max_action_dim: int  # action最大维度（如64）
    max_length: int = 512  # 语言序列最大长度
    embodiment_tag: EmbodimentTag | None = None  # 当前本体标签
    state_horizon: int  # state时间步数（如1）
    action_horizon: int  # action时间步数（如24）
    num_views: int = 3  # 相机视角数

    # Tokenizer（用于长度检查，实际tokenize在Collator中进行）
    tokenizer_path: str = Field(
        default="google/umt5-xxl",
        description="Tokenizer路径（与Collator一致）。"
    )
    _tokenizer: Optional[HuggingfaceTokenizer] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        """
        初始化DreamTransform。

        Args:
            **kwargs: 配置参数，包括default_instruction, max_state_dim, max_action_dim等。
        """
        super().__init__(**kwargs)
        # Initialize the tokenizer
        self._tokenizer = HuggingfaceTokenizer(
            name=self.tokenizer_path, 
            seq_len=self.max_length, 
            clean='whitespace'
        )
    
    @property
    def tokenizer(self):
        """获取tokenizer实例。"""
        return self._tokenizer

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """
        从数据集metadata设置本体标签。

        Args:
            dataset_metadata (DatasetMetadata): 包含embodiment_tag的数据集元数据。
        """
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
                if "annotation" in key:
                    modality = "language"
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            self._language_keys = language_keys  # Store all keys for random selection
            if len(language_keys) == 1:
                self._language_key = language_keys[0]
            else:
                self._language_key = None  # Will be selected randomly in _prepare_language
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        images = batch["images"]  # [V, T, C, H, W]

        np_images = rearrange(images, "v t c h w -> (t v) h w c")
        if "language" in batch:
            lang = batch["language"]
            if isinstance(lang, list) or isinstance(lang, np.ndarray):
                lang = lang[0]

        inputs = {}
        inputs["images"] = np_images
        inputs["text"] = lang

        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            v, t, c, h, w = images.shape
            
            # For DROID embodiment: 2x2 grid where the wrist view spans the full top row,
            # and the two exterior views occupy the bottom row.
            #
            # View indices (expected):
            # - View 0: left exterior
            # - View 1: right exterior
            # - View 2: wrist
            #
            # Layout:
            #   [wrist, wrist]     (wrist duplicated to have 2x width)
            #   [left_ext | right_ext]
            #
            # Training-time augmentation:
            # - Randomly drop (black out) either left_ext or right_ext.
            if self.embodiment_tag == EmbodimentTag.OXE_DROID and v >= 3:
                left_exterior = images[0]   # (t, c, h, w)
                right_exterior = images[1]  # (t, c, h, w)
                wrist_image = images[2]     # (t, c, h, w)

                concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

                # Top row: a SINGLE wrist view, resized to be 2x wider (same height).
                # We use nearest-neighbor upscaling by repeating pixels along width.
                wrist_wide = np.repeat(wrist_image, 2, axis=-1)  # (t, c, h, 2w)
                concat_images[0, :, :, :h, :] = wrist_wide

                # # Bottom row: left/right exteriors.
                # drop_exterior_idx = None
                # if self.training:
                #     # Always drop exactly one exterior view during training.
                #     drop_exterior_idx = random.choice([0, 1])  # 0=left, 1=right

                # if drop_exterior_idx != 0:
                concat_images[0, :, :, h:, :w] = left_exterior
                # if drop_exterior_idx != 1:
                concat_images[0, :, :, h:, w:] = right_exterior

                return concat_images
            
            # For other embodiments: use 2x2 grid layout
            # Layout: [head, right]
            #         [left, black]
            
            # Create output tensor with doubled height and width
            concat_images = np.zeros((1, t, c, 2*h, 2*w), dtype=images.dtype)
            
            # Place images in the 2x2 grid
            # Left upper: head image (view 0)
            if v > 0:
                concat_images[0, :, :, :h, :w] = images[0]

            # Left bottom: left image (view 1)
            if v > 1:
                concat_images[0, :, :, h:, :w] = images[1]

            # Right top: right image (view 2)
            if v > 2:
                concat_images[0, :, :, :h, w:] = images[2]

            # Right bottom: black pixels (already zeros from initialization)

            return concat_images
        
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        # Determine which language key to use
        selected_key = self._language_key
        
        # For DROID embodiment during training, randomly select from available language keys
        if (self._language_keys is not None and 
            len(self._language_keys) > 1 and 
            self.training and 
            self.embodiment_tag == EmbodimentTag.OXE_DROID):
            selected_key = random.choice(self._language_keys)
        elif self._language_keys is not None and len(self._language_keys) > 0 and selected_key is None:
            selected_key = self._language_keys[0]
        
        if selected_key is not None:
            raw_language = data[selected_key]
            if isinstance(raw_language, np.ndarray):
                raw_language = raw_language.item() if raw_language.size == 1 else raw_language[0]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            # WARNING: this is not compatible with LAPA and DREAM
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        if "<LAPA>" in raw_language:
            raw_language = raw_language.replace("<LAPA>", "")
            is_lapa_instance = True
        else:
            is_lapa_instance = False

        if "<DREAM>" in raw_language:
            raw_language = raw_language.replace("<DREAM>", "")
            is_dream_instance = True
        else:
            is_dream_instance = False
        
        if "<COTRAIN>" in raw_language:
            raw_language = raw_language.replace("<COTRAIN>", "")
            is_cotrain_instance = True
        else:
            is_cotrain_instance = False

        if self.always_use_default_instruction:
            raw_language = self.default_instruction
        
        # print("raw_language", raw_language)

        # Formalize language
        if self.formalize_language:
            formalized_language = formalize_language(raw_language)
            return formalized_language, is_lapa_instance, is_dream_instance, is_cotrain_instance
        else:
            return raw_language, is_lapa_instance, is_dream_instance, is_cotrain_instance

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """

        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] % self.state_horizon == 0, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] % self.action_horizon == 0, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        """
        处理单样本，将LeRobot格式转换为DreamZero模型输入格式。

        【输入】
        - data (dict): ConcatTransform输出的字典，包含：
          * "video": (T, V, H, W, C) uint8，多view已拼接
          * "state": (T, D_state) 已归一化
          * "action": (T_a, D_action) 已归一化
          * "annotation.language.xxx": str 语言指令（可能有多个键）
          * 其他：timestamp, embodiment_id等

        【处理流程】
        1. Video拼图：调用_prepare_video()进行2×2网格拼接
        2. 语言处理：调用_prepare_language()选择语言键，解析<LAPA>/<DREAM>/<COTRAIN>标签
        3. VLM预处理：调用_apply_vlm_processing()调整维度顺序
        4. State处理：调用_prepare_state()填充到max_state_dim，生成mask
        5. Action处理：调用_prepare_action()填充到max_action_dim，生成mask
        6. 特殊模式：根据标签设置is_lapa_instance/is_dream_instance等路由标记
        7. 组装输出字典

        【输出】
        - dict: 模型就绪的单样本字典，包含：
          * "images": (1, T, C, 2H, 2W) uint8（拼图后）→ VLM处理后可展平为(T, 2H, 2W, C)
          * "text": str 语言指令（待Collator tokenize）
          * "state": (T_s, max_state_dim) float，已填充
          * "state_mask": (T_s, max_state_dim) bool，有效维为True
          * "action": (T_a, max_action_dim) float，已填充
          * "action_mask": (T_a, max_action_dim) bool，有效维为True
          * "has_real_action": bool，是否计算action loss
          * "embodiment_id": int，本体标签
          * "is_lapa_instance"/"is_dream_instance"/"is_cotrain_instance": bool 路由标记
          * "text_negative": str 负向提示（用于CFG）

        【Shape示例】（DROID，num_frames=25, action_horizon=24）
        - 输入video: (25, 3, 256, 320, 3)
        - 输出images: (1, 25, 3, 512, 640)（拼图后2H=512, 2W=640）
        - 输出state: (1, 64)（原8维填充到64）
        - 输出action: (24, 64)（原8维填充到64）

        【调用关系】
        - 被: apply()（batch拆分后）
        - 调用: _prepare_video(), _prepare_language(), _prepare_state(), _prepare_action(),
                _apply_vlm_processing()

        Returns:
            dict: 处理后的单样本字典。
        """
        transformed_data = {}

        # 1) Video拼图：多view → 2×2网格
        images = self._prepare_video(data)
        images = images.astype(np.uint8)

        # 2) 语言处理：选择语言键，解析特殊标签
        language, is_lapa_instance, is_dream_instance, is_cotrain_instance = self._prepare_language(data)
        batch_data = {"images": images, "language": language}

        # 3) VLM预处理：调整维度顺序
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 4) State处理：填充+mask
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 5) Action处理：填充+mask，检测特殊模式
            is_detection_instance = self.embodiment_tag == EmbodimentTag.GR1_UNIFIED_SEGMENTATION
            if is_detection_instance:
                transformed_data["segmentation_target"] = data["action"][0, -3:-1]
                transformed_data["segmentation_target_mask"] = data["action"][0, -1:]
                transformed_data["has_real_action"] = np.zeros((), dtype=bool)
            else:
                transformed_data["segmentation_target"] = np.zeros((2,))
                transformed_data["segmentation_target_mask"] = np.zeros((1,))
                transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

            # default for lapa instance
            transformed_data["lapa_action"] = np.zeros_like(transformed_data["action"])
            transformed_data["lapa_action_mask"] = np.zeros_like(transformed_data["action_mask"])
        # else:
        transformed_data["text_negative"] = "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, three legs, many people in the background, walking backwards."

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.embodiment_tag == EmbodimentTag.MECKA_HANDS: 
            is_cotrain_instance = True
        else:
            is_cotrain_instance = False

        transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
        # print("dreamzero_fixed", is_cotrain_instance)
        if is_cotrain_instance:
            transformed_data["is_cotrain_instance"] = np.ones((), dtype=bool)
        else:
            transformed_data["is_cotrain_instance"] = np.zeros((), dtype=bool)

        if is_dream_instance:
            assert "dream_actions" in data
            transformed_data["embodiment_id"] = self.embodiment_tag_mapping["dream"]
            transformed_data["state"] = np.zeros_like(transformed_data["state"])
            actions_shape = transformed_data["action"].shape

            # Treat the "dream" IDM action as a real action so that flow matching loss will be applied.
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)

            dream_actions = data["dream_actions"]
            assert (
                dream_actions.size == actions_shape[0] * actions_shape[1]
            ), f"dream_actions size {dream_actions.size} does not match action shape {actions_shape}"
            transformed_data["action"] = dream_actions.reshape(actions_shape)

        if is_lapa_instance:
            assert "lapa_action" in data
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
            transformed_data["embodiment_id"] = self.embodiment_tag_mapping["lapa"]
            transformed_data["state"] = np.zeros_like(transformed_data["state"])
            actions_shape = transformed_data["action"].shape
            lapa_actions = data["lapa_action"]
            # Ensure total elements match before reshaping
            assert (
                lapa_actions.size == actions_shape[0] * actions_shape[1]
            ), f"Cannot reshape lapa_actions of size {lapa_actions.size} to {actions_shape}"
            # Reshape the lapa_actions to match the expected shape
            reshaped_lapa_actions = lapa_actions.reshape(actions_shape)
            # lapa_action should be between -1 and 1
            assert np.all(reshaped_lapa_actions >= -1) and np.all(
                reshaped_lapa_actions <= 1
            ), "LAPA action values should be between -1 and 1"
            transformed_data["action"] = reshaped_lapa_actions
            transformed_data["action_mask"] = np.ones(actions_shape, dtype=bool)

        if self.training:
            action_and_mask_keys = ["action", "action_mask", "lapa_action", "lapa_action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        # delete lapa_action and lapa_action_mask
        data.pop("lapa_action", None)
        # data.pop("lapa_action_mask", None)
        data.pop("dream_actions", None)
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.tokenizer, self.num_views, self.embodiment_tag_mapping)

    def apply(self, data: dict) -> dict:
        if not self.training and data["video"].ndim == 5:
            data["video"] = data["video"][None, ...]
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

