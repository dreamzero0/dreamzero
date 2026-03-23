"""
IdentityBackbone: 恒等Backbone，用于独立训练ActionHead而不依赖视觉-语言编码器。

【设计原理】
DreamZero架构中，WANPolicyHead（基于Wan2.1-I2V-14B）已经包含了完整的视觉编码（VAE+CLIP）和
语言编码（T5）能力。因此，当使用WANPolicyHead时，不需要额外的backbone提取特征。

IdentityBackbone作为占位符存在：
1. forward()返回空的backbone_features（shape: B,1,0），满足VLA接口要求
2. prepare_input()仅用于推断batch size
3. 实际的多模态编码由WANPolicyHead内部处理

【使用场景】
- DreamZero训练：使用WANPolicyHead + IdentityBackbone组合
- 未来扩展：若需添加独立VLM backbone（如Qwen2-VL），可替换此类
"""

import torch
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.backbone.base_backbone import Backbone


class IdentityBackbone(Backbone):
    """
    恒等Backbone：用于ActionHead独立训练的占位符实现。

    【作用与原理】
    该类是Backbone基类的极简实现，满足VLA模型接口的同时不执行实际计算：
    1. forward()返回空的backbone_features张量（B,1,0）
    2. prepare_input()仅从batch中提取一个张量用于推断batch size
    3. 不定义可训练参数（set_trainable_parameters为空操作）

    【数据流位置】
    上游：VLA.prepare_input()传递的backbone_inputs
    当前：IdentityBackbone
    下游：VLA.forward()将backbone_outputs传递给action_head

    【与其他Backbone的区别】
    - IdentityBackbone: 无实际计算，特征维度为0
    - 真实Backbone（如Qwen2-VL）: 提取图像+文本特征，输出高维张量

    【设计动机】
    WANPolicyHead内部已集成VAE（视觉）+ CLIP（图像特征）+ T5（语言）编码器，
    因此不需要外部backbone。IdentityBackbone使VLA接口保持一致性。
    """

    def set_trainable_parameters(self, **kwargs):
        """
        设置可训练参数（IdentityBackbone无参数，为空操作）。

        Args:
            **kwargs: 忽略所有参数。
        """
        return

    def forward(self, backbone_input: BatchFeature) -> BatchFeature:
        """
        前向传播：返回空的backbone_features占位符。

        【输入】
        - backbone_input (BatchFeature): prepare_input()的输出，包含任意一个张量
          （如{"action": (B, T_a, D_a)}），仅用于推断batch size。

        【处理】
        1. 取backbone_input的第一个值（任意张量）
        2. 提取其shape[0]作为batch size B
        3. 创建空的backbone_features: torch.empty(B, 1, 0)

        【输出】
        - BatchFeature: {"backbone_features": (B, 1, 0) float32 tensor}
          shape含义: (batch, seq_len=1, hidden_dim=0)

        【Shape示例】
        - 输入batch size=8 → 输出backbone_features: (8, 1, 0)

        【调用关系】
        - 被: VLA.forward() / VLA.get_action() 等
        - 调用: 无（仅创建空张量）

        Args:
            backbone_input (BatchFeature): Backbone输入（任意，仅取batch size）。

        Returns:
            BatchFeature: 包含空的backbone_features。
        """
        backbone_input_first_value = next(iter(backbone_input.values()))
        B = backbone_input_first_value.shape[0]

        # 创建空的特征张量: (B, 1, 0)
        backbone_features = torch.empty(
            B, 1, 0, dtype=torch.float32, device=backbone_input_first_value.device
        )
        output_dict = {
            "backbone_features": backbone_features,
        }

        return BatchFeature(data=output_dict)

    def prepare_input(self, batch: dict) -> BatchFeature:
        """
        输入准备：从batch中提取一个张量用于推断batch size。

        【输入】
        - batch (dict): VLA接收的完整batch字典，包含images/text/state/action等。

        【处理优先级】
        1. 若存在"action"，返回{"action": batch["action"]}（训练时常见）
        2. 否则若存在"state"，返回{"state": batch["state"]}（推理时可能）
        3. 否则若存在"video"，转换numpy→tensor返回（处理numpy数组）
        4. 否则返回整个batch

        【输出】
        - BatchFeature: 包含至少一个张量的字典，仅用于forward()推断batch size。

        【调用关系】
        - 被: VLA.prepare_input() → 生成backbone_inputs
        - 调用: 无

        Args:
            batch (dict): 完整batch字典。

        Returns:
            BatchFeature: 包含至少一个张量的BatchFeature（用于推断B）。
        """
        if "action" in batch:
            return BatchFeature(data={"action": batch["action"]})
        else:
            # 推理时使用state或video推断batch size
            if "state" in batch:
                return BatchFeature(data={"state": batch["state"]})
            elif "video" in batch:
                # video为numpy数组，需转为tensor以兼容BatchFeature.to()
                video = batch["video"]
                video_tensor = torch.from_numpy(video)
                return BatchFeature(data={"video": video_tensor})
            else:
                return BatchFeature(data=batch)
