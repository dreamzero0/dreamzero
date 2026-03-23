"""
FlowMatchScheduler: 流匹配（Flow Matching）扩散模型的核心调度器。

【作用与原理】
本类实现了流匹配（Flow Matching）框架下的扩散调度器，用于管理加噪（训练）和去噪（推理）过程。
流匹配将扩散过程建模为从噪声分布到数据分布的直线路径（向量场），通过预测速度向量（noise - sample）
来实现高效的去噪生成。

【核心概念】
1. Sigma (σ): 噪声级别，范围 [0, ~5.0]，σ=0 表示干净数据，σ=1 表示纯噪声
2. Timestep (t): 离散化的 sigma 索引，范围 [0, num_train_timesteps=1000]
3. Shift 变换: 将原始的 [0,1] 均匀分布压缩为偏向低噪声区域的分布，提高训练稳定性
4. 流匹配目标: v = ε - x₀，即噪声与干净样本的差（预测该向量场即为训练目标）

【数据流位置】
上游: WANPolicyHead.forward() 训练时调用
当前: FlowMatchScheduler（采样 timestep、加噪、计算目标、加权 loss）
下游: 返回 noisy_sample 和 training_target 给 DiT 模型预测

【与 DDPM/DDIM 的区别】
- DDPM/DDIM: 预测噪声 ε 或 v-prediction，需要多步迭代
- Flow Matching: 预测速度向量场 v = ε - x₀，训练更稳定，支持少步推理（4-16步）

参考论文: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
"""

import torch


class FlowMatchScheduler():
    """
    流匹配调度器：管理噪声级别的采样、加噪、去噪和 loss 加权。

    【典型配置】（DreamZero DROID 训练）
    - num_train_timesteps=1000: 训练时 timestep 离散 bucket 数
    - num_inference_steps=16: 推理时去噪步数
    - shift=5.0: 将 sigma 分布压缩，更多采样低噪声区域
    - sigma_max=1.0, sigma_min=0.003/1.002: 噪声级别范围
    - extra_one_step=True: 推理时额外一步用于稳定生成

    【关键属性】
    - self.sigmas: Tensor (num_inference_steps,), 每个推理步骤的 sigma 值
    - self.timesteps: Tensor (num_inference_steps,), sigma * 1000 映射后的 timestep 值
    - self.linear_timesteps_weights: Tensor (num_inference_steps,), 训练时各 timestep 的 loss 权重
    """

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        """
        初始化 FlowMatchScheduler。

        Args:
            num_inference_steps (int): 推理时的去噪步数（如 16 或 4）。
            num_train_timesteps (int): 训练时 timestep 的离散 bucket 数（默认 1000）。
            shift (float): Shift 变换参数（默认 3.0，DreamZero 用 5.0），控制 sigma 分布形状。
                shift 越大，低噪声区域（接近 0）越密集。
            sigma_max (float): 最大噪声级别（默认 1.0，纯噪声）。
            sigma_min (float): 最小噪声级别（默认 ~0.003，接近干净数据）。
            inverse_timesteps (bool): 是否反转 timestep 顺序（从干净到噪声）。
            extra_one_step (bool): 推理时是否额外添加一步用于稳定（默认 True）。
            reverse_sigmas (bool): 是否反转 sigma 值（1-sigma）。

        【初始化流程】
        1. 保存所有配置参数
        2. 调用 set_timesteps() 创建 sigma 和 timestep 调度表
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        """
        创建 sigma 和 timestep 调度表，支持训练和推理两种模式。

        【数学原理】
        原始 sigma 调度: sigma ∈ [sigma_start, sigma_min] 线性均匀分布
        Shift 变换: sigma_shifted = shift * sigma / (1 + (shift-1) * sigma)
            该变换将 [0,1] 映射为 [0,1] 但分布更集中在低 sigma 区域
        Timestep 映射: timestep = sigma_shifted * num_train_timesteps
            将连续的 sigma 映射到离散的 [0, 1000] 范围

        【训练模式特殊处理】
        当 training=True 时，额外计算每个 timestep 的权重（高斯加权）：
        - 中间 timestep（~500）权重最高
        - 两端（0 和 1000）权重较低
        这使得模型更关注"中等难度"的 timestep，提高训练效率。

        【Shape 说明】
        - self.sigmas: (num_inference_steps,) 的 1D Tensor
        - self.timesteps: (num_inference_steps,) 的 1D Tensor
        - self.linear_timesteps_weights: (num_inference_steps,) 的 1D Tensor（仅训练时）

        【具体 Shape 示例】（num_inference_steps=16, num_train_timesteps=1000）
        - self.sigmas: torch.Size([16])，值如 [0.999, 0.937, ..., 0.003]
        - self.timesteps: torch.Size([16])，值如 [999, 937, ..., 3]（sigma * 1000）
        - self.linear_timesteps_weights: torch.Size([16])，值如 [0.5, 0.8, ..., 0.3]

        Args:
            num_inference_steps (int): 推理步数。
            denoising_strength (float): 去噪强度（0-1，1.0 表示完整去噪）。
            training (bool): 是否为训练模式（决定是否需要计算 timestep 权重）。
            shift (float | None): 覆盖默认的 shift 参数。

        【调用时机】
        - __init__() 时自动调用（推理模式）
        - WANPolicyHead.__init__() 中显式调用 training=True 模式
        """
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """
        推理时执行单步去噪（Euler 积分）。

        【数学原理】
        流匹配 ODE: dx/dt = v(x, t)
        离散化: x_{t-1} = x_t + v(x, t) * (sigma_{t-1} - sigma_t)
        其中 model_output = v(x, t)（预测的向量场）

        【输入 Shape】
        - model_output: (B, C, H, W) 或 (B, T, D)，模型预测的速度向量场 v
        - timestep: scalar 或 Tensor，当前 timestep（用于查找 sigma）
        - sample: (B, C, H, W) 或 (B, T, D)，当前带噪样本 x_t

        【输出 Shape】
        - prev_sample: 与 sample 同 Shape，去噪后的样本 x_{t-1}

        【调用时机】
        仅在推理时使用，每步去噪调用一次（共 num_inference_steps 次）。

        Args:
            model_output (Tensor): 模型预测的向量场 v(x, t)。
            timestep (int | Tensor): 当前 timestep。
            sample (Tensor): 当前带噪样本 x_t。
            to_final (bool): 是否为最后一步（使用 sigma=0 或 1）。
            **kwargs: 其他参数（兼容 HuggingFace 接口）。

        Returns:
            Tensor: 去噪后的样本 x_{t-1}。
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        """
        从稳定化样本反推模型输出（用于某些特殊推理场景）。

        【数学原理】
        x_t = x_0 + sigma * v
        => v = (x_t - x_0) / sigma

        【输入 Shape】
        - timestep: scalar 或 Tensor，目标 timestep
        - sample: (B, ...)，带噪样本 x_t
        - sample_stablized: (B, ...)，稳定化后的干净样本 x_0

        【输出 Shape】
        - model_output: 与 sample 同 Shape，反推的向量场 v

        Args:
            timestep (int | Tensor): 目标 timestep。
            sample (Tensor): 带噪样本。
            sample_stablized (Tensor): 稳定化后的干净样本。

        Returns:
            Tensor: 反推的模型输出（向量场）。
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    # def add_noise(self, original_samples, noise, timestep):
    #     """
    #     旧版加噪函数（已弃用），仅支持标量 timestep。
    #     """
    #     if isinstance(timestep, torch.Tensor):
    #         timestep = timestep.cpu()
    #     timestep_id = torch.argmin((self.timesteps - timestep).abs())
    #     sigma = self.sigmas[timestep_id]
    #     sample = (1 - sigma) * original_samples + sigma * noise
    #     return sample
    
    def add_noise(self, original_samples, noise, timestep):
        """
        训练时向干净样本添加噪声（流匹配前向过程）。

        【数学原理】
        流匹配加噪公式（直线路径）:
            x_t = (1 - σ) * x_0 + σ * ε
        其中:
            x_0 = original_samples (干净样本)
            ε = noise (标准高斯噪声)
            σ = timestep 对应的 sigma 值
            x_t = 加噪后的样本

        该公式实现了从干净数据 (σ=0) 到纯噪声 (σ=1) 的线性插值。

        【输入 Shape】
        - original_samples: (B, C, H, W) 或 (B, T, D)，干净样本
            例: (4, 16, 9, 22, 40) 视频潜变量 或 (4, 24, 64) 动作序列
        - noise: 与 original_samples 同 Shape，标准高斯噪声 N(0,1)
        - timestep: (B,) 或 (B, T) 等，每个样本/位置的 timestep ID
            注意: timestep 可以是每个位置独立的（支持每帧/每步不同噪声级别）

        【输出 Shape】
        - sample: 与 original_samples 完全同 Shape，加噪后的样本 x_t

        【具体 Shape 示例】（视频 + 动作联合加噪）
        视频:
        - original_samples: (4, 16, 9, 22, 40) = (B, C_lat, T_lat, H_lat, W_lat)
        - noise: (4, 16, 9, 22, 40)
        - timestep: (4, 9) 或 (36,)  # 可以是每帧独立或 flatten 后
        - 输出 sample: (4, 16, 9, 22, 40)

        动作:
        - original_samples: (4, 24, 64) = (B, T_a, D_a)
        - noise: (4, 24, 64)
        - timestep: (4, 24) 或 (96,)
        - 输出 sample: (4, 24, 64)

        【调用关系】
        - 被: WANPolicyHead.forward() 训练时调用
        - 调用后: noisy_sample 输入 CausalWanModel 预测噪声

        【与 timestep 的关系】
        timestep 参数可以是:
        1. 标量 int: 所有样本使用相同噪声级别
        2. Tensor (B,): batch 内每个样本不同噪声级别
        3. Tensor (B, T): 每个时间位置不同噪声级别（DreamZero 视频/动作常用）

        本函数通过 broadcast 机制自动处理各种 timestep shape。

        Args:
            original_samples (Tensor): 干净样本 x_0，任意 Shape。
            noise (Tensor): 高斯噪声 ε，与 original_samples 同 Shape。
            timestep (int | Tensor): timestep 值（会通过查找表映射为 sigma）。
                若为 Tensor，可以是 (B,) 或 (B, T) 等，支持每个位置独立噪声级别。

        Returns:
            Tensor: 加噪后的样本 x_t，与 original_samples 同 Shape。
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        # 向量化查找: timestep (N,) -> timestep_id (N,), 再取 sigma (N,)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim = 0) 
        sigma = self.sigmas[timestep_id].to(device=original_samples.device, dtype=original_samples.dtype)
        # 将 sigma  broadcast 到与 original_samples 相同的维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        """
        计算流匹配的训练目标（向量场 v = ε - x₀）。

        【数学原理】
        流匹配的核心思想：模型预测速度向量场 v(x, t)，使得沿着该场流动可以从噪声到达数据。
        理论证明最优的 v* = E[x₀ - ε | x_t] = x₀ - x_t（对于直线路径）
        由于 x_t = (1-σ)x₀ + σε，可得 v* = ε - x₀

        【输入 Shape】
        - sample: (B, ...)，干净样本 x₀（注意：不是加噪后的 x_t！）
        - noise: (B, ...)，高斯噪声 ε
        - timestep: 在此简单实现中未使用（因为目标与 t 无关），保留参数为了接口兼容

        【输出 Shape】
        - target: (B, ...)，与 sample/noise 同 Shape，流匹配目标 v = ε - x₀

        【具体 Shape 示例】
        视频:
        - sample: (4, 16, 9, 22, 40)，干净视频潜变量
        - noise: (4, 16, 9, 22, 40)
        - target: (4, 16, 9, 22, 40)，视频流匹配目标

        动作:
        - sample: (4, 24, 64)，干净动作
        - noise: (4, 24, 64)
        - target: (4, 24, 64)，动作流匹配目标

        【调用关系】
        - 被: WANPolicyHead.forward() 训练时调用
        - 输出 target 与模型预测 model_output 计算 MSE loss

        Args:
            sample (Tensor): 干净样本 x₀。
            noise (Tensor): 高斯噪声 ε。
            timestep (Tensor): 当前 timestep（本实现中未使用，保留接口兼容）。

        Returns:
            Tensor: 流匹配目标 v = ε - x₀。
        """
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        """
        获取训练时各 timestep 的 loss 权重（用于加权 MSE）。

        【原理】
        不同 timestep 的训练难度不同：
        - 极低 sigma（接近干净数据）：模型容易预测，loss 权重应较低
        - 极高 sigma（接近纯噪声）：模型容易预测，loss 权重应较低
        - 中等 sigma（~0.5）：模型最难学习，loss 权重应较高

        因此使用高斯分布加权：权重 ∝ exp(-2*((t - T/2)/T)²)
        使得中间 timestep 获得更高权重。

        【输入 Shape】
        - timestep: (N,) 的 1D Tensor，每个样本/位置的 timestep 值
            例: (36,) 表示 4 batch * 9 帧 = 36 个 timestep

        【输出 Shape】
        - weights: (N,) 的 1D Tensor，每个样本的 loss 权重
            值范围通常在 [0.3, 1.0] 之间，中间 timestep ~1.0，两端 ~0.3

        【具体 Shape 示例】
        视频:
        - 输入 timestep: (36,)  # flatten 后的 (B*T,)
        - 输出 weights: (36,)，每个位置的 loss 权重
        - 通常再 unflatten 为 (4, 9) 与视频 shape 对齐

        动作:
        - 输入 timestep: (96,)  # flatten 后的 (B*T_a,) = 4*24
        - 输出 weights: (96,)
        - 通常再 unflatten 为 (4, 24)

        【调用关系】
        - 被: WANPolicyHead.forward() 训练时调用
        - 用法: weighted_loss = MSE_loss * training_weight(timestep)

        Args:
            timestep (Tensor): timestep 值，Shape 为 (N,)。

        Returns:
            Tensor: 各位置的 loss 权重，Shape 为 (N,)。
        """
        # 向量化查找 timestep_id
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0).to(self.timesteps.device)).abs(), dim = 0) 
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
