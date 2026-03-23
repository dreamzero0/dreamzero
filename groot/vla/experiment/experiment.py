import logging
import os
import time
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from groot.vla.experiment.base import BaseExperiment, BaseTrainer
from groot.vla.utils.action_args_override_utils import apply_action_overrides

logger = logging.getLogger(__name__)


INITIAL_ACTIONS_FILENAME = "initial_actions.npz"


class ForceRestart(ValueError):
    pass


class VLATrainer(BaseTrainer):
    """
    DreamZero专用Trainer：扩展BaseTrainer，添加ActionHead状态同步和时间预算功能。

    【作用与原理】
    VLATrainer在BaseTrainer基础上添加DreamZero特有功能：
    1. **ActionHead状态同步**: 将Trainer.global_step同步到model.action_head.global_step
       （Flow Matching中某些调度器需要知道当前训练步数）
    2. **时间预算控制**: restart_max_seconds限制单作业运行时间，超时后抛出ForceRestart
       （用于Slurm环境自动重新排队）
    3. **性能基准**: benchmark_time模式支持测量每步耗时（用于性能分析）
    4. **微步跟踪**: micro_global_step记录总训练微步（含gradient accumulation）

    【数据流位置】
    上游：DataLoader生成的inputs（与BaseTrainer相同）
    当前：VLATrainer.training_step()
    下游：super().training_step() → BaseTrainer → 优化器更新

    【关键扩展】
    - training_step(): 添加action_head.global_step同步和时间预算检查
    - 继承compute_loss(): 使用BaseTrainer的loss计算和跟踪逻辑

    【使用场景】
    - 标准DreamZero训练（默认配置）
    - 限时训练（restart_max_seconds > 0）
    - 性能基准测试（benchmark_time=True）
    """

    def __init__(self, **kwargs):
        """
        初始化VLATrainer。

        Args:
            **kwargs: Trainer参数，外加：
                - benchmark_time: 是否启用性能基准模式
                - num_trials: 基准测试的试验次数
                - restart_max_seconds: 单作业最大运行时间（秒）

        【初始化流程】
        1. 弹出并保存DreamZero特有参数
        2. 获取分布式rank
        3. 调用父类BaseTrainer.__init__()
        """
        self.benchmark_time = kwargs.pop("benchmark_time", False)
        self.step_timer = None
        self.num_trials = kwargs.pop("num_trials", 10)
        self.curr_trial = 0
        self.all_times = []
        self.start_time = time.time()
        self.restart_max_seconds = kwargs.pop("restart_max_seconds", 0)
        import torch.distributed as dist

        self.rank = dist.get_rank()

        self.micro_global_step = 0

        super().__init__(**kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        """
        执行单步训练：同步ActionHead状态，检查时间预算，调用父类训练。

        【输入】
        - model: VLA模型实例
        - inputs (dict): DataLoader输出的batch字典
        - args/kwargs: 传递给父类training_step的额外参数

        【处理流程】
        1. micro_global_step递增（记录总微步）
        2. 同步action_head.global_step（Flow Matching调度器需要）
        3. 性能基准模式：每100步计时，达到num_trials后退出
        4. 时间预算检查：若超restart_max_seconds则抛出ForceRestart
        5. 调用super().training_step()执行实际训练

        【输出】
        - loss_dict: 包含loss的字典（父类training_step返回）

        【调用关系】
        - 被: HuggingFace Trainer训练循环
        - 调用: super().training_step() → BaseTrainer.training_step()

        Args:
            model: VLA模型。
            inputs (dict): Batch输入。
            *args: 传递给父类的位置参数。
            **kwargs: 传递给父类的关键字参数。

        Returns:
            dict: 包含loss的字典。
        """
        self.micro_global_step += 1

        # Sync global_step to action_head（Flow Matching scheduler needs this）
        if hasattr(self.model.action_head, "global_step"):
            self.model.action_head.global_step = self.state.global_step

        # Performance benchmark mode
        if self.benchmark_time:
            if self.state.global_step % 100 == 0:
                if self.step_timer is not None:
                    elapsed_time = time.time() - self.step_timer
                    self.all_times.append(elapsed_time)
                    self.curr_trial += 1
                self.step_timer = time.time()
            if self.curr_trial >= self.num_trials:
                exit(0)

        # Time budget check
        if self.state.global_step % self.state.save_steps == 1:
            if self.restart_max_seconds > 0:
                cur_time = time.time()
                if (cur_time - self.start_time) > self.restart_max_seconds:
                    raise ForceRestart(f"Exceeded time limit {self.restart_max_seconds} seconds")

        # Execute actual training step via parent class
        loss_dict = super().training_step(model, inputs, *args, **kwargs)
        return loss_dict


class VLATrainerInferenceBenchmark(VLATrainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        warmup_steps = 100
        measure_steps = 100

        model.eval()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                for i in range(warmup_steps):
                    action = model.module.get_action(inputs)
                    action.keys()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                for i in range(measure_steps):
                    action = model.module.get_action(inputs)
                    action.keys()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        time_per_step = elapsed_time / measure_steps
        exit()


class VLAExperiment(BaseExperiment):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Dump the initial actions
        if hasattr(self.train_dataset, "get_initial_actions"):
            # We only dump the initial actions for the real robot dataset
            # Sim dataset doesn't have this function
            """
            initial_actions: list[dict[str, dict[str, np.ndarray]]]
            0: (the first dataset)
                trajectory_name:
                action_key:
                    action: np.ndarray
            1: (the second dataset)
                ...
            """
            initial_actions = self.train_dataset.get_initial_actions()
            if len(initial_actions) > 0:
                initial_actions_path = self.exp_cfg_dir / INITIAL_ACTIONS_FILENAME
                np.savez(str(initial_actions_path), initial_actions)
                print("Successfully dumped initial actions")
            else:
                print("No initial actions to dump")


@hydra.main(config_path="../configs", config_name="conf", version_base=None)
def main(cfg):
    # Automatically update action dim and action horizon keys if specified in the config
    cfg = apply_action_overrides(cfg)

    experiment = VLAExperiment(cfg)
    experiment.train()


if __name__ == "__main__":
    main()
