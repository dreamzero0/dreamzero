import logging
import os
import time
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from transformers import TrainerCallback

from groot.vla.experiment.base import BaseExperiment, BaseTrainer
from groot.vla.utils.action_args_override_utils import apply_action_overrides

logger = logging.getLogger(__name__)


INITIAL_ACTIONS_FILENAME = "initial_actions.npz"


class MilestoneSaveCallback(TrainerCallback):
    """Request checkpoints only at configured optimizer-step milestones."""

    def __init__(self, milestones):
        self.milestones = frozenset(int(step) for step in milestones)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.milestones:
            control.should_save = True
        return control


class ForceRestart(ValueError):
    pass


class VLATrainer(BaseTrainer):
    def __init__(self, **kwargs):
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
        milestones = kwargs.pop("milestone_save_steps", None)

        super().__init__(**kwargs)
        if milestones:
            self.add_callback(MilestoneSaveCallback(milestones))
            if self.rank == 0:
                print(
                    "[CHECKPOINT MILESTONES] "
                    + ",".join(str(int(step)) for step in milestones)
                )

    def training_step(self, model, inputs, *args, **kwargs):
        self.micro_global_step += 1

        if hasattr(self.model.action_head, "global_step"):
            self.model.action_head.global_step = self.state.global_step

        if self.benchmark_time:
            if self.state.global_step % 100 == 0:
                if self.step_timer is not None:
                    elapsed_time = time.time() - self.step_timer
                    self.all_times.append(elapsed_time)
                    self.curr_trial += 1
                self.step_timer = time.time()
            if self.curr_trial >= self.num_trials:
                exit(0)
        if self.state.global_step % self.state.save_steps == 1:
            if self.restart_max_seconds > 0:
                cur_time = time.time()
                if (cur_time - self.start_time) > self.restart_max_seconds:
                    raise ForceRestart(f"Exceeded time limit {self.restart_max_seconds} seconds")
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
