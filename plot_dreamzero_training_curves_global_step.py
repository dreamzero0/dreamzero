#!/usr/bin/env python3
"""
Plot DreamZero training metrics with the real global-step axis.

Default run:
  /data/wangk/checkpoints/dreamzero_g2_joint_lora_full

Data-source priority:
  1. loss_log.jsonl
  2. trainer_state.json
  3. TensorBoard events
  4. train.log

If a metric log contains N records without an explicit step while the completed
run has global_step=5000, the script maps the records over the real training
range instead of incorrectly using 0..N as the x-axis.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path(
    "/data/wangk/checkpoints/dreamzero_g2_joint_lora_full"
)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_metric_name(name: str) -> str:
    aliases = {
        "train/loss": "loss",
        "train_loss": "loss",
        "training_loss": "loss",
        "train/action_loss": "action_loss_avg",
        "action_loss": "action_loss_avg",
        "action/loss": "action_loss_avg",
        "lr": "learning_rate",
    }
    name = str(name).strip()
    return aliases.get(name, name)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_run_metadata(run_dir: Path) -> dict[str, int | float | None]:
    state_files = [run_dir / "trainer_state.json"]
    state_files.extend(sorted(run_dir.glob("checkpoint-*/trainer_state.json")))

    best: dict[str, Any] = {}
    best_step = -1
    for path in state_files:
        if not path.exists():
            continue
        data = load_json(path)
        step = int(data.get("global_step") or 0)
        if step >= best_step:
            best = data
            best_step = step

    global_step = int(best.get("global_step") or 0) or None
    max_steps = int(best.get("max_steps") or 0) or None

    logging_steps = best.get("logging_steps")
    try:
        logging_steps = int(logging_steps)
    except (TypeError, ValueError):
        logging_steps = None

    return {
        "global_step": global_step,
        "max_steps": max_steps,
        "logging_steps": logging_steps,
    }


def row_step(row: dict[str, Any]) -> int | None:
    for key in (
        "global_step",
        "step",
        "training_step",
        "train_step",
        "iteration",
        "iter",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


NON_METRIC_KEYS = {
    "global_step",
    "step",
    "training_step",
    "train_step",
    "iteration",
    "iter",
    "epoch",
    "timestamp",
    "time",
    "rank",
}


def infer_steps(
    rows: list[dict[str, Any]],
    global_step: int | None,
    logging_steps: int | None,
) -> list[int]:
    explicit = [row_step(row) for row in rows]
    if rows and all(step is not None for step in explicit):
        return [int(step) for step in explicit if step is not None]

    count = len(rows)
    if count == 0:
        return []

    # Prefer trainer logging_steps when it exactly or nearly spans the run.
    if logging_steps and logging_steps > 0:
        steps = [(index + 1) * logging_steps for index in range(count)]
        if global_step:
            steps = [min(step, global_step) for step in steps]
            if steps[-1] < global_step and global_step / count > logging_steps * 1.25:
                # The saved logging_steps does not explain the number of rows.
                steps = []
        if steps:
            return steps

    # Main fallback for this run:
    # 2500 scalar rows over a completed 5000-step run -> 2, 4, ..., 5000.
    if global_step and global_step > 0:
        scale = global_step / count
        steps = [
            max(1, min(global_step, int(round((index + 1) * scale))))
            for index in range(count)
        ]
        steps[-1] = global_step
        return steps

    return list(range(1, count + 1))


def read_loss_jsonl(
    path: Path,
    global_step: int | None,
    logging_steps: int | None,
) -> dict[str, list[tuple[int, float]]]:
    records: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return records

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)

    steps = infer_steps(rows, global_step, logging_steps)
    for step, row in zip(steps, rows):
        for key, value in row.items():
            if key in NON_METRIC_KEYS:
                continue
            number = finite_float(value)
            if number is not None:
                records[normalize_metric_name(key)].append((step, number))

    return records


def read_trainer_history(
    run_dir: Path,
) -> dict[str, list[tuple[int, float]]]:
    records: dict[str, list[tuple[int, float]]] = defaultdict(list)
    path = run_dir / "trainer_state.json"
    if not path.exists():
        return records

    data = load_json(path)
    for row in data.get("log_history", []):
        if not isinstance(row, dict):
            continue
        step = row_step(row)
        if step is None:
            continue
        for key, value in row.items():
            if key in NON_METRIC_KEYS:
                continue
            number = finite_float(value)
            if number is not None:
                records[normalize_metric_name(key)].append((step, number))
    return records


def read_tensorboard(
    run_dir: Path,
) -> dict[str, list[tuple[int, float]]]:
    records: dict[str, list[tuple[int, float]]] = defaultdict(list)
    files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not files:
        return records

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as exc:
        print(f"[WARN] TensorBoard unavailable: {exc}")
        return records

    for path in files:
        try:
            accumulator = EventAccumulator(
                str(path),
                size_guidance={"scalars": 0},
            )
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                name = normalize_metric_name(tag)
                for event in accumulator.Scalars(tag):
                    number = finite_float(event.value)
                    if number is not None:
                        records[name].append((int(event.step), number))
        except Exception as exc:
            print(f"[WARN] Cannot read {path}: {exc}")

    return records


DICT_RE = re.compile(r"\{.*\}")
PROGRESS_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def read_train_log(
    path: Path,
) -> dict[str, list[tuple[int, float]]]:
    records: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return records

    current_step: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            progress = PROGRESS_RE.search(line)
            if progress:
                current_step = int(progress.group(1))

            match = DICT_RE.search(line)
            if not match:
                continue
            try:
                row = ast.literal_eval(match.group(0))
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            explicit = row_step(row)
            if explicit is not None:
                current_step = explicit
            if current_step is None:
                continue

            for key, value in row.items():
                if key in NON_METRIC_KEYS:
                    continue
                number = finite_float(value)
                if number is not None:
                    records[normalize_metric_name(key)].append(
                        (current_step, number)
                    )
    return records


def merge_sources(
    sources: list[dict[str, list[tuple[int, float]]]],
) -> dict[str, list[tuple[int, float]]]:
    """
    Sources are ordered from highest to lowest priority.

    A higher-priority source owns a metric when it contains that metric. This
    prevents a lower-priority source with an incorrect synthetic axis from
    overwriting loss_log.jsonl's corrected 0..5000 axis.
    """
    result: dict[str, list[tuple[int, float]]] = {}
    all_names = {
        name
        for source in sources
        for name, values in source.items()
        if values
    }

    for name in sorted(all_names):
        selected: list[tuple[int, float]] = []
        for source in sources:
            values = source.get(name, [])
            if values:
                selected = values
                break

        deduplicated: dict[int, float] = {}
        for step, value in selected:
            deduplicated[int(step)] = float(value)
        result[name] = sorted(deduplicated.items())

    return result


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]

    result: list[float] = []
    queue: list[float] = []
    running = 0.0
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        result.append(running / len(queue))
    return result


def find_metric(
    records: dict[str, list[tuple[int, float]]],
    names: list[str],
) -> str | None:
    for name in names:
        if records.get(name):
            return name
    for actual, values in records.items():
        lower = actual.lower()
        if values and any(name.lower() in lower for name in names):
            return actual
    return None


def plot_metric(
    values: list[tuple[int, float]],
    path: Path,
    title: str,
    ylabel: str,
    smooth_window: int,
    final_step: int | None,
) -> None:
    steps = [step for step, _ in values]
    raw = [value for _, value in values]
    smoothed = moving_average(raw, smooth_window)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, raw, linewidth=0.8, alpha=0.25, label="raw")
    plt.plot(
        steps,
        smoothed,
        linewidth=2.0,
        label=f"moving average ({smooth_window})",
    )
    plt.xlabel("Global training step")
    plt.ylabel(ylabel)
    plt.title(title)
    if final_step:
        plt.xlim(0, final_step)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_csv(
    records: dict[str, list[tuple[int, float]]],
    path: Path,
) -> None:
    metric_names = sorted(records)
    all_steps = sorted(
        {
            step
            for values in records.values()
            for step, _ in values
        }
    )
    maps = {
        name: dict(records[name])
        for name in metric_names
    }

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", *metric_names])
        for step in all_steps:
            writer.writerow(
                [step, *[maps[name].get(step, "") for name in metric_names]]
            )


def make_dashboard(
    records: dict[str, list[tuple[int, float]]],
    path: Path,
    smooth_window: int,
    final_step: int | None,
) -> None:
    specs = [
        (["model_forward_time"], "model_forward_time"),
        (["dynamics_loss_avg"], "dynamics_loss_avg"),
        (["action_loss_avg"], "action_loss_avg"),
        (["training_step_time"], "training_step_time"),
        (["loss"], "loss"),
        (["grad_norm"], "grad_norm"),
        (["learning_rate"], "learning_rate"),
    ]

    available = []
    for candidates, title in specs:
        key = find_metric(records, candidates)
        if key:
            available.append((key, title))

    if not available:
        return

    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(14, max(4, 3.1 * len(available))),
        sharex=True,
    )
    if len(available) == 1:
        axes = [axes]

    for axis, (key, title) in zip(axes, available):
        values = records[key]
        steps = [step for step, _ in values]
        raw = [value for _, value in values]
        smooth = moving_average(raw, smooth_window)

        axis.plot(steps, raw, linewidth=0.7, alpha=0.22)
        axis.plot(steps, smooth, linewidth=1.7)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        if final_step:
            axis.set_xlim(0, final_step)

    axes[-1].set_xlabel("Global training step")
    fig.suptitle(
        f"DreamZero G2 Training Metrics — final global step: "
        f"{final_step if final_step else 'unknown'}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_file = (
        args.log_file.resolve()
        if args.log_file
        else run_dir / "train.log"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "training_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_run_metadata(run_dir)
    final_step = (
        metadata["global_step"]
        or metadata["max_steps"]
    )
    final_step = int(final_step) if final_step else None

    print("[INFO] run directory:", run_dir)
    print("[INFO] trainer global_step:", metadata["global_step"])
    print("[INFO] trainer max_steps:", metadata["max_steps"])
    print("[INFO] trainer logging_steps:", metadata["logging_steps"])

    loss_jsonl = read_loss_jsonl(
        run_dir / "loss_log.jsonl",
        final_step,
        (
            int(metadata["logging_steps"])
            if metadata["logging_steps"]
            else None
        ),
    )
    trainer = read_trainer_history(run_dir)
    events = read_tensorboard(run_dir)
    plain_log = read_train_log(log_file)

    records = merge_sources(
        [loss_jsonl, trainer, events, plain_log]
    )
    records = {
        name: values
        for name, values in records.items()
        if values
    }
    if not records:
        print("[ERROR] No scalar training metrics were found.")
        return 1

    print("[INFO] selected metrics:")
    for name, values in records.items():
        print(
            f"  {name}: {len(values)} points, "
            f"step {values[0][0]} -> {values[-1][0]}"
        )

    write_csv(records, output_dir / "training_metrics_global_step.csv")

    individual_specs = [
        (
            ["loss"],
            "01_loss_global_step.png",
            "DreamZero G2 Total Loss",
            "Loss",
        ),
        (
            ["action_loss_avg"],
            "02_action_loss_global_step.png",
            "DreamZero G2 Action Loss",
            "Action loss",
        ),
        (
            ["dynamics_loss_avg"],
            "03_dynamics_loss_global_step.png",
            "DreamZero G2 Dynamics Loss",
            "Dynamics loss",
        ),
        (
            ["learning_rate"],
            "04_learning_rate_global_step.png",
            "Learning Rate",
            "Learning rate",
        ),
        (
            ["grad_norm"],
            "05_grad_norm_global_step.png",
            "Gradient Norm",
            "Gradient norm",
        ),
        (
            ["training_step_time"],
            "06_training_step_time_global_step.png",
            "Training Step Time",
            "Seconds",
        ),
        (
            ["model_forward_time"],
            "07_model_forward_time_global_step.png",
            "Model Forward Time",
            "Seconds",
        ),
    ]

    for candidates, filename, title, ylabel in individual_specs:
        key = find_metric(records, candidates)
        if not key:
            continue
        plot_metric(
            records[key],
            output_dir / filename,
            title,
            ylabel,
            args.smooth_window,
            final_step,
        )

    make_dashboard(
        records,
        output_dir / "training_metrics_global_step.png",
        args.smooth_window,
        final_step,
    )

    print("[OK] plots written to:", output_dir)
    print(
        "[OK] main figure:",
        output_dir / "training_metrics_global_step.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
