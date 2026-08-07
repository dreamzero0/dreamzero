#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIM_NAMES = [
    "l.joint1",
    "l.joint2",
    "l.joint3",
    "l.joint4",
    "l.joint5",
    "l.joint6",
    "l.joint7",
    "l.gripper",
    "r.joint1",
    "r.joint2",
    "r.joint3",
    "r.joint4",
    "r.joint5",
    "r.joint6",
    "r.joint7",
    "r.gripper",
]


def load_train_actions(dataset_root: str, max_files: int | None = None):
    """
    读取训练集 parquet:
      - absolute action: df["action"]
      - relative action: df["action"] - df["observation.state"]

    返回:
      train_abs: (N, 16)
      train_rel: (N, 16)
    """
    pattern = os.path.join(dataset_root, "data", "chunk-*", "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {pattern}")

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    abs_list = []
    rel_list = []

    total_rows = 0
    for i, path in enumerate(parquet_files, 1):
        df = pd.read_parquet(path, columns=["observation.state", "action"])

        # 列里每个元素都是 list[16]
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)

        if states.shape[1] != 16 or actions.shape[1] != 16:
            raise ValueError(
                f"Expected 16-dim state/action, got state={states.shape}, action={actions.shape}, file={path}"
            )

        rel = actions - states

        abs_list.append(actions)
        rel_list.append(rel)

        total_rows += len(df)
        print(f"[train] loaded {i}/{len(parquet_files)}: {path} rows={len(df)} total_rows={total_rows}")

    train_abs = np.concatenate(abs_list, axis=0)
    train_rel = np.concatenate(rel_list, axis=0)

    print(f"[train] absolute actions shape: {train_abs.shape}")
    print(f"[train] relative actions shape: {train_rel.shape}")
    return train_abs, train_rel


def load_infer_json(json_path: str):
    """
    读取 robot_json / agibot_actions_xxx.json
    返回:
      infer_raw   : 模型原始输出动作 (T_all, 16)   -> 优先用 raw_actions
      infer_exec  : 最终执行动作     (T_all, 16)   -> 用 actions
      records     : 原始 step 记录
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Expected a non-empty list in {json_path}")

    raw_list = []
    exec_list = []

    for idx, rec in enumerate(records):
        if "raw_actions" in rec and rec["raw_actions"] is not None:
            arr = np.asarray(rec["raw_actions"], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] == 16:
                raw_list.append(arr)

        if "actions" in rec and rec["actions"] is not None:
            arr = np.asarray(rec["actions"], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] == 16:
                exec_list.append(arr)

    if len(raw_list) == 0:
        raise ValueError(f"No 16-D raw_actions found in {json_path}")

    if len(exec_list) == 0:
        raise ValueError(f"No 16-D actions found in {json_path}")

    infer_raw = np.concatenate(raw_list, axis=0)
    infer_exec = np.concatenate(exec_list, axis=0)

    print(f"[infer] raw_actions shape   : {infer_raw.shape}")
    print(f"[infer] actions shape       : {infer_exec.shape}")
    print(f"[infer] num step records    : {len(records)}")
    return infer_raw, infer_exec, records


def summarize_array(name: str, arr: np.ndarray):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"shape: {arr.shape}")
    for d in range(arr.shape[1]):
        col = arr[:, d]
        print(
            f"{d:2d} {DIM_NAMES[d]:>10s} | "
            f"mean={col.mean(): .5f} std={col.std(): .5f} "
            f"min={col.min(): .5f} p01={np.percentile(col,1): .5f} "
            f"p99={np.percentile(col,99): .5f} max={col.max(): .5f}"
        )


def make_output_dir(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def _auto_grid(n: int):
    cols = 4
    rows = math.ceil(n / cols)
    return rows, cols


def plot_hist_compare(
    ref_arr: np.ndarray,
    cmp_arr: np.ndarray,
    ref_name: str,
    cmp_name: str,
    out_path: str,
    bins: int = 80,
):
    """
    每个维度一张子图，画两个直方图轮廓
    """
    rows, cols = _auto_grid(ref_arr.shape[1])
    fig = plt.figure(figsize=(20, 4 * rows))

    for d in range(ref_arr.shape[1]):
        ax = fig.add_subplot(rows, cols, d + 1)
        ref_col = ref_arr[:, d]
        cmp_col = cmp_arr[:, d]

        all_min = min(ref_col.min(), cmp_col.min())
        all_max = max(ref_col.max(), cmp_col.max())

        if all_min == all_max:
            all_min -= 1e-3
            all_max += 1e-3

        edges = np.linspace(all_min, all_max, bins + 1)

        ax.hist(
            ref_col,
            bins=edges,
            histtype="step",
            density=True,
            label=ref_name,
            linewidth=1.5,
        )
        ax.hist(
            cmp_col,
            bins=edges,
            histtype="step",
            density=True,
            label=cmp_name,
            linewidth=1.5,
        )

        ax.set_title(f"{d}: {DIM_NAMES[d]}")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend()

    fig.suptitle(f"{ref_name} vs {cmp_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_percentile_compare(
    ref_arr: np.ndarray,
    cmp_arr: np.ndarray,
    ref_name: str,
    cmp_name: str,
    out_path: str,
):
    """
    每个维度画 p01/p50/p99 对比柱状图
    """
    dims = np.arange(ref_arr.shape[1])

    ref_p01 = np.percentile(ref_arr, 1, axis=0)
    ref_p50 = np.percentile(ref_arr, 50, axis=0)
    ref_p99 = np.percentile(ref_arr, 99, axis=0)

    cmp_p01 = np.percentile(cmp_arr, 1, axis=0)
    cmp_p50 = np.percentile(cmp_arr, 50, axis=0)
    cmp_p99 = np.percentile(cmp_arr, 99, axis=0)

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)

    width = 0.12
    ax.bar(dims - 2.5 * width, ref_p01, width=width, label=f"{ref_name} p01")
    ax.bar(dims - 1.5 * width, ref_p50, width=width, label=f"{ref_name} p50")
    ax.bar(dims - 0.5 * width, ref_p99, width=width, label=f"{ref_name} p99")

    ax.bar(dims + 0.5 * width, cmp_p01, width=width, label=f"{cmp_name} p01")
    ax.bar(dims + 1.5 * width, cmp_p50, width=width, label=f"{cmp_name} p50")
    ax.bar(dims + 2.5 * width, cmp_p99, width=width, label=f"{cmp_name} p99")

    ax.set_xticks(dims)
    ax.set_xticklabels(DIM_NAMES, rotation=45, ha="right")
    ax.set_title(f"Percentile Comparison: {ref_name} vs {cmp_name}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_single_rollout_timeseries(
    records: list,
    out_path: str,
    use_key: str = "actions",
):
    """
    画单个推理 JSON 中所有 step 的 action 时间序列。
    这里把所有 step 的 action 直接串起来。
    """
    arrs = []
    boundaries = []

    cur = 0
    for rec in records:
        if use_key not in rec or rec[use_key] is None:
            continue
        arr = np.asarray(rec[use_key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 16:
            continue
        arrs.append(arr)
        cur += arr.shape[0]
        boundaries.append(cur)

    if not arrs:
        raise ValueError(f"No valid {use_key} found for time-series plot.")

    seq = np.concatenate(arrs, axis=0)
    rows, cols = _auto_grid(seq.shape[1])

    fig = plt.figure(figsize=(20, 4 * rows))
    x = np.arange(seq.shape[0])

    for d in range(seq.shape[1]):
        ax = fig.add_subplot(rows, cols, d + 1)
        ax.plot(x, seq[:, d], linewidth=1.2)
        for b in boundaries[:-1]:
            ax.axvline(b, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{d}: {DIM_NAMES[d]}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Inference {use_key} Time Series", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_summary_csv(
    ref_arr: np.ndarray,
    cmp_arr: np.ndarray,
    ref_name: str,
    cmp_name: str,
    out_csv: str,
):
    rows = []
    for d in range(ref_arr.shape[1]):
        r = ref_arr[:, d]
        c = cmp_arr[:, d]
        rows.append({
            "dim": d,
            "name": DIM_NAMES[d],
            f"{ref_name}_mean": float(r.mean()),
            f"{ref_name}_std": float(r.std()),
            f"{ref_name}_p01": float(np.percentile(r, 1)),
            f"{ref_name}_p50": float(np.percentile(r, 50)),
            f"{ref_name}_p99": float(np.percentile(r, 99)),
            f"{cmp_name}_mean": float(c.mean()),
            f"{cmp_name}_std": float(c.std()),
            f"{cmp_name}_p01": float(np.percentile(c, 1)),
            f"{cmp_name}_p50": float(np.percentile(c, 50)),
            f"{cmp_name}_p99": float(np.percentile(c, 99)),
            "mean_abs_gap": float(abs(r.mean() - c.mean())),
            "p99_abs_gap": float(abs(np.percentile(r, 99) - np.percentile(c, 99))),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root",
        required=True,
        help="GEAR train root, e.g. /data/training_data/teleop/g2/g2_mock_light_module_joint_gear/train",
    )
    parser.add_argument(
        "--infer-json",
        required=True,
        help="robot_json/agibot_actions_xxx.json",
    )
    parser.add_argument(
        "--out-dir",
        default="./action_compare_outputs",
        help="output directory",
    )
    parser.add_argument(
        "--max-train-files",
        type=int,
        default=None,
        help="only load first N parquet files for quick debug",
    )
    args = parser.parse_args()

    out_dir = make_output_dir(args.out_dir)

    # 1) load
    train_abs, train_rel = load_train_actions(args.train_root, args.max_train_files)
    infer_raw, infer_exec, records = load_infer_json(args.infer_json)

    # 2) summary
    summarize_array("TRAIN ABSOLUTE ACTION", train_abs)
    summarize_array("TRAIN RELATIVE ACTION", train_rel)
    summarize_array("INFER RAW ACTION", infer_raw)
    summarize_array("INFER EXEC ACTION", infer_exec)

    # 3) save csv
    save_summary_csv(
        train_abs,
        infer_exec,
        "train_abs",
        "infer_exec",
        os.path.join(out_dir, "summary_train_abs_vs_infer_exec.csv"),
    )

    save_summary_csv(
        train_rel,
        infer_raw,
        "train_rel",
        "infer_raw",
        os.path.join(out_dir, "summary_train_rel_vs_infer_raw.csv"),
    )

    # 4) plots
    # A. 最重要：训练 absolute vs 推理最终执行
    plot_hist_compare(
        train_abs,
        infer_exec,
        "train_abs",
        "infer_exec",
        os.path.join(out_dir, "hist_train_abs_vs_infer_exec.png"),
    )
    plot_percentile_compare(
        train_abs,
        infer_exec,
        "train_abs",
        "infer_exec",
        os.path.join(out_dir, "percentile_train_abs_vs_infer_exec.png"),
    )

    # B. 用来查 relative/absolute 是否错位：训练 relative vs 推理原始输出
    plot_hist_compare(
        train_rel,
        infer_raw,
        "train_rel",
        "infer_raw",
        os.path.join(out_dir, "hist_train_rel_vs_infer_raw.png"),
    )
    plot_percentile_compare(
        train_rel,
        infer_raw,
        "train_rel",
        "infer_raw",
        os.path.join(out_dir, "percentile_train_rel_vs_infer_raw.png"),
    )

    # C. 单次 rollout 时间序列
    plot_single_rollout_timeseries(
        records,
        os.path.join(out_dir, "timeseries_infer_raw_actions.png"),
        use_key="raw_actions",
    )
    plot_single_rollout_timeseries(
        records,
        os.path.join(out_dir, "timeseries_infer_exec_actions.png"),
        use_key="actions",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
