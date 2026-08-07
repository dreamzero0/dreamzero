#!/usr/bin/env python3
"""Re-render plots and presentation sheets from saved sensitivity outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action_input_sensitivity import (
    _condition_label,
    _label,
    _make_montage_video,
    _read_mp4,
    _save_video,
    _stage_info,
    _write_action_plots,
    _write_round_contact_sheet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    for pair_dir in sorted(args.root.glob("target_*_donor_*")):
        if not pair_dir.is_dir():
            continue
        report_path = next(pair_dir.glob("*_report.json"))
        arrays_path = next(pair_dir.glob("*_action_arrays.npz"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        arrays = np.load(arrays_path)
        predictions = {
            key.removeprefix("pred_"): arrays[key]
            for key in arrays.files
            if key.startswith("pred_")
        }
        target_episode = int(report["target_episode"])
        donor_episode = int(report["donor_episode"])
        anchors = [int(value) for value in report["anchors"]]
        donor_anchors = [int(value) for value in report["donor_anchors"]]
        target_stage = _stage_info(anchors[0], int(report["target_rows"]))
        donor_stage = _stage_info(donor_anchors[0], int(report["donor_rows"]))
        labels = {
            condition: _condition_label(condition, target_stage, donor_stage)
            for condition in predictions
        }

        plot_report = _write_action_plots(
            pair_dir,
            target_episode,
            donor_episode,
            anchors,
            arrays["ground_truth"],
            predictions,
        )

        condition_frames: dict[str, list[np.ndarray]] = {}
        gt_frames: list[np.ndarray] | None = None
        for condition in predictions:
            slow_path = Path(report["condition_videos"][condition]["slow_video"])
            combined = _read_mp4(slow_path)
            half = combined[0].shape[1] // 2
            predicted = [frame[:, :half].copy() for frame in combined]
            ground_truth = [frame[:, half:].copy() for frame in combined]
            relabeled = [
                np.concatenate([_label(frame, labels[condition]), gt], axis=1)
                for frame, gt in zip(predicted, ground_truth)
            ]
            _save_video(slow_path, relabeled, int(report["display_video_fps"]))
            condition_frames[condition] = predicted
            if gt_frames is None:
                gt_frames = ground_truth

        assert gt_frames is not None
        montage_path = Path(report["video_montage_slow"])
        _make_montage_video(
            montage_path,
            condition_frames,
            gt_frames,
            int(report["display_video_fps"]),
            labels,
        )
        contact_sheet_path = Path(report["rounds_contact_sheet"])
        _write_round_contact_sheet(
            contact_sheet_path,
            {},
            [int(value) for value in report["starts"]],
            condition_frames,
            gt_frames,
            labels,
        )

        report["target_stage"] = target_stage
        report["donor_stage"] = donor_stage
        report["presentation_labels"] = labels
        report.update(plot_report)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(pair_dir)


if __name__ == "__main__":
    main()
