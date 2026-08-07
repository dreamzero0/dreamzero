#!/usr/bin/env python3
"""Create a zero-delta LoRA checkpoint for base-model inference controls."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")

    source_weights = source / "model.safetensors"
    source_config = source / "config.json"
    source_experiment = source / "experiment_cfg"
    for path in (source_weights, source_config, source_experiment):
        if not path.exists():
            raise FileNotFoundError(path)

    state = load_file(source_weights)
    zero_lora = {
        key: torch.zeros_like(value)
        for key, value in state.items()
        if "lora_A" in key or "lora_B" in key
    }
    if not zero_lora:
        raise RuntimeError(f"No LoRA tensors found in {source_weights}")

    output.mkdir(parents=True)
    save_file(zero_lora, output / "model.safetensors")
    shutil.copy2(source_config, output / "config.json")
    shutil.copytree(source_experiment, output / "experiment_cfg")
    (output / "CONTROL_INFO.json").write_text(
        json.dumps(
            {
                "purpose": "DreamZero-AgiBot base control in G2 architecture",
                "source_checkpoint": str(source),
                "zeroed_lora_tensors": len(zero_lora),
                "non_lora_checkpoint_tensors_loaded": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Created {output} with {len(zero_lora)} zero LoRA tensors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
