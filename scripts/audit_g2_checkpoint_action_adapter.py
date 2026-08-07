#!/usr/bin/env python3
"""CPU-only validation of a DreamZero G2 LoRA deployment checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


EXPECTED_ACTION_HORIZON = 24
EXPECTED_NUM_FRAMES = 33
EXPECTED_ACTION_DIM = 32
EXPECTED_OUTPUT_DIM = 16


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"{path} has an incomplete safetensors header")
        header_size = int.from_bytes(header_size_bytes, "little")
        if header_size <= 0 or header_size > path.stat().st_size - 8:
            raise ValueError(
                f"{path} has an invalid safetensors header size: {header_size}"
            )
        return json.loads(stream.read(header_size))


def _nested(config: dict, *keys: str):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def audit(checkpoint: Path) -> list[str]:
    errors: list[str] = []
    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "experiment_cfg" / "conf.yaml",
        checkpoint / "experiment_cfg" / "metadata.json",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing required file: {path}")
    if errors:
        return errors

    config = json.loads((checkpoint / "config.json").read_text())
    inner = _nested(config, "action_head_cfg", "config") or {}
    diffusion = inner.get("diffusion_model_cfg", {})
    checks = {
        "config.action_horizon": (
            config.get("action_horizon"),
            EXPECTED_ACTION_HORIZON,
        ),
        "action_head.action_horizon": (
            inner.get("action_horizon"),
            EXPECTED_ACTION_HORIZON,
        ),
        "action_head.num_frames": (
            inner.get("num_frames"),
            EXPECTED_NUM_FRAMES,
        ),
        "action_head.action_dim": (
            inner.get("action_dim"),
            EXPECTED_ACTION_DIM,
        ),
        "diffusion.num_action_per_block": (
            diffusion.get("num_action_per_block"),
            EXPECTED_ACTION_HORIZON,
        ),
        "diffusion.out_dim": (
            diffusion.get("out_dim"),
            EXPECTED_OUTPUT_DIM,
        ),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name}: expected {expected}, got {actual!r}")

    conf_text = (
        checkpoint / "experiment_cfg" / "conf.yaml"
    ).read_text(errors="replace")
    if not re.search(r"(?m)^save_lora_only:\s*true\s*$", conf_text):
        errors.append("experiment config does not declare save_lora_only: true")
    pretrained_match = re.search(
        r"(?m)^pretrained_model_path:\s*(\S+)\s*$",
        conf_text,
    )
    if not pretrained_match:
        errors.append("experiment config has no pretrained_model_path")
    else:
        base_path = Path(pretrained_match.group(1))
        if not base_path.is_dir():
            errors.append(f"pretrained base directory is missing: {base_path}")
        elif not (
            (base_path / "model.safetensors").is_file()
            or (base_path / "model.safetensors.index.json").is_file()
        ):
            errors.append(
                f"pretrained base has no safetensors weights: {base_path}"
            )

    header = _read_safetensors_header(checkpoint / "model.safetensors")
    keys = [key for key in header if key != "__metadata__"]
    buckets = Counter()
    for key in keys:
        lowered = key.lower()
        if "lora_a" in lowered:
            buckets["lora_A"] += 1
        elif "lora_b" in lowered:
            buckets["lora_B"] += 1
        elif "action_encoder" in lowered:
            buckets["action_encoder"] += 1
        elif "action_decoder" in lowered:
            buckets["action_decoder"] += 1
        elif "state_encoder" in lowered:
            buckets["state_encoder"] += 1
        else:
            buckets["other"] += 1

    if buckets["lora_A"] == 0:
        errors.append("checkpoint contains no LoRA A weights")
    if buckets["lora_A"] != buckets["lora_B"]:
        errors.append(
            "unbalanced LoRA weights: "
            f"A={buckets['lora_A']} B={buckets['lora_B']}"
        )
    for name, minimum in (
        ("action_encoder", 6),
        ("action_decoder", 4),
        ("state_encoder", 4),
    ):
        if buckets[name] < minimum:
            errors.append(
                f"incomplete {name}: expected at least {minimum}, "
                f"got {buckets[name]}"
            )

    print(f"checkpoint: {checkpoint}")
    print(f"pretrained_model_path: {pretrained_match.group(1) if pretrained_match else 'MISSING'}")
    print(f"tensor_keys: {len(keys)}")
    print(f"tensor_buckets: {dict(buckets)}")
    for name, (actual, expected) in checks.items():
        print(f"{name}: {actual!r} (expected {expected})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    errors = audit(args.checkpoint.expanduser().resolve())
    if errors:
        print("G2 CHECKPOINT AUDIT FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("G2 CHECKPOINT AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())