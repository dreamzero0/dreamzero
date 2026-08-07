#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/ubuntu/projects/wangk/dreamzero}
G2_DATA_ROOT=${G2_DATA_ROOT:-/data/training_data/teleop/g2/g2_mock_light_module_joint_gear_policy_gripper/train}
OUTPUT_DIR=${OUTPUT_DIR:-/data/wangk/checkpoints/dreamzero_g2_absolute_video1_action10_v1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-/data/wangk/checkpoints/umt5-xxl}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-/data/wangk/checkpoints/DreamZero-AgiBot}
PYTHON_BIN=${PYTHON_BIN:-/data/wangk/conda/envs/dreamzero/bin/python}

GPU_IDS=${GPU_IDS:-0,1,2,3}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-110}
MAX_STEPS=${MAX_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-500}
WANDB_MODE=${WANDB_MODE:-offline}
HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

export G2_DATA_ROOT OUTPUT_DIR WAN_CKPT_DIR TOKENIZER_DIR
export PRETRAINED_MODEL_PATH GPU_IDS EXPECTED_EPISODES MAX_STEPS SAVE_STEPS
export WANDB_MODE HYDRA_FULL_ERROR CUDA_VISIBLE_DEVICES="$GPU_IDS"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ "$GPU_IDS" == "0,1,2,3" ]] || fail "This run is pinned to GPUs 0,1,2,3"
for path in "$PROJECT_ROOT" "$G2_DATA_ROOT" "$WAN_CKPT_DIR" \
    "$TOKENIZER_DIR" "$PRETRAINED_MODEL_PATH"; do
    [[ -d "$path" ]] || fail "Missing directory: $path"
done
[[ -x "$PYTHON_BIN" ]] || fail "Missing Python interpreter: $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G2_DATA_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
expected = int(os.environ["EXPECTED_EPISODES"])
if info["total_episodes"] != expected:
    raise RuntimeError(f"Expected {expected} episodes, got {info['total_episodes']}")
if info["features"]["observation.state"]["shape"] != [16]:
    raise RuntimeError("G2 state must be 16D")
if info["features"]["action"]["shape"] != [16]:
    raise RuntimeError("G2 action must be 16D")
modality = json.loads((root / "meta/modality.json").read_text())
expected_video = {"top_head", "hand_left", "hand_right"}
if set(modality["video"]) != expected_video:
    raise RuntimeError(f"Unexpected video modalities: {sorted(modality['video'])}")
stats = json.loads((root / "meta/stats.json").read_text())
for feature_name in ("observation.state", "action"):
    values = stats[feature_name]
    for index, key in ((7, "left_gripper_position"), (15, "right_gripper_position")):
        if values["min"][index] < -1e-4 or values["max"][index] > 1.0001:
            raise RuntimeError(
                f"{feature_name} {key} is not policy-space [0,1]: "
                f"min={values['min'][index]} max={values['max'][index]}"
            )
# All-absolute action contract: joint positions must be absolute (not near-zero
# relative deltas). An absolute joint position spans a wide [q01, q99] range,
# whereas a relative delta sits in a narrow band around 0. Check the SPREAD.
for feature_name in ("observation.state", "action"):
    values = stats[feature_name]
    arm_indices = [*range(0, 7), *range(8, 15)]
    for i in arm_indices:
        spread = values["q99"][i] - values["q01"][i]
        if spread < 0.1:
            raise RuntimeError(
                f"{feature_name} dim {i} looks like a relative delta "
                f"(q01={values['q01'][i]:.4f} q99={values['q99'][i]:.4f} "
                f"spread={spread:.4f}); absolute joint-position stats expected."
            )
active_hold = json.loads((root / "meta/g2_active_hold_windows.json").read_text())
if active_hold["action_horizon"] != 24:
    raise RuntimeError("Active/hold index must use a 24-step horizon")
if active_hold["active_count"] <= 0 or active_hold["hold_count"] <= 0:
    raise RuntimeError("Active/hold index is empty")
tasks_path = root / "meta/tasks.jsonl"
tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
if len(tasks) != 1 or not str(tasks[0].get("task", "")).strip():
    raise RuntimeError("Expected exactly one non-empty G2 task prompt")
print(f"Prompt contract: {tasks[0]['task']}")
parquets = list(root.glob("data/chunk-*/*.parquet"))
videos = list(root.glob("videos/chunk-*/*/*.mp4"))
if len(parquets) != expected or len(videos) != expected * 3:
    raise RuntimeError(
        f"File count mismatch: parquets={len(parquets)} videos={len(videos)}"
    )
print(
    "Validated G2 all-absolute dataset: "
    f"episodes={expected} frames={info['total_frames']} tasks={info['total_tasks']} "
    f"videos={len(videos)} active={active_hold['active_count']} "
    f"hold={active_hold['hold_count']}"
)
PY

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    fail "Output directory is not empty: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
TRAIN_LOG="$OUTPUT_DIR/train.log"
cd "$PROJECT_ROOT"

echo "G2 all-absolute joint training: Future-Video→Action bottleneck + state late gated residual"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Base: $PRETRAINED_MODEL_PATH"
echo "Dataset: $G2_DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Objective: LoRA rank8 + flow action + decoded-action + endpoint (motion mask OFF)"

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node 4 \
    --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/g2_absolute \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    training_args.gradient_accumulation_steps=2 \
    save_steps="$SAVE_STEPS" \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=9 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    active_window_ratio=0.9 \
    g2_data_root="$G2_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    pretrained_lora_path=null \
    ++model_specific_transform.embodiment_tag_mapping.g2=33 \
    ++model_specific_transform.always_use_default_instruction=false \
    language_dropout_prob=0.0 \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    action_head_cfg.config.action_only_training=false \
    action_head_cfg.config.tune_diffusion_model=true \
    ++action_head_cfg.config.lora_rank=8 \
    ++action_head_cfg.config.lora_alpha=8 \
    action_head_cfg.config.decouple_video_action_noise=false \
    action_head_cfg.config.dynamics_loss_weight=1.0 \
    action_head_cfg.config.action_loss_weight=10.0 \
    ++action_head_cfg.config.action_x0_loss_weight=1.0 \
    ++action_head_cfg.config.action_endpoint_loss_weight=0.5 \
    ++action_head_cfg.config.action_velocity_loss_weight=0.5 \
    ++action_head_cfg.config.action_acc_loss_weight=0.1 \
    ++action_head_cfg.config.state_dropout=0.0 \
    ++action_head_cfg.config.cut_state_attention=true \
    ++action_head_cfg.config.motion_mask_enabled=false \
    2>&1 | tee "$TRAIN_LOG"

echo "Completed G2 all-absolute joint training"
