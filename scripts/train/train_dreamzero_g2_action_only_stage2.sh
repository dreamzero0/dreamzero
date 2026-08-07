#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/ubuntu/projects/wangk/dreamzero}
G2_DATA_ROOT=${G2_DATA_ROOT:-/data/training_data/teleop/g2/g2_mock_light_module_joint_gear_policy_gripper/train}
OUTPUT_DIR=${OUTPUT_DIR:-/data/wangk/checkpoints/dreamzero_g2_action_adapter_clean_v1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-/data/wangk/checkpoints/umt5-xxl}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-/data/wangk/checkpoints/DreamZero-AgiBot}
PYTHON_BIN=${PYTHON_BIN:-/data/wangk/conda/envs/dreamzero/bin/python}

GPU_IDS=${GPU_IDS:-0,1,2,3}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-110}
MAX_STEPS=${MAX_STEPS:-4000}
WANDB_MODE=${WANDB_MODE:-offline}
HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

export G2_DATA_ROOT OUTPUT_DIR WAN_CKPT_DIR TOKENIZER_DIR
export PRETRAINED_MODEL_PATH GPU_IDS
export EXPECTED_EPISODES MAX_STEPS WANDB_MODE HYDRA_FULL_ERROR

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

for path in \
    "$PROJECT_ROOT" \
    "$G2_DATA_ROOT" \
    "$WAN_CKPT_DIR" \
    "$TOKENIZER_DIR" \
    "$PRETRAINED_MODEL_PATH"; do
    [[ -d "$path" ]] || fail "Missing directory: $path"
done
[[ -x "$PYTHON_BIN" ]] || fail "Missing Python interpreter: $PYTHON_BIN"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
[[ ${#GPU_ARRAY[@]} -eq 4 ]] || fail "Stage 2 requires exactly four GPUs: 0,1,2,3"
[[ "$GPU_IDS" == "0,1,2,3" ]] || fail "Refusing GPU selection other than 0,1,2,3"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G2_DATA_ROOT"])
with (root / "meta/info.json").open() as stream:
    info = json.load(stream)
expected = int(os.environ["EXPECTED_EPISODES"])
if int(info["total_episodes"]) != expected:
    raise RuntimeError(
        f"Expected {expected} G2 episodes, got {info['total_episodes']}"
    )
if info["features"]["observation.state"]["shape"] != [16]:
    raise RuntimeError("G2 state must be 16D")
if info["features"]["action"]["shape"] != [16]:
    raise RuntimeError("G2 action must be 16D")
with (root / "meta/embodiment.json").open() as stream:
    embodiment = json.load(stream)
if embodiment.get("embodiment_tag") != "g2":
    raise RuntimeError(f"Expected embodiment_tag=g2, got {embodiment}")
with (root / "meta/g2_active_hold_windows.json").open() as stream:
    windows = json.load(stream)
if windows["active_count"] <= 0 or windows["hold_count"] <= 0:
    raise RuntimeError("Active/hold index must contain both groups")
print(
    "Validated G2 action-adapter dataset: "
    f"{expected} episodes, 16D state/action, "
    f"active={windows['active_count']} hold={windows['hold_count']}"
)
PY

mkdir -p "$OUTPUT_DIR"
TRAIN_LOG="$OUTPUT_DIR/train.log"
cd "$PROJECT_ROOT"

echo "G2 Action-Adapter-Only training"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "DreamZero base: $PRETRAINED_MODEL_PATH"
echo "G2 LoRA/action initialization: none (clean official-base start)"
echo "Output: $OUTPUT_DIR"

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node 4 \
    --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/g2_relative \
    wandb_project=dreamzero \
    train_architecture=action_only \
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
    milestone_save_steps="[100,250,500,1000,2000,3000,4000]" \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=7 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=no \
    g2_data_root="$G2_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    pretrained_lora_path=null \
    ++model_specific_transform.embodiment_tag_mapping.g2=33 \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=false \
    action_head_cfg.config.action_only_training=true \
    action_head_cfg.config.tune_diffusion_model=false \
    action_head_cfg.config.dynamics_loss_weight=1.0 \
    action_head_cfg.config.action_loss_weight=1.0 \
    2>&1 | tee "$TRAIN_LOG"

echo "Completed G2 Action-Adapter-Only training: $OUTPUT_DIR"
