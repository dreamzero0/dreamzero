#!/usr/bin/env bash
set -Eeuo pipefail

# DreamZero G2 joint-space LoRA training.
# Point G2_DATA_ROOT at the GEAR train split, never at its parent/test folder.

PROJECT_ROOT=${PROJECT_ROOT:-/home/ubuntu/projects/wangk/dreamzero}
G2_DATA_ROOT=${G2_DATA_ROOT:-/data/training_data/teleop/g2/g2_tasks_g1_g7_joint_gear_subtask_v2/train}

OUTPUT_DIR=${OUTPUT_DIR:-/data/wangk/checkpoints/dreamzero_g2_joint_subtask_lora_v2}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-/data/wangk/checkpoints/umt5-xxl}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-/data/wangk/checkpoints/DreamZero-AgiBot}

# Only use physical GPUs 4 and above. GPU_IDS is preferred, while an existing
# CUDA_VISIBLE_DEVICES remains supported. The torchrun process count is always
# derived from the resulting list (for example, 6,7 means exactly 2 workers).
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-1346}
MAX_STEPS=${MAX_STEPS:-3000}
SAVE_STEPS=${SAVE_STEPS:-500}
WANDB_MODE=${WANDB_MODE:-offline}
HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

export G2_DATA_ROOT OUTPUT_DIR WAN_CKPT_DIR TOKENIZER_DIR
export PRETRAINED_MODEL_PATH GPU_IDS
export EXPECTED_EPISODES MAX_STEPS SAVE_STEPS WANDB_MODE HYDRA_FULL_ERROR

# Pin every Python entry point to the currently activated conda environment.
# This prevents ~/.local/bin/torchrun from launching /usr/bin/python3.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
[[ -x "$PYTHON_BIN" ]] || {
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    exit 1
}
export PYTHON_BIN

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_dir() {
    [[ -d "$1" ]] || fail "Missing directory: $1"
}

require_file() {
    [[ -f "$1" ]] || fail "Missing file: $1"
}

trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

require_dir "$PROJECT_ROOT"
require_dir "$G2_DATA_ROOT"
require_dir "$WAN_CKPT_DIR"
require_dir "$TOKENIZER_DIR"
require_dir "$PRETRAINED_MODEL_PATH"
require_file "$PROJECT_ROOT/groot/vla/experiment/experiment.py"
require_file "$PROJECT_ROOT/groot/vla/configs/data/dreamzero/g2_relative.yaml"
require_file "$PROJECT_ROOT/groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml"
require_file "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
require_file "$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
require_file "$WAN_CKPT_DIR/Wan2.1_VAE.pth"
require_file "$G2_DATA_ROOT/meta/info.json"
require_file "$G2_DATA_ROOT/meta/modality.json"
require_file "$G2_DATA_ROOT/meta/embodiment.json"
require_file "$G2_DATA_ROOT/meta/stats.json"
require_file "$G2_DATA_ROOT/meta/relative_stats_dreamzero.json"

"$PYTHON_BIN" - <<'PY'
import sys

required = {
    "hydra": "hydra-core",
    "torch": "torch",
    "omegaconf": "omegaconf",
}

missing = []
for module_name, package_name in required.items():
    try:
        __import__(module_name)
    except Exception:
        missing.append(package_name)

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))

if missing:
    raise SystemExit(
        "Missing packages in the active Python environment: "
        + ", ".join(missing)
    )
PY

# Keep GPU visibility and torchrun world size under one source of truth. This
# also prevents an inherited NUM_GPUS from launching more workers than visible
# devices.
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
(( ${#GPU_ARRAY[@]} > 0 )) || fail "GPU_IDS must contain at least one GPU index"

declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU_IDS must be comma-separated physical GPU indices; got: $GPU_IDS"
    (( gpu >= 4 )) || fail "Refusing to use physical GPU $gpu: GPUs 0-3 are reserved"
    [[ -z "${SEEN_GPUS[$gpu]+x}" ]] || fail "Duplicate GPU index in GPU_IDS: $gpu"
    SEEN_GPUS[$gpu]=1
done

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_ARRAY[*]}")
NPROC_PER_NODE=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES

# Fail before torchrun if any selected physical index does not exist.
for gpu in "${GPU_ARRAY[@]}"; do
    nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null \
        || fail "Physical GPU $gpu is unavailable"
done

cd "$PROJECT_ROOT"

echo "[1/2] Validating the G2 joint subtask training split"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G2_DATA_ROOT"])
with (root / "meta/info.json").open() as handle:
    info = json.load(handle)
with (root / "meta/embodiment.json").open() as handle:
    embodiment = json.load(handle)

episodes = int(info["total_episodes"])
expected_episodes = int(os.environ["EXPECTED_EPISODES"])
parquets = list(root.glob("data/chunk-*/*.parquet"))
videos = list(root.glob("videos/chunk-*/*/*.mp4"))

if episodes != expected_episodes:
    raise RuntimeError(
        f"Expected {expected_episodes} training episodes, got {episodes}"
    )
if len(parquets) != episodes:
    raise RuntimeError(f"Expected {episodes} parquets, got {len(parquets)}")
if len(videos) != episodes * 3:
    raise RuntimeError(f"Expected {episodes * 3} videos, got {len(videos)}")
if info["features"]["observation.state"]["shape"] != [16]:
    raise RuntimeError("observation.state must be 16-dimensional G2 joint state")
if info["features"]["action"]["shape"] != [16]:
    raise RuntimeError("action must be 16-dimensional G2 joint action")
if embodiment.get("embodiment_tag") != "g2":
    raise RuntimeError(f"Expected embodiment_tag=g2, got {embodiment}")

video_features = {
    key: feature
    for key, feature in info["features"].items()
    if feature.get("dtype") == "video"
}
expected_video_keys = {
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
}
if set(video_features) != expected_video_keys:
    raise RuntimeError(f"Unexpected video features: {sorted(video_features)}")
for key, feature in video_features.items():
    if feature["shape"] != [176, 320, 3]:
        raise RuntimeError(f"Unexpected shape for {key}: {feature['shape']}")

print(
    f"Validation OK: {episodes} train episodes, "
    f"{len(parquets)} parquets, {len(videos)} videos, G2 joint 16D"
)
PY

mkdir -p "$OUTPUT_DIR"
TRAIN_LOG="$OUTPUT_DIR/train.log"

echo "Physical GPUs selected: $CUDA_VISIBLE_DEVICES"
echo "Training processes: $NPROC_PER_NODE (one per selected GPU)"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv

echo "W&B mode: $WANDB_MODE"
echo "Checkpoint interval: every $SAVE_STEPS steps"
echo "[2/2] Starting ${MAX_STEPS}-step DreamZero G2 joint LoRA training"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$NPROC_PER_NODE" \
    --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/g2_relative \
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
    save_steps="$SAVE_STEPS" \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=5 \
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
    mixture_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec \
    single_dataset_cls=groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID \
    frame_seqlen=880 \
    save_strategy=steps \
    g2_data_root="$G2_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    ++model_specific_transform.embodiment_tag_mapping.g2=26 \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    2>&1 | tee "$TRAIN_LOG"

echo "Completed successfully"
echo "Dataset: $G2_DATA_ROOT"
echo "Training output: $OUTPUT_DIR"
echo "Training log: $TRAIN_LOG"