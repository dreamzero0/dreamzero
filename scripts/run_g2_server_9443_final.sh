#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset ws_proxy wss_proxy WS_PROXY WSS_PROXY

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-FA2}"

PYTHON_BIN="${PYTHON_BIN:-/data/wangk/conda/envs/dreamzero/bin/python}"

PORT="${PORT:-9443}"

# inference only supports 1 or 2 GPUs
NUM_GPUS=2

# 固定使用 GPU0 + GPU1
CUDA_VISIBLE_DEVICES_VALUE="0,1"


# ==============================
# 新 action adapter checkpoint
# ==============================
MODEL_PATH="${MODEL_PATH:-/data/wangk/checkpoints/dreamzero_g2_nostate_v1_actvid_1to1_x001_3gpu/checkpoint-1500}"


WAN_CKPT_DIR="${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}"

TOKENIZER_PATH="${TOKENIZER_PATH:-/data/wangk/checkpoints/umt5-xxl}"

EMBODIMENT_TAG="g2"

SERVER_MODULE="${SERVER_MODULE:-socket_optimized_AR_g2.py}"


VIDEO_SAVE_MODE="${VIDEO_SAVE_MODE:-full}"

NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-/data/wangk/dreamzero/video_rollout_g2}"


if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: python missing:"
    echo "$PYTHON_BIN"
    exit 1
fi


if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR checkpoint missing:"
    echo "$MODEL_PATH"
    exit 1
fi


echo "=============================="
echo "DreamZero G2 Action Adapter"
echo "=============================="

echo "checkpoint:"
echo "$MODEL_PATH"

echo "GPU:"
echo "$CUDA_VISIBLE_DEVICES_VALUE"


echo ""
echo "Auditing checkpoint..."

"$PYTHON_BIN" scripts/audit_g2_checkpoint.py "$MODEL_PATH"


echo ""
echo "Starting DreamZero server"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    "$SERVER_MODULE" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --wan-ckpt-dir "$WAN_CKPT_DIR" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --video-save-mode "$VIDEO_SAVE_MODE" \
    --num-inference-timesteps "$NUM_INFERENCE_TIMESTEPS" \
    --output-dir "$OUTPUT_DIR"