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

PYTHON_BIN="${PYTHON_BIN:-/home/user/miniconda3/envs/dreamzero/bin/python}"
PORT="${PORT:-9443}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-/data/wangk/checkpoints/DreamZero-AgiBot}"
VIDEO_SAVE_MODE="${VIDEO_SAVE_MODE:-full}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-0}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
  echo "Set MODEL_PATH to your DreamZero-AgiBot checkpoint directory before starting."
  echo "Example:"
  echo "  MODEL_PATH=/data1/wangk/checkpoints/DreamZero-AgiBot bash scripts/run_agibot_server_6006.sh"
  exit 1
fi

echo "Starting DreamZero AgiBot server"
echo "  root: $ROOT_DIR"
echo "  port: $PORT"
echo "  num_gpus: $NUM_GPUS"
echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES_VALUE"
echo "  python_bin: $PYTHON_BIN"
echo "  model_path: $MODEL_PATH"
echo "  video_save_mode: $VIDEO_SAVE_MODE"
echo "  num_inference_timesteps: $NUM_INFERENCE_TIMESTEPS"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  socket_test_optimized_AR.py \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --embodiment-tag agibot \
  --video-save-mode "$VIDEO_SAVE_MODE" \
  --num-inference-timesteps "$NUM_INFERENCE_TIMESTEPS"
