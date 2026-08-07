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
PORT="${PORT:-6006}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"

SERVER_MODULE="${SERVER_MODULE:-socket_test_optimized_AR_agibot_fruit_video.py}"
EMBODIMENT_TAG="agibot"

# Select one experiment explicitly.
MODEL_PATH="${MODEL_PATH:-/data/wangk/checkpoints/DreamZero-AgiBot}"
RUN_NAME="${RUN_NAME:-official_agibot}"

WAN_CKPT_DIR="${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/wangk/checkpoints/umt5-xxl}"

# Generated-model video and real first-person video are separate.
VIDEO_SAVE_MODE="${VIDEO_SAVE_MODE:-full}"
INPUT_VIDEO_SAVE_MODE="${INPUT_VIDEO_SAVE_MODE:-top_head}"
INPUT_VIDEO_FPS="${INPUT_VIDEO_FPS:-30}"

NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/wangk/dreamzero/fruit_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi

if [ ! -f "$SERVER_MODULE" ]; then
  echo "ERROR: SERVER_MODULE does not exist: $ROOT_DIR/$SERVER_MODULE"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
  exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ]; then
  echo "ERROR: WAN_CKPT_DIR does not exist: $WAN_CKPT_DIR"
  exit 1
fi

if [ ! -e "$TOKENIZER_PATH" ]; then
  echo "ERROR: TOKENIZER_PATH does not exist: $TOKENIZER_PATH"
  exit 1
fi

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES_VALUE"
if [ "${#CUDA_DEVICE_ARRAY[@]}" -ne "$NUM_GPUS" ]; then
  echo "ERROR: NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES has ${#CUDA_DEVICE_ARRAY[@]} device(s)"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/run_manifest.txt" <<EOF
run_name=$RUN_NAME
model_path=$MODEL_PATH
embodiment_tag=$EMBODIMENT_TAG
port=$PORT
cuda_visible_devices=$CUDA_VISIBLE_DEVICES_VALUE
video_save_mode=$VIDEO_SAVE_MODE
input_video_save_mode=$INPUT_VIDEO_SAVE_MODE
input_video_fps=$INPUT_VIDEO_FPS
num_inference_timesteps=$NUM_INFERENCE_TIMESTEPS
started_at=$(date --iso-8601=seconds)
EOF

echo "Starting DreamZero AgiBot fruit evaluation"
echo "  run_name: $RUN_NAME"
echo "  model_path: $MODEL_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  generated_video: $VIDEO_SAVE_MODE"
echo "  real_input_video: $INPUT_VIDEO_SAVE_MODE"
echo "  input_video_fps: $INPUT_VIDEO_FPS"
echo "  port: $PORT"
echo "  GPUs: $CUDA_VISIBLE_DEVICES_VALUE"

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
  --input-video-save-mode "$INPUT_VIDEO_SAVE_MODE" \
  --input-video-fps "$INPUT_VIDEO_FPS" \
  --num-inference-timesteps "$NUM_INFERENCE_TIMESTEPS" \
  --output-dir "$OUTPUT_DIR"
