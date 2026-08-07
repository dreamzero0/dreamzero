#!/usr/bin/env bash
set -Eeuo pipefail

# Offline G2 held-out test-set video prediction.
# No robot, WebSocket client, JPEG transport, or SDK action execution.

PROJECT_ROOT=${PROJECT_ROOT:-/home/ubuntu/projects/wangk/dreamzero}
PYTHON_BIN=${PYTHON_BIN:-/data/wangk/conda/envs/dreamzero/bin/python}
EVAL_SCRIPT=${EVAL_SCRIPT:-$PROJECT_ROOT/eval_g2_checkpoint_on_testset.py}
TEST_DATA_ROOT=${TEST_DATA_ROOT:-/data/training_data/teleop/g2/g2_tasks_g1_g7_joint_gear_subtask_v2/test}

# Change this path when sweeping joint/video-only checkpoints.
MODEL_PATH=${MODEL_PATH:-/data/wangk/checkpoints/dreamzero_g2_video_only_lora_1k/checkpoint-1000}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data/wangk/checkpoints/umt5-xxl}

RUN_NAME="$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")"
OUTPUT_DIR=${OUTPUT_DIR:-/data/wangk/dreamzero/g2_testset_video_eval/$RUN_NAME}

GPU_IDS=${GPU_IDS:-0,1}
EPISODE_INDICES=${EPISODE_INDICES:-0}
FRAME_INDEX=${FRAME_INDEX:-30}
FUTURE_FRAMES=${FUTURE_FRAMES:-33}
NUM_INFERENCE_TIMESTEPS=${NUM_INFERENCE_TIMESTEPS:-16}
NUM_DIT_STEPS=${NUM_DIT_STEPS:-8}
SEED=${SEED:-42}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
PROMPT_OVERRIDE=${PROMPT_OVERRIDE:-}
WINDOWED=${WINDOWED:-0}
WINDOW_HISTORY=${WINDOW_HISTORY:-4}
WINDOW_STRIDE=${WINDOW_STRIDE:-4}
WINDOW_STARTS=${WINDOW_STARTS:-0,4,8,12,16,20,24,28}
ROLLOUT_FUTURE_FRAMES=${ROLLOUT_FUTURE_FRAMES:-33}
ROLLOUT_BLOCKS=${ROLLOUT_BLOCKS:-4}
SAVE_WINDOW_ARTIFACTS=${SAVE_WINDOW_ARTIFACTS:-0}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
[[ -d "$PROJECT_ROOT" ]] || fail "Missing project root: $PROJECT_ROOT"
[[ -f "$EVAL_SCRIPT" ]] || fail "Missing evaluator: $EVAL_SCRIPT"
[[ -d "$TEST_DATA_ROOT" ]] || fail "Missing G2 test split: $TEST_DATA_ROOT"
[[ -f "$TEST_DATA_ROOT/meta/info.json" ]] || fail "Missing test info.json"
[[ -f "$TEST_DATA_ROOT/meta/modality.json" ]] || fail "Missing test modality.json"
[[ -f "$TEST_DATA_ROOT/meta/embodiment.json" ]] || fail "Missing test embodiment.json"
[[ -d "$MODEL_PATH" ]] || fail "Missing checkpoint: $MODEL_PATH"
[[ -f "$MODEL_PATH/experiment_cfg/conf.yaml" ]] || fail "Missing checkpoint conf.yaml"
[[ -f "$MODEL_PATH/experiment_cfg/metadata.json" ]] || fail "Missing checkpoint metadata.json"
[[ -d "$WAN_CKPT_DIR" ]] || fail "Missing Wan checkpoint directory: $WAN_CKPT_DIR"
[[ -d "$TOKENIZER_PATH" ]] || fail "Missing tokenizer directory: $TOKENIZER_PATH"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

ARGS=(
  --model-path "$MODEL_PATH"
  --wan-ckpt-dir "$WAN_CKPT_DIR"
  --tokenizer-path "$TOKENIZER_PATH"
  --test-data-root "$TEST_DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --episode-indices "$EPISODE_INDICES"
  --frame-index "$FRAME_INDEX"
  --future-frames "$FUTURE_FRAMES"
  --num-inference-timesteps "$NUM_INFERENCE_TIMESTEPS"
  --seed "$SEED"
  --embodiment-tag g2
)

if [[ -n "$PROMPT_OVERRIDE" ]]; then
    ARGS+=(--prompt-override "$PROMPT_OVERRIDE")
fi
if [[ -n "$NUM_DIT_STEPS" ]]; then
    ARGS+=(--num-dit-steps "$NUM_DIT_STEPS")
fi
if [[ "$WINDOWED" == "1" ]]; then
    ARGS+=(
      --windowed
      --window-history "$WINDOW_HISTORY"
      --window-stride "$WINDOW_STRIDE"
      --window-starts "$WINDOW_STARTS"
      --rollout-future-frames "$ROLLOUT_FUTURE_FRAMES"
      --rollout-blocks "$ROLLOUT_BLOCKS"
    )
    if [[ "$SAVE_WINDOW_ARTIFACTS" == "1" ]]; then
        ARGS+=(--save-window-artifacts)
    fi
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[INFO] Running CPU preflight only; the 14B model will not be loaded."
    "$PYTHON_BIN" "$EVAL_SCRIPT" "${ARGS[@]}" --preflight-only
    exit 0
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC_PER_NODE=${#GPU_ARRAY[@]}
(( NPROC_PER_NODE == 1 || NPROC_PER_NODE == 2 )) \
  || fail "DreamZero inference supports 1 or 2 GPUs here; got GPU_IDS=$GPU_IDS"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

echo "[INFO] checkpoint: $MODEL_PATH"
echo "[INFO] test split: $TEST_DATA_ROOT"
echo "[INFO] episodes:   $EPISODE_INDICES"
echo "[INFO] frame:      $FRAME_INDEX"
echo "[INFO] output:     $OUTPUT_DIR"
echo "[INFO] GPUs:       $GPU_IDS"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$EVAL_SCRIPT" \
  "${ARGS[@]}"
