#!/usr/bin/env bash
set -Eeuo pipefail

# Replay one fixed G2 held-out test sample through an AgiBot/G1 checkpoint.
# Only future-video quality is evaluated. Native AgiBot actions are discarded.

PROJECT_ROOT=${PROJECT_ROOT:-/home/ubuntu/projects/wangk/dreamzero}
EVAL_SCRIPT=${EVAL_SCRIPT:-$PROJECT_ROOT/eval_agibot_checkpoint_on_g2_testset.py}

MODEL_PATH=${MODEL_PATH:-/data/wangk/checkpoints/dreamzero_agibot_fruit_lora_20k/checkpoint-12000}
TEST_DATA_ROOT=${TEST_DATA_ROOT:-/data/training_data/teleop/g2/g2_tasks_g1_g7_joint_gear_subtask_v2/test}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data/wangk/checkpoints/umt5-xxl}

CHECKPOINT_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-/data/wangk/dreamzero/video_compare_agibot_vs_g1ft_vs_g2ft/$CHECKPOINT_NAME}

GPU_IDS=${GPU_IDS:-0,1}
EPISODE_INDICES=${EPISODE_INDICES:-0}
FRAME_INDEX=${FRAME_INDEX:-30}
FUTURE_FRAMES=${FUTURE_FRAMES:-33}
NUM_INFERENCE_TIMESTEPS=${NUM_INFERENCE_TIMESTEPS:-4}
SEED=${SEED:-42}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
PROMPT_OVERRIDE=${PROMPT_OVERRIDE:-}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -d "$PROJECT_ROOT" ]] || fail "Missing project root: $PROJECT_ROOT"
[[ -f "$EVAL_SCRIPT" ]] || fail "Missing evaluator: $EVAL_SCRIPT"
[[ -d "$TEST_DATA_ROOT" ]] || fail "Missing G2 test split: $TEST_DATA_ROOT"
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
  --embodiment-tag agibot
)

if [[ -n "$PROMPT_OVERRIDE" ]]; then
    ARGS+=(--prompt-override "$PROMPT_OVERRIDE")
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[INFO] Cheap preflight only; model will not be loaded."
    python "$EVAL_SCRIPT" "${ARGS[@]}" --preflight-only
    exit 0
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC_PER_NODE=${#GPU_ARRAY[@]}
(( NPROC_PER_NODE == 1 || NPROC_PER_NODE == 2 )) \
  || fail "Expected one or two GPUs; got GPU_IDS=$GPU_IDS"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

echo "[INFO] checkpoint embodiment: agibot"
echo "[INFO] visual source:         G2 held-out test videos"
echo "[INFO] checkpoint:            $MODEL_PATH"
echo "[INFO] test split:            $TEST_DATA_ROOT"
echo "[INFO] episodes:              $EPISODE_INDICES"
echo "[INFO] frame:                 $FRAME_INDEX"
echo "[INFO] output:                $OUTPUT_DIR"
echo "[INFO] GPUs:                  $GPU_IDS"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$EVAL_SCRIPT" \
  "${ARGS[@]}"
