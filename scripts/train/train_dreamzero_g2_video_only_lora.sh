#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/home/ubuntu/projects/wangk/dreamzero
G2_DATA_ROOT=/data/training_data/teleop/g2/g2_tasks_g1_g7_joint_gear_subtask_v2/train
WAN_CKPT_DIR=/data/wangk/checkpoints/Wan2.1-I2V-14B-480P
TOKENIZER_DIR=/data/wangk/checkpoints/umt5-xxl
PRETRAINED_MODEL_PATH=/data/wangk/checkpoints/DreamZero-AgiBot
OUTPUT_DIR=/data/wangk/checkpoints/dreamzero_g2_video_only_lora_1k
RUN_NAME=g2_video_only_wan_lora_1k

MAX_STEPS=1000
SAVE_STEPS=250

# 用户使用物理 GPU 0-3；物理 GPU 4-7 留给同事。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

# 始终使用已激活的 dreamzero Conda 环境，避免命中 ~/.local/bin/torchrun。
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
TRAIN_LOG="$OUTPUT_DIR/train.log"

echo "Python: $PYTHON_BIN"
echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training processes: 4"
echo "Dataset: $G2_DATA_ROOT"
echo "Run name: $RUN_NAME"
echo "Schedule: $MAX_STEPS steps, save every $SAVE_STEPS steps"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv

echo "Starting DreamZero G2 video-only Wan LoRA training"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node 4 \
    --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    +run_name="$RUN_NAME" \
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
    ++action_head_cfg.config.video_only_training=true \
    2>&1 | tee "$TRAIN_LOG"

echo "Completed successfully"
echo "Training output: $OUTPUT_DIR"
echo "Training log: $TRAIN_LOG"