#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Disable all extension pruning by default so the run is pure offline+lora+sdpa.
: ${STREAMVLN_EXT_FLAGS:='{"enable_sliding_kv": false, "enable_memory_loss": false, "enable_voxel_proxy": false, "enable_token_selection": false, "enable_dynamic_memory": false, "enable_multiscale_memory": false, "enable_voxel_rgbd": false}'}
export STREAMVLN_EXT_FLAGS

cd /home/ubuntu/project/StreamVLN

VIDEO_FOLDER=${VIDEO_FOLDER:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R}
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/streamvln_offline_lora_sdpa_$(date +%Y%m%d_%H%M%S)}

NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-100}
USE_DEEPSPEED=${USE_DEEPSPEED:-0}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
VISION_LORA_LAST_N=${VISION_LORA_LAST_N:-2}

LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "StreamVLN Offline LoRA + SDPA Training"
echo "=========================================="
echo "model_path:        $MODEL_PATH"
echo "video_folder:      $VIDEO_FOLDER"
echo "output_dir:        $OUTPUT_DIR"
echo "num_epochs:        $NUM_EPOCHS"
echo "max_steps:         $MAX_STEPS"
echo "lora_target:       $LORA_TARGET_MODULES"
echo "vit_last_n_layers: $VISION_LORA_LAST_N"
echo "token_pruning:     $STREAMVLN_EXT_FLAGS"
echo "=========================================="

cmd=(
  python streamvln_ext/entrypoints/train_ext.py
  --model_name_or_path "$MODEL_PATH"
  --video_folder "$VIDEO_FOLDER"
  --version qwen_1_5
  --num_history 8
  --num_future_steps 4
  --num_frames 32
  --data_augmentation True
  --mm_tunable_parts "mm_lora_layer"
  --vision_tower "google/siglip-so400m-patch14-384"
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --image_aspect_ratio anyres_max_9
  --image_grid_pinpoints "(1x1),...,(6x6)"
  --bf16 True
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs "$NUM_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "$GRAD_ACC_STEPS"
  --evaluation_strategy no
  --save_strategy steps
  --save_steps 100
  --save_total_limit 2
  --learning_rate "$LEARNING_RATE"
  --weight_decay 0.
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 10
  --tf32 True
  --model_max_length 32768
  --gradient_checkpointing True
  --dataloader_num_workers 2
  --lazy_preprocess True
  --dataloader_drop_last True
  --report_to none
  --attn_implementation sdpa
  --lora_enable True
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_target_modules "$LORA_TARGET_MODULES"
  --lora_include_mm_projector True
  --lora_include_vision_tower True
  --lora_vision_tower_last_n_layers "$VISION_LORA_LAST_N"
)

if [[ "$MAX_STEPS" -gt 0 ]]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi

if [[ "$USE_DEEPSPEED" -eq 1 ]]; then
  cmd+=(--deepspeed scripts/zero2.json)
fi

start_ts=$(date +%s)
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
end_ts=$(date +%s)

echo "training_elapsed_seconds=$((end_ts - start_ts))" | tee "$OUTPUT_DIR/timing.txt"
echo "[done] training output: $OUTPUT_DIR"
