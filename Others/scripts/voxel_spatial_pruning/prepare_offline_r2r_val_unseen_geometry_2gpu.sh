#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

cd /home/ubuntu/project/StreamVLN

ROOT=${ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen}
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/model/Depth-Anything-V2-Metric-Indoor-Base-hf}
BATCH_SIZE=${BATCH_SIZE:-64}
DTYPE=${DTYPE:-float16}
POSE_MODE=${POSE_MODE:-odometry}
LOG_DIR=${LOG_DIR:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen/geometry_logs}

mkdir -p "$LOG_DIR"

for shard_id in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES=$shard_id
    python scripts_ext/prepare_r2r_depth_anything_geometry.py \
      --roots "$ROOT" \
      --model-path "$MODEL_PATH" \
      --batch-size "$BATCH_SIZE" \
      --dtype "$DTYPE" \
      --pose-mode "$POSE_MODE" \
      --num-shards 2 \
      --shard-id "$shard_id" \
      --summary-path "$LOG_DIR/summary_shard${shard_id}.json"
  ) > "$LOG_DIR/shard${shard_id}.log" 2>&1 &
  echo "[started] shard=${shard_id} gpu=${shard_id} log=$LOG_DIR/shard${shard_id}.log"
done

wait
echo "[done] summaries: $LOG_DIR/summary_shard0.json $LOG_DIR/summary_shard1.json"
