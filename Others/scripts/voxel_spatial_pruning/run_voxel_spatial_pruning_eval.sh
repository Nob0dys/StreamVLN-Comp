#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/ubuntu/project/StreamVLN

MODEL_PATH=${MODEL_PATH:-/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3}
DATASET_ROOT=${DATASET_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen}
OUTPUT_DIR=${OUTPUT_DIR:-./experiments_ext/results/voxel_spatial_pruning_$(date +%Y%m%d_%H%M%S)}
EVAL_MODE=${EVAL_MODE:-teacher_forcing}
MAX_EPISODES=${MAX_EPISODES:-0}
NUM_HISTORY=${NUM_HISTORY:-8}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}

VOXEL_SIZE=${VOXEL_SIZE:-0.25}
VOXEL_STRIDE_K=${VOXEL_STRIDE_K:-4}
VOXEL_FRAME_THRESHOLD=${VOXEL_FRAME_THRESHOLD:-0.05}
VOXEL_MIN_DEPTH=${VOXEL_MIN_DEPTH:-0.05}
VOXEL_MAX_DEPTH=${VOXEL_MAX_DEPTH:-10.0}

# Offline eval first loads saved per-frame depth/pose/intrinsic from each
# episode's depth/, pose/, and intrinsic/ directories. OFFLINE_GEOM_MODE only
# controls the fallback path for frames whose saved geometry is absent.
OFFLINE_GEOM_MODE=${OFFLINE_GEOM_MODE:-odometry}
OFFLINE_UNIT_DEPTH_M=${OFFLINE_UNIT_DEPTH_M:-2.0}
OFFLINE_HFOV_DEG=${OFFLINE_HFOV_DEG:-75.17817894}

export STREAMVLN_EXT_FLAGS=$(python - <<PY
import json
print(json.dumps({
    "enable_sliding_kv": False,
    "enable_memory_loss": False,
    "enable_voxel_proxy": False,
    "enable_token_selection": False,
    "enable_dynamic_memory": False,
    "enable_multiscale_memory": False,
    "enable_voxel_rgbd": False,
    "enable_video_token_compressor": False,
    "enable_voxel_spatial_pruning": True,
    "voxel_spatial_size": float("${VOXEL_SIZE}"),
    "voxel_spatial_stride_k": int("${VOXEL_STRIDE_K}"),
    "voxel_spatial_frame_threshold": float("${VOXEL_FRAME_THRESHOLD}"),
    "voxel_spatial_min_depth": float("${VOXEL_MIN_DEPTH}"),
    "voxel_spatial_max_depth": float("${VOXEL_MAX_DEPTH}"),
    "voxel_spatial_offline_geom_mode": "${OFFLINE_GEOM_MODE}",
    "voxel_spatial_offline_unit_depth_m": float("${OFFLINE_UNIT_DEPTH_M}"),
    "voxel_spatial_offline_hfov_deg": float("${OFFLINE_HFOV_DEG}"),
}))
PY
)

export MODEL_PATH
export DATASET_ROOT
export OUTPUT_DIR
export EVAL_MODE
export MAX_EPISODES
export NUM_HISTORY
export MODEL_MAX_LENGTH

echo "[voxel-spatial] flags: ${STREAMVLN_EXT_FLAGS}"
zsh test_r2r_offline_action_acc.sh
