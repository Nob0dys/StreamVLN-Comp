#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}

cd /home/ubuntu/project/StreamVLN

SUITE_TS=${SUITE_TS:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_fixed_train_subsets/${SUITE_TS}}
R2R_ROOT=${R2R_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R}
RXR_ROOT=${RXR_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/RxR}
ENVDROP_ROOT=${ENVDROP_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/EnvDrop}
R2R_SAMPLES=${R2R_SAMPLES:-5000}
MIX_SAMPLES=${MIX_SAMPLES:-20000}
R2R_SEED=${R2R_SEED:-12042}
MIX_SEED=${MIX_SEED:-12142}
LINK_MODE=${LINK_MODE:-symlink}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "$SUITE_ROOT"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

python streamvln_ext/tools/create_training_aware_fixed_train_subsets.py \
  --r2r-root "$R2R_ROOT" \
  --rxr-root "$RXR_ROOT" \
  --envdrop-root "$ENVDROP_ROOT" \
  --dst-root "$SUITE_ROOT" \
  --r2r-samples "$R2R_SAMPLES" \
  --mix-samples "$MIX_SAMPLES" \
  --r2r-seed "$R2R_SEED" \
  --mix-seed "$MIX_SEED" \
  --link-mode "$LINK_MODE" \
  "${overwrite_args[@]}"

ln -sfn "$SUITE_ROOT" /home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_fixed_train_subsets/latest_suite

echo "[run_120] suite_root=$SUITE_ROOT"
echo "[run_120] manifest=$SUITE_ROOT/subset_manifest.md"
