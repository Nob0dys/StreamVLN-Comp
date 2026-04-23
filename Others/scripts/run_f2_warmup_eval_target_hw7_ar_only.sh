#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/ubuntu/project/StreamVLN

PYTHON_BIN=${PYTHON_BIN:-$(which python)}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3}
STAGEF_ROOT=${STAGEF_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/token_pruning_stageF_runs/20260419_164637}
MODEL_PATH=${MODEL_PATH:-${STAGEF_ROOT}/checkpoints/combo_hcst_kr072_hw7_f2_warmup}
PHASE0_PROTOCOL_JSON=${PHASE0_PROTOCOL_JSON:-/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_phase0_runs/latest_suite/phase0_protocol_manifest.json}
AR_EPISODES=${AR_EPISODES:-80}
NUM_FRAMES=${NUM_FRAMES:-32}
NUM_HISTORY=${NUM_HISTORY:-8}
NUM_FUTURE_STEPS=${NUM_FUTURE_STEPS:-4}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MAIN_HISTORY_WINDOW=${MAIN_HISTORY_WINDOW:-7}
MIN_TOKENS=${MIN_TOKENS:-64}
KEEP_RATIOS=${KEEP_RATIOS:-"0.3 0.5 0.7"}
SUITE_TS=${SUITE_TS:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/f2_warmup_eval_target_hw7_ar_runs/${SUITE_TS}}

mkdir -p "$SUITE_ROOT"

read_json_field() {
  local path="$1"
  local field="$2"
  "$PYTHON_BIN" - <<PY
import json
with open("${path}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data${field})
PY
}

calc_text_keep_ratio() {
  local visual_ratio="$1"
  "$PYTHON_BIN" - <<PY
v = float("${visual_ratio}")
print(min(0.95, round(v + 0.10, 2)))
PY
}

build_flags() {
  local keep_ratio="$1"
  local text_keep_ratio="$2"
  "$PYTHON_BIN" - <<PY
import json
flags = {
    "enable_sliding_kv": False,
    "enable_memory_loss": False,
    "enable_voxel_proxy": False,
    "enable_token_selection": False,
    "enable_dynamic_memory": False,
    "enable_multiscale_memory": False,
    "enable_voxel_rgbd": False,
    "enable_hc_st_pruning": True,
    "hc_st_keep_ratio": float("${keep_ratio}"),
    "hc_st_min_tokens": int("${MIN_TOKENS}"),
    "hc_st_history_window": int("${MAIN_HISTORY_WINDOW}"),
    "hc_st_recent_boost": 1.2,
    "enable_tuning_free_mm_pruning": True,
    "mm_prune_visual_keep_ratio": float("${keep_ratio}"),
    "mm_prune_text_keep_ratio": float("${text_keep_ratio}"),
    "mm_prune_visual_score_type": "l2",
    "mm_prune_text_score_type": "l2",
    "mm_prune_keep_special_tokens": True,
    "enable_tome_visual_merge": False,
}
print(json.dumps(flags))
PY
}

AR_SUBSET_ROOT=$(read_json_field "$PHASE0_PROTOCOL_JSON" "['autoregressive_subset_root']")
AR_SUBSET_SIGNATURE=$(read_json_field "$PHASE0_PROTOCOL_JSON" "['autoregressive_subset_signature']")
AR_BASELINE_TOKENS=$(read_json_field "$PHASE0_PROTOCOL_JSON" "['autoregressive_avg_total_tokens_per_step']")

echo "method,experiment_name,keep_ratio,history_window,text_keep_ratio,subset_signature,summary_path" > "$SUITE_ROOT/run_index.csv"

for keep_ratio in ${=KEEP_RATIOS}; do
  kr_tag="${keep_ratio//./_}"
  experiment_name="f2_warmup_eval_target_hw7_keep_ratio_${kr_tag}"
  text_keep_ratio=$(calc_text_keep_ratio "$keep_ratio")
  flags_json=$(build_flags "$keep_ratio" "$text_keep_ratio")
  output_dir="$SUITE_ROOT/${experiment_name}_autoregressive"
  mkdir -p "$output_dir"

  env MODEL_PATH="$MODEL_PATH" \
      BASE_MODEL_PATH="$BASE_MODEL_PATH" \
      DATASET_ROOT="$AR_SUBSET_ROOT" \
      OUTPUT_DIR="$output_dir" \
      EVAL_MODE=autoregressive \
      MAX_EPISODES="$AR_EPISODES" \
      NUM_FRAMES="$NUM_FRAMES" \
      NUM_HISTORY="$NUM_HISTORY" \
      NUM_FUTURE_STEPS="$NUM_FUTURE_STEPS" \
      MODEL_MAX_LENGTH="$MODEL_MAX_LENGTH" \
      BASELINE_AVG_TOTAL_TOKENS="$AR_BASELINE_TOKENS" \
      STREAMVLN_EXT_FLAGS="$flags_json" \
      zsh /home/ubuntu/project/StreamVLN/test_r2r_offline_action_acc.sh > "$output_dir/runner.log" 2>&1

  echo "f2_warmup_eval_target_hw7,${experiment_name},${keep_ratio},${MAIN_HISTORY_WINDOW},${text_keep_ratio},${AR_SUBSET_SIGNATURE},${output_dir}/summary.json" >> "$SUITE_ROOT/run_index.csv"
  echo "[done] ${experiment_name}"
done

echo "[done] suite_root=$SUITE_ROOT"
