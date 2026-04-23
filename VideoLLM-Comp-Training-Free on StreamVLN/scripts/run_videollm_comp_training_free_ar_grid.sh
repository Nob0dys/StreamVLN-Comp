#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/ubuntu/project/StreamVLN

PYTHON_BIN=${PYTHON_BIN:-$(which python)}
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MODEL_PATH}
PHASE0_PROTOCOL_JSON=${PHASE0_PROTOCOL_JSON:-/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_phase0_runs/latest_suite/phase0_protocol_manifest.json}
AR_EPISODES=${AR_EPISODES:-80}
NUM_FRAMES=${NUM_FRAMES:-32}
NUM_HISTORY=${NUM_HISTORY:-8}
NUM_FUTURE_STEPS=${NUM_FUTURE_STEPS:-4}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SUITE_TS=${SUITE_TS:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_free_ar_runs/${SUITE_TS}}

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

AR_SUBSET_ROOT=$(read_json_field "$PHASE0_PROTOCOL_JSON" "['autoregressive_subset_root']")
AR_SUBSET_SIGNATURE=$(read_json_field "$PHASE0_PROTOCOL_JSON" "['autoregressive_subset_signature']")
AR_BASELINE_TOKENS=$(read_json_field "$PHASE0_PROTOCOL_JSON" "['autoregressive_avg_total_tokens_per_step']")

build_flags() {
  local method="$1"
  local keep_ratio="$2"
  local p1="${3:-}"
  local p2="${4:-}"
  local p3="${5:-}"
  "$PYTHON_BIN" - <<PY
import json

method = "${method}"
keep_ratio = float("${keep_ratio}")
p1 = "${p1}"
p2 = "${p2}"
p3 = "${p3}"

flags = {
    "enable_video_token_compressor": True,
    "video_token_compressor_type": method,
    "video_token_target_keep_ratio": keep_ratio,
    "enable_sliding_kv": False,
    "enable_memory_loss": False,
    "enable_voxel_proxy": False,
    "enable_token_selection": False,
    "enable_dynamic_memory": False,
    "enable_multiscale_memory": False,
    "enable_voxel_rgbd": False,
    "enable_hc_st_pruning": False,
    "enable_tuning_free_mm_pruning": False,
    "enable_tome_visual_merge": False,
}

if method == "visionzip":
    flags.update({
        "visionzip_dominant_num": int(p1),
        "visionzip_contextual_num": int(p2),
        "visionzip_score_type": "attn_proxy",
    })
elif method == "prunevid":
    cluster_ratio = float(p2)
    temporal_ratio = float(p3)
    flags["video_token_target_keep_ratio"] = max(0.05, min(1.0, cluster_ratio * max(temporal_ratio, 0.1)))
    flags.update({
        "prunevid_tau": float(p1),
        "prunevid_cluster_ratio": cluster_ratio,
        "prunevid_temporal_ratio": temporal_ratio,
        "prunevid_k": 7,
        "prunevid_min_tokens_for_cluster": 14,
    })
elif method == "dytok_static":
    flags.update({
        "dytok_static_base_compressor": "visionzip",
        "dytok_static_upper_limit_ratio": keep_ratio,
        "dytok_static_min_ratio": float(p1),
        "visionzip_score_type": "attn_proxy",
    })
elif method == "fastvid":
    flags.update({
        "fastvid_retention_ratio": keep_ratio,
        "fastvid_dyseg_c": 8,
        "fastvid_dyseg_tau": 0.9,
        "fastvid_stprune_d": 0.4,
        "fastvid_dtm_p": 4,
        "fastvid_dtm_beta": 0.6,
        "fastvid_score_type": "attn_proxy",
    })
elif method == "vqtoken":
    flags.update({
        "vqtoken_num_clusters": int(p1),
        "vqtoken_adaptive": False,
        "vqtoken_max_clusters": 64,
        "vqtoken_adaptive_method": "silhouette",
        "vqtoken_use_cross_attention": False,
    })
else:
    raise ValueError(f"unknown method: {method}")

print(json.dumps(flags))
PY
}

run_eval() {
  local method="$1"
  local experiment_name="$2"
  local keep_ratio="$3"
  local p1="${4:-}"
  local p2="${5:-}"
  local p3="${6:-}"
  local flags_json
  flags_json=$(build_flags "$method" "$keep_ratio" "$p1" "$p2" "$p3")

  local output_dir="$SUITE_ROOT/${experiment_name}_autoregressive"
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

  echo "${method},${experiment_name},${keep_ratio},${AR_SUBSET_SIGNATURE},${output_dir}/summary.json" >> "$SUITE_ROOT/run_index.csv"
  echo "[done] ${experiment_name}"
}

echo "method,experiment_name,keep_ratio,subset_signature,summary_path" > "$SUITE_ROOT/run_index.csv"

run_eval visionzip visionzip_keep_ratio_0_3 0.3 24 6
run_eval visionzip visionzip_keep_ratio_0_5 0.5 36 12
run_eval visionzip visionzip_keep_ratio_0_7 0.7 48 20

run_eval prunevid prunevid_tau_0_7_cluster_ratio_0_3_temporal_ratio_0_5 0.3 0.7 0.30 0.50
run_eval prunevid prunevid_tau_0_8_cluster_ratio_0_5_temporal_ratio_0_25 0.5 0.8 0.50 0.25
run_eval prunevid prunevid_tau_0_9_cluster_ratio_0_7_temporal_ratio_0_1 0.7 0.9 0.70 0.10

run_eval dytok_static dytok_static_visionzip_keep_ratio_0_3 0.3 0.2
run_eval dytok_static dytok_static_visionzip_keep_ratio_0_5 0.5 0.3
run_eval dytok_static dytok_static_visionzip_keep_ratio_0_7 0.7 0.5

run_eval fastvid fastvid_keep_ratio_0_3 0.3
run_eval fastvid fastvid_keep_ratio_0_5 0.5
run_eval fastvid fastvid_keep_ratio_0_7 0.7

run_eval vqtoken vqtoken_keep_ratio_0_3 0.3 59
run_eval vqtoken vqtoken_keep_ratio_0_5 0.5 98
run_eval vqtoken vqtoken_keep_ratio_0_7 0.7 137

echo "[done] suite_root=$SUITE_ROOT"
