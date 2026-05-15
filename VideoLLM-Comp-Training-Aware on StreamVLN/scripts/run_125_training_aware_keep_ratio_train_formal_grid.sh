#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/ubuntu/project/StreamVLN

SUITE_TS=${SUITE_TS:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_keep_ratio_formal_runs/${SUITE_TS}}
TRAIN_SUBSET_ROOT=${TRAIN_SUBSET_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_fixed_train_subsets/latest_suite/train_fixed_r2r_5k}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3}
DINO_PATH=${DINO_PATH:-/home/ubuntu/model/dinov2-base}
BERT_PATH=${BERT_PATH:-/home/ubuntu/model/bert-base-uncased}
MAX_STEPS=${MAX_STEPS:-1000}
NUM_EPOCHS=${NUM_EPOCHS:-1}
USE_DEEPSPEED=${USE_DEEPSPEED:-0}
FAIL_FAST=${FAIL_FAST:-1}
REUSE_IF_EXISTS=${REUSE_IF_EXISTS:-1}
METHODS=${METHODS:-"llamavid longvu"}
KEEP_RATIOS=${KEEP_RATIOS:-"0.3 0.5 0.7"}
TRAIN_SEED=${TRAIN_SEED:-12042}
BASELINE_TOKENS_PER_FRAME=${BASELINE_TOKENS_PER_FRAME:-169}

mkdir -p /home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_keep_ratio_formal_runs
mkdir -p "$SUITE_ROOT/checkpoints"
ln -sfn "$SUITE_ROOT" /home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_keep_ratio_formal_runs/latest_suite

RUN_INDEX="$SUITE_ROOT/run_index.csv"
echo "method,candidate,target_keep_ratio,actual_tokens_per_frame,actual_keep_ratio,phase,status,exit_code,path,flags_path,trainer_state_path" > "$RUN_INDEX"

ratio_tag() {
  print -r -- "${1//./_}"
}

llamavid_grid_side_for_keep_ratio() {
  case "$1" in
    0.3) echo "7" ;;
    0.5) echo "9" ;;
    0.7) echo "11" ;;
    *)
      echo "[error] unsupported keep_ratio for llamavid: $1" >&2
      return 1
      ;;
  esac
}

longvu_query_num_for_keep_ratio() {
  case "$1" in
    0.3) echo "49" ;;
    0.5) echo "81" ;;
    0.7) echo "121" ;;
    *)
      echo "[error] unsupported keep_ratio for longvu: $1" >&2
      return 1
      ;;
  esac
}

actual_tokens_per_frame_for() {
  local method="$1"
  local keep_ratio="$2"
  case "$method" in
    llamavid)
      local grid_side
      grid_side=$(llamavid_grid_side_for_keep_ratio "$keep_ratio")
      echo $((grid_side * grid_side + 1))
      ;;
    longvu)
      longvu_query_num_for_keep_ratio "$keep_ratio"
      ;;
    *)
      echo "[error] unsupported method: $method" >&2
      return 1
      ;;
  esac
}

actual_keep_ratio_from_tokens() {
  local tokens="$1"
  python - <<PY
tokens=float("${tokens}")
baseline=float("${BASELINE_TOKENS_PER_FRAME}")
print(f"{tokens / baseline:.6f}")
PY
}

candidate_for() {
  local method="$1"
  local keep_ratio="$2"
  local tag
  tag=$(ratio_tag "$keep_ratio")
  case "$method" in
    llamavid)
      local grid_side
      grid_side=$(llamavid_grid_side_for_keep_ratio "$keep_ratio")
      echo "llamavid_keep_ratio_${tag}_q32_grid${grid_side}_r2r5k"
      ;;
    longvu)
      local query_num
      query_num=$(longvu_query_num_for_keep_ratio "$keep_ratio")
      echo "longvu_keep_ratio_${tag}_q${query_num}_d1_r2r5k"
      ;;
    *)
      echo "[error] unsupported method: $method" >&2
      return 1
      ;;
  esac
}

flags_for() {
  local method="$1"
  local keep_ratio="$2"
  case "$method" in
    llamavid)
      local grid_side
      grid_side=$(llamavid_grid_side_for_keep_ratio "$keep_ratio")
      print -r -- "{\"enable_training_aware_video_compressor\": true, \"training_aware_video_compressor_type\": \"llamavid\", \"training_aware_compress_memory\": true, \"video_token_target_keep_ratio\": ${keep_ratio}, \"llamavid_bert_model_name\": \"$BERT_PATH\", \"llamavid_num_query\": 32, \"llamavid_compress_type\": \"grid:${grid_side}\", \"llamavid_qformer_hidden_size\": 768, \"llamavid_qformer_depth\": 2, \"llamavid_num_heads\": 8, \"enable_video_token_compressor\": false}"
      ;;
    longvu)
      local query_num
      query_num=$(longvu_query_num_for_keep_ratio "$keep_ratio")
      print -r -- "{\"enable_training_aware_video_compressor\": true, \"training_aware_video_compressor_type\": \"longvu\", \"training_aware_compress_memory\": true, \"video_token_target_keep_ratio\": ${keep_ratio}, \"longvu_dino_tower\": \"$DINO_PATH\", \"longvu_use_dino\": true, \"longvu_select_current_frames\": false, \"longvu_query_num\": ${query_num}, \"longvu_connector_depth\": 1, \"longvu_window_size\": 8, \"longvu_threshold\": 0.83, \"longvu_vision_hidden_size\": 1024, \"longvu_num_heads\": 8, \"enable_video_token_compressor\": false}"
      ;;
    *)
      echo "[error] unsupported method: $method" >&2
      return 1
      ;;
  esac
}

lr_for_method() {
  case "$1" in
    longvu) echo "5e-6" ;;
    llamavid) echo "2e-5" ;;
    *)
      echo "[error] unsupported method: $1" >&2
      return 1
      ;;
  esac
}

grad_acc_for_method() {
  case "$1" in
    longvu) echo "1" ;;
    llamavid) echo "4" ;;
    *)
      echo "[error] unsupported method: $1" >&2
      return 1
      ;;
  esac
}

model_max_length_for_method() {
  case "$1" in
    longvu) echo "8192" ;;
    llamavid) echo "4096" ;;
    *)
      echo "[error] unsupported method: $1" >&2
      return 1
      ;;
  esac
}

run_candidate() {
  local method="$1"
  local keep_ratio="$2"
  local candidate
  local flags
  local actual_tokens
  local actual_keep_ratio
  local train_dir
  local flags_path
  local trainer_state_path
  local lr
  local grad_acc
  local model_max_length

  candidate=$(candidate_for "$method" "$keep_ratio")
  flags=$(flags_for "$method" "$keep_ratio")
  actual_tokens=$(actual_tokens_per_frame_for "$method" "$keep_ratio")
  actual_keep_ratio=$(actual_keep_ratio_from_tokens "$actual_tokens")
  train_dir="$SUITE_ROOT/checkpoints/$candidate"
  flags_path="$train_dir/flags.json"
  trainer_state_path="$train_dir/trainer_state.json"
  lr=$(lr_for_method "$method")
  grad_acc=$(grad_acc_for_method "$method")
  model_max_length=$(model_max_length_for_method "$method")

  mkdir -p "$train_dir"
  print -r -- "$flags" > "$flags_path"

  if [[ "$REUSE_IF_EXISTS" == "1" && -f "$train_dir/training_aware_video_compressor.bin" && -f "$trainer_state_path" ]]; then
    echo "[run_125] reuse candidate=$candidate"
    echo "${method},${candidate},${keep_ratio},${actual_tokens},${actual_keep_ratio},train,reused,0,${train_dir},${flags_path},${trainer_state_path}" >> "$RUN_INDEX"
    return
  fi

  ds_args=()
  if [[ "$USE_DEEPSPEED" == "1" ]]; then
    ds_args=(--deepspeed scripts/zero2.json)
  fi

  echo "[run_125] train method=$method candidate=$candidate keep_ratio=$keep_ratio actual_keep_ratio=$actual_keep_ratio lr=$lr grad_acc=$grad_acc max_steps=$MAX_STEPS"
  set +e
  env STREAMVLN_EXT_FLAGS="$flags" \
    python streamvln_ext/entrypoints/train_ext.py \
      --model_name_or_path "$BASE_MODEL_PATH" \
      --video_folder "$TRAIN_SUBSET_ROOT" \
      "${ds_args[@]}" \
      --version qwen_1_5 \
      --num_history 8 \
      --num_future_steps 4 \
      --num_frames 32 \
      --data_augmentation False \
      --mm_tunable_parts "mm_video_token_compressor" \
      --vision_tower "google/siglip-so400m-patch14-384" \
      --mm_projector_type mlp2x_gelu \
      --mm_vision_select_layer -2 \
      --mm_use_im_start_end False \
      --mm_use_im_patch_token False \
      --image_aspect_ratio anyres_max_9 \
      --image_grid_pinpoints "(1x1),...,(6x6)" \
      --bf16 True \
      --output_dir "$train_dir" \
      --num_train_epochs "$NUM_EPOCHS" \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps "$grad_acc" \
      --evaluation_strategy no \
      --save_strategy no \
      --learning_rate "$lr" \
      --weight_decay 0.0 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 10 \
      --seed "$TRAIN_SEED" \
      --data_seed "$TRAIN_SEED" \
      --tf32 True \
      --model_max_length "$model_max_length" \
      --gradient_checkpointing True \
      --dataloader_num_workers 2 \
      --lazy_preprocess True \
      --dataloader_drop_last True \
      --max_steps "$MAX_STEPS" \
      --report_to none \
      --attn_implementation sdpa \
      2>&1 | tee "$train_dir/train.log"
  local train_code=${pipestatus[1]}
  set -e

  local train_status="ok"
  if [[ "$train_code" -ne 0 ]]; then
    train_status="failed"
  fi
  echo "${method},${candidate},${keep_ratio},${actual_tokens},${actual_keep_ratio},train,${train_status},${train_code},${train_dir},${flags_path},${trainer_state_path}" >> "$RUN_INDEX"
  if [[ "$train_code" -ne 0 && "$FAIL_FAST" == "1" ]]; then
    exit "$train_code"
  fi
}

echo "[run_125] suite_root=$SUITE_ROOT"
echo "[run_125] train_subset=$TRAIN_SUBSET_ROOT"
echo "[run_125] methods=$METHODS"
echo "[run_125] keep_ratios=$KEEP_RATIOS"

for method in ${=METHODS}; do
  for keep_ratio in ${=KEEP_RATIOS}; do
    run_candidate "$method" "$keep_ratio"
  done
done

export SUITE_ROOT
python - <<'PY'
import csv
import json
import os

suite_root = os.environ["SUITE_ROOT"]
index_path = os.path.join(suite_root, "run_index.csv")
rows = list(csv.DictReader(open(index_path, "r", encoding="utf-8")))
train_rows = [r for r in rows if r["phase"] == "train" and r["status"] in {"ok", "reused"}]

overview = []
for row in train_rows:
    trainer_state_path = row.get("trainer_state_path", "")
    if not trainer_state_path or not os.path.isfile(trainer_state_path):
        continue
    trainer_state = json.load(open(trainer_state_path, "r", encoding="utf-8"))
    last_log = trainer_state.get("log_history", [])[-1] if trainer_state.get("log_history") else {}
    overview.append(
        {
            "method": row["method"],
            "candidate": row["candidate"],
            "target_keep_ratio": float(row["target_keep_ratio"]),
            "actual_tokens_per_frame": int(float(row["actual_tokens_per_frame"])),
            "actual_keep_ratio": float(row["actual_keep_ratio"]),
            "status": row["status"],
            "train_loss": float(last_log.get("train_loss", 0.0)),
            "train_runtime_sec": float(last_log.get("train_runtime", 0.0)),
            "train_steps_per_second": float(last_log.get("train_steps_per_second", 0.0)),
            "path": row["path"],
        }
    )

overview.sort(key=lambda x: (x["method"], x["target_keep_ratio"]))

csv_path = os.path.join(suite_root, "training_aware_keep_ratio_train_overview.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "method",
        "candidate",
        "target_keep_ratio",
        "actual_tokens_per_frame",
        "actual_keep_ratio",
        "status",
        "train_loss",
        "train_runtime_sec",
        "train_steps_per_second",
        "path",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in overview:
        writer.writerow(row)

md_path = os.path.join(suite_root, "training_aware_keep_ratio_train_overview.md")
lines = [
    "# Training-Aware Keep-Ratio Train Overview",
    "",
    f"- suite_root: `{suite_root}`",
    "",
    "| method | keep_ratio | actual_keep_ratio | actual_tokens_per_frame | candidate | status | train_loss | runtime_sec | steps_per_sec |",
    "|---|---:|---:|---:|---|---|---:|---:|---:|",
]
for row in overview:
    lines.append(
        f"| `{row['method']}` | {row['target_keep_ratio']:.1f} | {row['actual_keep_ratio']:.4f} | "
        f"{row['actual_tokens_per_frame']} | `{row['candidate']}` | `{row['status']}` | "
        f"{row['train_loss']:.4f} | {row['train_runtime_sec']:.2f} | {row['train_steps_per_second']:.3f} |"
    )
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"[run_125] overview={md_path}")
PY

echo "[run_125] suite_root=$SUITE_ROOT"
