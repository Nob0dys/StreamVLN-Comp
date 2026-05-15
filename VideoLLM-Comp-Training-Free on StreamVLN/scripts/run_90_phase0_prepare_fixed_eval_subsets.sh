#!/bin/zsh
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

cd /home/ubuntu/project/StreamVLN

SOURCE_DATASET_ROOT=${SOURCE_DATASET_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen}
TF_EPISODES=${TF_EPISODES:-240}
AR_EPISODES=${AR_EPISODES:-80}
TF_SEED=${TF_SEED:-24042}
AR_SEED=${AR_SEED:-80042}
LINK_MODE=${LINK_MODE:-symlink}
OVERWRITE=${OVERWRITE:-0}
SUITE_TS=${SUITE_TS:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_${SUITE_TS}}

mkdir -p "$(dirname "$SUITE_ROOT")"

if [[ -e "$SUITE_ROOT" ]]; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$SUITE_ROOT"
  else
    echo "[run_90] error: suite root already exists: $SUITE_ROOT"
    echo "[run_90] set OVERWRITE=1 to replace it."
    exit 1
  fi
fi

mkdir -p "$SUITE_ROOT"
ln -sfn "$SUITE_ROOT" /home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/latest_suite

export SOURCE_DATASET_ROOT TF_EPISODES AR_EPISODES TF_SEED AR_SEED LINK_MODE SUITE_ROOT

python - <<'PY'
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

source_root = Path(os.environ["SOURCE_DATASET_ROOT"])
suite_root = Path(os.environ["SUITE_ROOT"])
tf_episodes = int(os.environ["TF_EPISODES"])
ar_episodes = int(os.environ["AR_EPISODES"])
tf_seed = int(os.environ["TF_SEED"])
ar_seed = int(os.environ["AR_SEED"])
link_mode = os.environ["LINK_MODE"].strip().lower()

if link_mode not in {"symlink", "copy"}:
    raise ValueError(f"Unsupported LINK_MODE: {link_mode}")

anno_path = source_root / "annotations.json"
if not anno_path.is_file():
    raise FileNotFoundError(f"Missing annotations.json: {anno_path}")

with anno_path.open("r", encoding="utf-8") as f:
    items = json.load(f)

if not isinstance(items, list) or not items:
    raise ValueError("annotations.json must be a non-empty JSON list")


def sample_indices(count: int, seed: int):
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    return indices[: min(count, len(indices))]


def get_instruction_preview(item):
    value = item.get("instruction", item.get("instructions", ""))
    if isinstance(value, list):
        text = " ".join(str(x) for x in value)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text[:120]


def compute_signature(indices):
    joined = ",".join(str(i) for i in indices)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def materialize_subset(mode_name, count, seed):
    mode_dir = suite_root / f"{mode_name}_{count}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    indices = sample_indices(count, seed)
    selected = [items[i] for i in indices]

    records = []
    for idx, item in zip(indices, selected):
        records.append(
            {
                "dataset_index": idx,
                "annotation_id": item.get("id"),
                "video": item.get("video"),
                "instruction_preview": get_instruction_preview(item),
                "num_actions": len(item.get("actions", []) or []),
                "mode": mode_name,
            }
        )

    signature = compute_signature(indices)

    with (mode_dir / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False)

    src_images = source_root / "images"
    dst_images = mode_dir / "images"
    if dst_images.exists() or dst_images.is_symlink():
        if dst_images.is_dir() and not dst_images.is_symlink():
            shutil.rmtree(dst_images)
        else:
            dst_images.unlink()

    if link_mode == "copy":
        shutil.copytree(src_images, dst_images)
    else:
        os.symlink(src_images, dst_images)

    indices_path = suite_root / f"{mode_name}_{count}_indices.json"
    with indices_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode_name,
                "episodes": count,
                "seed": seed,
                "subset_signature": signature,
                "source_dataset_root": str(source_root),
                "indices": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "mode": mode_name,
        "episodes": count,
        "seed": seed,
        "subset_root": str(mode_dir),
        "indices_json": str(indices_path),
        "subset_signature": signature,
        "num_selected": len(selected),
    }


tf_meta = materialize_subset("teacher_forcing", tf_episodes, tf_seed)
ar_meta = materialize_subset("autoregressive", ar_episodes, ar_seed)

meta = {
    "source_dataset_root": str(source_root),
    "source_annotation_count": len(items),
    "link_mode": link_mode,
    "teacher_forcing": tf_meta,
    "autoregressive": ar_meta,
}

with (suite_root / "subset_meta.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

rows = [tf_meta, ar_meta]
csv_path = suite_root / "subset_manifest.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["mode", "episodes", "seed", "num_selected", "subset_root", "indices_json", "subset_signature"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

md_lines = []
md_lines.append("# Phase 0 Fixed Eval Subsets")
md_lines.append("")
md_lines.append(f"- source_dataset_root: `{source_root}`")
md_lines.append(f"- source_annotation_count: `{len(items)}`")
md_lines.append(f"- link_mode: `{link_mode}`")
md_lines.append("")
md_lines.append("| mode | episodes | seed | num_selected | subset_root | indices_json | subset_signature |")
md_lines.append("|---|---:|---:|---:|---|---|---|")
for row in rows:
    md_lines.append(
        f"| {row['mode']} | {row['episodes']} | {row['seed']} | {row['num_selected']} | "
        f"`{row['subset_root']}` | `{row['indices_json']}` | `{row['subset_signature']}` |"
    )

with (suite_root / "subset_manifest.md").open("w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

print(json.dumps(meta, indent=2, ensure_ascii=False))
PY

echo "[run_90] suite_root=$SUITE_ROOT"
echo "[run_90] manifest_md=$SUITE_ROOT/subset_manifest.md"
echo "[run_90] tf_indices=$SUITE_ROOT/teacher_forcing_${TF_EPISODES}_indices.json"
echo "[run_90] ar_indices=$SUITE_ROOT/autoregressive_${AR_EPISODES}_indices.json"
