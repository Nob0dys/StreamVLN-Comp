import argparse
import json
import os
import random
import shutil
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Create a lightweight R2R subset with linked trajectory folders.")
    parser.add_argument("--src", type=str, required=True, help="Source R2R root containing annotations.json")
    parser.add_argument("--dst", type=str, required=True, help="Destination subset root")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before sampling")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_rel_path(path: str) -> str:
    normalized = os.path.normpath(path)
    if normalized.startswith(".."):
        raise ValueError(f"Unsafe relative path: {path}")
    return normalized


def _load_annotations(src_root: str) -> List[Dict]:
    anno_path = os.path.join(src_root, "annotations.json")
    if not os.path.isfile(anno_path):
        raise FileNotFoundError(f"Missing annotations file: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not isinstance(annotations, list):
        raise ValueError("annotations.json must be a JSON list")
    return annotations


def _materialize_video_dirs(src_root: str, dst_root: str, rel_video_dirs: List[str], link_mode: str):
    for rel_dir in rel_video_dirs:
        rel_dir = _safe_rel_path(rel_dir)
        src_dir = os.path.join(src_root, rel_dir)
        dst_dir = os.path.join(dst_root, rel_dir)

        if os.path.exists(dst_dir):
            continue
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Missing source trajectory dir: {src_dir}")

        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)

        if link_mode == "copy":
            shutil.copytree(src_dir, dst_dir)
            continue

        try:
            os.symlink(src_dir, dst_dir)
        except OSError:
            # Fallback when symlink is unavailable.
            shutil.copytree(src_dir, dst_dir)


def main():
    args = parse_args()

    annotations = _load_annotations(args.src)
    if len(annotations) == 0:
        raise ValueError("No trajectory annotations found in source")

    if os.path.exists(args.dst) and args.overwrite:
        shutil.rmtree(args.dst)
    os.makedirs(args.dst, exist_ok=True)

    indices = list(range(len(annotations)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)

    keep_count = min(max(args.episodes, 1), len(indices))
    selected = [annotations[i] for i in indices[:keep_count]]

    rel_video_dirs = sorted({item["video"] for item in selected})
    _materialize_video_dirs(args.src, args.dst, rel_video_dirs, args.link_mode)

    with open(os.path.join(args.dst, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f)

    meta = {
        "src": args.src,
        "dst": args.dst,
        "num_total": len(annotations),
        "num_selected": keep_count,
        "num_video_dirs": len(rel_video_dirs),
        "seed": args.seed,
        "shuffle": args.shuffle,
        "link_mode": args.link_mode,
    }
    with open(os.path.join(args.dst, "subset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
