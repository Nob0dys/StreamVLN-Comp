import argparse
import csv
import copy
import hashlib
import json
import os
import random
import shutil
from typing import Dict, Iterable, List, Sequence, Tuple


SourceSpec = Tuple[str, str]


def parse_args():
    parser = argparse.ArgumentParser(description="Create fixed offline train subsets for training-aware StreamVLN experiments.")
    parser.add_argument("--r2r-root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R")
    parser.add_argument("--rxr-root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/RxR")
    parser.add_argument("--envdrop-root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/EnvDrop")
    parser.add_argument("--dst-root", type=str, required=True)
    parser.add_argument("--r2r-samples", type=int, default=5000)
    parser.add_argument("--mix-samples", type=int, default=20000)
    parser.add_argument("--r2r-seed", type=int, default=12042)
    parser.add_argument("--mix-seed", type=int, default=12142)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_rel_path(path: str) -> str:
    normalized = os.path.normpath(path)
    if os.path.isabs(normalized) or normalized in {"", ".", ".."} or normalized.startswith(f"..{os.sep}"):
        raise ValueError(f"Unsafe relative path: {path}")
    return normalized


def _load_annotations(src_root: str) -> List[Dict]:
    anno_path = os.path.join(src_root, "annotations.json")
    if not os.path.isfile(anno_path):
        raise FileNotFoundError(f"Missing annotations file: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    if not isinstance(annotations, list):
        raise ValueError(f"annotations.json must be a list: {anno_path}")
    return annotations


def _select_indices(indices: Sequence[int], count: int, seed: int) -> List[int]:
    if not indices:
        return []
    indices = list(indices)
    random.Random(seed).shuffle(indices)
    return indices[: min(max(count, 0), len(indices))]


def _valid_annotation_indices(src_root: str, annotations: Sequence[Dict]) -> List[int]:
    valid = []
    for idx, item in enumerate(annotations):
        video = item.get("video")
        if not video:
            continue
        try:
            rel_video = _safe_rel_path(video)
        except ValueError:
            continue
        if os.path.exists(os.path.join(src_root, rel_video)):
            valid.append(idx)
    return valid


def _signature(entries: Sequence[Dict]) -> str:
    payload = "\n".join(f"{entry['source']}:{entry['dataset_index']}" for entry in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _instruction_preview(item: Dict, max_chars: int = 120) -> str:
    instructions = item.get("instructions") or []
    if isinstance(instructions, list) and instructions:
        return str(instructions[0])[:max_chars]
    return ""


def _materialize_video_dirs(
    selected: Sequence[Tuple[str, str, Dict]],
    source_roots: Dict[str, str],
    dst_root: str,
    link_mode: str,
):
    linked = set()
    for source_name, _, item in selected:
        rel_dir = _safe_rel_path(item["video"])
        if rel_dir in linked:
            continue
        linked.add(rel_dir)

        src_dir = os.path.join(source_roots[source_name], rel_dir)
        dst_dir = os.path.join(dst_root, rel_dir)
        if os.path.lexists(dst_dir):
            continue
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Missing source trajectory dir: {src_dir}")

        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        if link_mode == "copy":
            shutil.copytree(src_dir, dst_dir)
        else:
            try:
                os.symlink(src_dir, dst_dir)
            except OSError:
                shutil.copytree(src_dir, dst_dir)


def _write_subset(
    subset_name: str,
    selected: Sequence[Tuple[str, str, Dict]],
    source_roots: Dict[str, str],
    dst_root: str,
    seed: int,
    requested: int,
    link_mode: str,
) -> Dict:
    subset_root = os.path.join(dst_root, subset_name)
    os.makedirs(subset_root, exist_ok=True)

    _materialize_video_dirs(selected, source_roots, subset_root, link_mode)

    annotations = [copy.deepcopy(item) for _, _, item in selected]
    with open(os.path.join(subset_root, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False)

    index_entries = []
    for source_name, dataset_index, item in selected:
        index_entries.append(
            {
                "source": source_name,
                "dataset_index": int(dataset_index),
                "annotation_id": item.get("id"),
                "video": item.get("video", ""),
                "instruction_preview": _instruction_preview(item),
                "num_actions": len(item.get("actions") or []),
            }
        )

    subset_signature = _signature(index_entries)
    indices_payload = {
        "subset_name": subset_name,
        "requested": requested,
        "num_selected": len(index_entries),
        "seed": seed,
        "subset_signature": subset_signature,
        "link_mode": link_mode,
        "subset_root": subset_root,
        "indices": index_entries,
    }
    indices_json = os.path.join(dst_root, f"{subset_name}_indices.json")
    with open(indices_json, "w", encoding="utf-8") as f:
        json.dump(indices_payload, f, ensure_ascii=False, indent=2)

    meta = {
        "subset_name": subset_name,
        "requested": requested,
        "num_selected": len(index_entries),
        "seed": seed,
        "subset_signature": subset_signature,
        "subset_root": subset_root,
        "indices_json": indices_json,
        "sources": sorted({entry["source"] for entry in index_entries}),
        "link_mode": link_mode,
    }
    with open(os.path.join(subset_root, "subset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _mix_targets(total: int) -> Dict[str, int]:
    r2r = total // 4
    rxr = total // 4
    envdrop = total - r2r - rxr
    return {"r2r": r2r, "rxr": rxr, "envdrop": envdrop}


def _selected_from_source(
    source_name: str,
    annotations: List[Dict],
    valid_indices: Sequence[int],
    count: int,
    seed: int,
) -> List[Tuple[str, str, Dict]]:
    return [(source_name, str(idx), annotations[idx]) for idx in _select_indices(valid_indices, count, seed)]


def _build_mix_selected(
    annotations: Dict[str, List[Dict]],
    valid_indices: Dict[str, Sequence[int]],
    requested_total: int,
    seed: int,
) -> List[Tuple[str, str, Dict]]:
    targets = _mix_targets(requested_total)
    selected: List[Tuple[str, str, Dict]] = []
    used = {source_name: set() for source_name in annotations}

    for offset, source_name in enumerate(["r2r", "rxr", "envdrop"]):
        indices = _select_indices(valid_indices[source_name], targets[source_name], seed + offset)
        used[source_name].update(indices)
        selected.extend((source_name, str(idx), annotations[source_name][idx]) for idx in indices)

    deficit = max(0, requested_total - len(selected))
    if deficit == 0:
        return selected

    for offset, source_name in enumerate(["rxr", "r2r", "envdrop"]):
        if deficit <= 0:
            break
        remaining = [idx for idx in valid_indices[source_name] if idx not in used[source_name]]
        extra = _select_indices(remaining, deficit, seed + 100 + offset)
        used[source_name].update(extra)
        selected.extend((source_name, str(idx), annotations[source_name][idx]) for idx in extra)
        deficit = max(0, requested_total - len(selected))

    return selected


def _write_manifest(dst_root: str, metas: Sequence[Dict], source_counts: Dict[str, int], args):
    manifest = {
        "dst_root": dst_root,
        "source_counts": source_counts,
        "r2r_samples": args.r2r_samples,
        "mix_samples": args.mix_samples,
        "r2r_seed": args.r2r_seed,
        "mix_seed": args.mix_seed,
        "mix_ratio": "r2r:rxr:envdrop=1:1:2",
        "link_mode": args.link_mode,
        "subsets": list(metas),
    }
    with open(os.path.join(dst_root, "subset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(dst_root, "subset_manifest.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset_name",
                "requested",
                "num_selected",
                "seed",
                "subset_signature",
                "subset_root",
                "indices_json",
                "sources",
                "link_mode",
            ],
        )
        writer.writeheader()
        for meta in metas:
            row = dict(meta)
            row["sources"] = ",".join(meta["sources"])
            writer.writerow(row)

    lines = [
        "# Training-Aware Fixed Train Subsets",
        "",
        f"- root: `{dst_root}`",
        f"- link_mode: `{args.link_mode}`",
        f"- mix_ratio: `r2r:rxr:envdrop=1:1:2`",
        "",
        "| subset | requested | selected | seed | signature | sources |",
        "|---|---:|---:|---:|---|---|",
    ]
    for meta in metas:
        lines.append(
            f"| `{meta['subset_name']}` | {meta['requested']} | {meta['num_selected']} | "
            f"{meta['seed']} | `{meta['subset_signature']}` | {', '.join(meta['sources'])} |"
        )
    lines.append("")
    lines.append("## Source Counts")
    lines.append("")
    for source_name, count in sorted(source_counts.items()):
        lines.append(f"- `{source_name}`: `{json.dumps(count, ensure_ascii=False)}`")
    with open(os.path.join(dst_root, "subset_manifest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if os.path.exists(args.dst_root) and args.overwrite:
        shutil.rmtree(args.dst_root)
    os.makedirs(args.dst_root, exist_ok=True)

    source_roots = {
        "r2r": args.r2r_root,
        "rxr": args.rxr_root,
        "envdrop": args.envdrop_root,
    }
    annotations = {name: _load_annotations(root) for name, root in source_roots.items()}
    source_counts = {name: len(data) for name, data in annotations.items()}
    valid_indices = {
        name: _valid_annotation_indices(source_roots[name], data)
        for name, data in annotations.items()
    }
    valid_source_counts = {name: len(indices) for name, indices in valid_indices.items()}

    r2r_selected = _selected_from_source(
        "r2r",
        annotations["r2r"],
        valid_indices["r2r"],
        args.r2r_samples,
        args.r2r_seed,
    )
    mix_selected = _build_mix_selected(annotations, valid_indices, args.mix_samples, args.mix_seed)

    metas = [
        _write_subset(
            "train_fixed_r2r_5k",
            r2r_selected,
            source_roots,
            args.dst_root,
            args.r2r_seed,
            args.r2r_samples,
            args.link_mode,
        ),
        _write_subset(
            "train_fixed_mix_20k",
            mix_selected,
            source_roots,
            args.dst_root,
            args.mix_seed,
            args.mix_samples,
            args.link_mode,
        ),
    ]
    source_counts_for_manifest = {
        name: {
            "annotations": source_counts[name],
            "valid_video_dirs": valid_source_counts[name],
            "missing_or_invalid": source_counts[name] - valid_source_counts[name],
        }
        for name in sorted(source_counts)
    }
    _write_manifest(args.dst_root, metas, source_counts_for_manifest, args)
    print(json.dumps({"dst_root": args.dst_root, "subsets": metas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
