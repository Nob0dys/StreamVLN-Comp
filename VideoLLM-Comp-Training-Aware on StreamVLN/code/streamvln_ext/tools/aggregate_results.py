import argparse
import csv
import json
import os
from glob import glob
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate StreamVLN experiment result.json files.")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


def _safe_json_loads(line: str) -> Optional[Dict]:
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _parse_result_file(path: str) -> Optional[Dict]:
    per_episode: List[Dict] = []
    summary: Optional[Dict] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            data = _safe_json_loads(raw)
            if data is None:
                continue

            if {"sucs_all", "spls_all", "oss_all", "ones_all"}.issubset(data.keys()):
                summary = data
            elif {"success", "spl", "os", "ne"}.issubset(data.keys()):
                per_episode.append(data)

    if summary is not None:
        return {
            "sr": float(summary.get("sucs_all", 0.0)),
            "spl": float(summary.get("spls_all", 0.0)),
            "os": float(summary.get("oss_all", 0.0)),
            "ne": float(summary.get("ones_all", 0.0)),
            "episodes": int(summary.get("length", len(per_episode))),
        }

    if len(per_episode) == 0:
        return None

    sr = sum(x["success"] for x in per_episode) / len(per_episode)
    spl = sum(x["spl"] for x in per_episode) / len(per_episode)
    os_ = sum(x["os"] for x in per_episode) / len(per_episode)
    ne = sum(x["ne"] for x in per_episode) / len(per_episode)

    return {
        "sr": float(sr),
        "spl": float(spl),
        "os": float(os_),
        "ne": float(ne),
        "episodes": len(per_episode),
    }


def main():
    args = parse_args()

    result_files = glob(os.path.join(args.results_root, "**", "result.json"), recursive=True)
    rows = []

    for result_file in sorted(result_files):
        metrics = _parse_result_file(result_file)
        if metrics is None:
            continue

        exp_dir = os.path.dirname(result_file)
        exp_id = os.path.relpath(exp_dir, args.results_root)
        rows.append(
            {
                "exp_id": exp_id,
                "result_file": result_file,
                "episodes": metrics["episodes"],
                "sr": metrics["sr"],
                "spl": metrics["spl"],
                "os": metrics["os"],
                "ne": metrics["ne"],
            }
        )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = ["exp_id", "result_file", "episodes", "sr", "spl", "os", "ne"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Aggregated {len(rows)} experiments -> {args.out_csv}")


if __name__ == "__main__":
    main()
