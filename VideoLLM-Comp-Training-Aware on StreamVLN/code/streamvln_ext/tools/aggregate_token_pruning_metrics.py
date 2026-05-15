import argparse
import csv
import json
import os
from typing import Dict, List


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def _metric_row(candidate: str, mode: str, summary: Dict, summary_path: str) -> Dict:
    return {
        "candidate": candidate,
        "mode": mode,
        "summary_path": summary_path,
        "overall_action_acc": _safe_float(summary.get("overall_action_acc", 0.0)),
        "avg_total_tokens_per_step": _safe_float(summary.get("avg_total_tokens_per_step", 0.0)),
        "fps": _safe_float(summary.get("fps", 0.0)),
        "latency_ms_p50": _safe_float(summary.get("latency_ms_p50", 0.0)),
        "approx_tflops_per_step": _safe_float(summary.get("approx_tflops_per_step", 0.0)),
        "gpu_peak_allocated_mib": _safe_float(summary.get("gpu_peak_allocated_mib", 0.0)),
        "elapsed_seconds": _safe_float(summary.get("elapsed_seconds", 0.0)),
        "num_episodes_eval": _safe_int(summary.get("num_episodes_eval", 0)),
    }


def _build_mode_baselines(rows: List[Dict]) -> Dict[str, Dict]:
    baselines: Dict[str, Dict] = {}
    for row in rows:
        if row["candidate"] == "baseline":
            baselines[row["mode"]] = row
    return baselines


def _add_relative_metrics(rows: List[Dict], baselines: Dict[str, Dict]) -> List[Dict]:
    out = []
    for row in rows:
        base = baselines.get(row["mode"], None)
        if base is None:
            row["token_reduction_vs_mode_baseline"] = 0.0
            row["fps_gain_vs_mode_baseline"] = 0.0
            row["latency_p50_reduction_vs_mode_baseline"] = 0.0
            row["tflops_reduction_vs_mode_baseline"] = 0.0
            row["acc_drop_vs_mode_baseline"] = 0.0
            row["score"] = 0.0
            out.append(row)
            continue

        b_tok = _safe_float(base["avg_total_tokens_per_step"], 0.0)
        b_fps = _safe_float(base["fps"], 0.0)
        b_lat = _safe_float(base["latency_ms_p50"], 0.0)
        b_tflops = _safe_float(base["approx_tflops_per_step"], 0.0)
        b_acc = _safe_float(base["overall_action_acc"], 0.0)

        token_red = (b_tok - row["avg_total_tokens_per_step"]) / b_tok if b_tok > 0 else 0.0
        fps_gain = (row["fps"] - b_fps) / b_fps if b_fps > 0 else 0.0
        lat_red = (b_lat - row["latency_ms_p50"]) / b_lat if b_lat > 0 else 0.0
        tflops_red = (b_tflops - row["approx_tflops_per_step"]) / b_tflops if b_tflops > 0 else 0.0
        acc_drop = b_acc - row["overall_action_acc"]

        # Token-centric ranking with accuracy penalty.
        score = 0.4 * token_red + 0.2 * fps_gain + 0.2 * lat_red + 0.2 * tflops_red
        if acc_drop > 0:
            score -= min(1.0, acc_drop / 0.03) * 0.5

        row["token_reduction_vs_mode_baseline"] = token_red
        row["fps_gain_vs_mode_baseline"] = fps_gain
        row["latency_p50_reduction_vs_mode_baseline"] = lat_red
        row["tflops_reduction_vs_mode_baseline"] = tflops_red
        row["acc_drop_vs_mode_baseline"] = acc_drop
        row["score"] = score
        out.append(row)
    return out


def _write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "candidate",
        "mode",
        "score",
        "overall_action_acc",
        "acc_drop_vs_mode_baseline",
        "avg_total_tokens_per_step",
        "token_reduction_vs_mode_baseline",
        "fps",
        "fps_gain_vs_mode_baseline",
        "latency_ms_p50",
        "latency_p50_reduction_vs_mode_baseline",
        "approx_tflops_per_step",
        "tflops_reduction_vs_mode_baseline",
        "gpu_peak_allocated_mib",
        "elapsed_seconds",
        "num_episodes_eval",
        "summary_path",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("# Token Pruning Leaderboard")
    lines.append("")
    lines.append("| rank | candidate | mode | score | acc | token_red | fps_gain | lat_p50_red | tflops_red |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {candidate} | {mode} | {score:.4f} | {acc:.4f} | {tok:.4f} | {fps:.4f} | {lat:.4f} | {tf:.4f} |".format(
                rank=i,
                candidate=row["candidate"],
                mode=row["mode"],
                score=_safe_float(row["score"]),
                acc=_safe_float(row["overall_action_acc"]),
                tok=_safe_float(row["token_reduction_vs_mode_baseline"]),
                fps=_safe_float(row["fps_gain_vs_mode_baseline"]),
                lat=_safe_float(row["latency_p50_reduction_vs_mode_baseline"]),
                tf=_safe_float(row["tflops_reduction_vs_mode_baseline"]),
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate token-pruning summaries into sortable leaderboard.")
    parser.add_argument("--run_index", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_md", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    with open(args.run_index, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            summary_path = r.get("summary_path", "").strip()
            candidate = r.get("candidate", "unknown").strip()
            mode = r.get("mode", "unknown").strip()
            status = r.get("status", "").strip()

            if status != "ok" or not summary_path or not os.path.isfile(summary_path):
                continue

            summary = _load_json(summary_path)
            rows.append(_metric_row(candidate, mode, summary, summary_path))

    baselines = _build_mode_baselines(rows)
    rows = _add_relative_metrics(rows, baselines)

    rows_sorted = sorted(rows, key=lambda x: (x["mode"], -_safe_float(x["score"]), x["candidate"]))

    _write_csv(args.out_csv, rows_sorted)
    _write_md(args.out_md, rows_sorted)

    print(f"[aggregate] rows={len(rows_sorted)}")
    print(f"[aggregate] csv={args.out_csv}")
    print(f"[aggregate] md={args.out_md}")


if __name__ == "__main__":
    main()
