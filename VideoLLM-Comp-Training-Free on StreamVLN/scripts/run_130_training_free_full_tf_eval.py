#!/usr/bin/env python3
"""Run full-dataset TF eval for the five training-free VideoLLM-Comp methods.

The script uses streamvln_ext/entrypoints/eval_offline_action_acc_tf_ext.py as
the evaluation entrypoint and aggregates the requested table/plot artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = "/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen"
DEFAULT_MODEL_PATH = "/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"


BASE_DISABLED_FLAGS = {
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


@dataclass(frozen=True)
class EvalConfig:
    phase: str
    method: str
    experiment_name: str
    keep_ratio: float
    setting: str
    flags: Dict[str, object]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_gpus(raw: str) -> List[str]:
    out = [part.strip() for part in raw.split(",") if part.strip()]
    return out or ["0"]


def _unique_ordered(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _json_dumps(data: Dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _compressor_flags(method: str, keep_ratio: float, **kwargs: object) -> Dict[str, object]:
    flags: Dict[str, object] = {
        **BASE_DISABLED_FLAGS,
        "enable_video_token_compressor": True,
        "video_token_compressor_type": method,
        "video_token_target_keep_ratio": keep_ratio,
    }
    flags.update(kwargs)
    return flags


def build_configs() -> List[EvalConfig]:
    return [
        EvalConfig(
            phase="Phase 1",
            method="visionzip",
            experiment_name="visionzip_keep_ratio_0_3",
            keep_ratio=0.3,
            setting="keep_ratio=0.3 (dominant=24, contextual=6)",
            flags=_compressor_flags(
                "visionzip",
                0.3,
                visionzip_dominant_num=24,
                visionzip_contextual_num=6,
                visionzip_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="visionzip",
            experiment_name="visionzip_keep_ratio_0_5",
            keep_ratio=0.5,
            setting="keep_ratio=0.5 (dominant=36, contextual=12)",
            flags=_compressor_flags(
                "visionzip",
                0.5,
                visionzip_dominant_num=36,
                visionzip_contextual_num=12,
                visionzip_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="visionzip",
            experiment_name="visionzip_keep_ratio_0_7",
            keep_ratio=0.7,
            setting="keep_ratio=0.7 (dominant=48, contextual=20)",
            flags=_compressor_flags(
                "visionzip",
                0.7,
                visionzip_dominant_num=48,
                visionzip_contextual_num=20,
                visionzip_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="prunevid",
            experiment_name="prunevid_tau_0_7_cluster_ratio_0_3_temporal_ratio_0_5",
            keep_ratio=0.3,
            setting="tau=0.7, cluster=0.30, temporal=0.50",
            flags=_compressor_flags(
                "prunevid",
                0.15,
                prunevid_tau=0.7,
                prunevid_cluster_ratio=0.3,
                prunevid_temporal_ratio=0.5,
                prunevid_k=7,
                prunevid_min_tokens_for_cluster=14,
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="prunevid",
            experiment_name="prunevid_tau_0_8_cluster_ratio_0_5_temporal_ratio_0_25",
            keep_ratio=0.5,
            setting="tau=0.8, cluster=0.50, temporal=0.25",
            flags=_compressor_flags(
                "prunevid",
                0.125,
                prunevid_tau=0.8,
                prunevid_cluster_ratio=0.5,
                prunevid_temporal_ratio=0.25,
                prunevid_k=7,
                prunevid_min_tokens_for_cluster=14,
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="prunevid",
            experiment_name="prunevid_tau_0_9_cluster_ratio_0_7_temporal_ratio_0_1",
            keep_ratio=0.7,
            setting="tau=0.9, cluster=0.70, temporal=0.10",
            flags=_compressor_flags(
                "prunevid",
                0.07,
                prunevid_tau=0.9,
                prunevid_cluster_ratio=0.7,
                prunevid_temporal_ratio=0.1,
                prunevid_k=7,
                prunevid_min_tokens_for_cluster=14,
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="dytok_static",
            experiment_name="dytok_static_visionzip_keep_ratio_0_3",
            keep_ratio=0.3,
            setting="upper=0.3, min=0.2, base=visionzip",
            flags=_compressor_flags(
                "dytok_static",
                0.3,
                dytok_static_base_compressor="visionzip",
                dytok_static_upper_limit_ratio=0.3,
                dytok_static_min_ratio=0.2,
                visionzip_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="dytok_static",
            experiment_name="dytok_static_visionzip_keep_ratio_0_5",
            keep_ratio=0.5,
            setting="upper=0.5, min=0.3, base=visionzip",
            flags=_compressor_flags(
                "dytok_static",
                0.5,
                dytok_static_base_compressor="visionzip",
                dytok_static_upper_limit_ratio=0.5,
                dytok_static_min_ratio=0.3,
                visionzip_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 1",
            method="dytok_static",
            experiment_name="dytok_static_visionzip_keep_ratio_0_7",
            keep_ratio=0.7,
            setting="upper=0.7, min=0.5, base=visionzip",
            flags=_compressor_flags(
                "dytok_static",
                0.7,
                dytok_static_base_compressor="visionzip",
                dytok_static_upper_limit_ratio=0.7,
                dytok_static_min_ratio=0.5,
                visionzip_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 2",
            method="fastvid",
            experiment_name="fastvid_keep_ratio_0_3",
            keep_ratio=0.3,
            setting="keep_ratio=0.3",
            flags=_compressor_flags(
                "fastvid",
                0.3,
                fastvid_retention_ratio=0.3,
                fastvid_dyseg_c=8,
                fastvid_dyseg_tau=0.9,
                fastvid_stprune_d=0.4,
                fastvid_dtm_p=4,
                fastvid_dtm_beta=0.6,
                fastvid_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 2",
            method="fastvid",
            experiment_name="fastvid_keep_ratio_0_5",
            keep_ratio=0.5,
            setting="keep_ratio=0.5",
            flags=_compressor_flags(
                "fastvid",
                0.5,
                fastvid_retention_ratio=0.5,
                fastvid_dyseg_c=8,
                fastvid_dyseg_tau=0.9,
                fastvid_stprune_d=0.4,
                fastvid_dtm_p=4,
                fastvid_dtm_beta=0.6,
                fastvid_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 2",
            method="fastvid",
            experiment_name="fastvid_keep_ratio_0_7",
            keep_ratio=0.7,
            setting="keep_ratio=0.7",
            flags=_compressor_flags(
                "fastvid",
                0.7,
                fastvid_retention_ratio=0.7,
                fastvid_dyseg_c=8,
                fastvid_dyseg_tau=0.9,
                fastvid_stprune_d=0.4,
                fastvid_dtm_p=4,
                fastvid_dtm_beta=0.6,
                fastvid_score_type="attn_proxy",
            ),
        ),
        EvalConfig(
            phase="Phase 2",
            method="vqtoken",
            experiment_name="vqtoken_keep_ratio_0_3",
            keep_ratio=0.3,
            setting="keep_ratio=0.3, clusters=59",
            flags=_compressor_flags(
                "vqtoken",
                0.3,
                vqtoken_num_clusters=59,
                vqtoken_adaptive=False,
                vqtoken_max_clusters=64,
                vqtoken_adaptive_method="silhouette",
                vqtoken_use_cross_attention=False,
            ),
        ),
        EvalConfig(
            phase="Phase 2",
            method="vqtoken",
            experiment_name="vqtoken_keep_ratio_0_5",
            keep_ratio=0.5,
            setting="keep_ratio=0.5, clusters=98",
            flags=_compressor_flags(
                "vqtoken",
                0.5,
                vqtoken_num_clusters=98,
                vqtoken_adaptive=False,
                vqtoken_max_clusters=64,
                vqtoken_adaptive_method="silhouette",
                vqtoken_use_cross_attention=False,
            ),
        ),
        EvalConfig(
            phase="Phase 2",
            method="vqtoken",
            experiment_name="vqtoken_keep_ratio_0_7",
            keep_ratio=0.7,
            setting="keep_ratio=0.7, clusters=137",
            flags=_compressor_flags(
                "vqtoken",
                0.7,
                vqtoken_num_clusters=137,
                vqtoken_adaptive=False,
                vqtoken_max_clusters=64,
                vqtoken_adaptive_method="silhouette",
                vqtoken_use_cross_attention=False,
            ),
        ),
    ]


def eval_command(output_dir: Path, baseline_tokens: Optional[float]) -> List[str]:
    cmd = [
        sys.executable,
        "streamvln_ext/entrypoints/eval_offline_action_acc_tf_ext.py",
        "--model_path",
        os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        "--base_model_path",
        os.environ.get("BASE_MODEL_PATH", os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)),
        "--dataset_root",
        os.environ.get("DATASET_ROOT", DEFAULT_DATASET_ROOT),
        "--output_path",
        str(output_dir),
        "--num_history",
        os.environ.get("NUM_HISTORY", "8"),
        "--model_max_length",
        os.environ.get("MODEL_MAX_LENGTH", "4096"),
        "--max_episodes",
        os.environ.get("MAX_EPISODES", "0"),
        "--eval_protocol",
        os.environ.get("EVAL_PROTOCOL", "aurora_replay_gt"),
        "--aurora_decode_max_new_tokens",
        os.environ.get("AURORA_DECODE_MAX_NEW_TOKENS", "16"),
        "--aurora_batch_size",
        os.environ.get("AURORA_BATCH_SIZE", "1"),
        "--aurora_step_mode",
        os.environ.get("AURORA_STEP_MODE", "generate"),
        "--aurora_vision_batch_size",
        os.environ.get("AURORA_VISION_BATCH_SIZE", "16"),
    ]
    if _bool_env("AURORA_PRECOMPUTE_VISION", False):
        cmd.append("--aurora_precompute_vision")
    if _bool_env("SAVE_STEP_DEBUG", False):
        cmd.append("--save_step_debug")
        cmd.extend(
            [
                "--debug_max_steps_per_episode",
                os.environ.get("DEBUG_MAX_STEPS_PER_EPISODE", "8"),
            ]
        )
    if baseline_tokens is not None and baseline_tokens > 0:
        cmd.extend(["--baseline_avg_total_tokens_per_step", f"{baseline_tokens:.12f}"])
    return cmd


def run_eval(
    name: str,
    output_dir: Path,
    flags: Dict[str, object],
    gpu_id: str,
    baseline_tokens: Optional[float],
    reuse: bool,
) -> int:
    summary_path = output_dir / "summary.json"
    if reuse and summary_path.is_file():
        print(f"[reuse] {name}: {summary_path}", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["STREAMVLN_EXT_FLAGS"] = _json_dumps(flags)

    cmd = eval_command(output_dir, baseline_tokens)
    log_path = output_dir / "eval.log"
    timing_path = output_dir / "timing.txt"
    started = time.time()
    print(f"[start] {name} gpu={gpu_id} output={output_dir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(cmd) + "\n")
        log.write("[flags] " + env["STREAMVLN_EXT_FLAGS"] + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = int(time.time() - started)
    timing_path.write_text(f"eval_elapsed_seconds={elapsed}\n", encoding="utf-8")
    status = "ok" if proc.returncode == 0 else "failed"
    print(f"[done] {name} status={status} code={proc.returncode} seconds={elapsed}", flush=True)
    return proc.returncode


def load_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from_summary(
    cfg: Optional[EvalConfig],
    summary_path: Path,
    baseline_tokens: float,
) -> Dict[str, object]:
    data = load_summary(summary_path)
    method = "baseline" if cfg is None else cfg.method
    phase = "Baseline" if cfg is None else cfg.phase
    keep_ratio = 1.0 if cfg is None else cfg.keep_ratio
    setting = "baseline" if cfg is None else cfg.setting
    avg_tokens = float(data.get("avg_total_tokens_per_step", 0.0))
    token_reduction = 0.0
    if baseline_tokens > 0:
        token_reduction = (baseline_tokens - avg_tokens) / baseline_tokens
    return {
        "phase": phase,
        "method": method,
        "setting": setting,
        "keep_ratio": keep_ratio,
        "overall_action_acc": float(data.get("overall_action_acc", 0.0)),
        "avg_total_tokens_per_step": avg_tokens,
        "token_reduction_ratio_vs_baseline": token_reduction,
        "fps": float(data.get("fps", 0.0)),
        "latency_ms_p50": float(data.get("latency_ms_p50", 0.0)),
        "latency_ms_p95": float(data.get("latency_ms_p95", 0.0)),
        "num_episodes_eval": int(data.get("num_episodes_eval", 0)),
        "num_actions_compared": int(data.get("num_actions_compared", 0)),
        "summary_path": str(summary_path),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields = [
        "phase",
        "method",
        "setting",
        "keep_ratio",
        "overall_action_acc",
        "avg_total_tokens_per_step",
        "token_reduction_ratio_vs_baseline",
        "fps",
        "latency_ms_p50",
        "latency_ms_p95",
        "num_episodes_eval",
        "num_actions_compared",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _setting_label(row: Dict[str, object]) -> str:
    keep_ratio = float(row["keep_ratio"])
    return f"kr={keep_ratio:.1f}: {row['setting']}"


def _metric_series(rows: List[Dict[str, object]], metric: str, digits: int = 4) -> str:
    parts = []
    for row in sorted(rows, key=lambda r: float(r["keep_ratio"])):
        keep_ratio = float(row["keep_ratio"])
        value = row[metric]
        parts.append(f"kr={keep_ratio:.1f}: {_fmt(value, digits)}")
    return "; ".join(parts)


def _method_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    order = ["baseline", "visionzip", "prunevid", "dytok_static", "fastvid", "vqtoken"]
    out = []
    for method in order:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        if method == "baseline":
            row = method_rows[0]
            out.append(
                {
                    "method": method,
                    "planned_settings": "baseline",
                    "overall_action_acc": _fmt(row["overall_action_acc"]),
                    "avg_total_tokens_per_step": _fmt(row["avg_total_tokens_per_step"]),
                    "token_reduction_ratio_vs_baseline": _fmt(row["token_reduction_ratio_vs_baseline"]),
                    "fps": _fmt(row["fps"]),
                }
            )
            continue

        method_rows = sorted(method_rows, key=lambda r: float(r["keep_ratio"]))
        out.append(
            {
                "method": method,
                "planned_settings": "; ".join(_setting_label(row) for row in method_rows),
                "overall_action_acc": _metric_series(method_rows, "overall_action_acc"),
                "avg_total_tokens_per_step": _metric_series(method_rows, "avg_total_tokens_per_step"),
                "token_reduction_ratio_vs_baseline": _metric_series(
                    method_rows, "token_reduction_ratio_vs_baseline"
                ),
                "fps": _metric_series(method_rows, "fps"),
            }
        )
    return out


def write_report(path: Path, rows: List[Dict[str, object]], plot_rel_path: str) -> None:
    lines = [
        "# Training-Free Full-Dataset TF Eval",
        "",
        f"- dataset_root: `{os.environ.get('DATASET_ROOT', DEFAULT_DATASET_ROOT)}`",
        f"- eval_entrypoint: `streamvln_ext/entrypoints/eval_offline_action_acc_tf_ext.py`",
        f"- eval_protocol: `{os.environ.get('EVAL_PROTOCOL', 'aurora_replay_gt')}`",
        f"- max_episodes: `{os.environ.get('MAX_EPISODES', '0')}`",
        "",
        "| method | planned_settings | overall_action_acc | avg_total_tokens_per_step | token_reduction_ratio_vs_baseline | fps |",
        "|---|---|---|---|---|---|",
    ]
    for row in _method_summary_rows(rows):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["planned_settings"]),
                    str(row["overall_action_acc"]),
                    str(row["avg_total_tokens_per_step"]),
                    str(row["token_reduction_ratio_vs_baseline"]),
                    str(row["fps"]),
                ]
            )
            + " |"
        )
    lines.extend(["", f"![overall_action_acc vs keep_ratio]({plot_rel_path})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path: Path, rows: List[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_rows: Dict[str, List[Dict[str, object]]] = {}
    baseline_acc = None
    for row in rows:
        if row["method"] == "baseline":
            baseline_acc = float(row["overall_action_acc"])
            continue
        method_rows.setdefault(str(row["method"]), []).append(row)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160)
    for method, method_data in sorted(method_rows.items()):
        method_data = sorted(method_data, key=lambda r: float(r["keep_ratio"]))
        xs = [float(r["keep_ratio"]) for r in method_data]
        ys = [float(r["overall_action_acc"]) for r in method_data]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=method)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    if baseline_acc is not None:
        ax.axhline(baseline_acc, linestyle="--", color="#555555", linewidth=1.4, label=f"baseline {baseline_acc:.4f}")

    ax.set_title("Full R2R Val-Unseen TF: overall_action_acc vs keep_ratio")
    ax.set_xlabel("keep_ratio")
    ax.set_ylabel("overall_action_acc")
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def aggregate(suite_root: Path, configs: List[EvalConfig]) -> List[Dict[str, object]]:
    baseline_summary = suite_root / "baseline" / "summary.json"
    baseline_tokens = float(load_summary(baseline_summary).get("avg_total_tokens_per_step", 0.0))
    rows = [row_from_summary(None, baseline_summary, baseline_tokens)]
    for cfg in configs:
        summary_path = suite_root / cfg.experiment_name / "summary.json"
        if not summary_path.is_file():
            print(f"[warn] missing summary for {cfg.experiment_name}: {summary_path}", flush=True)
            continue
        rows.append(row_from_summary(cfg, summary_path, baseline_tokens))
    return rows


def write_run_index(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["name", "method", "keep_ratio", "status", "exit_code", "output_dir", "summary_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    suite_ts = os.environ.get("SUITE_TS") or time.strftime("%Y%m%d_%H%M%S")
    suite_root = Path(
        os.environ.get(
            "SUITE_ROOT",
            str(PROJECT_ROOT / "experiments_ext" / "videollm_comp_full_eval_tf_runs" / suite_ts),
        )
    )
    latest_link = suite_root.parent / "latest_suite"
    suite_root.mkdir(parents=True, exist_ok=True)
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(suite_root)
    except OSError as exc:
        print(f"[warn] could not update latest_suite symlink: {exc}", flush=True)

    gpus = _unique_ordered(_split_gpus(os.environ.get("GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))))
    runs_per_gpu = max(1, int(os.environ.get("RUNS_PER_GPU", "1")))
    gpu_slots = [
        {"gpu_id": gpu, "slot_id": f"{gpu}:{slot_idx}"}
        for slot_idx in range(runs_per_gpu)
        for gpu in gpus
    ]
    max_parallel = int(os.environ.get("MAX_PARALLEL", str(len(gpu_slots))))
    max_parallel = max(1, min(max_parallel, len(gpu_slots)))
    reuse = _bool_env("REUSE_IF_EXISTS", True)
    configs = build_configs()
    config_limit = int(os.environ.get("CONFIG_LIMIT", "0"))
    if config_limit > 0:
        configs = configs[:config_limit]

    print(f"[suite] root={suite_root}", flush=True)
    print(
        f"[suite] gpus={','.join(gpus)} runs_per_gpu={runs_per_gpu} "
        f"slots={len(gpu_slots)} max_parallel={max_parallel}",
        flush=True,
    )
    print(f"[suite] dataset={os.environ.get('DATASET_ROOT', DEFAULT_DATASET_ROOT)}", flush=True)
    print(f"[suite] eval_protocol={os.environ.get('EVAL_PROTOCOL', 'aurora_replay_gt')}", flush=True)

    run_rows: List[Dict[str, object]] = []
    baseline_dir = suite_root / "baseline"
    run_baseline_concurrent = _bool_env("RUN_BASELINE_CONCURRENT", False)
    if run_baseline_concurrent:
        baseline_tokens: Optional[float] = None
        pending: List[Optional[EvalConfig]] = [None] + list(configs)
        print("[suite] baseline will run concurrently with configs", flush=True)
    else:
        baseline_code = run_eval(
            "baseline",
            baseline_dir,
            {**BASE_DISABLED_FLAGS, "enable_video_token_compressor": False},
            gpus[0],
            None,
            reuse,
        )
        run_rows.append(
            {
                "name": "baseline",
                "method": "baseline",
                "keep_ratio": 1.0,
                "status": "ok" if baseline_code == 0 else "failed",
                "exit_code": baseline_code,
                "output_dir": str(baseline_dir),
                "summary_path": str(baseline_dir / "summary.json"),
            }
        )
        if baseline_code != 0:
            write_run_index(suite_root / "run_index.csv", run_rows)
            return baseline_code

        baseline_tokens = float(load_summary(baseline_dir / "summary.json").get("avg_total_tokens_per_step", 0.0))
        print(f"[baseline] avg_total_tokens_per_step={baseline_tokens:.6f}", flush=True)
        pending = list(configs)

    running: Dict[subprocess.Popen[bytes], Dict[str, object]] = {}
    # Use Popen here so we can keep GPU slots busy without loading a worker pool inside CUDA contexts.
    while pending or running:
        active_slots = {str(info["slot_id"]) for info in running.values()}
        baseline_gpus = set()
        if not _bool_env("ALLOW_BASELINE_GPU_SHARING", False):
            baseline_gpus = {
                str(info["gpu_id"])
                for info in running.values()
                if str(info.get("method", "")) == "baseline"
            }
        available_slots = [
            slot
            for slot in gpu_slots
            if str(slot["slot_id"]) not in active_slots and str(slot["gpu_id"]) not in baseline_gpus
        ]
        while pending and len(running) < max_parallel and available_slots:
            cfg = pending.pop(0)
            slot = available_slots.pop(0)
            gpu_id = str(slot["gpu_id"])
            slot_id = str(slot["slot_id"])
            if cfg is None:
                name = "baseline"
                method = "baseline"
                keep_ratio = 1.0
                flags = {**BASE_DISABLED_FLAGS, "enable_video_token_compressor": False}
                output_dir = baseline_dir
            else:
                name = cfg.experiment_name
                method = cfg.method
                keep_ratio = cfg.keep_ratio
                flags = cfg.flags
                output_dir = suite_root / cfg.experiment_name
            summary_path = output_dir / "summary.json"
            if reuse and summary_path.is_file():
                print(f"[reuse] {name}: {summary_path}", flush=True)
                run_rows.append(
                    {
                        "name": name,
                        "method": method,
                        "keep_ratio": keep_ratio,
                        "status": "ok",
                        "exit_code": 0,
                        "output_dir": str(output_dir),
                        "summary_path": str(summary_path),
                    }
                )
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["STREAMVLN_EXT_FLAGS"] = _json_dumps(flags)
            cmd = eval_command(output_dir, baseline_tokens)
            log_path = output_dir / "eval.log"
            log = log_path.open("w", encoding="utf-8")
            log.write("[command] " + " ".join(cmd) + "\n")
            log.write("[flags] " + env["STREAMVLN_EXT_FLAGS"] + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[proc] = {
                "cfg": cfg,
                "name": name,
                "method": method,
                "keep_ratio": keep_ratio,
                "gpu_id": gpu_id,
                "slot_id": slot_id,
                "output_dir": output_dir,
                "log": log,
                "started": time.time(),
            }
            print(f"[start] {name} gpu={gpu_id} slot={slot_id} output={output_dir}", flush=True)
            active_slots = {str(info["slot_id"]) for info in running.values()}
            baseline_gpus = set()
            if not _bool_env("ALLOW_BASELINE_GPU_SHARING", False):
                baseline_gpus = {
                    str(info["gpu_id"])
                    for info in running.values()
                    if str(info.get("method", "")) == "baseline"
                }
            available_slots = [
                slot
                for slot in gpu_slots
                if str(slot["slot_id"]) not in active_slots and str(slot["gpu_id"]) not in baseline_gpus
            ]

        if not running:
            continue

        done = [proc for proc in running if proc.poll() is not None]
        if not done:
            active = ", ".join(
                f"{info['name']}@gpu{info['gpu_id']}/slot{info['slot_id']}"
                for info in running.values()  # type: ignore[index]
            )
            print(f"[heartbeat] running={active}", flush=True)
            time.sleep(30)
            continue

        for proc in done:
            info = running.pop(proc)
            name = str(info["name"])
            method = str(info["method"])
            keep_ratio = float(info["keep_ratio"])
            output_dir = Path(info["output_dir"])  # type: ignore[arg-type]
            elapsed = int(time.time() - float(info["started"]))
            log = info["log"]
            log.close()
            (output_dir / "timing.txt").write_text(f"eval_elapsed_seconds={elapsed}\n", encoding="utf-8")
            code = proc.returncode if proc.returncode is not None else proc.wait()
            status = "ok" if code == 0 else "failed"
            print(f"[done] {name} status={status} code={code} seconds={elapsed}", flush=True)
            run_rows.append(
                {
                    "name": name,
                    "method": method,
                    "keep_ratio": keep_ratio,
                    "status": status,
                    "exit_code": code,
                    "output_dir": str(output_dir),
                    "summary_path": str(output_dir / "summary.json"),
                }
            )

    write_run_index(suite_root / "run_index.csv", run_rows)
    failed = [row for row in run_rows if row["status"] != "ok"]
    if failed:
        print(f"[failed] {len(failed)} runs failed; see run_index.csv", flush=True)
        return 1

    if (baseline_dir / "summary.json").is_file():
        baseline_tokens = float(load_summary(baseline_dir / "summary.json").get("avg_total_tokens_per_step", 0.0))
        print(f"[baseline] avg_total_tokens_per_step={baseline_tokens:.6f}", flush=True)

    rows = aggregate(suite_root, configs)
    csv_path = suite_root / "training_free_full_eval_tf_overview.csv"
    report_path = suite_root / "training_free_full_eval_tf_report.md"
    plot_dir = suite_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / "overall_action_acc_vs_keep_ratio.png"
    write_csv(csv_path, rows)
    write_plot(plot_path, rows)
    write_report(report_path, rows, "plots/overall_action_acc_vs_keep_ratio.png")

    print(f"[wrote] {csv_path}", flush=True)
    print(f"[wrote] {report_path}", flush=True)
    print(f"[wrote] {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
