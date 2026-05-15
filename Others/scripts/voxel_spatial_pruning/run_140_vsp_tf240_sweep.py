#!/usr/bin/env python3
"""Run Voxel-Based Spatial Pruning evals on the fixed TF240 Aurora subset."""

from __future__ import annotations

import argparse
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
DEFAULT_MODEL_PATH = "/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
DEFAULT_DATASET_ROOT = (
    "/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/"
    "phase0_20260421_230201/teacher_forcing_240"
)
DEFAULT_BASELINE_TOKENS = 1736.783835514942
DEFAULT_BASELINE_ACC = 0.3362558656458385
DEFAULT_BASELINE_INVALID = 0.4803037787107928


BASE_DISABLED_FLAGS: Dict[str, object] = {
    "enable_sliding_kv": False,
    "enable_memory_loss": False,
    "enable_voxel_proxy": False,
    "enable_token_selection": False,
    "enable_dynamic_memory": False,
    "enable_multiscale_memory": False,
    "enable_voxel_rgbd": False,
    "enable_video_token_compressor": False,
    "enable_training_aware_video_compressor": False,
    "enable_hc_st_pruning": False,
    "enable_tuning_free_mm_pruning": False,
    "enable_tome_visual_merge": False,
}


@dataclass(frozen=True)
class VspConfig:
    stage: str
    name: str
    voxel_size: float
    stride_k: int
    frame_threshold: float
    min_depth: float = 0.05
    max_depth: float = 10.0
    note: str = ""


def _split_gpus(raw: str) -> List[str]:
    gpus = [part.strip() for part in raw.split(",") if part.strip()]
    return gpus or ["0"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _json_dumps(data: Dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def build_flags(cfg: VspConfig) -> Dict[str, object]:
    flags: Dict[str, object] = {
        **BASE_DISABLED_FLAGS,
        "enable_voxel_spatial_pruning": True,
        "enable_offline_saved_geometry": True,
        "enable_runtime_token_metrics": True,
        "enable_runtime_latency_metrics": True,
        "enable_runtime_tflops_estimate": True,
        "voxel_spatial_size": float(cfg.voxel_size),
        "voxel_spatial_stride_k": int(cfg.stride_k),
        "voxel_spatial_frame_threshold": float(cfg.frame_threshold),
        "voxel_spatial_min_depth": float(cfg.min_depth),
        "voxel_spatial_max_depth": float(cfg.max_depth),
        "voxel_spatial_offline_geom_mode": "odometry",
        "voxel_spatial_offline_unit_depth_m": 2.0,
        "voxel_spatial_offline_hfov_deg": 75.17817894,
    }
    return flags


def initial_configs() -> List[VspConfig]:
    return [
        VspConfig(
            stage="sanity",
            name="sanity_s0_25_k4_t0_00_d10",
            voxel_size=0.25,
            stride_k=4,
            frame_threshold=0.0,
            note="default size with threshold disabled for isolated size sweep",
        ),
        VspConfig(
            stage="sanity",
            name="sanity_stress_s1_00_k8_t0_20_d10",
            voxel_size=1.00,
            stride_k=8,
            frame_threshold=0.20,
            note="stress test for aggressive pruning and frame dropping",
        ),
        VspConfig(
            stage="size_sweep",
            name="size_s0_10_k4_t0_00_d10",
            voxel_size=0.10,
            stride_k=4,
            frame_threshold=0.0,
            note="conservative size point",
        ),
        VspConfig(
            stage="size_sweep",
            name="size_s0_25_k4_t0_00_d10",
            voxel_size=0.25,
            stride_k=4,
            frame_threshold=0.0,
            note="main default-size point",
        ),
        VspConfig(
            stage="size_sweep",
            name="size_s0_50_k4_t0_00_d10",
            voxel_size=0.50,
            stride_k=4,
            frame_threshold=0.0,
            note="mid-aggressive size point",
        ),
        VspConfig(
            stage="size_sweep",
            name="size_s1_00_k4_t0_00_d10",
            voxel_size=1.00,
            stride_k=4,
            frame_threshold=0.0,
            note="very aggressive size point without frame threshold",
        ),
    ]


def refine_configs(size: float) -> List[VspConfig]:
    tag = str(size).replace(".", "_")
    return [
        VspConfig("k_sweep", f"k_sweep_s{tag}_k1_t0_00_d10", size, 1, 0.0),
        VspConfig("k_sweep", f"k_sweep_s{tag}_k2_t0_00_d10", size, 2, 0.0),
        VspConfig("k_sweep", f"k_sweep_s{tag}_k4_t0_00_d10", size, 4, 0.0),
        VspConfig("k_sweep", f"k_sweep_s{tag}_k8_t0_00_d10", size, 8, 0.0),
        VspConfig("threshold", f"threshold_s{tag}_k8_t0_10_d10", size, 8, 0.10),
        VspConfig("threshold", f"threshold_s{tag}_k8_t0_20_d10", size, 8, 0.20),
        VspConfig("threshold", f"threshold_s{tag}_k8_t0_30_d10", size, 8, 0.30),
        VspConfig("depth", f"depth_s{tag}_k4_t0_00_d5", size, 4, 0.0, max_depth=5.0),
        VspConfig("depth", f"depth_s{tag}_k4_t0_00_d20", size, 4, 0.0, max_depth=20.0),
    ]


def eval_command(output_dir: Path, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "streamvln_ext/entrypoints/eval_offline_action_acc_tf_ext.py",
        "--model_path",
        args.model_path,
        "--dataset_root",
        args.dataset_root,
        "--output_path",
        str(output_dir),
        "--num_history",
        str(args.num_history),
        "--model_max_length",
        str(args.model_max_length),
        "--max_episodes",
        str(args.max_episodes),
        "--eval_protocol",
        "aurora_replay_gt",
        "--aurora_decode_max_new_tokens",
        str(args.aurora_decode_max_new_tokens),
        "--aurora_batch_size",
        str(args.aurora_batch_size),
        "--aurora_step_mode",
        args.aurora_step_mode,
        "--aurora_vision_batch_size",
        str(args.aurora_vision_batch_size),
        "--baseline_avg_total_tokens_per_step",
        f"{args.baseline_tokens:.12f}",
    ]
    if args.base_model_path:
        cmd.extend(["--base_model_path", args.base_model_path])
    if args.aurora_precompute_vision:
        cmd.append("--aurora_precompute_vision")
    return cmd


def load_summary(output_dir: Path) -> Optional[Dict[str, object]]:
    path = output_dir / "summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_one(cfg: VspConfig, gpu_id: str, suite_dir: Path, args: argparse.Namespace) -> subprocess.Popen:
    output_dir = suite_dir / cfg.stage / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = build_flags(cfg)

    (output_dir / "flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cmd = eval_command(output_dir, args)
    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["STREAMVLN_EXT_FLAGS"] = _json_dumps(flags)
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_f = (output_dir / "eval.log").open("w", encoding="utf-8")
    print(f"[launch] gpu={gpu_id} {cfg.name}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._vsp_log_file = log_f  # type: ignore[attr-defined]
    proc._vsp_cfg = cfg  # type: ignore[attr-defined]
    proc._vsp_output_dir = output_dir  # type: ignore[attr-defined]
    proc._vsp_gpu_id = str(gpu_id)  # type: ignore[attr-defined]
    return proc


def _safe_float(data: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def collect_rows(configs: Iterable[VspConfig], suite_dir: Path, baseline_tokens: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        output_dir = suite_dir / cfg.stage / cfg.name
        summary = load_summary(output_dir)
        row: Dict[str, object] = {
            "stage": cfg.stage,
            "name": cfg.name,
            "voxel_size": cfg.voxel_size,
            "stride_k": cfg.stride_k,
            "frame_threshold": cfg.frame_threshold,
            "min_depth": cfg.min_depth,
            "max_depth": cfg.max_depth,
            "status": "done" if summary else "missing",
            "output_dir": str(output_dir),
            "note": cfg.note,
        }
        if summary:
            total_tokens = _safe_float(summary, "avg_total_tokens_per_step")
            row.update(
                {
                    "episodes": int(summary.get("num_episodes_eval", 0) or 0),
                    "actions": int(summary.get("num_actions_compared", 0) or 0),
                    "acc": _safe_float(summary, "overall_action_acc"),
                    "acc_delta_pp": 100.0 * (_safe_float(summary, "overall_action_acc") - DEFAULT_BASELINE_ACC),
                    "invalid": _safe_float(summary, "invalid_prediction_rate"),
                    "tokens": total_tokens,
                    "token_reduction": (baseline_tokens - total_tokens) / baseline_tokens if baseline_tokens > 0 else 0.0,
                    "avg_visual_before": _safe_float(summary, "avg_visual_tokens_before"),
                    "avg_visual_after": _safe_float(summary, "avg_visual_tokens_per_step"),
                    "avg_memory_before": _safe_float(summary, "avg_memory_tokens_before"),
                    "avg_memory_after": _safe_float(summary, "avg_memory_tokens_per_step"),
                    "runtime_reduction": _safe_float(summary, "runtime_token_reduction_ratio"),
                    "fps": _safe_float(summary, "fps"),
                    "latency_mean": _safe_float(summary, "latency_ms_mean"),
                    "latency_p95": _safe_float(summary, "latency_ms_p95"),
                    "vsp_calls": int(summary.get("voxel_spatial_num_calls", 0) or 0),
                    "vsp_pruned_batches": int(summary.get("voxel_spatial_num_pruned_batches", 0) or 0),
                    "vsp_skipped_batches": int(summary.get("voxel_spatial_num_skipped_batches", 0) or 0),
                    "vsp_effective_slots": int(summary.get("voxel_spatial_num_effective_slots", 0) or 0),
                    "vsp_skipped_slots": int(summary.get("voxel_spatial_num_skipped_slots", 0) or 0),
                    "vsp_memory_keep_ratio": _safe_float(summary, "voxel_spatial_memory_keep_ratio"),
                    "vsp_slot_keep_ratio": _safe_float(summary, "voxel_spatial_avg_slot_keep_ratio"),
                    "vsp_skip_reasons": json.dumps(
                        summary.get("voxel_spatial_skip_reasons", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    keys = [
        "stage",
        "name",
        "status",
        "voxel_size",
        "stride_k",
        "frame_threshold",
        "min_depth",
        "max_depth",
        "episodes",
        "actions",
        "acc",
        "acc_delta_pp",
        "invalid",
        "tokens",
        "token_reduction",
        "avg_visual_before",
        "avg_visual_after",
        "avg_memory_before",
        "avg_memory_after",
        "runtime_reduction",
        "fps",
        "latency_mean",
        "latency_p95",
        "vsp_calls",
        "vsp_pruned_batches",
        "vsp_skipped_batches",
        "vsp_effective_slots",
        "vsp_skipped_slots",
        "vsp_memory_keep_ratio",
        "vsp_slot_keep_ratio",
        "vsp_skip_reasons",
        "output_dir",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _fmt_pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def write_report(rows: List[Dict[str, object]], suite_dir: Path, args: argparse.Namespace) -> None:
    report = suite_dir / "vsp_tf240_aurora_report.md"
    done_rows = [row for row in rows if row.get("status") == "done"]
    formal_rows = [row for row in done_rows if row.get("stage") == "size_sweep"] or done_rows
    best_acc = max(formal_rows, key=lambda row: float(row.get("acc", -1.0)), default=None)
    best_trade = max(
        formal_rows,
        key=lambda row: (
            float(row.get("acc", 0.0)) >= DEFAULT_BASELINE_ACC - 0.005,
            float(row.get("token_reduction", 0.0)),
            float(row.get("acc", 0.0)),
        ),
        default=None,
    )

    lines = [
        "# Voxel-Based Spatial Pruning on StreamVLN TF240",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Suite: `{suite_dir}`",
        f"- Dataset: `{args.dataset_root}`",
        "- Protocol: `AuroraReplay-GT`",
        f"- Model: `{args.model_path}`",
        f"- Base model: `{args.base_model_path}`",
        f"- Aurora mode: step_mode=`{args.aurora_step_mode}`, decode_max_new_tokens=`{args.aurora_decode_max_new_tokens}`, "
        f"vision_batch_size=`{args.aurora_vision_batch_size}`, precompute_vision=`{args.aurora_precompute_vision}`",
        f"- Baseline acc: `{DEFAULT_BASELINE_ACC:.6f}`",
        f"- Baseline invalid: `{DEFAULT_BASELINE_INVALID:.6f}`",
        f"- Baseline tokens/step: `{args.baseline_tokens:.6f}`",
        f"- Max episodes for this run: `{args.max_episodes}`",
        f"- GPUs: `{','.join(args.gpus)}`; one eval process per GPU",
        "",
        "## Current Best",
        "",
    ]
    if best_acc:
        lines.append(
            f"- Best accuracy: `{best_acc['name']}` acc={_fmt_pct(best_acc.get('acc'))}, "
            f"tokens={_fmt_float(best_acc.get('tokens'), 1)}, "
            f"reduction={_fmt_pct(best_acc.get('token_reduction'))}."
        )
    if best_trade:
        lines.append(
            f"- Best near-baseline tradeoff: `{best_trade['name']}` acc={_fmt_pct(best_trade.get('acc'))}, "
            f"tokens={_fmt_float(best_trade.get('tokens'), 1)}, "
            f"reduction={_fmt_pct(best_trade.get('token_reduction'))}."
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Stage | Config | size | K | threshold | max_depth | status | Episodes | Acc | Δacc | Invalid | Tokens/step | Reduction | Mem before | Mem after | VSP keep | Skip reasons | FPS | p95 ms |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("stage", "")),
                    f"`{row.get('name', '')}`",
                    _fmt_float(row.get("voxel_size"), 2),
                    str(row.get("stride_k", "")),
                    _fmt_float(row.get("frame_threshold"), 2),
                    _fmt_float(row.get("max_depth"), 1),
                    str(row.get("status", "")),
                    str(row.get("episodes", "")),
                    _fmt_pct(row.get("acc")),
                    _fmt_float(row.get("acc_delta_pp"), 2),
                    _fmt_pct(row.get("invalid")),
                    _fmt_float(row.get("tokens"), 1),
                    _fmt_pct(row.get("token_reduction")),
                    _fmt_float(row.get("avg_memory_before"), 1),
                    _fmt_float(row.get("avg_memory_after"), 1),
                    _fmt_pct(row.get("vsp_memory_keep_ratio")),
                    f"`{row.get('vsp_skip_reasons', '')}`",
                    _fmt_float(row.get("fps"), 2),
                    _fmt_float(row.get("latency_p95"), 1),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Reduction` is computed against the documented TF240 Aurora baseline tokens/step.",
            "- This suite uses the same Aurora replay decoding口径 as the documented baseline: next-token logits with one decoded token and vision precompute enabled.",
            "- `Mem before/after` comes from runtime feature-token metrics before and after VSP.",
            "- `VSP keep` is the aggregate VSP memory keep ratio from VSP-specific diagnostics.",
            "- Visual tokens should stay nearly unchanged because this VSP path only prunes historical memory tokens.",
            "",
            "## Output Files",
            "",
            f"- CSV: `{suite_dir / 'vsp_tf240_aurora_results.csv'}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_configs(configs: List[VspConfig], suite_dir: Path, args: argparse.Namespace) -> None:
    pending = list(configs)
    running: List[subprocess.Popen] = []
    gpu_queue = list(args.gpus)
    all_configs = configs

    while pending or running:
        while pending and gpu_queue:
            cfg = pending.pop(0)
            output_dir = suite_dir / cfg.stage / cfg.name
            if args.reuse and (output_dir / "summary.json").is_file():
                print(f"[reuse] {cfg.name}", flush=True)
                continue
            gpu_id = gpu_queue.pop(0)
            running.append(run_one(cfg, gpu_id, suite_dir, args))

        time.sleep(args.poll_seconds)
        still_running: List[subprocess.Popen] = []
        for proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append(proc)
                continue
            cfg = proc._vsp_cfg  # type: ignore[attr-defined]
            proc._vsp_log_file.close()  # type: ignore[attr-defined]
            print(f"[done] {cfg.name} returncode={ret}", flush=True)
            gpu_queue.append(proc._vsp_gpu_id)  # type: ignore[attr-defined]
            rows = collect_rows(all_configs, suite_dir, args.baseline_tokens)
            write_csv(rows, suite_dir / "vsp_tf240_aurora_results.csv")
            write_report(rows, suite_dir, args)
            if ret != 0 and not args.keep_going:
                raise RuntimeError(f"{cfg.name} failed with return code {ret}")
        running = still_running

    rows = collect_rows(all_configs, suite_dir, args.baseline_tokens)
    write_csv(rows, suite_dir / "vsp_tf240_aurora_results.csv")
    write_report(rows, suite_dir, args)


def sanity_passed(suite_dir: Path, baseline_tokens: float) -> bool:
    sanity_dirs = sorted((suite_dir / "sanity").glob("*/summary.json"))
    if len(sanity_dirs) < 2:
        return False
    ok = True
    for path in sanity_dirs:
        summary = json.loads(path.read_text(encoding="utf-8"))
        tokens = _safe_float(summary, "avg_total_tokens_per_step")
        skipped_slots = int(summary.get("voxel_spatial_num_skipped_slots", 0) or 0)
        effective_slots = int(summary.get("voxel_spatial_num_effective_slots", 0) or 0)
        reduction = (baseline_tokens - tokens) / baseline_tokens if baseline_tokens > 0 else 0.0
        if effective_slots <= 0 or reduction <= 0.0:
            ok = False
        if skipped_slots > 0:
            print(f"[sanity] warning: skipped slots in {path.parent.name}: {skipped_slots}", flush=True)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--model-path", type=str, default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--base-model-path", type=str, default=os.environ.get("BASE_MODEL_PATH", ""))
    parser.add_argument("--dataset-root", type=str, default=os.environ.get("DATASET_ROOT", DEFAULT_DATASET_ROOT))
    parser.add_argument("--baseline-tokens", type=float, default=float(os.environ.get("BASELINE_AVG_TOTAL_TOKENS", DEFAULT_BASELINE_TOKENS)))
    parser.add_argument("--num-history", type=int, default=int(os.environ.get("NUM_HISTORY", "8")))
    parser.add_argument("--model-max-length", type=int, default=int(os.environ.get("MODEL_MAX_LENGTH", "4096")))
    parser.add_argument("--max-episodes", type=int, default=int(os.environ.get("MAX_EPISODES", "0")))
    parser.add_argument("--sanity-episodes", type=int, default=int(os.environ.get("SANITY_EPISODES", "10")))
    parser.add_argument("--aurora-decode-max-new-tokens", type=int, default=int(os.environ.get("AURORA_DECODE_MAX_NEW_TOKENS", "1")))
    parser.add_argument("--aurora-batch-size", type=int, default=int(os.environ.get("AURORA_BATCH_SIZE", "1")))
    parser.add_argument("--aurora-step-mode", type=str, default=os.environ.get("AURORA_STEP_MODE", "next_token_logits"))
    parser.add_argument("--aurora-vision-batch-size", type=int, default=int(os.environ.get("AURORA_VISION_BATCH_SIZE", "16")))
    parser.add_argument("--aurora-precompute-vision", action=argparse.BooleanOptionalAction, default=_env_bool("AURORA_PRECOMPUTE_VISION", True))
    parser.add_argument("--gpus", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--plan", choices=["sanity_size", "refine"], default=os.environ.get("VSP_PLAN", "sanity_size"))
    parser.add_argument("--refine-size", type=float, default=float(os.environ.get("VSP_REFINE_SIZE", "0.5")))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--reuse", action="store_true", default=os.environ.get("REUSE", "").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--keep-going", action="store_true", default=True)
    args = parser.parse_args()
    if not args.base_model_path:
        args.base_model_path = args.model_path
    args.gpus = _split_gpus(args.gpus)
    return args


def main() -> None:
    args = parse_args()
    if args.suite_dir:
        suite_dir = Path(args.suite_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suite_dir = PROJECT_ROOT / "experiments_ext" / "vsp_tf240_aurora_runs" / stamp
    suite_dir.mkdir(parents=True, exist_ok=True)

    (suite_dir / "suite_config.json").write_text(
        json.dumps(
            {
                "plan": args.plan,
                "model_path": args.model_path,
                "base_model_path": args.base_model_path,
                "dataset_root": args.dataset_root,
                "baseline_tokens": args.baseline_tokens,
                "num_history": args.num_history,
                "model_max_length": args.model_max_length,
                "max_episodes": args.max_episodes,
                "sanity_episodes": args.sanity_episodes,
                "aurora_decode_max_new_tokens": args.aurora_decode_max_new_tokens,
                "aurora_batch_size": args.aurora_batch_size,
                "aurora_step_mode": args.aurora_step_mode,
                "aurora_vision_batch_size": args.aurora_vision_batch_size,
                "aurora_precompute_vision": args.aurora_precompute_vision,
                "gpus": args.gpus,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.plan == "refine":
        configs = refine_configs(args.refine_size)
        run_configs(configs, suite_dir, args)
        print(f"[report] {suite_dir / 'vsp_tf240_aurora_report.md'}", flush=True)
        return

    all_configs = initial_configs()
    sanity = [cfg for cfg in all_configs if cfg.stage == "sanity"]
    size = [cfg for cfg in all_configs if cfg.stage == "size_sweep"]

    full_max_episodes = args.max_episodes
    args.max_episodes = args.sanity_episodes
    run_configs(sanity, suite_dir, args)
    if not sanity_passed(suite_dir, args.baseline_tokens):
        print("[abort] sanity did not show effective VSP pruning; inspect report before full sweep.", flush=True)
        print(f"[report] {suite_dir / 'vsp_tf240_aurora_report.md'}", flush=True)
        sys.exit(2)

    args.max_episodes = full_max_episodes
    run_configs(size, suite_dir, args)
    rows = collect_rows(all_configs, suite_dir, args.baseline_tokens)
    write_csv(rows, suite_dir / "vsp_tf240_aurora_results.csv")
    write_report(rows, suite_dir, args)
    print(f"[report] {suite_dir / 'vsp_tf240_aurora_report.md'}", flush=True)


if __name__ == "__main__":
    main()
