#!/usr/bin/env python3
"""Evaluate training-aware VideoLLM-Comp checkpoints on fixed TF240 Aurora.

This script is the TF240 counterpart of the existing AR keep-ratio evaluator.
It runs StreamVLN's teacher-forcing AuroraReplay-GT entrypoint for LongVU and
LLaMAVID compressor-only checkpoints, then writes a compact leaderboard.
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
DEFAULT_BASE_MODEL_PATH = "/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
DEFAULT_DINO_PATH = "/home/ubuntu/model/dinov2-base"
DEFAULT_BERT_PATH = "/home/ubuntu/model/bert-base-uncased"
DEFAULT_EVAL_SUBSET_ROOT = (
    "/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/"
    "phase0_20260421_230201/teacher_forcing_240"
)
DEFAULT_EVAL_INDICES_JSON = (
    "/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/"
    "phase0_20260421_230201/teacher_forcing_240_indices.json"
)
DEFAULT_TRAIN_INDICES_JSON = (
    "/home/ubuntu/project/StreamVLN/experiments_ext/"
    "videollm_comp_training_aware_fixed_train_subsets/latest_suite/"
    "train_fixed_r2r_5k_indices.json"
)
DEFAULT_BASELINE_SUMMARY = (
    "/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/"
    "20260507_fixed_tf240_aurora_promptfix/baseline/summary.json"
)
DEFAULT_BASELINE_TOKENS = 1736.783835514942
DEFAULT_BASELINE_ACC = 0.3362558656458385
DEFAULT_BASELINE_INVALID_RATE = 0.4803037787107928
BASELINE_TOKENS_PER_FRAME = 169.0


@dataclass(frozen=True)
class EvalConfig:
    method: str
    keep_ratio: float
    candidate: str
    actual_tokens_per_frame: int
    actual_keep_ratio: float
    model_max_length: int
    flags: Dict[str, object]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_words(raw: str) -> List[str]:
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def _ratio_tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "_")


def _json_dumps(data: Dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _subset_signature(path: Path) -> str:
    data = _maybe_read_json(path)
    return str(data.get("subset_signature", ""))


def _method_checkpoint_root(method: str) -> Path:
    env_key = f"{method.upper()}_CHECKPOINT_ROOT"
    if os.environ.get(env_key):
        return Path(os.environ[env_key]).expanduser()
    if os.environ.get("CHECKPOINT_ROOT"):
        return Path(os.environ["CHECKPOINT_ROOT"]).expanduser()
    return Path(
        "/home/ubuntu/project/StreamVLN/experiments_ext/"
        "videollm_comp_training_aware_keep_ratio_formal_runs/latest_suite/checkpoints"
    )


def _train_root_for_checkpoint_root(checkpoint_root: Path) -> Path:
    if checkpoint_root.name == "checkpoints":
        return checkpoint_root.parent
    return checkpoint_root


def _grid_side_for_llamavid(keep_ratio: float) -> int:
    mapping = {0.3: 7, 0.5: 9, 0.7: 11}
    try:
        return mapping[round(float(keep_ratio), 1)]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLaMAVID keep_ratio: {keep_ratio}") from exc


def _query_num_for_longvu(keep_ratio: float) -> int:
    mapping = {0.3: 49, 0.5: 81, 0.7: 121}
    try:
        return mapping[round(float(keep_ratio), 1)]
    except KeyError as exc:
        raise ValueError(f"Unsupported LongVU keep_ratio: {keep_ratio}") from exc


def _flags_for(method: str, keep_ratio: float) -> Dict[str, object]:
    dino_path = os.environ.get("DINO_PATH", DEFAULT_DINO_PATH)
    bert_path = os.environ.get("BERT_PATH", DEFAULT_BERT_PATH)
    if method == "llamavid":
        grid_side = _grid_side_for_llamavid(keep_ratio)
        return {
            "enable_training_aware_video_compressor": True,
            "training_aware_video_compressor_type": "llamavid",
            "training_aware_compress_memory": True,
            "video_token_target_keep_ratio": keep_ratio,
            "llamavid_bert_model_name": bert_path,
            "llamavid_num_query": 32,
            "llamavid_compress_type": f"grid:{grid_side}",
            "llamavid_qformer_hidden_size": 768,
            "llamavid_qformer_depth": 2,
            "llamavid_num_heads": 8,
            "enable_video_token_compressor": False,
        }
    if method == "longvu":
        query_num = _query_num_for_longvu(keep_ratio)
        return {
            "enable_training_aware_video_compressor": True,
            "training_aware_video_compressor_type": "longvu",
            "training_aware_compress_memory": True,
            "video_token_target_keep_ratio": keep_ratio,
            "longvu_dino_tower": dino_path,
            "longvu_use_dino": True,
            "longvu_select_current_frames": False,
            "longvu_query_num": query_num,
            "longvu_connector_depth": 1,
            "longvu_window_size": 8,
            "longvu_threshold": 0.83,
            "longvu_vision_hidden_size": 1024,
            "longvu_num_heads": 8,
            "enable_video_token_compressor": False,
        }
    raise ValueError(f"Unsupported method: {method}")


def _candidate_for(method: str, keep_ratio: float) -> str:
    tag = _ratio_tag(keep_ratio)
    if method == "llamavid":
        return f"llamavid_keep_ratio_{tag}_q32_grid{_grid_side_for_llamavid(keep_ratio)}_r2r5k"
    if method == "longvu":
        return f"longvu_keep_ratio_{tag}_q{_query_num_for_longvu(keep_ratio)}_d1_r2r5k"
    raise ValueError(f"Unsupported method: {method}")


def _actual_tokens_per_frame(method: str, keep_ratio: float) -> int:
    if method == "llamavid":
        side = _grid_side_for_llamavid(keep_ratio)
        return side * side + 1
    if method == "longvu":
        return _query_num_for_longvu(keep_ratio)
    raise ValueError(f"Unsupported method: {method}")


def build_configs(methods: Iterable[str], keep_ratios: Iterable[float]) -> List[EvalConfig]:
    configs: List[EvalConfig] = []
    for method in methods:
        if method not in {"llamavid", "longvu"}:
            raise ValueError(f"Unsupported method: {method}")
        for keep_ratio in keep_ratios:
            tokens = _actual_tokens_per_frame(method, keep_ratio)
            configs.append(
                EvalConfig(
                    method=method,
                    keep_ratio=keep_ratio,
                    candidate=_candidate_for(method, keep_ratio),
                    actual_tokens_per_frame=tokens,
                    actual_keep_ratio=float(tokens / BASELINE_TOKENS_PER_FRAME),
                    model_max_length=8192 if method == "longvu" else 4096,
                    flags=_flags_for(method, keep_ratio),
                )
            )
    return configs


def _checkpoint_path(cfg: EvalConfig) -> Path:
    return _method_checkpoint_root(cfg.method) / cfg.candidate


def _flags_for_checkpoint(cfg: EvalConfig, checkpoint_path: Path) -> Dict[str, object]:
    flags_path = checkpoint_path / "flags.json"
    if flags_path.is_file():
        return _read_json(flags_path)
    return cfg.flags


def eval_command(output_dir: Path, checkpoint_path: Path, cfg: EvalConfig, baseline_tokens: float) -> List[str]:
    cmd = [
        sys.executable,
        "streamvln_ext/entrypoints/eval_offline_action_acc_tf_ext.py",
        "--model_path",
        str(checkpoint_path),
        "--base_model_path",
        os.environ.get("BASE_MODEL_PATH", DEFAULT_BASE_MODEL_PATH),
        "--dataset_root",
        os.environ.get("EVAL_SUBSET_ROOT", DEFAULT_EVAL_SUBSET_ROOT),
        "--output_path",
        str(output_dir),
        "--num_history",
        os.environ.get("NUM_HISTORY", "8"),
        "--model_max_length",
        str(cfg.model_max_length),
        "--max_episodes",
        os.environ.get("MAX_EPISODES", "240"),
        "--eval_protocol",
        "aurora_replay_gt",
        "--aurora_decode_max_new_tokens",
        os.environ.get("AURORA_DECODE_MAX_NEW_TOKENS", "1"),
        "--aurora_batch_size",
        os.environ.get("AURORA_BATCH_SIZE", "1"),
        "--aurora_step_mode",
        os.environ.get("AURORA_STEP_MODE", "next_token_logits"),
        "--aurora_vision_batch_size",
        os.environ.get("AURORA_VISION_BATCH_SIZE", "1"),
        "--baseline_avg_total_tokens_per_step",
        f"{baseline_tokens:.12f}",
    ]
    if _bool_env("AURORA_PRECOMPUTE_VISION", False):
        cmd.append("--aurora_precompute_vision")
    if _bool_env("SAVE_STEP_DEBUG", False):
        cmd.extend(
            [
                "--save_step_debug",
                "--debug_max_steps_per_episode",
                os.environ.get("DEBUG_MAX_STEPS_PER_EPISODE", "8"),
            ]
        )
    return cmd


def _status_from_summary(summary_path: Path) -> str:
    if not summary_path.is_file():
        return "missing_summary"
    data = _read_json(summary_path)
    if data.get("eval_protocol_name") != "AuroraReplay-GT":
        return "bad_protocol"
    if int(data.get("num_episodes_eval", 0) or 0) != int(os.environ.get("MAX_EPISODES", "240")):
        return "bad_episode_count"
    if int(data.get("num_actions_compared", 0) or 0) <= 0:
        return "no_actions"
    return "ok"


def _gpu_slots() -> List[Dict[str, str]]:
    gpus = _split_words(os.environ.get("GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")))
    if not gpus:
        gpus = ["0"]
    runs_per_gpu = max(1, int(os.environ.get("RUNS_PER_GPU", "1")))
    slots: List[Dict[str, str]] = []
    for gpu in gpus:
        for slot_idx in range(runs_per_gpu):
            slots.append({"slot_id": f"{gpu}:{slot_idx}", "gpu_id": gpu})
    return slots


def write_run_index(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "method",
        "candidate",
        "target_keep_ratio",
        "actual_tokens_per_frame",
        "actual_keep_ratio",
        "status",
        "exit_code",
        "gpu_id",
        "slot_id",
        "train_root",
        "checkpoint_path",
        "flags_path",
        "output_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _decision_bucket(row: Dict[str, object], baseline_acc: float, baseline_invalid_rate: float) -> str:
    acc = float(row["overall_action_acc"])
    token_red = float(row["token_reduction_ratio_vs_baseline"])
    invalid_rate = float(row["invalid_prediction_rate"])
    if invalid_rate > baseline_invalid_rate + 0.01:
        return "risky"
    if acc >= baseline_acc and token_red >= 0.30:
        return "strong"
    if acc >= baseline_acc - 0.005 and token_red >= 0.45:
        return "competitive"
    if acc >= baseline_acc - 0.010 and token_red >= 0.65:
        return "compression-only"
    return "weak"


def aggregate(
    suite_root: Path,
    run_rows: List[Dict[str, object]],
    train_signature: str,
    eval_signature: str,
    baseline_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    baseline_tokens = float(baseline_summary.get("avg_total_tokens_per_step", DEFAULT_BASELINE_TOKENS))
    baseline_acc = float(baseline_summary.get("overall_action_acc", DEFAULT_BASELINE_ACC))
    baseline_invalid_rate = float(baseline_summary.get("invalid_prediction_rate", DEFAULT_BASELINE_INVALID_RATE))

    rows: List[Dict[str, object]] = []
    for run in run_rows:
        summary_path = Path(str(run["summary_path"]))
        if str(run["status"]) not in {"ok", "reused"} or not summary_path.is_file():
            continue
        data = _read_json(summary_path)
        avg_tokens = float(data.get("avg_total_tokens_per_step", 0.0) or 0.0)
        token_red = (baseline_tokens - avg_tokens) / baseline_tokens if baseline_tokens > 0 else 0.0
        row = {
            "method": run["method"],
            "candidate": run["candidate"],
            "target_keep_ratio": float(run["target_keep_ratio"]),
            "actual_tokens_per_frame": int(run["actual_tokens_per_frame"]),
            "actual_keep_ratio": float(run["actual_keep_ratio"]),
            "train_subset_signature": train_signature,
            "eval_subset_signature": eval_signature,
            "overall_action_acc": float(data.get("overall_action_acc", 0.0) or 0.0),
            "invalid_prediction_rate": float(data.get("invalid_prediction_rate", 0.0) or 0.0),
            "avg_visual_tokens_per_step": float(data.get("avg_visual_tokens_per_step", 0.0) or 0.0),
            "avg_memory_tokens_per_step": float(data.get("avg_memory_tokens_per_step", 0.0) or 0.0),
            "avg_total_tokens_per_step": avg_tokens,
            "token_reduction_ratio_vs_baseline": token_red,
            "fps": float(data.get("fps", 0.0) or 0.0),
            "latency_ms_p50": float(data.get("latency_ms_p50", 0.0) or 0.0),
            "latency_ms_p95": float(data.get("latency_ms_p95", 0.0) or 0.0),
            "gpu_peak_allocated_mib": float(data.get("gpu_peak_allocated_mib", 0.0) or 0.0),
            "gpu_peak_reserved_mib": float(data.get("gpu_peak_reserved_mib", 0.0) or 0.0),
            "num_episodes_eval": int(data.get("num_episodes_eval", 0) or 0),
            "num_actions_compared": int(data.get("num_actions_compared", 0) or 0),
            "cuda_visible_devices": str(data.get("cuda_visible_devices", "")),
            "decision_bucket": "",
            "train_root": run["train_root"],
            "checkpoint_path": run["checkpoint_path"],
            "flags_path": run["flags_path"],
            "summary_path": str(summary_path),
        }
        row["decision_bucket"] = _decision_bucket(row, baseline_acc, baseline_invalid_rate)
        rows.append(row)

    rows.sort(
        key=lambda x: (
            float(x["overall_action_acc"]),
            -float(x["invalid_prediction_rate"]),
            float(x["token_reduction_ratio_vs_baseline"]),
            -float(x["latency_ms_p95"]),
        ),
        reverse=True,
    )
    return rows


def write_overview_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fields = [
        "method",
        "candidate",
        "target_keep_ratio",
        "actual_tokens_per_frame",
        "actual_keep_ratio",
        "train_subset_signature",
        "eval_subset_signature",
        "overall_action_acc",
        "invalid_prediction_rate",
        "avg_visual_tokens_per_step",
        "avg_memory_tokens_per_step",
        "avg_total_tokens_per_step",
        "token_reduction_ratio_vs_baseline",
        "fps",
        "latency_ms_p50",
        "latency_ms_p95",
        "gpu_peak_allocated_mib",
        "gpu_peak_reserved_mib",
        "num_episodes_eval",
        "num_actions_compared",
        "cuda_visible_devices",
        "decision_bucket",
        "train_root",
        "checkpoint_path",
        "flags_path",
        "summary_path",
    ]
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_overview_md(
    path: Path,
    rows: List[Dict[str, object]],
    baseline_summary: Dict[str, object],
    train_signature: str,
    eval_signature: str,
) -> None:
    baseline_acc = float(baseline_summary.get("overall_action_acc", DEFAULT_BASELINE_ACC))
    baseline_tokens = float(baseline_summary.get("avg_total_tokens_per_step", DEFAULT_BASELINE_TOKENS))
    baseline_invalid = float(baseline_summary.get("invalid_prediction_rate", DEFAULT_BASELINE_INVALID_RATE))
    lines = [
        "# Training-Aware VideoLLM-Comp TF240 Overview",
        "",
        f"- eval protocol: `AuroraReplay-GT`",
        f"- eval subset: `{os.environ.get('EVAL_SUBSET_ROOT', DEFAULT_EVAL_SUBSET_ROOT)}`",
        f"- train subset signature: `{train_signature}`",
        f"- eval subset signature: `{eval_signature}`",
        f"- baseline acc: `{_pct(baseline_acc)}`",
        f"- baseline tokens/step: `{baseline_tokens:.2f}`",
        f"- baseline invalid rate: `{_pct(baseline_invalid)}`",
        "",
        "| rank | method | keep | actual_keep | acc | delta_vs_base | invalid | tokens/step | token_red | fps | p50 | p95 | peak_alloc_mib | bucket |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        acc = float(row["overall_action_acc"])
        lines.append(
            f"| {rank} | `{row['method']}` | {float(row['target_keep_ratio']):.1f} | "
            f"{float(row['actual_keep_ratio']):.4f} | {_pct(acc)} | "
            f"{(acc - baseline_acc) * 100.0:+.2f}pp | "
            f"{_pct(float(row['invalid_prediction_rate']))} | "
            f"{float(row['avg_total_tokens_per_step']):.2f} | "
            f"{_pct(float(row['token_reduction_ratio_vs_baseline']))} | "
            f"{float(row['fps']):.2f} | {float(row['latency_ms_p50']):.2f} | "
            f"{float(row['latency_ms_p95']):.2f} | "
            f"{float(row['gpu_peak_allocated_mib']):.1f} | `{row['decision_bucket']}` |"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "| candidate | episodes | actions | protocol_ok | summary |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        protocol_ok = (
            int(row["num_episodes_eval"]) == int(os.environ.get("MAX_EPISODES", "240"))
            and int(row["num_actions_compared"]) == 16196
        )
        lines.append(
            f"| `{row['candidate']}` | {int(row['num_episodes_eval'])} | "
            f"{int(row['num_actions_compared'])} | `{protocol_ok}` | `{row['summary_path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    suite_ts = os.environ.get("SUITE_TS") or time.strftime("%Y%m%d_%H%M%S")
    suite_root = Path(
        os.environ.get(
            "SUITE_ROOT",
            str(PROJECT_ROOT / "experiments_ext" / "videollm_comp_training_aware_tf240_eval_runs" / suite_ts),
        )
    )
    suite_root.mkdir(parents=True, exist_ok=True)
    latest_link = suite_root.parent / "latest_suite"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(suite_root)
    except OSError as exc:
        print(f"[warn] could not update latest_suite symlink: {exc}", flush=True)

    methods = _split_words(os.environ.get("METHODS", "llamavid longvu"))
    keep_ratios = [float(x) for x in _split_words(os.environ.get("KEEP_RATIOS", "0.3 0.5 0.7"))]
    configs = build_configs(methods, keep_ratios)
    reuse = _bool_env("REUSE_IF_EXISTS", True)
    fail_fast = _bool_env("FAIL_FAST", True)
    baseline_summary_path = Path(os.environ.get("BASELINE_SUMMARY", DEFAULT_BASELINE_SUMMARY))
    baseline_summary = _maybe_read_json(baseline_summary_path)
    baseline_tokens = float(baseline_summary.get("avg_total_tokens_per_step", DEFAULT_BASELINE_TOKENS))
    train_signature = _subset_signature(Path(os.environ.get("TRAIN_INDICES_JSON", DEFAULT_TRAIN_INDICES_JSON)))
    eval_signature = _subset_signature(Path(os.environ.get("EVAL_INDICES_JSON", DEFAULT_EVAL_INDICES_JSON)))

    print(f"[suite] root={suite_root}", flush=True)
    print(f"[suite] methods={' '.join(methods)} keep_ratios={keep_ratios}", flush=True)
    print(f"[suite] eval_subset={os.environ.get('EVAL_SUBSET_ROOT', DEFAULT_EVAL_SUBSET_ROOT)}", flush=True)
    print(f"[suite] train_signature={train_signature} eval_signature={eval_signature}", flush=True)
    print(f"[suite] baseline_summary={baseline_summary_path}", flush=True)

    gpu_slots = _gpu_slots()
    max_parallel = max(1, min(int(os.environ.get("MAX_PARALLEL", str(len(gpu_slots)))), len(gpu_slots)))
    pending = list(configs)
    running: Dict[subprocess.Popen[str], Dict[str, object]] = {}
    run_rows: List[Dict[str, object]] = []

    while pending or running:
        active_slot_ids = {str(info["slot_id"]) for info in running.values()}
        available_slots = [slot for slot in gpu_slots if slot["slot_id"] not in active_slot_ids]
        while pending and len(running) < max_parallel and available_slots:
            cfg = pending.pop(0)
            slot = available_slots.pop(0)
            gpu_id = slot["gpu_id"]
            slot_id = slot["slot_id"]
            checkpoint_path = _checkpoint_path(cfg)
            flags_path = checkpoint_path / "flags.json"
            output_dir = suite_root / "eval" / f"{cfg.candidate}_teacher_forcing_240"
            summary_path = output_dir / "summary.json"
            train_root = _train_root_for_checkpoint_root(_method_checkpoint_root(cfg.method))

            base_row = {
                "method": cfg.method,
                "candidate": cfg.candidate,
                "target_keep_ratio": cfg.keep_ratio,
                "actual_tokens_per_frame": cfg.actual_tokens_per_frame,
                "actual_keep_ratio": cfg.actual_keep_ratio,
                "gpu_id": gpu_id,
                "slot_id": slot_id,
                "train_root": str(train_root),
                "checkpoint_path": str(checkpoint_path),
                "flags_path": str(flags_path),
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
            }

            if not checkpoint_path.is_dir():
                row = {**base_row, "status": "missing_checkpoint", "exit_code": 1}
                run_rows.append(row)
                print(f"[missing] {cfg.candidate}: {checkpoint_path}", flush=True)
                if fail_fast:
                    write_run_index(suite_root / "run_index.csv", run_rows)
                    return 1
                continue

            if reuse and summary_path.is_file():
                status = _status_from_summary(summary_path)
                row = {**base_row, "status": "reused" if status == "ok" else status, "exit_code": 0 if status == "ok" else 1}
                run_rows.append(row)
                print(f"[reuse] {cfg.candidate}: {summary_path} status={status}", flush=True)
                if fail_fast and status != "ok":
                    write_run_index(suite_root / "run_index.csv", run_rows)
                    return 1
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["STREAMVLN_EXT_FLAGS"] = _json_dumps(_flags_for_checkpoint(cfg, checkpoint_path))
            cmd = eval_command(output_dir, checkpoint_path, cfg, baseline_tokens)
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
                **base_row,
                "log": log,
                "started": time.time(),
            }
            print(f"[start] {cfg.candidate} gpu={gpu_id} slot={slot_id} output={output_dir}", flush=True)

        if not running:
            continue

        done = [proc for proc in running if proc.poll() is not None]
        if not done:
            active = ", ".join(
                f"{info['candidate']}@gpu{info['gpu_id']}/slot{info['slot_id']}" for info in running.values()
            )
            print(f"[heartbeat] running={active}", flush=True)
            time.sleep(30)
            continue

        for proc in done:
            info = running.pop(proc)
            log = info.pop("log")
            log.close()
            elapsed = int(time.time() - float(info.pop("started")))
            output_dir = Path(str(info["output_dir"]))
            (output_dir / "timing.txt").write_text(f"eval_elapsed_seconds={elapsed}\n", encoding="utf-8")
            code = int(proc.returncode if proc.returncode is not None else proc.wait())
            status = "ok" if code == 0 and _status_from_summary(Path(str(info["summary_path"]))) == "ok" else "failed"
            row = {**info, "status": status, "exit_code": code}
            run_rows.append(row)
            print(
                f"[done] {info['candidate']} status={status} code={code} seconds={elapsed}",
                flush=True,
            )
            if fail_fast and status != "ok":
                write_run_index(suite_root / "run_index.csv", run_rows)
                return code or 1

    write_run_index(suite_root / "run_index.csv", run_rows)
    failed = [row for row in run_rows if row["status"] not in {"ok", "reused"}]
    if failed:
        print(f"[failed] {len(failed)} runs failed; see run_index.csv", flush=True)
        if fail_fast:
            return 1

    overview_rows = aggregate(suite_root, run_rows, train_signature, eval_signature, baseline_summary)
    csv_path = suite_root / "training_aware_tf240_overview.csv"
    md_path = suite_root / "training_aware_tf240_overview.md"
    write_overview_csv(csv_path, overview_rows)
    write_overview_md(md_path, overview_rows, baseline_summary, train_signature, eval_signature)
    print(f"[wrote] {csv_path}", flush=True)
    print(f"[wrote] {md_path}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
