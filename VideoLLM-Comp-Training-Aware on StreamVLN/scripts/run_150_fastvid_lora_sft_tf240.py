#!/usr/bin/env python3
"""Run FastVid training-free compression + LoRA SFT and TF240 evaluation.

The suite trains one LoRA adapter for each FastVid keep ratio on the fixed
R2R-5k subset, evaluates each adapter on TF240 AuroraReplay-GT with matching
FastVid flags, and writes compact artifacts for report updates.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path("/home/ubuntu/project/StreamVLN")
SUITE_PARENT = PROJECT_ROOT / "experiments_ext" / "videollm_comp_fastvid_sft_tf240_runs"
MODEL_PARENT = Path("/home/ubuntu/model")

DEFAULT_TRAIN_ROOT = (
    PROJECT_ROOT
    / "experiments_ext"
    / "videollm_comp_training_aware_fixed_train_subsets"
    / "latest_suite"
    / "train_fixed_r2r_5k"
)
DEFAULT_TRAIN_INDICES = (
    PROJECT_ROOT
    / "experiments_ext"
    / "videollm_comp_training_aware_fixed_train_subsets"
    / "latest_suite"
    / "train_fixed_r2r_5k_indices.json"
)
DEFAULT_EVAL_ROOT = (
    PROJECT_ROOT
    / "experiments_ext"
    / "fixed_eval_subsets"
    / "phase0_20260421_230201"
    / "teacher_forcing_240"
)
DEFAULT_EVAL_INDICES = (
    PROJECT_ROOT
    / "experiments_ext"
    / "fixed_eval_subsets"
    / "phase0_20260421_230201"
    / "teacher_forcing_240_indices.json"
)
DEFAULT_BASE_MODEL = Path("/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3")
DEFAULT_TRAINING_FREE_FASTVID_ROOT = (
    PROJECT_ROOT
    / "experiments_ext"
    / "videollm_comp_full_eval_tf_runs"
    / "20260507_fixed_tf240_aurora_promptfix_oom_rerun_lowmem"
)
DEFAULT_BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "experiments_ext"
    / "videollm_comp_full_eval_tf_runs"
    / "20260507_fixed_tf240_aurora_promptfix"
    / "baseline"
    / "summary.json"
)

EXPECTED_TRAIN_SIGNATURE = "9a7056bbf4f395ac"
EXPECTED_EVAL_SIGNATURE = "b76cf3ac8c8f8afa"
BASELINE_AVG_TOTAL_TOKENS = 1736.783835514942
BASELINE_ACC = 0.3362558656458385
KEEP_RATIOS = (0.3, 0.5, 0.7)

BASE_DISABLED_FLAGS: Dict[str, object] = {
    "enable_training_aware_video_compressor": False,
    "enable_sliding_kv": False,
    "enable_memory_loss": False,
    "enable_voxel_proxy": False,
    "enable_token_selection": False,
    "enable_dynamic_memory": False,
    "enable_multiscale_memory": False,
    "enable_voxel_rgbd": False,
    "enable_voxel_spatial_pruning": False,
    "enable_hc_st_pruning": False,
    "enable_tuning_free_mm_pruning": False,
    "enable_tome_visual_merge": False,
}


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_words(raw: str) -> List[str]:
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def _ratio_tag(ratio: float) -> str:
    return f"{ratio:.1f}".replace(".", "_")


def _candidate_name(ratio: float) -> str:
    return f"fastvid_keep_ratio_{_ratio_tag(ratio)}"


def _json_dumps(data: Dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _delta_pp(value: float, base: float) -> str:
    return f"{(value - base) * 100.0:+.2f}pp"


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def fastvid_flags(ratio: float) -> Dict[str, object]:
    flags: Dict[str, object] = {
        **BASE_DISABLED_FLAGS,
        "enable_video_token_compressor": True,
        "video_token_compressor_type": "fastvid",
        "video_token_target_keep_ratio": ratio,
        "fastvid_retention_ratio": ratio,
        "fastvid_dyseg_c": 8,
        "fastvid_dyseg_tau": 0.9,
        "fastvid_stprune_d": 0.4,
        "fastvid_dtm_p": 4,
        "fastvid_dtm_beta": 0.6,
        "fastvid_score_type": "attn_proxy",
    }
    return flags


def _load_annotations_videos(root: Path) -> List[str]:
    annotations_path = root / "annotations.json"
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    videos = []
    for item in data:
        video = str(item.get("video", "")).strip()
        if video:
            videos.append(video)
    return videos


def _signature(path: Path) -> str:
    return str(_read_json(path).get("subset_signature", ""))


def _symlink_force(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or not link_path.exists():
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_path)
        return
    print(f"[warn] latest link exists and is not a symlink, leaving untouched: {link_path}", flush=True)


def _tail(path: Path, max_lines: int = 12) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _gpu_ids() -> List[str]:
    raw = os.environ.get("GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1"
    ids = _split_words(raw)
    return ids or ["0"]


def _gpu_uuid_map() -> Dict[str, str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            mapping[parts[1]] = parts[0]
    return mapping


def _compute_app_gpu_by_pid() -> Dict[int, str]:
    uuid_to_index = _gpu_uuid_map()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}
    mapping: Dict[int, str] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        mapping[pid] = uuid_to_index.get(parts[1], "")
    return mapping


def _running_train_pids_for_output(output_dir: Path) -> List[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except Exception:
        return []
    output_text = str(output_dir)
    pids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, cmd = line.partition(" ")
        if "streamvln_ext/entrypoints/train_ext.py" not in cmd:
            continue
        if output_text not in cmd:
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            continue
    return pids


def _external_gpu_for_pids(pids: List[int]) -> str:
    pid_to_gpu = _compute_app_gpu_by_pid()
    for pid in pids:
        gpu = pid_to_gpu.get(pid, "")
        if gpu:
            return gpu
    return os.environ.get("TRAIN_GPU_ID", "0")


def _run_logged(
    *,
    name: str,
    cmd: List[str],
    env: Dict[str, str],
    log_path: Path,
    timing_path: Path,
    heartbeat_sec: int = 30,
) -> int:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(cmd) + "\n")
        if "STREAMVLN_EXT_FLAGS" in env:
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
        while proc.poll() is None:
            elapsed = int(time.time() - started)
            print(f"[heartbeat] {name} running elapsed_sec={elapsed}", flush=True)
            time.sleep(heartbeat_sec)
        code = int(proc.returncode if proc.returncode is not None else proc.wait())

    elapsed = int(time.time() - started)
    timing_path.write_text(f"elapsed_seconds={elapsed}\nexit_code={code}\n", encoding="utf-8")
    print(f"[done] {name} code={code} elapsed_sec={elapsed}", flush=True)
    if code != 0:
        print(f"[tail] {log_path}\n{_tail(log_path)}", flush=True)
    return code


def _launch_logged(
    *,
    name: str,
    cmd: List[str],
    env: Dict[str, str],
    log_path: Path,
) -> Tuple[subprocess.Popen[str], object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write("[command] " + " ".join(cmd) + "\n")
    if "STREAMVLN_EXT_FLAGS" in env:
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
    print(f"[start] {name} pid={proc.pid}", flush=True)
    return proc, log


def build_train_command(base_model: Path, train_root: Path, output_dir: Path, max_steps: int) -> List[str]:
    return [
        sys.executable,
        "streamvln_ext/entrypoints/train_ext.py",
        "--model_name_or_path",
        str(base_model),
        "--video_folder",
        str(train_root),
        "--version",
        "qwen_1_5",
        "--num_history",
        "8",
        "--num_future_steps",
        "4",
        "--num_frames",
        "32",
        "--data_augmentation",
        "False",
        "--mm_tunable_parts",
        "mm_lora_layer",
        "--vision_tower",
        "google/siglip-so400m-patch14-384",
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "anyres_max_9",
        "--image_grid_pinpoints",
        "(1x1),...,(6x6)",
        "--bf16",
        "True",
        "--output_dir",
        str(output_dir),
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "4",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "no",
        "--learning_rate",
        "1e-4",
        "--weight_decay",
        "0.0",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "10",
        "--seed",
        "12042",
        "--data_seed",
        "12042",
        "--tf32",
        "True",
        "--model_max_length",
        "32768",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "2",
        "--lazy_preprocess",
        "True",
        "--dataloader_drop_last",
        "True",
        "--max_steps",
        str(max_steps),
        "--report_to",
        "none",
        "--attn_implementation",
        "sdpa",
        "--lora_enable",
        "True",
        "--lora_r",
        "64",
        "--lora_alpha",
        "16",
        "--lora_dropout",
        "0.05",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        "--lora_include_mm_projector",
        "True",
        "--lora_include_vision_tower",
        "True",
        "--lora_vision_tower_last_n_layers",
        "2",
    ]


def build_eval_command(base_model: Path, eval_root: Path, checkpoint_dir: Path, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        "streamvln_ext/entrypoints/eval_offline_action_acc_tf_ext.py",
        "--model_path",
        str(checkpoint_dir),
        "--base_model_path",
        str(base_model),
        "--dataset_root",
        str(eval_root),
        "--output_path",
        str(output_dir),
        "--num_history",
        "8",
        "--model_max_length",
        os.environ.get("EVAL_MODEL_MAX_LENGTH", "4096"),
        "--max_episodes",
        "240",
        "--eval_protocol",
        "aurora_replay_gt",
        "--aurora_step_mode",
        "next_token_logits",
        "--aurora_decode_max_new_tokens",
        "1",
        "--aurora_batch_size",
        "1",
        "--aurora_vision_batch_size",
        "1",
        "--baseline_avg_total_tokens_per_step",
        f"{BASELINE_AVG_TOTAL_TOKENS:.12f}",
    ]


def common_env(gpu_id: str, flags: Dict[str, object]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["STREAMVLN_EXT_FLAGS"] = _json_dumps(flags)
    return env


def final_train_loss(trainer_state_path: Path) -> Optional[float]:
    if not trainer_state_path.is_file():
        return None
    state = _read_json(trainer_state_path)
    for row in reversed(state.get("log_history", []) or []):
        if "train_loss" in row:
            return float(row["train_loss"])
    return None


def train_validation(checkpoint_dir: Path, train_code: int) -> Dict[str, object]:
    loss = final_train_loss(checkpoint_dir / "trainer_state.json")
    loss_ok = loss is not None and math.isfinite(loss)
    log_text = (checkpoint_dir / "train.log").read_text(encoding="utf-8", errors="replace") if (checkpoint_dir / "train.log").is_file() else ""
    result = {
        "exit_code": train_code,
        "exit_code_ok": train_code == 0,
        "adapter_config_exists": (checkpoint_dir / "adapter_config.json").is_file(),
        "trainer_state_exists": (checkpoint_dir / "trainer_state.json").is_file(),
        "final_train_loss": loss,
        "final_train_loss_ok": loss_ok,
        "train_log_has_fastvid_flags": "fastvid" in log_text and (
            "STREAMVLN_EXT_FLAGS" in log_text or "[flags]" in log_text or "video_token_compressor_type" in log_text
        ),
    }
    result["ok"] = bool(
        result["exit_code_ok"]
        and result["adapter_config_exists"]
        and result["trainer_state_exists"]
        and result["final_train_loss_ok"]
        and result["train_log_has_fastvid_flags"]
    )
    return result


def eval_validation(summary_path: Path, eval_code: int) -> Dict[str, object]:
    result: Dict[str, object] = {
        "exit_code": eval_code,
        "exit_code_ok": eval_code == 0,
        "summary_exists": summary_path.is_file(),
    }
    if summary_path.is_file():
        summary = _read_json(summary_path)
        required_fields = [
            "overall_action_acc",
            "num_invalid_predictions",
            "avg_total_tokens_per_step",
            "fps",
        ]
        result.update(
            {
                "num_episodes_eval": int(summary.get("num_episodes_eval", 0) or 0),
                "num_actions_compared": int(summary.get("num_actions_compared", 0) or 0),
                "eval_protocol_name": summary.get("eval_protocol_name", ""),
                "required_fields_present": all(field in summary for field in required_fields),
            }
        )
    result["ok"] = bool(
        result.get("exit_code_ok")
        and result.get("summary_exists")
        and result.get("num_episodes_eval") == 240
        and result.get("num_actions_compared") == 16196
        and result.get("eval_protocol_name") == "AuroraReplay-GT"
        and result.get("required_fields_present")
    )
    return result


def load_summary_row(
    *,
    kind: str,
    candidate: str,
    keep_ratio: float,
    checkpoint: str,
    summary_path: Path,
) -> Dict[str, object]:
    summary = _read_json(summary_path)
    acc = float(summary.get("overall_action_acc", 0.0) or 0.0)
    tokens = float(summary.get("avg_total_tokens_per_step", 0.0) or 0.0)
    token_reduction = (BASELINE_AVG_TOTAL_TOKENS - tokens) / BASELINE_AVG_TOTAL_TOKENS if BASELINE_AVG_TOTAL_TOKENS else 0.0
    invalid_rate = float(summary.get("invalid_prediction_rate", 0.0) or 0.0)
    invalid_count = int(summary.get("num_invalid_predictions", 0) or 0)
    return {
        "kind": kind,
        "candidate": candidate,
        "keep_ratio": keep_ratio,
        "overall_action_acc": acc,
        "delta_vs_base": acc - BASELINE_ACC,
        "invalid_prediction_rate": invalid_rate,
        "num_invalid_predictions": invalid_count,
        "avg_total_tokens_per_step": tokens,
        "token_reduction_ratio_vs_baseline": token_reduction,
        "fps": float(summary.get("fps", 0.0) or 0.0),
        "checkpoint": checkpoint,
        "summary_source": str(summary_path),
    }


def write_train_overview(suite_root: Path, train_rows: List[Dict[str, object]]) -> None:
    csv_path = suite_root / "fastvid_lora_sft_train_overview.csv"
    fields = [
        "candidate",
        "target_keep_ratio",
        "status",
        "exit_code",
        "final_train_loss",
        "train_runtime_sec",
        "train_steps_per_second",
        "checkpoint_path",
        "flags_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in train_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# FastVid LoRA SFT Train Overview",
        "",
        f"- suite_root: `{suite_root}`",
        "",
        "| candidate | keep | status | final train loss | runtime sec | steps/sec | checkpoint |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in train_rows:
        loss = row.get("final_train_loss")
        loss_text = "-" if loss is None else f"{float(loss):.4f}"
        runtime = row.get("train_runtime_sec")
        runtime_text = "-" if runtime in {None, ""} else f"{float(runtime):.1f}"
        steps_per_sec = row.get("train_steps_per_second")
        sps_text = "-" if steps_per_sec in {None, ""} else f"{float(steps_per_sec):.3f}"
        lines.append(
            f"| `{row['candidate']}` | {float(row['target_keep_ratio']):.1f} | `{row['status']}` | "
            f"{loss_text} | {runtime_text} | {sps_text} | `{row['checkpoint_path']}` |"
        )
    (suite_root / "fastvid_lora_sft_train_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_overview(suite_root: Path, eval_rows: List[Dict[str, object]]) -> None:
    csv_path = suite_root / "fastvid_lora_sft_tf240_overview.csv"
    fields = [
        "kind",
        "candidate",
        "keep_ratio",
        "overall_action_acc",
        "delta_vs_base",
        "invalid_prediction_rate",
        "num_invalid_predictions",
        "avg_total_tokens_per_step",
        "token_reduction_ratio_vs_baseline",
        "fps",
        "checkpoint",
        "summary_source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# FastVid LoRA SFT TF240 Overview",
        "",
        f"- suite_root: `{suite_root}`",
        f"- baseline acc: `{_pct(BASELINE_ACC)}`",
        f"- baseline tokens/step: `{BASELINE_AVG_TOTAL_TOKENS:.2f}`",
        "",
        "| kind | candidate | keep | acc | delta vs base | invalid | tokens/step | token reduction | fps | checkpoint | summary source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in eval_rows:
        lines.append(
            f"| `{row['kind']}` | `{row['candidate']}` | {float(row['keep_ratio']):.1f} | "
            f"{_pct(float(row['overall_action_acc']))} | {_delta_pp(float(row['overall_action_acc']), BASELINE_ACC)} | "
            f"{_pct(float(row['invalid_prediction_rate']))} ({int(row['num_invalid_predictions'])}) | "
            f"{float(row['avg_total_tokens_per_step']):.2f} | "
            f"{_pct(float(row['token_reduction_ratio_vs_baseline']))} | "
            f"{float(row['fps']):.2f} | `{row['checkpoint']}` | `{row['summary_source']}` |"
        )
    (suite_root / "fastvid_lora_sft_tf240_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_section(
    *,
    suite_root: Path,
    model_root: Path,
    train_root: Path,
    base_model: Path,
    train_rows: List[Dict[str, object]],
    comparison_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "## FastVid LoRA SFT on R2R-5k",
        "",
        "- FastVid remains training-free: it has no trainable compressor parameters. This experiment measures whether LoRA SFT helps the base StreamVLN model recover under the FastVid-compressed input distribution.",
        f"- Suite root: `{suite_root}`",
        f"- Train subset: `{train_root}`",
        f"- Train subset signature: `{EXPECTED_TRAIN_SIGNATURE}`",
        f"- Eval subset: `{DEFAULT_EVAL_ROOT}`",
        f"- Eval subset signature: `{EXPECTED_EVAL_SIGNATURE}`",
        f"- Base model: `{base_model}`",
        f"- Checkpoint root: `{model_root}`",
        "- LoRA: `r=64`, `alpha=16`, `dropout=0.05`; targets `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`; includes mm projector and last 2 vision tower layers.",
        "- Training: FastVid keep ratios `0.3/0.5/0.7`, R2R-5k, `max_steps=1000`, batch size 1, gradient accumulation 4, SDPA, bf16.",
        "- Evaluation: TF240 AuroraReplay-GT, `next_token_logits`, `max_new_tokens=1`, no vision precompute, one matched FastVid config per checkpoint.",
        "",
        "| keep | checkpoint | final train loss |",
        "|---:|---|---:|",
    ]
    for row in train_rows:
        loss = row.get("final_train_loss")
        loss_text = "-" if loss is None else f"{float(loss):.4f}"
        lines.append(
            f"| {float(row['target_keep_ratio']):.1f} | `{row['checkpoint_path']}` | {loss_text} |"
        )
    lines.extend(
        [
            "",
            "| method | keep | Acc | Delta vs Base | Invalid | Tokens/step | Token reduction | FPS | checkpoint | summary source |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| `{row['kind']}` | {float(row['keep_ratio']):.1f} | "
            f"{_pct(float(row['overall_action_acc']))} | {_delta_pp(float(row['overall_action_acc']), BASELINE_ACC)} | "
            f"{_pct(float(row['invalid_prediction_rate']))} ({int(row['num_invalid_predictions'])}) | "
            f"{float(row['avg_total_tokens_per_step']):.2f} | "
            f"{_pct(float(row['token_reduction_ratio_vs_baseline']))} | "
            f"{float(row['fps']):.2f} | `{row['checkpoint']}` | `{row['summary_source']}` |"
        )
    lines.append("")
    (suite_root / "fastvid_lora_sft_report_section.md").write_text("\n".join(lines), encoding="utf-8")


def trainer_stats(checkpoint_dir: Path) -> Dict[str, object]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        return {}
    state = _read_json(state_path)
    for row in reversed(state.get("log_history", []) or []):
        if "train_loss" in row:
            return {
                "final_train_loss": float(row.get("train_loss", 0.0)),
                "train_runtime_sec": float(row.get("train_runtime", 0.0) or 0.0),
                "train_steps_per_second": float(row.get("train_steps_per_second", 0.0) or 0.0),
            }
    return {}


def preflight(
    *,
    suite_root: Path,
    train_root: Path,
    train_indices: Path,
    eval_root: Path,
    eval_indices: Path,
    base_model: Path,
) -> Dict[str, object]:
    checks: Dict[str, object] = {
        "train_root": str(train_root),
        "train_indices": str(train_indices),
        "eval_root": str(eval_root),
        "eval_indices": str(eval_indices),
        "base_model": str(base_model),
    }
    checks["train_root_exists"] = train_root.is_dir()
    checks["eval_root_exists"] = eval_root.is_dir()
    checks["base_model_exists"] = base_model.is_dir()
    train_sig = _signature(train_indices)
    eval_sig = _signature(eval_indices)
    checks["train_subset_signature"] = train_sig
    checks["eval_subset_signature"] = eval_sig
    checks["train_signature_ok"] = train_sig == EXPECTED_TRAIN_SIGNATURE
    checks["eval_signature_ok"] = eval_sig == EXPECTED_EVAL_SIGNATURE
    train_videos = set(_load_annotations_videos(train_root))
    eval_videos = set(_load_annotations_videos(eval_root))
    overlap = sorted(train_videos.intersection(eval_videos))
    checks["train_video_count"] = len(train_videos)
    checks["eval_video_count"] = len(eval_videos)
    checks["video_overlap_count"] = len(overlap)
    checks["video_overlap_sample"] = overlap[:10]
    checks["ok"] = bool(
        checks["train_root_exists"]
        and checks["eval_root_exists"]
        and checks["base_model_exists"]
        and checks["train_signature_ok"]
        and checks["eval_signature_ok"]
        and len(overlap) == 0
    )
    _write_json(suite_root / "validation_summary.json", {"preflight": checks})
    if not checks["ok"]:
        raise RuntimeError(f"Preflight failed: {json.dumps(checks, ensure_ascii=False, indent=2)}")
    return checks


def main() -> int:
    suite_ts = os.environ.get("SUITE_TS") or time.strftime("%Y%m%d_%H%M%S")
    suite_root = _env_path("SUITE_ROOT", SUITE_PARENT / suite_ts)
    model_root = _env_path("CHECKPOINT_ROOT", MODEL_PARENT / f"StreamVLN_fastvid_lora_sft_r2r5k_{suite_ts}")
    train_root = _env_path("TRAIN_SUBSET_ROOT", DEFAULT_TRAIN_ROOT)
    train_indices = _env_path("TRAIN_INDICES_JSON", DEFAULT_TRAIN_INDICES)
    eval_root = _env_path("EVAL_SUBSET_ROOT", DEFAULT_EVAL_ROOT)
    eval_indices = _env_path("EVAL_INDICES_JSON", DEFAULT_EVAL_INDICES)
    base_model = _env_path("BASE_MODEL_PATH", DEFAULT_BASE_MODEL)
    training_free_root = _env_path("TRAINING_FREE_FASTVID_ROOT", DEFAULT_TRAINING_FREE_FASTVID_ROOT)
    max_steps = int(os.environ.get("MAX_STEPS", "1000"))
    train_gpus = _split_words(os.environ.get("TRAIN_GPU_IDS", "")) or _gpu_ids()
    eval_gpus = _split_words(os.environ.get("EVAL_GPU_IDS", "")) or _gpu_ids()
    train_max_parallel = max(1, min(int(os.environ.get("TRAIN_MAX_PARALLEL", str(len(train_gpus)))), len(train_gpus)))
    eval_runs_per_gpu = max(1, int(os.environ.get("EVAL_RUNS_PER_GPU", "1")))
    eval_slots = [
        {"gpu_id": gpu_id, "slot_id": f"{gpu_id}:{slot_idx}"}
        for gpu_id in eval_gpus
        for slot_idx in range(eval_runs_per_gpu)
    ]
    eval_max_parallel = max(1, min(int(os.environ.get("EVAL_MAX_PARALLEL", str(len(eval_slots)))), len(eval_slots)))
    reuse = _bool_env("REUSE_IF_EXISTS", True)
    fail_fast = _bool_env("FAIL_FAST", True)

    SUITE_PARENT.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    suite_root.mkdir(parents=True, exist_ok=True)
    _symlink_force(SUITE_PARENT / "latest_suite", suite_root)
    _symlink_force(MODEL_PARENT / "StreamVLN_fastvid_lora_sft_r2r5k_latest", model_root)

    run_index_path = suite_root / "run_index.csv"
    if run_index_path.exists() and not reuse:
        run_index_path.unlink()
    run_index_fields = [
        "phase",
        "candidate",
        "target_keep_ratio",
        "status",
        "exit_code",
        "gpu_id",
        "checkpoint_path",
        "output_dir",
        "flags_path",
        "summary_path",
    ]

    print(f"[suite] suite_root={suite_root}", flush=True)
    print(f"[suite] checkpoint_root={model_root}", flush=True)
    print(
        f"[suite] train_gpus={','.join(train_gpus)} train_max_parallel={train_max_parallel} "
        f"eval_gpus={','.join(eval_gpus)} eval_runs_per_gpu={eval_runs_per_gpu} "
        f"eval_max_parallel={eval_max_parallel} max_steps={max_steps}",
        flush=True,
    )

    validation: Dict[str, object] = {
        "preflight": preflight(
            suite_root=suite_root,
            train_root=train_root,
            train_indices=train_indices,
            eval_root=eval_root,
            eval_indices=eval_indices,
            base_model=base_model,
        ),
        "train": {},
        "eval": {},
    }

    train_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []

    pending_train = list(KEEP_RATIOS)
    running_train: Dict[str, Dict[str, object]] = {}

    def finalize_train(
        *,
        candidate: str,
        ratio: float,
        checkpoint_dir: Path,
        flags_path: Path,
        status: str,
        code: int,
        gpu_id: str,
        slot_id: str = "",
    ) -> bool:
        train_check = train_validation(checkpoint_dir, code)
        if status == "reused" and train_check["ok"]:
            train_check["exit_code"] = 0
            train_check["exit_code_ok"] = True
        validation["train"][candidate] = train_check
        train_row = {
            "candidate": candidate,
            "target_keep_ratio": ratio,
            "status": status if train_check["ok"] else "failed",
            "exit_code": code,
            "checkpoint_path": str(checkpoint_dir),
            "flags_path": str(flags_path),
            **trainer_stats(checkpoint_dir),
        }
        train_rows.append(train_row)
        train_rows.sort(key=lambda row: float(row["target_keep_ratio"]))
        _append_csv_row(
            run_index_path,
            run_index_fields,
            {
                "phase": "train",
                "candidate": candidate,
                "target_keep_ratio": ratio,
                "status": train_row["status"],
                "exit_code": code,
                "gpu_id": gpu_id,
                "slot_id": slot_id,
                "checkpoint_path": str(checkpoint_dir),
                "output_dir": str(checkpoint_dir),
                "flags_path": str(flags_path),
            },
        )
        _write_json(suite_root / "validation_summary.json", validation)
        return bool(train_check["ok"])

    while pending_train or running_train:
        occupied_gpus = {str(info["gpu_id"]) for info in running_train.values() if str(info.get("gpu_id", ""))}
        available_gpus = [gpu for gpu in train_gpus if gpu not in occupied_gpus]
        while pending_train and len(running_train) < train_max_parallel:
            ratio = pending_train.pop(0)
            candidate = _candidate_name(ratio)
            checkpoint_dir = model_root / candidate
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            flags = fastvid_flags(ratio)
            flags_path = checkpoint_dir / "flags.json"
            _write_json(flags_path, flags)

            if reuse and (checkpoint_dir / "adapter_config.json").is_file() and (checkpoint_dir / "trainer_state.json").is_file():
                print(f"[reuse] train {candidate}: {checkpoint_dir}", flush=True)
                ok = finalize_train(
                    candidate=candidate,
                    ratio=ratio,
                    checkpoint_dir=checkpoint_dir,
                    flags_path=flags_path,
                    status="reused",
                    code=0,
                    gpu_id="",
                )
                if fail_fast and not ok:
                    write_train_overview(suite_root, train_rows)
                    return 1
                continue

            external_pids = _running_train_pids_for_output(checkpoint_dir)
            if external_pids:
                gpu_id = _external_gpu_for_pids(external_pids)
                running_train[candidate] = {
                    "kind": "external",
                    "candidate": candidate,
                    "ratio": ratio,
                    "checkpoint_dir": checkpoint_dir,
                    "flags_path": flags_path,
                    "gpu_id": gpu_id,
                    "started": time.time(),
                    "pids": external_pids,
                }
                print(
                    f"[attach] train {candidate} pids={external_pids} gpu={gpu_id} checkpoint={checkpoint_dir}",
                    flush=True,
                )
                if gpu_id in available_gpus:
                    available_gpus.remove(gpu_id)
                continue

            if not available_gpus:
                pending_train.insert(0, ratio)
                break

            gpu_id = available_gpus.pop(0)
            proc, log = _launch_logged(
                name=f"train {candidate} keep={ratio} gpu={gpu_id}",
                cmd=build_train_command(base_model, train_root, checkpoint_dir, max_steps),
                env=common_env(gpu_id, flags),
                log_path=checkpoint_dir / "train.log",
            )
            running_train[candidate] = {
                "kind": "launched",
                "candidate": candidate,
                "ratio": ratio,
                "checkpoint_dir": checkpoint_dir,
                "flags_path": flags_path,
                "gpu_id": gpu_id,
                "proc": proc,
                "log": log,
                "started": time.time(),
            }

        if not running_train:
            continue

        done_candidates: List[str] = []
        for candidate, info in list(running_train.items()):
            checkpoint_dir = Path(str(info["checkpoint_dir"]))
            if info["kind"] == "external":
                pids = _running_train_pids_for_output(checkpoint_dir)
                if pids:
                    info["pids"] = pids
                    continue
                done_candidates.append(candidate)
                continue
            proc = info["proc"]
            if proc.poll() is not None:
                done_candidates.append(candidate)

        if not done_candidates:
            active = ", ".join(
                f"{name}@gpu{info.get('gpu_id', '')}/{info['kind']}" for name, info in running_train.items()
            )
            print(f"[heartbeat] train running={active}", flush=True)
            time.sleep(30)
            continue

        for candidate in done_candidates:
            info = running_train.pop(candidate)
            checkpoint_dir = Path(str(info["checkpoint_dir"]))
            flags_path = Path(str(info["flags_path"]))
            ratio = float(info["ratio"])
            gpu_id = str(info.get("gpu_id", ""))
            elapsed = int(time.time() - float(info.get("started", time.time())))
            if info["kind"] == "external":
                code = 0 if (checkpoint_dir / "adapter_config.json").is_file() else 1
                timing_path = checkpoint_dir / "timing.txt"
                if not timing_path.is_file():
                    timing_path.write_text(
                        f"elapsed_seconds_observed={elapsed}\nexit_code={code}\nexternal_attach=true\n",
                        encoding="utf-8",
                    )
                status = "ok" if code == 0 else "failed"
                print(f"[done] external train {candidate} status={status} observed_sec={elapsed}", flush=True)
            else:
                proc = info["proc"]
                log = info["log"]
                log.close()
                code = int(proc.returncode if proc.returncode is not None else proc.wait())
                (checkpoint_dir / "timing.txt").write_text(f"elapsed_seconds={elapsed}\nexit_code={code}\n", encoding="utf-8")
                status = "ok" if code == 0 else "failed"
                print(f"[done] train {candidate} status={status} code={code} elapsed_sec={elapsed}", flush=True)
                if code != 0:
                    print(f"[tail] {checkpoint_dir / 'train.log'}\n{_tail(checkpoint_dir / 'train.log')}", flush=True)

            ok = finalize_train(
                candidate=candidate,
                ratio=ratio,
                checkpoint_dir=checkpoint_dir,
                flags_path=flags_path,
                status=status,
                code=code,
                gpu_id=gpu_id,
            )
            if fail_fast and not ok:
                write_train_overview(suite_root, train_rows)
                return code or 1

    write_train_overview(suite_root, train_rows)

    pending_eval = list(KEEP_RATIOS)
    running_eval: Dict[str, Dict[str, object]] = {}

    def finalize_eval(
        *,
        candidate: str,
        ratio: float,
        checkpoint_dir: Path,
        output_dir: Path,
        flags_path: Path,
        summary_path: Path,
        status: str,
        code: int,
        gpu_id: str,
        slot_id: str = "",
    ) -> bool:
        eval_check = eval_validation(summary_path, code)
        if status == "reused" and eval_check["ok"]:
            eval_check["exit_code"] = 0
            eval_check["exit_code_ok"] = True
        validation["eval"][candidate] = eval_check
        _append_csv_row(
            run_index_path,
            run_index_fields,
            {
                "phase": "eval",
                "candidate": candidate,
                "target_keep_ratio": ratio,
                "status": status if eval_check["ok"] else "failed",
                "exit_code": code,
                "gpu_id": gpu_id,
                "checkpoint_path": str(checkpoint_dir),
                "output_dir": str(output_dir),
                "flags_path": str(flags_path),
                "summary_path": str(summary_path),
            },
        )
        _write_json(suite_root / "validation_summary.json", validation)
        if summary_path.is_file():
            eval_rows.append(
                load_summary_row(
                    kind="FastVid-SFT-LoRA",
                    candidate=candidate,
                    keep_ratio=ratio,
                    checkpoint=str(checkpoint_dir),
                    summary_path=summary_path,
                )
            )
            eval_rows.sort(key=lambda row: float(row["keep_ratio"]))
        return bool(eval_check["ok"])

    while pending_eval or running_eval:
        occupied_slots = {str(info["slot_id"]) for info in running_eval.values()}
        available_slots = [slot for slot in eval_slots if slot["slot_id"] not in occupied_slots]
        while pending_eval and len(running_eval) < eval_max_parallel and available_slots:
            ratio = pending_eval.pop(0)
            candidate = _candidate_name(ratio)
            checkpoint_dir = model_root / candidate
            output_dir = suite_root / "eval" / f"{candidate}_teacher_forcing_240"
            output_dir.mkdir(parents=True, exist_ok=True)
            flags = fastvid_flags(ratio)
            flags_path = output_dir / "flags.json"
            _write_json(flags_path, flags)
            summary_path = output_dir / "summary.json"

            if reuse and summary_path.is_file():
                print(f"[reuse] eval {candidate}: {summary_path}", flush=True)
                ok = finalize_eval(
                    candidate=candidate,
                    ratio=ratio,
                    checkpoint_dir=checkpoint_dir,
                    output_dir=output_dir,
                    flags_path=flags_path,
                    summary_path=summary_path,
                    status="reused",
                    code=0,
                    gpu_id="",
                    slot_id="",
                )
                if fail_fast and not ok:
                    return 1
                continue

            slot = available_slots.pop(0)
            gpu_id = str(slot["gpu_id"])
            slot_id = str(slot["slot_id"])
            proc, log = _launch_logged(
                name=f"eval {candidate} keep={ratio} gpu={gpu_id} slot={slot_id}",
                cmd=build_eval_command(base_model, eval_root, checkpoint_dir, output_dir),
                env=common_env(gpu_id, flags),
                log_path=output_dir / "eval.log",
            )
            running_eval[candidate] = {
                "candidate": candidate,
                "ratio": ratio,
                "checkpoint_dir": checkpoint_dir,
                "output_dir": output_dir,
                "flags_path": flags_path,
                "summary_path": summary_path,
                "gpu_id": gpu_id,
                "slot_id": slot_id,
                "proc": proc,
                "log": log,
                "started": time.time(),
            }

        if not running_eval:
            continue

        done_candidates = [
            candidate for candidate, info in running_eval.items() if info["proc"].poll() is not None
        ]
        if not done_candidates:
            active = ", ".join(
                f"{name}@gpu{info['gpu_id']}/slot{info['slot_id']}" for name, info in running_eval.items()
            )
            print(f"[heartbeat] eval running={active}", flush=True)
            time.sleep(30)
            continue

        for candidate in done_candidates:
            info = running_eval.pop(candidate)
            proc = info["proc"]
            log = info["log"]
            log.close()
            elapsed = int(time.time() - float(info.get("started", time.time())))
            code = int(proc.returncode if proc.returncode is not None else proc.wait())
            output_dir = Path(str(info["output_dir"]))
            (output_dir / "timing.txt").write_text(f"elapsed_seconds={elapsed}\nexit_code={code}\n", encoding="utf-8")
            status = "ok" if code == 0 else "failed"
            print(f"[done] eval {candidate} status={status} code={code} elapsed_sec={elapsed}", flush=True)
            if code != 0:
                print(f"[tail] {output_dir / 'eval.log'}\n{_tail(output_dir / 'eval.log')}", flush=True)
            ok = finalize_eval(
                candidate=candidate,
                ratio=float(info["ratio"]),
                checkpoint_dir=Path(str(info["checkpoint_dir"])),
                output_dir=output_dir,
                flags_path=Path(str(info["flags_path"])),
                summary_path=Path(str(info["summary_path"])),
                status=status,
                code=code,
                gpu_id=str(info["gpu_id"]),
                slot_id=str(info["slot_id"]),
            )
            if fail_fast and not ok:
                return code or 1

    training_free_rows: List[Dict[str, object]] = []
    for ratio in KEEP_RATIOS:
        candidate = _candidate_name(ratio)
        summary_path = training_free_root / candidate / "summary.json"
        if not summary_path.is_file():
            print(f"[warn] missing training-free summary: {summary_path}", flush=True)
            continue
        training_free_rows.append(
            load_summary_row(
                kind="FastVid training-free",
                candidate=candidate,
                keep_ratio=ratio,
                checkpoint=str(base_model),
                summary_path=summary_path,
            )
        )

    comparison_rows = sorted(training_free_rows + eval_rows, key=lambda r: (float(r["keep_ratio"]), str(r["kind"])))
    write_eval_overview(suite_root, comparison_rows)
    write_report_section(
        suite_root=suite_root,
        model_root=model_root,
        train_root=train_root,
        base_model=base_model,
        train_rows=train_rows,
        comparison_rows=comparison_rows,
    )

    all_train_ok = all(bool(item.get("ok")) for item in validation["train"].values())
    all_eval_ok = all(bool(item.get("ok")) for item in validation["eval"].values())
    checkpoints_ok = all((model_root / _candidate_name(ratio) / "adapter_config.json").is_file() for ratio in KEEP_RATIOS)
    validation["final_acceptance"] = {
        "all_three_training_runs_completed": all_train_ok and len(validation["train"]) == 3,
        "all_three_tf240_evals_completed": all_eval_ok and len(validation["eval"]) == 3,
        "three_loadable_checkpoints_exist": checkpoints_ok,
        "report_section_generated": (suite_root / "fastvid_lora_sft_report_section.md").is_file(),
        "ok": all_train_ok and all_eval_ok and checkpoints_ok,
    }
    _write_json(suite_root / "validation_summary.json", validation)

    print(f"[wrote] {suite_root / 'fastvid_lora_sft_train_overview.md'}", flush=True)
    print(f"[wrote] {suite_root / 'fastvid_lora_sft_tf240_overview.md'}", flush=True)
    print(f"[wrote] {suite_root / 'fastvid_lora_sft_report_section.md'}", flush=True)
    print(f"[wrote] {suite_root / 'validation_summary.json'}", flush=True)
    return 0 if validation["final_acceptance"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
