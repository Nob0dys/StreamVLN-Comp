#!/usr/bin/env python
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def _load_annotations(root: Path) -> Dict[str, List[int]]:
    candidates = [root / "annotations.json", root / "annotations_v1-3.json"]
    annot_path = next((path for path in candidates if path.is_file()), None)
    if annot_path is None:
        return {}

    with annot_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    by_video: Dict[str, List[int]] = {}
    for item in annotations:
        video = str(item.get("video", "")).strip()
        if not video:
            continue
        actions = item.get("actions", [])
        by_video[video] = [int(action) for action in actions if int(action) in {-1, 0, 1, 2, 3}]
    return by_video


def _iter_episodes(root: Path) -> Iterable[Tuple[str, Path, List[str], List[int]]]:
    annotations = _load_annotations(root)
    image_root = root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_root}")

    if annotations:
        items = sorted(annotations.items(), key=lambda kv: kv[0])
        for video, actions in items:
            episode_dir = root / video
            rgb_dir = episode_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            frame_files = sorted(name for name in os.listdir(rgb_dir) if name.lower().endswith((".jpg", ".jpeg", ".png")))
            if frame_files:
                yield video, episode_dir, frame_files, actions
        return

    for episode_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        rgb_dir = episode_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        frame_files = sorted(name for name in os.listdir(rgb_dir) if name.lower().endswith((".jpg", ".jpeg", ".png")))
        if frame_files:
            video = str(episode_dir.relative_to(root))
            yield video, episode_dir, frame_files, []


def _intrinsic_4x4(width: int, height: int, vfov_deg: float) -> np.ndarray:
    vfov_rad = math.radians(float(vfov_deg))
    fy = float(height) / (2.0 * math.tan(vfov_rad / 2.0))
    fx = fy
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def _pose_from_odometry(frame_idx: int, actions: List[int], pose_mode: str) -> np.ndarray:
    if pose_mode == "identity":
        return np.eye(4, dtype=np.float32)

    valid_actions = [int(action) for action in actions if int(action) in {0, 1, 2, 3}]
    x = 0.0
    z = 0.0
    yaw = 0.0
    step = 0.25
    turn = math.radians(15.0)

    for action in valid_actions[: max(int(frame_idx), 0)]:
        if action == 1:
            x += math.sin(yaw) * step
            z += math.cos(yaw) * step
        elif action == 2:
            yaw += turn
        elif action == 3:
            yaw -= turn

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 0] = cos_y
    pose[0, 2] = sin_y
    pose[2, 0] = -sin_y
    pose[2, 2] = cos_y
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def _ensure_geometry_files(
    episode_dir: Path,
    frame_files: List[str],
    actions: List[int],
    width: int,
    height: int,
    vfov_deg: float,
    pose_mode: str,
    overwrite: bool,
) -> None:
    pose_dir = episode_dir / "pose"
    intrinsic_dir = episode_dir / "intrinsic"
    pose_dir.mkdir(exist_ok=True)
    intrinsic_dir.mkdir(exist_ok=True)

    intrinsic = _intrinsic_4x4(width=width, height=height, vfov_deg=vfov_deg)
    for frame_idx, frame_name in enumerate(frame_files):
        stem = Path(frame_name).stem
        pose_path = pose_dir / f"{stem}.npy"
        intrinsic_path = intrinsic_dir / f"{stem}.npy"

        if overwrite or not pose_path.is_file():
            np.save(pose_path, _pose_from_odometry(frame_idx, actions, pose_mode=pose_mode))
        if overwrite or not intrinsic_path.is_file():
            np.save(intrinsic_path, intrinsic)


def _save_depth_png(depth_m: np.ndarray, out_path: Path, max_depth_m: float) -> None:
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=max_depth_m, neginf=0.0)
    depth_m = np.clip(depth_m, 0.0, float(max_depth_m))
    depth_mm = np.round(depth_m * 1000.0).astype(np.uint16)
    Image.fromarray(depth_mm, mode="I;16").save(out_path)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _blank_rgb(size: Tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, (0, 0, 0))


def _load_rgb_batch_or_blank(image_paths: List[Path]) -> Tuple[List[Image.Image], List[str]]:
    images: List[Optional[Image.Image]] = []
    bad_image_paths: List[str] = []
    fallback_size: Optional[Tuple[int, int]] = None

    for path in image_paths:
        try:
            image = _load_rgb(path)
            fallback_size = image.size
            images.append(image)
        except (UnidentifiedImageError, OSError):
            images.append(None)
            bad_image_paths.append(str(path))

    if fallback_size is None:
        fallback_size = (256, 256)

    return [image if image is not None else _blank_rgb(fallback_size) for image in images], bad_image_paths


def _load_rgb_or_blank(path: Path, target_size: Tuple[int, int]) -> Tuple[Image.Image, bool]:
    try:
        return _load_rgb(path), False
    except (UnidentifiedImageError, OSError):
        height, width = target_size
        return Image.new("RGB", (width, height), (0, 0, 0)), True


def _infer_depth_batch(
    model,
    processor,
    image_paths: List[Path],
    device: torch.device,
    dtype: torch.dtype,
    max_depth_m: float,
    target_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[str]]:
    images, bad_image_paths = _load_rgb_batch_or_blank(image_paths)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device=device) for key, value in inputs.items()}

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
            outputs = model(**inputs)
        predicted = outputs.predicted_depth.float().unsqueeze(1)

    depth_maps: List[np.ndarray] = []
    for idx in range(len(images)):
        depth = F.interpolate(
            predicted[idx : idx + 1],
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        depth = depth.clamp(min=0.0, max=float(max_depth_m)).detach().cpu().numpy().astype(np.float32)
        depth_maps.append(depth)
    return depth_maps, bad_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate R2R RGB depths with Depth Anything V2 and save per-frame geometry.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/home/ubuntu/dataset/VLN-Trajectory-Data/R2R",
            "/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen",
        ],
        help="Dataset roots containing images/ and annotations.",
    )
    parser.add_argument("--model-path", default="/home/ubuntu/model/Depth-Anything-V2-Metric-Indoor-Base-hf")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--vfov-deg", type=float, default=60.0)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--pose-mode", choices=["odometry", "identity"], default="odometry")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_path)
    model = model.to(device=device)
    if device.type == "cuda" and dtype != torch.float32:
        model = model.to(dtype=dtype)
    model.eval()

    summary = {
        "model_path": args.model_path,
        "roots": args.roots,
        "width": args.width,
        "height": args.height,
        "vfov_deg": args.vfov_deg,
        "intrinsic": _intrinsic_4x4(args.width, args.height, args.vfov_deg).tolist(),
        "pose_mode": args.pose_mode,
        "max_depth_m": args.max_depth_m,
        "episodes_seen": 0,
        "frames_seen": 0,
        "depth_frames_written": 0,
        "depth_frames_skipped_existing": 0,
        "bad_image_frames": 0,
        "bad_image_paths_sample": [],
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
    }

    pending_paths: List[Path] = []
    pending_outputs: List[Path] = []

    def flush_batch() -> None:
        if not pending_paths:
            return
        depth_maps, bad_image_paths = _infer_depth_batch(
            model=model,
            processor=processor,
            image_paths=list(pending_paths),
            device=device,
            dtype=dtype,
            max_depth_m=args.max_depth_m,
            target_size=(int(args.height), int(args.width)),
        )
        if bad_image_paths:
            summary["bad_image_frames"] += len(bad_image_paths)
            sample = summary["bad_image_paths_sample"]
            sample.extend(bad_image_paths[: max(0, 100 - len(sample))])
        for depth_map, out_path in zip(depth_maps, pending_outputs):
            _save_depth_png(depth_map, out_path, max_depth_m=args.max_depth_m)
            summary["depth_frames_written"] += 1
        pending_paths.clear()
        pending_outputs.clear()

    for root_str in args.roots:
        root = Path(root_str)
        episodes = list(_iter_episodes(root))
        num_shards = max(int(args.num_shards), 1)
        shard_id = int(args.shard_id)
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"--shard-id must be in [0, {num_shards}), got {shard_id}")
        if num_shards > 1:
            episodes = [episode for idx, episode in enumerate(episodes) if idx % num_shards == shard_id]
        if args.max_episodes > 0:
            episodes = episodes[: args.max_episodes]

        progress = tqdm(episodes, desc=f"depth_geometry:{root.name}", unit="episode")
        for _, episode_dir, frame_files, actions in progress:
            summary["episodes_seen"] += 1
            if args.max_frames > 0:
                remaining = args.max_frames - summary["frames_seen"]
                if remaining <= 0:
                    break
                frame_files = frame_files[:remaining]

            _ensure_geometry_files(
                episode_dir=episode_dir,
                frame_files=frame_files,
                actions=actions,
                width=args.width,
                height=args.height,
                vfov_deg=args.vfov_deg,
                pose_mode=args.pose_mode,
                overwrite=args.overwrite,
            )

            depth_dir = episode_dir / "depth"
            depth_dir.mkdir(exist_ok=True)
            rgb_dir = episode_dir / "rgb"
            meta = {
                "depth_model": args.model_path,
                "depth_format": "uint16_png_millimeters",
                "intrinsic_file_format": "npy_4x4",
                "pose_file_format": "npy_4x4_camera_to_world",
                "pose_mode": args.pose_mode,
                "width": args.width,
                "height": args.height,
                "vfov_deg": args.vfov_deg,
                "max_depth_m": args.max_depth_m,
                "intrinsic": summary["intrinsic"],
            }
            with (episode_dir / "geometry_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            for frame_name in frame_files:
                stem = Path(frame_name).stem
                rgb_path = rgb_dir / frame_name
                depth_path = depth_dir / f"{stem}.png"
                summary["frames_seen"] += 1
                if depth_path.is_file() and not args.overwrite:
                    summary["depth_frames_skipped_existing"] += 1
                    continue
                pending_paths.append(rgb_path)
                pending_outputs.append(depth_path)
                if len(pending_paths) >= max(int(args.batch_size), 1):
                    flush_batch()

            progress.set_postfix(
                written=summary["depth_frames_written"],
                skipped=summary["depth_frames_skipped_existing"],
            )

        flush_batch()

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = Path(args.roots[-1]).parent / "depth_anything_geometry_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
