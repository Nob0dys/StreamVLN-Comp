import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from streamvln_ext.config.feature_flags import ExtFeatureFlags


def _infer_square_hw(num_tokens: int) -> Optional[Tuple[int, int]]:
    side = int(round(math.sqrt(float(num_tokens))))
    if side > 0 and side * side == int(num_tokens):
        return side, side
    return None


def _valid_geometry(
    depths: Optional[torch.Tensor],
    poses: Optional[torch.Tensor],
    intrinsics: Optional[torch.Tensor],
) -> bool:
    if not isinstance(depths, torch.Tensor) or not isinstance(poses, torch.Tensor) or not isinstance(intrinsics, torch.Tensor):
        return False
    if depths.ndim != 3 or poses.ndim != 3 or intrinsics.ndim != 3:
        return False
    if depths.shape[0] == 0 or poses.shape[0] == 0 or intrinsics.shape[0] == 0:
        return False
    return True


def _project_patch_centers_to_voxels(
    depths: torch.Tensor,
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    feature_hw: Tuple[int, int],
    voxel_size: float,
    min_depth: float,
    max_depth: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Back-project feature-grid centers into world-space voxel indices.

    Args:
        depths: [T, H, W] depth in meters.
        poses: [T, 4, 4] camera-to-world transforms.
        intrinsics: [T, 4, 4] or [T, 3, 3] camera intrinsics adjusted to depth size.
        feature_hw: token grid height/width after StreamVLN spatial pooling.

    Returns:
        voxel_coords: [T, Hf, Wf, 3] integer voxel coordinates.
        valid_mask: [T, Hf, Wf] mask for finite depth points.
    """
    grid_h, grid_w = feature_hw
    device = depths.device
    depth_f = depths.float()
    poses_f = poses.float()
    intr_f = intrinsics.float()

    if depth_f.ndim != 3:
        raise ValueError(f"depths must be [T,H,W], got {tuple(depth_f.shape)}")

    num_frames, depth_h, depth_w = depth_f.shape
    depth_grid = F.interpolate(
        depth_f.unsqueeze(1),
        size=(grid_h, grid_w),
        mode="nearest",
    ).squeeze(1)

    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = (xs + 0.5) * (float(depth_w) / float(grid_w)) - 0.5
    v = (ys + 0.5) * (float(depth_h) / float(grid_h)) - 0.5

    fx = intr_f[:, 0, 0].view(num_frames, 1, 1).clamp(min=1e-6)
    fy = intr_f[:, 1, 1].view(num_frames, 1, 1).clamp(min=1e-6)
    cx = intr_f[:, 0, 2].view(num_frames, 1, 1)
    cy = intr_f[:, 1, 2].view(num_frames, 1, 1)

    z = depth_grid
    x = (u.unsqueeze(0) - cx) / fx * z
    y = (v.unsqueeze(0) - cy) / fy * z

    ones = torch.ones_like(z)
    camera_points = torch.stack((x, y, z, ones), dim=-1)
    world_points = torch.matmul(poses_f[:, None, None, :, :], camera_points.unsqueeze(-1)).squeeze(-1)[..., :3]

    valid = torch.isfinite(world_points).all(dim=-1)
    valid = valid & torch.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))

    safe_voxel_size = max(float(voxel_size), 1e-6)
    voxel_coords = torch.floor(world_points / safe_voxel_size).to(dtype=torch.long)
    return voxel_coords, valid


def build_voxel_spatial_pruning_mask(
    voxel_coords: torch.Tensor,
    valid_mask: torch.Tensor,
    stride_k: int,
    frame_threshold: float,
) -> torch.Tensor:
    """Implement StreamVLN paper Algorithm 1 for a precomputed voxel map."""
    if voxel_coords.ndim != 4 or voxel_coords.shape[-1] != 3:
        raise ValueError(f"voxel_coords must be [T,H,W,3], got {tuple(voxel_coords.shape)}")
    if valid_mask.shape != voxel_coords.shape[:3]:
        raise ValueError("valid_mask shape must match voxel_coords[:3]")

    num_frames, grid_h, grid_w = valid_mask.shape
    stride_k = max(int(stride_k), 1)

    keep_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    latest: Dict[Tuple[int, int, int, int], Tuple[int, int, int]] = {}

    coords_cpu = voxel_coords.detach().cpu()
    valid_cpu = valid_mask.detach().cpu()
    for t in range(num_frames):
        bucket = int(t // stride_k)
        for y in range(grid_h):
            for x in range(grid_w):
                if not bool(valid_cpu[t, y, x]):
                    continue
                vx, vy, vz = coords_cpu[t, y, x].tolist()
                latest[(bucket, int(vx), int(vy), int(vz))] = (t, y, x)

    for t, y, x in latest.values():
        keep_mask[t, y, x] = True

    if frame_threshold > 0:
        min_keep = float(frame_threshold) * float(grid_h * grid_w)
        per_frame_keep = keep_mask.flatten(1).sum(dim=1).float()
        drop_frames = per_frame_keep < min_keep
        keep_mask[drop_frames] = False

    return keep_mask


def prune_frame_sequence_by_voxels(
    frame_sequence: torch.Tensor,
    depths: Optional[torch.Tensor],
    poses: Optional[torch.Tensor],
    intrinsics: Optional[torch.Tensor],
    ext_flags: ExtFeatureFlags,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Prune a [T, H*W, C] history token sequence with voxel spatial pruning."""
    if not isinstance(frame_sequence, torch.Tensor) or frame_sequence.ndim != 3:
        return frame_sequence, {"skipped": True, "reason": "invalid_frame_sequence"}

    num_frames, tokens_per_frame, hidden_size = frame_sequence.shape
    feature_hw = _infer_square_hw(tokens_per_frame)
    if feature_hw is None:
        return frame_sequence.flatten(0, 1).contiguous(), {
            "skipped": True,
            "reason": "non_square_token_grid",
            "num_frames": int(num_frames),
            "tokens_per_frame": int(tokens_per_frame),
        }

    if not _valid_geometry(depths, poses, intrinsics):
        return frame_sequence.flatten(0, 1).contiguous(), {
            "skipped": True,
            "reason": "missing_geometry",
            "num_frames": int(num_frames),
            "tokens_per_frame": int(tokens_per_frame),
        }

    usable_frames = min(int(num_frames), int(depths.shape[0]), int(poses.shape[0]), int(intrinsics.shape[0]))
    if usable_frames <= 0:
        return frame_sequence.flatten(0, 1).contiguous(), {
            "skipped": True,
            "reason": "empty_geometry",
            "num_frames": int(num_frames),
            "tokens_per_frame": int(tokens_per_frame),
        }

    seq = frame_sequence[:usable_frames]
    depths = depths[:usable_frames].to(device=seq.device)
    poses = poses[:usable_frames].to(device=seq.device)
    intrinsics = intrinsics[:usable_frames].to(device=seq.device)

    voxel_coords, valid_mask = _project_patch_centers_to_voxels(
        depths=depths,
        poses=poses,
        intrinsics=intrinsics,
        feature_hw=feature_hw,
        voxel_size=float(ext_flags.voxel_spatial_size),
        min_depth=float(ext_flags.voxel_spatial_min_depth),
        max_depth=float(ext_flags.voxel_spatial_max_depth),
    )
    if int(valid_mask.sum().item()) <= 0:
        return frame_sequence.flatten(0, 1).contiguous(), {
            "skipped": True,
            "reason": "no_valid_depth",
            "num_frames": int(num_frames),
            "tokens_per_frame": int(tokens_per_frame),
        }

    keep_mask = build_voxel_spatial_pruning_mask(
        voxel_coords=voxel_coords,
        valid_mask=valid_mask,
        stride_k=int(ext_flags.voxel_spatial_stride_k),
        frame_threshold=float(ext_flags.voxel_spatial_frame_threshold),
    )

    seq_grid = seq.view(usable_frames, feature_hw[0], feature_hw[1], hidden_size)
    pruned = seq_grid[keep_mask].view(-1, hidden_size).contiguous()
    if usable_frames < num_frames:
        tail = frame_sequence[usable_frames:].flatten(0, 1)
        pruned = torch.cat([pruned, tail], dim=0)

    before = int(num_frames * tokens_per_frame)
    after = int(pruned.shape[0])
    return pruned, {
        "skipped": False,
        "num_frames": int(num_frames),
        "usable_frames": int(usable_frames),
        "tokens_per_frame": int(tokens_per_frame),
        "grid_h": int(feature_hw[0]),
        "grid_w": int(feature_hw[1]),
        "valid_tokens": int(valid_mask.sum().item()),
        "kept_tokens": after,
        "dropped_tokens": max(0, before - after),
        "keep_ratio": float(after / before) if before > 0 else 0.0,
    }


def apply_voxel_spatial_pruning_to_memory(
    image_features: List[torch.Tensor],
    memory_features: List[Optional[torch.Tensor]],
    depths: Optional[torch.Tensor],
    poses: Optional[torch.Tensor],
    intrinsics: Optional[torch.Tensor],
    ext_flags: ExtFeatureFlags,
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, object]]:
    """Apply voxel pruning to StreamVLN memory banks.

    StreamVLN stores sampled history frames as a single memory slot shaped
    [slot, T * tokens_per_frame, C].  This function reconstructs T from the
    current visual token grid, applies Algorithm 1 to each slot, and returns a
    new memory bank with variable-length pruned memory tokens.
    """
    pruned_memory_features: List[Optional[torch.Tensor]] = []
    batch_meta: List[Dict[str, object]] = []

    for batch_idx, memory_bank in enumerate(memory_features):
        if memory_bank is None or not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 3:
            pruned_memory_features.append(memory_bank)
            batch_meta.append({"skipped": True, "reason": "missing_memory_bank"})
            continue

        frame_features = image_features[batch_idx] if batch_idx < len(image_features) else None
        if not isinstance(frame_features, torch.Tensor) or frame_features.ndim != 3:
            pruned_memory_features.append(memory_bank)
            batch_meta.append({"skipped": True, "reason": "missing_image_feature_grid"})
            continue

        tokens_per_frame = int(frame_features.shape[1])
        if tokens_per_frame <= 0:
            pruned_memory_features.append(memory_bank)
            batch_meta.append({"skipped": True, "reason": "invalid_tokens_per_frame"})
            continue

        total_tokens = int(memory_bank.shape[1])
        if total_tokens % tokens_per_frame != 0:
            pruned_memory_features.append(memory_bank)
            batch_meta.append({
                "skipped": True,
                "reason": "memory_tokens_not_divisible_by_frame_tokens",
                "total_tokens": total_tokens,
                "tokens_per_frame": tokens_per_frame,
            })
            continue

        history_frames = total_tokens // tokens_per_frame
        depth_batch = depths[batch_idx, :history_frames] if isinstance(depths, torch.Tensor) and batch_idx < depths.shape[0] else None
        pose_batch = poses[batch_idx, :history_frames] if isinstance(poses, torch.Tensor) and batch_idx < poses.shape[0] else None
        intr_batch = intrinsics[batch_idx, :history_frames] if isinstance(intrinsics, torch.Tensor) and batch_idx < intrinsics.shape[0] else None

        slot_outputs: List[torch.Tensor] = []
        slot_meta: List[Dict[str, object]] = []
        for slot in memory_bank:
            slot_frames = slot.view(history_frames, tokens_per_frame, slot.shape[-1])
            pruned_slot, meta = prune_frame_sequence_by_voxels(
                slot_frames,
                depths=depth_batch,
                poses=pose_batch,
                intrinsics=intr_batch,
                ext_flags=ext_flags,
            )
            slot_outputs.append(pruned_slot)
            slot_meta.append(meta)

        if len(slot_outputs) == 0:
            pruned_memory_features.append(memory_bank)
            batch_meta.append({"skipped": True, "reason": "empty_slots"})
            continue

        min_len = min(int(slot.shape[0]) for slot in slot_outputs)
        aligned = [slot[:min_len] for slot in slot_outputs]
        pruned_memory_features.append(torch.stack(aligned, dim=0).contiguous())
        batch_meta.append({
            "skipped": False,
            "history_frames": int(history_frames),
            "tokens_per_frame": int(tokens_per_frame),
            "memory_tokens_before": int(memory_bank.shape[0] * memory_bank.shape[1]),
            "memory_tokens_after": int(len(aligned) * min_len),
            "slot_meta": slot_meta,
        })

    stats: Dict[str, object] = {
        "method": "voxel_spatial_pruning",
        "voxel_size": float(ext_flags.voxel_spatial_size),
        "stride_k": int(ext_flags.voxel_spatial_stride_k),
        "frame_threshold": float(ext_flags.voxel_spatial_frame_threshold),
        "batch_meta": batch_meta,
    }
    return pruned_memory_features, stats
