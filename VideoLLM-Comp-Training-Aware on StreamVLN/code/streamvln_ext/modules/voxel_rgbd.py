from typing import List, Optional, Tuple

import torch


def _depth_weight(depth_frame: Optional[torch.Tensor]) -> float:
    if depth_frame is None or not isinstance(depth_frame, torch.Tensor) or depth_frame.numel() == 0:
        return 1.0
    d = depth_frame.float()
    return float(torch.clamp(d.std() + 1e-3, min=1e-3, max=10.0).item())


def _select_by_score(tokens: torch.Tensor, score: torch.Tensor, keep_ratio: float, min_tokens: int) -> torch.Tensor:
    total = tokens.shape[0]
    keep = max(min_tokens, int(total * keep_ratio))
    keep = min(keep, total)
    if keep >= total:
        return tokens

    idx = torch.topk(score, k=keep, largest=True, sorted=False).indices
    idx, _ = torch.sort(idx)
    return tokens.index_select(0, idx).contiguous()


def prune_rgbd_features(
    image_features: List[torch.Tensor],
    depths: Optional[torch.Tensor],
    poses: Optional[torch.Tensor],
    intrinsics: Optional[torch.Tensor],
    keep_ratio: float = 0.7,
    min_tokens: int = 64,
) -> List[torch.Tensor]:
    # Current implementation is a practical RGB-D proxy: token-norm ranking re-weighted by depth variance.
    # pose/intrinsics are accepted for interface compatibility and future geometric pruning upgrades.
    _ = poses
    _ = intrinsics

    pruned = []
    for frame_id, frame_tokens in enumerate(image_features):
        if not isinstance(frame_tokens, torch.Tensor) or frame_tokens.ndim != 2:
            pruned.append(frame_tokens)
            continue

        if frame_tokens.shape[0] <= min_tokens:
            pruned.append(frame_tokens)
            continue

        base_score = torch.norm(frame_tokens, dim=-1)
        depth_frame = None
        if isinstance(depths, torch.Tensor) and depths.ndim >= 3 and frame_id < depths.shape[0]:
            depth_frame = depths[frame_id]
        dw = _depth_weight(depth_frame)
        score = base_score * dw
        pruned.append(_select_by_score(frame_tokens, score, keep_ratio=keep_ratio, min_tokens=min_tokens))

    return pruned
