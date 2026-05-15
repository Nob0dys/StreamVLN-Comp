import math
from typing import List

import torch


def _compute_keep_tokens(num_tokens: int, keep_ratio: float, min_tokens: int) -> int:
    keep_ratio = max(0.0, min(1.0, keep_ratio))
    keep = int(math.ceil(num_tokens * keep_ratio))
    keep = max(min_tokens, keep)
    keep = min(num_tokens, keep)
    return keep


def _select_by_norm(feature: torch.Tensor, keep_tokens: int) -> torch.Tensor:
    if feature.ndim != 2:
        raise ValueError(f"Expected [tokens, dim], got {feature.shape}")

    num_tokens = feature.size(0)
    if keep_tokens >= num_tokens:
        return feature

    scores = feature.norm(dim=-1)
    topk = torch.topk(scores, k=keep_tokens, largest=True, sorted=False).indices
    topk = torch.sort(topk).values
    return feature.index_select(0, topk)


def prune_tokens_per_frame(
    frame_features: torch.Tensor,
    keep_ratio: float,
    min_tokens: int,
) -> torch.Tensor:
    if frame_features.ndim != 3:
        raise ValueError(f"Expected [frames, tokens, dim], got {frame_features.shape}")

    pruned_frames: List[torch.Tensor] = []
    for feat in frame_features:
        keep_tokens = _compute_keep_tokens(feat.size(0), keep_ratio, min_tokens)
        pruned_frames.append(_select_by_norm(feat, keep_tokens))

    return torch.stack(pruned_frames, dim=0)


def prune_token_bank(
    token_bank: torch.Tensor,
    keep_ratio: float,
    min_tokens: int,
) -> torch.Tensor:
    if token_bank is None:
        return None
    if token_bank.ndim != 2:
        raise ValueError(f"Expected [tokens, dim], got {token_bank.shape}")

    keep_tokens = _compute_keep_tokens(token_bank.size(0), keep_ratio, min_tokens)
    return _select_by_norm(token_bank, keep_tokens)
