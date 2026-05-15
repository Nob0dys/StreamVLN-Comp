from typing import Optional

import torch
import torch.nn.functional as F

from streamvln.utils.utils import IGNORE_INDEX


def build_memory_pseudo_labels(
    labels: torch.Tensor,
    memory_mask: Optional[torch.Tensor],
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if labels is None or memory_mask is None:
        return labels

    pseudo_labels = labels.clone()
    batch_size, seq_len = labels.shape

    for b in range(batch_size):
        next_token = ignore_index
        next_targets = torch.full((seq_len,), ignore_index, dtype=labels.dtype, device=labels.device)

        for t in range(seq_len - 1, -1, -1):
            if labels[b, t] != ignore_index:
                next_token = labels[b, t]
            next_targets[t] = next_token

        valid_memory = memory_mask[b] & (next_targets != ignore_index)
        pseudo_labels[b, valid_memory] = next_targets[valid_memory]

    return pseudo_labels


def compute_weighted_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    memory_mask: Optional[torch.Tensor],
    memory_loss_weight: float,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    valid = shift_labels.ne(ignore_index)
    if not torch.any(valid):
        return shift_logits.new_zeros((), dtype=shift_logits.dtype)

    safe_labels = shift_labels.clone()
    safe_labels[~valid] = 0

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    weights = valid.to(token_loss.dtype)
    if memory_mask is not None:
        memory_shift = memory_mask[..., 1:].contiguous() & valid
        weights[memory_shift] = memory_loss_weight

    weighted_loss = token_loss * weights * valid.to(token_loss.dtype)
    denom = (weights * valid.to(token_loss.dtype)).sum().clamp_min(1.0)
    return weighted_loss.sum() / denom
