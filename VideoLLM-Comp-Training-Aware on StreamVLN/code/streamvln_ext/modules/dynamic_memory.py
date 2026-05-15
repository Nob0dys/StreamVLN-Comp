from typing import Optional

import torch
import torch.nn.functional as F


def _align_memory(prev_bank: torch.Tensor, cur_bank: torch.Tensor):
    slot_n = min(prev_bank.shape[0], cur_bank.shape[0])
    tok_n = min(prev_bank.shape[1], cur_bank.shape[1])
    return prev_bank[:slot_n, :tok_n], cur_bank[:slot_n, :tok_n]


def compute_memory_delta(prev_bank: Optional[torch.Tensor], cur_bank: Optional[torch.Tensor]) -> float:
    if prev_bank is None or cur_bank is None:
        return 1.0
    if prev_bank.ndim != 3 or cur_bank.ndim != 3:
        return 1.0

    prev_aligned, cur_aligned = _align_memory(prev_bank, cur_bank)
    if prev_aligned.numel() == 0 or cur_aligned.numel() == 0:
        return 1.0

    prev_norm = F.normalize(prev_aligned.float(), dim=-1)
    cur_norm = F.normalize(cur_aligned.float(), dim=-1)
    cos_sim = (prev_norm * cur_norm).sum(dim=-1).mean()
    return float(1.0 - cos_sim.item())


def update_memory_bank(
    prev_bank: Optional[torch.Tensor],
    cur_bank: Optional[torch.Tensor],
    delta_threshold: float = 0.04,
    blend: float = 0.5,
) -> Optional[torch.Tensor]:
    if cur_bank is None:
        return prev_bank
    if prev_bank is None:
        return cur_bank
    if prev_bank.ndim != 3 or cur_bank.ndim != 3:
        return cur_bank

    delta = compute_memory_delta(prev_bank, cur_bank)
    if delta < delta_threshold:
        return prev_bank

    prev_aligned, cur_aligned = _align_memory(prev_bank, cur_bank)
    alpha = max(0.0, min(1.0, float(blend)))
    mixed = (1.0 - alpha) * prev_aligned + alpha * cur_aligned

    out = cur_bank.clone()
    out[: mixed.shape[0], : mixed.shape[1]] = mixed.to(dtype=out.dtype)
    return out
