from typing import Optional

import torch


def _compute_token_scores(tokens: torch.Tensor) -> torch.Tensor:
    # tokens: [N, D]
    if tokens.numel() == 0:
        return torch.zeros((0,), device=tokens.device, dtype=tokens.dtype)
    return torch.norm(tokens, dim=-1)


def select_tokens(tokens: torch.Tensor, keep_ratio: float = 0.7, min_tokens: int = 64) -> torch.Tensor:
    if tokens is None or not isinstance(tokens, torch.Tensor):
        return tokens
    if tokens.ndim != 2:
        return tokens

    total = tokens.shape[0]
    if total <= min_tokens:
        return tokens

    keep = max(min_tokens, int(total * keep_ratio))
    keep = min(keep, total)
    if keep == total:
        return tokens

    scores = _compute_token_scores(tokens)
    keep_idx = torch.topk(scores, k=keep, largest=True, sorted=False).indices
    keep_idx, _ = torch.sort(keep_idx)
    return tokens.index_select(0, keep_idx).contiguous()


def select_memory_bank(memory_bank: Optional[torch.Tensor], keep_ratio: float = 0.7, min_tokens: int = 64) -> Optional[torch.Tensor]:
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        return memory_bank
    if memory_bank.ndim != 3:
        return memory_bank

    slots = []
    for slot in memory_bank:
        slots.append(select_tokens(slot, keep_ratio=keep_ratio, min_tokens=min_tokens))

    # Ensure all slots have the same length by clipping to the shortest one.
    min_len = min(slot.shape[0] for slot in slots)
    aligned = [slot[:min_len] for slot in slots]
    return torch.stack(aligned, dim=0)
