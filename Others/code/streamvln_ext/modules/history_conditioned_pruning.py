from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


def _safe_norm_scores(tokens: torch.Tensor) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=tokens.device if isinstance(tokens, torch.Tensor) else None)
    return torch.norm(tokens.float(), dim=-1)


def _merge_history_bank(
    history_bank: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    history_window: int,
) -> Optional[torch.Tensor]:
    if history_bank is None:
        return None

    if isinstance(history_bank, torch.Tensor):
        if history_bank.ndim == 2:
            return history_bank
        if history_bank.ndim == 3:
            slots = history_bank[-max(1, history_window) :]
            return slots.reshape(-1, slots.shape[-1])
        return None

    if isinstance(history_bank, (list, tuple)):
        tensors = [x for x in history_bank if isinstance(x, torch.Tensor)]
        if not tensors:
            return None
        tensors = tensors[-max(1, history_window) :]

        flattened = []
        for item in tensors:
            if item.ndim == 2:
                flattened.append(item)
            elif item.ndim == 3:
                flattened.append(item.reshape(-1, item.shape[-1]))
        if not flattened:
            return None
        return torch.cat(flattened, dim=0)

    return None


def _history_conditioned_scores(
    current_tokens: torch.Tensor,
    history_tokens: Optional[torch.Tensor],
    recent_boost: float,
) -> torch.Tensor:
    base = _safe_norm_scores(current_tokens)
    if history_tokens is None or not isinstance(history_tokens, torch.Tensor) or history_tokens.numel() == 0:
        return base

    cur_norm = F.normalize(current_tokens.float(), dim=-1)
    his_norm = F.normalize(history_tokens.float(), dim=-1)

    sim = torch.matmul(cur_norm, his_norm.transpose(0, 1))
    max_sim = sim.max(dim=1).values.clamp(-1.0, 1.0)
    novelty = 1.0 - max_sim

    energy = base / (base.mean().clamp_min(1e-6))
    score = novelty + float(recent_boost) * energy
    return score


def prune_tokens_history_conditioned(
    current_tokens: torch.Tensor,
    history_bank: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    keep_ratio: float = 0.8,
    min_tokens: int = 64,
    history_window: int = 8,
    recent_boost: float = 1.2,
) -> torch.Tensor:
    if current_tokens is None or not isinstance(current_tokens, torch.Tensor):
        return current_tokens
    if current_tokens.ndim != 2:
        return current_tokens

    total = current_tokens.shape[0]
    if total <= 0:
        return current_tokens

    keep = max(int(total * float(keep_ratio)), int(min_tokens))
    keep = max(1, min(keep, total))
    if keep >= total:
        return current_tokens

    merged_history = _merge_history_bank(history_bank, history_window=history_window)
    scores = _history_conditioned_scores(current_tokens, merged_history, recent_boost=recent_boost)

    keep_idx = torch.topk(scores, k=keep, largest=True, sorted=False).indices
    keep_idx, _ = torch.sort(keep_idx)
    return current_tokens.index_select(0, keep_idx).contiguous()
