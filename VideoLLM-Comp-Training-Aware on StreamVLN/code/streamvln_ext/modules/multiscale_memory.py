from typing import List, Optional

import torch
import torch.nn.functional as F


def _pool_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[0] <= target_tokens:
        return tokens

    # [N, D] -> [1, D, N]
    x = tokens.transpose(0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool1d(x, output_size=target_tokens)
    return pooled.squeeze(0).transpose(0, 1).contiguous()


def build_multiscale_memory(memory_bank: Optional[torch.Tensor], levels: int = 3) -> Optional[torch.Tensor]:
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        return memory_bank
    if memory_bank.ndim != 3 or levels <= 1:
        return memory_bank

    out_slots: List[torch.Tensor] = []
    for slot in memory_bank:
        level_tokens = [slot]
        cur_len = slot.shape[0]
        for _ in range(levels - 1):
            cur_len = max(1, cur_len // 2)
            level_tokens.append(_pool_tokens(slot, cur_len))
        out_slots.append(torch.cat(level_tokens, dim=0))

    min_len = min(slot.shape[0] for slot in out_slots)
    aligned = [slot[:min_len] for slot in out_slots]
    return torch.stack(aligned, dim=0)
