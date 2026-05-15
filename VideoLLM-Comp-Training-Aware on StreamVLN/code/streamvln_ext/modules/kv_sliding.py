from typing import Any, List, Sequence, Tuple

import torch


def _truncate_kv_tensor(tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if tensor is None or tensor.size(-2) <= max_tokens:
        return tensor
    return tensor[..., -max_tokens:, :].contiguous()


def _truncate_kv_any(value: Any, max_tokens: int) -> Any:
    if isinstance(value, torch.Tensor):
        return _truncate_kv_tensor(value, max_tokens)
    if isinstance(value, list):
        return [_truncate_kv_any(v, max_tokens) for v in value]
    if isinstance(value, tuple):
        return tuple(_truncate_kv_any(v, max_tokens) for v in value)
    return value


def truncate_past_key_values(past_key_values: Any, max_tokens: int) -> Any:
    if past_key_values is None or max_tokens <= 0:
        return past_key_values

    if not isinstance(past_key_values, (list, tuple)):
        return past_key_values

    trimmed: List[Tuple[Any, ...]] = []
    for layer in past_key_values:
        if not isinstance(layer, (list, tuple)) or len(layer) < 2:
            trimmed.append(layer)
            continue

        key_states = _truncate_kv_any(layer[0], max_tokens)
        value_states = _truncate_kv_any(layer[1], max_tokens)
        if len(layer) == 2:
            trimmed.append((key_states, value_states))
        else:
            trimmed.append((key_states, value_states, *layer[2:]))

    return tuple(trimmed) if isinstance(past_key_values, tuple) else trimmed
