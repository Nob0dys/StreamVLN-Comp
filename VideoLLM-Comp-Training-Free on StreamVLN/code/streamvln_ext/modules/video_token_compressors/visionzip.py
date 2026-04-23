from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from streamvln_ext.config.feature_flags import ExtFeatureFlags


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return scores.float()
    s_min = scores.min()
    s_max = scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-6)


def _score_l2(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=tokens.device)
    return torch.norm(tokens.float(), dim=-1)


def _score_attn_proxy(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=tokens.device)
    normalized = F.normalize(tokens.float(), dim=-1)
    query = normalized.mean(dim=0, keepdim=True)
    attn = torch.matmul(normalized, query.transpose(0, 1)).squeeze(-1)
    return _normalize_scores(attn)


def _token_scores(tokens: torch.Tensor, score_type: str) -> torch.Tensor:
    score_key = str(score_type).strip().lower()
    if score_key in {"l2", "norm", "magnitude"}:
        return _score_l2(tokens)
    return _score_attn_proxy(tokens)


def _evenly_spaced_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    if length <= 0 or count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    count = min(length, count)
    if count == 1:
        return torch.zeros((1,), dtype=torch.long, device=device)
    positions = torch.linspace(0, length - 1, steps=count, device=device)
    indices = torch.round(positions).long()
    indices = torch.unique(indices, sorted=True)
    if indices.numel() < count:
        full = torch.arange(length, device=device, dtype=torch.long)
        remaining = full[~torch.isin(full, indices)]
        fill = remaining[: max(0, count - indices.numel())]
        indices = torch.cat([indices, fill], dim=0)
        indices, _ = torch.sort(indices)
    return indices[:count]


def _aggregate_contextual_tokens(
    remaining_tokens: torch.Tensor,
    contextual_local_idx: torch.Tensor,
) -> torch.Tensor:
    hidden_dim = remaining_tokens.shape[-1]
    if contextual_local_idx.numel() == 0:
        return torch.empty((0, hidden_dim), dtype=remaining_tokens.dtype, device=remaining_tokens.device)

    target_tokens = remaining_tokens.index_select(0, contextual_local_idx)
    if contextual_local_idx.numel() == remaining_tokens.shape[0]:
        return target_tokens

    merge_mask = torch.ones((remaining_tokens.shape[0],), dtype=torch.bool, device=remaining_tokens.device)
    merge_mask[contextual_local_idx] = False
    tokens_to_merge = remaining_tokens[merge_mask]
    if tokens_to_merge.numel() == 0:
        return target_tokens

    target_norm = F.normalize(target_tokens.float(), dim=-1)
    merge_norm = F.normalize(tokens_to_merge.float(), dim=-1)
    similarity = torch.matmul(merge_norm, target_norm.transpose(0, 1))
    assignments = similarity.argmax(dim=1)

    counts = torch.bincount(assignments, minlength=target_tokens.shape[0]).to(device=remaining_tokens.device, dtype=torch.float32)
    counts = counts.clamp(min=1.0).unsqueeze(-1)
    aggregated = torch.zeros_like(target_tokens.float())
    aggregated.index_add_(0, assignments, tokens_to_merge.float())
    contextual_tokens = target_tokens.float() + aggregated / counts
    return contextual_tokens.to(dtype=remaining_tokens.dtype)


def compress_visionzip_tokens(
    tokens: torch.Tensor,
    dominant_num: int,
    contextual_num: int,
    score_type: str = "attn_proxy",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokens is None or not isinstance(tokens, torch.Tensor):
        return tokens, torch.empty((0,), dtype=torch.long)
    if tokens.ndim != 2:
        return tokens, torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.long)

    num_tokens = int(tokens.shape[0])
    if num_tokens <= 0:
        return tokens, torch.empty((0,), device=tokens.device, dtype=torch.long)

    dominant_num = max(0, min(int(dominant_num), num_tokens))
    contextual_num = max(0, min(int(contextual_num), max(0, num_tokens - dominant_num)))
    if dominant_num + contextual_num >= num_tokens:
        keep_idx = torch.arange(num_tokens, device=tokens.device, dtype=torch.long)
        return tokens, keep_idx

    scores = _token_scores(tokens, score_type=score_type)
    if dominant_num > 0:
        dominant_local_idx = torch.topk(scores, k=dominant_num, largest=True, sorted=False).indices
        dominant_local_idx, _ = torch.sort(dominant_local_idx)
    else:
        dominant_local_idx = torch.empty((0,), dtype=torch.long, device=tokens.device)

    keep_mask = torch.ones((num_tokens,), dtype=torch.bool, device=tokens.device)
    keep_mask[dominant_local_idx] = False
    dominant_tokens = tokens.index_select(0, dominant_local_idx) if dominant_local_idx.numel() > 0 else tokens.new_empty((0, tokens.shape[-1]))

    remaining_idx = torch.where(keep_mask)[0]
    remaining_tokens = tokens.index_select(0, remaining_idx) if remaining_idx.numel() > 0 else tokens.new_empty((0, tokens.shape[-1]))

    if contextual_num > 0 and remaining_tokens.shape[0] > 0:
        contextual_local_idx = _evenly_spaced_indices(remaining_tokens.shape[0], contextual_num, tokens.device)
        contextual_keep_idx = remaining_idx.index_select(0, contextual_local_idx)
        contextual_tokens = _aggregate_contextual_tokens(remaining_tokens, contextual_local_idx)
    else:
        contextual_keep_idx = torch.empty((0,), dtype=torch.long, device=tokens.device)
        contextual_tokens = tokens.new_empty((0, tokens.shape[-1]))

    combined_keep_idx = torch.cat([dominant_local_idx, contextual_keep_idx], dim=0)
    combined_tokens = torch.cat([dominant_tokens, contextual_tokens], dim=0)
    if combined_keep_idx.numel() == 0:
        keep_idx = torch.arange(num_tokens, device=tokens.device, dtype=torch.long)
        return tokens, keep_idx

    order = torch.argsort(combined_keep_idx)
    keep_idx = combined_keep_idx.index_select(0, order)
    compressed_tokens = combined_tokens.index_select(0, order).contiguous()
    return compressed_tokens, keep_idx


def _compress_frame_sequence(
    frame_sequence: torch.Tensor,
    dominant_num: int,
    contextual_num: int,
    score_type: str,
) -> Tuple[torch.Tensor, List[int]]:
    compressed_frames: List[torch.Tensor] = []
    frame_sizes: List[int] = []
    for frame in frame_sequence:
        compressed_frame, _ = compress_visionzip_tokens(
            frame,
            dominant_num=dominant_num,
            contextual_num=contextual_num,
            score_type=score_type,
        )
        compressed_frames.append(compressed_frame)
        frame_sizes.append(int(compressed_frame.shape[0]))

    min_len = min(frame_sizes) if frame_sizes else 0
    if min_len <= 0:
        hidden = int(frame_sequence.shape[-1])
        return frame_sequence.new_empty((0, 0, hidden)), frame_sizes

    aligned = [frame[:min_len] for frame in compressed_frames]
    return torch.stack(aligned, dim=0), frame_sizes


def _compress_memory_bank(
    memory_bank: torch.Tensor,
    tokens_per_frame: Optional[int],
    dominant_num: int,
    contextual_num: int,
    score_type: str,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        return memory_bank, {"memory_frames_inferred": 0, "memory_tokens_per_frame": 0}
    if memory_bank.ndim != 3:
        return memory_bank, {"memory_frames_inferred": 0, "memory_tokens_per_frame": 0}

    compressed_slots: List[torch.Tensor] = []
    slot_sizes: List[int] = []
    inferred_frames = 0
    inferred_tokens_per_frame = 0

    for slot in memory_bank:
        total_tokens = int(slot.shape[0])
        if tokens_per_frame and tokens_per_frame > 0 and total_tokens % tokens_per_frame == 0:
            num_frames = max(1, total_tokens // tokens_per_frame)
            inferred_frames = max(inferred_frames, num_frames)
            inferred_tokens_per_frame = tokens_per_frame
            slot_frames = slot.view(num_frames, tokens_per_frame, slot.shape[-1])
            compressed_slot_frames, _ = _compress_frame_sequence(
                slot_frames,
                dominant_num=dominant_num,
                contextual_num=contextual_num,
                score_type=score_type,
            )
            compressed_slot = compressed_slot_frames.flatten(0, 1)
        else:
            compressed_slot, _ = compress_visionzip_tokens(
                slot,
                dominant_num=dominant_num,
                contextual_num=contextual_num,
                score_type=score_type,
            )
        compressed_slots.append(compressed_slot)
        slot_sizes.append(int(compressed_slot.shape[0]))

    min_len = min(slot_sizes) if slot_sizes else 0
    if min_len <= 0:
        hidden = int(memory_bank.shape[-1])
        return memory_bank.new_empty((0, 0, hidden)), {
            "memory_frames_inferred": inferred_frames,
            "memory_tokens_per_frame": inferred_tokens_per_frame,
        }

    aligned = [slot[:min_len] for slot in compressed_slots]
    return torch.stack(aligned, dim=0), {
        "memory_frames_inferred": inferred_frames,
        "memory_tokens_per_frame": inferred_tokens_per_frame,
    }


def apply_visionzip_compression(
    image_features: List[torch.Tensor],
    memory_features: List[Optional[torch.Tensor]],
    ext_flags: ExtFeatureFlags,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], Dict[str, object]]:
    dominant_num = max(0, int(ext_flags.visionzip_dominant_num))
    contextual_num = max(0, int(ext_flags.visionzip_contextual_num))
    score_type = str(ext_flags.visionzip_score_type).strip().lower()

    compressed_image_features: List[torch.Tensor] = []
    compressed_memory_features: List[Optional[torch.Tensor]] = []
    image_frame_sizes: List[List[int]] = []
    memory_meta: List[Dict[str, int]] = []

    for batch_idx, frame_features in enumerate(image_features):
        if not isinstance(frame_features, torch.Tensor) or frame_features.ndim != 3:
            compressed_image_features.append(frame_features)
            image_frame_sizes.append([])
            tokens_per_frame = None
        else:
            compressed_frames, frame_sizes = _compress_frame_sequence(
                frame_features,
                dominant_num=dominant_num,
                contextual_num=contextual_num,
                score_type=score_type,
            )
            compressed_image_features.append(compressed_frames)
            image_frame_sizes.append(frame_sizes)
            tokens_per_frame = int(frame_features.shape[1])

        memory_bank = memory_features[batch_idx] if batch_idx < len(memory_features) else None
        if memory_bank is None:
            compressed_memory_features.append(None)
            memory_meta.append({"memory_frames_inferred": 0, "memory_tokens_per_frame": 0})
        else:
            compressed_memory_bank, meta = _compress_memory_bank(
                memory_bank,
                tokens_per_frame=tokens_per_frame,
                dominant_num=dominant_num,
                contextual_num=contextual_num,
                score_type=score_type,
            )
            compressed_memory_features.append(compressed_memory_bank)
            memory_meta.append(meta)

    stats: Dict[str, object] = {
        "method": "visionzip",
        "dominant_num": dominant_num,
        "contextual_num": contextual_num,
        "score_type": score_type,
        "image_frame_sizes": image_frame_sizes,
        "memory_meta": memory_meta,
    }
    return compressed_image_features, compressed_memory_features, stats
