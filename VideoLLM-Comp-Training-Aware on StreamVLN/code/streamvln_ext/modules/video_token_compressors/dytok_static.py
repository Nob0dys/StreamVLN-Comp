from typing import Dict, List, Optional, Tuple

import torch

from streamvln_ext.config.feature_flags import ExtFeatureFlags
from streamvln_ext.modules.video_token_compressors.visionzip import (
    _token_scores,
    compress_visionzip_tokens,
)


def _frame_importance_weights(frame_sequence: torch.Tensor, score_type: str) -> torch.Tensor:
    if frame_sequence is None or not isinstance(frame_sequence, torch.Tensor) or frame_sequence.ndim != 3:
        return torch.empty((0,), dtype=torch.float32, device=frame_sequence.device if isinstance(frame_sequence, torch.Tensor) else None)

    weights: List[torch.Tensor] = []
    for frame in frame_sequence:
        if frame.numel() == 0:
            weights.append(torch.zeros((1,), dtype=torch.float32, device=frame_sequence.device))
        else:
            token_scores = _token_scores(frame, score_type=score_type)
            weights.append(token_scores.float().mean().view(1))

    if not weights:
        return torch.empty((0,), dtype=torch.float32, device=frame_sequence.device)

    stacked = torch.cat(weights, dim=0)
    if not torch.isfinite(stacked).all() or float(stacked.abs().sum()) <= 0.0:
        return torch.full(
            (frame_sequence.shape[0],),
            fill_value=1.0 / max(1, int(frame_sequence.shape[0])),
            dtype=torch.float32,
            device=frame_sequence.device,
        )
    return torch.softmax(stacked, dim=0)


def _allocate_frame_budgets(
    num_frames: int,
    tokens_per_frame: int,
    upper_limit_ratio: float,
    min_ratio: float,
    frame_weights: torch.Tensor,
) -> torch.Tensor:
    if num_frames <= 0 or tokens_per_frame <= 0:
        return torch.empty((0,), dtype=torch.long, device=frame_weights.device)

    max_total = int(num_frames * tokens_per_frame)
    target_total = int(round(float(upper_limit_ratio) * max_total))
    target_total = max(num_frames, min(max_total, target_total))

    min_budget = int(round(float(min_ratio) * tokens_per_frame))
    min_budget = max(1, min(tokens_per_frame, min_budget))
    if min_budget * num_frames > target_total:
        min_budget = max(1, target_total // num_frames)

    budgets = torch.full((num_frames,), fill_value=min_budget, dtype=torch.long, device=frame_weights.device)
    remaining = int(target_total - budgets.sum().item())
    if remaining <= 0:
        return budgets

    weights = frame_weights.float()
    if weights.numel() != num_frames or not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = torch.full((num_frames,), fill_value=1.0 / num_frames, dtype=torch.float32, device=budgets.device)
    else:
        weights = weights / weights.sum()

    desired = weights * remaining
    increments = torch.floor(desired).long()
    budgets = budgets + increments
    budgets = budgets.clamp(max=tokens_per_frame)

    left = int(target_total - budgets.sum().item())
    if left > 0:
        remainders = desired - increments.float()
        while left > 0:
            candidates = torch.where(budgets < tokens_per_frame)[0]
            if candidates.numel() == 0:
                break
            candidate_scores = remainders[candidates]
            order = candidates[torch.argsort(candidate_scores, descending=True)]
            updated = False
            for idx in order:
                if left <= 0:
                    break
                if int(budgets[idx].item()) >= tokens_per_frame:
                    continue
                budgets[idx] += 1
                left -= 1
                updated = True
            if not updated:
                break

    return budgets.long()


def _split_budgets(total_budgets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if total_budgets.numel() == 0:
        empty = total_budgets.new_empty((0,))
        return empty, empty

    dominant = ((6 * total_budgets) // 7).clamp(min=1)
    dominant = torch.minimum(dominant, total_budgets)
    contextual = total_budgets - dominant
    return dominant.long(), contextual.long()


def _compress_sequence_with_budgets(
    frame_sequence: torch.Tensor,
    dominant_budgets: torch.Tensor,
    contextual_budgets: torch.Tensor,
    score_type: str,
) -> Tuple[List[torch.Tensor], List[int]]:
    compressed_frames: List[torch.Tensor] = []
    frame_sizes: List[int] = []

    for frame_idx, frame in enumerate(frame_sequence):
        compressed_frame, _ = compress_visionzip_tokens(
            frame,
            dominant_num=int(dominant_budgets[frame_idx].item()),
            contextual_num=int(contextual_budgets[frame_idx].item()),
            score_type=score_type,
        )
        compressed_frames.append(compressed_frame)
        frame_sizes.append(int(compressed_frame.shape[0]))

    return compressed_frames, frame_sizes


def _compress_image_sequence(
    frame_sequence: torch.Tensor,
    upper_limit_ratio: float,
    min_ratio: float,
    score_type: str,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if frame_sequence is None or not isinstance(frame_sequence, torch.Tensor) or frame_sequence.ndim != 3:
        return frame_sequence, {}

    num_frames, tokens_per_frame, hidden_dim = frame_sequence.shape
    frame_weights = _frame_importance_weights(frame_sequence, score_type=score_type)
    total_budgets = _allocate_frame_budgets(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        upper_limit_ratio=upper_limit_ratio,
        min_ratio=min_ratio,
        frame_weights=frame_weights,
    )
    dominant_budgets, contextual_budgets = _split_budgets(total_budgets)

    compressed_frames, frame_sizes = _compress_sequence_with_budgets(
        frame_sequence,
        dominant_budgets=dominant_budgets,
        contextual_budgets=contextual_budgets,
        score_type=score_type,
    )

    if len(compressed_frames) == 1:
        compressed = compressed_frames[0].unsqueeze(0)
    else:
        min_len = min(frame_sizes) if frame_sizes else 0
        if min_len <= 0:
            compressed = frame_sequence.new_empty((0, 0, hidden_dim))
        else:
            compressed = torch.stack([frame[:min_len] for frame in compressed_frames], dim=0)

    metadata: Dict[str, object] = {
        "num_frames": int(num_frames),
        "tokens_per_frame": int(tokens_per_frame),
        "frame_weights": [float(x) for x in frame_weights.detach().cpu().tolist()],
        "total_budgets": [int(x) for x in total_budgets.detach().cpu().tolist()],
        "dominant_budgets": [int(x) for x in dominant_budgets.detach().cpu().tolist()],
        "contextual_budgets": [int(x) for x in contextual_budgets.detach().cpu().tolist()],
        "frame_output_sizes": frame_sizes,
    }
    return compressed, metadata


def _compress_memory_bank(
    memory_bank: torch.Tensor,
    tokens_per_frame: Optional[int],
    upper_limit_ratio: float,
    min_ratio: float,
    score_type: str,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        return memory_bank, {"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []}
    if memory_bank.ndim != 3:
        return memory_bank, {"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []}

    compressed_slots: List[torch.Tensor] = []
    slot_sizes: List[int] = []
    slot_meta: List[Dict[str, object]] = []
    inferred_frames = 0
    inferred_tokens_per_frame = 0

    for slot in memory_bank:
        if tokens_per_frame and tokens_per_frame > 0 and int(slot.shape[0]) % tokens_per_frame == 0:
            num_frames = max(1, int(slot.shape[0]) // tokens_per_frame)
            inferred_frames = max(inferred_frames, num_frames)
            inferred_tokens_per_frame = int(tokens_per_frame)
            slot_frames = slot.view(num_frames, tokens_per_frame, slot.shape[-1])
        else:
            slot_frames = slot.unsqueeze(0)

        frame_weights = _frame_importance_weights(slot_frames, score_type=score_type)
        total_budgets = _allocate_frame_budgets(
            num_frames=int(slot_frames.shape[0]),
            tokens_per_frame=int(slot_frames.shape[1]),
            upper_limit_ratio=upper_limit_ratio,
            min_ratio=min_ratio,
            frame_weights=frame_weights,
        )
        dominant_budgets, contextual_budgets = _split_budgets(total_budgets)
        compressed_frames, frame_sizes = _compress_sequence_with_budgets(
            slot_frames,
            dominant_budgets=dominant_budgets,
            contextual_budgets=contextual_budgets,
            score_type=score_type,
        )
        compressed_slot = torch.cat(compressed_frames, dim=0).contiguous() if compressed_frames else slot.new_empty((0, slot.shape[-1]))
        compressed_slots.append(compressed_slot)
        slot_sizes.append(int(compressed_slot.shape[0]))
        slot_meta.append(
            {
                "num_frames": int(slot_frames.shape[0]),
                "tokens_per_frame": int(slot_frames.shape[1]),
                "frame_weights": [float(x) for x in frame_weights.detach().cpu().tolist()],
                "total_budgets": [int(x) for x in total_budgets.detach().cpu().tolist()],
                "dominant_budgets": [int(x) for x in dominant_budgets.detach().cpu().tolist()],
                "contextual_budgets": [int(x) for x in contextual_budgets.detach().cpu().tolist()],
                "frame_output_sizes": frame_sizes,
            }
        )

    min_len = min(slot_sizes) if slot_sizes else 0
    if min_len <= 0:
        hidden = int(memory_bank.shape[-1])
        return memory_bank.new_empty((0, 0, hidden)), {
            "memory_slots": int(memory_bank.shape[0]),
            "memory_frames_inferred": inferred_frames,
            "memory_tokens_per_frame": inferred_tokens_per_frame,
            "slot_meta": slot_meta,
        }

    aligned = [slot[:min_len] for slot in compressed_slots]
    return torch.stack(aligned, dim=0), {
        "memory_slots": int(memory_bank.shape[0]),
        "memory_frames_inferred": inferred_frames,
        "memory_tokens_per_frame": inferred_tokens_per_frame,
        "slot_meta": slot_meta,
    }


def apply_dytok_static_compression(
    image_features: List[torch.Tensor],
    memory_features: List[Optional[torch.Tensor]],
    ext_flags: ExtFeatureFlags,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], Dict[str, object]]:
    base_compressor = str(ext_flags.dytok_static_base_compressor).strip().lower()
    if base_compressor not in {"visionzip", ""}:
        raise ValueError(f"Unsupported DyToK static base compressor: {ext_flags.dytok_static_base_compressor}")

    upper_limit_ratio = float(ext_flags.dytok_static_upper_limit_ratio)
    min_ratio = float(ext_flags.dytok_static_min_ratio)
    score_type = str(ext_flags.visionzip_score_type).strip().lower()

    compressed_image_features: List[torch.Tensor] = []
    compressed_memory_features: List[Optional[torch.Tensor]] = []
    image_meta: List[Dict[str, object]] = []
    memory_meta: List[Dict[str, object]] = []

    for batch_idx, frame_features in enumerate(image_features):
        tokens_per_frame = None
        if not isinstance(frame_features, torch.Tensor) or frame_features.ndim != 3:
            compressed_image_features.append(frame_features)
            image_meta.append({})
        else:
            tokens_per_frame = int(frame_features.shape[1])
            compressed_frames, meta = _compress_image_sequence(
                frame_features,
                upper_limit_ratio=upper_limit_ratio,
                min_ratio=min_ratio,
                score_type=score_type,
            )
            compressed_image_features.append(compressed_frames)
            image_meta.append(meta)

        memory_bank = memory_features[batch_idx] if batch_idx < len(memory_features) else None
        if memory_bank is None:
            compressed_memory_features.append(None)
            memory_meta.append({"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []})
        else:
            compressed_memory_bank, meta = _compress_memory_bank(
                memory_bank,
                tokens_per_frame=tokens_per_frame,
                upper_limit_ratio=upper_limit_ratio,
                min_ratio=min_ratio,
                score_type=score_type,
            )
            compressed_memory_features.append(compressed_memory_bank)
            memory_meta.append(meta)

    stats: Dict[str, object] = {
        "method": "dytok_static",
        "base_compressor": "visionzip",
        "upper_limit_ratio": upper_limit_ratio,
        "min_ratio": min_ratio,
        "score_type": score_type,
        "image_meta": image_meta,
        "memory_meta": memory_meta,
    }
    return compressed_image_features, compressed_memory_features, stats
