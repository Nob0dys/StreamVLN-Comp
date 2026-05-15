from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F


def _compute_keep_tokens(num_tokens: int, keep_ratio: float, min_tokens: int) -> int:
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    keep = int(num_tokens * keep_ratio)
    keep = max(int(min_tokens), keep)
    keep = max(1, keep)
    keep = min(num_tokens, keep)
    return keep


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return scores.float()
    s_min = scores.min()
    s_max = scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-6)


def _score_l2(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), device=tokens.device, dtype=torch.float32)
    return torch.norm(tokens.float(), dim=-1)


def _score_attn_proxy(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), device=tokens.device, dtype=torch.float32)

    x = F.normalize(tokens.float(), dim=-1)
    query = x.mean(dim=0, keepdim=True)
    attn = torch.matmul(x, query.transpose(0, 1)).squeeze(-1)
    return _normalize_scores(attn)


def _score_random(tokens: torch.Tensor, random_seed: Optional[int]) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), device=tokens.device, dtype=torch.float32)

    seed = 42 if random_seed is None else int(random_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scores = torch.rand((tokens.shape[0],), generator=generator, dtype=torch.float32)
    return scores.to(device=tokens.device)


def _token_scores(
    tokens: torch.Tensor,
    score_type: str = "l2",
    hybrid_alpha: float = 0.5,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    score_key = str(score_type).strip().lower()

    if score_key in {"attn", "attention", "attn_proxy"}:
        return _score_attn_proxy(tokens)

    if score_key in {"hybrid", "attn_hybrid", "norm_attn_hybrid"}:
        alpha = max(0.0, min(1.0, float(hybrid_alpha)))
        l2 = _normalize_scores(_score_l2(tokens))
        attn = _score_attn_proxy(tokens)
        return alpha * l2 + (1.0 - alpha) * attn

    if score_key == "random":
        return _score_random(tokens, random_seed=random_seed)

    return _score_l2(tokens)


def _tome_similarity_matrix(
    src_tokens: torch.Tensor,
    dst_tokens: torch.Tensor,
    similarity_metric: str = "cosine",
) -> torch.Tensor:
    metric = str(similarity_metric).strip().lower()
    src_f = src_tokens.float()
    dst_f = dst_tokens.float()

    if metric in {"dot", "inner", "inner_product"}:
        return torch.matmul(src_f, dst_f.transpose(0, 1))

    src_n = F.normalize(src_f, dim=-1)
    dst_n = F.normalize(dst_f, dim=-1)
    return torch.matmul(src_n, dst_n.transpose(0, 1))


def _tome_merge_once(
    tokens: torch.Tensor,
    token_sizes: torch.Tensor,
    max_merges: int,
    similarity_metric: str = "cosine",
) -> Tuple[torch.Tensor, torch.Tensor]:
    total = int(tokens.shape[0])
    if total <= 1 or max_merges <= 0:
        return tokens, token_sizes

    device = tokens.device
    src_idx = torch.arange(0, total, 2, device=device, dtype=torch.long)
    dst_idx = torch.arange(1, total, 2, device=device, dtype=torch.long)
    if src_idx.numel() == 0 or dst_idx.numel() == 0:
        return tokens, token_sizes

    src_tokens = tokens.index_select(0, src_idx)
    dst_tokens = tokens.index_select(0, dst_idx)
    scores = _tome_similarity_matrix(src_tokens, dst_tokens, similarity_metric=similarity_metric)

    if scores.numel() == 0:
        return tokens, token_sizes

    target_merges = min(int(max_merges), int(src_idx.numel()), int(dst_idx.numel()))
    if target_merges <= 0:
        return tokens, token_sizes

    flat_sorted = torch.argsort(scores.reshape(-1), descending=True)
    src_used = torch.zeros((src_idx.numel(),), dtype=torch.bool, device=device)
    dst_used = torch.zeros((dst_idx.numel(),), dtype=torch.bool, device=device)
    dst_count = int(dst_idx.numel())

    selected_pairs = []
    for flat in flat_sorted.tolist():
        if len(selected_pairs) >= target_merges:
            break
        s_local = int(flat // dst_count)
        d_local = int(flat % dst_count)
        if src_used[s_local] or dst_used[d_local]:
            continue
        src_used[s_local] = True
        dst_used[d_local] = True
        selected_pairs.append((int(src_idx[s_local].item()), int(dst_idx[d_local].item())))

    if len(selected_pairs) == 0:
        return tokens, token_sizes

    token_f = tokens.float()
    size_f = token_sizes.float()

    src_remove: Set[int] = set()
    dst_to_merged: Dict[int, torch.Tensor] = {}
    dst_to_merged_size: Dict[int, torch.Tensor] = {}

    for src_global, dst_global in selected_pairs:
        src_w = size_f[src_global]
        dst_w = size_f[dst_global]
        merged_w = src_w + dst_w
        merged_token = (token_f[src_global] * src_w + token_f[dst_global] * dst_w) / (merged_w + 1e-6)

        src_remove.add(src_global)
        dst_to_merged[dst_global] = merged_token
        dst_to_merged_size[dst_global] = merged_w

    merged_tokens_out = []
    merged_sizes_out = []
    for idx in range(total):
        if idx in src_remove:
            continue
        if idx in dst_to_merged:
            merged_tokens_out.append(dst_to_merged[idx])
            merged_sizes_out.append(dst_to_merged_size[idx])
        else:
            merged_tokens_out.append(token_f[idx])
            merged_sizes_out.append(size_f[idx])

    if len(merged_tokens_out) == 0:
        return tokens, token_sizes

    new_tokens = torch.stack(merged_tokens_out, dim=0).to(dtype=tokens.dtype)
    new_sizes = torch.stack(merged_sizes_out, dim=0).to(device=tokens.device, dtype=torch.float32)
    return new_tokens, new_sizes


def merge_visual_tokens_tome(
    tokens: torch.Tensor,
    keep_ratio: float = 0.8,
    min_tokens: int = 64,
    similarity_metric: str = "cosine",
) -> torch.Tensor:
    if tokens is None or not isinstance(tokens, torch.Tensor):
        return tokens
    if tokens.ndim != 2:
        return tokens

    total = int(tokens.shape[0])
    if total <= 1:
        return tokens

    keep = _compute_keep_tokens(total, keep_ratio=keep_ratio, min_tokens=min_tokens)
    if keep >= total:
        return tokens

    current_tokens = tokens
    current_sizes = torch.ones((total,), dtype=torch.float32, device=tokens.device)

    while int(current_tokens.shape[0]) > keep:
        need_merge = int(current_tokens.shape[0]) - keep
        before = int(current_tokens.shape[0])
        current_tokens, current_sizes = _tome_merge_once(
            current_tokens,
            current_sizes,
            max_merges=need_merge,
            similarity_metric=similarity_metric,
        )
        after = int(current_tokens.shape[0])
        if after >= before:
            break

    if int(current_tokens.shape[0]) > keep:
        fallback_ratio = float(keep) / float(max(1, int(current_tokens.shape[0])))
        current_tokens = prune_visual_tokens_tuning_free(
            current_tokens,
            keep_ratio=fallback_ratio,
            min_tokens=keep,
            score_type="l2",
        )

    return current_tokens


def prune_visual_tokens_tuning_free(
    tokens: torch.Tensor,
    keep_ratio: float = 0.8,
    min_tokens: int = 64,
    score_type: str = "l2",
    hybrid_alpha: float = 0.5,
    random_seed: Optional[int] = None,
    external_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if tokens is None or not isinstance(tokens, torch.Tensor):
        return tokens
    if tokens.ndim != 2:
        return tokens

    total = tokens.shape[0]
    if total <= 0:
        return tokens

    keep = _compute_keep_tokens(total, keep_ratio=keep_ratio, min_tokens=min_tokens)
    if keep >= total:
        return tokens

    if external_scores is not None:
        scores = external_scores.to(device=tokens.device, dtype=torch.float32)
        if scores.ndim != 1 or scores.numel() != total:
            raise ValueError("external_scores must be 1D and match number of tokens")
    else:
        scores = _token_scores(
            tokens,
            score_type=score_type,
            hybrid_alpha=hybrid_alpha,
            random_seed=random_seed,
        )
    keep_idx = torch.topk(scores, k=keep, largest=True, sorted=False).indices
    keep_idx, _ = torch.sort(keep_idx)
    return tokens.index_select(0, keep_idx).contiguous()


def build_protected_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    memory_token_id: int,
    keep_special_tokens: bool = True,
) -> torch.Tensor:
    if not isinstance(input_ids, torch.Tensor):
        return torch.zeros((0,), dtype=torch.bool)

    if input_ids.ndim != 1:
        raise ValueError(f"Expected 1D input_ids, got {tuple(input_ids.shape)}")

    if not keep_special_tokens:
        return torch.zeros_like(input_ids, dtype=torch.bool)

    return (input_ids == int(image_token_id)) | (input_ids == int(memory_token_id))


def prune_text_embeds_tuning_free(
    text_embeds: torch.Tensor,
    keep_ratio: float = 0.9,
    min_tokens: int = 1,
    protected_mask: Optional[torch.Tensor] = None,
    score_type: str = "l2",
    hybrid_alpha: float = 0.5,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if text_embeds is None or not isinstance(text_embeds, torch.Tensor):
        return text_embeds, torch.tensor([], dtype=torch.long)
    if text_embeds.ndim != 2:
        keep_idx = torch.arange(text_embeds.shape[0], device=text_embeds.device, dtype=torch.long)
        return text_embeds, keep_idx

    total = text_embeds.shape[0]
    if total <= 0:
        keep_idx = torch.arange(0, device=text_embeds.device, dtype=torch.long)
        return text_embeds, keep_idx

    keep = _compute_keep_tokens(total, keep_ratio=keep_ratio, min_tokens=min_tokens)
    if keep >= total:
        keep_idx = torch.arange(total, device=text_embeds.device, dtype=torch.long)
        return text_embeds, keep_idx

    if protected_mask is None:
        protected_mask = torch.zeros((total,), dtype=torch.bool, device=text_embeds.device)
    else:
        protected_mask = protected_mask.to(device=text_embeds.device, dtype=torch.bool)
        if protected_mask.numel() != total:
            raise ValueError("protected_mask length must match number of text tokens")

    protected_idx = torch.where(protected_mask)[0]
    scores = _token_scores(
        text_embeds,
        score_type=score_type,
        hybrid_alpha=hybrid_alpha,
        random_seed=random_seed,
    )

    remaining_keep = max(0, keep - int(protected_idx.numel()))
    non_protected_idx = torch.where(~protected_mask)[0]

    if remaining_keep > 0 and non_protected_idx.numel() > 0:
        non_protected_scores = scores.index_select(0, non_protected_idx)
        take = min(int(remaining_keep), int(non_protected_idx.numel()))
        rel_idx = torch.topk(non_protected_scores, k=take, largest=True, sorted=False).indices
        selected_non_protected = non_protected_idx.index_select(0, rel_idx)
    else:
        selected_non_protected = torch.empty((0,), dtype=torch.long, device=text_embeds.device)

    keep_idx = torch.cat([protected_idx, selected_non_protected], dim=0)
    if keep_idx.numel() == 0:
        keep_idx = torch.topk(scores, k=max(1, keep), largest=True, sorted=False).indices

    keep_idx, _ = torch.sort(torch.unique(keep_idx, sorted=True))
    if keep_idx.numel() > keep:
        keep_idx = keep_idx[:keep]

    return text_embeds.index_select(0, keep_idx).contiguous(), keep_idx
