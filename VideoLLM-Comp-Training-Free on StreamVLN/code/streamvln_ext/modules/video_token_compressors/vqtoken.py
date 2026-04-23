from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from streamvln_ext.config.feature_flags import ExtFeatureFlags


def _sample_init_centroids(x_norm: torch.Tensor, num_clusters: int) -> torch.Tensor:
    n_samples, dim = x_norm.shape
    first_idx = torch.randint(0, n_samples, (1,), device=x_norm.device).item()
    centroids = torch.empty((num_clusters, dim), device=x_norm.device, dtype=x_norm.dtype)
    centroids[0] = x_norm[first_idx]
    if num_clusters == 1:
        return centroids

    min_dists = torch.cdist(x_norm, centroids[0:1], p=2).squeeze(1)
    eps = 1e-6
    for idx in range(1, num_clusters):
        weights = min_dists.pow(2).clamp(min=0.0) + eps
        probs = weights / weights.sum()
        try:
            next_idx = torch.multinomial(probs, 1).item()
        except RuntimeError:
            next_idx = torch.randint(0, n_samples, (1,), device=x_norm.device).item()
        centroids[idx] = x_norm[next_idx]
        new_dists = torch.cdist(x_norm, centroids[idx : idx + 1], p=2).squeeze(1)
        min_dists = torch.minimum(min_dists, new_dists)
    return centroids


def _kmeans_cosine(x: torch.Tensor, num_clusters: int, max_iteration: int = 50, tol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    n_samples = int(x.shape[0])
    num_clusters = max(1, min(int(num_clusters), n_samples))
    if num_clusters >= n_samples:
        labels = torch.arange(n_samples, device=x.device, dtype=torch.long)
        return labels, x

    x_norm = F.normalize(x, dim=-1)
    centroids = _sample_init_centroids(x_norm, num_clusters)

    for _ in range(max_iteration):
        centroid_norm = F.normalize(centroids, dim=-1)
        distances = 1 - torch.mm(x_norm, centroid_norm.transpose(0, 1))
        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(0, labels.unsqueeze(1).expand_as(x_norm), x_norm)
        counts = torch.zeros((num_clusters,), device=x.device, dtype=x_norm.dtype)
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=x_norm.dtype))
        empty_mask = counts == 0
        counts = counts.clamp(min=1.0)
        new_centroids = new_centroids / counts.unsqueeze(1)
        new_centroids[empty_mask] = centroids[empty_mask]

        if torch.norm(centroids - new_centroids) < float(tol):
            centroids = new_centroids
            break
        centroids = new_centroids

    return labels.long(), centroids


def _select_representatives(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    num_clusters = int(centroids.shape[0])
    device = tokens.device
    norm_tokens = F.normalize(tokens.float(), dim=-1)
    norm_centroids = F.normalize(centroids.float(), dim=-1)

    representative_indices = []
    for cluster_id in range(num_clusters):
        member_idx = torch.nonzero(labels == cluster_id, as_tuple=False).flatten()
        if member_idx.numel() == 0:
            representative_indices.append(torch.tensor(0, device=device, dtype=torch.long))
            continue
        member_tokens = norm_tokens.index_select(0, member_idx)
        similarity = torch.matmul(member_tokens, norm_centroids[cluster_id])
        best_local = torch.argmax(similarity)
        representative_indices.append(member_idx[best_local])

    rep_idx = torch.stack(representative_indices, dim=0)
    rep_idx, _ = torch.sort(rep_idx)
    return rep_idx.long()


def _compress_frame_tokens(
    frame_tokens: torch.Tensor,
    num_clusters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = int(frame_tokens.shape[0])
    num_clusters = max(1, min(int(num_clusters), num_tokens))
    if num_clusters >= num_tokens:
        keep_idx = torch.arange(num_tokens, device=frame_tokens.device, dtype=torch.long)
        return frame_tokens, keep_idx

    labels, centroids = _kmeans_cosine(frame_tokens, num_clusters=num_clusters)
    rep_idx = _select_representatives(frame_tokens, labels, centroids)
    compressed = frame_tokens.index_select(0, rep_idx)
    return compressed.contiguous(), rep_idx


def _compress_image_sequence(
    frame_sequence: torch.Tensor,
    ext_flags: ExtFeatureFlags,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if frame_sequence is None or not isinstance(frame_sequence, torch.Tensor) or frame_sequence.ndim != 3:
        return frame_sequence, {}

    if ext_flags.vqtoken_use_cross_attention:
        raise ValueError("Current StreamVLN Phase 2 VQToken only supports use_cross_attention=false.")
    if ext_flags.vqtoken_adaptive:
        raise ValueError("Current StreamVLN Phase 2 VQToken only supports adaptive=false.")

    num_frames = int(frame_sequence.shape[0])
    num_clusters = int(ext_flags.vqtoken_num_clusters)
    compressed_frames: List[torch.Tensor] = []
    keep_indices_per_frame: List[List[int]] = []
    frame_output_sizes: List[int] = []
    for frame in frame_sequence:
        compressed_frame, keep_idx = _compress_frame_tokens(frame, num_clusters=num_clusters)
        compressed_frames.append(compressed_frame)
        keep_indices_per_frame.append([int(x) for x in keep_idx.tolist()])
        frame_output_sizes.append(int(compressed_frame.shape[0]))

    compressed = torch.stack(compressed_frames, dim=0)
    meta = {
        "num_frames": num_frames,
        "num_clusters": int(compressed.shape[1]),
        "frame_output_sizes": frame_output_sizes,
        "frame_keep_indices": keep_indices_per_frame,
    }
    return compressed, meta


def _compress_memory_bank(
    memory_bank: torch.Tensor,
    tokens_per_frame: Optional[int],
    ext_flags: ExtFeatureFlags,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        return memory_bank, {"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []}
    if memory_bank.ndim != 3:
        return memory_bank, {"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []}

    if ext_flags.vqtoken_use_cross_attention:
        raise ValueError("Current StreamVLN Phase 2 VQToken only supports use_cross_attention=false.")
    if ext_flags.vqtoken_adaptive:
        raise ValueError("Current StreamVLN Phase 2 VQToken only supports adaptive=false.")

    compressed_slots: List[torch.Tensor] = []
    slot_meta: List[Dict[str, object]] = []
    slot_sizes: List[int] = []
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

        compressed_frames, meta = _compress_image_sequence(slot_frames, ext_flags=ext_flags)
        compressed_slot = compressed_frames.flatten(0, 1).contiguous()
        compressed_slots.append(compressed_slot)
        slot_sizes.append(int(compressed_slot.shape[0]))
        meta["num_frames"] = int(slot_frames.shape[0])
        meta["tokens_per_frame"] = int(slot_frames.shape[1])
        slot_meta.append(meta)

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


def apply_vqtoken_compression(
    image_features: List[torch.Tensor],
    memory_features: List[Optional[torch.Tensor]],
    ext_flags: ExtFeatureFlags,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], Dict[str, object]]:
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
            compressed_frames, meta = _compress_image_sequence(frame_features, ext_flags=ext_flags)
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
                ext_flags=ext_flags,
            )
            compressed_memory_features.append(compressed_memory_bank)
            memory_meta.append(meta)

    stats: Dict[str, object] = {
        "method": "vqtoken",
        "num_clusters": int(ext_flags.vqtoken_num_clusters),
        "adaptive": bool(ext_flags.vqtoken_adaptive),
        "max_clusters": int(ext_flags.vqtoken_max_clusters),
        "adaptive_method": str(ext_flags.vqtoken_adaptive_method),
        "use_cross_attention": bool(ext_flags.vqtoken_use_cross_attention),
        "image_meta": image_meta,
        "memory_meta": memory_meta,
    }
    return compressed_image_features, compressed_memory_features, stats
