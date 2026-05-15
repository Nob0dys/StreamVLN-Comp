from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from streamvln_ext.config.feature_flags import ExtFeatureFlags


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _cluster_dpc_knn(
    features: torch.Tensor,
    cluster_num: int,
    k: int = 5,
) -> Tuple[torch.Tensor, int]:
    with torch.no_grad():
        batch_size, num_tokens, hidden_dim = features.shape
        if num_tokens <= 0:
            empty = torch.empty((batch_size, 0), dtype=torch.long, device=features.device)
            return empty, 0

        cluster_num = max(1, min(int(cluster_num), num_tokens))
        if cluster_num >= num_tokens:
            full = torch.arange(num_tokens, dtype=torch.long, device=features.device).unsqueeze(0).expand(batch_size, -1)
            return full, num_tokens

        k = max(1, min(int(k), num_tokens))
        dist_matrix = torch.cdist(features.float(), features.float()) / (hidden_dim ** 0.5)
        dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        mask = (density[:, None, :] > density[:, :, None]).to(dtype=features.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, center_idx = torch.topk(score, k=cluster_num, dim=-1)

        center_dists = _index_points(dist_matrix, center_idx)
        cluster_idx = center_dists.argmin(dim=1)

        idx_batch = torch.arange(batch_size, device=features.device)[:, None].expand(batch_size, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=features.device)[None, :].expand(batch_size, cluster_num)
        cluster_idx[idx_batch.reshape(-1), center_idx.reshape(-1)] = idx_tmp.reshape(-1)

    return cluster_idx.long(), cluster_num


def _compute_cluster_features(
    features: torch.Tensor,
    cluster_idx: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    batch_size, _, hidden_dim = features.shape
    cluster_onehot = F.one_hot(cluster_idx, num_classes=num_clusters).to(dtype=features.dtype)
    cluster_sums = torch.bmm(cluster_onehot.permute(0, 2, 1), features)
    cluster_counts = cluster_onehot.sum(dim=1)
    cluster_counts_safe = cluster_counts.clone()
    cluster_counts_safe[cluster_counts_safe == 0] = 1
    cluster_features = cluster_sums / cluster_counts_safe.unsqueeze(-1)
    zero_mask = (cluster_counts == 0).unsqueeze(-1)
    cluster_features = cluster_features.masked_fill(zero_mask, 0)
    return cluster_features.view(batch_size, num_clusters, hidden_dim)


def _make_uniform_keep_indices(num_tokens: int, keep_count: int, device: torch.device) -> torch.Tensor:
    if num_tokens <= 0 or keep_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    keep_count = min(num_tokens, keep_count)
    if keep_count == 1:
        return torch.zeros((1,), dtype=torch.long, device=device)
    positions = torch.linspace(0, num_tokens - 1, steps=keep_count, device=device)
    return torch.round(positions).long()


def _select_representative_indices(
    features: torch.Tensor,
    cluster_idx: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    batch_size, num_tokens, _ = features.shape
    device = features.device

    cluster_range = torch.arange(num_clusters, device=device)
    membership = cluster_idx.unsqueeze(1) == cluster_range.view(1, -1, 1)
    member_counts = membership.sum(dim=-1)
    empty_mask = member_counts == 0

    norm_features = F.normalize(features.float(), dim=-1)
    membership_float = membership.float()
    cluster_sums = torch.bmm(membership_float, features.float())
    safe_counts = member_counts.float().clamp(min=1.0).unsqueeze(-1)
    norm_centers = F.normalize(cluster_sums / safe_counts, dim=-1)

    scores = torch.bmm(norm_centers, norm_features.transpose(1, 2))
    scores = scores.masked_fill(~membership, float("-inf"))
    reps = scores.argmax(dim=-1)

    if empty_mask.any():
        fallback = _make_uniform_keep_indices(num_tokens, num_clusters, device)
        reps = torch.where(empty_mask, fallback.unsqueeze(0).expand_as(reps), reps)

    return reps.long()


def _refine_clusters(cluster_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_tokens = cluster_idx.shape
    device = cluster_idx.device
    refined_idx = cluster_idx.clone()

    for batch_idx in range(batch_size):
        seq = cluster_idx[batch_idx]
        diff = seq[1:] != seq[:-1]
        change_pos = diff.nonzero(as_tuple=False).flatten()
        seg_bounds = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                change_pos + 1,
                torch.tensor([num_tokens], device=device, dtype=torch.long),
            ]
        )
        seg_starts = seg_bounds[:-1]
        seg_ends = seg_bounds[1:]
        seg_lens = seg_ends - seg_starts
        seg_labels = seq[seg_starts]

        num_cls = int(seq.max().item()) + 1 if seq.numel() > 0 else 0
        if num_cls <= 0:
            continue

        max_len_per_cls = torch.zeros(num_cls, device=device, dtype=seg_lens.dtype)
        max_len_per_cls.scatter_reduce_(
            0,
            seg_labels.long(),
            seg_lens,
            reduce="amax",
            include_self=True,
        )

        cls_max = max_len_per_cls[seg_labels.long()]
        keep_seg = (cls_max > 1) & (seg_lens == cls_max)

        seg_id = torch.zeros(num_tokens, device=device, dtype=torch.long)
        if seg_starts.shape[0] > 1:
            seg_id[seg_starts[1:]] = 1
        seg_id = seg_id.cumsum(0)
        pos_keep = keep_seg[seg_id]

        refined_batch = seq.clone()
        refined_batch[~pos_keep] = -1

        neg_mask = refined_batch == -1
        if neg_mask.any():
            positions = torch.arange(num_tokens, device=device, dtype=torch.long)
            valid_mask = ~neg_mask

            ff = torch.where(valid_mask, positions, torch.tensor(-1, device=device, dtype=torch.long))
            ff, _ = ff.cummax(dim=0)
            has_left = ff >= 0

            bf = torch.where(valid_mask, positions, torch.tensor(num_tokens, device=device, dtype=torch.long))
            bf_rev, _ = bf.flip(0).cummin(dim=0)
            bf = bf_rev.flip(0)
            has_right = bf < num_tokens

            diff2 = refined_batch[1:] != refined_batch[:-1]
            cp2 = diff2.nonzero(as_tuple=False).flatten()
            sb2 = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    cp2 + 1,
                    torch.tensor([num_tokens], device=device, dtype=torch.long),
                ]
            )
            ss2 = sb2[:-1]
            se2 = sb2[1:]
            sl2 = se2 - ss2
            sid2 = torch.zeros(num_tokens, device=device, dtype=torch.long)
            if ss2.shape[0] > 1:
                sid2[ss2[1:]] = 1
            sid2 = sid2.cumsum(0)
            pos_seg_len = sl2[sid2]

            ff_clamped = ff.clamp(min=0)
            bf_clamped = bf.clamp(max=num_tokens - 1)
            left_label = refined_batch[ff_clamped]
            right_label = refined_batch[bf_clamped]
            left_len = pos_seg_len[ff_clamped]
            right_len = pos_seg_len[bf_clamped]

            left_ok = has_left & (left_label != -1)
            right_ok = has_right & (right_label != -1)
            left_len = left_len * left_ok.long()
            right_len = right_len * right_ok.long()

            use_left = left_ok & (left_len >= right_len)
            new_label = torch.where(
                use_left,
                left_label,
                torch.where(right_ok, right_label, torch.zeros_like(refined_batch)),
            )
            refined_batch = torch.where(neg_mask, new_label, refined_batch)

        refined_idx[batch_idx] = refined_batch

    return refined_idx.long()


def _segment_lengths(cluster_idx: torch.Tensor) -> torch.Tensor:
    device = cluster_idx.device
    batch_size, num_tokens = cluster_idx.shape
    segment_lengths_list: List[torch.Tensor] = []
    max_segments = 0

    for batch_idx in range(batch_size):
        seq = cluster_idx[batch_idx]
        change_points = torch.where(seq[1:] != seq[:-1])[0] + 1
        boundaries = torch.cat(
            [
                torch.tensor([0], device=device),
                change_points,
                torch.tensor([num_tokens], device=device),
            ]
        )
        lengths = boundaries[1:] - boundaries[:-1]
        segment_lengths_list.append(lengths)
        max_segments = max(max_segments, int(lengths.numel()))

    result = torch.zeros((batch_size, max_segments), dtype=torch.long, device=device)
    for batch_idx in range(batch_size):
        lengths = segment_lengths_list[batch_idx]
        result[batch_idx, : lengths.numel()] = lengths
    return result


def _compute_window_similarity(window_frames: torch.Tensor) -> torch.Tensor:
    batch_size, window_size, _, _ = window_frames.shape
    if window_size <= 1:
        return torch.ones((batch_size, window_frames.shape[2]), device=window_frames.device, dtype=torch.float32)

    frames_normed = F.normalize(window_frames.float(), p=2, dim=-1)
    frames_sim = torch.einsum("b w l c, b t l c -> b w t l", frames_normed, frames_normed)
    frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size * (window_size - 1))
    return frames_sim


def _spatial_merge_with_indices(
    features: torch.Tensor,
    keep_indices: torch.Tensor,
    cluster_ratio: float,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = int(features.shape[1])
    num_clusters = max(1, min(num_tokens, int(num_tokens * float(cluster_ratio))))
    if num_clusters >= num_tokens:
        return features, keep_indices

    cluster_idx, cluster_num = _cluster_dpc_knn(features, cluster_num=num_clusters, k=k)
    merged_features = _compute_cluster_features(features, cluster_idx, cluster_num)
    rep_idx = _select_representative_indices(features, cluster_idx, cluster_num)[0]
    return merged_features, keep_indices[rep_idx]


def _process_static_features(
    window_frames: torch.Tensor,
    static_mask: torch.Tensor,
    center_frame: int,
    tokens_per_frame: int,
    cluster_ratio: float,
    k: int,
    min_tokens_for_cluster: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, _, hidden_dim = window_frames.shape
    mask_ref = static_mask[0]
    if int(mask_ref.sum().item()) <= 0:
        return (
            torch.empty((batch_size, 0, hidden_dim), device=window_frames.device, dtype=window_frames.dtype),
            torch.empty((0,), device=window_frames.device, dtype=torch.long),
        )

    selected = window_frames[:, :, mask_ref, :]
    static_feat = selected.mean(dim=1)
    static_keep = mask_ref.nonzero(as_tuple=False).flatten().to(dtype=torch.long, device=window_frames.device)
    static_keep = static_keep + center_frame * tokens_per_frame

    if static_feat.shape[1] > int(min_tokens_for_cluster):
        static_feat, static_keep = _spatial_merge_with_indices(
            static_feat,
            static_keep,
            cluster_ratio=cluster_ratio,
            k=k,
        )
    return static_feat, static_keep


def _process_dynamic_features(
    window_frames: torch.Tensor,
    dynamic_mask: torch.Tensor,
    start_frame: int,
    tokens_per_frame: int,
    cluster_ratio: float,
    k: int,
    min_tokens_for_cluster: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, window_size, _, hidden_dim = window_frames.shape
    device = window_frames.device
    mask_ref = dynamic_mask[0]
    num_dynamic = int(mask_ref.sum().item())

    if num_dynamic <= 0:
        return (
            torch.empty((batch_size, 0, hidden_dim), device=device, dtype=window_frames.dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    all_feats = window_frames[:, :, mask_ref, :]
    base_keep = mask_ref.nonzero(as_tuple=False).flatten().to(dtype=torch.long, device=device)
    offsets = (torch.arange(window_size, device=device) + start_frame) * tokens_per_frame
    all_keep = base_keep.unsqueeze(0) + offsets.unsqueeze(1)

    if num_dynamic > int(min_tokens_for_cluster):
        stacked = all_feats.reshape(batch_size * window_size, num_dynamic, hidden_dim)
        num_clusters = max(1, min(num_dynamic, int(num_dynamic * float(cluster_ratio))))
        if num_clusters < num_dynamic:
            cluster_idx, cluster_num = _cluster_dpc_knn(stacked, cluster_num=num_clusters, k=k)
            merged = _compute_cluster_features(stacked, cluster_idx, cluster_num)
            rep_idx = _select_representative_indices(stacked, cluster_idx, cluster_num)

            merged = merged.reshape(batch_size, window_size * cluster_num, hidden_dim)
            frame_ids = torch.arange(window_size, device=device).repeat(batch_size)
            expanded_keep = all_keep[frame_ids]
            rep_keep = expanded_keep.gather(1, rep_idx)
            dynamic_keep = rep_keep[:window_size].reshape(-1)
            return merged, dynamic_keep

    dynamic_features = all_feats.reshape(batch_size, window_size * num_dynamic, hidden_dim)
    dynamic_keep = all_keep.reshape(-1)
    return dynamic_features, dynamic_keep


def compress_prunevid_sequence(
    frame_sequence: torch.Tensor,
    tau: float,
    cluster_ratio: float,
    temporal_ratio: float,
    k: int,
    min_tokens_for_cluster: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if frame_sequence is None or not isinstance(frame_sequence, torch.Tensor):
        return frame_sequence, {}
    if frame_sequence.ndim != 3:
        return frame_sequence, {}

    num_frames, tokens_per_frame, hidden_dim = frame_sequence.shape
    if num_frames <= 0 or tokens_per_frame <= 0:
        return frame_sequence.new_empty((0, hidden_dim)), {
            "num_frames": int(num_frames),
            "tokens_per_frame": int(tokens_per_frame),
            "window_sizes": [],
            "static_sizes": [],
            "dynamic_sizes": [],
            "num_windows": 0,
        }

    frames = frame_sequence.unsqueeze(0)
    frame_means = frames.mean(dim=2)
    num_temporal_clusters = max(1, min(num_frames, int(num_frames * float(temporal_ratio))))
    if num_temporal_clusters >= num_frames:
        temporal_clusters = torch.arange(num_frames, device=frame_sequence.device, dtype=torch.long).unsqueeze(0)
    else:
        temporal_clusters, _ = _cluster_dpc_knn(
            frame_means,
            cluster_num=num_temporal_clusters,
            k=min(int(k), max(1, num_frames - 1)),
        )
        temporal_clusters = _refine_clusters(temporal_clusters)
    window_lengths = _segment_lengths(temporal_clusters)

    static_features_list: List[torch.Tensor] = []
    dynamic_features_list: List[torch.Tensor] = []
    static_sizes: List[int] = []
    dynamic_sizes: List[int] = []
    window_sizes: List[int] = []
    keep_indices_list: List[torch.Tensor] = []

    start_idx = 0
    for window_size_tensor in window_lengths[0]:
        window_size = int(window_size_tensor.item())
        if window_size <= 0:
            continue

        window_frames = frames[:, start_idx : start_idx + window_size, :, :]
        similarity = _compute_window_similarity(window_frames)
        static_mask = similarity > float(tau)

        center_frame = start_idx + (window_size // 2)
        static_feat, static_keep = _process_static_features(
            window_frames,
            static_mask,
            center_frame=center_frame,
            tokens_per_frame=tokens_per_frame,
            cluster_ratio=cluster_ratio,
            k=k,
            min_tokens_for_cluster=min_tokens_for_cluster,
        )
        dynamic_feat, dynamic_keep = _process_dynamic_features(
            window_frames,
            ~static_mask,
            start_frame=start_idx,
            tokens_per_frame=tokens_per_frame,
            cluster_ratio=cluster_ratio,
            k=k,
            min_tokens_for_cluster=min_tokens_for_cluster,
        )

        static_features_list.append(static_feat)
        dynamic_features_list.append(dynamic_feat)
        static_sizes.append(int(static_feat.shape[1]))
        dynamic_sizes.append(int(dynamic_feat.shape[1]))
        keep_indices_list.append(static_keep)
        keep_indices_list.append(dynamic_keep)
        window_sizes.append(window_size)
        start_idx += window_size

    all_features: List[torch.Tensor] = []
    for static_feat, dynamic_feat in zip(static_features_list, dynamic_features_list):
        all_features.append(static_feat)
        all_features.append(dynamic_feat)

    if all_features:
        compressed = torch.cat(all_features, dim=1).squeeze(0).contiguous()
        keep_indices = torch.cat(keep_indices_list, dim=0) if keep_indices_list else torch.empty((0,), device=frame_sequence.device, dtype=torch.long)
    else:
        compressed = frame_sequence.flatten(0, 1).contiguous()
        keep_indices = torch.arange(
            frame_sequence.shape[0] * frame_sequence.shape[1],
            device=frame_sequence.device,
            dtype=torch.long,
        )

    metadata: Dict[str, object] = {
        "num_frames": int(num_frames),
        "tokens_per_frame": int(tokens_per_frame),
        "window_sizes": window_sizes,
        "static_sizes": static_sizes,
        "dynamic_sizes": dynamic_sizes,
        "num_windows": len(window_sizes),
        "keep_indices": keep_indices,
        "output_tokens": int(compressed.shape[0]),
    }
    return compressed, metadata


def _compress_memory_bank(
    memory_bank: torch.Tensor,
    image_tokens_per_frame: Optional[int],
    tau: float,
    cluster_ratio: float,
    temporal_ratio: float,
    k: int,
    min_tokens_for_cluster: int,
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
        total_tokens = int(slot.shape[0])
        if image_tokens_per_frame and image_tokens_per_frame > 0 and total_tokens % image_tokens_per_frame == 0:
            num_frames = max(1, total_tokens // image_tokens_per_frame)
            inferred_frames = max(inferred_frames, num_frames)
            inferred_tokens_per_frame = int(image_tokens_per_frame)
            slot_frames = slot.view(num_frames, image_tokens_per_frame, slot.shape[-1])
        else:
            slot_frames = slot.unsqueeze(0)

        compressed_slot, meta = compress_prunevid_sequence(
            slot_frames,
            tau=tau,
            cluster_ratio=cluster_ratio,
            temporal_ratio=temporal_ratio,
            k=k,
            min_tokens_for_cluster=min_tokens_for_cluster,
        )
        compressed_slots.append(compressed_slot)
        slot_sizes.append(int(compressed_slot.shape[0]))
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


def apply_prunevid_compression(
    image_features: List[torch.Tensor],
    memory_features: List[Optional[torch.Tensor]],
    ext_flags: ExtFeatureFlags,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], Dict[str, object]]:
    tau = float(ext_flags.prunevid_tau)
    cluster_ratio = float(ext_flags.prunevid_cluster_ratio)
    temporal_ratio = float(ext_flags.prunevid_temporal_ratio)
    k = int(ext_flags.prunevid_k)
    min_tokens_for_cluster = int(ext_flags.prunevid_min_tokens_for_cluster)

    compressed_image_features: List[torch.Tensor] = []
    compressed_memory_features: List[Optional[torch.Tensor]] = []
    image_meta: List[Dict[str, object]] = []
    memory_meta: List[Dict[str, object]] = []

    for batch_idx, frame_features in enumerate(image_features):
        image_tokens_per_frame = None
        if not isinstance(frame_features, torch.Tensor) or frame_features.ndim != 3:
            compressed_image_features.append(frame_features)
            image_meta.append({})
        else:
            image_tokens_per_frame = int(frame_features.shape[1])
            compressed_tokens, meta = compress_prunevid_sequence(
                frame_features,
                tau=tau,
                cluster_ratio=cluster_ratio,
                temporal_ratio=temporal_ratio,
                k=k,
                min_tokens_for_cluster=min_tokens_for_cluster,
            )
            compressed_image_features.append(compressed_tokens.unsqueeze(0))
            image_meta.append(meta)

        memory_bank = memory_features[batch_idx] if batch_idx < len(memory_features) else None
        if memory_bank is None:
            compressed_memory_features.append(None)
            memory_meta.append({"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []})
        else:
            compressed_memory_bank, meta = _compress_memory_bank(
                memory_bank,
                image_tokens_per_frame=image_tokens_per_frame,
                tau=tau,
                cluster_ratio=cluster_ratio,
                temporal_ratio=temporal_ratio,
                k=k,
                min_tokens_for_cluster=min_tokens_for_cluster,
            )
            compressed_memory_features.append(compressed_memory_bank)
            memory_meta.append(meta)

    stats: Dict[str, object] = {
        "method": "prunevid",
        "tau": tau,
        "cluster_ratio": cluster_ratio,
        "temporal_ratio": temporal_ratio,
        "k": k,
        "min_tokens_for_cluster": min_tokens_for_cluster,
        "image_meta": image_meta,
        "memory_meta": memory_meta,
    }
    return compressed_image_features, compressed_memory_features, stats
