import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from streamvln.model.stream_video_vln import StreamVLNForCausalLM
from streamvln.utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
from streamvln_ext.config.feature_flags import ExtFeatureFlags, load_feature_flags
from streamvln_ext.modules.dynamic_memory import update_memory_bank
from streamvln_ext.modules.kv_sliding import truncate_past_key_values
from streamvln_ext.modules.memory_loss import build_memory_pseudo_labels, compute_weighted_causal_loss
from streamvln_ext.modules.multiscale_memory import build_multiscale_memory
from streamvln_ext.modules.runtime_metrics import RuntimeTokenLatencyMetrics
from streamvln_ext.modules.token_selection import select_memory_bank, select_tokens
from streamvln_ext.modules.history_conditioned_pruning import prune_tokens_history_conditioned
from streamvln_ext.modules.tuning_free_mm_pruning import (
    merge_visual_tokens_tome,
    prune_text_embeds_tuning_free,
    prune_visual_tokens_tuning_free,
)
from streamvln_ext.modules.video_token_compressors import (
    apply_fastvid_compression,
    apply_dytok_static_compression,
    LLaMAVIDStreamVLNCompressor,
    LongVUStreamVLNCompressor,
    apply_prunevid_compression,
    apply_vqtoken_compression,
    apply_visionzip_compression,
)
from streamvln_ext.modules.voxel_proxy import prune_token_bank, prune_tokens_per_frame
from streamvln_ext.modules.voxel_rgbd import prune_rgbd_features
from streamvln_ext.modules.voxel_spatial_pruning import apply_voxel_spatial_pruning_to_memory


class StreamVLNForCausalLMExt(StreamVLNForCausalLM):
    """Isolated extension model that keeps original StreamVLN files untouched."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.ext_flags: ExtFeatureFlags = load_feature_flags()
        self._last_memory_position_mask: Optional[torch.Tensor] = None
        self._last_video_token_compressor_stats: Optional[dict] = None
        self._last_voxel_spatial_pruning_stats: Optional[dict] = None
        self.training_aware_video_compressor: Optional[nn.Module] = None

        gate_hidden = int(getattr(config, "hidden_size", 0) or 0)
        if gate_hidden <= 0:
            gate_hidden = 1536
        self.mm_token_gate = nn.ModuleDict({
            "q_proj": nn.Linear(gate_hidden, 1, bias=True),
        })
        nn.init.zeros_(self.mm_token_gate["q_proj"].weight)
        nn.init.zeros_(self.mm_token_gate["q_proj"].bias)

        # Keep independent cache containers; the original implementation uses shared dict refs.
        self.curr_t: List[int] = []
        self.cache: List[dict] = []
        self._dynamic_memory_cache: List[Optional[torch.Tensor]] = []
        self.runtime_metrics = RuntimeTokenLatencyMetrics()
        self._init_training_aware_video_compressor()

    def _compute_learned_gate_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens is None or not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            return torch.zeros((0,), dtype=torch.float32, device=tokens.device if isinstance(tokens, torch.Tensor) else None)

        gate_layer = self.mm_token_gate["q_proj"]
        gate_dtype = gate_layer.weight.dtype
        logits = gate_layer(tokens.to(dtype=gate_dtype)).squeeze(-1)
        if self.ext_flags.mm_prune_gate_use_sigmoid:
            return torch.sigmoid(logits).to(dtype=torch.float32)
        return logits.to(dtype=torch.float32)

    def refresh_ext_flags(self):
        self.ext_flags = load_feature_flags()
        self._init_training_aware_video_compressor()

    def _init_training_aware_video_compressor(self):
        self.training_aware_video_compressor = None
        if not self.ext_flags.enable_training_aware_video_compressor:
            return

        compressor_type = str(self.ext_flags.training_aware_video_compressor_type).strip().lower()
        raw_hidden = int(getattr(self.config, "mm_hidden_size", 0) or getattr(self.config, "vision_hidden_size", 0) or 1152)
        llm_hidden = int(getattr(self.config, "hidden_size", 0) or 0)
        if llm_hidden <= 0:
            return

        if compressor_type == "longvu":
            self.training_aware_video_compressor = LongVUStreamVLNCompressor(
                self.ext_flags,
                raw_hidden_size=raw_hidden,
                llm_hidden_size=llm_hidden,
            )
        elif compressor_type == "llamavid":
            self.training_aware_video_compressor = LLaMAVIDStreamVLNCompressor(
                self.ext_flags,
                raw_hidden_size=raw_hidden,
                llm_hidden_size=llm_hidden,
            )
        elif compressor_type not in {"", "none"}:
            raise ValueError(f"Unsupported training-aware video compressor type: {compressor_type}")

    def _ensure_env_slots(self, env_idx: int):
        required = env_idx + 1
        while len(self.curr_t) < required:
            self.curr_t.append(0)
            self.cache.append({})

    @staticmethod
    def _count_feature_tokens(image_features, memory_features) -> Tuple[int, int]:
        visual_tokens = 0
        memory_tokens = 0

        for frame_features in image_features:
            if isinstance(frame_features, torch.Tensor) and frame_features.ndim == 3:
                visual_tokens += int(frame_features.shape[0] * frame_features.shape[1])

        for memory_bank in memory_features:
            if isinstance(memory_bank, torch.Tensor) and memory_bank.ndim == 3:
                memory_tokens += int(memory_bank.shape[0] * memory_bank.shape[1])

        return visual_tokens, memory_tokens

    @staticmethod
    def _count_feature_tokens_by_sample(image_features, memory_features) -> List[Tuple[int, int]]:
        counts: List[Tuple[int, int]] = []
        batch_size = max(len(image_features), len(memory_features))
        for batch_idx in range(batch_size):
            visual_tokens = 0
            memory_tokens = 0

            if batch_idx < len(image_features):
                frame_features = image_features[batch_idx]
                if isinstance(frame_features, torch.Tensor) and frame_features.ndim == 3:
                    visual_tokens = int(frame_features.shape[0] * frame_features.shape[1])

            if batch_idx < len(memory_features):
                memory_bank = memory_features[batch_idx]
                if isinstance(memory_bank, torch.Tensor) and memory_bank.ndim == 3:
                    memory_tokens = int(memory_bank.shape[0] * memory_bank.shape[1])

            counts.append((visual_tokens, memory_tokens))

        return counts

    def reset_runtime_metrics(self):
        self.runtime_metrics.reset()

    def consume_runtime_metrics_summary(self, reset: bool = True):
        summary = self.runtime_metrics.export_summary()
        if reset:
            self.runtime_metrics.reset()
        return summary

    def _estimate_tflops_per_step(self, total_tokens: int) -> float:
        if total_tokens <= 0:
            return 0.0

        hidden = int(getattr(self.config, "hidden_size", 0) or 0)
        layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        if hidden <= 0 or layers <= 0:
            return 0.0

        t = float(total_tokens)
        h = float(hidden)
        l = float(layers)
        flops = l * (8.0 * t * h * h + 4.0 * t * t * h)
        return float(flops / 1e12)

    def reset(self, env_num):
        self.curr_t = [0 for _ in range(env_num)]
        self.cache = [{} for _ in range(env_num)]
        self._dynamic_memory_cache = [None for _ in range(env_num)]
        self.reset_runtime_metrics()

    def reset_for_env(self, env_idx):
        self._ensure_env_slots(env_idx)
        self.curr_t[env_idx] = 0
        self.cache[env_idx] = {}
        while len(self._dynamic_memory_cache) <= env_idx:
            self._dynamic_memory_cache.append(None)
        self._dynamic_memory_cache[env_idx] = None

    @staticmethod
    def _time_start_value(time_ids, batch_idx: int) -> int:
        if time_ids is None:
            return 0
        cur = time_ids[batch_idx]
        if cur is None:
            return 0
        value = cur[0]
        if isinstance(value, torch.Tensor):
            value = value.item()
        return int(value)

    def _project_and_pool_raw_frames(self, raw_frames: torch.Tensor) -> torch.Tensor:
        projected = self.get_model().mm_projector(raw_frames)
        return self.get_2dPool(projected, 2)

    def _encode_rgbd_with_raw(self, images, depths, poses, intrinsics, time_ids=None, task_ids=None):
        batch_size, num_view, _, _, _ = images.shape
        raw_features = self.get_model().get_vision_tower()(images.flatten(0, 1))
        num_raw_tokens = int(raw_features.shape[1])
        raw_hidden = int(raw_features.shape[-1])
        raw_features = raw_features.view(batch_size, num_view, num_raw_tokens, raw_hidden)

        image_features = []
        memory_features = []
        raw_image_features = []
        raw_memory_features = []
        current_images = []

        if num_view != 1:
            for batch_idx in range(batch_size):
                start_idx = self._time_start_value(time_ids, batch_idx)
                if start_idx == 0:
                    cur_raw = raw_features[batch_idx]
                    image_features.append(self._project_and_pool_raw_frames(cur_raw))
                    memory_features.append(None)
                    raw_image_features.append(cur_raw)
                    raw_memory_features.append(None)
                    current_images.append(images[batch_idx])
                    continue

                history_idx = int(self.model.num_history)
                cur_raw = raw_features[batch_idx, history_idx:]
                his_raw = raw_features[batch_idx, :history_idx]
                image_features.append(self._project_and_pool_raw_frames(cur_raw))
                his_projected = self._project_and_pool_raw_frames(his_raw)
                memory_features.append(his_projected.flatten(0, 1).unsqueeze(0))
                raw_image_features.append(cur_raw)
                raw_memory_features.append(his_raw.flatten(0, 1).unsqueeze(0))
                current_images.append(images[batch_idx, history_idx:])
        else:
            for batch_idx in range(batch_size):
                cur_raw = raw_features[batch_idx]
                image_features.append(self._project_and_pool_raw_frames(cur_raw))
                memory_features.append(None)
                raw_image_features.append(cur_raw)
                raw_memory_features.append(None)
                current_images.append(images[batch_idx])

        meta = {
            "raw_tokens_per_frame": num_raw_tokens,
            "raw_hidden_size": raw_hidden,
            "current_images": current_images,
        }
        return image_features, memory_features, raw_image_features, raw_memory_features, meta

    def _apply_training_aware_video_compressor(
        self,
        image_features,
        memory_features,
        raw_image_features,
        raw_memory_features,
        raw_meta,
    ):
        compressor = self.training_aware_video_compressor
        if compressor is None:
            return image_features, memory_features, None

        compressor_type = str(self.ext_flags.training_aware_video_compressor_type).strip().lower()
        compressed_image_features = []
        compressed_memory_features = []
        image_meta = []
        memory_meta = []
        raw_tokens_per_frame = int(raw_meta.get("raw_tokens_per_frame", 0) or 0)
        current_images = raw_meta.get("current_images") or []

        for batch_idx, raw_frames in enumerate(raw_image_features):
            if raw_frames is None:
                compressed_image_features.append(image_features[batch_idx])
                image_meta.append({"skipped": True})
                compressed_memory_features.append(memory_features[batch_idx])
                memory_meta.append({"skipped": True})
                continue

            compressor.to(device=raw_frames.device, dtype=raw_frames.dtype)
            if compressor_type == "longvu":
                images_for_batch = current_images[batch_idx] if batch_idx < len(current_images) else None
                compressed_frames, meta = compressor.compress_frames(
                    raw_frames,
                    images=images_for_batch,
                    allow_frame_selection=False,
                )
            elif compressor_type == "llamavid":
                compressed_frames, meta = compressor.compress_frames(raw_frames)
            else:
                raise ValueError(f"Unsupported training-aware video compressor type: {compressor_type}")

            compressed_image_features.append(compressed_frames)
            image_meta.append(meta)

            memory_bank = memory_features[batch_idx] if batch_idx < len(memory_features) else None
            raw_memory_bank = raw_memory_features[batch_idx] if batch_idx < len(raw_memory_features) else None
            if self.ext_flags.training_aware_compress_memory and raw_memory_bank is not None:
                compressed_memory, mem_meta = compressor.compress_memory_bank(raw_memory_bank, tokens_per_frame=raw_tokens_per_frame)
                compressed_memory_features.append(compressed_memory)
                memory_meta.append(mem_meta)
            else:
                compressed_memory_features.append(memory_bank)
                memory_meta.append({"method": compressor_type, "memory_compression": False})

        stats = {
            "method": compressor_type,
            "raw_tokens_per_frame": raw_tokens_per_frame,
            "image_meta": image_meta,
            "memory_meta": memory_meta,
        }
        return compressed_image_features, compressed_memory_features, stats

    def encode_rgbd(self, images, depths, poses, intrinsics, time_ids=None, task_ids=None):
        precomputed_rgbd = getattr(self, "_precomputed_rgbd_batch", None)
        if isinstance(precomputed_rgbd, dict):
            image_features = precomputed_rgbd["image_features"]
            memory_features = precomputed_rgbd["memory_features"]
            raw_image_features = precomputed_rgbd.get("raw_image_features")
            raw_memory_features = precomputed_rgbd.get("raw_memory_features")
            raw_meta = precomputed_rgbd.get("raw_meta") or {}
        elif precomputed_rgbd is not None:
            image_features, memory_features = precomputed_rgbd
            raw_image_features = None
            raw_memory_features = None
            raw_meta = None
        elif self.ext_flags.enable_training_aware_video_compressor:
            image_features, memory_features, raw_image_features, raw_memory_features, raw_meta = self._encode_rgbd_with_raw(
                images,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids,
            )
        else:
            image_features, memory_features = super().encode_rgbd(images, depths, poses, intrinsics, time_ids, task_ids)
            raw_image_features = None
            raw_memory_features = None
            raw_meta = None

        before_counts = self._count_feature_tokens_by_sample(image_features, memory_features)
        self._last_video_token_compressor_stats = None
        self._last_voxel_spatial_pruning_stats = None

        if self.ext_flags.enable_training_aware_video_compressor:
            image_features, memory_features, compressor_stats = self._apply_training_aware_video_compressor(
                image_features=image_features,
                memory_features=memory_features,
                raw_image_features=raw_image_features,
                raw_memory_features=raw_memory_features,
                raw_meta=raw_meta or {},
            )
            self._last_video_token_compressor_stats = compressor_stats

        if self.ext_flags.enable_voxel_spatial_pruning:
            memory_features, voxel_spatial_stats = apply_voxel_spatial_pruning_to_memory(
                image_features=image_features,
                memory_features=memory_features,
                depths=depths,
                poses=poses,
                intrinsics=intrinsics,
                ext_flags=self.ext_flags,
            )
            self._last_voxel_spatial_pruning_stats = voxel_spatial_stats
            if self.ext_flags.enable_runtime_token_metrics:
                self.runtime_metrics.update_voxel_spatial_pruning_stats(voxel_spatial_stats)

        if self.ext_flags.enable_voxel_rgbd and depths is not None and poses is not None and intrinsics is not None:
            rgbd_pruned_image_features = []
            for batch_idx, frame_features in enumerate(image_features):
                depth_batch = depths[batch_idx] if isinstance(depths, torch.Tensor) and batch_idx < depths.shape[0] else None
                pose_batch = poses[batch_idx] if isinstance(poses, torch.Tensor) and batch_idx < poses.shape[0] else None
                intr_batch = intrinsics[batch_idx] if isinstance(intrinsics, torch.Tensor) and batch_idx < intrinsics.shape[0] else None
                rgbd_pruned_image_features.append(
                    prune_rgbd_features(
                        frame_features,
                        depth_batch,
                        pose_batch,
                        intr_batch,
                        keep_ratio=self.ext_flags.voxel_rgbd_keep_ratio,
                        min_tokens=self.ext_flags.voxel_rgbd_min_tokens,
                    )
                )
            image_features = rgbd_pruned_image_features

        if self.ext_flags.enable_voxel_proxy:
            keep_ratio = self.ext_flags.voxel_proxy_keep_ratio
            min_tokens = self.ext_flags.voxel_proxy_min_tokens

            pruned_image_features = []
            for frame_features in image_features:
                pruned_image_features.append(prune_tokens_per_frame(frame_features, keep_ratio=keep_ratio, min_tokens=min_tokens))
            image_features = pruned_image_features

            pruned_memory_features = []
            for memory_bank in memory_features:
                if memory_bank is None:
                    pruned_memory_features.append(None)
                    continue
                slot_features = []
                for slot in memory_bank:
                    slot_features.append(prune_token_bank(slot, keep_ratio=keep_ratio, min_tokens=min_tokens))
                pruned_memory_features.append(torch.stack(slot_features, dim=0))
            memory_features = pruned_memory_features

        if self.ext_flags.enable_token_selection:
            image_features = [
                [
                    select_tokens(
                        frame,
                        keep_ratio=self.ext_flags.token_selection_keep_ratio,
                        min_tokens=self.ext_flags.token_selection_min_tokens,
                    )
                    for frame in frame_features
                ]
                for frame_features in image_features
            ]
            memory_features = [
                select_memory_bank(
                    memory_bank,
                    keep_ratio=self.ext_flags.token_selection_keep_ratio,
                    min_tokens=self.ext_flags.token_selection_min_tokens,
                )
                for memory_bank in memory_features
            ]

        if self.ext_flags.enable_video_token_compressor:
            compressor_type = str(self.ext_flags.video_token_compressor_type).strip().lower()
            if compressor_type == "visionzip":
                image_features, memory_features, compressor_stats = apply_visionzip_compression(
                    image_features=image_features,
                    memory_features=memory_features,
                    ext_flags=self.ext_flags,
                )
                self._last_video_token_compressor_stats = compressor_stats
            elif compressor_type == "fastvid":
                image_features, memory_features, compressor_stats = apply_fastvid_compression(
                    image_features=image_features,
                    memory_features=memory_features,
                    ext_flags=self.ext_flags,
                )
                self._last_video_token_compressor_stats = compressor_stats
            elif compressor_type in {"dytok", "dytok_static"}:
                image_features, memory_features, compressor_stats = apply_dytok_static_compression(
                    image_features=image_features,
                    memory_features=memory_features,
                    ext_flags=self.ext_flags,
                )
                self._last_video_token_compressor_stats = compressor_stats
            elif compressor_type == "prunevid":
                image_features, memory_features, compressor_stats = apply_prunevid_compression(
                    image_features=image_features,
                    memory_features=memory_features,
                    ext_flags=self.ext_flags,
                )
                self._last_video_token_compressor_stats = compressor_stats
            elif compressor_type == "vqtoken":
                image_features, memory_features, compressor_stats = apply_vqtoken_compression(
                    image_features=image_features,
                    memory_features=memory_features,
                    ext_flags=self.ext_flags,
                )
                self._last_video_token_compressor_stats = compressor_stats
            elif compressor_type not in {"", "none"}:
                raise ValueError(f"Unsupported video token compressor type: {self.ext_flags.video_token_compressor_type}")

        if self.ext_flags.enable_hc_st_pruning:
            pruned_image_features = []
            for batch_idx, frame_features in enumerate(image_features):
                history_bank = memory_features[batch_idx] if batch_idx < len(memory_features) else None
                frame_list = []
                for frame in frame_features:
                    frame_list.append(
                        prune_tokens_history_conditioned(
                            frame,
                            history_bank,
                            keep_ratio=self.ext_flags.hc_st_keep_ratio,
                            min_tokens=self.ext_flags.hc_st_min_tokens,
                            history_window=self.ext_flags.hc_st_history_window,
                            recent_boost=self.ext_flags.hc_st_recent_boost,
                        )
                    )
                pruned_image_features.append(torch.stack(frame_list, dim=0))
            image_features = pruned_image_features

        if self.ext_flags.enable_tome_visual_merge:
            tome_image_features = []
            for frame_features in image_features:
                frame_list = []
                for frame in frame_features:
                    frame_list.append(
                        merge_visual_tokens_tome(
                            frame,
                            keep_ratio=self.ext_flags.tome_visual_keep_ratio,
                            min_tokens=max(1, self.ext_flags.tome_visual_min_tokens),
                            similarity_metric=self.ext_flags.tome_similarity_metric,
                        )
                    )
                tome_image_features.append(torch.stack(frame_list, dim=0))
            image_features = tome_image_features

            tome_memory_features = []
            for memory_bank in memory_features:
                if memory_bank is None:
                    tome_memory_features.append(None)
                    continue

                slot_features = []
                for slot in memory_bank:
                    slot_features.append(
                        merge_visual_tokens_tome(
                            slot,
                            keep_ratio=self.ext_flags.tome_visual_keep_ratio,
                            min_tokens=max(1, self.ext_flags.tome_visual_min_tokens),
                            similarity_metric=self.ext_flags.tome_similarity_metric,
                        )
                    )

                min_len = min(slot.shape[0] for slot in slot_features)
                aligned = [slot[:min_len] for slot in slot_features]
                tome_memory_features.append(torch.stack(aligned, dim=0))
            memory_features = tome_memory_features

        if self.ext_flags.enable_tuning_free_mm_pruning:
            tuned_image_features = []
            score_type = str(self.ext_flags.mm_prune_visual_score_type).strip().lower()
            for frame_features in image_features:
                frame_list = []
                for frame in frame_features:
                    learned_scores = None
                    if score_type == "learned_gate":
                        learned_scores = self._compute_learned_gate_scores(frame)
                    frame_list.append(
                        prune_visual_tokens_tuning_free(
                            frame,
                            keep_ratio=self.ext_flags.mm_prune_visual_keep_ratio,
                            min_tokens=max(1, self.ext_flags.hc_st_min_tokens),
                            score_type=self.ext_flags.mm_prune_visual_score_type,
                            hybrid_alpha=self.ext_flags.mm_prune_hybrid_alpha,
                            random_seed=self.ext_flags.mm_prune_random_seed,
                            external_scores=learned_scores,
                        )
                    )
                tuned_image_features.append(torch.stack(frame_list, dim=0))
            image_features = tuned_image_features

            tuned_memory_features = []
            for memory_bank in memory_features:
                if memory_bank is None:
                    tuned_memory_features.append(None)
                    continue

                slot_features = []
                for slot in memory_bank:
                    learned_scores = None
                    if score_type == "learned_gate":
                        learned_scores = self._compute_learned_gate_scores(slot)
                    slot_features.append(
                        prune_visual_tokens_tuning_free(
                            slot,
                            keep_ratio=self.ext_flags.mm_prune_visual_keep_ratio,
                            min_tokens=max(1, self.ext_flags.hc_st_min_tokens),
                            score_type=self.ext_flags.mm_prune_visual_score_type,
                            hybrid_alpha=self.ext_flags.mm_prune_hybrid_alpha,
                            random_seed=self.ext_flags.mm_prune_random_seed,
                            external_scores=learned_scores,
                        )
                    )

                min_len = min(slot.shape[0] for slot in slot_features)
                aligned = [slot[:min_len] for slot in slot_features]
                tuned_memory_features.append(torch.stack(aligned, dim=0))
            memory_features = tuned_memory_features

        if self.ext_flags.enable_dynamic_memory:
            while len(self._dynamic_memory_cache) < len(memory_features):
                self._dynamic_memory_cache.append(None)

            updated_memory_features = []
            for idx, memory_bank in enumerate(memory_features):
                updated_bank = update_memory_bank(
                    self._dynamic_memory_cache[idx],
                    memory_bank,
                    delta_threshold=self.ext_flags.dynamic_memory_delta_threshold,
                    blend=self.ext_flags.dynamic_memory_blend,
                )
                self._dynamic_memory_cache[idx] = updated_bank.detach() if isinstance(updated_bank, torch.Tensor) else None
                updated_memory_features.append(updated_bank)
            memory_features = updated_memory_features

        if self.ext_flags.enable_multiscale_memory:
            memory_features = [
                build_multiscale_memory(memory_bank, levels=self.ext_flags.multiscale_levels)
                for memory_bank in memory_features
            ]

        after_counts = self._count_feature_tokens_by_sample(image_features, memory_features)
        if self.ext_flags.enable_runtime_token_metrics:
            for before, after in zip(before_counts, after_counts):
                visual_before, memory_before = before
                visual_after, memory_after = after
                self.runtime_metrics.update_token_counts(
                    visual_before=visual_before,
                    visual_after=visual_after,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    total_before=visual_before + memory_before,
                    total_after=visual_after + memory_after,
                )

        return image_features, memory_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes,
        depths,
        poses,
        intrinsics,
        time_ids=None,
        task_ids=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            self._last_memory_position_mask = None
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features, memory_features = self.encode_rgbd(images, depths, poses, intrinsics, time_ids, task_ids)

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_memory_masks = []

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_memories = (cur_input_ids == MEMORY_TOKEN_INDEX).sum()
            num_specials = num_images + num_memories

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            memory_token_indices = torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + memory_token_indices)
            special_tokens = [cur_input_ids[index] for index in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i] + 1 : special_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i] + 1 : special_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_memory_mask = []
            total_before = 0
            total_after = 0

            cur_img_id = 0
            cur_mem_id = 0

            for i in range(num_specials + 1):
                text_embed = cur_input_embeds_no_im[i]
                text_labels = cur_labels_noim[i]
                total_before += int(text_embed.shape[0])

                if self.ext_flags.enable_tuning_free_mm_pruning and text_embed.shape[0] > 1:
                    text_embed, keep_idx = prune_text_embeds_tuning_free(
                        text_embed,
                        keep_ratio=self.ext_flags.mm_prune_text_keep_ratio,
                        min_tokens=1,
                        protected_mask=None,
                        score_type=self.ext_flags.mm_prune_text_score_type,
                        hybrid_alpha=self.ext_flags.mm_prune_hybrid_alpha,
                        random_seed=self.ext_flags.mm_prune_random_seed,
                    )
                    if keep_idx.numel() > 0:
                        text_labels = text_labels.index_select(0, keep_idx.to(text_labels.device))

                total_after += int(text_embed.shape[0])

                cur_new_input_embeds.append(text_embed)
                cur_new_labels.append(text_labels)
                cur_new_memory_mask.append(
                    torch.zeros(
                        (text_embed.shape[0],),
                        dtype=torch.bool,
                        device=cur_labels.device,
                    )
                )

                if i >= num_specials:
                    continue

                special_token = special_tokens[i]
                if special_token == IMAGE_TOKEN_INDEX:
                    cur_image_feature = image_features[batch_idx][cur_img_id]
                    cur_img_id += 1
                    total_before += int(cur_image_feature.shape[0])
                    total_after += int(cur_image_feature.shape[0])
                    cur_new_input_embeds.append(cur_image_feature)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_feature.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    cur_new_memory_mask.append(
                        torch.zeros(
                            (cur_image_feature.shape[0],),
                            dtype=torch.bool,
                            device=cur_labels.device,
                        )
                    )

                elif special_token == MEMORY_TOKEN_INDEX:
                    cur_memory_bank = memory_features[batch_idx]
                    if cur_memory_bank is None:
                        continue

                    cur_memory_feature = cur_memory_bank[cur_mem_id]
                    cur_mem_id += 1
                    total_before += int(cur_memory_feature.shape[0])
                    total_after += int(cur_memory_feature.shape[0])
                    cur_new_input_embeds.append(cur_memory_feature)
                    cur_new_labels.append(
                        torch.full(
                            (cur_memory_feature.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    cur_new_memory_mask.append(
                        torch.ones(
                            (cur_memory_feature.shape[0],),
                            dtype=torch.bool,
                            device=cur_labels.device,
                        )
                    )
                else:
                    raise NotImplementedError

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_memory_mask = torch.cat(cur_new_memory_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_memory_masks.append(cur_new_memory_mask)

            if self.ext_flags.enable_runtime_token_metrics:
                self.runtime_metrics.update_token_counts(
                    total_before=total_before,
                    total_after=total_after,
                )

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_memory_masks = [x[:tokenizer_model_max_length] for x in new_memory_masks]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        new_memory_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_memory_mask) in enumerate(
            zip(new_input_embeds, new_labels, new_memory_masks)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_memory_mask_padded[i, -cur_len:] = cur_new_memory_mask
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_memory_mask_padded[i, :cur_len] = cur_new_memory_mask
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        self._last_memory_position_mask = new_memory_mask_padded

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: torch.FloatTensor = None,
        depths: torch.FloatTensor = None,
        poses: torch.FloatTensor = None,
        intrinsics: torch.FloatTensor = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if not self.ext_flags.enable_memory_loss or labels is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                depths=depths,
                poses=poses,
                intrinsics=intrinsics,
                image_sizes=image_sizes,
                return_dict=return_dict,
                modalities=modalities,
                **kwargs,
            )

        time_ids = kwargs.get("time_ids", None)
        task_ids = kwargs.get("task_type", None)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids,
            )

        pseudo_labels = build_memory_pseudo_labels(labels, self._last_memory_position_mask, ignore_index=IGNORE_INDEX)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = Qwen2ForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        loss = compute_weighted_causal_loss(
            outputs.logits,
            pseudo_labels,
            self._last_memory_position_mask,
            memory_loss_weight=self.ext_flags.memory_loss_weight,
            ignore_index=IGNORE_INDEX,
        )

        if not return_dict:
            output_tuple = (outputs.logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output_tuple

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        cache_position = kwargs.get("cache_position", None)
        if cache_position is not None and hasattr(cache_position, "numel") and cache_position.numel() == 0:
            device = input_ids.device if input_ids is not None else self.device
            kwargs["cache_position"] = torch.tensor([0], device=device, dtype=torch.long)

        return StreamVLNForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        time_ids = kwargs.pop("time_ids", None)
        task_ids = kwargs.pop("task_type", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        env_id = kwargs.pop("env_id", None)
        if env_id is not None:
            self._ensure_env_slots(env_id)

            if self.curr_t[env_id] == 0:
                self.cache[env_id]["inputs_embeds"] = inputs_embeds
            else:
                self.cache[env_id]["inputs_embeds"] = torch.cat(
                    [self.cache[env_id]["inputs_embeds"], inputs_embeds],
                    dim=1,
                )
            self.curr_t[env_id] += 1

            if self.ext_flags.enable_sliding_kv and self.ext_flags.kv_window_tokens > 0:
                cache_embed = self.cache[env_id]["inputs_embeds"]
                if cache_embed.size(1) > self.ext_flags.kv_window_tokens:
                    self.cache[env_id]["inputs_embeds"] = cache_embed[:, -self.ext_flags.kv_window_tokens :, :].contiguous()

            inputs_embeds = self.cache[env_id]["inputs_embeds"]

        if self.ext_flags.enable_sliding_kv:
            if self.ext_flags.sliding_kv_use_past:
                if "past_key_values" in kwargs:
                    kwargs["past_key_values"] = truncate_past_key_values(
                        kwargs.get("past_key_values"),
                        self.ext_flags.kv_window_tokens,
                    )
            else:
                # Stable fallback path: keep sliding context window but avoid cache-shape mismatch.
                kwargs["past_key_values"] = None

        # Transformers can pass an empty cache_position tensor in single-process eval.
        # Fall back to fresh decoding to avoid indexing errors in upstream helper code.
        cache_position = kwargs.get("cache_position", None)
        if kwargs.get("past_key_values", None) is not None and cache_position is not None:
            try:
                if hasattr(cache_position, "numel") and cache_position.numel() == 0:
                    kwargs["past_key_values"] = None
            except Exception:
                kwargs["past_key_values"] = None

        step_start = time.perf_counter()
        outputs = Qwen2ForCausalLM.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        step_latency_ms = (time.perf_counter() - step_start) * 1000.0

        if self.ext_flags.enable_runtime_latency_metrics:
            self.runtime_metrics.update_latency_ms(step_latency_ms)

        if self.ext_flags.enable_runtime_tflops_estimate and isinstance(inputs_embeds, torch.Tensor):
            total_tokens = int(inputs_embeds.shape[1]) if inputs_embeds.ndim >= 2 else 0
            self.runtime_metrics.update_tflops_estimate(self._estimate_tflops_per_step(total_tokens))

        return outputs
