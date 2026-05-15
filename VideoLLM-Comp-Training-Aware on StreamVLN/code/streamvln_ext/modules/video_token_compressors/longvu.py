import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamvln_ext.config.feature_flags import ExtFeatureFlags


class _LongVUCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, queries: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(queries)
        kv = self.kv_norm(kv_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.ffn(queries)
        return queries


class LongVUStreamVLNCompressor(nn.Module):
    """StreamVLN-adapted LongVU compressor.

    StreamVLN has one image feature block per <image> token, so the default
    adaptation keeps the current-frame count unchanged and applies LongVU-style
    multi-encoder token sampling per frame. The optional frame selector is kept
    for memory/hard analysis but is disabled for current frames by default.
    """

    def __init__(self, ext_flags: ExtFeatureFlags, raw_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.raw_hidden_size = int(raw_hidden_size)
        self.llm_hidden_size = int(llm_hidden_size)
        self.dino_tower = str(ext_flags.longvu_dino_tower)
        self.window_size = int(ext_flags.longvu_window_size)
        self.threshold = float(ext_flags.longvu_threshold)
        self.max_frames = int(ext_flags.longvu_max_frames)
        self.min_frames = int(ext_flags.longvu_min_frames)
        self.dino_interp_size = int(ext_flags.longvu_dino_interp_size)
        self.vision_hidden_size = int(ext_flags.longvu_vision_hidden_size)
        self.query_num = int(ext_flags.longvu_query_num)
        self.use_dino = bool(ext_flags.longvu_use_dino)
        self.select_current_frames = bool(ext_flags.longvu_select_current_frames)

        num_heads = max(1, int(ext_flags.longvu_num_heads))
        if self.vision_hidden_size % num_heads != 0:
            num_heads = 1

        self.siglip_projector = nn.Sequential(
            nn.Linear(self.raw_hidden_size, self.vision_hidden_size),
            nn.GELU(),
            nn.Linear(self.vision_hidden_size, self.vision_hidden_size),
            nn.LayerNorm(self.vision_hidden_size),
        )
        # The encoder weights are loaded lazily, but the projector must exist
        # before the optimizer is built.
        self.dino_projector: Optional[nn.Module] = None
        self.dino_encoder: Optional[nn.Module] = None
        self.dino_hidden_size: Optional[int] = None
        if self.use_dino:
            try:
                from transformers import Dinov2Config

                self.dino_hidden_size = int(Dinov2Config.from_pretrained(self.dino_tower).hidden_size)
                self.dino_projector = nn.Sequential(
                    nn.Linear(self.dino_hidden_size, self.vision_hidden_size),
                    nn.GELU(),
                    nn.Linear(self.vision_hidden_size, self.vision_hidden_size),
                    nn.LayerNorm(self.vision_hidden_size),
                )
            except Exception:
                self.dino_hidden_size = None

        self.vision_query = nn.Parameter(torch.randn(self.query_num, self.vision_hidden_size) * (self.vision_hidden_size ** -0.5))
        self.layers = nn.ModuleList(
            [_LongVUCrossAttentionLayer(self.vision_hidden_size, num_heads=num_heads) for _ in range(max(1, int(ext_flags.longvu_connector_depth)))]
        )
        self.output_projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
        )

    def frozen_parameter_prefixes(self) -> List[str]:
        return ["dino_encoder"]

    def _ensure_dino(self, device: torch.device, dtype: torch.dtype):
        if not self.use_dino or self.dino_encoder is not None:
            return

        from transformers import Dinov2Model

        encoder = Dinov2Model.from_pretrained(self.dino_tower)
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.to(device=device, dtype=dtype)
        self.dino_encoder = encoder
        self.dino_hidden_size = int(encoder.config.hidden_size)
        if self.dino_projector is None:
            self.dino_projector = nn.Sequential(
                nn.Linear(self.dino_hidden_size, self.vision_hidden_size),
                nn.GELU(),
                nn.Linear(self.vision_hidden_size, self.vision_hidden_size),
                nn.LayerNorm(self.vision_hidden_size),
            )
        self.dino_projector.to(device=device, dtype=dtype)

    @staticmethod
    def _siglip_to_imagenet(images: torch.Tensor) -> torch.Tensor:
        images_01 = images * 0.5 + 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images_01 - mean) / std

    @staticmethod
    def _interpolate_tokens(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if features.shape[1] == target_tokens:
            return features

        source_tokens = int(features.shape[1])
        source_side = int(math.sqrt(source_tokens))
        target_side = int(math.sqrt(target_tokens))
        if source_side * source_side == source_tokens and target_side * target_side == target_tokens:
            x = features.view(features.shape[0], source_side, source_side, features.shape[-1])
            x = x.permute(0, 3, 1, 2).contiguous()
            x = F.interpolate(x.float(), size=(target_side, target_side), mode="bilinear", align_corners=False).to(features.dtype)
            return x.permute(0, 2, 3, 1).flatten(1, 2).contiguous()

        return F.adaptive_avg_pool1d(features.transpose(1, 2), target_tokens).transpose(1, 2).contiguous()

    def _encode_dino(self, images: Optional[torch.Tensor], target_tokens: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if images is None or not self.use_dino:
            return None

        self._ensure_dino(images.device, dtype)
        if self.dino_encoder is None:
            return None

        with torch.no_grad():
            dino_images = self._siglip_to_imagenet(images)
            try:
                outputs = self.dino_encoder(dino_images, interpolate_pos_encoding=True)
            except TypeError:
                outputs = self.dino_encoder(dino_images)
            dino_tokens = outputs.last_hidden_state[:, 1:].to(dtype=dtype)

        return self._interpolate_tokens(dino_tokens, target_tokens)

    def _select_frames(self, frame_features: torch.Tensor) -> torch.Tensor:
        num_frames = int(frame_features.shape[0])
        if num_frames <= 1:
            return torch.arange(num_frames, device=frame_features.device)

        all_indices = []
        window = max(1, self.window_size)
        for start in range(0, num_frames, window):
            end = min(start + window, num_frames)
            segment = frame_features[start:end].flatten(1, 2)
            segment = F.normalize(segment.float(), dim=1)
            sim = torch.mean(segment @ segment.transpose(0, 1), dim=1)
            sim[(end - start) // 2] = 0.0
            local = torch.where(sim < self.threshold)[0]
            if local.numel() == 0:
                local = torch.tensor([(end - start) // 2], device=frame_features.device, dtype=torch.long)
            all_indices.append(local + start)

        indices = torch.cat(all_indices, dim=0)
        if indices.numel() > self.max_frames:
            keep = torch.linspace(0, indices.numel() - 1, steps=self.max_frames, device=indices.device).round().long()
            indices = indices.index_select(0, keep)
        if indices.numel() < self.min_frames:
            fallback = torch.linspace(0, num_frames - 1, steps=min(num_frames, self.min_frames), device=indices.device).round().long()
            indices = torch.unique(torch.cat([indices, fallback], dim=0), sorted=True)
        return indices.long()

    def compress_frames(
        self,
        raw_frames: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        allow_frame_selection: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if raw_frames is None or raw_frames.ndim != 3:
            return raw_frames, {"method": "longvu", "skipped": True}

        original_frames, tokens_per_frame, _ = raw_frames.shape
        selected = torch.arange(original_frames, device=raw_frames.device, dtype=torch.long)
        if allow_frame_selection:
            selected = self._select_frames(raw_frames)

        raw_selected = raw_frames.index_select(0, selected)
        image_selected = images.index_select(0, selected) if images is not None and allow_frame_selection else images
        if image_selected is not None and image_selected.shape[0] != raw_selected.shape[0]:
            image_selected = None

        siglip_tokens = self.siglip_projector(raw_selected)
        kv_tokens = [siglip_tokens]

        dino_tokens = self._encode_dino(image_selected, target_tokens=tokens_per_frame, dtype=raw_frames.dtype)
        if dino_tokens is not None and self.dino_projector is not None:
            kv_tokens.append(self.dino_projector(dino_tokens))

        kv = torch.cat(kv_tokens, dim=1)
        queries = self.vision_query.to(device=raw_frames.device, dtype=raw_frames.dtype).unsqueeze(0).expand(raw_selected.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, kv)

        output = self.output_projector(queries).contiguous()
        meta = {
            "method": "longvu",
            "original_frames": int(original_frames),
            "selected_frames": int(raw_selected.shape[0]),
            "selected_indices": [int(x) for x in selected.detach().cpu().tolist()],
            "input_tokens_per_frame": int(tokens_per_frame),
            "output_tokens_per_frame": int(output.shape[1]),
            "query_num": int(self.query_num),
            "used_dino": bool(dino_tokens is not None),
        }
        return output, meta

    def compress_memory_bank(self, raw_memory_bank: Optional[torch.Tensor], tokens_per_frame: int) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
        if raw_memory_bank is None or not isinstance(raw_memory_bank, torch.Tensor) or raw_memory_bank.ndim != 3:
            return raw_memory_bank, {"method": "longvu", "memory_slots": 0}

        compressed_slots: List[torch.Tensor] = []
        slot_meta: List[Dict[str, object]] = []
        for slot in raw_memory_bank:
            if tokens_per_frame > 0 and slot.shape[0] % tokens_per_frame == 0:
                frames = slot.view(slot.shape[0] // tokens_per_frame, tokens_per_frame, slot.shape[-1])
            else:
                frames = slot.unsqueeze(0)
            compressed, meta = self.compress_frames(frames, images=None, allow_frame_selection=False)
            compressed_slots.append(compressed.flatten(0, 1).contiguous())
            slot_meta.append(meta)

        min_len = min(slot.shape[0] for slot in compressed_slots)
        aligned = [slot[:min_len] for slot in compressed_slots]
        return torch.stack(aligned, dim=0), {
            "method": "longvu",
            "memory_slots": int(raw_memory_bank.shape[0]),
            "slot_meta": slot_meta,
        }
