import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamvln_ext.config.feature_flags import ExtFeatureFlags


def _grid_representative_indices(input_side: int, output_side: int, device: torch.device) -> torch.Tensor:
    if output_side <= 1:
        center = (input_side // 2) * input_side + (input_side // 2)
        return torch.tensor([center], device=device, dtype=torch.long)

    ys = torch.linspace(0, input_side - 1, steps=output_side, device=device).round().long()
    xs = torch.linspace(0, input_side - 1, steps=output_side, device=device).round().long()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return (yy * input_side + xx).flatten().long()


def _pool_grid_tokens(tokens: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_tokens, hidden = tokens.shape
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), grid_size * grid_size).transpose(1, 2)
        keep = torch.linspace(0, num_tokens - 1, steps=grid_size * grid_size, device=tokens.device).round().long()
        return pooled, keep

    x = tokens.view(num_frames, side, side, hidden).permute(0, 3, 1, 2).contiguous()
    pooled = F.adaptive_avg_pool2d(x.float(), output_size=(grid_size, grid_size)).to(tokens.dtype)
    pooled = pooled.permute(0, 2, 3, 1).flatten(1, 2).contiguous()
    keep = _grid_representative_indices(side, grid_size, tokens.device)
    return pooled, keep


class _CrossAttentionBlock(nn.Module):
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

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.query_norm(queries), self.kv_norm(context), self.kv_norm(context), need_weights=False)
        queries = queries + attn_out
        queries = queries + self.ffn(queries)
        return queries


class LLaMAVIDStreamVLNCompressor(nn.Module):
    """StreamVLN-adapted LLaMA-VID context/content token compressor.

    The upstream LLaMA-VID path uses a BERT/QFormer module. This implementation
    keeps the same context/content contract while avoiding text-template
    assumptions that do not fit StreamVLN's per-step image-token layout.
    """

    def __init__(self, ext_flags: ExtFeatureFlags, raw_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.raw_hidden_size = int(raw_hidden_size)
        self.llm_hidden_size = int(llm_hidden_size)
        self.num_query = int(ext_flags.llamavid_num_query)
        self.compress_type = str(ext_flags.llamavid_compress_type)
        self.q_hidden = int(ext_flags.llamavid_qformer_hidden_size)

        num_heads = max(1, int(ext_flags.llamavid_num_heads))
        if self.q_hidden % num_heads != 0:
            num_heads = 1

        self.query_tokens = nn.Parameter(torch.randn(self.num_query, self.q_hidden) * (self.q_hidden ** -0.5))
        self.visual_in = nn.Sequential(
            nn.LayerNorm(self.raw_hidden_size),
            nn.Linear(self.raw_hidden_size, self.q_hidden),
        )
        self.layers = nn.ModuleList(
            [_CrossAttentionBlock(self.q_hidden, num_heads=num_heads) for _ in range(max(1, int(ext_flags.llamavid_qformer_depth)))]
        )

        self.query_to_raw = nn.Linear(self.q_hidden, self.raw_hidden_size)
        self.context_key = nn.Linear(self.raw_hidden_size, self.raw_hidden_size)
        self.context_value = nn.Linear(self.raw_hidden_size, self.llm_hidden_size)
        self.content_projector = nn.Sequential(
            nn.Linear(self.raw_hidden_size, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
        )

    def _content_tokens(self, raw_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        compress_type = self.compress_type.strip().lower()
        if compress_type == "none":
            keep = torch.arange(raw_frames.shape[1], device=raw_frames.device, dtype=torch.long)
            return self.content_projector(raw_frames), keep
        if compress_type.startswith("grid:"):
            grid_size = max(1, int(compress_type.split("grid:", 1)[1]))
            pooled, keep = _pool_grid_tokens(raw_frames, grid_size)
            return self.content_projector(pooled), keep

        # Default LLaMA-VID stage-2 style content path.
        content = raw_frames.mean(dim=1, keepdim=True)
        keep = _grid_representative_indices(int(math.sqrt(raw_frames.shape[1])), 1, raw_frames.device)
        if keep.numel() == 0:
            keep = torch.zeros((1,), device=raw_frames.device, dtype=torch.long)
        return self.content_projector(content), keep[:1]

    def compress_frames(self, raw_frames: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        if raw_frames is None or raw_frames.ndim != 3:
            return raw_frames, {"method": "llamavid", "skipped": True}

        num_frames, num_tokens, _ = raw_frames.shape
        context = self.visual_in(raw_frames)
        queries = self.query_tokens.to(dtype=context.dtype, device=context.device).unsqueeze(0).expand(num_frames, -1, -1)

        for layer in self.layers:
            queries = layer(queries, context)

        text_q = self.query_to_raw(queries)
        keys = self.context_key(raw_frames)
        attn = torch.matmul(text_q, keys.transpose(1, 2)) / math.sqrt(max(1, self.raw_hidden_size))
        context_raw = torch.matmul(attn.softmax(dim=-1), raw_frames).mean(dim=1, keepdim=True)
        context_token = self.context_value(context_raw)

        content_tokens, content_keep = self._content_tokens(raw_frames)
        output = torch.cat([context_token, content_tokens], dim=1).contiguous()

        meta = {
            "method": "llamavid",
            "num_frames": int(num_frames),
            "input_tokens_per_frame": int(num_tokens),
            "context_tokens_per_frame": 1,
            "content_tokens_per_frame": int(content_tokens.shape[1]),
            "output_tokens_per_frame": int(output.shape[1]),
            "compress_type": self.compress_type,
            "content_keep_indices": [int(x) for x in content_keep.detach().cpu().tolist()],
        }
        return output, meta

    def compress_memory_bank(self, raw_memory_bank: Optional[torch.Tensor], tokens_per_frame: int) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
        if raw_memory_bank is None or not isinstance(raw_memory_bank, torch.Tensor) or raw_memory_bank.ndim != 3:
            return raw_memory_bank, {"method": "llamavid", "memory_slots": 0}

        compressed_slots: List[torch.Tensor] = []
        slot_meta: List[Dict[str, object]] = []
        for slot in raw_memory_bank:
            if tokens_per_frame > 0 and slot.shape[0] % tokens_per_frame == 0:
                frames = slot.view(slot.shape[0] // tokens_per_frame, tokens_per_frame, slot.shape[-1])
            else:
                frames = slot.unsqueeze(0)
            compressed, meta = self.compress_frames(frames)
            compressed_slots.append(compressed.flatten(0, 1).contiguous())
            slot_meta.append(meta)

        min_len = min(slot.shape[0] for slot in compressed_slots)
        aligned = [slot[:min_len] for slot in compressed_slots]
        return torch.stack(aligned, dim=0), {
            "method": "llamavid",
            "memory_slots": int(raw_memory_bank.shape[0]),
            "slot_meta": slot_meta,
        }
