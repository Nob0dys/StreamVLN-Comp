import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


@dataclass
class ExtFeatureFlags:
    enable_training_aware_video_compressor: bool = False
    training_aware_video_compressor_type: str = "none"
    training_aware_compress_memory: bool = True
    longvu_dino_tower: str = "/home/ubuntu/model/dinov2-base"
    longvu_window_size: int = 16
    longvu_threshold: float = 0.83
    longvu_max_frames: int = 400
    longvu_min_frames: int = 1
    longvu_dino_interp_size: int = 0
    longvu_vision_hidden_size: int = 1024
    longvu_query_num: int = 144
    longvu_connector_depth: int = 3
    longvu_num_heads: int = 8
    longvu_use_dino: bool = True
    longvu_select_current_frames: bool = False
    llamavid_bert_model_name: str = "/home/ubuntu/model/bert-base-uncased"
    llamavid_num_query: int = 32
    llamavid_compress_type: str = "mean"
    llamavid_qformer_hidden_size: int = 768
    llamavid_qformer_depth: int = 2
    llamavid_num_heads: int = 8
    enable_video_token_compressor: bool = False
    video_token_compressor_type: str = "none"
    video_token_target_keep_ratio: float = 1.0
    visionzip_dominant_num: int = 36
    visionzip_contextual_num: int = 12
    visionzip_score_type: str = "attn_proxy"
    visionzip_attn_layer: int = -2
    visionzip_hidden_state_layer: int = -2
    prunevid_tau: float = 0.8
    prunevid_cluster_ratio: float = 0.5
    prunevid_temporal_ratio: float = 0.25
    prunevid_k: int = 7
    prunevid_min_tokens_for_cluster: int = 14
    fastvid_retention_ratio: float = 0.5
    fastvid_dyseg_c: int = 8
    fastvid_dyseg_tau: float = 0.9
    fastvid_stprune_d: float = 0.4
    fastvid_dtm_p: int = 4
    fastvid_dtm_beta: float = 0.6
    fastvid_score_type: str = "attn_proxy"
    vqtoken_num_clusters: int = 32
    vqtoken_adaptive: bool = False
    vqtoken_max_clusters: int = 64
    vqtoken_adaptive_method: str = "silhouette"
    vqtoken_use_cross_attention: bool = False
    dytok_static_base_compressor: str = "visionzip"
    dytok_static_upper_limit_ratio: float = 0.5
    dytok_static_min_ratio: float = 0.3
    enable_sliding_kv: bool = False
    kv_window_tokens: int = 2048
    sliding_kv_use_past: bool = False
    enable_memory_loss: bool = False
    memory_loss_weight: float = 0.2
    enable_voxel_proxy: bool = False
    voxel_proxy_keep_ratio: float = 0.7
    voxel_proxy_min_tokens: int = 64
    enable_dynamic_memory: bool = False
    dynamic_memory_delta_threshold: float = 0.04
    dynamic_memory_blend: float = 0.5
    enable_token_selection: bool = False
    token_selection_keep_ratio: float = 0.7
    token_selection_min_tokens: int = 64
    enable_multiscale_memory: bool = False
    multiscale_levels: int = 3
    enable_voxel_rgbd: bool = False
    voxel_rgbd_keep_ratio: float = 0.7
    voxel_rgbd_min_tokens: int = 64
    enable_voxel_spatial_pruning: bool = False
    voxel_spatial_size: float = 0.25
    voxel_spatial_stride_k: int = 4
    voxel_spatial_frame_threshold: float = 0.05
    voxel_spatial_min_depth: float = 0.05
    voxel_spatial_max_depth: float = 10.0
    voxel_spatial_offline_geom_mode: str = "dummy"
    voxel_spatial_offline_unit_depth_m: float = 2.0
    voxel_spatial_offline_hfov_deg: float = 75.17817894
    enable_offline_saved_geometry: bool = True
    enable_hc_st_pruning: bool = False
    hc_st_keep_ratio: float = 0.8
    hc_st_min_tokens: int = 64
    hc_st_history_window: int = 8
    hc_st_recent_boost: float = 1.2
    enable_tuning_free_mm_pruning: bool = False
    mm_prune_visual_keep_ratio: float = 0.8
    mm_prune_text_keep_ratio: float = 0.9
    mm_prune_visual_score_type: str = "l2"
    mm_prune_text_score_type: str = "l2"
    mm_prune_hybrid_alpha: float = 0.5
    mm_prune_random_seed: int = 42
    mm_prune_gate_use_sigmoid: bool = True
    mm_prune_keep_special_tokens: bool = True
    enable_tome_visual_merge: bool = False
    tome_visual_keep_ratio: float = 0.8
    tome_visual_min_tokens: int = 64
    tome_similarity_metric: str = "cosine"
    enable_runtime_token_metrics: bool = True
    enable_runtime_latency_metrics: bool = True
    enable_runtime_tflops_estimate: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtFeatureFlags":
        flags = cls()
        for key, value in data.items():
            if not hasattr(flags, key):
                continue
            if key.startswith("enable_"):
                setattr(flags, key, _to_bool(value))
            elif key in {
                "mm_prune_keep_special_tokens",
                "mm_prune_gate_use_sigmoid",
                "vqtoken_adaptive",
                "vqtoken_use_cross_attention",
                "training_aware_compress_memory",
                "longvu_use_dino",
                "longvu_select_current_frames",
            }:
                setattr(flags, key, _to_bool(value))
            elif key in {
                "longvu_window_size",
                "longvu_max_frames",
                "longvu_min_frames",
                "longvu_dino_interp_size",
                "longvu_vision_hidden_size",
                "longvu_query_num",
                "longvu_connector_depth",
                "longvu_num_heads",
                "llamavid_num_query",
                "llamavid_qformer_hidden_size",
                "llamavid_qformer_depth",
                "llamavid_num_heads",
                "visionzip_dominant_num",
                "visionzip_contextual_num",
                "visionzip_attn_layer",
                "visionzip_hidden_state_layer",
                "prunevid_k",
                "prunevid_min_tokens_for_cluster",
                "fastvid_dyseg_c",
                "fastvid_dtm_p",
                "vqtoken_num_clusters",
                "vqtoken_max_clusters",
                "kv_window_tokens",
                "voxel_proxy_min_tokens",
                "token_selection_min_tokens",
                "multiscale_levels",
                "voxel_rgbd_min_tokens",
                "voxel_spatial_stride_k",
                "hc_st_min_tokens",
                "hc_st_history_window",
                "mm_prune_random_seed",
                "tome_visual_min_tokens",
            }:
                setattr(flags, key, int(value))
            elif key in {
                "longvu_threshold",
                "video_token_target_keep_ratio",
                "prunevid_tau",
                "prunevid_cluster_ratio",
                "prunevid_temporal_ratio",
                "fastvid_retention_ratio",
                "fastvid_dyseg_tau",
                "fastvid_stprune_d",
                "fastvid_dtm_beta",
                "dytok_static_upper_limit_ratio",
                "dytok_static_min_ratio",
                "memory_loss_weight",
                "voxel_proxy_keep_ratio",
                "dynamic_memory_delta_threshold",
                "dynamic_memory_blend",
                "token_selection_keep_ratio",
                "voxel_rgbd_keep_ratio",
                "voxel_spatial_size",
                "voxel_spatial_frame_threshold",
                "voxel_spatial_min_depth",
                "voxel_spatial_max_depth",
                "voxel_spatial_offline_unit_depth_m",
                "voxel_spatial_offline_hfov_deg",
                "hc_st_keep_ratio",
                "hc_st_recent_boost",
                "mm_prune_visual_keep_ratio",
                "mm_prune_text_keep_ratio",
                "mm_prune_hybrid_alpha",
                "tome_visual_keep_ratio",
            }:
                setattr(flags, key, float(value))
            else:
                setattr(flags, key, value)
        return flags

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_json_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_env(env_name: str) -> Dict[str, Any]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {env_name}: {exc}") from exc


def load_feature_flags(extra_overrides: Optional[Dict[str, Any]] = None) -> ExtFeatureFlags:
    merged: Dict[str, Any] = {}

    file_path = os.environ.get("STREAMVLN_EXT_FLAGS_FILE", "").strip()
    merged.update(_load_json_file(file_path))
    merged.update(_load_json_env("STREAMVLN_EXT_FLAGS"))

    if extra_overrides:
        merged.update(extra_overrides)

    return ExtFeatureFlags.from_dict(merged)
