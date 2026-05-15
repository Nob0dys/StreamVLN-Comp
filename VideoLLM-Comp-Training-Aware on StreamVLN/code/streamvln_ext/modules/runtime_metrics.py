from collections import Counter
from typing import Dict, List, Optional


class RuntimeTokenLatencyMetrics:
    """Lightweight container for runtime token/latency/TFLOPs statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.visual_tokens_before: List[int] = []
        self.visual_tokens_after: List[int] = []
        self.memory_tokens_before: List[int] = []
        self.memory_tokens_after: List[int] = []
        self.total_tokens_before: List[int] = []
        self.total_tokens_after: List[int] = []
        self.step_latency_ms: List[float] = []
        self.tflops_estimates: List[float] = []
        self.voxel_spatial_pruning_stats: List[Dict[str, object]] = []

    def update_token_counts(
        self,
        visual_before: int = 0,
        visual_after: int = 0,
        memory_before: int = 0,
        memory_after: int = 0,
        total_before: int = 0,
        total_after: int = 0,
    ):
        self.visual_tokens_before.append(int(visual_before))
        self.visual_tokens_after.append(int(visual_after))
        self.memory_tokens_before.append(int(memory_before))
        self.memory_tokens_after.append(int(memory_after))
        self.total_tokens_before.append(int(total_before))
        self.total_tokens_after.append(int(total_after))

    def update_latency_ms(self, latency_ms: float):
        self.step_latency_ms.append(float(latency_ms))

    def update_tflops_estimate(self, tflops: float):
        self.tflops_estimates.append(float(tflops))

    def update_voxel_spatial_pruning_stats(self, stats: Optional[Dict[str, object]]):
        if isinstance(stats, dict):
            self.voxel_spatial_pruning_stats.append(stats)

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def export_summary(self) -> Dict[str, float]:
        avg_total_before = self._mean(self.total_tokens_before)
        avg_total_after = self._mean(self.total_tokens_after)

        reduction = 0.0
        if avg_total_before > 0:
            reduction = max(0.0, (avg_total_before - avg_total_after) / avg_total_before)

        summary = {
            "avg_visual_tokens_before": self._mean(self.visual_tokens_before),
            "avg_visual_tokens_after": self._mean(self.visual_tokens_after),
            "avg_memory_tokens_before": self._mean(self.memory_tokens_before),
            "avg_memory_tokens_after": self._mean(self.memory_tokens_after),
            "avg_total_tokens_before": avg_total_before,
            "avg_total_tokens_after": avg_total_after,
            "token_reduction_ratio": reduction,
            "latency_ms_mean": self._mean(self.step_latency_ms),
            "approx_tflops_per_step": self._mean(self.tflops_estimates),
            "num_runtime_steps": float(len(self.step_latency_ms)),
        }
        summary.update(self._export_voxel_spatial_summary())
        return summary

    def _export_voxel_spatial_summary(self) -> Dict[str, object]:
        stats_list = self.voxel_spatial_pruning_stats
        if not stats_list:
            return {
                "voxel_spatial_num_calls": 0,
                "voxel_spatial_num_batch_items": 0,
                "voxel_spatial_num_slot_items": 0,
                "voxel_spatial_num_effective_slots": 0,
                "voxel_spatial_num_skipped_slots": 0,
                "voxel_spatial_skip_reasons": {},
            }

        reason_counts: Counter = Counter()
        keep_ratios: List[float] = []
        valid_token_counts: List[int] = []
        kept_token_counts: List[int] = []
        memory_before_sum = 0
        memory_after_sum = 0
        num_batch_items = 0
        num_batch_skipped = 0
        num_batch_pruned = 0
        num_slot_items = 0
        num_effective_slots = 0
        num_skipped_slots = 0

        for stats in stats_list:
            for batch_meta in stats.get("batch_meta", []) or []:
                if not isinstance(batch_meta, dict):
                    continue
                num_batch_items += 1
                if bool(batch_meta.get("skipped", False)):
                    num_batch_skipped += 1
                    reason_counts[str(batch_meta.get("reason", "unknown"))] += 1
                    continue

                before = int(batch_meta.get("memory_tokens_before", 0) or 0)
                after = int(batch_meta.get("memory_tokens_after", 0) or 0)
                memory_before_sum += before
                memory_after_sum += after
                if before > 0 and after < before:
                    num_batch_pruned += 1

                for slot_meta in batch_meta.get("slot_meta", []) or []:
                    if not isinstance(slot_meta, dict):
                        continue
                    num_slot_items += 1
                    if bool(slot_meta.get("skipped", False)):
                        num_skipped_slots += 1
                        reason_counts[str(slot_meta.get("reason", "unknown"))] += 1
                        continue
                    num_effective_slots += 1
                    keep_ratios.append(float(slot_meta.get("keep_ratio", 0.0) or 0.0))
                    valid_token_counts.append(int(slot_meta.get("valid_tokens", 0) or 0))
                    kept_token_counts.append(int(slot_meta.get("kept_tokens", 0) or 0))

        memory_keep_ratio = 0.0
        if memory_before_sum > 0:
            memory_keep_ratio = float(memory_after_sum / memory_before_sum)

        return {
            "voxel_spatial_num_calls": int(len(stats_list)),
            "voxel_spatial_num_batch_items": int(num_batch_items),
            "voxel_spatial_num_skipped_batches": int(num_batch_skipped),
            "voxel_spatial_num_pruned_batches": int(num_batch_pruned),
            "voxel_spatial_num_slot_items": int(num_slot_items),
            "voxel_spatial_num_effective_slots": int(num_effective_slots),
            "voxel_spatial_num_skipped_slots": int(num_skipped_slots),
            "voxel_spatial_skip_reasons": dict(sorted(reason_counts.items())),
            "voxel_spatial_memory_tokens_before_sum": int(memory_before_sum),
            "voxel_spatial_memory_tokens_after_sum": int(memory_after_sum),
            "voxel_spatial_memory_keep_ratio": memory_keep_ratio,
            "voxel_spatial_avg_slot_keep_ratio": self._mean(keep_ratios),
            "voxel_spatial_avg_valid_tokens_per_slot": self._mean(valid_token_counts),
            "voxel_spatial_avg_kept_tokens_per_slot": self._mean(kept_token_counts),
        }
