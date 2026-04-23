from typing import Dict, List


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

        return {
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
