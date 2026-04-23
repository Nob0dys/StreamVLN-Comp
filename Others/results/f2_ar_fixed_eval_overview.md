# f2_warmup_eval_target_hw7 AR Fixed-Eval Overview

Baseline: `overall_action_acc=0.6178`, `stop_acc=0.1750`, `avg_total_tokens_per_step=346.4022`, `fps=8.2924`, `latency_ms_p50=130.1628`, `latency_ms_p95=265.1825`.

| method | setting | overall_action_acc | stop_acc | avg_total_tokens_per_step | token_reduction_ratio_vs_baseline | fps | latency_ms_p50 | latency_ms_p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| f2_warmup_eval_target_hw7 | keep_ratio=0.3 | 0.5979 | 0.0506 | 110.9428 | 0.6797 | 9.3307 | 82.7140 | 165.0251 |
| f2_warmup_eval_target_hw7 | keep_ratio=0.5 | 0.6020 | 0.0758 | 130.9031 | 0.6221 | 7.5816 | 94.9315 | 222.6512 |
| f2_warmup_eval_target_hw7 | keep_ratio=0.7 | 0.6183 | 0.0875 | 201.1454 | 0.4193 | 9.2399 | 86.1458 | 200.7274 |

Main takeaways:

- `keep_ratio=0.7` is the best fixed-eval point for this internal method.
- It slightly exceeds baseline action accuracy while reducing token count by about `41.93%`.
- Its latency is better than baseline at both p50 and p95.
- STOP accuracy remains the main unresolved risk.
