# Autoregressive Overview

Baseline: `overall_action_acc=0.6178`, `avg_total_tokens_per_step=346.4022`, `fps=8.2924`, `latency_ms_p50=130.1628`, `latency_ms_p95=265.1825`.

| method | setting | overall_action_acc | avg_total_tokens_per_step | token_reduction_ratio_vs_baseline | fps | latency_ms_p50 | latency_ms_p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| VisionZip | keep_ratio=0.3, dominant=24, contextual=6 | 0.5637 | 71.5577 | 0.7934 | 8.8970 | 121.0319 | 202.0550 |
| VisionZip | keep_ratio=0.5, dominant=36, contextual=12 | 0.5948 | 102.8375 | 0.7031 | 8.8908 | 122.9757 | 208.6705 |
| VisionZip | keep_ratio=0.7, dominant=48, contextual=20 | 0.6172 | 137.2619 | 0.6037 | 8.8994 | 124.1004 | 215.9799 |
| PruneVid | tau=0.7, cluster=0.30, temporal=0.50 | 0.6059 | 117.9163 | 0.6596 | 8.9603 | 123.7171 | 212.1687 |
| PruneVid | tau=0.8, cluster=0.50, temporal=0.25 | 0.6098 | 185.3302 | 0.4650 | 8.6891 | 126.7378 | 230.6429 |
| PruneVid | tau=0.9, cluster=0.70, temporal=0.10 | 0.6069 | 248.9876 | 0.2812 | 8.7488 | 128.3335 | 244.7573 |
| DyToK static | upper=0.3, min=0.2, base=visionzip | 0.6028 | 121.4961 | 0.6493 | 8.8611 | 124.2410 | 216.9615 |
| DyToK static | upper=0.5, min=0.3, base=visionzip | 0.6142 | 187.9710 | 0.4574 | 8.8726 | 126.5705 | 232.9431 |
| DyToK static | upper=0.7, min=0.5, base=visionzip | 0.6116 | 254.3027 | 0.2659 | 8.6997 | 128.3327 | 248.5843 |
| FastVID | keep_ratio=0.3 | 0.6136 | 121.0457 | 0.6506 | 8.9162 | 123.8540 | 215.3458 |
| FastVID | keep_ratio=0.5 | 0.6140 | 187.8421 | 0.4577 | 8.6722 | 126.9156 | 231.7565 |
| FastVID | keep_ratio=0.7 | 0.6133 | 252.2505 | 0.2718 | 8.7195 | 128.4641 | 247.6133 |
| VQToken | keep_ratio=0.3, clusters=59 | 0.5990 | 121.8397 | 0.6483 | 8.4805 | 136.6803 | 328.3257 |
| VQToken | keep_ratio=0.5, clusters=98 | 0.6149 | 187.7950 | 0.4579 | 8.2271 | 146.9684 | 412.4889 |
| VQToken | keep_ratio=0.7, clusters=137 | 0.6192 | 254.3558 | 0.2657 | 7.7916 | 157.2306 | 498.7116 |

Main takeaways:

- `VQToken keep_ratio=0.7` has the highest action accuracy among these migrated methods.
- `VisionZip keep_ratio=0.7` is the strongest balanced option when accuracy, token reduction, and latency are considered together.
- `FastVID keep_ratio=0.3/0.5` is stable and close to baseline while reducing tokens heavily.
- `DyToK static` is useful as a budget-control direction.
