# Training-Aware VideoLLM-Comp TF240 Overview

- eval protocol: `AuroraReplay-GT`
- eval subset: `/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_20260421_230201/teacher_forcing_240`
- train subset signature: `9a7056bbf4f395ac`
- eval subset signature: `b76cf3ac8c8f8afa`
- baseline acc: `33.63%`
- baseline tokens/step: `1736.78`
- baseline invalid rate: `48.03%`

| rank | method | keep | actual_keep | acc | delta_vs_base | invalid | tokens/step | token_red | fps | p50 | p95 | peak_alloc_mib | bucket |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `longvu` | 0.3 | 0.2899 | 45.40% | +11.77pp | 32.06% | 492.20 | 71.66% | 10.62 | 43.32 | 48.14 | 18331.2 | `strong` |
| 2 | `longvu` | 0.5 | 0.4793 | 41.22% | +7.59pp | 39.31% | 763.13 | 56.06% | 9.24 | 56.43 | 62.87 | 18331.2 | `strong` |
| 3 | `longvu` | 0.7 | 0.7160 | 39.97% | +6.34pp | 40.41% | 1101.79 | 36.56% | 8.27 | 71.49 | 77.09 | 18476.6 | `strong` |

## Acceptance

| candidate | episodes | actions | protocol_ok | summary |
|---|---:|---:|---|---|
| `longvu_keep_ratio_0_3_q49_d1_r2r5k` | 240 | 16196 | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_longvu_nexttoken_precompute/eval/longvu_keep_ratio_0_3_q49_d1_r2r5k_teacher_forcing_240/summary.json` |
| `longvu_keep_ratio_0_5_q81_d1_r2r5k` | 240 | 16196 | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_longvu_nexttoken_precompute/eval/longvu_keep_ratio_0_5_q81_d1_r2r5k_teacher_forcing_240/summary.json` |
| `longvu_keep_ratio_0_7_q121_d1_r2r5k` | 240 | 16196 | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_longvu_nexttoken_precompute/eval/longvu_keep_ratio_0_7_q121_d1_r2r5k_teacher_forcing_240/summary.json` |
