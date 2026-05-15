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
| 1 | `llamavid` | 0.5 | 0.4852 | 47.76% | +14.14pp | 29.59% | 771.60 | 55.57% | 9.76 | 50.28 | 56.80 | 18157.4 | `strong` |
| 2 | `llamavid` | 0.3 | 0.2959 | 43.38% | +9.76pp | 35.15% | 500.67 | 71.17% | 11.32 | 37.01 | 41.90 | 18157.4 | `strong` |
| 3 | `llamavid` | 0.7 | 0.7219 | 42.47% | +8.85pp | 36.48% | 1110.26 | 36.07% | 8.68 | 65.49 | 71.25 | 18313.2 | `strong` |

## Acceptance

| candidate | episodes | actions | protocol_ok | summary |
|---|---:|---:|---|---|
| `llamavid_keep_ratio_0_5_q32_grid9_r2r5k` | 240 | 16196 | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_llamavid_nexttoken_precompute/eval/llamavid_keep_ratio_0_5_q32_grid9_r2r5k_teacher_forcing_240/summary.json` |
| `llamavid_keep_ratio_0_3_q32_grid7_r2r5k` | 240 | 16196 | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_llamavid_nexttoken_precompute/eval/llamavid_keep_ratio_0_3_q32_grid7_r2r5k_teacher_forcing_240/summary.json` |
| `llamavid_keep_ratio_0_7_q32_grid11_r2r5k` | 240 | 16196 | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_llamavid_nexttoken_precompute/eval/llamavid_keep_ratio_0_7_q32_grid11_r2r5k_teacher_forcing_240/summary.json` |
