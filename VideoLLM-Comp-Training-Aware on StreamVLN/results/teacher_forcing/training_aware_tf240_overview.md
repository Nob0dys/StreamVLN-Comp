# Training-Aware VideoLLM-Comp TF240 Overview

- master root: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240`
- baseline summary: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/20260507_fixed_tf240_aurora_promptfix/baseline/summary.json`
- baseline acc: `33.63%`
- baseline invalid: `48.03%`
- baseline tokens/step: `1736.78`
- final eval mode: `AuroraReplay-GT`, `next_token_logits`, `max_new_tokens=1`, `aurora_precompute_vision=1`

| rank | method | keep | actual_keep | acc | delta | invalid | tokens/step | token_red | fps | p50 | p95 | peak_reserved_mib | bucket |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `llamavid` | 0.5 | 0.4852 | 47.76% | +14.14pp | 29.59% | 771.60 | 55.57% | 9.76 | 50.28 | 56.80 | 70588 | `strong` |
| 2 | `longvu` | 0.3 | 0.2899 | 45.40% | +11.77pp | 32.06% | 492.20 | 71.66% | 10.62 | 43.32 | 48.14 | 18946 | `strong` |
| 3 | `llamavid` | 0.3 | 0.2959 | 43.38% | +9.76pp | 35.15% | 500.67 | 71.17% | 11.32 | 37.01 | 41.90 | 18944 | `strong` |
| 4 | `llamavid` | 0.7 | 0.7219 | 42.47% | +8.85pp | 36.48% | 1110.26 | 36.07% | 8.68 | 65.49 | 71.25 | 96034 | `strong` |
| 5 | `longvu` | 0.5 | 0.4793 | 41.22% | +7.59pp | 39.31% | 763.13 | 56.06% | 9.24 | 56.43 | 62.87 | 67536 | `strong` |
| 6 | `longvu` | 0.7 | 0.7160 | 39.97% | +6.34pp | 40.41% | 1101.79 | 36.56% | 8.27 | 71.49 | 77.09 | 96346 | `strong` |

## Acceptance

| candidate | episodes | actions | protocol | dataset | summary |
|---|---:|---:|---|---|---|
| `llamavid_keep_ratio_0_5_q32_grid9_r2r5k` | 240 | 16196 | `True` | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_llamavid_nexttoken_precompute/eval/llamavid_keep_ratio_0_5_q32_grid9_r2r5k_teacher_forcing_240/summary.json` |
| `longvu_keep_ratio_0_3_q49_d1_r2r5k` | 240 | 16196 | `True` | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_longvu_nexttoken_precompute/eval/longvu_keep_ratio_0_3_q49_d1_r2r5k_teacher_forcing_240/summary.json` |
| `llamavid_keep_ratio_0_3_q32_grid7_r2r5k` | 240 | 16196 | `True` | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_llamavid_nexttoken_precompute/eval/llamavid_keep_ratio_0_3_q32_grid7_r2r5k_teacher_forcing_240/summary.json` |
| `llamavid_keep_ratio_0_7_q32_grid11_r2r5k` | 240 | 16196 | `True` | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_llamavid_nexttoken_precompute/eval/llamavid_keep_ratio_0_7_q32_grid11_r2r5k_teacher_forcing_240/summary.json` |
| `longvu_keep_ratio_0_5_q81_d1_r2r5k` | 240 | 16196 | `True` | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_longvu_nexttoken_precompute/eval/longvu_keep_ratio_0_5_q81_d1_r2r5k_teacher_forcing_240/summary.json` |
| `longvu_keep_ratio_0_7_q121_d1_r2r5k` | 240 | 16196 | `True` | `True` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/tf240_eval_longvu_nexttoken_precompute/eval/longvu_keep_ratio_0_7_q121_d1_r2r5k_teacher_forcing_240/summary.json` |
