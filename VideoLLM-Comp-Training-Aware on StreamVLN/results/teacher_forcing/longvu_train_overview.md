# Training-Aware Keep-Ratio Train Overview

- suite_root: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/train_longvu`

| method | keep_ratio | actual_keep_ratio | actual_tokens_per_frame | candidate | status | train_loss | runtime_sec | steps_per_sec |
|---|---:|---:|---:|---|---|---:|---:|---:|
| `longvu` | 0.3 | 0.2899 | 49 | `longvu_keep_ratio_0_3_q49_d1_r2r5k` | `ok` | 0.2368 | 263.07 | 3.801 |
| `longvu` | 0.5 | 0.4793 | 81 | `longvu_keep_ratio_0_5_q81_d1_r2r5k` | `ok` | 0.2351 | 326.10 | 3.067 |
| `longvu` | 0.7 | 0.7160 | 121 | `longvu_keep_ratio_0_7_q121_d1_r2r5k` | `ok` | 0.2319 | 388.75 | 2.572 |
