# Training-Aware Keep-Ratio Train Overview

- suite_root: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_training_aware_tf240_runs/20260508_204619_ta_tf240/train_llamavid`

| method | keep_ratio | actual_keep_ratio | actual_tokens_per_frame | candidate | status | train_loss | runtime_sec | steps_per_sec |
|---|---:|---:|---:|---|---|---:|---:|---:|
| `llamavid` | 0.3 | 0.2959 | 50 | `llamavid_keep_ratio_0_3_q32_grid7_r2r5k` | `reused` | 0.2201 | 1007.06 | 0.993 |
| `llamavid` | 0.5 | 0.4852 | 82 | `llamavid_keep_ratio_0_5_q32_grid9_r2r5k` | `ok` | 0.2142 | 1249.42 | 0.800 |
| `llamavid` | 0.7 | 0.7219 | 122 | `llamavid_keep_ratio_0_7_q32_grid11_r2r5k` | `ok` | 0.2075 | 1486.60 | 0.673 |
