# Low-Memory OOM Rerun Overview

- Suite root: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/20260507_fixed_tf240_aurora_promptfix_oom_rerun_lowmem`
- Original suite: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/20260507_fixed_tf240_aurora_promptfix`
- Low-memory settings: no vision precompute, `aurora_vision_batch_size=1`, `aurora_batch_size=1`, one process per GPU.

| Config | Status | Episodes | Actions | Acc | Token reduction vs original baseline | FPS |
|---|---|---:|---:|---:|---:|---:|
| dytok_static_visionzip_keep_ratio_0_7 | valid-full | 240/240 | 16196 | 33.66% | 28.7% | 5.33 |
| fastvid_keep_ratio_0_3 | valid-full | 240/240 | 16196 | 31.96% | 66.8% | 6.40 |
| fastvid_keep_ratio_0_5 | valid-full | 240/240 | 16196 | 33.37% | 47.8% | 5.75 |
| fastvid_keep_ratio_0_7 | valid-full | 240/240 | 16196 | 33.61% | 28.8% | 5.36 |
| vqtoken_keep_ratio_0_3 | valid-full | 240/240 | 16196 | 31.61% | 66.8% | 3.78 |
| vqtoken_keep_ratio_0_5 | valid-full | 240/240 | 16196 | 32.84% | 47.8% | 2.90 |
