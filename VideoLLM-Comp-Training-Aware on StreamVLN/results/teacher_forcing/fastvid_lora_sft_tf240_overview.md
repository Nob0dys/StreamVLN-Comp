# FastVid LoRA SFT TF240 Overview

- suite_root: `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_fastvid_sft_tf240_runs/20260509_133700`
- baseline acc: `33.63%`
- baseline tokens/step: `1736.78`

| kind | candidate | keep | acc | delta vs base | invalid | tokens/step | token reduction | fps | checkpoint | summary source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `FastVid training-free` | `fastvid_keep_ratio_0_3` | 0.3 | 31.96% | -1.66pp | 49.32% (7988) | 576.87 | 66.79% | 6.40 | `/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/20260507_fixed_tf240_aurora_promptfix_oom_rerun_lowmem/fastvid_keep_ratio_0_3/summary.json` |
| `FastVid-SFT-LoRA` | `fastvid_keep_ratio_0_3` | 0.3 | 47.37% | +13.74pp | 26.77% (4335) | 576.87 | 66.79% | 5.04 | `/home/ubuntu/model/StreamVLN_fastvid_lora_sft_r2r5k_20260509_133700/fastvid_keep_ratio_0_3` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_fastvid_sft_tf240_runs/20260509_133700/eval/fastvid_keep_ratio_0_3_teacher_forcing_240/summary.json` |
| `FastVid training-free` | `fastvid_keep_ratio_0_5` | 0.5 | 33.37% | -0.25pp | 47.32% (7664) | 907.06 | 47.77% | 5.75 | `/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/20260507_fixed_tf240_aurora_promptfix_oom_rerun_lowmem/fastvid_keep_ratio_0_5/summary.json` |
| `FastVid-SFT-LoRA` | `fastvid_keep_ratio_0_5` | 0.5 | 46.43% | +12.80pp | 26.54% (4299) | 907.06 | 47.77% | 4.65 | `/home/ubuntu/model/StreamVLN_fastvid_lora_sft_r2r5k_20260509_133700/fastvid_keep_ratio_0_5` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_fastvid_sft_tf240_runs/20260509_133700/eval/fastvid_keep_ratio_0_5_teacher_forcing_240/summary.json` |
| `FastVid training-free` | `fastvid_keep_ratio_0_7` | 0.7 | 33.61% | -0.02pp | 48.02% (7778) | 1237.26 | 28.76% | 5.36 | `/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_full_eval_tf_runs/20260507_fixed_tf240_aurora_promptfix_oom_rerun_lowmem/fastvid_keep_ratio_0_7/summary.json` |
| `FastVid-SFT-LoRA` | `fastvid_keep_ratio_0_7` | 0.7 | 36.06% | +2.43pp | 42.23% (6839) | 1237.26 | 28.76% | 4.38 | `/home/ubuntu/model/StreamVLN_fastvid_lora_sft_r2r5k_20260509_133700/fastvid_keep_ratio_0_7` | `/home/ubuntu/project/StreamVLN/experiments_ext/videollm_comp_fastvid_sft_tf240_runs/20260509_133700/eval/fastvid_keep_ratio_0_7_teacher_forcing_240/summary.json` |
