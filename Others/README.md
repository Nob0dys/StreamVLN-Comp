# Others: f2_warmup_eval_target_hw7

This folder contains the internal StreamVLN pruning line centered on `f2_warmup_eval_target_hw7`.

## Method

`f2_warmup_eval_target_hw7` combines:

- HC-ST visual and memory pruning
- tuning-free multimodal pruning
- text token pruning
- a lightweight recovery checkpoint trained with a higher warmup keep ratio and evaluated at the target keep ratio

For the main reported setting:

- history window: `7`
- target keep ratio: `0.72` in the original internal line
- fixed-eval sweep: `keep_ratio={0.3, 0.5, 0.7}`
- text keep ratio during eval: `min(0.95, keep_ratio + 0.10)`

## Code

Core pruning/eval files:

- `code/streamvln_ext/modules/history_conditioned_pruning.py`
- `code/streamvln_ext/modules/tuning_free_mm_pruning.py`
- `code/streamvln_ext/modules/token_selection.py`
- `code/streamvln_ext/modules/runtime_metrics.py`
- `code/streamvln_ext/model/stream_video_vln_ext.py`
- `code/streamvln_ext/config/feature_flags.py`
- `code/streamvln_ext/entrypoints/eval_offline_action_acc_ext.py`

Training-related patch files:

- `code/training/train_r2r_offline_lora_sdpa.sh`
- `code/training/streamvln/args.py`
- `code/training/streamvln/streamvln_train.py`
- `code/training/streamvln/dataset/vln_action_dataset.py`
- `code/training/llava/train/llava_trainer.py`

## Results

The fixed-subset AR-only overview is in:

- `results/f2_ar_fixed_eval_overview.md`
- `results/f2_ar_fixed_eval_overview.csv`

Raw summaries are under `results/autoregressive/`.

## Run Script

`scripts/run_f2_warmup_eval_target_hw7_ar_only.sh` evaluates the existing warmup recovery checkpoint across `keep_ratio={0.3, 0.5, 0.7}`.
