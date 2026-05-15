# VideoLLM-Comp Training-Free on StreamVLN

This folder contains the migrated VideoLLM-Comp training-free variants on StreamVLN:

- `VisionZip`
- `PruneVid`
- `DyToK (static)`
- `FastVID`
- `VQToken`, with `use_cross_attention=false`

## Code

Core integration files:

- `code/streamvln_ext/model/stream_video_vln_ext.py`
- `code/streamvln_ext/config/feature_flags.py`
- `code/streamvln_ext/modules/video_token_compressors/`
- `code/streamvln_ext/modules/runtime_metrics.py`
- `code/streamvln_ext/entrypoints/eval_offline_action_acc_ext.py`
- `code/streamvln_ext/entrypoints/common.py`

`stream_video_vln_ext.py` wires the compressor into `encode_rgbd(...)`, so both current visual features and memory features can be compressed before action generation.

## Results

The overview table is in `results/autoregressive_overview.md`.

The full teacher-forcing report is in `REPORT.md`, with archived TF240 summaries under `results/teacher_forcing/`.

Raw summaries are under `results/autoregressive/`, one subfolder per method setting. Each subfolder keeps:

- `summary.json`
- `timing.txt`

The included plot is:

- `assets/videollm_comp_autoregressive_overall_action_acc.png`

## Run Script

`scripts/run_videollm_comp_training_free_ar_grid.sh` is a compact AR-only grid script for the five migrated methods. It expects the original StreamVLN project layout and uses `STREAMVLN_EXT_FLAGS` to choose the compressor configuration.

The TF240 fixed-subset evaluation scripts are:

- `scripts/run_90_phase0_prepare_fixed_eval_subsets.sh`
- `scripts/run_130_training_free_full_tf_eval.py`
