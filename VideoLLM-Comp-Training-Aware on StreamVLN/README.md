# VideoLLM-Comp Training-Aware on StreamVLN

This folder is intentionally reserved.

The current migrated StreamVLN upload contains the practical training-free VideoLLM-Comp variants:

- `VisionZip`
- `PruneVid`
- `DyToK (static)`
- `FastVID`
- `VQToken` with `use_cross_attention=false`

Heavier VideoLLM-Comp-style variants such as `LongVU`, `LLaMAVID`, and cross-attention-style `VQToken` are not included here because they were not part of the completed StreamVLN migration in the current report. They require additional model-side integration or learned components before a clean upload.

The internal recovery method `f2_warmup_eval_target_hw7` is not a VideoLLM-Comp method, so it is placed under `Others`.
