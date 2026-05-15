# VideoLLM-Comp Training-Aware on StreamVLN

This folder contains the StreamVLN training-aware and training-adapted compression experiments summarized in `REPORT.md`.

## Methods

- `LLaMAVID`: trainable query-based compressor.
- `LongVU`: trainable query-based compressor with optional DINO features.
- `FastVid-SFT-LoRA`: training-free FastVid compression adapted with LoRA SFT.

## Contents

- `REPORT.md`: full experiment report copied from the cleaned StreamVLN workspace.
- `code/streamvln_ext/`: StreamVLN extension code used by the training-aware and FastVid-SFT-LoRA runs.
- `scripts/`: fixed-subset preparation, training, TF240 eval, and FastVid LoRA SFT scripts.
- `results/teacher_forcing/`: lightweight overviews, validation summary, and archived summary JSON files.
- `assets/plots/`: exported figures referenced by the report.
