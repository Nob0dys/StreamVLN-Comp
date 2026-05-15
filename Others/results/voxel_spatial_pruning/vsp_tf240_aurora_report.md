# Voxel-Based Spatial Pruning on StreamVLN TF240

- Generated: 2026-05-08 19:29:37 CST
- Suite: `/home/ubuntu/project/StreamVLN/experiments_ext/vsp_tf240_aurora_runs/20260508_vsp_initial_baseline_compatible`
- Dataset: `/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_20260421_230201/teacher_forcing_240`
- Protocol: `AuroraReplay-GT`
- Model: `/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- Base model: `/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- Aurora mode: step_mode=`next_token_logits`, decode_max_new_tokens=`1`, vision_batch_size=`16`, precompute_vision=`True`
- Baseline acc: `0.336256`
- Baseline invalid: `0.480304`
- Baseline tokens/step: `1736.783836`
- Max episodes for this run: `0`
- GPUs: `0,1`; one eval process per GPU

## Current Best

- Best accuracy: `size_s0_10_k4_t0_00_d10` acc=33.48%, tokens=1600.4, reduction=7.85%.
- Best near-baseline tradeoff: `size_s0_25_k4_t0_00_d10` acc=33.36%, tokens=1420.9, reduction=18.19%.

## Results

| Stage | Config | size | K | threshold | max_depth | status | Episodes | Acc | Δacc | Invalid | Tokens/step | Reduction | Mem before | Mem after | VSP keep | Skip reasons | FPS | p95 ms |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sanity | `sanity_s0_25_k4_t0_00_d10` | 0.25 | 4 | 0.00 | 10.0 | done | 10 | 33.53% | -0.10 | 48.30% | 1406.4 | 19.02% | 731.9 | 565.4 | 77.25% | `{"missing_memory_bank": 10}` | 6.83 | 106.7 |
| sanity | `sanity_stress_s1_00_k8_t0_20_d10` | 1.00 | 8 | 0.20 | 10.0 | done | 10 | 35.75% | 2.12 | 44.61% | 593.7 | 65.82% | 731.9 | 159.0 | 21.73% | `{"missing_memory_bank": 10}` | 9.89 | 64.4 |
| size_sweep | `size_s0_10_k4_t0_00_d10` | 0.10 | 4 | 0.00 | 10.0 | done | 240 | 33.48% | -0.15 | 48.47% | 1600.4 | 7.85% | 731.7 | 663.5 | 90.68% | `{"missing_memory_bank": 240}` | 6.48 | 111.8 |
| size_sweep | `size_s0_25_k4_t0_00_d10` | 0.25 | 4 | 0.00 | 10.0 | done | 240 | 33.36% | -0.27 | 48.67% | 1420.9 | 18.19% | 731.7 | 573.8 | 78.42% | `{"missing_memory_bank": 240}` | 6.82 | 109.5 |
| size_sweep | `size_s0_50_k4_t0_00_d10` | 0.50 | 4 | 0.00 | 10.0 | done | 240 | 32.74% | -0.88 | 48.38% | 1094.8 | 36.96% | 731.7 | 410.7 | 56.13% | `{"missing_memory_bank": 240}` | 7.92 | 87.6 |
| size_sweep | `size_s1_00_k4_t0_00_d10` | 1.00 | 4 | 0.00 | 10.0 | done | 240 | 32.80% | -0.82 | 48.02% | 680.7 | 60.81% | 731.7 | 203.7 | 27.83% | `{"missing_memory_bank": 240}` | 9.53 | 66.0 |

## Notes

- `Reduction` is computed against the documented TF240 Aurora baseline tokens/step.
- This suite uses the same Aurora replay decoding口径 as the documented baseline: next-token logits with one decoded token and vision precompute enabled.
- `Mem before/after` comes from runtime feature-token metrics before and after VSP.
- `VSP keep` is the aggregate VSP memory keep ratio from VSP-specific diagnostics.
- Visual tokens should stay nearly unchanged because this VSP path only prunes historical memory tokens.

## Output Files

- CSV: `/home/ubuntu/project/StreamVLN/experiments_ext/vsp_tf240_aurora_runs/20260508_vsp_initial_baseline_compatible/vsp_tf240_aurora_results.csv`
