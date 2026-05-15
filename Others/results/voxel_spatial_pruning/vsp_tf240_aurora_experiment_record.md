# StreamVLN TF240 Voxel-Based Spatial Pruning 实验记录

- 记录时间：2026-05-08 19:30 CST
- 实验目录：`/home/ubuntu/project/StreamVLN/experiments_ext/vsp_tf240_aurora_runs/20260508_vsp_initial_baseline_compatible`
- 数据集：`/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_20260421_230201/teacher_forcing_240`
- 协议：`AuroraReplay-GT`
- checkpoint：`/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- conda 环境：`streamvln`
- GPU 策略：2 张 96GB GPU，每张 GPU 1 个 eval process

## Baseline

| episodes | actions | acc | invalid | tokens/step |
|---:|---:|---:|---:|---:|
| 240 | 16196 | 33.63% | 48.03% | 1736.8 |

baseline 来自 `training-free-videollm-comp-on-streamvln.md`。本轮 VSP 正式实验使用与 baseline 对齐的 Aurora replay 口径：

- `aurora_step_mode=next_token_logits`
- `aurora_decode_max_new_tokens=1`
- `aurora_vision_batch_size=16`
- `aurora_precompute_vision=True`
- `base_model_path` 与 `model_path` 相同

备注：最开始有一轮 `20260508_vsp_initial` 结果使用了不一致的 Aurora 生成口径，导致 `invalid=100%`，该轮 accuracy 不纳入正式结论；其中 token 剪枝诊断仅作为排查参考。

## 本轮实验范围

本轮先做 baseline-compatible 的 size sweep，并把 `voxel_spatial_frame_threshold` 固定为 `0`，目的是先隔离 `voxel_spatial_size` 的影响，避免整帧丢弃阈值混入第一轮结论。

固定项：

- `enable_voxel_spatial_pruning=True`
- `enable_offline_saved_geometry=True`
- `enable_voxel_rgbd=False`
- `enable_video_token_compressor=False`
- `voxel_spatial_stride_k=4`
- `voxel_spatial_frame_threshold=0`
- `voxel_spatial_min_depth=0.05`
- `voxel_spatial_max_depth=10`

## 参数说明（对应论文 Algorithm 1）

本实验中的三个核心参数对应 StreamVLN 论文（arXiv:2507.05240）第 3.3 节及 Algorithm 1 中的实现细节，具体含义如下：

| 实验参数 | 代码字段 | Algorithm 1 对应含义 | 作用说明 |
|---|---|---|---|
| **size** | `voxel_spatial_size` | 3D voxel grid 的体素边长（voxel size） | 将世界坐标系下的点云离散化为 uniform voxels 时的空间粒度。size 越大，单个 voxel 覆盖的物理空间越大，不同帧投影到同一 voxel 的概率越高，剪枝越激进。 |
| **K** | `voxel_spatial_stride_k` | 时间分桶的跨度（temporal stride / bucket size） | 在 `build_voxel_spatial_pruning_mask` 中实现为 `bucket = t // stride_k`。每 K 帧为一个时间桶，同一桶内若多个 token 落入同一 voxel，仅保留最新帧的 token。K 越大，跨帧合并的时间范围越宽。 |
| **t** | `voxel_spatial_frame_threshold` | 整帧丢弃阈值（frame-level pruning threshold） | 若某帧经 voxel pruning 后保留的 token 数低于 `t × (grid_h × grid_w)`，则将该帧全部丢弃。t=0 表示关闭整帧丢弃，仅做逐 token 的空间剪枝。 |

> 注：论文 Algorithm 1 的核心逻辑为：先将 2D image patches 通过深度反投影到共享 3D 空间，再按 `size` 离散化为 voxels；随后在每个长度为 `K` 的时间桶内，对落入同一 voxel 的多个 token 仅保留最新一帧的 token；最后若启用 `t>0`，则剔除保留 token 比例过低的整帧。

## 正式结果

| config | acc | delta vs baseline | invalid | tokens/step | token reduction | memory before | memory after | VSP keep | fps | latency mean | latency p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline (no VSP)` | 33.63% | 0.00 pp | 48.03% | 1736.8 | 0.00% | 731.7 | 731.7 | 100.00% | 9.23 | 96.9 ms | 105.7 ms |
| `size=0.10,K=4,t=0` | 33.48% | -0.15 pp | 48.47% | 1600.4 | 7.85% | 731.7 | 663.5 | 90.68% | 6.48 | 102.7 ms | 111.8 ms |
| `size=0.25,K=4,t=0` | 33.36% | -0.27 pp | 48.67% | 1420.9 | 18.19% | 731.7 | 573.8 | 78.42% | 6.82 | 94.9 ms | 109.5 ms |
| `size=0.50,K=4,t=0` | 32.74% | -0.88 pp | 48.38% | 1094.8 | 36.96% | 731.7 | 410.7 | 56.13% | 7.92 | 74.5 ms | 87.6 ms |
| `size=1.00,K=4,t=0` | 32.80% | -0.82 pp | 48.02% | 680.7 | 60.81% | 731.7 | 203.7 | 27.83% | 9.53 | 53.4 ms | 66.0 ms |

VSP 诊断：

- 所有正式配置的 visual tokens 均为 `98.0 -> 98.0`，符合当前 VSP 只剪 historical memory tokens 的预期。
- 所有正式配置的 skip reason 均为 `{"missing_memory_bank": 240}`，对应每个 episode 第一步没有历史 memory bank，属于预期行为。
- 未观察到 `missing_geometry`、`no_valid_depth` 等 geometry/depth 相关失败原因。

## 结论

1. VSP 在 TF240 AuroraReplay-GT 上工作正常，memory tokens 随 voxel size 增大单调下降，invalid rate 没有明显恶化。
2. 精度优先建议从 `size=0.25,K=4,t=0` 开始：只掉 `0.27 pp`，tokens/step 降低 `18.19%`。
3. 如果目标是吞吐或长上下文压力，`size=0.50,K=4,t=0` 是值得继续看的点：tokens/step 降低 `36.96%`，但 acc 下降接近 `0.9 pp`。
4. `size=1.00,K=4,t=0` 的 token reduction 很强，达到 `60.81%`，invalid 也稳定，但 acc 下降约 `0.82 pp`，更适合作为 aggressive/latency-oriented 配置。

## 下一步建议

优先细扫两个区域：

1. 精度优先区域：`size=0.25`，扫 `K=1,2,4,8`，先保持 `threshold=0`。
2. 性价比区域：`size=0.50`，扫 `K=1,2,4,8`，确认 stride 是否能追回部分 accuracy。
3. 在最佳 `size,K` 上再扫 `frame_threshold=0.02,0.05,0.10`。建议暂缓 `0.20`，因为整帧丢弃会显著改变行为分布。
4. `max_depth=5,20` 放到最后做噪声过滤验证。

## 输出文件

- 自动汇总报告：`/home/ubuntu/project/StreamVLN/experiments_ext/vsp_tf240_aurora_runs/20260508_vsp_initial_baseline_compatible/vsp_tf240_aurora_report.md`
- CSV：`/home/ubuntu/project/StreamVLN/experiments_ext/vsp_tf240_aurora_runs/20260508_vsp_initial_baseline_compatible/vsp_tf240_aurora_results.csv`
- 每个配置目录中保留 `summary.json`、`flags.json`、`command.txt`、`eval.log`
