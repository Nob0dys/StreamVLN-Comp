# Training-Free VideoLLM Compression on StreamVLN

- Generated: 2026-05-07 19:39:48 Asia/Shanghai
- Archived original summaries: `results/teacher_forcing/summaries/original`
- Archived low-memory rerun summaries: `results/teacher_forcing/summaries/lowmem_rerun`
- Dataset: `AuroraReplay-GT` fixed teacher-forcing subset, 240 episodes; raw fixed-subset files were removed during repository cleanup.
- Protocol: `AuroraReplay-GT` fixed teacher-forcing subset, 240 episodes
- Model: `/home/ubuntu/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- Metric: offline action micro accuracy over four actions; invalid model outputs are counted as wrong, not converted to STOP.

## Executive Summary

- Official configs summarized: 16. Full valid: **16/16**; partial: 0; failed/OOM: 0 after low-memory rerun.
- Baseline action accuracy is **33.63%** with **1736.8** total tokens/step.
- Best compressed accuracy: **DyTok-static 0.7** at **33.66%** (+0.04pp vs baseline), token reduction **28.7%**.
- Strongest compression while valid-full: **VisionZip 0.3** with **80.9%** token reduction and **32.43%** accuracy.
- Closest-to-baseline compressed points: FastVid 0.7: 33.61% (-0.02pp), 28.8% token reduction; DyTok-static 0.7: 33.66% (+0.04pp), 28.7% token reduction; FastVid 0.5: 33.37% (-0.25pp), 47.8% token reduction; DyTok-static 0.5: 33.30% (-0.32pp), 47.8% token reduction; PruneVid 0.5: 33.27% (-0.35pp), 48.9% token reduction.
- Source policy: original valid runs are kept; the six original OOM/partial configs are replaced by low-memory reruns. Accuracy/token metrics are comparable; FPS for rerun rows used lower-memory settings and should be interpreted with that caveat.

## Training-Free 方法机制与启发

| 方法 | 代码中具体是什么 | 作用在哪些模块上 | 从本次 StreamVLN 结果得到的启发 |
|---|---|---|---|
| VisionZip | `algorithm/LLaVA-NeXT/llava/model/multimodal_compressor/visionzip/compressor.py` 中的逐帧视觉 token 选择与合并器。它先按视觉编码器注意力保留高分 `dominant` token，再从剩余 token 中采样少量 `contextual` 锚点，并用 hidden-state 相似度把其它 token 聚合到这些锚点上。 | 在 VideoLLM-Comp 中由 `llava_arch.py` 在 `encode_images` 和 `mm_projector` 之后、视觉 embedding 插入 LLM prompt 之前调用；同时读取视觉编码器的 attention 和 hidden states 做打分。在本 StreamVLN 迁移里，同一思路作用到当前 `image_features` 和历史 `memory_features`。 | 它给出最强有效压缩：ratio 0.3 减少 80.9% token，仍有 32.43% accuracy。更高 keep ratio 没有单调提升精度，说明少量 attention 选出的 dominant/contextual token 已能保留不少 teacher-forcing 导航信号，额外局部 token 的边际收益有限。 |
| PruneVid | `algorithm/LLaVA-NeXT/llava/model/multimodal_compressor/prunevid/compressor.py` 中的结构化时空剪枝和合并方法。它用 DPC-KNN 根据帧均值形成 temporal window，再用 `tau` 把 token 分成 static/dynamic；static token 跨帧平均，dynamic token 按帧保留，最后由 `cluster_ratio` 控制空间聚类合并。 | 主要压缩插入 LLM 前的 projected video tokens。原代码还支持 `enable_vtp=true` 时进入 Qwen2 decoder 内部做 VTP 剪枝，但本表对应的 StreamVLN training-free 实验走的是 visual/memory token compressor 路径，不启用 LLM 内部剪枝。 | 中等预算最有价值：ratio 0.5 达到 33.27%，同时减少 48.9% token，接近 baseline。ratio 0.3 降到 31.42%，说明过强的时序/static 合并会删掉动作相关细节；结构化剪枝适合保守到中等强度。 |
| DyTok-static | `algorithm/LLaVA-NeXT/llava/model/multimodal_compressor/dytok/compressor.py` 中的 token 预算调度 wrapper。static 模式把实际压缩委托给基础压缩器，这里配置为 VisionZip；dynamic 模式可以通过 tiny model detection pass 和 LLM attention 分配帧预算，但本次静态实验没有使用这条动态路径。 | 通过底层 VisionZip 作用于插入 LLM 前的 projected video tokens。在 StreamVLN 实现中，`dytok_static` 会给当前视觉 token 和 memory token 分配有上限和下限的预算，再在预算内执行 VisionZip 风格选择。 | 它是本表最佳压缩点：ratio 0.7 达到 33.66%，比 baseline 高 0.04pp，同时减少 28.7% token；ratio 0.5 也有 33.30%，减少 47.8% token。这说明轻到中等强度的预算化压缩可能有去噪效果，不一定损害 teacher-forcing 决策。 |
| FastVid | `algorithm/LLaVA-NeXT/llava/model/multimodal_compressor/fastvid/compressor.py` 中的视频感知选择器。它注册 `SigLipPoolingHead`，用 projector 前的 raw features 得到帧级特征和 token attention，做动态分段，保留 salient token，并通过 density-based token merging 保留上下文 token。 | 打分阶段需要 `raw_features_before_proj`，但真正压缩的是插入 LLM 前的 projected video features。原始压缩器是 video-only；StreamVLN 迁移中把它同时用于当前视觉 token 和 memory-token 序列。 | ratio 0.5 和 0.7 很稳，分别为 33.37% 和 33.61%，token reduction 为 47.8% 和 28.8%。这说明视频动态分段对中等压缩很有效；不过它有额外 pooling/scoring 路径，token 变少不等于实际 FPS 一定更高。 |
| VQToken | `algorithm/LLaVA-NeXT/llava/model/multimodal_compressor/vqtoken/compressor.py` 中的向量量化式压缩器。本次 training-free 变体使用 fixed-K K-means，把大量视觉 token 聚成 cluster centers；代码里也有 adaptive K 和 cross-attention 选项，但这里都关闭。 | 作用于插入 LLM 前的 projected visual tokens。当前配置 `use_cross_attention=false`，所以不引入可训练 cross-attention 模块；StreamVLN 中同样把 fixed-cluster 压缩用于当前视觉 token 和 memory token。 | ratio 0.7 仍有 32.98%，但 FPS 最差，ratio 0.7 只有 2.47。聚类代表 token 在语义上有潜力，但部署时需要单独优化或 profile 聚类开销；单纯增加 cluster 数不能保证用足够低的延迟换来足够高的精度。 |

## Complete Result Table

| Method | Ratio | Status | Source | Episodes | Actions | Acc | Delta vs Base | Invalid | Tokens/step | Token reduction | FPS | GPU |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | - | valid-full | original | 240/240 | 16196 | 33.63% | +0.00pp | 48.03% | 1736.8 | 0.0% | 9.23 | 0 |
| VisionZip | 0.3 | valid-full | original | 240/240 | 16196 | 32.43% | -1.19pp | 48.21% | 331.3 | 80.9% | 6.28 | 1 |
| VisionZip | 0.5 | valid-full | original | 240/240 | 16196 | 31.54% | -2.08pp | 48.65% | 483.7 | 72.1% | 6.21 | 1 |
| VisionZip | 0.7 | valid-full | original | 240/240 | 16196 | 31.93% | -1.69pp | 49.24% | 653.1 | 62.4% | 5.99 | 1 |
| PruneVid | 0.3 | valid-full | original | 240/240 | 16196 | 31.42% | -2.21pp | 49.46% | 539.7 | 68.9% | 4.63 | 0 |
| PruneVid | 0.5 | valid-full | original | 240/240 | 16196 | 33.27% | -0.35pp | 47.37% | 886.7 | 48.9% | 4.58 | 0 |
| PruneVid | 0.7 | valid-full | original | 240/240 | 16196 | 33.01% | -0.62pp | 48.45% | 1228.4 | 29.3% | 4.59 | 0 |
| DyTok-static | 0.3 | valid-full | original | 240/240 | 16196 | 31.84% | -1.79pp | 49.36% | 575.0 | 66.9% | 6.43 | 1 |
| DyTok-static | 0.5 | valid-full | original | 240/240 | 16196 | 33.30% | -0.32pp | 47.55% | 907.1 | 47.8% | 6.21 | 1 |
| DyTok-static | 0.7 | valid-full | lowmem-rerun | 240/240 | 16196 | 33.66% | +0.04pp | 47.79% | 1239.1 | 28.7% | 5.33 | 0 |
| FastVid | 0.3 | valid-full | lowmem-rerun | 240/240 | 16196 | 31.96% | -1.66pp | 49.32% | 576.9 | 66.8% | 6.40 | 1 |
| FastVid | 0.5 | valid-full | lowmem-rerun | 240/240 | 16196 | 33.37% | -0.25pp | 47.32% | 907.1 | 47.8% | 5.75 | 1 |
| FastVid | 0.7 | valid-full | lowmem-rerun | 240/240 | 16196 | 33.61% | -0.02pp | 48.02% | 1237.3 | 28.8% | 5.36 | 0 |
| VQToken | 0.3 | valid-full | lowmem-rerun | 240/240 | 16196 | 31.61% | -2.01pp | 49.71% | 576.9 | 66.8% | 3.78 | 1 |
| VQToken | 0.5 | valid-full | lowmem-rerun | 240/240 | 16196 | 32.84% | -0.78pp | 48.09% | 907.1 | 47.8% | 2.90 | 0 |
| VQToken | 0.7 | valid-full | original | 240/240 | 16196 | 32.98% | -0.65pp | 48.52% | 1237.3 | 28.8% | 2.47 | 1 |

## Acc / Ratio Chart

<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="600" viewBox="0 0 1040 600" role="img" aria-label="Action accuracy versus keep or cluster ratio">
<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:22px;font-weight:700}.axis{font-size:13px;fill:#111827}.tick{font-size:12px;fill:#374151}.legend{font-size:13px;fill:#111827}.note{font-size:12px;fill:#4b5563}.small{font-size:11px;fill:#111827}</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="520.0" y="32" text-anchor="middle" class="title">Action Accuracy vs Keep / Cluster Ratio</text>
<text x="520.0" y="54" text-anchor="middle" class="note">Fixed TF240 AuroraReplay-GT. All points use the same marker; source details are listed below.</text>
<line x1="78" y1="514.0" x2="805" y2="514.0" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="66" y="518.0" text-anchor="end" class="tick">31</text>
<line x1="78" y1="368.0" x2="805" y2="368.0" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="66" y="372.0" text-anchor="end" class="tick">32</text>
<line x1="78" y1="222.0" x2="805" y2="222.0" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="66" y="226.0" text-anchor="end" class="tick">33</text>
<line x1="78" y1="76.0" x2="805" y2="76.0" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="66" y="80.0" text-anchor="end" class="tick">34</text>
<line x1="111.0" y1="76" x2="111.0" y2="514" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="111.0" y="542" text-anchor="middle" class="tick">0.3</text>
<line x1="441.5" y1="76" x2="441.5" y2="514" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="441.5" y="542" text-anchor="middle" class="tick">0.5</text>
<line x1="772.0" y1="76" x2="772.0" y2="514" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="5 5"/>
<text x="772.0" y="542" text-anchor="middle" class="tick">0.7</text>
<line x1="78" y1="514" x2="805" y2="514" stroke="#111827" stroke-width="1.5"/>
<line x1="78" y1="76" x2="78" y2="514" stroke="#111827" stroke-width="1.5"/>
<text x="441.5" y="578" text-anchor="middle" class="axis">Keep / cluster ratio</text>
<text x="22" y="295.0" text-anchor="middle" class="axis" transform="rotate(-90 22 295.0)">Action accuracy (%)</text>
<line x1="78" y1="130.7" x2="805" y2="130.7" stroke="#111827" stroke-width="2" stroke-dasharray="8 6"/>
<text x="801" y="123.7" text-anchor="end" class="small">Baseline 33.63%</text>
<polyline points="111.0,304.6 441.5,434.5 772.0,377.7" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="111.0" cy="304.6" r="6.8" fill="#2563eb" stroke="#ffffff" stroke-width="1.5"/>
<text x="118.0" y="294.6" class="small">32.43</text>
<circle cx="441.5" cy="434.5" r="6.8" fill="#2563eb" stroke="#ffffff" stroke-width="1.5"/>
<text x="448.5" y="452.5" class="small">31.54</text>
<circle cx="772.0" cy="377.7" r="6.8" fill="#2563eb" stroke="#ffffff" stroke-width="1.5"/>
<text x="779.0" y="395.7" class="small">31.93</text>
<polyline points="111.0,453.4 441.5,182.0 772.0,220.8" fill="none" stroke="#d97706" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="111.0" cy="453.4" r="6.8" fill="#d97706" stroke="#ffffff" stroke-width="1.5"/>
<text x="118.0" y="471.4" class="small">31.42</text>
<circle cx="441.5" cy="182.0" r="6.8" fill="#d97706" stroke="#ffffff" stroke-width="1.5"/>
<text x="448.5" y="200.0" class="small">33.27</text>
<circle cx="772.0" cy="220.8" r="6.8" fill="#d97706" stroke="#ffffff" stroke-width="1.5"/>
<text x="779.0" y="238.8" class="small">33.01</text>
<polyline points="111.0,392.1 441.5,177.5 772.0,125.3" fill="none" stroke="#059669" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="111.0" cy="392.1" r="6.8" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>
<text x="118.0" y="382.1" class="small">31.84</text>
<circle cx="441.5" cy="177.5" r="6.8" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>
<text x="448.5" y="167.5" class="small">33.30</text>
<circle cx="772.0" cy="125.3" r="6.8" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>
<text x="779.0" y="115.3" class="small">33.66</text>
<polyline points="111.0,373.2 441.5,167.6 772.0,133.4" fill="none" stroke="#64748b" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="111.0" cy="373.2" r="6.8" fill="#64748b" stroke="#ffffff" stroke-width="1.5"/>
<text x="118.0" y="391.2" class="small">31.96</text>
<circle cx="441.5" cy="167.6" r="6.8" fill="#64748b" stroke="#ffffff" stroke-width="1.5"/>
<text x="448.5" y="157.6" class="small">33.37</text>
<circle cx="772.0" cy="133.4" r="6.8" fill="#64748b" stroke="#ffffff" stroke-width="1.5"/>
<text x="779.0" y="151.4" class="small">33.61</text>
<polyline points="111.0,424.5 441.5,245.1 772.0,225.3" fill="none" stroke="#7c3aed" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="111.0" cy="424.5" r="6.8" fill="#7c3aed" stroke="#ffffff" stroke-width="1.5"/>
<text x="118.0" y="442.5" class="small">31.61</text>
<circle cx="441.5" cy="245.1" r="6.8" fill="#7c3aed" stroke="#ffffff" stroke-width="1.5"/>
<text x="448.5" y="263.1" class="small">32.84</text>
<circle cx="772.0" cy="225.3" r="6.8" fill="#7c3aed" stroke="#ffffff" stroke-width="1.5"/>
<text x="779.0" y="215.3" class="small">32.98</text>
<line x1="843" y1="92" x2="871" y2="92" stroke="#2563eb" stroke-width="3"/>
<circle cx="857" cy="92" r="5.5" fill="#2563eb"/>
<text x="881" y="96" class="legend">VisionZip</text>
<line x1="843" y1="120" x2="871" y2="120" stroke="#d97706" stroke-width="3"/>
<circle cx="857" cy="120" r="5.5" fill="#d97706"/>
<text x="881" y="124" class="legend">PruneVid</text>
<line x1="843" y1="148" x2="871" y2="148" stroke="#059669" stroke-width="3"/>
<circle cx="857" cy="148" r="5.5" fill="#059669"/>
<text x="881" y="152" class="legend">DyTok-static</text>
<line x1="843" y1="176" x2="871" y2="176" stroke="#64748b" stroke-width="3"/>
<circle cx="857" cy="176" r="5.5" fill="#64748b"/>
<text x="881" y="180" class="legend">FastVid</text>
<line x1="843" y1="204" x2="871" y2="204" stroke="#7c3aed" stroke-width="3"/>
<circle cx="857" cy="204" r="5.5" fill="#7c3aed"/>
<text x="881" y="208" class="legend">VQToken</text>
<line x1="843" y1="232" x2="871" y2="232" stroke="#111827" stroke-width="2" stroke-dasharray="7 5"/>
<text x="881" y="236" class="legend">Baseline</text>
</svg>

## Source Details

| Method | Ratio | Summary source |
|---|---:|---|
| Baseline | - | `results/teacher_forcing/summaries/original/baseline/summary.json` |
| VisionZip | 0.3 | `results/teacher_forcing/summaries/original/visionzip_keep_ratio_0_3/summary.json` |
| VisionZip | 0.5 | `results/teacher_forcing/summaries/original/visionzip_keep_ratio_0_5/summary.json` |
| VisionZip | 0.7 | `results/teacher_forcing/summaries/original/visionzip_keep_ratio_0_7/summary.json` |
| PruneVid | 0.3 | `results/teacher_forcing/summaries/original/prunevid_tau_0_7_cluster_ratio_0_3_temporal_ratio_0_5/summary.json` |
| PruneVid | 0.5 | `results/teacher_forcing/summaries/original/prunevid_tau_0_8_cluster_ratio_0_5_temporal_ratio_0_25/summary.json` |
| PruneVid | 0.7 | `results/teacher_forcing/summaries/original/prunevid_tau_0_9_cluster_ratio_0_7_temporal_ratio_0_1/summary.json` |
| DyTok-static | 0.3 | `results/teacher_forcing/summaries/original/dytok_static_visionzip_keep_ratio_0_3/summary.json` |
| DyTok-static | 0.5 | `results/teacher_forcing/summaries/original/dytok_static_visionzip_keep_ratio_0_5/summary.json` |
| DyTok-static | 0.7 | `results/teacher_forcing/summaries/lowmem_rerun/dytok_static_visionzip_keep_ratio_0_7/summary.json` |
| FastVid | 0.3 | `results/teacher_forcing/summaries/lowmem_rerun/fastvid_keep_ratio_0_3/summary.json` |
| FastVid | 0.5 | `results/teacher_forcing/summaries/lowmem_rerun/fastvid_keep_ratio_0_5/summary.json` |
| FastVid | 0.7 | `results/teacher_forcing/summaries/lowmem_rerun/fastvid_keep_ratio_0_7/summary.json` |
| VQToken | 0.3 | `results/teacher_forcing/summaries/lowmem_rerun/vqtoken_keep_ratio_0_3/summary.json` |
| VQToken | 0.5 | `results/teacher_forcing/summaries/lowmem_rerun/vqtoken_keep_ratio_0_5/summary.json` |
| VQToken | 0.7 | `results/teacher_forcing/summaries/original/vqtoken_keep_ratio_0_7/summary.json` |

## Reporting Notes

- All official configurations now have complete `240/240` fixed-subset results.
- Low-memory rerun settings: no vision precompute, `aurora_vision_batch_size=1`, `aurora_batch_size=1`, one process per GPU.
- The previous all-`0.0146` action-accuracy issue is not present; invalid predictions are tracked separately and counted as wrong.
