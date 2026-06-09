# QATA 阶段性实验总结

日期：2026-06-10  
当前分支：`exp-qata`  
范围：只总结已有 QATA / memory 实验，不包含新训练。

## 1. 统一结果表

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| MARS | Baseline | `configs/vit_clipreid.yml` | `logs/mars_vit_clip_reid_newprompt+dense_meanp` | 88.9 | 93.0 | 98.1 | 56 | 当前主 baseline |
| MARS | Plain QATA | `configs/vit_clipreid_qata.yml` | `logs/qata_mars_20260603_214227` | 88.2 | 92.3 | 97.0 | 32 | 明确下降 |
| MARS | Residual QATA a=0.1 | `configs/vit_clipreid_qata_residual_a01.yml` | `logs/qata_residual_a01_mars_20260604_185152` | 89.2 | 93.0 | 98.1 | 42 | mAP +0.3，Rank 持平 |
| MARS | Residual QATA a=0.1 repeat | `configs/vit_clipreid_qata_residual_a01.yml` | `logs/qata_residual_a01_mars_repeat_20260605_022703` | 89.2 | 93.0 | 98.1 | 56/64/66/68/74/76/78/80 | 复现 +0.3 mAP plateau |
| MARS | Residual QATA a=0.2 | `configs/vit_clipreid_qata_residual_a02.yml` | `logs/qata_residual_a02_mars_20260607_032430` | 89.2 | 93.1 | 97.9 | 54 | 不明显优于 a=0.1 |
| MARS | Consistency Memory a=0.1 | `configs/vit_clipreid_memory_consistency_a01.yml` | `logs/memory_consistency_a01_mars_20260608_023602` | 88.7 | 92.7 | 97.4 | 48 | 低于 baseline，memory 权重近似均匀 |
| iLIDS-VID | Baseline | `configs/vit_clipreid_ilids.yml` | `logs/ilids_vit_clip_reid` | 76.9 | 81.2 | 86.7 | 28 | 当前复现 baseline |
| iLIDS-VID | Plain QATA | `configs/vit_clipreid_ilids_qata.yml` | `logs/qata_ilids_20260603_214227` | 75.6 | 79.8 | 86.8 | 26 | mAP / Rank-1 下降 |
| iLIDS-VID | Residual QATA a=0.1 | `configs/vit_clipreid_ilids_qata_residual_a01.yml` | `logs/qata_residual_a01_ilids_20260607_031850` | 76.2 | 80.3 | 86.2 | 28 | 好于 plain，但低于 baseline |

诊断实验：

| Dataset | Method | Output Dir | Epochs | 关键观察 |
|---|---|---|---:|---|
| MARS | Plain QATA weight diag | `logs/qata_diag_mars_20260604_154211` | 5 | effective frames 接近 8，top1 约 0.14，权重近似均匀 |
| iLIDS-VID | Plain QATA weight diag | `logs/qata_diag_ilids_20260604_154211` | 5 | effective frames 接近 8，top1 约 0.14，权重近似均匀 |

## 2. Plain QATA 为什么失败

Plain QATA 直接把两处基础 mean pooling 替换成 learnable quality-weighted pooling：

- `img_feature.mean(1)`
- `img_feature_proj.mean(1)`

实验结果显示它在 MARS 和 iLIDS-VID 上同时下降。权重诊断显示主要问题不是权重塌缩，而是权重几乎均匀：

- `SEQ_LEN=8`
- effective frames 接近 8
- top1 weight 约 0.14，均匀权重为 0.125

这说明 qpool 没学到可靠的帧质量区分，但它仍然引入了额外参数、LayerNorm/MLP 变换和 softmax 权重扰动。对已经调好的 CLIP video feature 来说，这种“弱学习但强替换”的改动会破坏原本稳定的 mean pooling 表征。

Plain QATA 的失败可以概括为：

- 质量估计没有形成强判别。
- 直接替换 mean pooling，扰动没有边界。
- CLIP 特征本身已较稳，轻微错误加权也可能降低 rank。
- 小数据集 iLIDS 上更容易出现训练饱和但验证下降。

## 3. Residual QATA 为什么更稳定

Residual QATA 使用：

```text
pooled = mean_pool + alpha * (qpool - mean_pool)
```

其中 `alpha=0.1` 或 `0.2`。它比 plain QATA 稳定的原因是：

- mean pooling 仍是主路径，保留 baseline 的强先验。
- qpool 只作为小幅修正项，不再完全接管视频聚合。
- 即使 qpool 权重接近均匀或略有噪声，最终 feature 偏移也被 alpha 限制。
- 它把问题从“替换聚合器”变成“学习 residual correction”，优化更保守。

从结果看，Residual QATA a=0.1 在 MARS 两次复跑都达到 mAP 89.2，plain QATA 只有 88.2。这说明 residual 形式确实解决了最小 QATA 的主要不稳定来源。

## 4. MARS 上稳定 +0.3 mAP 的意义和局限

意义：

- 两次 MARS a=0.1 结果都达到 mAP 89.2，相对 baseline 88.9 稳定 +0.3。
- Rank-1 / Rank-5 基本不损失，说明 residual 修正没有破坏主检索能力。
- a=0.2 也维持 mAP 89.2，说明 residual QATA 在 MARS 上不是偶然单次异常。
- 与 plain QATA 相比，Residual QATA 明显更优，证明“受限质量修正”比“直接质量替换”更合理。

局限：

- +0.3 mAP 属于小幅收益，Rank-1/Rank-5 没有同步稳定提升。
- a=0.2 没有带来明确增益，说明普通 feature aggregation 的收益可能已接近上限。
- QATA 权重仍接近均匀，当前机制更像轻微 feature regularization，而不是强质量选择。
- 仅 MARS 为正不足以支撑“大幅质量感知建模”主张。

因此，MARS 结果可以作为有效 ablation，但不适合作为唯一核心贡献。

## 5. iLIDS 上 Residual QATA 未超过 baseline 的可能原因

iLIDS-VID 结果：

- Baseline: 76.9 / 81.2 / 86.7
- Plain QATA: 75.6 / 79.8 / 86.8
- Residual QATA a=0.1: 76.2 / 80.3 / 86.2

Residual QATA 明显好于 plain QATA，但仍低于 baseline。可能原因：

- iLIDS 数据规模更小，训练波动更大，新增 qpool 参数更容易带来噪声。
- iLIDS 当前 baseline/protocol 已知没有完全对齐论文指标，因此该数据集上的结论可信度低于 MARS。
- iLIDS 的短序列、遮挡和视角变化可能使“帧质量”与“身份判别性”不完全一致；高一致性或高权重帧未必是最有辨识度的帧。
- 权重统计显示 iLIDS 上 QATA 更接近均匀，说明模型几乎没有学到有效质量区分。
- 验证曲线在 26-28 epoch 达峰后回落，说明 residual 修正没有提供后期泛化收益。

结论：iLIDS 不支持继续扩大普通 feature-level QATA 的实验网格，但也不完全否定 MARS 上的小幅稳定收益。

## 6. Consistency Memory 当前版本为什么失败

Consistency Memory a=0.1 的设计是对每个 ID 的 sequence-level features 计算与 ID mean 的 cosine similarity，再做 residual weighted memory。

结果：

- mAP 88.7 / Rank-1 92.7 / Rank-5 97.4
- 低于 MARS baseline 88.9 / 93.0 / 98.1
- 低于 Residual QATA feature-only 89.2 / 93.0 / 98.1

memory stats：

- mean effective samples: 13.2734
- mean samples per ID: 13.2768
- mean top1 weight: 0.1304

失败原因：

- effective samples 几乎等于 samples per ID，说明权重近似均匀。
- cosine-to-ID-mean consistency 对当前 CLIP sequence features 区分力不足，或者 `tau=1.0` 过于平滑。
- 当前 memory 改动几乎退化成原始 mean memory 的极小 perturbation，却没有带来 Rank 稳定性。
- memory 是 stage2 开头构建一次并固定，后续训练无法修正 memory construction 的质量。

因此当前版本不值得继续盲目做 `MEMORY_ALPHA` / `MEMORY_TEMP` / iLIDS memory 消融。

## 7. 是否继续 QATA 方向

建议继续，但要收缩问题定义。

继续的理由：

- Residual QATA feature-only 在 MARS 上有稳定、可复现的小幅收益。
- plain vs residual 的对比清楚，能形成有价值的负例和设计动机。
- 该模块改动小、可开关、易消融，不破坏 TF-CLIP 主结构。

需要停止或暂缓的内容：

- 暂停 plain QATA。
- 暂停 memory consistency 分支。
- 暂停 iLIDS memory-only。
- 暂停盲目 alpha/temp 网格。
- 暂停扩展到 TMD / CLIP-Memory SSP / dense inference。

当前最合理定位：Residual QATA 是一个低风险 feature aggregation regularizer，而不是完整解决 Video ReID 低质量帧问题的最终方案。

## 8. 如果写文章，QATA 可以作为哪个部分

可以作为：

- 一个轻量 residual temporal aggregation module。
- 一个 baseline-compatible plug-in，用于减少直接质量加权带来的扰动。
- 一个 ablation-driven design：plain QATA 失败，residual QATA 修复。
- 一个辅助模块，配合更强的主创新使用。
- “quality-aware but conservative aggregation”的分析案例。

不适合作为：

- 单独主贡献。
- “显著提升 SOTA”的核心模块。
- 强帧选择或强质量估计方法，因为当前权重接近均匀。
- CLIP-Memory learning 的核心证据，因为 memory consistency 当前失败。
- 大规模泛化结论，因为 iLIDS 未超过 baseline。

更稳的论文叙述方式：

- 把 Residual QATA 放在 method 的一个子模块。
- 把 plain QATA 作为 negative ablation。
- 强调 residual constraint 的必要性。
- 不夸大质量权重本身的可解释性，除非后续有更强可视化证据。

## 9. 下一步最值得探索的 3 个方向

按风险从低到高排序：

### 方向 1：Residual QATA 可视化与统计分析

风险：低。

内容：

- 可视化 high-weight / low-weight frames。
- 统计不同遮挡、模糊、背景干扰情况下的权重差异。
- 比较 MARS 与 iLIDS 的权重分布。
- 分析 alpha=0.1 和 alpha=0.2 为什么收益接近。

价值：

- 不需要新训练。
- 可以支撑论文中的机制分析。
- 可以判断 QATA 到底学到质量，还是只是 regularization。

### 方向 2：Residual QATA + 更稳的质量先验

风险：中。

内容：

- 不再只依赖 learnable MLP qpool。
- 引入 deterministic quality hints，例如 frame-to-sequence consistency、feature norm、temporal stability。
- 仍使用 residual 形式，避免直接替换 mean pooling。

价值：

- 针对 plain QATA “学不到质量”的问题。
- 比 memory 分支更小、更接近已有正结果。
- 易做 ablation：learned-only vs deterministic-only vs hybrid residual。

### 方向 3：Robust Memory，而不是 Softmax Memory

风险：中高。

内容：

- 放弃当前 softmax consistency memory。
- 只做 outlier suppression / trimmed mean / residual robust mean。
- 目标不是选择 top sequence，而是压低明显异常 sequence。

价值：

- 更符合当前 memory 诊断：问题不是 top1 不够尖，而是需要判断是否存在 outlier。
- 比 learnable qpool memory 更公平，因为不依赖随机初始化模块。

风险：

- 需要先恢复 `exp-memory-consistency` 或重新实现。
- 需要做 feature/similarity 诊断，否则仍可能盲目。
- 如果 cosine score 本身没区分力，该方向也应停止。

最终建议：

短期不要继续跑训练。先把 Residual QATA feature-only 的实验、可视化、失败案例和机制解释整理完整。如果需要新的技术推进，优先做 feature-level residual quality prior，而不是继续 memory temperature/alpha 消融。
