# TF-CLIP QATA / Memory 阶段性研究决策报告

日期：2026-06-13

当前结论先行：现阶段不建议继续盲目训练 QATA 网格、Consistency Memory、KMeans/max/iLIDS multi-prototype 或 Residual QATA 组合。当前唯一可信正结果是 **Residual QATA alpha=0.1 在 MARS 上稳定 +0.3 mAP**，但幅度较小，更适合作为辅助组件或分析模块，不足以单独支撑主创新。下一阶段应转向更明确的问题场景，例如公开困难场景迁移、显式质量退化建模、camera/view-aware memory。

## 1. 已完成实验总表

### MARS 主实验

| Method | Config / Output | mAP | Rank-1 | Rank-5 | Best Epoch | 结论 |
|---|---|---:|---:|---:|---:|---|
| Baseline | `configs/vit_clipreid.yml`; `logs/mars_vit_clip_reid_newprompt+dense_meanp` | 88.9 | 93.0 | 98.1 | 56 | 当前复现基线 |
| Plain QATA | `configs/vit_clipreid_qata.yml`; `logs/qata_mars_20260603_214227` | 88.2 | 92.3 | 97.0 | 32 | 明显下降 |
| Residual QATA a=0.1 | `configs/vit_clipreid_qata_residual_a01.yml`; `logs/qata_residual_a01_mars_20260604_185152` | 89.2 | 93.0 | 98.1 | 42 | 正收益 |
| Residual QATA a=0.1 repeat | `logs/qata_residual_a01_mars_repeat_20260605_022703` | 89.2 | 93.0 | 98.1 | 56/64/66/68/74/76/78/80 | 复跑稳定 |
| Residual QATA a=0.2 | `configs/vit_clipreid_qata_residual_a02.yml`; `logs/qata_residual_a02_mars_20260607_032430` | 89.2 | 93.1 | 97.9 | 54 | 不明显优于 a=0.1 |
| Consistency Memory a=0.1 | `configs/vit_clipreid_memory_consistency_a01.yml`; `logs/memory_consistency_a01_mars_20260608_023602` | 88.7 | 92.7 | 97.4 | 48 | 低于 baseline |
| Old Multi-prototype farthest LSE | `logs/multiproto_k2_farthest_lse_mars_20260610_143142` | 83.9 | 90.9 | 97.8 | 62 | 大幅下降，后确认 scoring path 不等价 |
| Old duplicate mean LSE | `logs/multiproto_k2_dupmean_lse_mars_20260610_195020` | 84.6 | 90.7 | 97.3 | 66 | 控制实验也下降，定位 scoring path bug |
| Fixed duplicate mean LSE | `logs/multiproto_k2_dupmean_lse_fixed_mars_20260611_015110` | 88.9 | 93.0 | 97.3 | 64 | 修复后恢复 baseline-level mAP |
| Fixed farthest-two LSE | `logs/multiproto_k2_farthest_lse_fixed_mars_20260611_173726` | 88.3 | 93.0 | 97.4 | 48 | 修复后不崩，但仍低于 baseline |

### iLIDS-VID 参考实验

| Method | Config / Output | mAP | Rank-1 | Rank-5 | Best Epoch | 结论 |
|---|---|---:|---:|---:|---:|---|
| Baseline | `configs/vit_clipreid_ilids.yml`; `logs/ilids_vit_clip_reid` | 76.9 | 81.2 | 86.7 | 28 | 当前复现基线，未完全对齐论文 |
| Plain QATA | `configs/vit_clipreid_ilids_qata.yml`; `logs/qata_ilids_20260603_214227` | 75.6 | 79.8 | 86.8 | 26 | mAP / Rank-1 下降 |
| Residual QATA a=0.1 | `configs/vit_clipreid_ilids_qata_residual_a01.yml`; `logs/qata_residual_a01_ilids_20260607_031850` | 76.2 | 80.3 | 86.2 | 28 | 好于 plain QATA，但低于 baseline |

## 2. 路线结论

### 2.1 Plain QATA 为什么失败

Plain QATA 直接用 learnable quality pooling 替换 `img_feature.mean(1)` 和 `img_feature_proj.mean(1)`。诊断显示权重没有塌缩，而是接近均匀：SEQ_LEN=8 时 effective frames 接近 8，top1 weight 约 0.14，均匀权重 top1 为 0.125。

这说明 plain qpool 没有学到可靠的帧质量区分，只是在一个已经调好的 CLIP video representation 上引入了额外扰动。由于原始 mean pooling 本身很强，直接替换会破坏稳定性，MARS 和 iLIDS 都下降。

### 2.2 Residual QATA 为什么是当前唯一正收益

Residual QATA 使用：

```text
pooled = mean_pool + alpha * (qpool - mean_pool)
```

它保留 mean pooling 作为主路径，只让 QATA 做小幅修正。alpha=0.1 时，MARS 从 88.9 mAP 提升到 89.2 mAP，复跑稳定；Rank-1 / Rank-5 基本保持 baseline。

本质上，它不是强质量选择器，而是一个受限扰动项。它能降低 plain QATA 的风险，但也限制了上限。alpha=0.2 没有带来明确增益，说明继续做 alpha/temp 网格的收益有限。

### 2.3 Consistency Memory 为什么退化为 mean

Consistency Memory a=0.1 使用 sequence feature 与 ID mean prototype 的 cosine similarity 做 softmax 权重，再 residual 更新 memory。实验结果低于 baseline：88.7 / 92.7 / 97.4。

memory stats 显示：

- mean effective samples: 13.2734
- mean samples per ID: 13.2768
- mean top1 weight: 0.1304

effective samples 几乎等于每个 ID 的平均样本数，说明权重基本均匀。原因是 ID 内 cosine similarity 分布区分度不足，tau=1.0 下 softmax 过平，最终几乎回到 mean memory。它没有有效筛选高质量 sequence，反而引入了轻微 prototype 偏移。

### 2.4 Multi-prototype scoring path bug 如何被 duplicate mean 定位

旧版 farthest LSE 大幅下降到 83.9 mAP，但不能直接归因于 multi-prototype 假设失败。随后设计了 duplicate mean 控制实验：每个 ID 的两个 prototype 都复制原始 mean memory，理论上应等价或接近 single-prototype baseline。

旧 duplicate mean LSE 仍下降到 84.6 mAP，说明问题在 scoring path / normalization / logsumexp / SSP shape 处理，而不仅是 farthest prototype 构建。

定位出的关键问题包括：

- multi path 改成了 cosine/normalize 风格，而原始 single path 是 raw dot 兼容路径。
- logsumexp 对 duplicate prototype 引入 `temp * log(K)` 常数，需要校正。
- SSP 曾把 `[C,K,D]` flatten 成 `C*K` memory tokens，改变了 self-attention token 数。
- duplicate_mean 在控制路径中不应 normalize，否则不再严格等价原始 mean memory。

修复后，数值等价检查通过：duplicate max vs single 差异为 0，corrected duplicate logsumexp vs single 约 `3.8e-6`。Fixed duplicate mean LSE 恢复到 88.9 mAP / 93.0 Rank-1，证明 scoring path 已基本修复。

### 2.5 修正后 farthest-two 为什么仍不适合作为主线

修正 scoring path 后，farthest-two LSE 从旧版 83.9 恢复到 88.3，但仍低于 baseline 88.9，也低于 Residual QATA 89.2。

memory stats:

- prototype cosine mean/std: 0.9322 / 0.0329
- fallback IDs: 1
- empty cluster fallbacks: 0

两个 prototype 不是完全重复，但仍高度相似，说明 feature-space farthest split 没有形成有意义的身份模式分解。duplicate mean 能恢复 baseline-level mAP，而 farthest-two 下降，当前更像是 farthest construction / assignment 本身不够有效，而不是 scoring path 仍有主要错误。

因此，farthest-two 不适合作为下一阶段主线。继续跑 KMeans/max/iLIDS 之前，必须先有更强的 prototype 语义依据或质量诊断，否则只是消耗 GPU。

## 3. 当前最可信正结果

最可信正结果是：

```text
Residual QATA alpha=0.1 on MARS:
Baseline 88.9 / 93.0 / 98.1
Residual QATA 89.2 / 93.0 / 98.1
Repeat stable around 89.2 mAP
```

可信点：

- 两次 MARS 完整训练都复现 +0.3 mAP。
- Rank-1 / Rank-5 未受损。
- 机制保守，baseline-compatible，容易消融。

局限：

- 增益只有 +0.3 mAP，可能不足以支撑主贡献。
- iLIDS 上未超过 baseline。
- 权重仍接近均匀，不支持“强质量选择”的叙事。
- 更合理的表述是 conservative residual aggregation，而不是 robust quality-aware temporal selection。

## 4. 当前不建议继续投入的方向

1. 普通 QATA alpha/temp 网格

Plain QATA 已经证明直接替换 mean pooling 有风险。Residual a=0.2 没有明显优于 a=0.1，继续网格搜索更像调参，不像新问题驱动。

2. Consistency Memory temp/alpha

a=0.1 / tau=1.0 已经显示权重近似 mean。直接跑 low-temp 或 alpha 网格没有先验证 similarity 分布，风险高。若继续，必须先做 CPU-only 分布诊断，而不是完整训练。

3. Farthest-two Multi-prototype

修正后仍低于 baseline，prototype cosine 高，说明当前 feature-space split 不够有意义。

4. KMeans / max / iLIDS multi-prototype 盲跑

KMeans 可能只是更复杂的 feature-space clustering；max 可能引入 hard assignment 噪声；iLIDS baseline 本身不够稳。没有新的诊断依据前，不建议继续跑。

## 5. 下一阶段更有论文潜力的方向

### 优先级 1：公开困难场景迁移，例如 UAV / G2A / S2R

核心动机：当前 TF-CLIP 在常规 Video ReID 数据集上已经很强，小模块很难显著提升。困难场景更容易暴露 TF-CLIP 的真实缺陷，如视角极端、低分辨率、运动模糊、背景复杂、跨域分布变化。

最小方案：

- 选择一个公开困难场景数据集或协议。
- 先复现 TF-CLIP baseline。
- 分析 failure case，再引入轻量改造。
- Residual QATA 可以作为辅助模块参与对比，但不是主线。

优点：问题驱动更明确，论文叙事更强。

风险：数据集适配和 baseline 复现成本较高。

### 优先级 2：质量退化建模，但必须有显式 degradation cue

核心动机：隐式 attention/qpool 没有学到质量区分。下一步如果继续质量方向，应显式建模退化因素，而不是期待网络自己从 ID loss 中学质量。

可能 cue：

- blur / sharpness
- detection box truncation
- occlusion ratio
- frame-text CLIP confidence
- temporal consistency
- low-resolution / motion degradation augmentation label

最小方案：

- 构造 degradation augmentation。
- 给模型明确的 degradation cue 或 auxiliary quality target。
- 让 aggregation 或 loss 使用这个 cue。

优点：比 QATA 的隐式权重更可解释。

风险：需要定义可靠质量标签或 proxy，实验设计更复杂。

### 优先级 3：camera/view-aware memory，而不是 feature-space farthest split

核心动机：multi-prototype 的想法仍有合理性，但 prototype 应该对应真实模式，例如 camera/view/domain，而不是单纯 feature-space farthest。

最小方案：

- 按 camera ID 或 view/domain meta 信息为每个 ID 构建多个 prototype。
- query 时对 camera-aware prototypes 做 max/logsumexp 或 camera-conditioned selection。
- 保持 loss 输出 `[B,num_classes]`，不改 loss。

优点：比 farthest split 更有语义依据。

风险：依赖数据集是否有 camera/view 信息；跨数据集泛化需要谨慎。

### 优先级 4：pseudo-attribute / semantic memory

核心动机：TF-CLIP 的 memory 当前是视觉 feature mean，缺少可解释的语义结构。属性或 pseudo-caption 可以把 identity memory 从纯视觉均值扩展为视觉-语义 memory。

最小方案：

- 用外部 caption/attribute 模型生成 tracklet-level pseudo attributes。
- 构建 identity-level semantic memory。
- 与 CLIP text/image branch 做对齐。

优点：创新强度高，贴近 CLIP。

风险：依赖额外模型，噪声和工程复杂度高，公平性和 reproducibility 成本较大。

## 6. 明确建议

### 是否继续当前模块路线？

不建议把当前 QATA / multi-prototype 模块路线继续作为主线推进。已有结果显示，小模块在 MARS 上只能提供很小收益，iLIDS 不稳，memory 方向目前没有正收益。继续做 alpha/temp/KMeans/max 更像调参搜索，论文风险高。

### 是否应转向“公开困难场景问题 + 成熟 baseline 改造”？

建议转向。TF-CLIP baseline 在常规 MARS/iLIDS 上已经较强，简单 aggregation/memory 改动空间有限。公开困难场景可以提供更清晰的问题定义，也更容易解释为什么需要质量、视角、跨域或语义增强。

推荐下一阶段主线：

```text
公开困难场景迁移 / 退化场景 Video ReID
+ 显式质量退化 cue
+ Residual QATA 作为辅助稳定组件
```

### Residual QATA 适合当主贡献还是辅助组件？

Residual QATA 不适合作为主贡献。它适合作为：

- 一个 low-risk auxiliary module；
- 一个稳定 baseline enhancement；
- 一个 ablation 中的 conservative temporal correction；
- 一个质量退化建模主线下的辅助 aggregation layer。

如果写文章，Residual QATA 可以放在方法的一小节或 ablation support 中，用于证明“直接 learnable quality attention 不稳定，residual constraint 更稳”。但主贡献应来自更明确的问题设置，例如困难场景迁移、显式 degradation-aware learning、camera/view-aware memory 或 semantic memory。

## 7. 当前停止线

停止以下训练：

- Plain QATA 新网格
- Residual QATA alpha/temp 大规模搜索
- Consistency Memory temp/alpha
- Farthest-two multi-prototype
- KMeans multi-prototype
- max aggregation
- iLIDS multi-prototype
- Residual QATA + multi-prototype 组合

保留以下资产：

- Residual QATA a=0.1 MARS 稳定正结果。
- duplicate mean control 定位 scoring path bug 的完整诊断链。
- Fixed farthest-two negative result，用于证明简单 feature-space split 不够。

下一步应先确定困难场景数据集和问题协议，再设计有显式 cue 的方法，而不是继续堆模块。
