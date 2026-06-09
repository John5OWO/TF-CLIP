# 下一阶段创新方向设计

日期：2026-06-10  
当前分支：`exp-qata`  
约束：不修改代码，不启动训练，不运行 GPU 任务。  
依据：TF-CLIP baseline、QATA 阶段实验、memory consistency 负结果。

## 0. 当前结论基线

已有关键结果：

| Dataset | Method | mAP | Rank-1 | Rank-5 | 结论 |
|---|---|---:|---:|---:|---|
| MARS | Baseline | 88.9 | 93.0 | 98.1 | 当前主 baseline |
| MARS | Plain QATA | 88.2 | 92.3 | 97.0 | 直接替换 mean pooling 失败 |
| MARS | Residual QATA a=0.1 | 89.2 | 93.0 | 98.1 | 稳定 +0.3 mAP |
| MARS | Residual QATA a=0.2 | 89.2 | 93.1 | 97.9 | 不明显优于 a=0.1 |
| MARS | Consistency Memory a=0.1 | 88.7 | 92.7 | 97.4 | memory 权重近似均匀，失败 |
| iLIDS-VID | Baseline | 76.9 | 81.2 | 86.7 | 当前复现 baseline |
| iLIDS-VID | Plain QATA | 75.6 | 79.8 | 86.8 | 失败 |
| iLIDS-VID | Residual QATA a=0.1 | 76.2 | 80.3 | 86.2 | 好于 plain，但低于 baseline |

阶段性判断：

- 普通 learnable frame weighting 不足以作为强主线。
- Residual 约束有效，但收益小，适合作为辅助模块或设计原则。
- 当前 CLIP-Memory 是每个 ID 一个 mean prototype，且 stage2 开头一次性固定，这是 TF-CLIP 的明显可改进点。
- Consistency Memory 失败的原因不是 memory 方向一定错，而是当前 score 近似均匀，没有形成有效结构。

## 1. 方向一：Multi-prototype CLIP-Memory

### A. 核心动机与 TF-CLIP 缺陷

TF-CLIP 当前为每个 identity 构建一个固定 memory prototype：

```text
memory_i = mean(sequence_features of ID_i)
```

这个单 prototype 假设同一个 ID 的所有 tracklet 可以被一个中心表示覆盖。但 Video ReID 中同一行人可能存在：

- 多视角：front / back / side。
- 多摄像头风格差异。
- 遮挡程度差异。
- 光照和背景变化。
- 检测框偏移。

单一 mean prototype 会把不同模式混在一起，可能削弱 I2T / SSP 对 hard tracklet 的监督。QATA 阶段也说明“对所有样本做一个平滑权重”不够，下一步更有潜力的是让 memory 本身从一个中心变成多个身份子原型。

### B. 最小实现方案

最小版本：Identity-local K-prototype memory。

1. 仍使用 `model(img, get_image=True)` 提取 sequence-level feature `[N, 512]`。
2. 对每个 ID 内的 features 做轻量聚类，得到 `K` 个 prototype。
3. 得到 memory shape：

```text
cluster_features: [num_classes, K, 512]
```

4. stage2 forward 中，对每个 image feature 与该 ID memory bank 做 similarity：

```text
logits[b, id, k] = cos(img_b, proto_id_k)
logits_id = max_k logits[b, id, k]
```

或用 soft aggregation：

```text
logits_id = logsumexp_k(logits[b, id, k] / tau) * tau
```

5. 初始 K 建议：

- `K=2`：低风险，前后/侧面或 clean/noisy 分裂。
- `K=3`：中等风险。

6. 如果某 ID 样本数少于 K，则复制 mean prototype 或使用实际样本数加 mask。

第一版只改 memory 和 logits，不动 TMD、dense inference、QATA。

### C. 需要修改哪些文件

建议新增：

- `model/multi_prototype_memory.py`
  - `build_multi_prototype_memory(features, labels, num_classes, k, mode, eps)`
  - 支持 `kmeans`、`camera_split`、`random_init_kmeans`、`fallback_mean`

需要修改：

- `config/defaults.py`
  - 增加 `MODEL.MEMORY.MULTI_ENABLED`
  - `MODEL.MEMORY.NUM_PROTOTYPES`
  - `MODEL.MEMORY.AGG_MODE = "max" | "logsumexp" | "mean"`
  - `MODEL.MEMORY.CLUSTER_MODE = "kmeans" | "camera" | "hybrid"`
- `configs/vit_clipreid_memory_multiproto_k2.yml`
- `processor/processor_clipreid_stage2.py`
  - 替换或包装 `generate_cluster_features()`
  - 让 `cluster_features` 支持 `[C, K, 512]`
- `model/make_model_clipreid.py`
  - 处理 `text_features2` 为 `[B, C, K, 512]`
  - 对 prototype 维度做 `max/logsumexp`
- `loss/make_loss.py`
  - 如果 `i2tscore` 最终仍是 `[B, C]`，可不改
  - 如果保留 `[B, C, K]`，需要改 loss 前聚合

### D. 实验和消融

主实验：

| Dataset | Variant | 目的 |
|---|---|---|
| MARS | baseline single prototype | reference |
| MARS | K=2 max | 最小 multi-prototype |
| MARS | K=2 logsumexp | 稳定聚合对比 |
| MARS | K=3 logsumexp | 检查更多 prototype 是否有效 |
| iLIDS | K=2 best mode | 跨数据集验证 |

关键消融：

- `K=1/2/3`
- `max` vs `logsumexp` vs `mean over prototypes`
- k-means prototype vs camera-based prototype
- fixed memory vs periodic refresh，不建议第一轮做 refresh
- feature-only Residual QATA 是否叠加，不建议第一轮叠加

诊断：

- 每个 ID 内 prototype 间 cosine distance。
- 每个 prototype 分配的 tracklet 数。
- query/gallery 命中哪个 prototype。
- hard IDs 是否受益。

### E. 评估

| 维度 | 评价 |
|---|---|
| 创新强度 | 高。直接改 TF-CLIP 的 CLIP-Memory 表示假设，从 single identity prototype 扩展到 multi-modal identity memory。 |
| 实现难度 | 中高。需要改 memory shape 和 model logits，但不必改 backbone。 |
| 训练成本 | 中。memory 构建多一点 CPU/GPU 聚类成本，stage2 logits 增加 K 倍，但 K=2/3 可控。 |
| 发论文潜力 | 高。动机清楚，和 Video ReID 多视角/遮挡天然相关，消融清晰，能独立成主线。 |
| 主要风险 | 如果 ID 内样本太少或聚类不稳定，prototype 可能噪声化；iLIDS 小数据集风险更高。 |

## 2. 方向二：Occlusion-aware Temporal Aggregation

### A. 核心动机与 TF-CLIP 缺陷

TF-CLIP 当前视频级聚合主要依赖 mean pooling 和 TMD。QATA 证明简单 learnable frame weighting 不可靠，但它没有显式建模遮挡。Video ReID 中很多失败来自：

- 行人被局部遮挡。
- 检测框包含大量背景。
- 某些帧只有局部身体可见。
- 同一 tracklet 内可见部位变化大。

Occlusion-aware temporal aggregation 的目标不是泛泛学习“质量”，而是显式估计每帧或每 patch 的可见性，用可见性指导时间聚合。

### B. 最小实现方案

建议第一版不要做复杂分割，而做 CLIP token consistency visibility。

1. 从 CLIP image encoder 获取 patch tokens `[B, T, P, C]`。
2. 计算每帧 patch token 与 sequence prototype 的一致性。
3. 得到 frame-level occlusion score：

```text
visible_score[t] = mean_top_p cosine(patch_tokens[t], sequence_mean)
```

或：

```text
occlusion_score[t] = patch_token_variance / background_dominance
```

4. 使用 residual temporal aggregation：

```text
pooled = mean_pool + alpha * (visible_weighted_pool - mean_pool)
```

5. 第一版只在 `img_feature_proj` 分支做，不改 TMD 和 dense inference。

更稳的替代：不改 forward，先做 occlusion score 诊断和可视化，确认 score 能识别遮挡帧。

### C. 需要修改哪些文件

建议新增：

- `model/occlusion_aggregation.py`
  - `PatchVisibilityEstimator`
  - `OcclusionAwareTemporalPooling`

可能需要修改：

- `model/make_model_clipreid.py`
  - 暴露 patch tokens 或中间 tokens。
  - 在基础 pooling 位置加入可开关 residual occlusion pooling。
- `config/defaults.py`
  - `MODEL.OCC_AGG.ENABLED`
  - `MODEL.OCC_AGG.ALPHA`
  - `MODEL.OCC_AGG.TOP_P`
  - `MODEL.OCC_AGG.TEMP`
- `configs/vit_clipreid_occagg.yml`
- `processor/processor_clipreid_stage2.py`
  - 只在需要统计日志时增加轻量记录。

如果 CLIP encoder 当前不方便返回 patch tokens，实现难度会明显上升。

### D. 实验和消融

主实验：

- MARS baseline vs occlusion-aware residual aggregation。
- iLIDS baseline vs occlusion-aware residual aggregation。
- 与 Residual QATA a=0.1 对比。

消融：

- frame CLS score vs patch visibility score。
- `top_p=0.3/0.5/0.7`
- `alpha=0.1/0.2`
- 只作用于 `img_feature_proj` vs 同时作用于 `img_feature`
- 不同遮挡强度样本上的分组指标。

诊断：

- 高遮挡帧是否低权重。
- 背景占比高的检测框是否低权重。
- 可视化 patch visibility heatmap。

### E. 评估

| 维度 | 评价 |
|---|---|
| 创新强度 | 中高。比普通 QATA 更具体，针对遮挡失败场景。 |
| 实现难度 | 中高。关键取决于当前 CLIP ViT 是否容易返回 patch tokens。 |
| 训练成本 | 中。额外 token 计算和日志可控，但显存可能增加。 |
| 发论文潜力 | 中高。如果可视化成立，故事较强；如果只做 frame score，可能和 QATA 太接近。 |
| 主要风险 | 没有遮挡标注，visibility score 可能仍不可靠；可能重复 QATA 的“近似均匀权重”问题。 |

## 3. 方向三：Pseudo-caption / Attribute-enhanced CLIP Memory

### A. 核心动机与 TF-CLIP 缺陷

TF-CLIP 名义上利用 CLIP，但当前 stage2 的 `cluster_features` 本质是 image feature memory，不是真正的文本语义 memory。它缺少：

- 衣服颜色。
- 上下装类型。
- 背包/帽子等属性。
- 性别/发型等弱语义。
- 跨视角稳定的身份描述。

Pseudo-caption / attribute-enhanced memory 的目标是用属性或伪 caption 给 identity memory 注入更稳定的语义约束，让 CLIP 的 text/image 对齐能力真正参与 Video ReID。

### B. 最小实现方案

最小版本：attribute prompt memory。

1. 为每个 tracklet 或 ID 生成 pseudo attributes：

- 颜色：upper/lower color。
- clothing：long sleeve, short sleeve, pants, skirt。
- accessories：bag, backpack, hat。

2. 用模板生成 pseudo caption：

```text
"a person wearing {upper_color} top and {lower_color} pants, {accessory}"
```

3. 用 CLIP text encoder 得到 text memory `[num_classes, 512]`。
4. 与原始 image memory 融合：

```text
memory = image_memory + beta * (text_attr_memory - image_memory)
```

或双分支 I2T：

```text
logits = logits_image_memory + lambda * logits_attr_memory
```

第一版不需要训练 caption generator。可以用离线 attribute predictor 或简单 CLIP zero-shot prompts 生成属性。

### C. 需要修改哪些文件

建议新增：

- `tools/extract_pseudo_attributes.py`
  - 离线生成 `pseudo_attributes.json`
- `model/attribute_memory.py`
  - prompt 模板
  - text feature construction
  - image/text memory fusion

需要修改：

- `config/defaults.py`
  - `MODEL.ATTR_MEMORY.ENABLED`
  - `MODEL.ATTR_MEMORY.ATTR_FILE`
  - `MODEL.ATTR_MEMORY.FUSION_BETA`
  - `MODEL.ATTR_MEMORY.LOGITS_LAMBDA`
- `processor/processor_clipreid_stage2.py`
  - memory construction 时读取 attribute memory。
- `model/make_model_clipreid.py`
  - 支持额外 text memory logits 或 fused memory。
- `datasets` 相关文件
  - 如果需要 tracklet path 到 ID 的映射，可能要扩展 dataset 返回 meta。

### D. 实验和消融

主实验：

- baseline image memory。
- image memory + pseudo attribute text memory。
- only attribute text memory，不一定预期好，但可做消融。
- Residual fusion beta 0.1/0.2/0.5。

消融：

- manual template 数量。
- color-only vs color+accessory。
- ID-level majority attribute vs tracklet-level attribute。
- fusion feature-level vs logits-level。
- 属性置信度过滤。

诊断：

- pseudo attribute 准确率人工抽样。
- 错误 caption 对性能影响。
- 提升是否来自颜色类属性。

### E. 评估

| 维度 | 评价 |
|---|---|
| 创新强度 | 高。更充分利用 CLIP 的语言侧，和 TF-CLIP 名称/框架契合。 |
| 实现难度 | 高。需要属性生成、prompt 设计、memory 融合和错误属性鲁棒性。 |
| 训练成本 | 中。训练本身不一定贵，但离线属性提取和调试成本高。 |
| 发论文潜力 | 高。如果属性质量可靠，故事比 QATA 更强。 |
| 主要风险 | pseudo caption 噪声大；需要外部 attribute 模型会影响公平性；如果只用简单 prompt，可能效果弱。 |

## 4. 方向四：Frame Degradation Augmentation + Quality-aware Training

### A. 核心动机与 TF-CLIP 缺陷

QATA 失败的核心原因之一是没有质量监督。模型只通过 ReID loss 间接学习 frame weights，结果权重近似均匀。Frame degradation augmentation 的思路是主动构造低质量帧，让模型知道哪些帧被模糊、遮挡、噪声、背景污染，从而获得质量学习信号。

它直接解决 TF-CLIP 的缺陷：

- mean pooling 默认所有帧同质量。
- QATA 没有显式质量标签。
- Video ReID 真实退化包括 blur、occlusion、低分辨率、检测偏移。

### B. 最小实现方案

最小版本：degradation-aware residual training。

1. 在训练 dataloader 中随机对部分帧施加退化：

- Gaussian blur。
- random erasing / block occlusion。
- color jitter。
- downsample-upsample。
- bbox shift/crop padding。

2. 同时生成 degradation label：

```text
quality_target[t] = 1.0 for clean frame
quality_target[t] = q_low for degraded frame
```

3. 给 `FrameQualityEstimator` 增加辅助 loss：

```text
L_quality = BCE or MSE(pred_quality, quality_target)
```

4. 聚合仍然使用 residual：

```text
pooled = mean + alpha * (qpool - mean)
```

5. 第一版只训练 MARS，不碰 memory/TMD/dense inference。

该方向本质上修复 QATA 的关键短板：没有质量监督。

### C. 需要修改哪些文件

建议新增：

- `datasets/frame_degradation.py`
  - degrade transform 和 quality label 生成
- `loss/quality_loss.py`
  - quality supervision loss

需要修改：

- `datasets/video_loader_wyq.py` 或 transform pipeline
  - 返回 degraded frames 和 per-frame quality labels
- `processor/processor_clipreid_stage2.py`
  - 接收 quality labels
  - 计算 `L_quality`
  - 记录 quality prediction stats
- `model/quality_aggregation.py`
  - 暴露 quality scores，不一定改核心结构
- `model/make_model_clipreid.py`
  - forward 返回 quality scores 或保存在 model 状态中
- `config/defaults.py`
  - `INPUT.DEGRADATION.ENABLED`
  - `MODEL.QA_TRAIN.ENABLED`
  - `MODEL.QA_TRAIN.LOSS_WEIGHT`
  - degradation 类型和概率
- `configs/vit_clipreid_degrade_qtrain.yml`

### D. 实验和消融

主实验：

- baseline。
- Residual QATA a=0.1。
- degradation augmentation only，不使用 quality loss。
- degradation + quality loss。
- degradation + quality loss + residual QATA。

消融：

- blur / occlusion / low-res / bbox shift 单独测试。
- degradation probability。
- quality loss weight。
- clean/degraded mixed ratio。
- train with degradation, test clean。
- train with degradation, test synthetic degraded validation。

诊断：

- quality score 是否能区分 clean/degraded。
- high degradation frames 是否低权重。
- 对真实遮挡/模糊 tracklet 的 subgroup performance。

### E. 评估

| 维度 | 评价 |
|---|---|
| 创新强度 | 中高。不是只改结构，而是引入质量监督和鲁棒训练。 |
| 实现难度 | 中。需要改 dataloader/processor/loss，但不需要改 CLIP-Memory。 |
| 训练成本 | 中。训练成本基本不变，数据增强略增 CPU 开销。 |
| 发论文潜力 | 中高。能解释 QATA 失败并提出监督式质量学习，实验故事清晰。 |
| 主要风险 | 合成退化与真实低质量帧存在 domain gap；过强增强可能损害 clean performance。 |

## 5. 横向比较

| Direction | 创新强度 | 实现难度 | 训练成本 | 与当前结果兼容性 | 论文潜力 | 主要风险 |
|---|---|---|---|---|---|---|
| Multi-prototype CLIP-Memory | 高 | 中高 | 中 | 高，直接针对 single mean memory 缺陷 | 高 | memory/logits 改动较大，聚类不稳定 |
| Occlusion-aware temporal aggregation | 中高 | 中高 | 中 | 中，可能复用 residual 思想 | 中高 | 无遮挡标注，score 可能仍近似均匀 |
| Pseudo-caption / attribute memory | 高 | 高 | 中 | 中，增强 CLIP 语义侧 | 高 | pseudo attribute 噪声和外部模型公平性 |
| Degradation augmentation + QA training | 中高 | 中 | 中 | 高，直接修复 QATA 无监督问题 | 中高 | 合成退化 domain gap |

## 6. 最推荐主线

推荐主线：Multi-prototype CLIP-Memory。

原因：

1. 它直接针对 TF-CLIP 的核心结构缺陷：每个 ID 只有一个 fixed mean memory。
2. 它比继续 temporal pooling 更有论文空间，因为 memory 是 TF-CLIP 的关键模块，不只是边缘 aggregation trick。
3. 它可以自然解释 Video ReID 的多模态身份分布：视角、摄像头、遮挡、光照。
4. 它不依赖随机 qpool，也不需要外部 attribute/caption 模型。
5. 消融清楚：`K=1` baseline、`K=2/3`、`max/logsumexp`、camera split/kmeans。
6. 如果做得好，可以和 Residual QATA 形成组合：feature aggregation 是辅助，multi-prototype memory 是主创新。

第一阶段建议只做 MARS：

| Step | Experiment | 目的 |
|---:|---|---|
| 1 | K=2, logsumexp aggregation | 最稳的 multi-prototype memory 验证 |
| 2 | K=2, max aggregation | 检查 hard prototype selection |
| 3 | K=3, logsumexp | 检查 prototype 数量上限 |
| 4 | camera-based prototype | 验证是否视角/摄像头驱动 |
| 5 | best multi-proto + Residual QATA a=0.1 | 检查是否互补 |

当前不推荐作为主线：

- Pseudo-caption / attribute memory：论文潜力高，但工程和噪声风险最大，适合作为后续增强。
- Occlusion-aware aggregation：动机好，但容易重复 QATA 的“权重近似均匀”失败，需要可见性诊断先行。
- Degradation QA training：适合作为 QATA 的修复支线，但主创新强度略弱于 multi-prototype memory。

## 7. 下一步执行建议

不要马上训练。下一步应先做 Multi-prototype CLIP-Memory 的代码审计和最小设计确认：

1. 确认 `generate_cluster_features()` 当前输入输出 shape。
2. 确认 `text_features2` 在 `model/make_model_clipreid.py` 中进入 SSP 和 logits 的 shape。
3. 设计 `[num_classes, K, 512]` memory 如何经过 SSP。
4. 决定第一版是：
   - SSP 前展开 `[C*K, 512]`，再 reshape。
   - 还是 SSP 后对 K 维聚合。
5. 先写 toy CPU 测试，确保 logits `[B, C]` 与 baseline loss 兼容。

推荐最小原则：

- 不动 TMD。
- 不动 dense inference。
- 不动 QATA。
- 不引入外部模型。
- 不做 online memory update。
- 第一版只替换 single mean memory 为 fixed multi-prototype memory。
