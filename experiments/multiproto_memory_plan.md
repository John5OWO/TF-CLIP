# Multi-prototype CLIP-Memory Feasibility Check

日期：2026-06-10  
当前分支：`exp-multiproto-memory`  
约束：未修改核心代码，未修改配置文件，未启动训练，未运行 GPU 任务。

## 0. 分支与工作区状态

已执行：

```bash
git branch
git status
```

确认结果：

- 当前分支：`exp-multiproto-memory`
- 远端状态：up to date with `origin/exp-multiproto-memory`
- 当前未跟踪文件：`experiments/next_idea_plan.md`

本次只新增本设计文档：`experiments/multiproto_memory_plan.md`。

## A. 当前 cluster_features 构建路径

### A1. generate_cluster_features() 如何生成 cluster_features

位置：`processor/processor_clipreid_stage2.py:50-64`

当前逻辑：

```python
centers = defaultdict(list)
for i, label in enumerate(labels):
    if label == -1:
        continue
    centers[labels[i]].append(features[i])

centers = [
    torch.stack(centers[idx], dim=0).mean(0)
    for idx in sorted(centers.keys())
]
centers = torch.stack(centers, dim=0)
```

含义：

- 输入 `features` 是每个训练 sequence/tracklet 的 sequence-level CLIP image feature。
- 按 person ID 分组。
- 每个 ID 内做等权 mean。
- 最后按 `sorted(centers.keys())` 堆叠为 memory。

### A2. cluster_features 的 shape

构建前：

- `image_features_list`: `[N, 512]`
- MARS 当前 `N = 8298` train tracklets。

构建后：

- `cluster_features`: `[num_classes, 512]`
- MARS 当前 `num_classes = 625`
- 即 `[625, 512]`

相关代码：

- `processor/processor_clipreid_stage2.py:189-192`
- `datasets/make_dataloader_clipreid.py:143`

### A3. labels / person ID 如何对应 num_classes

MARS 数据集在 `datasets/set/mars.py` 中 relabel：

- `pid2label = {pid: label for label, pid in enumerate(pid_list)}`
- tracklet 保存为 `(img_paths, int(pid), int(camid), 1)`

相关位置：

- `datasets/set/mars.py:151-163`
- `datasets/set/mars.py:177-199`

因此 MARS train labels 是连续的 `0..624`，与 `num_classes=625` 对齐。

风险点：

- `generate_cluster_features()` 当前只按出现的 sorted labels 堆叠，没有显式检查 `len(centers) == num_classes`。
- 对 MARS 正常成立；如果未来扩展到其他数据集或过滤 ID，建议加 assert。

### A4. 是否每个 ID 有多个 sequence feature

是。MARS train 有 625 个 ID、8298 个 tracklets。每个 tracklet 通过 `model(img, get_image=True)` 得到一个 `[512]` sequence feature。

数据来自：

- `train_loader_stage1 = VideoDataset(dataset.train, sample='dense')`
- `batch_size=1`
- 每个 dataset item 对应一个 tracklet。

相关位置：

- `datasets/make_dataloader_clipreid.py:140-168`
- `processor/processor_clipreid_stage2.py:163-192`

### A5. 每个 ID 平均/最少/最多多少 sequence feature

从 `/data1/lgf/TF-CLIP/data/Mars/split_train.json` 读取 metadata，不加载图像、不使用 GPU：

```text
num_ids = 625
num_tracklets = 8298
avg_tracklets_per_id = 13.2768
min_tracklets_per_id = 1
max_tracklets_per_id = 271
ids_eq_1 = 1
ids_lt_4 = 45
```

分布：

| tracklets per ID | # IDs |
|---:|---:|
| 1 | 1 |
| 2 | 16 |
| 3 | 28 |
| 4 | 35 |
| 5-9 | 213 |
| 10-19 | 233 |
| 20-49 | 89 |
| >=50 | 10 |

结论：

- `K=2` 对绝大多数 ID 可行。
- 但少量 ID 样本很少，必须有 fallback。
- `K=3` 第一轮风险更高，因为 45 个 ID 少于 4 个 tracklets。

## B. 当前 cluster_features 使用路径

### B1. cluster_features 在 do_train_stage2() 中传到哪里

位置：`processor/processor_clipreid_stage2.py:220`

```python
score, feat, logits1 = model(
    x=img,
    cam_label=target_cam,
    view_label=target_view,
    text_features2=cluster_features
)
```

其中 `logits1` 就是 I2T / image-to-memory logits，后续传给 loss：

```python
loss1 = loss_fn(score1, feat, target, target_cam, logits1)
```

位置：`processor/processor_clipreid_stage2.py:225-228`

### B2. model forward 哪个分支使用 text_features2

位置：`model/make_model_clipreid.py:321-333`

仅 training 分支使用 `text_features2`：

```python
text_features2 = text_features2.unsqueeze(0).expand(B, -1, -1)
image_features_proj_raw2 = image_features_proj_raw.view(B, T, -1, image_features_proj_raw.shape[-1])
video_feature_project = image_features_proj_raw2.mean(1)
text_features2 = text_features2 + self.SSP(text_features2, video_feature_project)
logits = torch.einsum("bd,bkd->bk", img_feature_proj, text_features2)
```

当前 shape：

- 输入 `text_features2`: `[num_classes, 512]`
- expand 后：`[B, num_classes, 512]`
- `video_feature_project`: `[B, 512]`
- SSP 输出：应为 `[B, num_classes, 512]`
- `img_feature_proj`: `[B, 512]`
- `logits`: `[B, num_classes]`

### B3. i2tscore 在哪里计算

位置：`model/make_model_clipreid.py:327`

```python
logits = torch.einsum("bd,bkd->bk", img_feature_proj, text_features2)
```

返回位置：`model/make_model_clipreid.py:333`

```python
return [...], [img_feature, img_feature_proj, cls_f_tp], logits
```

在 processor 中命名为 `logits1`。

### B4. i2tscore 当前 shape 是否是 [B, num_classes]

是。

当前 `einsum("bd,bkd->bk")` 明确输出 `[B, K_class]`，这里的 `k` 是 class 维，不是 prototype 维。

### B5. loss/make_loss.py 是否要求 i2tscore 必须是 [B, num_classes]

是。位置：`loss/make_loss.py:56-62`

```python
I2TLOSS = xent(i2tscore, target)
```

`CrossEntropyLabelSmooth(num_classes=num_classes)` 预期输入是 `[B, num_classes]`，target 是 `[B]`。

因此 Multi-prototype 第一版必须在进入 loss 前聚合 prototype 维，保持：

```text
i2tscore: [B, num_classes]
```

这样 `loss/make_loss.py` 可以不改。

### B6. frame loss / I2T loss / SSP loss 依赖哪些 tensor shape

当前没有单独显式 SSP loss，SSP 是 model 内对 memory feature 的 prompt-style adjustment，I2T loss 作用在 SSP 后 logits 上。

训练 loss 依赖：

1. ID loss：
   - `score1 = score[0:3]`
   - `score[0]`: `[B, num_classes]`
   - `score[1]`: `[B, num_classes]`
   - `score[2]`: `[B, num_classes]`
   - 位置：`processor/processor_clipreid_stage2.py:222-228`

2. Frame loss：
   - `score2 = score[3]`
   - `targetX`: `[B*T]`
   - `score2`: `[B*T, num_classes]`
   - 位置：`processor/processor_clipreid_stage2.py:230-238`

3. Triplet loss：
   - `feat`: list
   - `[img_feature, img_feature_proj, cls_f_tp]`
   - shape 大致为 `[B,768]`, `[B,512]`, `[B,768]`
   - 位置：`loss/make_loss.py:47-51`

4. I2T loss：
   - `i2tscore/logits1`: `[B, num_classes]`
   - 位置：`loss/make_loss.py:56-62`

Multi-prototype 最小设计只应改变 I2T logits 的内部计算，不改变返回给 loss 的最终 shape。

## C. Multi-prototype 最小可行设计

目标：

```text
cluster_features: [num_classes, 512]
```

扩展为：

```text
cluster_features: [num_classes, K, 512]
```

但最终保持：

```text
i2tscore: [B, num_classes]
```

### C1. Memory 构建

新建一个可开关函数：

```python
build_multi_prototype_memory(labels, features, num_classes, k, mode)
```

输入：

- `labels`: `[N]`
- `features`: `[N, 512]`
- `num_classes`: int
- `k=2`

输出：

- baseline disabled: `[num_classes, 512]`
- multi enabled: `[num_classes, K, 512]`

### C2. Model 内相似度计算

当 `text_features2.dim() == 2`：

```python
text_features2 = text_features2.unsqueeze(0).expand(B, -1, -1)
text_features2 = text_features2 + self.SSP(text_features2, video_feature_project)
logits = torch.einsum("bd,bkd->bk", img_feature_proj, text_features2)
```

保持 baseline。

当 `text_features2.dim() == 3`，即 `[C, K, 512]`：

推荐最小兼容设计：

1. 展开 prototype：

```text
[C, K, 512] -> [C*K, 512]
```

2. 按 baseline 方式进入 SSP：

```text
[C*K, 512] -> [B, C*K, 512]
```

3. 计算 prototype logits：

```text
sim_flat = einsum("bd,bpd->bp") -> [B, C*K]
sim = sim_flat.view(B, C, K)
```

4. K 维聚合：

```python
if AGG_MODE == "max":
    logits = sim.max(dim=2).values
elif AGG_MODE == "logsumexp":
    logits = tau * torch.logsumexp(sim / tau, dim=2)
```

5. 返回 `[B, C]` 给原 loss。

优点：

- `loss/make_loss.py` 不需要改。
- SSP 仍可复用，只是把 prototype 当作更多 memory tokens。
- baseline path 可以通过 `MULTI_ENABLED=False` 完全保留。

注意：

- `logsumexp` 会随 K 增加引入常数偏移。对 fixed K=2 问题不大；如需严格归一，可用 `tau * (logsumexp(sim/tau) - log(K))`，使其接近 smooth max 而不是增加类别常数。

## D. Prototype 构建方式比较

### 方案 1：KMeans within each ID

做法：

- 每个 ID 内对 `[n_i, 512]` sequence features 做 KMeans。
- K=2。
- 每个 cluster 的 mean 作为 prototype。
- `n_i < K` 时 duplicate mean 或 duplicate available feature。

优点：

- 最符合 multi-modal memory 的直觉。
- 能利用 ID 内所有 tracklets。
- 对大 ID，能分出视角/摄像头/外观模式。

缺点：

- 需要实现 torch kmeans 或引入 sklearn。
- 每个 ID 聚类一次，有 CPU 开销。
- 小样本 ID 聚类不稳定。
- 初始化和随机性需要固定 seed。

适合程度：

- 适合作为正式版本，但不一定适合作为第一版落地。

### 方案 2：Farthest-two prototypes

做法：

- 每个 ID 先算 mean。
- 找离 mean 最远的一个 feature `a`。
- 再找离 `a` 最远的一个 feature `b`。
- 用 `a` 和 `b` 作为两个 seed。
- 可选：按更近 seed 分配样本，再求两个 cluster mean。

推荐第一版使用“farthest seed + one-step assignment mean”，而不是直接用两个单样本：

```text
seed_a, seed_b -> assign all features -> proto_a_mean, proto_b_mean
```

优点：

- 不依赖 sklearn。
- deterministic，方便复现。
- 计算量小。
- 能显式拉开两个 prototype，比 random split 更有意义。
- 对 K=2 非常自然。

缺点：

- 容易选到 outlier。
- 如果 ID 内特征分布是单峰，两个 prototype 可能只是噪声边界。
- 需要记录 prototype separation 和 cluster size，判断是否合理。

适合程度：

- 最适合作为第一版。

### 方案 3：Random split mean

做法：

- 每个 ID 内随机打乱 tracklets。
- 分成两组，各自求 mean。
- 样本不足时 fallback。

优点：

- 实现最简单。
- 可作为 sanity baseline。
- 能判断收益是否只是来自 K 维度容量增加，而不是结构化 prototype。

缺点：

- 不具备明确语义。
- 方差较大，需要固定 seed 或多次重复。
- 如果它也涨，说明 multi-prototype 可能只是 regularization/capacity；如果它不涨而 farthest/kmeans 涨，说明结构化分裂有效。

适合程度：

- 不建议作为第一版主实验，但应该作为 ablation。

### 第一版推荐

第一版推荐：

```text
CLUSTER_MODE = "farthest"
NUM_PROTOTYPES = 2
AGG_MODE = "logsumexp" first, "max" second
```

原因：

- 不引入 sklearn。
- deterministic，便于 debug。
- K=2 下实现直接。
- 相比 random split 更有结构假设。
- 相比 KMeans 更小改动，更适合 feasibility-first。

KMeans 建议作为第二版，等 farthest 证明 shape 和训练流程可行后再实现。

## E. 配置设计

建议新增独立 memory 配置，不复用 `MODEL.QATA`：

```python
MODEL.MEMORY = CN()
MODEL.MEMORY.MULTI_ENABLED = False
MODEL.MEMORY.NUM_PROTOTYPES = 2
MODEL.MEMORY.CLUSTER_MODE = "farthest"
MODEL.MEMORY.AGG_MODE = "logsumexp"
MODEL.MEMORY.AGG_TEMP = 0.07
MODEL.MEMORY.MIN_SAMPLES_PER_PROTO = 1
MODEL.MEMORY.NORMALIZE_PROTOTYPES = True
MODEL.MEMORY.LOG_STATS = True
MODEL.MEMORY.STATS_FILE = "multiproto_memory_stats.txt"
```

用户给出的必需项：

```python
MODEL.MEMORY.MULTI_ENABLED = False
MODEL.MEMORY.NUM_PROTOTYPES = 2
MODEL.MEMORY.CLUSTER_MODE = "kmeans" | "farthest" | "random_split"
MODEL.MEMORY.AGG_MODE = "max" | "logsumexp"
MODEL.MEMORY.AGG_TEMP = 0.07
MODEL.MEMORY.MIN_SAMPLES_PER_PROTO = 1
MODEL.MEMORY.NORMALIZE_PROTOTYPES = True
```

第一轮实验必须：

```yaml
MODEL:
  QATA:
    ENABLED: False
  MEMORY:
    MULTI_ENABLED: True
    NUM_PROTOTYPES: 2
```

理由：

- QATA 阶段已结束，不能让 feature aggregation 与 memory 改动混在一起。
- 第一轮要隔离 Multi-prototype CLIP-Memory 本身收益。

## F. 最小实验矩阵

第一轮只跑 MARS。

| Order | Dataset | Method | Config intent | Purpose |
|---:|---|---|---|---|
| 1 | MARS | Baseline single prototype | existing `configs/vit_clipreid.yml` | reference: 88.9 / 93.0 / 98.1 |
| 2 | MARS | Multi-proto K=2 + logsumexp | `MULTI_ENABLED=True`, `K=2`, `CLUSTER_MODE=farthest`, `AGG_MODE=logsumexp` | 稳定 smooth aggregation |
| 3 | MARS | Multi-proto K=2 + max | `MULTI_ENABLED=True`, `K=2`, `CLUSTER_MODE=farthest`, `AGG_MODE=max` | hard prototype selection 对比 |
| 4 | MARS | Random split K=2 + best agg | `CLUSTER_MODE=random_split` | sanity baseline，验证不是 K 维度本身 |
| 5 | iLIDS | Best MARS variant | only if MARS improves | 跨数据集验证 |

暂时不要：

- 不叠加 Residual QATA。
- 不做 K=3。
- 不做 online memory update。
- 不做 feature + memory 组合。
- 不做 iLIDS，除非 MARS 至少不损 Rank-1/Rank-5 且 mAP 有正信号。

建议成功判据：

- MARS mAP 至少达到 baseline `88.9`，最好超过 `89.1`。
- Rank-1 不低于 `93.0`。
- Rank-5 不明显低于 `98.1`，最多允许 `-0.1/-0.2` 作为观察。

## G. 风险分析

### G1. 哪些代码位置最容易出 shape bug

1. `processor/processor_clipreid_stage2.py`
   - 当前 `generate_cluster_features()` 返回 `[C,512]`。
   - Multi-proto 返回 `[C,K,512]` 后，必须保证旧 path disabled 时仍返回 `[C,512]`。

2. `model/make_model_clipreid.py:322`
   - 当前代码固定假设 `text_features2.unsqueeze(0).expand(B,-1,-1)`。
   - 对 `[C,K,512]` 不能直接这样 expand 后进入旧 `einsum`。
   - 需要显式分支处理 `dim()==3`。

3. `model/make_model_clipreid.py:325`
   - SSP 当前输入 `[B,C,512]`。
   - 如果使用 `[B,C*K,512]` 展开，SSP 应可复用，但要确认内部是否依赖 class count。

4. `model/make_model_clipreid.py:327`
   - 当前 `einsum("bd,bkd->bk")` 输出 `[B,C]`。
   - Multi-proto 需要先输出 `[B,C*K]`，reshape `[B,C,K]`，再聚合 `[B,C]`。

5. `loss/make_loss.py:56-62`
   - loss 只接受最终 `[B,C]`。
   - 如果误传 `[B,C,K]` 会直接 shape 错或语义错。

### G2. KMeans 是否会引入 CPU 开销

会，但可控。

MARS：

- 625 IDs。
- 8298 train tracklets。
- 平均每 ID 13.28 features。
- feature dim 512。

K=2 的 ID-local KMeans 很小，理论开销不大。但如果用 sklearn，会引入依赖和 CPU/GIL/数据搬运问题；如果用 torch 实现，可保持依赖简单。

第一版建议 farthest，避免把 feasibility 阶段复杂化。

### G3. K=2 是否可能过拟合

可能，但风险中等。

原因：

- 少样本 ID 中，两个 prototype 可能记住单个 tracklet。
- `max` 聚合可能鼓励每个样本只对某个 prototype 过强匹配。
- 对 iLIDS 这类小数据集风险更高。

缓解：

- prototype 用 cluster mean，不直接用单个样本。
- 少样本 ID fallback duplicate mean。
- 第一轮只 MARS。
- 记录 cluster size 和 prototype separation。

### G4. max 聚合是否 hard assignment 过强

是。

`max_k sim` 的优点是能让每个 query 匹配最接近的 identity mode；缺点是：

- 梯度只来自 winner prototype。
- 容易过度依赖 noisy prototype。
- 对 hard negative 可能放大错误相似度。

因此 max 适合作为对比实验，不建议作为第一优先。

### G5. logsumexp 是否更稳定

更稳定。

`logsumexp` 是 smooth max：

```text
i2tscore = tau * logsumexp(sim / tau, dim=K)
```

优点：

- 两个 prototype 都有梯度贡献。
- 比 max 更不容易被单个 noisy prototype 支配。
- `tau=0.07` 与 CLIP similarity temperature 直觉接近。

注意：

- 可考虑归一化版本：

```text
i2tscore = tau * (logsumexp(sim / tau, dim=K) - log(K))
```

避免 K 增加带来的常数偏移。虽然同一类别 K 固定时 CE 对类别间相对值影响有限，但归一化更严谨。

### G6. 如果结果不涨，如何判断是 prototype 构建失败还是假设不成立

必须记录诊断统计：

1. Prototype separation：
   - 每 ID 两个 prototype 的 cosine distance。
   - 如果大多数 separation 接近 0，说明构建没有产生多模态。

2. Cluster size balance：
   - 每 ID 两个 prototype 的样本数。
   - 如果大量为 `1 vs n-1`，可能只是 outlier split。

3. Prototype usage：
   - 训练 batch 中每个 sample 的 winner prototype 分布。
   - 如果几乎永远选 proto 0，proto 1 无效。

4. Hard ID analysis：
   - baseline 错误 ID 是否在 multi-proto 中改善。
   - 如果 only easy IDs 改善，意义有限。

5. Random split baseline：
   - 如果 random split 也涨，说明收益可能来自容量/regularization，不是结构化 prototype。
   - 如果 farthest/kmeans 涨而 random split 不涨，说明多模态构建有效。

6. K=1 compatibility：
   - Multi-proto path 设置 K=1 时必须复现 single mean baseline。
   - 如果 K=1 都不一致，说明实现引入了额外变量。

判断标准：

- 如果 prototype separation 很小，构建失败或 ID 内确实单峰。
- 如果 separation 大但性能不涨，可能 aggregation/loss 使用方式不合适。
- 如果 separation 大、usage 均衡、random split 不涨、structured split 涨，则 multi-prototype 假设成立。

## H. 最小实现建议顺序

虽然本阶段不写代码，但后续建议按以下顺序实现：

1. 新增 `MODEL.MEMORY` 配置，默认全关。
2. 新增纯函数：
   - `build_multi_prototype_memory()`
   - `aggregate_multi_prototype_logits()`
3. 写 CPU toy tests：
   - baseline `[C,512]` path 不变。
   - multi `[C,K,512]` 聚合后 logits `[B,C]`。
   - K=1 等价 baseline。
   - single-sample ID duplicate mean。
4. 修改 `generate_cluster_features()`，保持 disabled 完全原逻辑。
5. 修改 `make_model_clipreid.py` training 分支，仅处理 `text_features2.dim()==3`。
6. 运行 py_compile 和 toy forward。
7. 再考虑完整训练。

第一版推荐配置：

```yaml
MODEL:
  QATA:
    ENABLED: False
  MEMORY:
    MULTI_ENABLED: True
    NUM_PROTOTYPES: 2
    CLUSTER_MODE: "farthest"
    AGG_MODE: "logsumexp"
    AGG_TEMP: 0.07
    MIN_SAMPLES_PER_PROTO: 1
    NORMALIZE_PROTOTYPES: True
```

## I. 结论

Multi-prototype CLIP-Memory 在当前 TF-CLIP 代码中是可行的，且可以做到不改 loss：

- memory 从 `[C,512]` 扩为 `[C,K,512]`
- model 内部计算 `[B,C,K]`
- K 维聚合回 `[B,C]`
- 原 `loss/make_loss.py` 保持不变

第一版最推荐：

- `K=2`
- `CLUSTER_MODE=farthest`
- `AGG_MODE=logsumexp`
- `QATA.ENABLED=False`
- 只跑 MARS

最主要风险是 shape 分支和 SSP 输入维度处理。只要保证最终 `i2tscore=[B,num_classes]`，这个方向可以作为下一阶段主线推进。
