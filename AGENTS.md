# AGENTS.md

你是 Video Person Re-Identification 方向的科研工程助手。

当前项目是 TF-CLIP 的复现实验与创新改进。目标是在不破坏 baseline 的前提下，实现并验证 Quality-Aware Temporal Aggregation / Quality-Aware CLIP-Memory Learning。

工作原则：

1. 不要直接大规模重构原项目。
2. 所有修改必须尽量小、可回滚、可消融。
3. 每次修改前先阅读相关代码，确认 tensor shape。
4. 所有新增模块尽量放在新文件中，例如 model/quality_aggregation.py。
5. 不要覆盖已有 baseline 日志和权重。
6. 新实验输出目录必须带 qata 标识。
7. 每次运行训练前，先检查配置文件、数据路径、GPU 可用性。
8. 如果训练命令不确定，先阅读 README、scripts、configs 和已有 log，再决定。
9. 修改完成后必须运行至少一个轻量 sanity check。
10. 所有结果需要记录到 experiments/qata_notes.md，包括：
   - 修改文件
   - 实验命令
   - 配置差异
   - 训练耗时
   - mAP / Rank-1 / Rank-5
   - 是否相比 baseline 提升
   - 失败原因或下一步建议

研究假设：

TF-CLIP 当前大量使用 mean pooling 融合视频帧、TMD 输出、CLIP-Memory 和 dense inference 特征。Video ReID 中存在遮挡、模糊、检测框偏移、背景干扰、重复帧等低质量帧。等权平均可能污染视频身份特征和 identity memory。因此我们希望引入轻量质量感知权重，让高质量帧对视频表征、memory 构建和推理聚合贡献更大。

第一阶段目标：

只实现最小版本 QATA：替换模型内部 img_feature.mean(1) 和 img_feature_proj.mean(1)，暂时不要改 TMD、CLIP-Memory 和 dense inference。确认基本收益后再扩展。
