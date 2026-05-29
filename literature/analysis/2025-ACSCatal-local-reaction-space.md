文件重命名：evaluating-predictive-accuracy-in-asymmetric-catalysis-a-machine-learning-perspective-on-local-reaction-space.pdf → 2025-ACSCatal-local-reaction-space.pdf

# Evaluating Predictive Accuracy in Asymmetric Catalysis: A Machine Learning Perspective on Local Reaction Space

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Evaluating Predictive Accuracy in Asymmetric Catalysis: A Machine Learning Perspective on Local Reaction Space |
| 作者 | Isaiah O. Betinol / Jolene P. Reid |
| 期刊 | ACS Catal. 2025, 15, 6067-6077 |
| DOI | 10.1021/acscatal.5c01051 |
| 规范化文件名 | 2025-ACSCatal-local-reaction-space.pdf |
| 开源代码 / 数据集 | 是；Supporting Information ZIP repository |

## §1 核心贡献

- **问题**：评估 asymmetric catalysis ML 中预测准确率到底来自局部邻居还是全局多样性。
- **方法创新**：提出 RaRFRegression，用 ECFP4 2048-bit reaction fingerprints + Jaccard/Tanimoto radius，为每个测试点动态选择局部训练集。
- **结果**：多数 case 中局部模型用远少于全训练集的数据即可达到相近 MAE；无直接邻居时，额外多样性最多改善约 0.25-0.5 kcal/mol。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：8 个 asymmetric catalysis case studies，覆盖 HTE 与文献数据。
- **预测目标**：ΔΔG‡ 回归，由 enantioselectivity 转换。
- **立体化学挑战**：模型性能高度依赖测试点附近是否有相似训练反应，随机划分可能高估全局泛化。
- **与本项目对比**：AldolRxnMaster 的 Evans-only TSCV 高而 full-data TSCV 低，可能正是 local reaction space 覆盖差异。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| reaction components | concatenated fingerprint | component ECFP4, 2048 bits | RDKit |
| similarity | Jaccard/Tanimoto distance | radius threshold selects neighbors | RaRFRegression |
| model input | local subset | each query has own training subset | scikit-learn RF |
| output | ΔΔG‡ | RT ln(er) transformation | 自定义 |

### 2.3 特征提取管线

```text
Step 1: asymmetric catalysis datasets → components standardized。
Step 2: each component → ECFP4 2048-bit fingerprint。
Step 3: concatenate reaction fingerprint → compute Tanimoto/Jaccard distance。
Step 4: for each test point select training points within radius r。
最终: train local RF → predict ΔΔG‡ for that query。
```

### 2.4 模型选择与架构

- **模型类型**：Radius-adaptive Random Forest Regression。
- **选择理由**：隔离 local neighborhood coverage 对预测性能的贡献。
- **关键超参数**：radius sweep；random forest baseline；kReduction diversity comparison。
- **小数据设计**：直接研究小训练子集/局部邻域大小的影响。
- **多任务/多输出**：每个 case study 单独回归。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | random 80/20、LORO/LOCO、kReduction |
| 底物外推 | LORO/LOCO 明确评估 component holdout |
| 时序泄漏 | 非核心 |
| CV | 多 case study / split 对比 |
| 类别不平衡 | 回归任务 |
| HPO | radius and RF settings comparison |

### 2.6 推理输出与后处理

每个测试点先生成局部训练集，再训练局部 RF 输出 ΔΔG‡；同时邻域大小和距离天然提供 confidence diagnostic。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| random MAE | A 0.26 vs control 0.27 | case A | Table 1 |
| random MAE | F 0.27 vs control 0.32 | case F | Table 1 |
| radius 0.6 | 通常 <20% training data 即可接近全数据 | random split | Results |
| LORO/LOCO | D 0.75 vs control 0.54, F 0.57 vs 0.46 | holdout split | Table 2 |
| diversity sampling | 小训练集可差到约 2 kcal/mol | kReduction | Results |

### 2.8 化学价值

该工作把“模型是否学到机制”转化为可度量的 local coverage 问题。对本项目，每个错误样本都应报告最近邻距离和同辅基邻居数。

## §3 任务与数据对齐

- **任务类型**：ΔΔG‡ 回归评估框架；与本项目相似度 4/5。
- **数据**：8 个 asymmetric catalysis case studies，240 到 12,619 条，含文献与 HTE 数据。
- **特征**：每个 reaction component 计算 ECFP4, nbits=2048，拼接为 reaction fingerprint；Jaccard distance 定义局部半径。
- **模型**：radius-selected Random Forest；baseline RF/NN/kNN/full training；kReduction diversity baseline。
- **评估**：80/20 random split；LORO/LOCO；MAE。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：使用 2D fingerprints，不重点讨论 `@/@@`。
- **CIP vs 相对构型**：预测 ΔΔG‡，不直接处理 R/S。
- **局部环境**：核心是 reaction-space neighborhood，而非 stereocenter local feature。
- **标签噪声**：通过局部邻域判断训练代表性，间接识别 OOD。

## §5 可操作技术建议

#### 建议 A：局部邻居诊断

| 字段 | 内容 |
|---|---|
| 来自论文 | RaRFRegression Figure 2 |
| 论文怎么做 | 对每个测试点只用 radius 内训练样本训练 RF。 |
| 在本项目中怎么用 | 新增 `scripts/run_local_space_v4.py`：基于 v4 features 或 Morgan fingerprint 计算每个测试样本最近邻距离、邻居数、local RF prediction。 |
| 预期收益 | 解释 Evans-only 0.795 vs 全数据 0.624 是局部密度问题还是模型问题。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 B：失败样本邻域表

| 字段 | 内容 |
|---|---|
| 来自论文 | Figure 5 local vs full predictions |
| 论文怎么做 | 比较 local model 和 full model，每点都有邻居数/半径。 |
| 在本项目中怎么用 | 对每个错误预测输出 `nearest_train_distance`、`n_neighbors_r05`、`same_aux_neighbors` 到 `results/tables/error_neighborhood_v4.csv`。 |
| 预期收益 | 判断 class 0/3 错误是否因训练集中缺少类似芳香醛。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 C：目标导向采样

| 字段 | 内容 |
|---|---|
| 来自论文 | kReduction vs RaRF Figure 9 |
| 论文怎么做 | 局部覆盖优于盲目最大化 diversity。 |
| 在本项目中怎么用 | 未来补数据时优先围绕错误/OOD aldehyde scaffold 采样，而不是平均扩展所有 auxiliary。 |
| 预期收益 | 更少实验提升模型局部可靠性。 |
| 实现难度 | 低 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 泛化机制诊断 | 中 | 高 |
| B | 错误解释 | 低 | 高 |
| C | 数据补充策略 | 低 | 中 |

## §6 写作借鉴

1. **故事结构**：把“准确率高”拆成“局部空间是否被覆盖”；本项目可用这个框架解释 Evans-only 与 full-data 差距。
2. **方法论证**：建议在论文结果中增加 distance-to-training 和 local neighbor count，而不仅是整体 TSCV。
3. **Benchmark 严格性**：LORO/LOCO 比 random split 更能暴露泛化问题；AldolRxnMaster 可做 aldehyde/auxiliary holdout。
4. **新文献线索**：Reid group asymmetric catalysis descriptor/ML benchmark 系列可作为评价框架背景。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 4 | 直接解释局部反应空间问题 |
| 特征工程启示 | 3 | reaction fingerprint 简单有效 |
| 手性处理启示 | 2 | 不处理 CIP |
| 小数据策略 | 5 | 局部小训练集思想很强 |
| 写作/叙事借鉴 | 4 | 可解释全局泛化限制 |
| 综合优先级 | 4 | 应实现 local diagnostics |

**一句话总结**：这篇论文为本项目全数据性能下降提供了最好的诊断框架：先问测试样本附近有没有训练邻居，再谈模型复杂度。
