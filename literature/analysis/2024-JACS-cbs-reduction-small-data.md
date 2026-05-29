文件重命名：leveraging-limited-experimental-data-with-machine-learning-differentiating-a-methyl-from-an-ethyl-group-in-the-corey.pdf → 2024-JACS-cbs-reduction-small-data.pdf

# Leveraging Limited Experimental Data with Machine Learning: CBS Reduction

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Leveraging Limited Experimental Data with Machine Learning: Differentiating a Methyl from an Ethyl Group in the Corey-Bakshi-Shibata Reduction |
| 作者 | Oliver Pereira / Peter R. Schreiner |
| 期刊 | J. Am. Chem. Soc. 2024, 146, 14576-14586 |
| DOI | 10.1021/jacs.4c01286 |
| 规范化文件名 | 2024-JACS-cbs-reduction-small-data.pdf |
| 开源代码 / 数据集 | 是；dx.doi.org/10.22029/jlupub-17995 |

## §1 核心贡献

- **问题**：CBS reduction of butanone 中区分 methyl/ethyl face，约 100 个高质量三重复实验。
- **方法创新**：用 RDKit/DeepChem/PyG 构建 substrate-catalyst key-intermediate graph，GNN 预测 ΔΔG‡，避免数千 CPU 小时的 DFT 描述符。
- **结果**：发现 ML1 催化剂将 butanone CBS reduction 提升到 80% ee；模型 RMSE 0.12、MAE 0.16 kcal/mol。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：Corey-Bakshi-Shibata reduction of butanone，目标是用手性 oxazaborolidine catalyst 控制羰基还原面选择。
- **预测目标**：ΔΔG‡ 回归，并由符号/大小对应 ee 与主产物构型。
- **立体化学挑战**：butanone 的 methyl/ethyl 差异极小，经典 Corey 模型与 DFT 设计只能把 ee 推到 60-72% 左右。
- **与本项目对比**：同为小数据手性选择性体系，差别在 CBS 只有一个主要反应面选择，而 AldolRxnMaster 同时有 Ca/Cb 两个立体中心和 CIP 翻转。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| ketone substrate | graph | RDKit molecular graph；与 catalyst 通过关键相互作用边连接 | RDKit, DeepChem |
| CBS catalyst | graph | oxazaborolidine scaffold + substituents；B 原子参与 key-intermediate edge | RDKit, DeepChem |
| 反应条件 | 固定条件 | 小规模实验设计中条件基本统一 | 实验设计 |
| 手性信息 | node feature | DeepChem 默认特征含 chirality/CIP code；解释分析显示 CIP code 很重要 | DeepChem, PyG |

### 2.3 特征提取管线

```text
Step 1: substrate/catalyst SMILES → RDKit mol → 标准分子图。
Step 2: 在 catalyst B 与 substrate carbonyl O 之间加 key-intermediate edge → combined graph。
Step 3: DeepChem graph featurizer → atom/bond features，包含 atom type、valence、aromaticity、chirality/CIP code 等。
Step 4: PyTorch Geometric batching → GNN 输入。
最终: graph embedding → ΔΔG‡ regression。
```

### 2.4 模型选择与架构

- **模型类型**：graph neural network with graph attention + global pooling + feed-forward regression head。
- **选择理由**：关键中间体图能直接表达 catalyst-substrate interaction；DFT 描述符耗费约 7000 CPU h 且表现不如图模型。
- **关键超参数**：论文使用 Bayesian optimization 搜索 GNN 超参数；主文重点报告优化框架而非固定一组通用参数。
- **小数据设计**：约 100 条样本，因此使用三重复实验、early stopping、Bayesian HPO、LOOCV。
- **多任务/多输出**：单任务回归 ΔΔG‡。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | random 80/20 train/test |
| 底物外推 | LOOCV excluding substrate/catalyst |
| 时序泄漏 | 非文献时序任务，未做 temporal split |
| CV | LOOCV 与多次 random split |
| 类别不平衡 | 回归任务，不按类别处理 |
| HPO | Bayesian optimization + Gaussian process |

### 2.6 推理输出与后处理

模型输出 ΔΔG‡ 连续值，再转换为 ee 或用于催化剂排序；没有复杂阈值，但实验验证用候选催化剂的预测排序指导合成。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| RMSE | 0.12 kcal/mol | held-out test | Results |
| MAE | 0.16 kcal/mol | held-out test | Results |
| 实验提升 | butanone ee 到 80% | 新 ML1 catalyst，5 次平均 | Results |
| DFT 对比 | 约 7000 CPU h 描述符仍不如图模型 | descriptor baseline | Discussion |
| 与本项目对比 | 小数据质量更高，任务更窄 | Aldol ET Evans TSCV=0.795 | 本项目背景 |

### 2.8 化学价值

这篇论文证明在极小数据下，只要表示能贴近反应关键中间体，ML 可以给出真实可合成的催化剂改进，而不是只做 retrospective benchmark。

## §3 任务与数据对齐

- **任务类型**：ΔΔG‡ 回归；与本项目相似度 4/5。
- **数据规模**：约 100 条，且每条三重复；比本项目更小但质量更高。
- **输入/特征**：关键中间体图把 catalyst boron 与 substrate carbonyl oxygen 连接；节点/边用 DeepChem 默认 graph features，并包含 CIP code。
- **模型**：GNN with graph attention layer；80/20 split；early stopping；Bayesian optimization + Gaussian process；LOOCV 检查 substrate/catalyst 外推。
- **性能**：RMSE 0.12、MAE 0.16 kcal/mol；ML1 平均五次实验 80% ee。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：不是纯 SMILES；RDKit 生成图并显式使用 CIP code node feature。
- **CIP vs 相对构型**：模型目标为 ΔΔG‡ 符号和 ee；解释中 CIP code 是最重要 node feature 之一。
- **局部环境**：key-intermediate graph 直接把反应中心连接起来，优于分离 catalyst/substrate graph。
- **标签噪声**：通过三重复实验和统一条件降低噪声。

## §5 可操作技术建议

#### 建议 A：伪中间体图

| 字段 | 内容 |
|---|---|
| 来自论文 | Figure 1 key-intermediate graph |
| 论文怎么做 | 连接 catalyst boron 和 substrate carbonyl oxygen，构造反应关键中间体图。 |
| 在本项目中怎么用 | 在 `scripts/run_build_graphs.py` 新增 aldol pseudo-TS graph：ketone enolate alpha carbon 与 aldehyde carbonyl carbon 加 forming edge，auxiliary 和 aldehyde 保留原子 CIP 特征。 |
| 预期收益 | GNN 不再学习两个孤立分子，能捕捉 Ca/Cb 生成关系。 |
| 实现难度 | 高 |
| 优先级 | 中 |

#### 建议 B：重复/置信权重

| 字段 | 内容 |
|---|---|
| 来自论文 | 100 reactions run in triplicate |
| 论文怎么做 | 小数据但每条重复，减少测量噪声。 |
| 在本项目中怎么用 | 在 `scripts/run_rebuild_v4.py` 根据同一 reactant/condition 重复记录一致性生成 `sample_weight`；在 `scripts/run_all_models_v4.py` 传入 tree model sample_weight。 |
| 预期收益 | 降低疑似 CIP/文献冲突样本权重。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 C：解释错误反推特征

| 字段 | 内容 |
|---|---|
| 来自论文 | GNNExplainer / GNN-LRP Figure 9 |
| 论文怎么做 | 查看重要节点/边，确认模型关注反应中心和 CIP code。 |
| 在本项目中怎么用 | 对 `scripts/run_chienn_product.py` 或 GNN benchmark 输出错误样本的 atom attribution，重点检查 aldehyde priority atoms。 |
| 预期收益 | 提升对 class 0/3 错误的化学解释。 |
| 实现难度 | 中 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 机制 GNN | 高 | 中 |
| B | 噪声鲁棒 | 中 | 高 |
| C | 错误解释 | 中 | 中 |

## §6 写作借鉴

1. **故事结构**：从“methyl 与 ethyl 如此相似以至经验模型失效”切入，再展示 ML 设计真实催化剂；本项目可用“芳香醛 CIP flip 使经典标签失配”作为同等级痛点。
2. **方法论证**：论文把 DFT、高质量实验、GNN、实验回验串成闭环；AldolRxnMaster 应补充 feature ablation、label audit 和少量人工核验示例。
3. **Benchmark 严格性**：CBS 的数据量小但重复性强；本项目数据更大，应额外报告重复/冲突样本权重。
4. **新文献线索**：Corey 原始 CBS 模型、Gerbig/Schreiner 的 DED 设计工作可作为小数据合成优化背景文献。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 4 | chiral auxiliary/boron reduction 与 aldol 机制相近 |
| 特征工程启示 | 5 | key-intermediate graph 很可迁移 |
| 手性处理启示 | 4 | CIP 作为图特征显式加入 |
| 小数据策略 | 5 | 约 100 条仍成功 |
| 写作/叙事借鉴 | 4 | ML→合成验证很强 |
| 综合优先级 | 4 | 阶段2中优先实现部分思想 |

**一句话总结**：它说明对于少量高质量立体选择性数据，最有效的图表示不是分子图，而是化学家心中的关键中间体图。
