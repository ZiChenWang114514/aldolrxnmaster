文件重命名：s41467-025-58854-8.pdf → 2025-NatCommun-meta-learning-selectivity.pdf

# A meta-learning approach for selectivity prediction in asymmetric catalysis

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | A meta-learning approach for selectivity prediction in asymmetric catalysis |
| 作者 | Sukriti Singh / José Miguel Hernández-Lobato |
| 期刊 | Nat. Commun. 2025, 16, 3599 |
| DOI | 10.1038/s41467-025-58854-8 |
| 规范化文件名 | 2025-NatCommun-meta-learning-selectivity.pdf |
| 开源代码 / 数据集 | 是；https://github.com/sukriti243/Meta-learning-for-selectivity-；数据 asymcatml.net |

## §1 核心贡献

- **问题**：用 11,932 条 Ir/Rh/Co AHO 文献数据做 highly enantioselective (%ee>80) 二分类，目标是 few-shot 新任务。
- **方法创新**：Prototypical Networks 从多个 reaction tasks 中学习共享表示，并提出 meta-cluster support selection，避免在测试新反应时额外做 support 实验。
- **结果**：Protonet 64 support AUPRC 0.9117±0.0026；meta-cluster AUPRC 0.9300±0.0011；外部 245 条 2023-2024 反应上 Protonet_cluster 0.9147±0.0022。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：asymmetric hydrogenation of olefins，金属包含 Ir/Rh/Co。
- **预测目标**：binary classification，%ee >80%。
- **立体化学挑战**：不同金属、配体、底物组合形成多个局部任务；随机取 support 不一定贴近 query。
- **与本项目对比**：AldolRxnMaster 的 auxiliary type 与 aldehyde class 可自然定义任务/cluster。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| olefin | Morgan fingerprint | radius 2, 512-bit | RDKit |
| ligand | Morgan fingerprint | radius 2, 512-bit | RDKit |
| solvent | Morgan fingerprint | radius 2, 512-bit | RDKit |
| metal/additive/condition | one-hot + numeric | metal/additive, pressure, temperature, catalyst loading | Python ML stack |
| GNN alternative | learned embedding | MPNN 3 message-passing, GRU, node dim 64, set2set 3 | PyTorch |

### 2.3 特征提取管线

```text
Step 1: 11,932 AHO reactions → component parsing。
Step 2: component fingerprints + condition encodings → 1544d vector。
Step 3: optional GNN embedding also mapped to 1544d。
Step 4: UMAP 1544d → 1048d；k-means cluster = 10/15/20 alternatives。
最终: query 从相似 cluster 抽 support → Protonet high-ee prediction。
```

### 2.4 模型选择与架构

- **模型类型**：Prototypical Networks；baseline RF/GNN；meta-cluster support selection。
- **选择理由**：few-shot 分类中 prototype 可用少量 support 快速形成类别中心。
- **关键超参数**：train support 512/query 64；meta-test support 16/32/64、query 128；cluster 10/15/20。
- **小数据设计**：support set selection 是核心。
- **多任务/多输出**：episode-based high-ee binary classification。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | train/test/validation = 9032/2400/500 |
| 底物外推 | recent/out-of-sample 245 reactions |
| 时序泄漏 | recent test 单独评估 |
| CV | 多 repeats/support sizes |
| 类别不平衡 | AUPRC 作为主指标 |
| HPO | cluster number and support size comparisons |

### 2.6 推理输出与后处理

对 query 先选择 cluster-near support，计算正/负类 prototypes，再输出 high-ee probability；AUPRC 用于评价不平衡二分类。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| dataset | 11,932 reactions | 5009 Ir, 6391 Rh, 532 Co | Methods |
| Protonet support64 | AUPRC 0.9117±0.0026 | main test | Results |
| RF baseline | AUPRC 0.8369±0.0055 | same | Results |
| meta-cluster | AUPRC 0.9300±0.0011 | Dcluster support | Results |
| recent reactions | Protonet support64 AUPRC 0.9341±0.0053 | 245 out-of-sample | Results |

### 2.8 化学价值

这篇论文表明少样本立体选择性预测的关键不是把所有数据混在一起，而是为每个 query 找到最相似、最有信息量的 support。

## §3 任务与数据对齐

- **任务类型**：二分类；与本项目 4-class 相似度 3/5。
- **数据规模**：Ir 5,009；Rh 6,391；Co 532；类别 65% >80% ee。
- **输入表示**：olefin/ligand/solvent Morgan fingerprint radius=2, 512-bit；metal/additive OHE；pressure/temperature/S/C ratio；总 1544d。GNN 替代表示同样 1544d。
- **模型**：Protonet；baseline RF/GNN；UMAP 1544→1048 + k-means for meta-cluster。
- **评估**：random tasks、cluster-based split、out-of-sample 245 reactions。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：不重点处理；任务是 ee threshold。
- **CIP vs 相对构型**：不涉及 absolute R/S。
- **局部环境**：fingerprint/GNN 为通用表示，无专门 chirality descriptor。
- **标签噪声**：用 AUPRC 处理类别不平衡；无噪声建模。

## §5 可操作技术建议

#### 建议 A：Meta-cluster支持集

| 字段 | 内容 |
|---|---|
| 来自论文 | Fig. 5 meta-cluster |
| 论文怎么做 | 对反应表示聚类，测试 query 的 support 从相同 cluster 的训练数据取。 |
| 在本项目中怎么用 | 在 `scripts/run_protonet.py` 中对 `data/features_v4/v4_features.csv` 用 UMAP/k-means 聚类；对非 Evans query 选相同 cluster 的 Evans/全数据 support。 |
| 预期收益 | 不需要新增实验即可改善少样本子集预测。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 B：AUPRC补充指标

| 字段 | 内容 |
|---|---|
| 来自论文 | AUPRC for imbalanced %ee threshold |
| 论文怎么做 | 用 AUPRC 而非 accuracy 评价不平衡二分类。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 对 Ca/Cb 二分类和每个 one-vs-rest class 输出 AUPRC、macro F1。 |
| 预期收益 | 更清楚衡量 class 0/3 改善，而非被 majority class 掩盖。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 C：任务定义实验

| 字段 | 内容 |
|---|---|
| 来自论文 | random/cluster/out-of-sample tasks |
| 论文怎么做 | 多种 task construction 比较 meta-learning 表现。 |
| 在本项目中怎么用 | 比较 `auxiliary_type`、`aldehyde_aromaticity`、`source_year`、`feature_cluster` 四种 task 定义，保留最稳定者。 |
| 预期收益 | 找到最适合 AldolRxnMaster 的迁移单位。 |
| 实现难度 | 中 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 少样本支持集选择 | 中 | 高 |
| B | 更敏感评估 | 低 | 高 |
| C | 迁移任务选择 | 中 | 中 |

## §6 写作借鉴

1. **故事结构**：从 high-ee screening 的实际需求切入，再展示少量 support 即可超过传统模型；本项目可转化为“新辅基只有几十条也能适配”。
2. **方法论证**：AUPRC、support size、cluster sensitivity 三者构成完整 benchmark；本项目可加入 macro F1/AUPRC 而非只看 accuracy。
3. **Benchmark 严格性**：recent out-of-sample test 很有价值；AldolRxnMaster 可按文献年份拆分 newest aldol examples。
4. **新文献线索**：Prototypical Networks、UMAP/k-means support selection、asymcatml 数据平台。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 3 | 不预测 R/S |
| 特征工程启示 | 3 | 标准反应 fingerprint |
| 手性处理启示 | 2 | 手性处理弱 |
| 小数据策略 | 5 | few-shot 实验完整 |
| 写作/叙事借鉴 | 4 | literature sparse data 叙事强 |
| 综合优先级 | 4 | 适合非 Evans 泛化 |

**一句话总结**：该文应作为 AldolRxnMaster 跨 auxiliary 少样本迁移和 meta-cluster support selection 的主要参考。
