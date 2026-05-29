文件重命名：Angew Chem Int Ed - 2025 - Singh - Bayesian Meta‐Learning for Few‐Shot Reaction Outcome Prediction of Asymmetric.pdf → 2025-AngewChem-bayesian-meta-learning.pdf

# Bayesian Meta-Learning for Few-Shot Reaction Outcome Prediction of Asymmetric Hydrogenation of Olefins

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Bayesian Meta-Learning for Few-Shot Reaction Outcome Prediction of Asymmetric Hydrogenation of Olefins |
| 作者 | Sukriti Singh / José Miguel Hernández-Lobato |
| 期刊 | Angew. Chem. Int. Ed. 2025, 64, e202503821 |
| DOI | 10.1002/anie.202503821 |
| 规范化文件名 | 2025-AngewChem-bayesian-meta-learning.pdf |
| 开源代码 / 数据集 | 代码：https://github.com/sukriti243/Bayesian-meta-learning-for-reaction-outcome-prediction；数据需 asymcatml.net 许可 |

## §1 核心贡献

- **问题**：在约 12,000 条 AHO 文献数据上，用少量 support examples 预测新反应是否 highly enantioselective。
- **方法创新**：比较 DKT、ADKF，并提出 ADKF-prior，将 deep kernel GP 与 meta-learned Gaussian prior 结合以改善 low-data。
- **结果**：time-based split 中 ADKF-prior 仅 8 个 support examples 达 AUPRC 0.9062±0.0221，超过 DKL full-training 0.8782±0.0150、RF 0.8504、GNN 0.8558。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：Ir/Rh asymmetric hydrogenation of olefins。
- **预测目标**：binary classification，%ee >80% 视为 high-selectivity positive。
- **立体化学挑战**：新底物/配体组合可用数据极少，需要从相关任务快速迁移。
- **与本项目对比**：AldolRxnMaster 的非 Evans 辅基就是典型 few-shot target；可从 Evans 大子集迁移。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| olefin | Morgan fingerprint | radius 2, 512-bit | RDKit |
| ligand | Morgan fingerprint | radius 2, 512-bit | RDKit |
| solvent | Morgan fingerprint | radius 2, 512-bit | RDKit |
| metal/additive/condition | one-hot + numeric | metal/additive one-hot；temperature/pressure/catalyst loading | scikit-learn/PyTorch |
| total | feature vector | 1544d | 自定义 |

### 2.3 特征提取管线

```text
Step 1: AHO dataset → reaction components separated。
Step 2: olefin/ligand/solvent → Morgan radius 2 512-bit each。
Step 3: metal/additive one-hot + temp/pressure/loading numeric。
Step 4: concatenate → 1544d vector；GNN alternative 也转成 1544d embedding。
最终: meta-learning task support/query batches。
```

### 2.4 模型选择与架构

- **模型类型**：DKT、ADKF、ADKF-prior；baseline Protonet、RF、GNN、single-task DKL。
- **选择理由**：Bayesian GP head 提供 few-shot uncertainty；ADKF-prior 用任务先验改善极小 support。
- **关键超参数**：support sizes 5/8/10/16/32/64/128；query 128；meta-training batch 5 tasks。
- **小数据设计**：核心就是 few-shot support adaptation。
- **多任务/多输出**：多任务 episode 学习，单任务输出 high-ee probability。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | random、substrate-based、time-based splits |
| 底物外推 | substrate split |
| 时序泄漏 | time split，2023-2024 作为 recent test |
| CV | 多 support size/repeats |
| 类别不平衡 | high-ee binary 约 65/35 |
| HPO | meta-learning settings and baseline tuning |

### 2.6 推理输出与后处理

给定少量 support reactions，模型输出 query 为 high-ee 的概率；Bayesian 方法同时给 uncertainty，可用于优先实验或低置信样本标记。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| dataset | ~12,000 reactions | AHO Ir/Rh | Methods |
| class ratio | 65/35 | %ee >80 binary | Methods |
| time split RF | AUPRC 0.8504±0.0077 | 2023-2024, 245 reactions | Results |
| time split DKL | AUPRC 0.8782±0.0150 | same | Results |
| ADKF-prior | AUPRC 0.9062±0.0221 | support 8 | Results |

### 2.8 化学价值

该文把“新反应族只有少量数据”变成可训练的 support/query 问题。AldolRxnMaster 可据此把 Crimmins/Oppolzer/Myers 从全局低权重样本改为 target tasks。

## §3 任务与数据对齐

- **任务类型**：二分类，%ee >80 vs <80；与本项目 4-class 相似度 3/5。
- **数据/特征**：Ir/Rh AHO；olefin、ligand、solvent 用 Morgan fingerprint 512-bit radius 2；metal/additive OHE；temperature/pressure/catalyst loading；总维度 1544。
- **模型**：DKT、ADKF、ADKF-prior、Protonet；baseline RF/GNN/DKL/DT/XGBoost/AdaBoost/ExtraTrees/SVM。
- **划分**：random、substrate-based、time-based；support sizes 5-128，query 128。
- **不确定性**：Bayesian deep kernel GP 给 posterior uncertainty，可用于 BO。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：使用 fingerprint/graph，不专门解决 R/S；目标为 ee threshold。
- **CIP vs 相对构型**：未处理 absolute configuration；不直接解决本项目 CIP 翻转。
- **局部环境**：靠 Morgan/GNN 表示，未设计 reaction-center priority feature。
- **标签噪声**：通过 meta-learning 适应稀疏文献任务，未做显式 label-noise correction。

## §5 可操作技术建议

#### 建议 A：辅基元学习

| 字段 | 内容 |
|---|---|
| 来自论文 | substrate/time split results |
| 论文怎么做 | 预训练 meta-model，在新 task 上用 8-64 个 support examples 适应。 |
| 在本项目中怎么用 | 改造 `scripts/run_protonet.py`：task 定义为 auxiliary_type × aldehyde_class；Evans 为大 task，Crimmins/Oppolzer/Myers 为 few-shot target。 |
| 预期收益 | 改善非 Evans 子集，尤其全数据 TSCV 0.624 的跨辅基损失。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 B：深核不确定性

| 字段 | 内容 |
|---|---|
| 来自论文 | ADKF/DKT posterior uncertainty |
| 论文怎么做 | deep kernel GP 同时输出预测和不确定性。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 旁新增 `scripts/run_dkl_v4.py`，用 128d v4 features 训练 DKL/GP 分类近似，输出 uncertainty 分位数。 |
| 预期收益 | 识别低置信 class 0/3 与 OOD aldehyde。 |
| 实现难度 | 高 |
| 优先级 | 中 |

#### 建议 C：时间外推报告

| 字段 | 内容 |
|---|---|
| 来自论文 | 2023-2024 time-based split |
| 论文怎么做 | 用旧数据训练，最新文献作测试。 |
| 在本项目中怎么用 | 在 `scripts/run_splits_v4.py` 增加 `source_year` 切分，若无年份则从原始 Reaxys/source 字段补齐。 |
| 预期收益 | 投稿时比随机 split 更可信。 |
| 实现难度 | 中 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 少样本跨辅基泛化 | 中 | 高 |
| B | 置信度/OOD | 高 | 中 |
| C | 严格评估 | 中 | 中 |

## §6 写作借鉴

1. **故事结构**：用“source data 不等于 target reaction 有足够标签”引出 few-shot；本项目可用 Evans vs non-Evans 同样叙事。
2. **方法论证**：support size curve 很有说服力；AldolRxnMaster 可报告每个 auxiliary 的 k-shot 曲线。
3. **Benchmark 严格性**：time split 和 substrate split 值得直接借鉴。
4. **新文献线索**：DKT、ADKF、Prototypical Networks、asymcatml 数据库原文是 meta-learning 方法背景。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 3 | ee 二分类而非绝对构型 |
| 特征工程启示 | 3 | 标准 fingerprint 管线 |
| 手性处理启示 | 2 | 不处理 CIP |
| 小数据策略 | 5 | few-shot 核心论文 |
| 写作/叙事借鉴 | 4 | sparse literature data 叙事可用 |
| 综合优先级 | 4 | 适合解决非 Evans 少样本 |

**一句话总结**：这篇论文对本项目的最大价值不是手性表示，而是把 Evans→非 Evans 的迁移问题形式化为 few-shot/meta-learning。
