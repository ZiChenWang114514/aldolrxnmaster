文件重命名：stereoisomers-are-not-machine-learning-s-best-friends.pdf → 2024-JCIM-stereoisomers-ml.pdf

# Stereoisomers Are Not Machine Learning's Best Friends

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Stereoisomers Are Not Machine Learning's Best Friends |
| 作者 | Gökhan Tahıl / Sébastien Tilloy |
| 期刊 | J. Chem. Inf. Model. 2024, 64, 5451-5469 |
| DOI | 10.1021/acs.jcim.4c00318 |
| 规范化文件名 | 2024-JCIM-stereoisomers-ml.pdf |
| 开源代码 / 数据集 | 论文报告 Stereo2vec 相关资源；主文未给单一 GitHub 链接 |

## §1 核心贡献

- **问题**：系统测试常用分子描述符、fingerprints 和 embedding 是否能区分 stereoisomers。
- **方法创新**：提出 MolStereo2vec、IsoString2vec、IsoSymbol2vec、IsoOrder2vec，并设计多组 stereoisomer discrimination tests。
- **结果**：RDKit descriptors、ECFP with chirality、Mol2vec/MolChiral2vec 都无法完全区分 stereoisomers；MolStereo2vec、IsoString2vec、IsoOrder2vec 表现更稳健。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应/性质类型**：不是反应预测，而是 stereoisomer descriptor distinguishability 与 cyclodextrin host-guest binding prediction。
- **预测目标**：描述符是否能唯一区分 stereoisomer；下游为结合性质预测。
- **立体化学挑战**：许多分子表示在连接相同、只构型不同的分子上给出相同向量，ML 因而先天无法学习立体差异。
- **与本项目对比**：AldolRxnMaster 的 Ca/Cb 标签正是 stereochemical identity；若 128d 特征对关键异构体碰撞，模型调参无法修复。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| stereoisomer sets | RDKit descriptors / ECFP / Mol2vec variants | RDKit descriptor, ECFP with chirality, Mol2vec, MolChiral2vec, MolStereo2vec | RDKit, embedding workflow |
| cyclodextrin host-guest pairs | descriptor vectors | 3,459 pairs，含 only-stereoisomer subset | OpenCycloDB |
| 模型输入 | concatenated descriptors | host/guest descriptors 拼接或组合 | scikit-learn / boosting |
| 手性信息 | explicit stereo tokens | IsoString/IsoSymbol/IsoOrder2vec 显式编码异构关系 | 自定义 |

### 2.3 特征提取管线

```text
Step 1: stereoisomer molecules → RDKit canonicalization / fingerprinting。
Step 2: 计算 RDKit descriptors、ECFP with chirality、Mol2vec/MolChiral2vec/MolStereo2vec。
Step 3: 检查同一 stereoisomer group 内 descriptor collision 数。
Step 4: host-guest pair descriptors → RF/XGB/LGBM。
最终: 全集与 only-stereoisomer subset 的模型性能比较。
```

### 2.4 模型选择与架构

- **模型类型**：RF、XGBoost、LightGBM；描述符本身是论文重点。
- **选择理由**：用强 tabular 模型验证“表示是否区分 stereoisomers”会不会影响下游。
- **关键超参数**：Optuna HPO；具体参数随模型/描述符组合优化。
- **小数据设计**：stratified 5-fold + HPO；重点在 OS subset。
- **多任务/多输出**：单一性质预测；多表示对比。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | stratified 5-fold |
| 底物外推 | OS subset 专门检验 stereoisomer-only difficulty |
| 时序泄漏 | 非时序数据，未做 temporal split |
| CV | 5-fold |
| 类别不平衡 | stratification |
| HPO | Optuna |

### 2.6 推理输出与后处理

模型输出 binding/property prediction；更关键的后处理是对 stereoisomer group 内表示碰撞计数，判断输入特征是否先天不可区分。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| OpenCycloDB collision | RDKit 510/510, ECFP 505/510 identical | stereoisomer distinguishability | Table 16/17 |
| distinguish all | MolStereo2vec/IsoString/IsoOrder2vec | tested stereoisomers | Results |
| host-guest pairs | 3,459 | cyclodextrin dataset | Methods |
| OS subset | MolStereo2vec 更稳健 | only stereoisomers | Table 21 |

### 2.8 化学价值

该文提醒：模型分数差不一定来自算法，可能来自输入表示已经把立体异构体折叠成同一个点。对 AldolRxnMaster，立体特征碰撞审计应放在模型调参之前。

## §3 任务与数据对齐

- **任务类型**：分子表示诊断 + cyclodextrin association constant 回归；与本项目相似度 3/5。
- **关键数据**：ZINC20 文件测试显示 selected RDKit descriptors/ECFP with chirality 对大量分子给出 identical representation；OpenCycloDB 510 stereoisomers 进一步验证。
- **模型**：RF、XGBoost、LightGBM；Optuna HPO；Stratified 5-fold CV。
- **性能定位**：Table 21 显示整体测试分数不足以揭示 stereoisomer 失败，必须拆 OS (only stereoisomers) 和 WS 子集。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：Isomeric SMILES 可表达 stereochemistry，但文本方法会受顺序/符号问题影响。
- **CIP vs 相对构型**：MolStereo2vec 显式使用 RDKit `ChiralType`、`CIPCode`、E/Z、r/s 信息生成 stereochemistry-sensitive identifiers。
- **局部环境**：MolStereo2vec 在 Morgan identifier 层面加入 stereochemistry，接近“局部手性环境 fingerprint”。
- **标签噪声**：论文重点是表示不可分，而非标签噪声。

## §5 可操作技术建议

#### 建议 A：立体表示单元测试

| 字段 | 内容 |
|---|---|
| 来自论文 | Tables 3/13-17 stereoisomer discrimination |
| 论文怎么做 | 用成对 stereoisomers 测表示是否真的不同。 |
| 在本项目中怎么用 | 新增 `tests/test_stereo_features.py`：构造 Evans 4R/4S、aldehyde priority flip、syn/anti product pairs，断言 `run_features_v4.py` 输出的 chirality/aldpri 特征不同。 |
| 预期收益 | 防止 future feature refactor 破坏手性分辨能力。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 B：OS子集报告

| 字段 | 内容 |
|---|---|
| 来自论文 | Table 21 OS/WS split |
| 论文怎么做 | 单独统计 testing molecules whose stereoisomer exists in train。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 增加 `stereo_pair_subset` 评估：训练/测试中同骨架不同 Ca/Cb 标签的样本单独算 accuracy/F1。 |
| 预期收益 | 暴露模型是否只是识别骨架而非立体。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 C：MolStereo式指纹

| 字段 | 内容 |
|---|---|
| 来自论文 | MolStereo2vec |
| 论文怎么做 | 在 Morgan identifiers 中加入 CIP/EZ/r/s stereochemical identifiers。 |
| 在本项目中怎么用 | 在 `scripts/run_features_v4.py` 加 `morgan_chiral_count_*` 或局部 hashed stereochemical identifiers，替代仅全局 `chiral_sum_signs`。 |
| 预期收益 | 增强 Evans/Crimmins 立体异构体区分。 |
| 实现难度 | 中 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 防回归 | 低 | 高 |
| B | 真实手性能力评估 | 中 | 高 |
| C | 表示增强 | 中 | 中 |

## §6 写作借鉴

1. **故事结构**：论文不是先堆模型，而是先问“表示有没有能力区分目标”；本项目 Introduction 可用这个逻辑引出 `ald_pri_*` 与几何标签。
2. **方法论证**：建议增加 feature collision audit 表，比单纯 ablation 更能证明特征必要性。
3. **Benchmark 严格性**：OS subset 很适合借鉴；AldolRxnMaster 可构造 class 0/3、芳香醛 priority flip 子集。
4. **新文献线索**：MolChiral2vec、MolStereo2vec 原始表示学习文献可作为 stereochemical descriptor 背景。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 2 | 非反应预测 |
| 特征工程启示 | 4 | stereo-sensitive identifiers 可迁移 |
| 手性处理启示 | 5 | 直接检验手性表示失效 |
| 小数据策略 | 2 | 不是重点 |
| 写作/叙事借鉴 | 4 | Introduction 负面证据 |
| 综合优先级 | 4 | 适合做测试与审计 |

**一句话总结**：该文提醒本项目必须测试“特征是否真的区分立体异构体”，不能只看总体 accuracy。
