文件重命名：data-driven-prediction-of-enantioselectivity-for-the-sharpless-asymmetric-dihydroxylation-model-development-and.pdf → 2025-ACSCentSci-sharpless-dihydroxylation.pdf

# Data-Driven Prediction of Enantioselectivity for the Sharpless Asymmetric Dihydroxylation

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Data-Driven Prediction of Enantioselectivity for the Sharpless Asymmetric Dihydroxylation: Model Development and Experimental Validation |
| 作者 | Blake E. Ocampo, Bilal Altundas / Scott E. Denmark |
| 期刊 | ACS Cent. Sci. 2025, 11, 1640-1650 |
| DOI | 10.1021/acscentsci.5c00900 |
| 规范化文件名 | 2025-ACSCentSci-sharpless-dihydroxylation.pdf |
| 开源代码 / 数据集 | 是；https://github.com/SEDenmarkLab/SAD |

## §1 核心贡献

- **问题**：用 1,007 条 Sharpless asymmetric dihydroxylation 文献反应预测不同 alkene class 的 enantioselectivity magnitude。
- **方法创新**：把 Sharpless/Norrby mnemonic 转为 quadrant alignment，再用 fragment-based 57 维 steric/electronic/RDF/NBO 描述符建模。
- **结果**：总体 train R2=0.91、test r2=0.72、Q2F3=0.93、MAE=0.30 kcal/mol；15 个未报道底物实验验证中 10 个误差在 ±16% ee 内。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：Sharpless asymmetric dihydroxylation，AD-mix alpha/beta。
- **预测目标**：ΔΔG‡ 回归，代表 ee magnitude；不预测具体 enantiomer identity。
- **立体化学挑战**：Sharpless mnemonic 能判断 face，但底物类别和电子效应会导致低选择性或 mnemonic 失效。
- **与本项目对比**：都是经典立体模型已存在但不够定量；本文把 mnemonic 工程化为 alignment/descriptor，本项目也应把 Evans aldol mnemonic 工程化。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| alkene | fragment/quadrant descriptors | Mono/Gem/Cis/Trans/TriQ2/TriQ3/TriQ4/Tetra 分类 | RDKit, molli |
| steric | Sterimol + volume | B/L；Max Volume；3-BFS Volume | Morfeus, molli |
| electronics | ESPMIN/ESP99/NBO | fragment ESP；full alkene NBO | Gaussian/NBO workflow via molli |
| conformational | RDF | alkene atom-centered radial distribution | custom RDF |
| conditions | mostly omitted | temperature normalized to 0 °C；AD-mix alpha/beta averaged | custom |

### 2.3 特征提取管线

```text
Step 1: 文献/百科/primary literature → 1,007 unique SAD entries。
Step 2: ChemDraw → canonical SMILES via RDKit → 去重/标准化。
Step 3: 根据 Sharpless mnemonic 选 Q1，并按 counterclockwise 拼接 quadrants。
Step 4: 每个 quadrant 计算 Sterimol、Max/3-BFS volume、ESPMIN/ESP99；全 alkene 计算 RDF/NBO。
Step 5: 拼接为 57 维 descriptor；按 8 个 alkene class 分别建模。
最终: class-specific regression → normalized positive ΔΔG‡。
```

### 2.4 模型选择与架构

- **模型类型**：PLS/SVR/Ridge/Lasso/RF/GBR/XGBoost/kNN/GPR 对比；最终不同 class 选不同模型，GBR 多数最佳，Trans 用 RF，Tetra 用 GPR。
- **选择理由**：不同 alkene class 关系不同，class-specific model 优于统一模型。
- **关键超参数**：80/20 split；5-fold randomized search；再 5-fold Bayesian optimization。
- **小数据设计**：低样本 class 仍建模但用 cross-validation 暴露过拟合风险。
- **多输出**：单回归输出 ΔΔG‡。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | 80/20 train/test，按 alkene class 建模 |
| OOD | 从 2,524 个外部 osmium dihydroxylation alkene 中选 15 个实验验证 |
| 时序泄漏 | 未按时间划分 |
| CV | 5-fold randomized search + Bayesian optimization；低样本 class 另看 q2 |
| 类别不平衡 | 按 class 拆分；承认 Trans 高 ee 正偏 |
| HPO | randomized search + Bayesian optimization |

### 2.6 推理输出与后处理

输出 normalized positive ΔΔG‡，即 ee magnitude；不直接输出 absolute stereochemistry，因为 AD-mix face 方向由 mnemonic/alignment 决定。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| 总体 | train R2=0.91, train MAE=0.16, test r2=0.72, Q2F3=0.93, MAE=0.30 kcal/mol | all class models | Modeling Results |
| Mono | Q2F3=0.91, MAE=0.27, r2=0.63 | class-specific | Fig. 4 |
| Gem | Q2F3=0.94, MAE=0.32, r2=0.74 | class-specific | Fig. 4 |
| Cis | Q2F3=0.97, MAE=0.22, r2=0.74 | 41 cis alkenes | Fig. 4 |
| Tetra | Q2F3=0.89, MAE=0.22, r2=0.89 | 21 tetra alkenes；CV poor | Fig. 4 |
| 实验验证 | 10/15 within ±16% ee；mono 全部 within ±7% ee；tri 全部 at most ±7% ee | 15 new alkenes | Fig. 6 / Conclusions |

### 2.8 化学价值

该工作展示了文献数据虽然偏正且异质，但如果先用经典 mnemonic 对齐，再用物理描述符建模，仍可指导未报道底物的实验验证。

## §3 任务与数据对齐

1. **任务类型**：回归；与 4-class 相似度 3/5。
2. **数据规模与质量**：1,007 条，接近本项目 Evans-only 规模；文献数据需人工校验。
3. **底物空间**：alkenes，不是 aldol；但 chiral reagent/catalyst mnemonic 工程化很相关。
4. **标签定义**：避免直接预测 enantiomer identity，而预测 magnitude；本项目可借鉴“先几何对齐再建模”。
5. **OOD 评估**：15 个实验验证是顶刊叙事关键。

## §4 手性/立体化学处理方式

1. **SMILES 手性编码**：不依赖 SMILES 手性；用 quadrant alignment 表达 face selectivity。
2. **CIP vs 相对构型**：明确用 top/bottom face independent of CIP，避免 diverse substituent 下 CIP priority 不一致。
3. **局部环境**：四象限 fragment descriptors 是围绕 alkene reaction center 的局部环境特征。
4. **标签噪声**：人工查原始文献和 expert encyclopedia，减少 Reaxys/Scifinder 转录错误。

## §5 可操作技术建议

#### 建议 A：Evans象限化

| 字段 | 内容 |
|---|---|
| 来自论文 | Figure 3 alignment-dependent fragment descriptors |
| 论文怎么做 | 将 Sharpless mnemonic 转为 Q1-Q4 对齐坐标，再拼接描述符。 |
| 在本项目中怎么用 | 在 `scripts/run_features_v4.py` 增加 Evans aldol quadrant descriptors：以 enolate C=C/auxiliary C4 为参考轴，对 aldehyde substituent 与 auxiliary R-group 生成 left/right/up/down steric/electronic columns。 |
| 预期收益 | 比全局 Morgan/steric 更贴近 Zimmerman-Traxler TS；改善 class 0/3。 |
| 实现难度 | 高 |
| 优先级 | 高 |

#### 建议 B：按子类建模

| 字段 | 内容 |
|---|---|
| 来自论文 | eight alkene class models |
| 论文怎么做 | Mono/Gem/Cis/Trans/Tri/Tetra 分别建模，不强行统一。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 增加 auxiliary-specific 和 aldehyde-class-specific 模型：Evans、Crimmins、Oppolzer、Myers 分开训练，再做 gating/ensemble。 |
| 预期收益 | 降低全数据 TSCV 0.624 的异质性损失。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 C：实验验证叙事

| 字段 | 内容 |
|---|---|
| 来自论文 | Experimental validation Figure 6 |
| 论文怎么做 | 从外部 2,524 个 alkene 中聚类选 15 个底物，实验验证模型。 |
| 在本项目中怎么用 | 在论文写作计划中设计 8-12 个未见 aldehyde/auxiliary 组合的湿实验验证，优先选模型高置信但规则不确定样本。 |
| 预期收益 | 显著提升 Nat Commun/JACS 说服力。 |
| 实现难度 | 高 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 机制特征提升 | 高 | 高 |
| B | 处理辅基异质性 | 中 | 高 |
| C | 投稿说服力 | 高 | 中 |

## §6 写作借鉴

- 这篇是本项目投稿 ACS Central Science/JACS 的直接写作模板：文献数据库 → mnemonic quantification → ML → SHAP → 实验验证。
- 本项目需要类似图：Evans mnemonic/TS model 被转为可计算特征，并展示对错误 class 的纠正。
- SHAP 只作 hypothesis generator，不作因果证明，这个措辞值得照搬。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 4 | 经典立体模型工程化高度相关 |
| 特征工程启示 | 5 | quadrant descriptors 可直接启发 |
| 手性处理启示 | 5 | face alignment 独立于 CIP |
| 小数据策略 | 4 | 分 class 小数据建模并公开风险 |
| 写作/叙事借鉴 | 5 | 实验验证闭环强 |
| 综合优先级 | 5 | 阶段1核心 |

**一句话总结**：它展示了如何把经典立体化学 mnemonic 变成机器学习坐标系，这正是 AldolRxnMaster 下一步从经验特征走向论文级机制特征的路径。
