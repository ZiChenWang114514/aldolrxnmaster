文件重命名：s41598-024-62158-0.pdf → 2024-SciRep-composite-stereoselectivity.pdf

# Predicting the stereoselectivity of chemical reactions by composite machine learning method

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Predicting the stereoselectivity of chemical reactions by composite machine learning method |
| 作者 | Jihoon Chung / Pengyu Hong, Zhenyu Kong |
| 期刊 | Sci. Rep. 2024, 14, 12124 |
| DOI | 10.1038/s41598-024-62158-0 |
| 规范化文件名 | 2024-SciRep-composite-stereoselectivity.pdf |
| 开源代码 / 数据集 | 主文未报告 |

## §1 核心贡献

- **问题**：预测 CPA-catalyzed protic nucleophile addition to imines 的 enantioselectivity ΔΔG‡，数据 342 条。
- **方法创新**：将 SVR、RF、LASSO 组成 composite model，用 GMM 判断测试样本与训练 imine/nucleophile feature distribution 的相似性并选择子模型。
- **结果**：全特征 SVR R2=0.936、MSE=0.182；composite model 三个外部反应类型 R2 分别 0.84、0.97、0.75，MSE 较先前方法降低 >70%。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：chiral phosphoric acid catalyzed addition of protic nucleophiles to imines。
- **预测目标**：ΔΔG‡ 回归，对应 enantioselectivity。
- **立体化学挑战**：不同测试样本可能由 nucleophile、imine 或整体相互作用主导，单一全局模型容易平均化。
- **与本项目对比**：AldolRxnMaster 也存在 auxiliary type、aldehyde class、condition regime 多子空间；composite routing 对非 Evans 和芳香醛子集很相关。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| nucleophile | molecular descriptors | 用于 GMM similarity 与模型训练 | descriptor workflow |
| imine | molecular descriptors | 可包含或排除以构建 no-imine model | descriptor workflow |
| catalyst/condition | descriptor/context features | CPA 数据中反应条件相对固定 | 文献数据 |
| similarity | GMM likelihood | nucleophile 14 components, imine 12 components by BIC | GMM |

### 2.3 特征提取管线

```text
Step 1: CPA reaction dataset → 307 training reactions + external tests。
Step 2: 计算 nucleophile/imine/catalyst descriptors。
Step 3: 训练三类模型：overall all-feature SVR、nucleophile-focused RF、LASSO fallback。
Step 4: 用 GMM 对 nucleophile/imine feature space 估计 test similarity。
最终: 根据 similarity route 选择子模型输出 ΔΔG‡。
```

### 2.4 模型选择与架构

- **模型类型**：LASSO、RF、SVR；外层 composite router。
- **选择理由**：不同局部化学空间由不同特征块主导，固定单模型不稳。
- **关键超参数**：SVR best MSE 0.182/R2 0.936；GMM components nucleophile 14、imine 12。
- **小数据设计**：Bayesian optimization、permutation importance、GMM similarity gating。
- **多任务/多输出**：单任务回归，模型集成通过 gating 实现。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | 训练集 + Reid 等外部测试集 |
| 底物外推 | 通过 GMM similarity 分析测试样本是否接近训练子空间 |
| 时序泄漏 | 未以年份为主 |
| CV | 用于模型评估/HPO |
| 类别不平衡 | 回归任务 |
| HPO | Bayesian optimization |

### 2.6 推理输出与后处理

先计算测试样本在 nucleophile/imine GMM 中的 similarity，再选择 overall SVR、nucleophile RF 或 LASSO，输出 ΔΔG‡。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| SVR all features | MSE 0.182, R2 0.936 | training/CV | Table 1 |
| RF without imine | R2 0.932, MSE 0.192 | feature ablation | Table 3 |
| GMM components | nucleophile 14, imine 12 | BIC selection | Results |
| composite tests | R2 0.84, 0.97, 0.75 | three external sets | Table 5 |
| MSE improvement | >70% decrease | vs previous SOTA | Discussion |

### 2.8 化学价值

该文说明“测试样本属于哪个局部反应空间”本身可以是模型的一部分。AldolRxnMaster 应把 non-Evans 与芳香醛看成需要路由/诊断的子空间。

## §3 任务与数据对齐

- **任务类型**：ΔΔG‡ 回归；与本项目相似度 3/5。
- **数据规模**：342 CPA reactions；307 training + 多个外部测试组合。
- **模型**：LASSO、RF、SVR；Bayesian optimization；permutation importance；GMM gating。
- **关键特征**：imine、nucleophile、catalyst、solvent descriptors；重要特征包括 nucleophile “H-X-CNu”“H-X-Nu”、nXH、HOMO、L 等。
- **验证**：100 replications；Table 5 外部测试。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：未讨论。
- **CIP vs 相对构型**：预测 ΔΔG‡，不处理 absolute configuration。
- **局部环境**：descriptor 以 imine/nucleophile/catalyst/solvent 为模块；GMM 用关键局部特征判断相似性。
- **标签噪声**：未显式处理。

## §5 可操作技术建议

#### 建议 A：GMM模型门控

| 字段 | 内容 |
|---|---|
| 来自论文 | Figure 7 composite model |
| 论文怎么做 | 用 GMM 判断 imine/nucleophile 是否在训练分布内，再选择 SVR/RF/LASSO。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 增加 gate：若 aldehyde `ald_pri_*` 与训练 GMM log-likelihood 低，使用 conservative ExtraTrees/nearest-neighbor fallback。 |
| 预期收益 | 减少 OOD 芳香醛或罕见 auxiliary 的过度自信预测。 |
| 实现难度 | 中 |
| 优先级 | 中 |

#### 建议 B：特征重要性删减

| 字段 | 内容 |
|---|---|
| 来自论文 | permutation importance Tables 2/4 |
| 论文怎么做 | 移除低重要性特征并比较模型。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 加 permutation importance 导出，并自动生成 top-k feature ablation。 |
| 预期收益 | 降低 128d 过拟合；解释模型。 |
| 实现难度 | 低 |
| 优先级 | 中 |

#### 建议 C：子模型专家

| 字段 | 内容 |
|---|---|
| 来自论文 | overall SVR / nucleophile-focused RF / LASSO |
| 论文怎么做 | 不同化学相似性区域使用不同模型。 |
| 在本项目中怎么用 | 训练 Evans-only expert、non-Evans expert、aromatic-aldehyde expert，再按 `auxiliary_type` 和 `ald_pri_is_aromatic` 路由。 |
| 预期收益 | 改善全数据 TSCV 和 class 0/3。 |
| 实现难度 | 中 |
| 优先级 | 高 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | OOD 降风险 | 中 | 中 |
| B | 特征瘦身 | 低 | 中 |
| C | 异质数据建模 | 中 | 高 |

## §6 写作借鉴

1. **故事结构**：从“全局模型无法覆盖所有局部机制”引出 composite model；本项目可用 Evans-only 与 full-data TSCV gap 做相同论证。
2. **方法论证**：GMM similarity 让失败样本解释更具体；AldolRxnMaster 可报告 nearest-neighbor/local-likelihood。
3. **Benchmark 严格性**：外部测试集比单 random split 更有说服力；本项目可按发表时间或 auxiliary holdout 增强。
4. **新文献线索**：Reid CPA 数据集及其早期 multivariate linear regression/selectivity prediction 文献值得补入背景。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 3 | CPA ee 回归 |
| 特征工程启示 | 3 | 重要性分析可用 |
| 手性处理启示 | 2 | 不处理 CIP |
| 小数据策略 | 4 | 342 条 + composite |
| 写作/叙事借鉴 | 3 | 方法朴素但可解释 |
| 综合优先级 | 3 | 可借鉴 gate/ensemble |

**一句话总结**：这篇论文最可用的是“相似性判断后选择专家模型”，适合 AldolRxnMaster 的 auxiliary/aldehyde 异质空间。
