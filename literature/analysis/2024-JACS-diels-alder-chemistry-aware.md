文件重命名：data-efficient-chemistry-aware-machine-learning-predictions-of-diels-alder-reaction-outcomes.pdf → 2024-JACS-diels-alder-chemistry-aware.pdf

# Data-Efficient, Chemistry-Aware Machine Learning Predictions of Diels-Alder Reaction Outcomes

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Data-Efficient, Chemistry-Aware Machine Learning Predictions of Diels-Alder Reaction Outcomes |
| 作者 | Angus Keto / Olaf Wiest |
| 期刊 | J. Am. Chem. Soc. 2024, 146, 16052-16061 |
| DOI | 10.1021/jacs.4c03131 |
| 规范化文件名 | 2024-JACS-diels-alder-chemistry-aware.pdf |
| 开源代码 / 数据集 | Supporting Information 提供数据/代码生成细节；主文未给单一仓库链接 |

## §1 核心贡献

- **问题**：预测 9,537 条 Diels-Alder reaction 的 major product regio/site selectivity，覆盖 intramolecular、hetero、aromatic、IED 等子类。
- **方法创新**：使用 NERF 预测从 reactants 到 products 的 simultaneous edge changes，显式模拟 Diels-Alder bond reorganization。
- **结果**：NERF 用 40% 训练数据即可 Top-1 accuracy 91.4%，85:5:10 split 达 96.1%；Chemformer 需要 >45% 数据和预训练才过 90%。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：intermolecular/intramolecular Diels-Alder reaction，含 hetero Diels-Alder、aromatic Diels-Alder、inverse-electron-demand 等子类。
- **预测目标**：major product structure / regio-site selectivity，等价于预测哪些键形成或重排。
- **立体化学挑战**：DA 的 regioselectivity 受 diene/dienophile electronics、构象、endo/exo 与芳香性影响，模板或字符串模型在稀有子类上容易失败。
- **与本项目对比**：论文不直接预测 Ca/Cb R/S；但“反应中心边变化优先于全局字符串生成”与 aldol C-C 形成键建模高度相关。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| reactants | molecular graph | atom type、charge、aromaticity、segment id、positional embedding | NERF implementation |
| reaction center | edge change candidates | nonautoregressive simultaneous edge changes | 自定义 NERF |
| 反应条件 | 未作为核心输入 | Reaxys DA 数据主要基于反应物到产物 | Reaxys |
| DFT subset | optional 3D/QM features | 7,171 条 DFT-optimized subset；最终未显著提升 NERF | DFT workflow |

### 2.3 特征提取管线

```text
Step 1: Reaxys 37,891 entries → 去重/清洗 → 9,537 DA reactions。
Step 2: reactant/product atom mapping → 标记形成/断裂键。
Step 3: reactant graph → atom/edge features + segment/position encoding。
Step 4: NERF 预测候选 edge changes → 生成 product candidates。
最终: candidate ranking → Top-k product accuracy。
```

### 2.4 模型选择与架构

- **模型类型**：Nonautoregressive Electron Redistribution Framework。
- **选择理由**：DA 反应中心明确，预测边变化比 SMILES 序列生成更数据高效。
- **关键超参数**：论文主要比较训练集比例、子类、Top-k；具体网络参数见 SI/code。
- **小数据设计**：不依赖预训练；低训练比例 5%-40% 下仍与大模型比较。
- **多任务/多输出**：本质为多候选边变化排序。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | 10 个 random splits；training:validation:test 从 5:47.5:47.5 到 92.5:2.5:5 |
| 底物外推 | 按 DA 子类型报告 intermolecular/intramolecular/aromatic/hetero |
| 时序泄漏 | 未以发表年份为主 |
| CV | 多 random split |
| 类别不平衡 | 子类分层报告 |
| HPO | 主文未作为核心 |

### 2.6 推理输出与后处理

模型输出 edge change ranking，后处理为产物图并计算 Top-1/Top-k accuracy；不是直接输出 R/S 标签。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| NERF Top-1 | 91.4% | 40% training | Results |
| 85% training Top-1 | T5Chem 79.0, Chemformer 92.6, NERF 96.1 | random split | Results |
| intermolecular | 95.7% NERF | 80:10:10 | Results |
| intramolecular | 90.0% NERF | 80:10:10 | Results |
| triazine few-shot | 6.0% → 47.0% | 加少量新类数据 | Results |

### 2.8 化学价值

该工作说明对机制清楚的周环反应，显式 reaction-center 表示比通用序列模型更省数据；AldolRxnMaster 可把 forming C-C bond 与两个新立体中心作为类似中心对象。

## §3 任务与数据对齐

- **任务类型**：major product 生成/分类；与本项目相似度 3/5。
- **数据规模**：Reaxys 37,891 entries → 27,349 unique reactions → 9,537 working dataset。
- **模型**：NERF、Chemformer、T5Chem、template GNN/RF 等。
- **输入表示**：NERF 用 molecular graph，节点包括 atom type、charge、aromaticity、segment/positional embedding，多重键为多 edge values。
- **评估**：10 random splits，training:validation:test 从 5:47.5:47.5 到 92.5:2.5:5。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：本文主要评估 regio/site；当前 NERF 版本不编码 stereochemistry，不能预测 diastereoselectivity。
- **CIP vs 相对构型**：未解决；作者明确指出 stereochemical features 被忽略。
- **局部环境**：模型核心是 reaction center 的 edge changes；这对本项目 forming C-C bond 很有启发。
- **标签噪声**：严格过滤 Diels-Alder bonding pattern 和 six-membered ring formation。

## §5 可操作技术建议

#### 建议 A：反应中心边变化

| 字段 | 内容 |
|---|---|
| 来自论文 | NERF edge-change model |
| 论文怎么做 | 预测发生键变化的 atom pairs，而非直接背产品 SMILES。 |
| 在本项目中怎么用 | 在 `scripts/run_stereorank.py` 或 `run_stereorank_product_feats.py` 中显式生成四个 stereochemical product candidates，并用 reaction-center features 排序。 |
| 预期收益 | 将 4-class 预测转为 candidate ranking，减少标签抽象度。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 B：少量新类微调

| 字段 | 内容 |
|---|---|
| 来自论文 | 少量 triazine/oxazole 数据提升新子类 |
| 论文怎么做 | 加入很少新体系训练数据后，NERF 新体系准确率明显提升。 |
| 在本项目中怎么用 | 针对 Crimmins/Oppolzer/Myers 分别进行 few-shot fine-tuning 或 sample-weight upweight，而非与 Evans 等权混合。 |
| 预期收益 | 提升非 Evans 小子集。 |
| 实现难度 | 低 |
| 优先级 | 中 |

#### 建议 C：反应子类报告

| 字段 | 内容 |
|---|---|
| 来自论文 | aromatic/hetero/intramolecular subclass accuracy |
| 论文怎么做 | 对 underrepresented subclasses 单独报告 accuracy。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 增加 subgroup metrics：aromatic aldehyde、halogen aldehyde、auxiliary type、class 0/3。 |
| 预期收益 | 让模型弱点对审稿人透明。 |
| 实现难度 | 低 |
| 优先级 | 高 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 任务重构 | 中 | 高 |
| B | 小子类提升 | 低 | 中 |
| C | 评估透明 | 低 | 高 |

## §6 写作借鉴

1. **故事结构**：先强调 DA 是规则明确但子类复杂的反应，再证明 chemistry-aware model 比预训练 Transformer 更省数据；本项目可对应“Evans 规则明确但 CIP/底物子类复杂”。
2. **方法论证**：论文用训练数据比例曲线展示 data efficiency；AldolRxnMaster 可加入按 Evans 样本比例训练的 learning curve。
3. **Benchmark 严格性**：多 random split 与子类拆分清楚，但时序/OOD 不如本项目 TSCV；可借鉴子类报告。
4. **新文献线索**：NERF、Chemformer、T5Chem 相关 reaction prediction 文献可作为“通用模型不足”背景。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 3 | 反应预测但非手性绝对构型 |
| 特征工程启示 | 4 | reaction-center edge changes 可迁移 |
| 手性处理启示 | 2 | 当前模型未编码 stereochemistry |
| 小数据策略 | 4 | 数据高效和少量新类学习 |
| 写作/叙事借鉴 | 4 | 子类评估清楚 |
| 综合优先级 | 3 | 适合启发 stereorank/candidate ranking |

**一句话总结**：这篇论文提示本项目可把 4-class 分类改写为四个 product candidate 的机制排序任务。
