文件重命名：d4sc07461k.pdf → 2025-ChemSci-stereoselective-synthesis-tools.pdf

# Connecting the complexity of stereoselective synthesis to the evolution of predictive tools

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Connecting the complexity of stereoselective synthesis to the evolution of predictive tools |
| 作者 | Jiajing Li / Jolene P. Reid |
| 期刊 | Chem. Sci. 2025, 16, 3832-3851 |
| DOI | 10.1039/d4sc07461k |
| 规范化文件名 | 2025-ChemSci-stereoselective-synthesis-tools.pdf |
| 开源代码 / 数据集 | 否；综述 |

## §1 核心贡献

- **问题**：综述 stereoselective synthesis 中从定性投影模型、量子化学、LFER/QSSR 到非线性 ML 的预测工具演化。
- **方法创新**：按 substrate control、stoichiometric chiral reagent control、asymmetric catalysis 与工具复杂度组织历史，而非简单按模型分类。
- **结果**：给出本项目 Introduction 可用的叙事框架：反应越依赖弱 NCI 与多组件交互，越需要从经验规则转向描述符和 ML。

## §2 建模体系全链路拆解

这是一篇综述，不提供单一建模管线。对 AldolRxnMaster 关键的“管线抽象”如下：

```text
定性模型: Cram/Felkin-Anh/Evans-style projections → 可解释但边界模糊。
量子化学: TS/conformer/interaction energy → 精准但昂贵，不适合大规模文献数据。
LFER/QSSR: Hammett/Sterimol/NBO/DFT descriptors → 小中型数据可解释建模。
非线性 ML: fingerprint/fragment/GNN/key-intermediate graph → 可处理复杂非线性，但需验证和解释。
```

该综述强调 chiral auxiliary/substrate control 通常比 asymmetric catalysis 更容易形成定性模型，因为已有 stereocenter 或刚性骨架约束更强。这正是本项目 Evans aldol 的论文定位优势。

## §3 任务与数据对齐

1. **任务类型**：综述，不是预测任务；与本项目相似度 4/5，主要在写作定位。
2. **数据规模与质量**：覆盖多个经典案例，无统一数据集。
3. **底物空间**：广泛覆盖 carbonyl stereoselective transformations、asymmetric catalysis、organocatalysis。
4. **标签定义**：多为 ee/enantioselectivity/absolute configuration 讨论。
5. **OOD 评估**：强调局部模型与泛化边界，未提供统一 benchmark。

## §4 手性/立体化学处理方式

1. **SMILES 手性编码**：综述未以 SMILES 为核心；强调物理有机模型与描述符。
2. **CIP vs 相对构型**：可作为引言支撑：绝对构型来自 chirality transfer，但可预测对象应由机制坐标定义。
3. **局部环境**：Sterimol、CoMFA、grid descriptors、fragment descriptors、key-intermediate graph 都是局部环境表征。
4. **标签噪声**：综述层面提醒文献数据与复杂反应需要谨慎使用。

## §5 可操作技术建议

#### 建议 A：引言三段式

| 字段 | 内容 |
|---|---|
| 来自论文 | Introduction / Periodization |
| 论文怎么做 | 从简单投影规则讲到量子化学、LFER、ML，强调复杂反应推动工具演化。 |
| 在本项目中怎么用 | 在论文 `docs` 或 manuscript draft 中按“Evans aldol rule 已知 → CIP 绝对标签和文献异质性导致规则不足 → AldolRxnMaster 量化学习”组织 Introduction。 |
| 预期收益 | 清楚证明项目不是又一个黑箱分类器，而是物理有机预测工具演化中的缺口填补。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 B：工具对照图

| 字段 | 内容 |
|---|---|
| 来自论文 | Table 2 predictive tools |
| 论文怎么做 | 对比定性模型、QM、LFER、ML 的用途和局限。 |
| 在本项目中怎么用 | 增加一个 manuscript figure/table：Evans mnemonic、RDKit priority heuristic、ExtraTrees v4、mechanism-aware model 的数据需求/准确率/解释性对比。 |
| 预期收益 | 提升论文方法论说服力。 |
| 实现难度 | 低 |
| 优先级 | 中 |

#### 建议 C：关键中间体图

| 字段 | 内容 |
|---|---|
| 来自论文 | 对 Pereira 2024 key-intermediate graph 的综述 |
| 论文怎么做 | 把反应关键中间体作为 GNN graph，而非孤立分子图。 |
| 在本项目中怎么用 | 在 `scripts/run_build_graphs.py` 或后续 GNN 脚本中构建 Zimmerman-Traxler-like pseudo complex graph：ketone enolate + aldehyde carbonyl 通过 forming C-C edge 连接。 |
| 预期收益 | GNN 模型更具机制含义。 |
| 实现难度 | 高 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 写作定位 | 低 | 高 |
| B | 方法论论证 | 低 | 中 |
| C | 机制 GNN | 高 | 中 |

## §6 写作借鉴

- 综述的核心句式可转化为本项目卖点：stereoselective synthesis 中微小结构变化会不可预测地改变 selectivity，ML 是量化复杂交互的工具。
- 可用作 Introduction 高级引用，说明 substrate/chiral auxiliary control 是经典但尚未被系统 ML 化的领域。
- 未纳入但高度相关线索：Zahrt et al. Science 2019；Milo/Sigman Nature 2014；Reid & Sigman Nat. Rev. Chem. 2018；Pereira et al. JACS 2024。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 3 | 非原始模型论文 |
| 特征工程启示 | 4 | 总结多个描述符家族 |
| 手性处理启示 | 3 | 主要是宏观框架 |
| 小数据策略 | 4 | 强调 LFER/QSSR 与局部模型 |
| 写作/叙事借鉴 | 5 | Introduction 必引 |
| 综合优先级 | 4 | 写作核心，代码启示次之 |

**一句话总结**：这篇综述为 AldolRxnMaster 提供最重要的论文叙事：经典立体模型不是被 ML 取代，而是被 ML 量化、验证并扩展。
