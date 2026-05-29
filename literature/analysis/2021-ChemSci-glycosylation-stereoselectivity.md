文件重命名：d0sc06222g.pdf → 2021-ChemSci-glycosylation-stereoselectivity.pdf

# Predicting glycosylation stereoselectivity using machine learning

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Predicting glycosylation stereoselectivity using machine learning |
| 作者 | Sooyeon Moon / Peter H. Seeberger, Kerry Gilmore |
| 期刊 | Chem. Sci. 2021, 12, 2931-2939 |
| DOI | 10.1039/d0sc06222g |
| 规范化文件名 | 2021-ChemSci-glycosylation-stereoselectivity.pdf |
| 开源代码 / 数据集 | 是；https://github.com/DrSouravChemEng/GlyMecH |

## §1 核心贡献

- **问题**：预测糖苷化反应 alpha/beta 立体选择性，训练集 268 条，变量覆盖 7 个 electrophile、6 个 nucleophile、4 个 acid catalyst、7 个 solvent 与 -50 到 100 °C 温度。
- **方法创新**：用化学直觉筛选 10 个 QM/物理有机描述符，加温度形成小数据 Random Forest 回归模型，而不是直接使用高维指纹。
- **结果**：RF 平均 RMSE 5.9%，两个 holdout 合并总体 RMSE 6.8%，并预测到 solvent 可开关 leaving group orientation 影响这一新立体控制方式。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：Lewis/Brønsted acid 促进的 glycosylation，生成 C-O 键和 anomeric stereocenter。
- **预测目标**：回归 alpha-product percentage / alpha:beta stereoselectivity。
- **立体化学挑战**：glycosylation 位于 SN1/SN2 连续谱，永久因素和环境因素至少 11 个，经验规则难以定量。
- **与本项目对比**：二者都是小数据、底物/条件共同决定立体结果；本文标签是相对 alpha/beta 比例，本项目是 Ca/Cb 绝对 CIP 4-class，CIP 翻转噪声更严重。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| electrophile | 计算描述符 + 二元构型编码 | C1 13C NMR shift；C2/C4 axial=1/equatorial=0 | SPARTAN, B3LYP/6-31G(d) 或 6-311G(d) |
| nucleophile | 计算描述符 | 17O NMR shift；nucleophilic O exposed surface area；alpha-C exposed surface area | SPARTAN |
| acid catalyst | conjugate base 描述符 | HOMO energy；anion exposed surface area | SPARTAN |
| solvent/temperature | 电子描述符 + 标量 | solvent min/max electrostatic potential；temperature -50 到 100 °C | SPARTAN |

SMILES 手性没有作为主表示；糖环构型用 axial/equatorial 二元特征表达，避免让模型自己从字符串学习立体化学。

### 2.3 特征提取管线

```text
Step 1: 原始反应表 → 清理为 268 条训练反应 + HD1/HD2 holdout。
Step 2: 起始物/溶剂/酸共轭碱 → DFT 优化和电子/空间描述符计算。
Step 3: 人工筛选候选描述符 → 保留 10 个描述符，控制 data:descriptor > 10:1。
Step 4: 10 个描述符 + 温度 → 11 维 RF 输入。
最终: Random Forest 回归 → alpha product percentage。
```

手性专属特征主要是 electrophile C2/C4 axial/equatorial 二元编码，不是 CIP R/S。

### 2.4 模型选择与架构

- **模型类型**：Random Forest regression；对比 GPR、SVM、RT。
- **选择理由**：RF 对异质描述符不需归一化，CART+pruning 降低过拟合；适合 268 条小数据。
- **关键超参数**：论文将超参数细节放在 ESI pS35；主文未逐项列出。
- **小数据设计**：限制描述符数量、人工化学筛选、两个实验 holdout；没有深度模型。
- **多输出**：单输出 alpha selectivity。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | 268 条训练；HD1 测 unseen species；HD2 测 leaving-group orientation 新机制 |
| 底物外推 | 是，HD1 覆盖 unseen electrophile/nucleophile/acid/solvent |
| 时序泄漏 | 未按发表时间；数据为统一微反应器生成 |
| 交叉验证 | 主文强调 holdout；超参数细节在 ESI |
| 类别不平衡 | 回归任务，不适用 |
| HPO | RF tuning，细节见 ESI |

与 AldolRxnMaster 相比：本文没有 TSCV，但 holdout 是实验设计驱动，外推更干净。本项目可保留 TSCV，同时补充 reagent-family holdout。

### 2.6 推理输出与后处理

RF 直接输出 alpha product percentage 随温度变化曲线；无概率校准；不确定性未系统报告。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| RF 平均 RMSE | 5.9% | 训练后模型比较 | Model training and algorithm comparison |
| Baseline | GPR 7.9%, SVM 10%, RT 11% | 同一描述符集 | 同上 |
| 总体 RMSE | 6.8% | 两个 holdout 数据集 | Abstract / Conclusion |
| leaving group orientation 新机制 | RMSE 3.1-6.4 | HD2 solvent-controlled effect | Fig. 7 |
| 因素重要性 | electrophile 27%, nucleophile 20%, solvent 27%, temperature 19% | RF importance | Fig. 8 |

与本项目 Evans ET TSCV=0.795 相比：任务不同，本文是回归；其优势在于外部实验 holdout 和特征可解释性。

### 2.8 化学价值

模型能在实验前预测 glycosylation 条件对 alpha/beta 比例的影响，并识别 solvent 对 leaving-group orientation 的开关效应，说明小数据只要特征机制明确，也能发现非直觉立体控制因素。

## §3 任务与数据对齐

1. **任务类型**：回归；与 4-class 相似度 3/5。
2. **数据规模与质量**：268 条训练，统一微反应器，高重复性；明显优于普通文献挖掘。
3. **底物空间**：糖 donor/acceptor，不像醛醇，但同为底物控制和条件控制共存。
4. **标签定义**：alpha/beta 相对构型，不涉及 CIP 噪声。
5. **OOD 评估**：HD1/HD2 值得效仿，尤其是一次只换一个反应组件。

## §4 手性/立体化学处理方式

1. **SMILES 手性编码**：未依赖 `@/@@`；以环构象方向的二元特征处理糖环手性。
2. **CIP vs 相对构型**：输出 alpha/beta，相对构型比 CIP 稳定；对本项目的启示是不要把几何 syn/anti 和 CIP R/S 混为一谈。
3. **立体中心局部环境**：C2/C4 axial/equatorial 特征相当于围绕 anomeric center 的局部构型代理。
4. **标签噪声处理**：通过统一平台实验生成数据，绕开文献标签噪声；无 label smoothing。

## §5 可操作技术建议

#### 建议 A：单点外推

| 字段 | 内容 |
|---|---|
| 来自论文 | HD1 设计；Model training |
| 论文怎么做 | holdout 中分别放 unseen electrophile/nucleophile/acid/solvent，检验单组件外推。 |
| 在本项目中怎么用 | 在 `scripts/run_splits_v4.py` 新增 `component_holdout`：按 aldehyde scaffold、auxiliary_type、activator、base 分别留出，输出到 `data/splits_v4/component_holdout_*.json`。 |
| 预期收益 | 明确 Evans-only 0.795 是插值还是可迁移；发现 class 0/3 失败集中在哪类 aldehyde。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 B：CIP旁路标签

| 字段 | 内容 |
|---|---|
| 来自论文 | Fig. 3 electrophile axial/equatorial encoding |
| 论文怎么做 | 用 C2/C4 axial/equatorial 直接表达几何关系，而不是 CIP。 |
| 在本项目中怎么用 | 在 `scripts/run_rebuild_v4.py` 生成 `label_geom_synanti` 和 `cb_priority_flip_flag`，并在 `scripts/run_all_models_v4.py` 同时报 Ca/Cb、CIP 4-class、几何 syn/anti 三个评估。 |
| 预期收益 | 定位芳香醛 CIP 翻转噪声；class 0/3 F1 可解释性增强。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 C：描述符瘦身

| 字段 | 内容 |
|---|---|
| 来自论文 | 10 descriptors + temperature，data:descriptor > 10:1 |
| 论文怎么做 | 小数据下限制描述符数量，避免高维过拟合。 |
| 在本项目中怎么用 | 在 `scripts/run_all_models_v4.py` 增加 `compact_mech` 特征组：仅 steric 34d + condition 44d + chirality 7d + aldpri 8d，排除弱贡献 rgroup/chiralenv 做 ablation。 |
| 预期收益 | 验证 128d 是否过宽；可能提升全数据 TSCV 稳定性。 |
| 实现难度 | 低 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 外推诊断 | 中 | 高 |
| B | 降低 CIP 噪声 | 中 | 高 |
| C | 降低过拟合 | 低 | 中 |

## §6 写作借鉴

- Introduction 写法可借鉴：先强调经验规则在机制模糊反应中的不足，再说明 ML 不是黑箱替代，而是量化多个化学因素的工具。
- Results 图表逻辑清晰：数据/描述符 → 模型对比 → holdout 验证 → 发现新机制 → feature importance。
- 本项目投稿时应增加“经验规则/简单基线”对比，例如 Evans auxiliary rule、aldehyde priority rule、ExtraTrees ablation。
- 相关未纳入线索：Zahrt et al. Science 2019 10.1126/science.aau5631；Reid & Sigman Nature 2019 10.1038/s41586-019-1384-z。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 4 | 小数据立体选择性 + 多反应组件，非常接近 |
| 特征工程启示 | 5 | 低维机制特征最值得迁移 |
| 手性处理启示 | 4 | 几何构型编码绕开 CIP 问题 |
| 小数据策略 | 5 | 268 条仍可做稳健外推 |
| 写作/叙事借鉴 | 4 | 机制发现叙事强 |
| 综合优先级 | 5 | 阶段1必读，建议直接实现 |

**一句话总结**：这篇论文证明了“小而准的机制描述符 + 设计良好的 holdout”比盲目堆高维表征更适合立体选择性小数据任务。
