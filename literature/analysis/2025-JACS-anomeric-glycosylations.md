文件重命名：anomeric-selectivity-of-glycosylations-through-a-machine-learning-lens.pdf → 2025-JACS-anomeric-glycosylations.pdf

# Anomeric Selectivity of Glycosylations through a Machine Learning Lens

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Anomeric Selectivity of Glycosylations through a Machine Learning Lens |
| 作者 | Natasha Videcrantz Faurschou / Christian Marcus Pedersen, Connor W. Coley |
| 期刊 | J. Am. Chem. Soc. 2025, 147, 36197-36209 |
| DOI | 10.1021/jacs.5c07561 |
| 规范化文件名 | 2025-JACS-anomeric-glycosylations.pdf |
| 开源代码 / 数据集 | 是；https://github.com/NatashaVF/GlycoTools |

## §1 核心贡献

- **问题**：从 CAS 2010-2015 文献数据预测 glycosylation 主 anomer、minor anomer 是否出现、以及 anomeric ratio，主/次分类数据 10,434 条，ratio 数据 1,846 条。
- **方法创新**：比较 Chemprop CGR-GNN、scikit-learn fingerprint RF/DT、OHE，并引入 Mills projection donor/product vector 来规避 alpha/beta 与 CIP 标签不一致。
- **结果**：RF OHE 主 anomer AUC random 0.97、publication 0.93；minor product AUC random 0.95、publication 0.88；ratio 回归 publication R2 仅 0.11，显示定性构型比比例外推更可靠。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：文献 glycosylation-type substitution，覆盖 O/S/N glycosylation、pyranoside/furanoside。
- **预测目标**：三阶段多任务：主 anomer 二分类、minor product 二分类、anomeric ratio 回归。
- **立体化学挑战**：alpha/beta 依赖 Fischer projection 远端 stereocenter；CIP R/S 又会受近端取代基优先级影响，二者都可能不等价于实际 axial/equatorial 倾向。
- **与本项目对比**：本文对“CIP 标签不稳定”有直接论述，和 AldolRxnMaster 的芳香醛 Cb CIP 翻转问题高度同构。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| donor/acceptor | CGR graph / Morgan fingerprint / OHE | Chemprop CGR；FP 模型全物种 fingerprint；OHE 按 donor ring bonds 拆 substituent |
| activator | graph / fingerprint / OHE | GNN 中 activator 为额外 graph；FP/OHE 模型按物种编码 | Chemprop, scikit-learn |
| solvent | Morgan fingerprint | 多溶剂时 fingerprint 求和 | RDKit |
| temperature | 标量 | Celsius | scikit-learn |
| 手性 | donor/product Mills vector | out of plane = 1, into plane = -1, deoxy = 0 | 自定义 |

### 2.3 特征提取管线

```text
Step 1: CAS Content Collection → hand-defined reaction SMARTS → glycosylation-type reactions。
Step 2: 人工补充 solvent/temperature，过滤 enzyme、glycal、非单化合物 reagent。
Step 3A: GNN 路线 → donor+acceptor CGR + activator graph + solvent Morgan FP。
Step 3B: FP 路线 → donor/acceptor/activator/solvent fingerprints + temperature。
Step 3C: OHE 路线 → donor substituent OHE + donor vector + activator/solvent OHE + temperature。
最终: 三个模型串联为 GlycoPredictor。
```

### 2.4 模型选择与架构

- **模型类型**：RF、DT、Chemprop GNN。
- **选择理由**：RF 表现最佳；DT 可解释；GNN 灵活但没有优于 tabular 模型。
- **关键超参数**：主文说明除 GNN 小调参外使用 standard hyperparameters；完整参数在 SI S3。
- **小数据设计**：学习曲线从 100 条开始；>1500 条时模型超过专家树。
- **多输出**：不是共享多任务网络，而是三模型级联：major → minor → ratio。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | random split 与 publication split；GNN 使用 8:1:1 train/test/validation |
| OOD | publication split 模拟新 glycosylation methodology |
| 时序泄漏 | publication split 使用最新 publication 作测试 |
| 交叉验证 | 10 test sets learning curves；主表为 3 trials |
| 类别不平衡 | minor product 稀疏，单独建模 |
| HPO | GNN minor optimization；Table 2 simplified model grid search |

### 2.6 推理输出与后处理

输入 reaction SMILES 与条件后先预测 major Mills configuration，再判断 minor 是否出现；若有 minor，再预测 ratio，否则 ratio 置为 100% major，并翻译为 alpha:beta。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| Major AUC | RF OHE random 0.97±0.00；publication 0.93±0.00 | 10,434 条 | Table 1 |
| Minor AUC | RF OHE random 0.95±0.01；publication 0.88±0.00 | 10,434 条 | Table 1 |
| Ratio RMSE/R2 | RF FP random 19.3±0.9%, R2 0.57±0.06；publication 28.0±0.3%, R2 0.05±0.02 | 1,846 条 | Table 1 |
| 外部 Barresi-Hindsgaul | simplified major AUC 0.72；minor AUC 0.24；ratio RMSE 49%, R2 -0.31 | 1994 外部数据 | Table 2 |
| 与本项目对比 | publication major AUC 高于本项目 4-class accuracy；ratio 外推明显较弱 | 不同任务 | Table 1 |

### 2.8 化学价值

GlycoPredictor 能作为 synthesis planning 的局部专家判断 anomeric feasibility，但作者明确认为当前更适合评估已知转化的稳健性，而非完全设计新方法。

## §3 任务与数据对齐

1. **任务类型**：二分类 + 二分类 + 回归；与本项目 4-class 相似度 4/5。
2. **数据规模与质量**：CAS 文献挖掘，10,434 条主/次分类；ratio 数据 1,846 条；报告方式年代差异造成外部集性能下降。
3. **底物空间**：糖类，与醛醇不同，但都是底物构型和条件共同控制。
4. **标签定义**：Mills vector 优于 alpha/beta/CIP，直接启发本项目建立几何标签。
5. **OOD 评估**：publication split 和 1994 外部数据都值得本项目照做。

## §4 手性/立体化学处理方式

1. **SMILES 手性编码**：GNN/FP 使用结构输入，但真正稳定的手性信息来自 donor vector，而非期待模型理解 `@/@@`。
2. **CIP vs 相对构型**：论文明确指出 alpha/beta 和 CIP 都可能随远端或近端取代基改变而不代表相同构象方向。
3. **立体中心局部环境**：Mills vector 对每个 ring stereocenter 编码 -1/0/1，类似本项目可围绕 Cb 建局部 priority vector。
4. **标签噪声处理**：通过重新定义 product configuration 减少标签语义噪声；对旧文献外部集承认报告标准差异。

## §5 可操作技术建议

#### 建议 A：几何标签并行

| 字段 | 内容 |
|---|---|
| 来自论文 | Figure 2 common stereochemical descriptors / Mills projection |
| 论文怎么做 | 放弃 alpha/beta 和 CIP 作为唯一构型标签，改用 out-of-plane / into-plane Mills vector。 |
| 在本项目中怎么用 | 在 `scripts/run_rebuild_v4.py` 增加 `label_cb_geom` 与 `label_joint_geom`；在 `scripts/run_all_models_v4.py` 输出 CIP 4-class 与 geom 4-class 双报告。 |
| 预期收益 | 直接隔离芳香醛 CIP 翻转造成的 class 0/3 噪声。 |
| 实现难度 | 中 |
| 优先级 | 高 |

#### 建议 B：级联输出

| 字段 | 内容 |
|---|---|
| 来自论文 | GlycoPredictor Figure 4 |
| 论文怎么做 | major → minor → ratio 分阶段预测，而非单模型一次预测所有内容。 |
| 在本项目中怎么用 | 新增 `scripts/run_factorized_v4.py`：先预测 Ca，再预测 Cb，最后组合 label_joint；报告 joint accuracy 与两个 binary accuracy。 |
| 预期收益 | 降低 4-class 稀疏性；可能提升 Cb 与 class 0/3 F1。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 C：文献时序集

| 字段 | 内容 |
|---|---|
| 来自论文 | publication split, Table 1 |
| 论文怎么做 | 用最新 publication 作测试，模拟新方法外推。 |
| 在本项目中怎么用 | 若数据有 `year/source`，在 `scripts/run_splits_v4.py` 新增 latest-source split；没有则先在 `data/clean_v4/substrate_aldol_clean.csv` 补 `source_year`。 |
| 预期收益 | 让 TSCV 更接近真实投稿审稿要求。 |
| 实现难度 | 中 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 消除 CIP/几何混淆 | 中 | 高 |
| B | 提升 joint 预测稳健性 | 低 | 高 |
| C | 更严格外推验证 | 中 | 中 |

## §6 写作借鉴

- 很适合作为本项目“CIP R/S 标签未必等于化学几何构型”的核心引用。
- 图表顺序可仿照：标签定义问题 → 多表示模型对比 → random/publication split → 可解释 DT 与专家规则对比。
- 本项目应加入专家规则 baseline，例如 Evans mnemonic、aldehyde priority heuristic，并说明 ML 何处超过规则。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 5 | 底物立体选择性多分类最接近 |
| 特征工程启示 | 4 | donor/product vector 可直接迁移思想 |
| 手性处理启示 | 5 | 直击 CIP 标签不稳定 |
| 小数据策略 | 4 | 大数据主模型，但也有学习曲线 |
| 写作/叙事借鉴 | 5 | 标签定义和专家规则对比非常可用 |
| 综合优先级 | 5 | 必须精读并引用 |

**一句话总结**：这篇论文最重要的价值是把“构型标签定义本身”作为建模问题处理，而这正是 AldolRxnMaster 当前 class 0/3 噪声的根源。
