# AldolRxnMaster 阶段1/阶段2文献综合总结

本文件汇总 `reading_guide.md` 中阶段 1 的 5 篇精读文献和阶段 2 的 8 篇方法文献。逐篇分析见 `literature/analysis/`。

## 1. 总体结论

阶段 1/2 文献形成一个清楚共识：立体选择性预测不能只依赖通用 SMILES/通用 fingerprint。表现最好的工作通常至少满足以下一个条件：

- 把经典有机化学模型转成特征坐标：glycosylation 的 axial/equatorial 或 Mills vector，Sharpless AD 的 quadrant alignment，CBS 的 key-intermediate graph。
- 把直接 R/S 绝对构型改写为更接近机制的中间标签：ChemAHNet 的 interaction mode，glycosylation 的 Mills configuration。
- 对外推做严格定义：publication split、component holdout、scaffold/value-range split、local reaction-space diagnostic。
- 在小数据场景中主动控制复杂度：低维机制描述符、子类专家模型、few-shot/meta-learning、局部邻域模型。

对 AldolRxnMaster 最关键的判断是：当前 class 0/3 问题不只是模型容量问题，而是 **CIP 绝对标签与反应几何标签不一致** 叠加 **局部反应空间稀疏** 的问题。

## 2. 对本项目最重要的技术启示

### 2.1 标签层：必须并行建立几何标签

最强证据来自 JACS 2025 glycosylation 和 Nat. Comput. Sci. 2025 ChemAHNet。二者都指出 CIP/R/S 或 alpha/beta 不一定稳定对应同一种反应几何。AldolRxnMaster 的芳香醛 Cb priority flip 正是同类问题。

建议优先实现：

- 在 `scripts/run_rebuild_v4.py` 新增 `label_geom_synanti`、`label_cb_geom`、`cb_priority_flip_flag`。
- 在 `scripts/run_all_models_v4.py` 同时报 CIP 4-class、几何 4-class、Ca binary、Cb binary。
- 把 class 0/3 错误按 `ald_pri_is_aromatic`、`ald_pri_priority_proxy`、`cb_priority_flip_flag` 分层报告。

### 2.2 特征层：从全局描述符转向机制对齐特征

最有价值的特征设计来自三篇：

- Chem. Sci. 2021 glycosylation：10 个低维 QM/物理有机描述符 + 温度，data:descriptor >10:1。
- ACS Cent. Sci. 2025 SAD：Sharpless mnemonic → quadrant alignment → 57 维 fragment descriptors。
- JACS 2024 CBS：substrate+catalyst → key-intermediate graph。

建议优先实现：

- 在 `scripts/run_features_v4.py` 增加 aldehyde×auxiliary interaction columns，例如 `ald_pri_priority_proxy * chiral_aux_c4_R`、`ald_Vbur_total_mean * aux_rg_*`。
- 中期实现 Evans aldol quadrant/pseudo-TS features：以 enolate/auxiliary 为坐标系，对 aldehyde substituent 和 auxiliary R-group 做相对方位描述。
- 新增 stereo feature unit tests，确保 Evans 4R/4S、aldehyde priority flip、syn/anti candidates 的特征确实不同。

### 2.3 模型层：优先做 factorized/expert/local，而非盲目深度模型

阶段 2 文献支持三类改造：

- Factorized model：先 Ca/Cb，再 joint label；对应 GlycoPredictor 的级联预测。
- Expert/gating model：按 auxiliary type、aldehyde class、GMM likelihood 路由；对应 Sci Rep composite model。
- Local diagnostics：对每个测试点输出近邻距离和邻居数；对应 ACS Catal. local reaction space。

建议优先实现：

- 新增 `scripts/run_factorized_v4.py`：Ca binary + Cb binary → joint。
- 在 `scripts/run_all_models_v4.py` 增加 subgroup metrics 和 one-vs-rest AUPRC/macro F1。
- 新增 `scripts/run_local_space_v4.py` 输出 `nearest_train_distance`、`n_neighbors_r05`、`same_aux_neighbors`、local RF prediction。

### 2.4 少样本迁移：Evans 到非 Evans 应按 meta-learning 问题处理

Nat. Commun. 2025 与 Angew. Chem. 2025 的 meta-learning 论文说明：大 source task + 小 target task 时，support selection 和 task construction 比单纯合并数据更重要。

建议：

- 改造 `scripts/run_protonet.py`：task 定义比较 `auxiliary_type`、`aldehyde_class`、`feature_cluster`。
- 对 Crimmins/Oppolzer/Myers 做 few-shot adaptation 或 sample weighting。
- 用 meta-cluster 支持集：query 从同一 feature cluster 的训练样本获得 support，而不是随机 support。

## 3. 论文写作建议

推荐 Introduction 主线：

1. 经典手性辅基醛醇反应有经验规则，但文献中尚无 Evans/Crimmins/Oppolzer/Myers aldol 4-class 绝对构型 ML 预测工作。
2. 现代立体选择性 ML 文献显示，纯字符串/通用表示难以可靠处理 chirality，尤其 Transformer 对 `@/@@` 学习困难。
3. 本项目的科学难点不是简单产物预测，而是几何 syn/anti、Ca/Cb、CIP R/S 三套标签体系在芳香醛等场景下不完全一致。
4. AldolRxnMaster 的贡献应表述为：构建首个手性辅基 aldol 立体化学数据库和机制感知 4-class 预测基线，并系统诊断 CIP 标签噪声与外推边界。

推荐 Results 图表顺序：

- 数据集与标签定义：Ca/Cb、label_joint、CIP flip examples。
- 模型基线：ET/XGB/RF 等在 TSCV、grouped、scaffold split 下结果。
- 特征 ablation：steric、condition、chirality、aldpri、interaction。
- 错误与局部空间诊断：class 0/3、aromatic aldehyde、nearest-neighbor distance。
- 改进模型：factorized/expert/local/meta-learning。
- 可解释性：SHAP 或 permutation importance，重点展示 `ald_pri_*` 与 auxiliary chirality 特征。

## 4. 优先级矩阵

| 优先级 | 改动 | 来源文献 | 目标文件/脚本 | 预期收益 |
|---|---|---|---|---|
| 高 | 几何标签并行与 CIP flip flag | JACS 2025 glycosylation；ChemAHNet | `scripts/run_rebuild_v4.py`, `scripts/run_all_models_v4.py` | 根治 class 0/3 解释问题 |
| 高 | Ca/Cb factorized model | JACS 2025 GlycoPredictor | `scripts/run_factorized_v4.py` | 缓解 4-class 稀疏性 |
| 高 | subgroup metrics + AUPRC | Nat Commun 2025 meta-learning；JACS 2024 Diels-Alder | `scripts/run_all_models_v4.py` | 更敏感评价 class 0/3 |
| 高 | local reaction-space diagnostics | ACS Catal. 2025 local space | `scripts/run_local_space_v4.py` | 区分 OOD 与模型错误 |
| 高 | sample_weight / conflict weight | JACS 2024 CBS | `scripts/run_rebuild_v4.py`, `scripts/run_all_models_v4.py` | 降低噪声样本影响 |
| 中 | aldehyde×auxiliary interaction features | ChemAHNet | `scripts/run_features_v4.py` | 捕获组合效应 |
| 中 | meta-cluster few-shot | Nat Commun 2025；Angew 2025 | `scripts/run_protonet.py` | 改善非 Evans |
| 中 | pseudo-TS graph / stereorank | JACS 2024 CBS；JACS 2024 Diels-Alder | `scripts/run_build_graphs.py`, `scripts/run_stereorank.py` | 长期机制模型 |

## 5. 阶段1/2文献清单

| 阶段 | 文件 | 最重要启示 |
|---|---|---|
| 1 | 2021-ChemSci-glycosylation-stereoselectivity.md | 小数据低维机制描述符 + component holdout |
| 1 | 2025-JACS-anomeric-glycosylations.md | Mills/geometric label 避免 CIP/alpha-beta 混淆 |
| 1 | 2025-NatCompSci-asymmetric-hydrogenation.md | interaction mode 先于 absolute configuration |
| 1 | 2025-ACSCentSci-sharpless-dihydroxylation.md | mnemonic → quadrant descriptors → 实验验证 |
| 1 | 2025-ChemSci-stereoselective-synthesis-tools.md | Introduction 和工具演化叙事 |
| 2 | 2024-JACS-cbs-reduction-small-data.md | key-intermediate graph 与高质量小数据 |
| 2 | 2025-AngewChem-bayesian-meta-learning.md | Bayesian meta-learning 处理 few-shot |
| 2 | 2025-NatCommun-meta-learning-selectivity.md | meta-cluster support selection |
| 2 | 2024-NatCommun-chirality-transformers.md | Transformer 对 `@/@@` 手性 token 识别困难 |
| 2 | 2024-JCIM-stereoisomers-ml.md | 必须测试表示是否区分 stereoisomers |
| 2 | 2024-SciRep-composite-stereoselectivity.md | GMM gating / expert model |
| 2 | 2025-ACSCatal-local-reaction-space.md | 局部邻居决定多数预测准确率 |
| 2 | 2024-JACS-diels-alder-chemistry-aware.md | reaction-center edge change 和 candidate ranking |

## 6. 最终建议

下一轮代码工作不应优先追求新深度模型，而应先完成四件事：

1. 建立几何标签与 CIP 标签并行评估。
2. 拆解 Ca/Cb factorized prediction。
3. 输出 subgroup + local-neighbor diagnostics。
4. 加入最小 interaction features 和 stereo feature tests。

完成这四项后，再评估 meta-learning 或 pseudo-TS graph 是否值得投入。
