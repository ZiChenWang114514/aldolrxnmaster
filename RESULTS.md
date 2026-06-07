# Results — AldolRxnMaster

基于底物控制（手性辅基）的 Aldol 反应 4-class 立体化学预测基准。

## V5 基准 (2026-05-30)

**数据**: 2434 substrate-controlled aldol reactions (V5, 从 134K Reaxys 重建, 9 种辅基)
**辅基**: Evans (1661) + Crimmins thione (260) + Crimmins oxathione (169) + Oppolzer (141) + **Abiko (127)** + **Menthyl ester (32)** + **Oxazoline (21)** + **Myers (16)** + Other (7)
**VALID**: 2427 行 (10 种辅基, 排除 7 行 other_auxiliary)
**特征**: Steric(34d) + Conditions(44d) + Aux one-hot(9d) + Aux mechanistic(6d) + Chirality(7d) + R-group(7d) + ChiralEnv(21d) + AldPriority(8d) + DeltaChiral(16d) + ChiralDet(3d) + n_stereo(1d) = **156d**

### V5 默认模型 (156d)

| Rank | Model | Category | TSCV mean±std | Scaffold | Grouped mean±std |
|------|-------|----------|---------------|----------|-----------------|
| 1 | **v4b_full_et** | v4b | **0.677±0.048** | 0.717 | 0.728±0.024 |
| 2 | **v4b_full_rf** | v4b | 0.660±0.077 | 0.773 | 0.744±0.019 |
| 3 | **v4b_full_xgb** | v4b | 0.652±0.041 | **0.831** | **0.760±0.016** |
| 4 | v4b_full_lgbm | v4b | 0.625±0.058 | 0.744 | 0.748±0.016 |
| 5 | v4b_no_chiral_xgb | ablation | 0.613±0.105 | 0.672 | 0.687±0.024 |
| 6 | steronly_xgb | steric | 0.573±0.029 | 0.708 | 0.638±0.012 |
| 7 | v4b_condaux_xgb | v4b | 0.480±0.023 | 0.526 | 0.591±0.028 |
| 8 | v4b_chiral_only_xgb | ablation | 0.452±0.028 | 0.452 | 0.514±0.013 |
| 9 | cond_xgb | baseline | 0.252±0.038 | 0.261 | 0.409±0.016 |
| 10 | majority | baseline | 0.250±0.000 | 0.250 | 0.250±0.000 |

### V5 Optuna-Tuned Models (200 trials/model)

| Rank | Model | TSCV mean±std | Scaffold | Grouped mean±std |
|------|-------|---------------|----------|-----------------|
| 1 | **xgb_optuna** | **0.739±0.074** | — | **0.760** |
| 2 | et_optuna | 0.722±0.061 | — | 0.730 |
| 3 | ma_bw_xgb_optuna | 0.666±0.034 | — | 0.752 |

### V5 ZT-GNN Models (Evans-only, 1661 rows, TSCV 4-fold)

| Rank | Model | TSCV mean±std | 说明 |
|------|-------|---------------|------|
| 1 | **ZT-Chiral+feat** | **0.818±0.017** | 手性消息传递 + 156d global features |
| 2 | ZT-ComENet+feat | 0.784±0.051 | 组合等变网络 |
| 3 | ZT-Hybrid+feat | 0.776±0.058 | 多视角融合 |
| 4 | ZT-GAT+feat | 0.753±0.041 | 图注意力 |
| 5 | ZT-GIN+feat | 0.731±0.081 | 图同构网络 |
| 6 | ZT-ChiDeK+feat | 0.721±0.061 | 手性 + DekeyNetwork |
| 7 | ZT-GCPNet+feat | 0.715±0.046 | 图属性网络 |
| — | *Baselines* | | |
| — | Chemprop+156d+ZT | 0.809 | MPNN + ZT 32d |
| — | Evans ET (tree) | 0.710 | 无图表示 |

**负面结果**: MultiTS (多 TS 注意力, 未完成): fold1-2 估计 ~0.715, 远低于 ZT-Chiral 0.818。额外的多 TS 构象注意力机制未带来提升。

### V5 按轴分解 + Gold-Test 诚实评测 (2026-06-07)

**核心结论**: 4-class CIP 准确率 ≈ **α轴 × 羰醇轴** 的乘积。三层廉价 gate 实证（详见 `LESSONS.md` L14/L15）锁定真天花板结构——
- **α 中心 (label_Ca, 辅基控制)** 在可信标签上 ≈ **0.93–0.96**，已近其噪声上限；现报 0.88 大半是**评测 test 集的标签噪声**（~6% test 行标签本身错/矛盾，模型对了被判错）。
- **羰醇中心 (label_Cb, 醛面选择)** ≈ **0.82–0.83**，是真正的瓶颈。
- 机理坐标系重标注、标签去噪均被证否（`label_syn_anti_3d` 是噪声，grouped lift 仅 0.025；broad-SMARTS 交叉校验 CIP **零分歧**；去 214 行 Type A 矛盾仅 +0.007；删数据有害——数据量 > 标签纯度）。

**Gold-test 评测协议**: gold = 高 mapping-confidence ∧ 非 Type A 矛盾 ∧ broad-SMARTS CIP 一致（可信标签子集）。同一 ExtraTrees 模型在 full / gold / non-gold 三个 test 子集上评测，分离"标签噪声压低指标"与"分子更易"的选择效应。

| Scope | 轴 | split | full | **gold** | non-gold | 读法 |
|---|---|---|---|---|---|---|
| Evans | Ca (α) | grouped | 0.894 | **0.963** | 0.865 | gold≫non-gold → 误差主要是标签噪声 |
| Evans | Cb (羰醇) | grouped | 0.815 | **0.833** | 0.808 | gold≈non-gold → **真实化学瓶颈,非标签伪影** |
| Evans | SA (主产物对) | tscv | 0.833 | 0.836 | 0.829 | — |
| Evans | joint (4-class) | tscv | 0.762 | **0.793** | 0.740 | =两轴乘积 |
| 全数据集 | Ca (α) | grouped | 0.864 | **0.932** | 0.836 | α 全集亦 ~0.93 |
| 全数据集 | Cb (羰醇) | grouped | 0.791 | 0.827 | 0.776 | 瓶颈一致 |
| 全数据集 | joint | grouped | 0.729 | 0.776 | 0.705 | — |

**Evans joint gold-test 混淆矩阵** (行=真, 列=预测, 序 RR/RS/SR/SS): `[[84,43,6,1],[3,157,4,2],[5,1,131,4],[9,8,19,69]]` — RS/SR(Ca≠Cb) recall 0.93–0.95，RR/SS(Ca==Cb) 仅 0.63–0.66，主错在 Cb。

**诚实边界**: 现有冠军 ZT-Chiral 0.818 基本到顶；4-class 冲 90% 需两轴**都** >0.95，而羰醇轴受真实化学限制 ~0.82。可发表叙事："底物控制的 α-立体诱导可达 ~94%，相对构型(羰醇)轴是预测瓶颈"。脚本: `scripts/run_honest_eval.py` → `results/tables/honest_axis_eval.csv`。

**羰醇轴攻坚 (Phase B, 负结果, LESSONS L16)**: 4 个廉价杠杆均未通过 gold-test gate(Δ≥+0.01)——`+face_map(24d 真实构象面图)` 有害, `+SPMS(16d)` 混合, `+两阶段Cα→Cb` 与 `+显式醛CIP优先级` 在 full-test 显示虚假提升但 gold-test 蒸发。羰醇面选择需真实 TS 能量学(ΔΔG‡, xTB/qTS 已证失败), **0.82 确认为真上限**。脚本: `scripts/run_carbinol_gate.py`。

### V5 特征有效性 / 标签消融（null-importance）(2026-06-07)

**方法**: 三角度 × 三轴(Ca/Cb/joint)识别噪声特征。① **标签消融 (M1, null-importance)**: 打乱标签重训 30 次得每特征重要性的 null 分布，真重要性未显著高于 null(p>0.05)即噪声。② **gold-test permutation (M2)**: fold 内 permute 每列、测 gold(可信标签)子集 balanced-acc 跌幅，gold_drop≤0 即噪声。③ **整组消融 (M4, LOGO)**: 删整个特征组测 gold Δ。脚本: `scripts/run_feature_ablation.py` → `results/tables/feature_{null_importance,perm_importance,group_ablation}.csv` + `noise_feature_list.csv`。

**核心发现：噪声是"轴条件性"的，特征集正交可分但全局不可剪。** 整组消融 gold_delta(负=该组有用)清楚显示两轴需要**互斥**的特征：

| 特征组 | α轴(Ca) | 羰醇轴(Cb) | 4-class(joint) | 结论 |
|---|---|---|---|---|
| **steric**(42d) | **+0.0018** | **−0.1508** | **−0.1995** | α 的噪声 / 羰醇的命脉 |
| **conditions**(44d) | **−0.0908** | **+0.0195** | −0.0595 | α 的命脉 / 羰醇的噪声 |
| **aldpri**(8d) | −0.0075 | −0.0568 | −0.1054 | 羰醇/4-class 关键 |
| chirality(10d) | −0.0058 | +0.0067 | −0.0172 | 羰醇的噪声 |
| delta_chiral(19d) | −0.0094 | +0.0043 | −0.0084 | 羰醇的噪声 |
| chiralenv(21d) | −0.0061 | −0.0134 | −0.0218 | 两轴弱有用 |
| auxiliary(23d) | −0.0119 | −0.0037 | −0.0061 | α 略有用 |
| rgroup(8d) | −0.0078 | −0.0024 | +0.0123 | 4-class 的噪声 |

数据驱动复现 **Zimmerman-Traxler 图景**: **α 轴由条件+辅基控制(烯醇 Z/E 几何, 位阻是噪声)、羰醇轴由位阻+醛优先级控制(条件/手性是噪声)**——一个特征对一个轴是噪声、对另一个轴常是头号信号。M1/M2 top 信号一致佐证(Ca: `chiralenv_ket_*`/`chiral_aux_c4_R`/`feat_act_Bu2BOTf`; Cb/joint: `ald_pri_max_atomic_num`/`ald_pri_priority_proxy`/`ald_pri_alpha_branching`)。

**剪枝验证(非破坏性) FAIL — 全局不可剪**: 保守 AND 判据(M1 且 M2 三轴皆噪声)仅得 4 个纯噪声特征(`Vbur_diff_std`/`feat_solvent_epsilon`/`aux_rg_phenyl`/`chiralenv_ald_n_heavy`)。剪掉后 gold: **Ca +0.0046、Cb +0.0121(单轴均改善)，但 4-class −0.0181(FAIL gate −0.005)**。即：连最噪的 4 个特征也**协同**携带 4-class 弱信号，单轴看是冗余、合起来对 joint 有用——再次印证 L15"数据量/聚合信号 > 局部纯度"。**结论：单一 4-class 模型几乎不可全局精简；但若拆成轴特异模型，可各删对侧一半(α 删 steric 42d +0.0018、Cb 删 conditions 44d +0.0195)，更简且略好。** 另: full-test 上有 6 个特征显示虚假重要性而 gold 无(如 `aux_oxazoline`/`feat_solvent_epsilon`)——L15/L16 的"评测噪声造假信号"第三次出现。

### V4d → V5 变化 (2215 → 2427 VALID 行)

| 指标 | V4d (2215行, 154d) | V5 (2427行, 156d) | 变化 |
|------|-------------------|-------------------|------|
| TSCV champion (XGB) | 0.657 (Optuna) | 0.652 (default) | -0.005 |
| TSCV ET | — | 0.677 | 新基线 |
| Grouped XGB | 0.760 | 0.760 | 持平 |
| Scaffold XGB | — | 0.831 | 新基线 |
| 辅基类型 | 4 valid | 10 valid | +6 |
| 数据量 | 2215 | 2427 | +9.6% |

### V5 Per-Auxiliary 分析 (XGB, grouped)

| 辅基 | Balanced Accuracy | 测试样本 |
|------|:-:|:-:|
| crimmins_oxathione | 0.893 | 33 |
| evans | 0.746-0.783 | 307-328 |
| crimmins_thione | 0.654-0.730 | 65-79 |
| oppolzer | 0.607-0.720 | 27-32 |
| **abiko** ★ | 0.568-0.646 | 24-25 |
| **oxazoline** ★ | 0.500 | 4 |
| **menthyl_ester** ★ | 0.250 | 7 |

---

## V4d 基准 (2026-05-27, 历史)

**数据**: 2334 substrate-controlled aldol reactions (V4d, 从 134K Reaxys 原始数据重建)
**辅基**: Evans (1654) + Crimmins thione (259) + Crimmins oxathione (161) + Oppolzer (141) + Other (105) + Myers (14)
**特征**: Steric (34d) + Conditions (50d) + Auxiliary (6d) + Chirality (7d) + R-group (8d) + ChiralEnv (21d) + AldPriority (8d) + DeltaChiral (16d) + ChiralDet (3d) = **153d**
**MechAware**: BW (112d) / Full (328d) 可选叠加
**划分**: TSCV 4-fold temporal + Scaffold (Murcko) + Grouped random (5 seeds)
**评估**: balanced accuracy (macro-averaged recall, 4-class)
**无泄漏**: role-aware group_id, DRFP 已排除, 手性特征仅从酮 SMILES 提取
**3D syn/anti**: step08b 二面角法计算 label_syn_anti_3d（98.7% 成功率），仅作分析标签

### Champion: ma_bw_xgb_optuna (TSCV = 0.657, 153d, Optuna-tuned)

### Optuna-Tuned Models on 153d (2026-05-28, re-searched on 153d features)

Optuna 200 trials per model on 153d features, full splits benchmark:

| Rank | Model | Category | Dim | TSCV mean±std | Scaffold | Grouped mean±std |
|------|-------|----------|-----|---------------|----------|-----------------|
| 1 | **ma_bw_xgb_optuna** | optuna | 175d | **0.657±0.054** | 0.593 | 0.752±0.020 |
| 2 | et_optuna | optuna | 153d | 0.642±0.052 | 0.583 | 0.730±0.029 |
| 3 | xgb_optuna | optuna | 153d | 0.638±0.052 | **0.612** | **0.760±0.027** |

### Chemprop v2 MPNN Baseline (2026-05-28)

| Rank | Model | Input | TSCV mean±std | Scaffold | Grouped mean±std |
|------|-------|-------|---------------|----------|-----------------|
| 4 | Chemprop+Features | SMILES+153d | 0.626±0.032 | 0.594 | **0.789±0.010** |
| 5 | Chemprop | SMILES only | 0.601±0.032 | 0.616 | 0.730±0.013 |

Key findings:
- **Tree > MPNN on TSCV** (+0.031): handcrafted features generalize better across time
- **MPNN > Tree on Grouped** (+0.037): neural graph representation captures more in-distribution patterns
- **153d features help MPNN**: Chemprop+Features (0.626) >> Chemprop (0.601), confirming 153d provides signal beyond SMILES
- **Low learning rate + strong gamma** remain the dominant Optuna pattern across both 128d and 153d

### Default Models (V4d baseline)

| Rank | Model | Category | Dim | TSCV mean±std | Scaffold | Grouped mean±std |
|------|-------|----------|-----|---------------|----------|-----------------|
| 1 | **v4b_full_et** | v4b | 128d | **0.624±0.031** | **0.613** | 0.738±0.024 |
| 2 | **ma_bw_xgb** | mechaware | 156d | 0.604±0.040 | 0.607 | **0.752±0.022** |
| 3 | v4b_full_xgb | v4b | 128d | 0.602±0.036 | 0.589 | 0.747±0.020 |
| 4 | v4b_full_rf | v4b | 128d | 0.585±0.035 | 0.598 | 0.742±0.028 |
| 5 | ma_full_xgb | mechaware | 372d | 0.578±0.039 | 0.599 | 0.741±0.023 |
| 6 | v4b_full_lgbm | v4b | 128d | 0.568±0.031 | 0.577 | 0.733±0.020 |
| 7 | steronly_xgb | steric | 42d | 0.542±0.016 | 0.482 | 0.657±0.037 |
| 8 | v4b_no_chiral_xgb | ablation | 84d | 0.528±0.044 | 0.447 | 0.644±0.019 |
| 9 | v4b_condaux_xgb | v4b | 65d | 0.436±0.020 | 0.454 | 0.583±0.015 |
| 10 | v4b_chiral_only_xgb | ablation | 7d | 0.401±0.030 | 0.419 | 0.507±0.020 |
| 11 | cond_xgb | baseline | 44d | 0.261±0.039 | 0.309 | 0.426±0.013 |
| 12 | majority | baseline | — | 0.250±0.000 | 0.250 | 0.250±0.000 |

### V4c → V4d 变化 (2288 → 2334 行)

| 指标 | V4c (2288行) | V4d (2334行) | 变化 |
|------|-------------|-------------|------|
| TSCV champion | ma_bw_xgb 0.625 | v4b_full_et 0.624 | -0.001 (冠军换人) |
| TSCV ma_bw_xgb | 0.625 | 0.604 | -0.021 |
| TSCV v4b_full_et | 0.621 | 0.624 | +0.003 |
| Scaffold champion | v4b_full_et 0.607 | v4b_full_et 0.613 | +0.006 |
| Grouped champion | ma_bw_xgb 0.773 | ma_bw_xgb 0.752 | -0.021 |

**分析**: 新增 46 行（保护 OH 模板扩展）后 ma_bw_xgb 轻微下降（-0.021 TSCV），v4b_full_et 轻微上升（+0.003），总体在正常波动范围内。V4d 冠军由 ExtraTrees 128d 夺得，MechAware BW 在 Grouped 上仍最强。

### V4 → V4d-153d 改进历程

| 指标 | V4 (84d) | V4b (120d) | V4d default (128d) | V4d Optuna (153d) | Chemprop+Feat | 总提升 |
|------|----------|-----------|-------------------|------------------|--------------|--------|
| TSCV champion | 0.507 | 0.565 | 0.624 | **0.657** | 0.626 | **+29.6%** |
| Grouped champion | 0.662 | 0.694 | 0.752 | 0.760 | **0.789** | **+19.2%** |
| Scaffold champion | 0.480 | 0.513 | 0.613 | 0.612 | **0.616** | **+28.3%** |

### 新增特征 (44d over V4 baseline)

1. **辅基手性 (7d)**: 从酮 SMILES 提取 CIP — `chiral_dominant_sign` (r=-0.376), `chiral_aux_c4_R` (r=-0.362)
2. **R-基团 (8d)**: Evans C4 取代基分类 (benzyl/isopropyl/phenyl 等) + Oppolzer one-hot
3. **手性环境 (21d)**: 距离分层立体中心计数 (≤3/≤4/≤5 键), 手性梯度, 醛手性环境
4. **醛 CIP 优先级 (8d)**: 芳香性、alpha 支链度、原子序数、卤素/杂原子、链长、优先级代理

### 关键诊断发现

2-class 相对立体 (label_SA) TSCV = **0.746**，4-class 绝对 CIP TSCV = 0.624。差距来自 CIP 优先级效应：同一 Evans-syn 产物，脂肪醛给 class 2 (60%)，芳香醛给 class 3 (37%)。醛 CIP 优先级特征帮助模型学习此映射。

**3D syn/anti 发现**: `label_SA` (CIP 启发式) vs `label_syn_anti_3d` (3D 二面角) 一致率仅 45.6%，证实 CIP R/S 不能可靠推断 syn/anti。3D 二面角分布呈清晰双峰（~±60° syn, ~±180° anti），与 Zimmerman-Traxler TS 模型预测一致。

### 消融实验

| 对比 | TSCV | 说明 |
|------|------|------|
| 128d 全特征 (ET) | **0.624** | V4d default 冠军 |
| MechAware-BW (XGB) | 0.604 | Grouped 最强 (0.752) |
| 84d 无手性 (XGB) | 0.528 | ≈ V4 baseline |
| 仅 7d 手性 (XGB) | 0.401 | 7 个特征已远超随机 |

### Per-Auxiliary Performance (化学空间审计, 2026-05-28)

使用 v4b_full_et 默认参数在全 TSCV 上的 per-auxiliary 分层结果:

| 辅基类型 | n_test | bal_acc | simple_acc | 评价 |
|----------|--------|---------|------------|------|
| **Evans** | 1339 | **0.771** | 0.789 | 优秀 |
| Crimmins thione | 377 | 0.453 | 0.414 | 中等 |
| Crimmins oxathione | 70 | 0.350 | 0.286 | 差 |
| Oppolzer | 108 | 0.371 | 0.315 | 差 |
| Other | 238 | 0.266 | 0.214 | 接近随机 |

TSCV 距离-精度相关: r = **-0.916**（几乎完美负相关）

### Stacking 实验 (2026-05-28)

| 方法 | TSCV | Scaffold | Grouped |
|------|------|----------|---------|
| Stacking (ET+XGB+MA-BW→LR) | 0.617 | 0.577 | 0.741 |
| Level-1 OOF-LR (理想 bound) | **0.660** | — | — |

简单 stacking 未提升（inner-val 20% 数据量不足训练 meta-learner）。

### 关键发现

1. **Optuna 大幅提升 TSCV**: ma_bw_xgb 0.604→0.666 (+10.3%)，默认超参严重过拟合
2. **Evans 单独 0.771**: 辅基异质性是整体性能瓶颈（0.624 vs Evans-only 0.771）
3. **辅基手性是关键缺失**: 加入 7d 手性特征后 condaux 从 0.243→0.436 (+79%)
4. **Steric 仍是核心**: steronly_xgb (0.542) 仍远超 condaux (0.436)，3D 结构不可或缺
5. **3D syn/anti 不是 CIP**: label_SA vs label_syn_anti_3d 一致率仅 45.6%
6. **化学距离决定性能**: TSCV distance-accuracy r=-0.916

### SHAP Feature Importance (153d, 2026-05-28)

| Rank | Feature | Global SHAP | 说明 |
|------|---------|------------|------|
| 1 | ald_pri_priority_proxy | 0.179 | 醛 CIP 优先级代理 |
| 2 | **delta_chiral_0** | **0.103** | 🆕 Delta 手性 PC1 |
| 3 | ald_pri_max_atomic_num | 0.074 | 醛最大原子序数 |
| 4 | ald_pri_alpha_branching | 0.074 | 醛 alpha 支链 |
| 5 | ald_B1_mean | 0.057 | 醛 Sterimol B1 |
| 6 | chiralenv_ald_frac_sp3 | 0.057 | 醛 sp3 比例 |
| 7 | ald_Vbur_total_mean | 0.055 | 醛埋藏体积 |
| 8 | chiralenv_ket_chirality_gradient | 0.054 | 酮手性梯度 |
| 9 | **chiral_det_mean** | **0.050** | 🆕 连续手性行列式 |
| 10 | chiral_aux_c4_R | 0.047 | 辅基 C4 CIP |

Feature group importance:
- steric 34d: 0.559 | aldpri 8d: 0.394 | conditions 50d: 0.357
- **delta_chiral 16d: 0.325** | chiralenv 21d: 0.277 | chirality 10d: 0.178
- **chiral_det 3d: 0.092** | auxiliary 5d: 0.017 | rgroup 8d: 0.004

### Per-Auxiliary Independent Models (2026-05-28)

| 辅基 | n | XGB TSCV | ET TSCV | 评价 |
|------|---|----------|---------|------|
| **Evans** | 1654 | 0.678 | **0.710** | 优秀 |
| Crimmins thione | 259 | 0.447 | **0.567** | 中等 |
| Crimmins oxathione | 161 | 0.353 | 0.348 | 差 |
| Oppolzer | 141 | 0.400 | **0.425** | 差 |
| All (统一) | 2334 | 0.641 | 0.646 | 基线 |

### Error Analysis (2026-05-28)

Confusion matrix (全 TSCV, XGB Optuna):
- Class 0 (RR) recall: 0.486 | Class 1 (RS): 0.768 | Class 2 (SR): 0.786 | Class 3 (SS): 0.615
- 最常见错误: RR→SS (175), RR→RS (114), SS→RR (75) — CIP 优先级翻转
- 高置信度错误 (prob>0.8): 0 个
- 标注错误候选: 259 个

### V3 → V4 数据变化说明

V4 管线完全重建自 134K Reaxys 原始导出:
- **管线可复现**: 13 步清洗(含 step08b 3D syn/anti) + 行级审计
- **范围扩展**: Evans → 6 种手性辅基 (排除手性催化)
- **标签重提取**: CIP 从产物 SMILES 重新提取 (与 V3 有 61% 不一致 — V3 非金标准)
- **DRFP 排除**: 确认产物 @/@@ 直接编码答案 → 标签泄漏
- **3D syn/anti**: step08b 通过 ETKDGv3+MMFF 二面角计算真实 syn/anti（98.7% 成功率）

---

## 历史: V4c 基准 (2026-05-27, 已被 V4d 取代)

V4c 数据 (2288 行) 的冠军: ma_bw_xgb TSCV=0.625±0.040, Grouped=0.773±0.024。

## 历史: V3 基准 (2026-05-16, 已被 V4 取代)

V3 数据 (1655 Evans-only) 的结果见 `archive/` 目录。
冠军为 MechAware-Full (TSCV=0.733)，但基于不可复现的数据管线。
