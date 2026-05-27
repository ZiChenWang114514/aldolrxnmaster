# Results — AldolRxnMaster

基于底物控制（手性辅基）的 Aldol 反应 4-class 立体化学预测基准。

## V4d 基准 (2026-05-27)

**数据**: 2334 substrate-controlled aldol reactions (V4d, 从 134K Reaxys 原始数据重建)
**辅基**: Evans (1654) + Crimmins thione (259) + Crimmins oxathione (161) + Oppolzer (141) + Other (105) + Myers (14)
**特征**: Steric (34d) + Conditions (44d) + Auxiliary (6d) + Chirality (7d) + R-group (8d) + ChiralEnv (21d) + AldPriority (8d) = **128d**
**MechAware**: BW (112d) / Full (328d) 可选叠加
**划分**: TSCV 4-fold temporal + Scaffold (Murcko) + Grouped random (5 seeds)
**评估**: balanced accuracy (macro-averaged recall, 4-class)
**无泄漏**: role-aware group_id, DRFP 已排除, 手性特征仅从酮 SMILES 提取
**3D syn/anti**: step08b 二面角法计算 label_syn_anti_3d（98.7% 成功率），仅作分析标签

### Champion: v4b_full_et (TSCV = 0.624, Scaffold = 0.613)

### 完整排名

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

### V4 → V4d 改进历程

| 指标 | V4 (84d) | V4b (120d) | V4c (128d) | V4d (128d, 2334行) | 总提升 |
|------|----------|-----------|-----------|-------------------|--------|
| TSCV champion | 0.507 | 0.565 | 0.625 | **0.624** | **+23.1%** |
| Grouped champion | 0.662 | 0.694 | 0.773 | **0.752** | **+13.6%** |
| Scaffold champion | 0.480 | 0.513 | 0.607 | **0.613** | **+27.7%** |

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
| 128d 全特征 (ET) | **0.624** | V4d 冠军 |
| MechAware-BW (XGB) | 0.604 | Grouped 最强 (0.752) |
| 84d 无手性 (XGB) | 0.528 | ≈ V4 baseline |
| 仅 7d 手性 (XGB) | 0.401 | 7 个特征已远超随机 |

### 关键发现

1. **ExtraTrees 最稳定**: ET (0.624 TSCV, 0.613 scaffold) 在所有 OOD 指标上最均衡
2. **MechAware BW 在 Grouped 最强**: 0.752（同分布最优），但 TSCV 0.604 低于 ET
3. **辅基手性是关键缺失**: 加入 7d 手性特征后 condaux 从 0.243→0.436 (+79%)
4. **Steric 仍是核心**: steronly_xgb (0.542) 仍远超 condaux (0.436)，3D 结构不可或缺
5. **3D syn/anti 不是 CIP**: label_SA vs label_syn_anti_3d 一致率仅 45.6%

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
