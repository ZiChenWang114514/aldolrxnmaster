# Results — AldolRxnMaster

基于底物控制（手性辅基）的 Aldol 反应 4-class 立体化学预测基准。

## V4c 基准 (2026-05-27)

**数据**: 2288 substrate-controlled aldol reactions (V4, 从 134K Reaxys 原始数据重建)
**辅基**: Evans (1636) + Crimmins thione (258) + Crimmins oxathione (139) + Oppolzer (137) + Other (104) + Myers (14)
**特征**: Steric (34d) + Conditions (44d) + Auxiliary (6d) + Chirality (7d) + R-group (8d) + ChiralEnv (21d) + AldPriority (8d) = **128d**
**MechAware**: BW (112d) / Full (328d) 可选叠加
**划分**: TSCV 4-fold temporal + Scaffold (Murcko) + Grouped random (5 seeds)
**评估**: balanced accuracy (macro-averaged recall, 4-class)
**无泄漏**: role-aware group_id, DRFP 已排除, 手性特征仅从酮 SMILES 提取

### Champion: ma_bw_xgb (TSCV = 0.625, Grouped = 0.773)

### 完整排名

| Rank | Model | Category | Dim | TSCV mean±std | Scaffold | Grouped mean±std |
|------|-------|----------|-----|---------------|----------|-----------------|
| 1 | **ma_bw_xgb** | mechaware | 156d | **0.625±0.040** | 0.595 | **0.773±0.024** |
| 2 | **v4b_full_et** | v4b | 128d | 0.621±0.032 | **0.607** | 0.748±0.028 |
| 3 | ma_full_xgb | mechaware | 372d | 0.602±0.032 | 0.596 | 0.754±0.025 |
| 4 | v4b_full_xgb | v4b | 128d | 0.602±0.038 | 0.594 | 0.759±0.023 |
| 5 | v4b_full_rf | v4b | 128d | 0.587±0.044 | 0.586 | 0.752±0.025 |
| 6 | v4b_full_lgbm | v4b | 128d | 0.571±0.033 | 0.591 | 0.741±0.033 |
| 7 | steronly_xgb | steric | 34d | 0.553±0.017 | 0.523 | 0.675±0.030 |
| 8 | v4b_no_chiral_xgb | ablation | 84d | 0.507±0.024 | 0.454 | 0.662±0.011 |
| 9 | v4b_condaux_xgb | v4b | 65d | 0.450±0.018 | 0.466 | 0.573±0.018 |
| 10 | v4b_chiral_only_xgb | ablation | 7d | 0.410±0.028 | 0.424 | 0.496±0.007 |
| 11 | cond_xgb | baseline | 44d | 0.227±0.035 | 0.313 | 0.405±0.009 |
| 12 | majority | baseline | — | 0.250±0.000 | 0.250 | 0.250±0.000 |

### V4 → V4c 改进历程 (2026-05-27)

| 指标 | V4 (84d) | V4b (120d) | V4c (128d) | 总提升 |
|------|----------|-----------|-----------|--------|
| TSCV champion | 0.507 | 0.565 | **0.625** | **+23.3%** |
| Grouped champion | 0.662 | 0.694 | **0.773** | **+16.8%** |
| Scaffold champion | 0.480 | 0.513 | **0.607** | **+26.5%** |

### 新增特征 (44d over V4 baseline)

1. **辅基手性 (7d)**: 从酮 SMILES 提取 CIP — `chiral_dominant_sign` (r=-0.376), `chiral_aux_c4_R` (r=-0.362)
2. **R-基团 (8d)**: Evans C4 取代基分类 (benzyl/isopropyl/phenyl 等) + Oppolzer one-hot
3. **手性环境 (21d)**: 距离分层立体中心计数 (≤3/≤4/≤5 键), 手性梯度, 醛手性环境
4. **醛 CIP 优先级 (8d)**: 芳香性、alpha 支链度、原子序数、卤素/杂原子、链长、优先级代理

### 关键诊断发现

2-class 相对立体 (label_SA) TSCV = **0.746**，4-class 绝对 CIP TSCV = 0.625。差距来自 CIP 优先级效应：同一 Evans-syn 产物，脂肪醛给 class 2 (60%)，芳香醛给 class 3 (37%)。醛 CIP 优先级特征帮助模型学习此映射。

### 消融实验

| 对比 | TSCV | 说明 |
|------|------|------|
| 128d 全特征 (ET) | **0.621** | V4c 冠军 (无 MechAware) |
| MechAware-BW + chirality (XGB) | **0.625** | 总冠军 |
| 84d 无手性 (XGB) | 0.507 | = V4 baseline |
| 仅 7d 手性 (XGB) | 0.410 | 7 个特征已远超随机 |

### 关键发现

1. **辅基手性是关键缺失**: 加入 7d 手性特征后 condaux 从 0.243→0.450 (+85%)
2. **ExtraTrees 超越 XGBoost**: ET (0.565) > XGB (0.551)，可能因为 120d 下 ET 的随机特征选择更抗过拟合
3. **Scaffold 全面提升**: 0.454→0.513，手性特征帮助泛化到新骨架
4. **Steric 仍是核心**: steronly_xgb (0.500) 仍远超 condaux (0.450)，3D 结构不可或缺

### V3 → V4 数据变化说明

V4 管线完全重建自 134K Reaxys 原始导出:
- **管线可复现**: 12 步清洗 + 行级审计 (V3 的 134K→4751 步骤已丢失)
- **范围扩展**: Evans → 6 种手性辅基 (排除手性催化)
- **标签重提取**: CIP 从产物 SMILES 重新提取 (与 V3 有 61% 不一致 — V3 非金标准)
- **DRFP 排除**: 确认产物 @/@@ 直接编码答案 → 标签泄漏

---

## 历史: V3 基准 (2026-05-16, 已被 V4 取代)

V3 数据 (1655 Evans-only) 的结果见 `archive/` 目录。
冠军为 MechAware-Full (TSCV=0.733)，但基于不可复现的数据管线。
