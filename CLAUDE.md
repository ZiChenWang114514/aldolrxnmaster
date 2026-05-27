# AldolRxnMaster

基于底物控制（手性辅基）的 aldol 反应 4-class 立体化学预测。

## 当前状态 (2026-05-27)

- **数据**: V4 管线从 134K Reaxys 原始数据重建，2288 行（6 种辅基类型）
- **辅基类型**: Evans (1636) + Crimmins thione (258) + Crimmins oxathione (139) + Oppolzer (137) + Other (104) + Myers (14)
- **冠军**: **ma_bw_xgb** (156d), TSCV = **0.625 ± 0.040**, Grouped = **0.773 ± 0.024**
- **特征**: Steric(34d) + Conditions(44d) + Auxiliary(6d) + Chirality(7d) + R-group(8d) + ChiralEnv(21d) + AldPriority(8d) = **128d**; MechAware BW(112d)/Full(328d) 可选叠加
- **泄漏已排除**: DRFP 已确认标签泄漏（产物 @/@@ 编码答案），不再使用；手性特征仅从酮 SMILES 提取
- **标签编码**: 4-class `label_joint = Ca × 2 + Cb`; 2-class `label_SA` TSCV=0.746

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11)

## 数据路径

```
data/
  data.csv            原始 Reaxys 导出 (134,027 行)
  clean_v4/           V4 清洗数据
    substrate_aldol_clean.csv  (2288 行, 38 列)
    evans_clean.csv            (1636 行, Evans 子集)
    labels.csv                 (4-class 标签)
    condition_features.csv     (44d 条件特征)
    audit/                     行级审计报告
  features_v4/        V4 特征
    v4_features.csv            (2288 × 128d 完整特征矩阵)
    steric_features.csv        (34d 空间位阻特征)
    labels.csv
    conformers/                构象 pickle 缓存
  splits_v4/          V4 划分
    tscv_fold{1-4}.json        时间序列 CV
    scaffold.json              Murcko 骨架划分
    grouped_seed{42..1024}.json  role-aware 分组划分
  raw/                (已归档到 archive/data_raw_v3/)
  clean/              V3 清洗数据 (已被 V4 取代)
  features/           V3 特征 (已被 V4 取代)
  splits/             V3 划分 (已被 V4 取代)
```

## 脚本 (统一 run_*.py 命名)

```bash
# V4 数据清洗 (12 步: 134K Reaxys → 2179 辅基 aldol)
conda run -n aldol-rxn python scripts/run_rebuild_v4.py

# V4 特征工程 (构象 + steric + chirality + rgroup + chiralenv + aldpri → 128d)
conda run -n aldol-rxn python scripts/run_features_v4.py

# V4 数据划分 (TSCV + scaffold + grouped)
conda run -n aldol-rxn python scripts/run_splits_v4.py

# V4 模型基准 (11 models × 10 splits)
conda run -n aldol-rxn python scripts/run_all_models_v4.py

# (旧) V3 管线 — 仅 Evans, 已被 V4 取代
conda run -n aldol-rxn python scripts/run_rebuild.py
conda run -n aldol-rxn python scripts/run_all_models_v3.py
```

## 约定

- 脚本命名: `scripts/run_*.py`
- Predictions: `results/predictions_v4/{category}/{model_key}_{split}.csv`
  - Categories: v4b, mechaware, steric, ablation, baseline
  - CSV 格式: `idx, y_true, y_pred, prob_0, prob_1, prob_2, prob_3`
- 4-class label: `label_joint = Ca * 2 + Cb` (R=0, S=1)
- 所有 split 用 V4 role-aware group_id (无泄漏)
- 归档: `archive/` (旧数据/旧预测/废弃脚本)
