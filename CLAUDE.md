# AldolRxnMaster

基于底物控制（手性辅基）的 aldol 反应 4-class 立体化学预测。

## 当前状态 (2026-05-27)

- **数据**: V4d 管线从 134K Reaxys 原始数据重建，**2334 行**（6 种辅基类型）
- **辅基类型**: Evans (1654) + Crimmins thione (259) + Crimmins oxathione (161) + Oppolzer (141) + Other (105) + Myers (14)
- **冠军**: **ma_bw_xgb** (156d), TSCV = **0.625 ± 0.040**, Grouped = **0.773 ± 0.024** (基于 V4c/2288 行，V4d 重新基准进行中)
- **特征**: Steric(34d) + Conditions(44d) + Auxiliary(6d) + Chirality(7d) + R-group(8d) + ChiralEnv(21d) + AldPriority(8d) = **128d**; MechAware BW(112d)/Full(328d) 可选叠加
- **泄漏已排除**: DRFP 已确认标签泄漏（产物 @/@@ 编码答案），不再使用；手性特征仅从酮 SMILES 提取
- **标签编码**: 4-class `label_joint = Ca × 2 + Cb` (R=0, S=1); 2-class `label_SA` (CIP 启发式，非 syn/anti)
- **3D syn/anti**: `label_syn_anti_3d` 由 step08b 3D 二面角法计算（98.7% 成功率），仅作分析标签不入 ML 特征

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11)

## 数据路径

```
data/
  data.csv            原始 Reaxys 导出 (134,027 行)
  clean_v4/           V4d 清洗数据
    substrate_aldol_clean.csv  (2334 行, 42 列)
    evans_clean.csv            (1654 行, Evans 子集)
    labels.csv                 (标签: Ca, Cb, SA, joint + 3D syn/anti)
    condition_features.csv     (44d 条件特征)
    audit/                     行级审计报告
  features_v4/        V4 特征
    v4_features.csv            (2334 × 128d 完整特征矩阵)
    steric_features.csv        (34d 空间位阻特征)
    labels.csv
    conformers/                构象 pickle 缓存
    v4_mechaware_bw.csv        (112d 基加权 MechAware)
    v4_mechaware_full.csv      (328d 完整 Z/E MechAware)
  splits_v4/          V4 划分
    tscv_fold{1-4}.json        时间序列 CV
    scaffold.json              Murcko 骨架划分
    grouped_seed{42..1024}.json  role-aware 分组划分
  raw/                (已归档到 archive/data_raw_v3/)
  clean/              V3 清洗数据 (已被 V4 取代)
  features/           V3 特征 (已被 V4 取代)
  splits/             V3 划分 (已被 V4 取代)
```

## 清洗管线 (13 步)

```
step01 Load+Filter → step02 Parse → step03 Auxiliary → step04 Canonicalize →
step05 Stereo → step06 AtomMapping → step07 LabelExtract → step08 LabelValidate →
step08b 3D_SynAnti → step09 Dedup → step10 Conditions → step11 CondEngineer →
step12 AuditOutput
```

## 关键标签列

| 列名 | 含义 | 值域 | 用途 |
|------|------|------|------|
| `label_joint` | 4-class 绝对 CIP (Ca×2+Cb) | 0,1,2,3 | **主 ML 目标** |
| `label_Ca` / `label_Cb` | Ca/Cb CIP 编码 | 0(R), 1(S) | 标签组件 |
| `label_SA` | int(Ca==Cb) CIP 启发式 | 0, 1 | 向后兼容，**不是** syn/anti |
| `label_syn_anti_3d` | 3D 二面角真实 syn/anti | 1(syn), 0(anti), None(失败) | 分析标签 |
| `dihedral_oh_cb_ca_co` | OH-Cb-Ca-C(=O) 二面角 | -180°~+180° | 分析 |
| `conformer_energy` | MMFF 力场能量 | kcal/mol | 分析 |
| `synanti_confidence` | syn/anti 置信度 | 0~1 | 分析 |

## 脚本 (统一 run_*.py 命名)

```bash
# V4 数据清洗 (13 步含 step08b: 134K Reaxys → 2334 辅基 aldol)
conda run -n aldol-rxn python scripts/run_rebuild_v4.py

# V4 特征工程 (构象 + steric + chirality + rgroup + chiralenv + aldpri → 128d)
conda run -n aldol-rxn python scripts/run_features_v4.py

# V4 数据划分 (TSCV + scaffold + grouped)
conda run -n aldol-rxn python scripts/run_splits_v4.py

# V4 MechAware 特征 (Z/E 分离 + BW 加权)
conda run -n aldol-rxn python scripts/run_mechaware_v4.py

# V4 模型基准 (11 models × 10 splits)
conda run -n aldol-rxn python scripts/run_all_models_v4.py
```

## 约定

- 脚本命名: `scripts/run_*.py`
- Predictions: `results/predictions_v4/{category}/{model_key}_{split}.csv`
  - Categories: v4b, mechaware, steric, ablation, baseline
  - CSV 格式: `idx, y_true, y_pred, prob_0, prob_1, prob_2, prob_3`
- 4-class label: `label_joint = Ca * 2 + Cb` (R=0, S=1)
- 所有 split 用 V4 role-aware group_id (无泄漏)
- 归档: `archive/` (旧数据/旧预测/废弃脚本)
