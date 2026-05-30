# AldolRxnMaster

基于底物控制（手性辅基）的 aldol 反应 4-class 立体化学预测。

## 当前状态 (2026-05-30)

- **数据**: V5 管线从 134K Reaxys 原始数据重建，**2434 行**（9 种辅基类型 + 7 other）
- **辅基类型**: Evans (1661) + Crimmins thione (260) + Crimmins oxathione (169) + Oppolzer (141) + **Abiko (127)** + **Menthyl ester (32)** + **Oxazoline (21)** + **Myers (16)** + Other (7)
- **V5 新增**: 5 种新辅基 SMARTS (abiko/menthyl/borneol/oxazoline/super_quat)，宽泛 Myers SMARTS，酯型/恶唑啉型产物 SMARTS，ynamide 排除，step08 标签恢复
- **VALID_AUXILIARIES**: 10 种（+6 vs V4），**2427 行** (vs V4 的 2215 行, +9.6%)
- **冠军** (V5): **v4b_full_xgb** (156d), TSCV = **0.652±0.041**, Grouped = **0.760±0.016**, Scaffold = **0.831**
- **特征**: Steric(34d) + Conditions(44d) + Aux one-hot(9d) + Aux mechanistic(6d) + Chirality(7d) + R-group(7d) + ChiralEnv(21d) + AldPriority(8d) + DeltaChiral(16d) + ChiralDet(3d) + n_stereo(1d) = **156d**
- **泄漏已排除**: DRFP 已确认标签泄漏（产物 @/@@ 编码答案），不再使用；手性特征仅从酮 SMILES 提取
- **标签编码**: 4-class `label_joint = Ca × 2 + Cb` (R=0, S=1); 2-class `label_SA` (CIP 启发式，非 syn/anti)
- **3D syn/anti**: `label_syn_anti_3d` 由 step08b 3D 二面角法计算（97.9% 成功率），仅作分析标签不入 ML 特征

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11)

## 项目结构

```
chiralaldol/              核心包
  config.py               路径常量 (FEAT_DIR, SPLITS_DIR, PRED_DIR, ...)
  data_io.py              数据加载 (prepare_Xy, load_splits, load_mechaware_bw, ...)
  model_trainers.py       模型训练 (train_xgb, train_et, train_rf, train_lgbm, MajorityClassifier)
  feature_registry.py     特征子集 (select_features, FEATURE_SUBSETS)
  rebuild/                V5 数据清洗管线 (13 步, 含辅基扩展+标签恢复)
  rebuild_legacy/         V3 计算工具 (conformers, steric — 被 run_features.py 引用)
  gnn/                    GNN 模块 (equiformer, schnet_3d, etc.)
  steric_descriptors.py   Sterimol/Vbur 计算
  conformer_sampler.py    3D 构象采样
  ...

scripts/                  可执行脚本 (15 个)
  run_rebuild.py          数据清洗 (134K Reaxys → 2434, 9 种辅基)
  run_features.py         特征工程 (→ 156d)
  run_splits.py           数据划分 (TSCV + scaffold + grouped)
  run_mechaware.py        MechAware 特征 (Z/E 分离 + BW 加权)
  run_benchmark.py        模型基准 (11 models × 10 splits)
  run_benchmark_full.py   完整基准 (含 MechAware)
  run_benchmark_evans.py  Evans-only 基准
  run_optuna.py           Optuna 超参搜索
  run_optuna_benchmark.py Optuna 参数全 split 评估
  run_stacking.py         Stacking 集成
  run_chemprop.py         Chemprop MPNN baseline
  run_aux_models.py       Per-auxiliary 独立模型
  run_shap_analysis.py    SHAP 特征重要性
  run_error_analysis.py   错误分析
  run_chem_space_audit.py 化学空间审计

data/
  clean_v5/               清洗数据 (2434 行, 9 种辅基)
  features_v5/            156d 特征
  splits_v5/              TSCV + scaffold + grouped 划分

results/
  predictions_v5/         预测 CSV (v4b, mechaware, steric, ablation, baseline)
  optuna/                 Optuna 最优参数
  tables/                 汇总表
  shap/                   SHAP 分析
  chem_space_audit/       化学空间诊断

archive/                  归档 (V3 数据/脚本/notebooks)
```

## 数据路径

```
data/
  clean_v5/
    substrate_aldol_clean.csv  (2434 行, 42 列, 9 种辅基)
    evans_clean.csv            (1661 行, Evans 子集)
    labels.csv                 (标签: Ca, Cb, SA, joint + 3D syn/anti)
    condition_features.csv     (44d 条件特征)
    audit/                     行级审计报告
  features_v5/
    v5_features.csv            (2427 × 156d 完整特征矩阵)
    steric_features.csv        (34d 空间位阻特征)
    labels.csv
    conformers/                构象 pickle 缓存
  splits_v5/
    tscv_fold{1-4}.json        时间序列 CV
    scaffold.json              Murcko 骨架划分
    grouped_seed{42..1024}.json  role-aware 分组划分
  clean_v4/                    V4 历史数据 (2334 行, 保留)
  features_v4/                 V4 历史特征 (154d, 保留)
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

## 脚本

```bash
# 数据清洗 (13 步含 step08b: 134K Reaxys → 2434 辅基 aldol, 9 种辅基)
conda run -n aldol-rxn python scripts/run_rebuild.py

# 特征工程 (构象 + steric + chirality + rgroup + chiralenv + aldpri + delta_chiral + chiral_det → 156d)
conda run -n aldol-rxn python scripts/run_features.py

# 数据划分 (TSCV + scaffold + grouped)
conda run -n aldol-rxn python scripts/run_splits.py

# MechAware 特征 (Z/E 分离 + BW 加权)
conda run -n aldol-rxn python scripts/run_mechaware.py

# 模型基准 (11 models × 10 splits)
conda run -n aldol-rxn python scripts/run_benchmark.py

# Optuna 超参搜索
conda run -n aldol-rxn python scripts/run_optuna.py

# Chemprop MPNN baseline
CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_chemprop.py
```

## 约定

- 脚本命名: `scripts/run_*.py`
- 共享模块: `chiralaldol/{config,data_io,model_trainers,feature_registry}.py`
- Predictions: `results/predictions_v5/{category}/{model_key}_{split}.csv`
  - Categories: v4b, mechaware, steric, ablation, baseline
  - CSV 格式: `idx, y_true, y_pred, prob_0, prob_1, prob_2, prob_3`
- 4-class label: `label_joint = Ca * 2 + Cb` (R=0, S=1)
- 所有 split 用 V5 role-aware group_id (无泄漏)
- 归档: `archive/` (旧数据/旧预测/废弃脚本), `data/clean_v4/` + `data/features_v4/` (V4 历史)
