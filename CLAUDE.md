# AldolRxnMaster

基于底物控制（手性辅基）的 aldol 反应 4-class 立体化学预测。

## 当前状态 (2026-06-07)

- **诚实按轴评测 (2026-06-07)**: 4-class CIP ≈ **α轴 × 羰醇轴**。gold-test(可信标签)上 **α(Ca)~0.94 / 羰醇(Cb)~0.82 / 4-class~0.79**。α 误差大半是评测标签噪声(模型已学对)；**羰醇轴是真实化学瓶颈**(gold≈non-gold)。机理重标注/标签去噪均证否(见 LESSONS L14/L15)。冲 90% 需两轴都>0.95，羰醇受限~0.82。详见 RESULTS.md 诚实评测节。
- **数据**: V5 管线从 134K Reaxys 原始数据重建，**2434 行**（9 种辅基类型 + 7 other）
- **辅基类型**: Evans (1661) + Crimmins thione (260) + Crimmins oxathione (169) + Oppolzer (141) + **Abiko (127)** + **Menthyl ester (32)** + **Oxazoline (21)** + **Myers (16)** + Other (7)
- **VALID_AUXILIARIES**: 10 种（+6 vs V4），**2427 行** (vs V4 的 2215 行, +9.6%)
- **冠军** (V5 Evans-only): **ZT-Chiral+feat** (ZT 图 + 156d global-feat), TSCV = **0.818**
- **冠军** (V5 全数据集): **XGB Optuna** (156d), TSCV = **0.739±0.074**, Grouped = **0.760**
- **特征**: Steric(34d) + Conditions(44d) + Aux one-hot(9d) + Aux mechanistic(6d) + Chirality(7d) + R-group(7d) + ChiralEnv(21d) + AldPriority(8d) + DeltaChiral(16d) + ChiralDet(3d) + n_stereo(1d) = **156d**
- **SPMS 特征**: 球面投影位阻 (16d stats) + Si/Re Face Map (24d)
- **泄漏已排除**: DRFP 已确认标签泄漏（产物 @/@@ 编码答案），不再使用；手性特征仅从酮 SMILES 提取
- **标签编码**: 4-class `label_joint = Ca × 2 + Cb` (R=0, S=1); 2-class `label_SA` (CIP 启发式，非 syn/anti)
- **3D syn/anti**: `label_syn_anti_3d` 由 step08b 3D 二面角法计算（97.9% 成功率），仅作分析标签不入 ML 特征

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11)
- **包安装**: `pip install -e .` (editable mode, chiralaldol 包)

## 项目结构

```
chiralaldol/              核心包 (pip install -e . 可直接 import)
  config.py               路径常量 (FEAT_DIR, SPLITS_DIR, PRED_DIR, ...)
  data_io.py              数据加载 (prepare_Xy, load_splits, load_mechaware_bw, ...)
  model_trainers.py       模型训练 (train_xgb, train_et, train_rf, train_lgbm, MajorityClassifier)
  feature_registry.py     特征子集 (select_features, FEATURE_SUBSETS)
  utils.py                共享工具 (clean_mol, ACYL_ALPHA_SMARTS, VDW_RADII, ...)
  steric_descriptors.py   Sterimol/Vbur 计算
  conformer_sampler.py    3D 构象采样
  enolate_generator.py    酮→烯醇化 SMILES 转换
  ze_enolate_generator.py Z/E 烯醇化构象生成
  aldehyde_steric.py      醛基位阻描述符
  solvent_lookup.py       Kamlet-Taft 溶剂参数
  spms.py                 SPMS 球面投影位阻
  spms_compressor.py      SPMS 压缩 (AE/PCA/stats)
  face_steric_map.py      Si/Re 面立体地图
  zt_graph_builder.py     ZT 过渡态图构建
  zt_features.py          ZT 图 → 32d 平坦特征
  zt_3d_coords.py         ZT 椅式 3D 坐标
  rebuild/                V5 数据清洗管线 (13 步, 含辅基扩展+标签恢复)
  rebuild_legacy/         V3 计算工具 (conformers, steric — 被 run_features.py 引用)
  gnn/                    GNN 模块 (zt_models, graph_builder, trainer, ...)

scripts/                  可执行脚本 (22 个)
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
  run_zt_chemprop.py      ZT+Chemprop (SMILES+156d+ZT 32d)
  run_build_zt_graphs.py  构建 ZT 过渡态图 (Evans subset)
  run_zt_gnn.py           ZT-GIN / ZT-GAT / ZT-Chiral GNN
  run_zt_chienn.py        ChiENN 外部模型集成
  run_spms_features.py    SPMS 球面投影计算 (Phase A/B/C)
  run_spms_benchmark.py   SPMS 特征 × Tree 模型基准
  run_spms_remaining.py   SPMS + ZT-GNN / Chemprop 组合实验
  run_aux_models.py       Per-auxiliary 独立模型
  run_shap_analysis.py    SHAP 特征重要性
  run_error_analysis.py   错误分析
  run_chem_space_audit.py 化学空间审计

data/
  clean_v5/               清洗数据 (2434 行, 9 种辅基)
  features_v5/            156d 特征 + mechaware + conformers
    spms/                 SPMS 球面投影矩阵 + face map
    zt_graphs/            ZT 过渡态图 (evans_zt_graphs.pkl)
  splits_v5/              TSCV + scaffold + grouped 划分

results/
  predictions_v5/         预测 CSV
    v4b/                  XGB/ET/RF/LGBM (156d)
    mechaware/            MechAware-BW/Full-XGB
    steric/               位阻-only
    ablation/             特征消融
    baseline/             多数类/KNN
    chemprop/             Chemprop MPNN
    zt_chemprop/          ZT+Chemprop
    zt_gnn/               ZT-GIN/GAT/Chiral (+feat/+face_map)
    optuna/               Optuna-tuned
    spms/                 SPMS Tree 模型
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
    v5_features_spms.csv       (2427 × 172d SPMS 增强特征)
    steric_features.csv        (34d 空间位阻特征)
    mechaware_bw.csv           (BW 加权 MechAware 特征)
    mechaware_full.csv         (完整 MechAware 特征)
    labels.csv
    conformers/                构象 pickle 缓存
    mechaware/                 Z/E 构象 + 位阻 CSV
    spms/                      SPMS 矩阵 + face map
    zt_graphs/                 evans_zt_graphs.pkl
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

## V5 模型排行榜 (TSCV balanced accuracy)

| Rank | 模型 | TSCV | 范围 |
|------|------|------|------|
| 1 | ZT-Chiral+feat | 0.818 | Evans-only |
| 2 | ZT-Chemprop SMILES+156d+ZT | 0.785 | Evans-only |
| 3 | ZT-ComENet+feat | 0.784 | Evans-only |
| 4 | ZT-GIN+face_map | 0.783 | Evans-only |
| 5 | ZT-Hybrid+feat | 0.776 | Evans-only |
| 6 | ZT-GAT+feat | 0.753 | Evans-only |
| 7 | ZT-GIN+feat | 0.731 | Evans-only |
| 8 | XGB Optuna | 0.739 | 全数据集 |
| 9 | ET Optuna | 0.722 | 全数据集 |
| 10 | XGB+face_map | 0.685 | 全数据集 |

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

# ZT 过渡态图构建 + GNN
conda run -n aldol-rxn python scripts/run_build_zt_graphs.py
CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py --model all --epochs 100 --use-global-feat

# SPMS 球面投影特征
conda run -n aldol-rxn python scripts/run_spms_features.py --phase A
conda run -n aldol-rxn python scripts/run_spms_features.py --phase B --method stats
conda run -n aldol-rxn python scripts/run_spms_features.py --phase C
```

## 约定

- 脚本命名: `scripts/run_*.py`
- 共享模块: `chiralaldol/{config,data_io,model_trainers,feature_registry,utils}.py`
- 共享常量: `ACYL_ALPHA_SMARTS` 定义在 `utils.py`，其他模块统一 import
- Predictions: `results/predictions_v5/{category}/{model_key}_{split}.csv`
  - Categories: v4b, mechaware, steric, ablation, baseline, chemprop, zt_chemprop, zt_gnn, optuna, spms
  - CSV 格式: `idx, y_true, y_pred, prob_0, prob_1, prob_2, prob_3`
- 4-class label: `label_joint = Ca * 2 + Cb` (R=0, S=1)
- 所有 split 用 V5 role-aware group_id (无泄漏)
- 归档: `archive/` (旧数据/旧预测/废弃脚本), `data/clean_v4/` + `data/features_v4/` (V4 历史)
