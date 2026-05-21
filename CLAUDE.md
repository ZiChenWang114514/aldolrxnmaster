# AldolRxnMaster

Evans 不对称 aldol 反应 4-class 立体化学预测。

## 当前状态 (2026-05-16)

- **数据**: 1655 Evans reactions (V3 严格清洗, 从 4751 原始行)
- **冠军**: **MechAware-Full** (151d), TSCV 4-fold mean bal_acc = **0.733 ± 0.074**
- **方法**: 显式 Z/E 烯醇盐构象分离 + base-dependent 加权 + XGBoost
- **泄漏已修复**: 旧 scaffold 0.826 确认有 group_id 泄漏 (真实 0.757); DRFP 0.87 有标签泄漏
- **真实天花板**: ~0.73 (TSCV), ~0.74 (scaffold/grouped)

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11)
- 不要用其他环境

## 数据路径

```
data/
  raw/          原始 alldata.csv (4751 行)
  clean/        evans_clean.csv (1655), non_evans_clean.csv (430)
  features/     v3_features.csv (87d), labels.csv, condition_features.csv,
                steric_features.csv, mechaware/{ketone,z,e}_steric.csv
  splits/       tscv_fold{1-4}.json, scaffold.json, grouped_seed{42..1024}.json
  interim/      中间产物 (调试用)
  audit/        行级审计报告
```

## 脚本 (统一 run_*.py 命名)

```bash
# MechAware 管线 (Z/E 构象 → steric → 模型)
conda run -n aldol-rxn python scripts/run_mechaware_conformers.py
conda run -n aldol-rxn python scripts/run_mechaware.py

# V3 数据重建 (16 步全管线)
conda run -n aldol-rxn python scripts/run_rebuild.py

# 全部模型基准 (15 active models × 10 splits)
conda run -n aldol-rxn python scripts/run_all_models_v3.py

# 公平对比 (同数据同划分 + 泄漏检测)
conda run -n aldol-rxn python scripts/run_comparison.py

# ChiralAldol 原始管线
conda run -n aldol-rxn python scripts/run_chiralaldol.py
```

## 约定

- 脚本命名: `scripts/run_*.py`
- Predictions: `results/predictions/{category}/{model_key}_{split}.csv`
  - Categories: steric, fp, gnn, meta, baseline
  - CSV 格式: `idx, y_true, y_pred, prob_0, prob_1, prob_2, prob_3`
- 模型注册: `MODEL_REGISTRY.md` (全部 47+ 模型含状态)
- 4-class label: `label_joint = Ca * 2 + Cb`
- 所有 split 用 V3 role-aware group_id (无泄漏)
- 归档: `archive/` (旧数据/旧预测/废弃脚本)

## 关键发现

- Z/E 烯醇盐分离 steric 特征 (+5.3% TSCV vs V2)
- DRFP 有标签泄漏 (产物 @/@@ 编码了答案)
- 手工 3D 特征 >> GNN/Transformer/Fingerprint (在小数据量下)
