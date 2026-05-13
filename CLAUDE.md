# AldolRxnMaster

Evans 不对称 aldol 反应 4-class 立体化学预测 — 35 模型 benchmark + ChiralAldol 创新方法。

## 当前状态 (2026-05-13)

- **37 模型 × 3 splits = 111 prediction CSVs**
- **冠军**: ChiralAldolV2-XGB (enolate+aldehyde 3D steric + cond + aux, 75d), temporal bal_acc=**0.783**
- **前冠军**: ChiralAldol-Stack (0.725), 被 V2-XGB 超越 +5.8%
- **数据**: 1822 Evans 反应, 4-class joint Ca×Cb label
- **SHAP**: sin_tau1 (#1), Vbur_diff (#4), top-10 中 3D 特征占 6/10
- **Phase 11-A1 完成**: 醛基 Sterimol/%Vbur 10d → 0.664→0.783 (+11.9%)

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11, PyTorch 2.11+cu128)
- **EquiReact 环境**: `conda activate equireact` (仅 run_equireact.py 使用)
- 不要用 `membrane-vhppi` 环境

## 关键路径

- 数据: `data/processed/features/` (labels.csv, reaction_smiles.csv, drfp_fps.npz, rxnfp_fps.npz, tabular_features.npz)
- 分割: `data/processed/splits/` (evans_temporal.json, evans_scaffold.json, evans_grouped_random_seed42.json)
- 结果: `results/predictions/` (35 models × 3 splits = 105 CSVs), `results/tables/` (comparison 汇总)
- 外部模型: `external/` (9 个 clone 的 SOTA repo)
- **ChiralAldol 方法**: `chiralaldol/` (enolate_generator, conformer_sampler, steric_descriptors, feature_builder)
- **ChiralAldol 数据**: `data/processed/chiralaldol/` (enolates.csv, conformer_ensembles.pkl, steric_features.csv)
- **分析**: `notebooks/02_shap_analysis/` (shap_importance.csv, hard_cases.csv)

## 常用命令

```bash
# ChiralAldol 全管线 (enolate → conformers → steric → models including Stacking/WtVote)
conda run -n aldol-rxn python scripts/run_chiralaldol_pipeline.py

# 跑全部 baseline + fingerprint + transformer 模型
conda run -n aldol-rxn python scripts/run_all_models.py

# 跑单个新模型
conda run -n aldol-rxn python scripts/run_chemprop.py
conda run -n aldol-rxn python scripts/run_protonet.py
conda run -n aldol-rxn python scripts/run_chemahnet.py
conda run -n aldol-rxn python scripts/run_chienn_product.py
conda run -n equireact python scripts/run_equireact.py

# 重建对比表（新增模型后必跑）
conda run -n aldol-rxn python scripts/rebuild_comparison.py

# SHAP + 误差分析
conda run -n aldol-rxn python notebooks/02_shap_analysis/shap_and_error_analysis.py
```

## 约定

- 新模型的 predictions 统一存到 `results/predictions/{model_name}_{split_name}.csv`
- CSV 格式: `idx, y_true, y_pred, prob_0, prob_1, prob_2, prob_3`
- 模型名在 `scripts/rebuild_comparison.py` 的 NAME_MAP 中注册显示名
- 评估函数: `src/aldolrxnmaster/evaluation/metrics.py` 的 `compute_all_metrics()` + `compute_metrics_with_ci()`
- 4-class label: `label_joint = Ca * 2 + Cb` (0=syn-R, 1=anti-1, 2=anti-2, 3=syn-S)
- 所有 split 尊重 group_id，无数据泄漏
