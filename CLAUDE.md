# AldolRxnMaster

Evans 不对称 aldol 反应 4-class 立体化学预测 — 35 模型 benchmark + ChiralAldol 创新方法。

## 当前状态 (2026-05-14)

- **40 模型 × 3 splits = 120 prediction CSVs**
- **冠军**: ChiralAldolV2-XGB (enolate+aldehyde 3D steric + cond + aux, 75d), temporal bal_acc=**0.783**
- **前冠军**: ChiralAldol-Stack (0.725), 被 V2-XGB 超越 +5.8%
- **数据**: 1822 Evans 反应, 4-class joint Ca×Cb label
- **SHAP**: sin_tau1 (#1), Vbur_diff (#4), top-10 中 3D 特征占 6/10
- **Phase 11-A1 完成**: 醛基 Sterimol/%Vbur 10d → 0.664→0.783 (+11.9%)
- **Phase 11-B1 完成 (负面结果)**: GFN2-xTB 电子描述符无增益
  - V3-XGB (87d, 全量 xTB): temporal **0.696** (退步 -8.7%)
  - V3b-XGB (80d, 5d clean ald xTB): temporal **0.721** (仍不如 V2)
  - 根因: 烯醇盐 xTB 59% 计算失败；Evans aldol 是立体控制反应，HOMO/LUMO 无关
- **Phase C1 完成 (负面结果)**: qTS 准过渡态 VDW steric 特征无增益
  - V4-XGB (79d = 75d V2 + 4d qTS VDW): temporal **0.628** (退步 -15.5%)
  - 根因: 近似 ZT 坐标构建中 si/re 面分配不一致 (不同醛基 C=O 方向不同)；
    VDW clash 特征与 label 相关性 r ≈ −0.03 (接近零)
  - GFN2/GFN1-xTB 真实 TS 优化速度 50-120s/分子，1822×4 = ~60h，不可行
  - **结论**: V2 (75d) 仍是最优方案；qTS 需要更严格的 ZT 环坐标 + 正确的 si/re 几何
- **下一步候选**:
  1. **更好的 qTS**: 正确 ZT 6-元环坐标 (chair/twist-boat 显式环几何)，确保 si/re = 等轴/直轴 R 基
  2. **交叉项特征**: (Vbur_si - Vbur_re) × ald_B5 = 面选择性与醛基体积的直积
  3. **基于产物构建 TS**: 从产物逆向设计 TS 几何，更准确但更复杂

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
