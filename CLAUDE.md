# AldolRxnMaster

Evans 不对称 aldol 反应 4-class 立体化学预测 — 47+ 模型 benchmark + ChiralAldol + GNN 方法。

## 当前状态 (2026-05-15)

- **数据**: 1801 Evans 反应 (清洗后), 4258 全量反应 (含非 Evans 用于迁移学习)
- **冠军**: ChiralAldolV2-XGB (75d), TSCV 4-fold mean bal_acc=**0.682 ± 0.044**
- **单 temporal split**: 0.69 (清洗后) / 0.783 (旧数据, 因 C1 仅 5 样本不稳定)
- **scaffold**: 0.826, **grouped_random**: 0.807
- **SHAP**: sin_tau1 (#1), Vbur_diff (#4), top-10 中 3D 特征占 6/10
- **表格方法天花板已确认**: 4 次扩展全部失败 (B1 xTB, C1 qTS, V5 cross, feature fusion)
- **GNN 全部失败**: 4 架构 × 3 融合 = 12 组合, 最佳 MPNN+FiLM=0.497 (远低于 V2)
- **手工 3D 特征 > 所有学到的表示**: GNN/Transformer/fingerprint 均不如 75d 手工特征

### Phase A: 数据清洗 ✓
- 1822 → 1801 行 (删 21 缺失分子)
- **chirality_valid bug 修复**: Product_ 列丢失立体标注 → 改用 Raw_Product_Smiles → 100% valid
- **溶剂推断**: 497 unknown → 109 (B→CH2Cl2, Li→THF, etc.)
- **Time-series CV**: 4-fold temporal mean = 0.682 ± 0.044
- 输出: `data/processed/evans_v2_clean.csv`, `data/processed/all_clean.csv` (4258 行)

### Phase B: 图表示构建 ✓
- 4 种 PyG 图: diff (100%), multiview (100%), 3D spatial (99.8%), TS approx (100%)
- Atom mapping 验证: 100% 化学正确 (新 C-C 键 Cb=OH, Ca=α-carbon)
- 输出: `data/processed/graphs/`

### Phase C: GNN 实验 ✓ (负面结果)
- MPNN+FiLM (diff graph): **0.497** (最佳 GNN)
- Equiformer (SE3, 3D): 0.458
- SchNet (3D): 0.389
- GAT 多视图: 修复了 batching 问题, 待重跑
- **根因**: 1801 样本不足以训练 GNN; 手工 3D 特征编码了 ZT 机理知识

### Phase D: 特征融合 ✓ (负面结果)
- V2+DRFP: 0.625 (不如 V2 alone)
- V2+RXNFP: 0.621
- V2+DRFP+RXNFP: 0.548
- **加任何 fingerprint 都降低 temporal 性能** (curse of dimensionality)

## 环境

- **主环境**: `conda activate aldol-rxn` (Python 3.11, PyTorch 2.11+cu128)
- **EquiReact 环境**: `conda activate equireact` (仅 run_equireact.py 使用)
- 不要用 `membrane-vhppi` 环境

## 关键路径

- **清洗后数据**: `data/processed/evans_v2_clean.csv` (1801 行), `data/processed/all_clean.csv` (4258 行)
- **特征**: `data/processed/features/` (labels.csv, reaction_conditions.csv, auxchiral_features.csv, drfp_fps.npz, rxnfp_fps.npz)
- **分割**: `data/processed/splits/` (evans_temporal.json, evans_scaffold.json, evans_grouped_random_seed42.json)
- **图表示**: `data/processed/graphs/` (diff_graphs.pt, multiview_graphs.pt, spatial_3d_graphs.pt, ts_approx_graphs.pt)
- **结果**: `results/predictions/`, `results/tables/` (comparison, gnn_coarse_screening.csv, tscv_results.json)
- **ChiralAldol 方法**: `chiralaldol/` (enolate_generator, conformer_sampler, steric_descriptors, feature_builder, aldehyde_steric, solvent_lookup)
- **GNN 模块**: `chiralaldol/gnn/` (graph_builder, mpnn_diff, gat_multiview, equiformer, schnet_3d, condition_fusion, trainer)
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
