# Reproduction Guide

从零复现 AldolRxnMaster V3 基准结果。

## 前置条件

- Linux, CUDA GPU (可选，XGBoost 模型只需 CPU)
- Conda (Anaconda/Miniconda)
- `git clone` 本项目

---

## Step 0: 环境

```bash
conda create -n aldol-rxn python=3.11 -y
conda activate aldol-rxn

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install rdkit xgboost lightgbm scikit-learn pandas numpy scipy
pip install transformers

# 外部包
cd external/drfp && pip install -e . && cd ../..
```

## Step 1: V3 数据重建 (16 步管线)

```bash
conda run -n aldol-rxn python scripts/run_rebuild.py --date-suffix 20260516
```

**输入**: `data/raw/alldata.csv` (4751 行)  
**输出**:
- `data/clean/evans_clean.csv` (1655 行)
- `data/features/` (v3_features.csv, labels.csv, condition_features.csv, steric_features.csv)
- `data/splits/` (tscv_fold{1-4}.json, scaffold.json, grouped_seed*.json)
- `data/audit/` (row_audit.csv, summary_report.json)

## Step 2: MechAware Z/E 构象

```bash
conda run -n aldol-rxn python scripts/run_mechaware_conformers.py
```

**输出**: `data/features/mechaware/{ketone,z_enolate,e_enolate}_steric.csv`

## Step 3: 全部模型基准

```bash
conda run -n aldol-rxn python scripts/run_all_models_v3.py
```

**输出**: 
- `results/predictions/{steric,fp,baseline}/{model}_{split}.csv`
- `results/tables/benchmark_v3_20260516.csv`

## Step 4: 公平对比 + 泄漏检测

```bash
conda run -n aldol-rxn python scripts/run_comparison.py
```

**输出**: `results/tables/fair_comparison_20260516.csv`

---

## 预期结果 (V3 基准, 2026-05-16)

| Rank | Model | TSCV mean±std |
|------|-------|---------------|
| 1 | MechAware-Full | 0.733 ± 0.074 |
| 2 | MechAware-BW | 0.725 ± 0.060 |
| 3 | ChiralAldol-V2-XGB | 0.696 ± 0.023 |
| 4 | CondAux-XGB | 0.689 ± 0.086 |
| 5 | ChiralAldol-V2-RF | 0.680 ± 0.049 |

## 注意事项

- **DRFP 有标签泄漏**: 产物 @/@@ 编码了答案。去除后真实 TSCV=0.577。
- **旧 scaffold 0.826 有泄漏**: group_id 修复后降至 0.757。
- `archive/` 目录包含旧管线和旧结果（仅供参考，不应使用）。
