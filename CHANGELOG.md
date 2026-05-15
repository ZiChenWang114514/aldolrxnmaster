# Changelog

## 2026-05-14: Phase V5 — 交叉项+多模型集成 (负面结果)

### Added
- `chiralaldol/feature_builder.py`: `build_chiralaldol_v5_features()` (87d = V2 75d + 12d cross/Z-E/derived)
- `scripts/run_v5_pipeline.py`: XGB/LGBM/ET + 4-model OOF stacking + auto feature-selection
- 6 交叉项: sin_tau1×ald_Vbur/B5, Vbur_diff×ald_L/B5/ze_z, sin_tau1×ze_z
- 3 衍生: vbur_norm_diff, vbur_si_re_ratio, sin_tau1_sq
- 3 Z/E encoding: ze_z_dominant, ze_mixed, ze_unknown

### Result: NEGATIVE
- V5-XGB temporal **0.758** (不如 V2 0.783)
- V5-Stack temporal **0.694** (最差); V5-Stack scaffold **0.864** (最优)
- 交叉项 r=0.33 on train, 不迁移到 temporal test (2019+)
- **结论: 0.783 是表格方法天花板。V2 (75d) 的 XGBoost depth-6 已隐式学到交互**

## 2026-05-14: Phase C1 — qTS 准过渡态 (负面结果)

### Added
- `chiralaldol/qts_builder.py`: Zimmerman-Traxler 6-元环脚手架 + VDW Gaussian steric 特征 (4d)
- `scripts/run_qts_pipeline.py`: 全管线 (醛基提取 → qTS 计算 → V4 训练)
- 4 通道: si/re × chair/twist-boat, 1822 反应约 3 分钟完成

### Result: NEGATIVE
- V4-XGB temporal **0.628** (退步 -15.5% vs V2 0.783)
- 根因: 近似 ZT 坐标不同醛基的 C=O 方向不同 → si/re 面不一致 (r≈-0.03)
- GFN1/GFN2-xTB 太慢: 50-120s/molecule × 7288 = 60+ h

## 2026-05-14: Phase 11-B1 — xTB 电子描述符 (负面结果)

### Added
- `chiralaldol/xtb_descriptors.py`: GFN2-xTB HOMO/LUMO/gap/dipole/Mulliken/Fukui 12d
- V3-XGB (87d = 24+10+12+35+6), V3b-XGB (80d = 24+10+5+35+6)

### Result: NEGATIVE
- V3-XGB temporal **0.696** (退步 -8.7%); V3b-XGB temporal **0.721** (退步 -6.2%)
- 根因: 烯醇盐 xTB 59% 计算失败 (大分子+带电荷); Evans aldol 是立体控制非电子控制

### Changed
- `scripts/rebuild_comparison.py`: 注册 V3/V3b/V4/V5 共 8 个新模型
- `chiralaldol/feature_builder.py`: 新增 build_chiralaldol_v3/v3b/v5_features()
- Benchmark: 37 → 45 models, 111 → 135 prediction CSVs

## 2026-05-13: Phase 11-A1 — 醛基 3D 立体特征 + ChiralAldolV2 模型

### Added
- **chiralaldol/aldehyde_steric.py**: 醛基 3D 立体描述符模块
  - ALDEHYDE_SMARTS `[CX3;H1]=[OX1]` 匹配醛基碳
  - Sterimol (L/B1/B5) + %Vbur_total for aldehyde R-group (10d features)
  - Boltzmann-weighted ensemble aggregation (mean + std)
  - 667 unique aldehydes, 100% conformer generation success
- **ChiralAldolV2-XGB**: enolate_steric(24) + ald_steric(10) + cond(35) + aux(6) = **75d**
  - Temporal **0.783** (**NEW CHAMPION**, +11.9% over V1-XGB 0.664, +5.8% over V1-Stack 0.725)
  - Scaffold **0.819**, Grouped 0.789
- **ChiralAldolV2-Stack**: V2-XGB + DRFP+Cond → LogReg stacking
  - Temporal 0.782 (essentially same as V2-XGB — DRFP fusion now unnecessary)
  - Scaffold 0.808, Grouped 0.788

### Key Insights
1. Aldehyde steric features are the single most impactful feature addition in the entire project (+11.9% from 10 new features)
2. V2-XGB ≈ V2-Stack confirms that aldehyde features captured what DRFP was compensating for
3. Per-class: C1 (anti-1) improved from 40% → 60%, C3 (syn-S) from 71.6% → 75.3%
4. Benchmark: 35 → 37 models, 105 → 111 prediction CSVs

### Changed
- `scripts/run_chiralaldol_pipeline.py` — added stage3b + V2 model training
- `scripts/rebuild_comparison.py` — registered ChiralAldolV2-XGB, ChiralAldolV2-Stack
- `chiralaldol/feature_builder.py` — added build_chiralaldol_v2_features()
- `data/processed/chiralaldol/aldehyde_steric_features.csv` — 1822×10 features

## 2026-05-12: Phase 11 规划 — 0.725 → 0.90+ 突破路径分析

### 分析与规划
- 深度研读 Gemini 生成的六维分析报告（GFN2-xTB / qTS / SOTA架构 / 目标函数 / 数据增强 / 醛基位阻）
- **核心发现**: 当前 24d 特征全部描述烯醇盐（亲核体），醛（亲电体）完全未建模，这是 #1 盲点
- sin_tau1 (#1 SHAP=0.794) 的优势源于其对 Zimmerman-Traxler TS 几何的隐式近似；若直接提供 TS 几何则可大幅改善
- 确定 Phase 11 优先级：A1（醛基 Sterimol/%Vbur）→ A2（对映体增强可行性）→ A3（dr值确认）→ B1（xTB电子）→ B2（qTS建模）
- 识别报告中对本项目帮助有限的方向：ChemAHNet从零训练（1822样本必然过拟合）、Levenshtein SMILES增强（树模型无用）、多保真建模（无高保真DFT数据）

### 关键参考文献确认
- qTS 方法: Cell Reports Phys. Sci. 2024, DOI: 10.1016/j.xcrp.2024.102043（Universal descriptors of qTS for small-data asymmetric catalysis）
- Δ²-learning: PMC10686042（GFN2-xTB → DFT delta learning for reaction properties）
- 醛位阻量化：直接延伸 Morfeus %Vbur 方法至醛的 R-group

## 2026-05-10: Phase 10 — Late Fusion + SHAP + Error Analysis

### Added
- **ChiralAldol-Stack**: 5-fold OOF stacking of ChiralAldol-XGB + DRFP+Cond-XGB → LogisticRegression meta-learner
  - Temporal **0.725** (NEW CHAMPION, surpasses DRFP+Cond 0.711 by +1.4%)
  - Scaffold 0.709, Grouped Random 0.778
- **ChiralAldol-WtVote**: Performance-weighted soft voting of both base models
  - Temporal 0.710 (ties prev. champion), Grouped Random **0.790** (best ever)
- Early fusion (feature concatenation, 193d) attempted → 0.636, abandoned due to dimensionality imbalance
- **SHAP analysis** on ChiralAldol-XGB (65 features):
  - sin_tau1_mean is #1 feature (SHAP=0.794, 2.7x higher than #2)
  - Vbur_diff_mean ranks #4 (SHAP=0.213)
  - Top-10: 6 from 3D steric, 4 from reaction conditions
  - Output: `notebooks/02_shap_analysis/shap_importance.csv`
- **Error complementarity analysis**:
  - Stack ONLY correct: 20/155 (12.9%) vs DRFP ONLY correct: 5/155 (3.2%) — 4x unique contribution
  - 11 hard cases (7.1%) universally mispredicted by all 7 models, 63.6% are C3 (syn-S)
  - Per-class: C0=85.5%, C2=92.9%, C3=71.6%
  - Output: `notebooks/02_shap_analysis/hard_cases.csv`

### Key Insights
1. Late fusion at prediction-probability level avoids the dimensionality problem of early fusion (DRFP 128d dominated over steric 24d in concatenation)
2. sin_tau1 (auxiliary-enolate torsion angle) is the single most important feature — far more predictive than any individual condition feature
3. 3D steric descriptors provide genuinely orthogonal information to DRFP: 12.9% of test samples are correctly predicted ONLY by the 3D-informed model

### Changed
- `scripts/run_chiralaldol_pipeline.py` — Stage 4 adds WtVote + Stacking models
- `scripts/rebuild_comparison.py` — registered 3 new models (WtVote, Stack, early fusion)
- Benchmark: 32 → 35 models, 96 → 105 prediction CSVs

## 2026-05-10: Phase 9 — ChiralAldol Novel Method

### Added
- **ChiralAldol**: Chemistry-informed 3D steric descriptor method for Evans aldol stereoselectivity prediction
  - M1: Enolate generator — SMARTS-based ketone → Z/E enolate conversion (1801/1822 = 98.8%)
  - M2: Conformer ensemble sampler — ETKDG v3 × 100 conformers + MMFF + RMSD clustering → K representatives with Boltzmann weights
  - M3: 3D steric descriptors — face-dependent %Vbur (si/re), Sterimol L/B1/B5, dihedral angles sin/cos (24 features total)
  - M4: Feature builder — steric(24d) + reaction conditions(35d) + aux chirality(6d) = 65d
- **ChiralAldol-XGB**: temporal 0.664 (rank 4/32), scaffold 0.676, grouped_random 0.757
- **SterOnly-XGB**: steric-only ablation — temporal 0.564
- **CondAux-XGB**: condition+aux ablation (no 3D) — temporal 0.522
- `chiralaldol/` module: enolate_generator.py, conformer_sampler.py, steric_descriptors.py, feature_builder.py, utils.py
- `scripts/run_chiralaldol_pipeline.py` — full pipeline with checkpointing and progress logging
- `data/processed/chiralaldol/`: enolates.csv, conformer_ensembles.pkl (2.8MB), steric_features.csv
- Chemical validation: R-auxiliary → Vbur_diff < 0, S-auxiliary → Vbur_diff > 0 (consistent with Evans model)

### Changed
- `scripts/rebuild_comparison.py` — added ChiralAldol-XGB, SterOnly-XGB, CondAux-XGB to NAME_MAP and MODEL_ORDER
- Benchmark: 29 → 32 models, 87 → 96 prediction CSVs

### Key Result
- ChiralAldol-XGB (0.664) surpasses all Transformer models (best: DistilBERT 0.591) and all end-to-end 3D DL models (ChiENN 0.257, EquiReact 0.318) on temporal split
- 3D steric features alone (SterOnly 0.564) outperform condition+auxiliary features (CondAux 0.522) on temporal split, demonstrating that %Vbur captures genuine stereochemical information
- Gap to champion DRFP+Cond+XGBoost (0.711) is -0.047, within overlapping 95% CIs

## 2026-05-09: Phase 8 — AuxChiral Models

### Added
- **AuxChiral-XGB**: Evans auxiliary R/S + conditions → XGBoost (temporal 0.517)
- AuxChiral+Ald-XGB, AuxChiral-LGBM, CondOnly-XGB, AuxNoBase-XGB, DRFP+Aux+Cond-XGB
- `scripts/run_auxchiral.py`, auxiliary chirality features (6d) in `compute_all.py`
- Code-results consistency audit → TODO.md (P1-P3 issues)

## 2026-05-08: Phase 6 Complete — 23-Model Benchmark

### Added
- **Chemprop v2** (MPNN): reaction SMILES classification, with and without conditions
- **Chemprop+Cond**: Chemprop + 14-dim reaction conditions (metal+solvent) — #2 overall (0.6327)
- **ProtoNet**: Prototypical Networks (meta-learning) on DRFP+Cond features — #5 (0.5707)
- **ChemAHNet-Aldol**: Chemistry-informed DL with cross-attention fusion — #12 (0.4046 temporal, 0.7261 scaffold)
- **ChiENN-Product**: Chirality-aware GNN on product 3D graphs — #23 (0.2226)
- **EquiReact**: 3D equivariant classifier using SchNet-style architecture — #21 (0.2812)
- `scripts/run_chemprop.py`, `run_protonet.py`, `run_chemahnet.py`, `run_chienn_product.py`, `run_equireact.py`
- `scripts/generate_3d_conformers.py` — RDKit ETKDG+MMFF for all molecules
- `data/processed/conformers/conformers.pkl` — 1801/1822 valid 3D conformers
- `conda create -n equireact` — separate environment for 3D models (e3nn, torch-scatter)

### Changed
- `scripts/rebuild_comparison.py` — added NAME_MAP entries for all new models
- Benchmark: 17 → 23 models, all with 3 splits and bootstrap 95% CI

## 2026-05-08: Phase 5b — Fair Comparison Fix

### Fixed
- DistilBERT-Rxn / RoBERTa-Rxn: installed `httpx[socks]` for SOCKS proxy, pre-downloaded models
- MolT5-base: reduced freeze from 10→6 layers, increased lr 1e-5→3e-5, epochs 10→15
- Created `scripts/rebuild_comparison.py` to regenerate unified comparison tables from prediction CSVs
- Created `scripts/rerun_failed_models.py` for targeted re-training

## 2026-05-08: Phase 5a — Chemistry SOTA Models

### Added
- **DRFP+Cond+XGBoost** — new champion (0.7056 temporal bal_acc)
- DRFP+XGBoost, DRFP+LightGBM, DRFP+Cond+XGBoost
- RXNFP+XGBoost, RXNFP+LightGBM, RXNFP+MLP
- ChemBERTa-77M, MolT5-base
- `scripts/precompute_chem_fps.py` — DRFP (2048-dim) + RXNFP (256-dim)
- `aldol-rxn` conda environment (Python 3.11, PyTorch 2.11+cu128)
- Cloned 9 external SOTA repos to `external/`
- Downloaded ChemBERTa-77M + MolT5-base pretrained weights

## 2026-05-08: Phase 1-4 — Data Cleaning + Baseline Benchmark

### Added
- 7-step data cleaning pipeline: consolidate → deduplicate → validate → unify_labels → impute → split → quality_report
- Feature engineering: Morgan FP 2048-bit, RDKit 2D descriptors, reaction conditions (Kamlet-Taft)
- 11 baseline models: XGBoost, LightGBM, RF, 1-NN, 5-NN, Morgan-MLP, DistilBERT-Rxn, RoBERTa-Rxn, XGBoost-FullFP, MajorityClass, Random
- 3 evaluation splits: temporal, scaffold, grouped-random (all group-aware, zero leakage)
- Evaluation: balanced_accuracy, MCC, joint_accuracy, per-class F1, bootstrap 95% CI
- SA inconsistency audit: `notebooks/01_data_cleaning_audit/`
