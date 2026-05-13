# AldolRxnMaster — Publication Roadmap (JACS Target)

## Project Status (2026-05-10)

- **Data**: 4751 → 4447 (deduplicated) → 1822 Evans clean, 7-step pipeline
- **Features**: Morgan FP, DRFP, RXNFP, RDKit desc, reaction conditions (35d), aux chirality (6d), **3D steric descriptors (24d, NEW)**
- **Models**: 32 unique × 3 splits = 96 prediction CSVs
- **NEW Champion**: ChiralAldol-Stack, temporal bal_acc=**0.725** (stacking of 3D steric + DRFP)
- **Previous Champion**: DRFP+Cond+XGBoost, temporal bal_acc=0.711
- **Standalone novel method**: ChiralAldol-XGB, temporal bal_acc=0.664 (rank 4/35)
- **Gap to practical utility**: 0.725 → 0.90+ needed for real synthetic planning

---

## Publication Tasks (JACS/Nature Comms target)

### T1: ChiralAldol + DRFP Fusion Model [DONE]

- [x] Early fusion (feature concatenation 193d) → 0.636, FAILED (dimensionality imbalance)
- [x] Late fusion — Weighted Voting → 0.710 (ties prev. champion)
- [x] Late fusion — Stacking (5-fold OOF + LogReg) → **0.725 (NEW CHAMPION)**
- [x] Paper story confirmed: "3D steric + 2D fingerprint are complementary; stacking fusion surpasses either alone"

### T2: SHAP Feature Importance Analysis [DONE]

- [x] SHAP TreeExplainer on ChiralAldol-XGB (65d) — completed
- [x] Result: sin_tau1_mean is #1 (SHAP=0.794), Vbur_diff_mean is #4 (SHAP=0.213)
- [x] Top-10: 6 from 3D steric, 4 from conditions — 3D features dominate
- [x] Output: `notebooks/02_shap_analysis/shap_importance.csv`
- [ ] SHAP TreeExplainer on DRFP+Cond+XGBoost → identify which DRFP bits matter (deferred)
- [ ] Generate publication-quality SHAP plots (deferred to paper writing)

### T3: Error Complementarity Analysis [DONE]

- [x] Stack ONLY correct: 20/155 (12.9%) vs DRFP ONLY correct: 5/155 (3.2%) → 4x more unique value from 3D
- [x] 11/155 (7.1%) hard cases wrong by all 7 models — C3 (syn-S) = 63.6% of hard cases
- [x] Per-class: C0=85.5%, C2=92.9%, C3=71.6%, C1=40% (n=5, unreliable)
- [x] Output: `notebooks/02_shap_analysis/hard_cases.csv`
- [ ] Chemical motif analysis of hard cases (deferred — requires manual chemistry inspection)

### T4: Publication Figures [MEDIUM PRIORITY]

- [ ] Confusion matrices (4-class) for top-5 models × temporal split
- [ ] %Vbur_diff vs label: violin plot showing R/S auxiliary face selectivity
- [ ] Radar chart: multi-method, multi-metric comparison
- [ ] t-SNE: ChiralAldol feature space colored by 4-class label
- [ ] Learning curve: training set size vs performance

### T5: Manuscript-Ready Figures & Tables [LOW — pre-submission]

- [ ] Figure 1: ChiralAldol pipeline schematic (enolate → conformers → %Vbur → prediction)
- [ ] Figure 2: 32-model benchmark heatmap (3 splits × bal_acc)
- [ ] Figure 3: SHAP + %Vbur chemical interpretation
- [ ] Figure 4: Error complementarity analysis
- [ ] SI Table S1-S3: Full results per split (all 32 models × all metrics)
- [ ] SI Table S4: Per-class F1 scores
- [ ] SI Figure S1: Conformer ensemble statistics

---

---

## Phase 11: Bridging the 0.725 → 0.90+ Gap

Current best (ChiralAldol-Stack 0.725) is far from 0.90+ accuracy needed for practical utility.

**Root cause diagnosis** (2026-05-12, based on SHAP + Gemini deep research):
1. 24d steric features **全部描述烯醇盐**（亲核体），对醛（亲电体）零描述——这是当前最大盲点
2. sin_tau1 (#1 SHAP) 在用基态二面角近似 TS 几何，表达能力有限
3. 无量子化学层面的电子效应描述符
4. C3 (syn-S) 是主要错误类（71.6% 准确率，硬案例中占 63.6%），暗示 class imbalance 未充分处理

### Phase A — 立即可执行（复用现有管线）

- [ ] **A1: 醛基 Sterimol/%Vbur 特征** ← 优先级最高
  - 对醛的 R 基团做构象采样 + Sterimol (L/B1/B5) + %Vbur
  - 直接复用 `chiralaldol/steric_descriptors.py` 中的现有函数，换算作对象为醛
  - 新增 ~12d 特征，与烯醇盐 24d 特征分开拼接（aldehyde_steric_12d）
  - 输出: `data/processed/chiralaldol/aldehyde_steric_features.csv`

- [ ] **A2: 确认对映体增强可行性**
  - 统计数据集中 R-辅基 vs S-辅基的反应数量分布
  - 如果同一底物 R/S 辅基不同时存在，则可做镜像增强（3644 条）
  - 若两种辅基均已覆盖，增强意义有限

- [x] **A3: 确认原始 dr/ee 数值是否可用** — **BLOCKED**
  - 已确认: `data/raw/alldata.csv` 仅有离散标签 (label_Ca/Cb/SA)，无 dr/ee 连续值
  - ΔΔG‡ 回归路线无法执行，维持当前 4-class 分类路线
  - 注: raw data 中有 `Aldehyde` SMILES 列，可直接用于 A1

- [ ] **A4: 类别不平衡专项处理**
  - 在 Stacking 的 LogisticRegression meta-learner 中加入 class_weight='balanced'（已有）
  - 在 ChiralAldol-XGB base learner 中尝试 Focal Loss 风格的 scale_pos_weight 调整
  - 重点提升 C3 (syn-S) 的准确率

### Phase B — 工程量较大（A 阶段有提升后执行）

- [ ] **B1: xTB 电子描述符**
  - GFN2-xTB 计算烯醇盐和醛的 HOMO-LUMO gap、Fukui f-/f+、偶极矩
  - 提取 Cα 的 f- 指数（亲核位点）和醛羰基碳的 f+ 指数（亲电位点）
  - 工具: xtb CLI + Python subprocess，或 tblite Python binding

- [ ] **B2: 准过渡态 (qTS) 建模** ← 最高天花板，工程量最大
  - 构建 Zimmerman-Traxler 六元环椅式骨架模板
  - 固定新生 C-C 键距离 ~2.1 Å + 金属-O 配位约束
  - GFN2-xTB 约束优化 4 条竞争通道（si/re × chair/twist）
  - 提取 ΔE_qTS（相对准活化能）作为关键特征
  - 规模: 1822 反应 × 4 通道 = ~7300 次 xTB 计算（约数小时）

- [ ] **B3: 迁移学习特征增强**
  - 从 ChemBERTa/RXNFP 提取预训练反应 embedding 作为附加特征
  - 与 A1 醛基特征拼接后重新训练 Stacking

---

## Legacy Issues (from 2026-05-09 audit)

### P1 — Missing baseline prediction CSVs

- [ ] `MajorityClass` and `Random` in `run_all_models.py` computed metrics but did not call `save_preds()`. CSVs missing from `results/predictions/`. Fix: re-run with save enabled.

### P2 — Unscripted prediction file

- [ ] `drfp_aux_cond_xgboost_*.csv` exists but was generated via inline Python (Phase 8 discussion), no standalone script. Fix: document provenance or scriptify.

### P3 — Phantom NAME_MAP entries

- [ ] `t5chem_clf` (T5Chem-Clf): API incompatible, no predictions generated
- [ ] `gcpnet` (GCPNet): protein-oriented, abandoned
- Fix: remove from NAME_MAP or mark as "Not Pursued"

---

## Completed

- [x] Phase 1-2: Project structure + 7-step data cleaning pipeline (4751→4447→1822)
- [x] Phase 3: Feature engineering (Morgan FP, descriptors, DRFP, RXNFP, conditions)
- [x] Phase 4-5: 17 models baseline benchmark
- [x] Phase 5b: Transformer fair comparison (DistilBERT/RoBERTa/ChemBERTa/MolT5)
- [x] Phase 6: Chemprop ± Cond, ProtoNet, ChemAHNet, ChiENN, EquiReact
- [x] Phase 7: Reagent/Base role classification encoding (21d)
- [x] Phase 8: AuxChiral models (aux 6d + cond 35d)
- [x] Phase 10: **Late Fusion — Stacking + Weighted Voting**
  - [x] Early fusion (feature concat 193d) → 0.636, failed due to dimensionality imbalance
  - [x] Weighted Voting → 0.710 (ties prev. champion)
  - [x] Stacking (5-fold OOF + LogReg) → **0.725 (NEW CHAMPION)**
  - [x] SHAP analysis: sin_tau1 #1, Vbur_diff #4, 6/10 top features are 3D steric
  - [x] Error analysis: Stack provides 20 unique correct (12.9%) vs DRFP 5 (3.2%)
  - [x] 11 hard cases (7.1%) universally mispredicted, 63.6% are C3 (syn-S)
- [x] Phase 9: **ChiralAldol — Novel 3D steric descriptor method**
  - [x] M1: Enolate generator (1801/1822, 98.8%)
  - [x] M2: Conformer ensemble sampler (100 conf/mol, RMSD clustering)
  - [x] M3: 3D steric descriptors (%Vbur si/re, Sterimol L/B1/B5, dihedrals, 24d total)
  - [x] M4: Feature integration (steric 24d + cond 35d + aux 6d = 65d)
  - [x] M5: XGBoost training + evaluation (3 models × 3 splits)
  - [x] Ablation study (SterOnly vs CondAux vs Full)
  - [x] Chemical validation (R/S → Vbur_diff sign inversion confirmed)
- [x] 3D conformer generation (1801/1822 valid)
- [x] Unified comparison table (32 models × 3 splits = 96 CSVs)

## Abandoned (with reasons)

- T5Chem-Clf: transformers 5.8 API incompatible
- ChemFormer: requires Python 3.7 + old pytorch-lightning
- MolecularTransformer: Python 3.5 + OpenNMT 0.4.1, completely outdated
- GCPNet: protein-oriented (PDB/Bio.PDB), not applicable
