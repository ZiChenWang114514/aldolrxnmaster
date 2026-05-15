# AldolRxnMaster — Publication Roadmap

## Project Status (2026-05-15)

- **Data**: 4751 → 4447 → 1801 Evans clean (Phase A 清洗), 4258 全量 (含非 Evans)
- **Features**: 75d V2 (enolate 24d + aldehyde 10d + cond 35d + aux 6d) = **严格最优**
- **Models**: 47 tabular + 9 GNN + feature fusion = **56+ 模型**
- **Champion**: ChiralAldolV2-XGB (75d)
  - TSCV 4-fold mean: **0.682 ± 0.044** (可靠评估)
  - Scaffold: **0.826**, Grouped: **0.807**
- **天花板确认**: 表格 + GNN + fingerprint 融合 = 全部不如 V2 (75d) alone
- **关键发现**: 单 temporal split (0.783) 因 C1=5 样本不稳定, TSCV 是更可靠指标

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

- [x] **A1: 醛基 Sterimol/%Vbur 特征** ✓ DONE (2026-05-13)
  - `chiralaldol/aldehyde_steric.py` — 667 unique aldehydes, 100% success
  - 新增 10d 特征 (L/B1/B5 + Vbur_total, mean/std + metadata)
  - **ChiralAldolV2-XGB**: 0.783 temporal (+11.9% over V1-XGB)
  - **ChiralAldolV2-Stack**: 0.782 temporal (DRFP fusion now marginal)
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

### Phase A-cont — 快速验证

- [x] **A2: 对映体增强** — **无效，已放弃** (2026-05-13)
  - 94.5% 底物对仅含单侧辅基，理论上可增强 1546 条
  - 实测: temporal -1.8%, scaffold -10.5%, grouped -0.6% → 全部下降
  - 原因: 特征层翻转(Vbur_si↔re, sin_τ→-sin_τ)是近似非精确的；
    模型已通过 aux_config_R + Vbur_diff 符号隐式学到 R/S 对称性

- [ ] **A4: 类别不平衡专项处理**
  - C1 (anti-1) 仅 10.3%，C2 (anti-2) 仅 7.4%
  - 尝试 Focal Loss / SMOTE / class-weighted XGBoost 调整
  - 重点: V2-XGB 的 C1 仍只有 60%（temporal）

### Phase B — xTB 电子描述符 ✗ 完成 (负面结果)

- [x] **B1: GFN2-xTB 电子描述符** — ✗ 负面 (2026-05-14)
  - `chiralaldol/xtb_descriptors.py` (tblite-python 0.5.0)
  - V3-XGB (87d, 全量 xTB): temporal **0.696** (退步 -8.7%)
  - V3b-XGB (80d, 5d clean ald xTB): temporal **0.721** (仍不如 V2)
  - 根因: 烯醇盐 xTB 59% 计算失败 (大分子+带电荷)；Evans aldol 是立体控制反应

### Phase C — qTS 过渡态建模 ✗ 完成 (负面结果)

- [x] **C1: qTS VDW steric** — ✗ 负面 (2026-05-14)
  - `chiralaldol/qts_builder.py` + `scripts/run_qts_pipeline.py`
  - V4-XGB (79d = 75d V2 + 4d qTS): temporal **0.628** (退步 -15.5%)
  - 根因: 近似 ZT 坐标 si/re 面不一致 (r≈-0.03); GFN1/2-xTB 太慢 (50-120s/mol)

### Phase V5 — 交叉项特征 ✗ 完成 (负面结果)

- [x] **V5: 交叉项 + Z/E + 多模型** — ✗ 负面 (2026-05-14)
  - `chiralaldol/feature_builder.py:build_chiralaldol_v5_features()` + `scripts/run_v5_pipeline.py`
  - V5-XGB (87d): temporal **0.758**; V5-LGBM: 0.749; V5-Stack: 0.694
  - 根因: 交叉项在训练集 r=0.33 但不迁移到 temporal test set (2019+)
  - **结论: 0.783 是表格方法天花板**

---

## Legacy Issues (from 2026-05-09 audit)

- [x] **P1**: MajorityClass/Random save_preds() — RESOLVED (Phase 11 规整)
- [ ] **P2**: drfp_aux_cond_xgboost 无独立脚本 — 低优先级
- [x] **P3**: t5chem_clf/gcpnet phantom entries — RESOLVED (已从 NAME_MAP 移除)

## Phase A-D (2026-05-15) — 数据清洗 + GNN + 融合

- [x] **A1**: chirality_valid bug 修复 + 21 行删除 → 1801 行
- [x] **A3**: 溶剂推断 (386/497 填充, 93.9% known)
- [x] **A4**: Time-series CV (4-fold mean = 0.682 ± 0.044)
- [x] **A5**: V2-XGB 重训练 (scaffold 0.826, grouped 0.807)
- [x] **B2**: 全量数据集 all_clean.csv (4258 行)
- [x] **B4**: 4 种图表示 (diff/multiview/3D/TS)
- [x] **C**: GNN 12 组合 (**负面**: 最佳 0.497, 远低于 V2)
- [x] **D**: 特征融合 (**负面**: V2 alone > V2+any fingerprint)

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
