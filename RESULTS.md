# Benchmark Results

56+ models evaluated on Evans asymmetric aldol reaction stereochemistry prediction.
Cleaned dataset: **1801 reactions** (from 1822), 4-class joint Ca×Cb label.

## Evaluation Metrics

| Metric | Champion (V2-XGB) | Notes |
|--------|-------------------|-------|
| **TSCV 4-fold mean** | **0.682 ± 0.044** | Most reliable; 4 temporal windows |
| Temporal single split | 0.69 (clean) / 0.783 (old) | C1=5 samples → unstable |
| Scaffold | **0.826** | In-distribution generalization |
| Grouped random | **0.807** | Stratified random |

> **注意**: 单 temporal split 的 0.783 因 C1 仅 5 样本不稳定 (2 预测变化 = ±0.11)。
> TSCV 4-fold mean (0.682 ± 0.044) 是更可靠的 temporal 评估。

## GNN 实验结果 (Phase C, 2026-05-15)

| Model | Graph | Fusion | Temporal bal_acc |
|-------|-------|--------|-----------------|
| MPNN+FiLM | diff | FiLM | **0.497** |
| Equiformer+concat | 3D spatial | concat | 0.458 |
| SchNet+FiLM | 3D spatial | FiLM | 0.389 |
| SchNet+concat | 3D spatial | concat | 0.343 |
| MPNN+concat | diff | concat | 0.312 |
| MPNN+inject | diff | inject | 0.288 |

**结论**: GNN 全面不如 V2-XGB (0.69)。1801 样本不足以训练 GNN。

## 特征融合结果 (Phase D, 2026-05-15)

| Features | Temporal bal_acc |
|----------|-----------------|
| **V2 (75d)** | **0.690** |
| V2+DRFP (203d) | 0.625 |
| V2+RXNFP (331d) | 0.621 |
| DRFP+Cond (163d) | 0.624 |
| V2+DRFP+RXNFP (459d) | 0.548 |

**结论**: V2 (75d) strictly 最优。加任何 fingerprint 都降低 temporal 性能。

## Evans Temporal Split — Tabular Models (原始 1822 行数据)

Train ≤2015 (1500), Val 2016-2018 (167), Test ≥2019 (155)

| Rank | Model | Bal.Acc | MCC | Joint Acc | F1m | 95% CI | Type |
|------|-------|---------|-----|-----------|-----|--------|------|
| **1** | **ChiralAldolV2-XGB** | **0.7829** | **0.7115** | **0.8194** | **0.8160** | **[0.630,0.902]** | **3D Enolate+Ald+Cond** |
| **2** | **ChiralAldolV2-Stack** | **0.7815** | **0.7076** | **0.8194** | **0.8157** | **[0.629,0.899]** | **Late Fusion V2** |
| 3 | ChiralAldol-Stack | 0.7248 | 0.6350 | 0.7742 | 0.7560 | [0.598,0.849] | Late Fusion |
| 4 | DRFP+Cond+XGBoost | 0.7106 | 0.5105 | 0.6774 | 0.7105 | [0.556,0.839] | FP+Cond+GBDT |
| 5 | ChiralAldol-WtVote | 0.7100 | 0.6327 | 0.7742 | 0.7481 | [0.572,0.849] | Late Fusion |
| 6 | DRFP+Aux+Cond-XGB | 0.7061 | 0.4961 | 0.6710 | 0.7136 | [0.553,0.832] | FP+Cond+GBDT |
| 5 | ProtoNet | 0.6842 | 0.6011 | 0.7419 | 0.6700 | [0.553,0.828] | Meta-learning |
| 6 | ChiralAldol-XGB | 0.6644 | 0.4894 | 0.6387 | 0.6587 | [0.545,0.793] | 3D Steric+Cond |
| 7 | ChiralAldol+DRFP-XGB | 0.6360 | 0.5905 | 0.7484 | 0.6577 | [0.524,0.771] | Early Fusion |
| 8 | Chemprop+Cond | 0.6251 | 0.5524 | 0.7226 | 0.6617 | [0.519,0.762] | MPNN+Cond |
| 9 | DistilBERT-Rxn | 0.5910 | 0.5560 | 0.7161 | 0.5618 | [0.491,0.724] | Transformer |
| 10 | DRFP+XGBoost | 0.5870 | 0.5697 | 0.7484 | 0.5981 | [0.456,0.733] | FP+GBDT |
| 11 | AuxChiral+Ald-XGB | 0.5826 | 0.4786 | 0.6774 | 0.5531 | [0.441,0.724] | Cond+GBDT |
| 12 | SterOnly-XGB | 0.5638 | 0.4992 | 0.6968 | 0.5480 | [0.425,0.701] | 3D Steric only |
| 13 | DRFP+LightGBM | 0.5248 | 0.5225 | 0.7226 | 0.5430 | [0.420,0.671] | FP+GBDT |
| 14 | AuxChiral-LGBM | 0.5244 | 0.3621 | 0.5806 | 0.4900 | [0.392,0.669] | Cond+GBDT |
| 15 | CondAux-XGB | 0.5215 | 0.3435 | 0.5806 | 0.4955 | [0.385,0.670] | Cond only |
| 16 | AuxChiral-XGB | 0.5168 | 0.3343 | 0.5677 | 0.4879 | [0.378,0.666] | Cond+GBDT |
| 17 | AuxNoBase-XGB | 0.5153 | 0.3312 | 0.5677 | 0.4863 | [0.379,0.661] | Cond+GBDT |
| 18 | ChemAHNet-Aldol | 0.4975 | 0.4393 | 0.6387 | 0.4548 | [0.374,0.640] | Chem-informed DL |
| 19 | Chemprop | 0.4894 | 0.4204 | 0.6387 | 0.4628 | [0.411,0.567] | MPNN |
| 20 | RoBERTa-Rxn | 0.4799 | 0.3671 | 0.4903 | 0.4260 | [0.333,0.634] | Transformer |
| 21 | Morgan-MLP | 0.4741 | 0.1143 | 0.4839 | 0.4998 | [0.318,0.643] | FP+MLP |
| 22 | ChemBERTa-77M | 0.4247 | 0.0897 | 0.4194 | 0.2917 | [0.266,0.508] | Chem-Transformer |
| 23 | XGBoost | 0.4181 | 0.1164 | 0.4774 | 0.4476 | [0.299,0.547] | GBDT |
| 24 | LightGBM | 0.3983 | 0.0631 | 0.4452 | 0.4290 | [0.278,0.532] | GBDT |
| 25 | XGBoost-FullFP | 0.3710 | 0.0215 | 0.4129 | 0.3963 | [0.253,0.511] | FP+GBDT |
| 26 | 5-NN | 0.3348 | -0.004 | 0.3935 | 0.3019 | [0.211,0.485] | kNN |
| 27 | RF | 0.3338 | -0.005 | 0.4000 | 0.3511 | [0.213,0.458] | RF |
| 28 | RXNFP+MLP | 0.3324 | 0.2152 | 0.5419 | 0.3176 | [0.280,0.396] | Learned FP+MLP |
| 29 | 1-NN | 0.3251 | 0.0317 | 0.4065 | 0.3212 | [0.224,0.460] | kNN |
| 30 | RXNFP+XGBoost | 0.3225 | 0.1453 | 0.3484 | 0.2913 | [0.211,0.456] | Learned FP+GBDT |
| 31 | EquiReact | 0.3179 | 0.1773 | 0.5355 | 0.2444 | [0.250,0.431] | 3D Equivariant |
| 32 | RXNFP+LightGBM | 0.3150 | 0.1285 | 0.3419 | 0.2814 | [0.209,0.451] | Learned FP+GBDT |
| 33 | MolT5-base | 0.2623 | 0.1245 | 0.3806 | 0.1570 | [0.253,0.278] | Chem-Transformer |
| 34 | ChiENN-Product | 0.2572 | -0.045 | 0.3355 | 0.1665 | [0.147,0.414] | Chirality GNN |
| 35 | CondOnly-XGB | 0.1992 | -0.075 | 0.2387 | 0.1972 | [0.113,0.325] | Cond only |

## Evans Scaffold Split

Train 1457, Val 182, Test 183

| Rank | Model | Bal.Acc | MCC | Joint Acc | Type |
|------|-------|---------|-----|-----------|------|
| 1 | DRFP+Cond+XGBoost | 0.8257 | 0.7138 | 0.8087 | FP+Cond+GBDT |
| 2 | DRFP+Aux+Cond-XGB | 0.7647 | 0.6733 | 0.7869 | FP+Cond+GBDT |
| 3 | ChiralAldol-WtVote | 0.7218 | 0.6770 | 0.7814 | Late Fusion |
| 4 | DRFP+XGBoost | 0.7220 | 0.5903 | 0.7268 | FP+GBDT |
| 5 | ChiralAldol-Stack | 0.7089 | 0.6397 | 0.7486 | Late Fusion |
| 6 | AuxChiral-XGB | 0.6911 | 0.6045 | 0.7213 | Cond+GBDT |
| 7 | ChiralAldol-XGB | 0.6759 | 0.6037 | 0.7213 | 3D Steric+Cond |

## Evans Grouped Random (seed42)

Train 1458, Val 179, Test 185

| Rank | Model | Bal.Acc | MCC | Joint Acc | Type |
|------|-------|---------|-----|-----------|------|
| 1 | DRFP+Aux+Cond-XGB | 0.7976 | 0.7580 | 0.8486 | FP+Cond+GBDT |
| 2 | ChiralAldol-WtVote | 0.7898 | 0.7636 | 0.8541 | Late Fusion |
| 3 | ChiralAldol-Stack | 0.7775 | 0.7330 | 0.8324 | Late Fusion |
| 4 | DRFP+Cond+XGBoost | 0.7751 | 0.7584 | 0.8486 | FP+Cond+GBDT |
| 5 | ChiralAldol-XGB | 0.7565 | 0.7071 | 0.8162 | 3D Steric+Cond |

## ChiralAldol Ablation Study

| Model | EnolSter(24d) | AldSter(10d) | Cond(35d) | Aux(6d) | DRFP(128d) | Temporal | Scaffold | Grouped |
|-------|:---:|:---:|:---:|:---:|:---:|---------|---------|---------|
| **ChiralAldolV2-XGB** | **yes** | **yes** | **yes** | **yes** | | **0.783** | **0.819** | 0.789 |
| **ChiralAldolV2-Stack** | **yes** | **yes** | **yes** | **yes** | **yes (late)** | **0.782** | **0.808** | 0.788 |
| ChiralAldol-Stack | yes | | yes | yes | yes (late) | 0.725 | 0.709 | 0.778 |
| ChiralAldol-WtVote | yes | | yes | yes | yes (late) | 0.710 | 0.722 | **0.790** |
| ChiralAldol+DRFP-XGB | yes | | yes | yes | yes (early) | 0.636 | — | — |
| ChiralAldol-XGB | yes | | yes | yes | | 0.664 | 0.676 | 0.757 |
| SterOnly-XGB | yes | | | | | 0.564 | 0.621 | 0.594 |
| CondAux-XGB | | | yes | yes | | 0.522 | 0.667 | 0.721 |

Key findings from ablation:
- **Aldehyde steric is the single most impactful feature addition**: V2-XGB (0.783) vs V1-XGB (0.664) = **+11.9%** absolute on temporal, from just 10 new features.
- **V2 eliminates the need for DRFP fusion**: V2-XGB (0.783) ≈ V2-Stack (0.782). Adding DRFP via stacking provides <0.1% marginal gain. The 10d aldehyde features captured what DRFP was compensating for.
- **Late fusion >> Early fusion**: Stacking (0.725) >> feature concatenation (0.636). Early fusion dilutes 3D steric signal due to DRFP's dimensional dominance (128d vs 24d).
- **SterOnly (0.564) > CondAux (0.522) on temporal**: 3D steric descriptors alone carry more predictive signal than reaction conditions on out-of-distribution future reactions.

## SHAP Feature Importance (ChiralAldol-XGB, 65 features)

| Rank | Feature | SHAP | Category |
|------|---------|------|----------|
| 1 | sin_tau1_mean | 0.794 | 3D dihedral |
| 2 | solvent_pi_star | 0.295 | Condition |
| 3 | solvent_beta | 0.268 | Condition |
| 4 | **Vbur_diff_mean** | **0.213** | **3D %Vbur** |
| 5 | sin_tau2_mean | 0.194 | 3D dihedral |
| 6 | activator_TiCl4 | 0.146 | Condition |
| 7 | cos_tau1_mean | 0.144 | 3D dihedral |
| 8 | Vbur_si_std | 0.127 | 3D %Vbur |
| 9 | Vbur_total_std | 0.126 | 3D %Vbur |
| 10 | base_DIPEA | 0.123 | Condition |

Top-10 features: **6 from 3D steric descriptors, 4 from reaction conditions**. The dihedral angle sin_tau1 (auxiliary-enolate torsion) is the single most important feature, with SHAP value 2.7x higher than the next feature. Vbur_diff_mean (face selectivity) ranks 4th.

## Error Complementarity (ChiralAldol-Stack vs DRFP+Cond, temporal test set)

| Category | Count | % |
|----------|-------|---|
| Both correct | 100 | 64.5% |
| **Stack ONLY correct** | **20** | **12.9%** |
| DRFP ONLY correct | 5 | 3.2% |
| Both wrong | 30 | 19.4% |

ChiralAldol-Stack provides 4x more unique correct predictions (20) than DRFP+Cond provides alone (5). This confirms that 3D steric descriptors encode information invisible to 2D reaction fingerprints.

## Hard Cases

11/155 test samples (7.1%) are mispredicted by all 7 evaluated models. Class distribution: C3 (syn-S) accounts for 63.6% of hard cases, suggesting certain S-auxiliary reactions follow non-standard selectivity mechanisms.

## Per-Class Accuracy (ChiralAldol-Stack, temporal)

| Class | Accuracy | n |
|-------|----------|---|
| C0 (syn-R) | 85.5% | 47/55 |
| C1 (anti-1) | 40.0% | 2/5 |
| C2 (anti-2) | 92.9% | 13/14 |
| C3 (syn-S) | 71.6% | 58/81 |

C2 (anti) achieves 92.9% accuracy. C1 (anti) has only 5 test samples, too few for reliable assessment. The main bottleneck is C3 (syn-S) at 71.6%.

## Per-Class Accuracy (ChiralAldolV2-XGB, temporal)

| Class | V2 Accuracy | V1 Accuracy | Δ |
|-------|------------|------------|---|
| C0 (syn-R) | 85.5% | 85.5% | 0 |
| C1 (anti-1) | 60.0% | 40.0% | +20% |
| C2 (anti-2) | 92.9% | 92.9% | 0 |
| C3 (syn-S) | 75.3% | 71.6% | +3.7% |

V2's gain is concentrated in C1 (anti-1, +20%) — the hardest minority class — and C3 (syn-S, +3.7%).

## Honest Assessment

ChiralAldolV2-XGB (0.783) closes half the gap from 0.725 to 0.90. Remaining gap (0.783 → 0.90+) likely requires: (1) QM-level transition state features (qTS modeling with GFN2-xTB), (2) electronic descriptors (Fukui, HOMO-LUMO), or (3) class-imbalance-aware training (Focal Loss for C1/C2 anti classes).

## Key Findings

1. **ChiralAldolV2-XGB is the champion on temporal split** (0.783 bal_acc), surpassing all 36 other models.
2. **Aldehyde steric is the single most impactful feature**: V2-XGB (0.783) vs V1-XGB (0.664) = +11.9% from just 10 new features. This confirms that the aldehyde R-group's equatorial/axial preference in the Zimmerman-Traxler TS was the critical missing information.
3. **V2 renders DRFP fusion unnecessary**: V2-XGB (0.783) ≈ V2-Stack (0.782). The 10d aldehyde features captured what DRFP was compensating for.
4. **Chemistry-informed 3D > end-to-end 3D DL**: ChiralAldolV2 (0.783) >> ChiENN (0.257), EquiReact (0.318). Explicit mechanistic modeling is far more effective than learning from raw coordinates at this data scale.
5. **SHAP validates chemical intuition**: sin_tau1 (auxiliary-enolate torsion) is the #1 feature; %Vbur_diff (face selectivity) is #4. 3D descriptors dominate the top-10.
6. **%Vbur_diff correctly captures Evans model**: R-auxiliary gives negative Vbur_diff (re-face shielded), S-auxiliary gives positive (si-face shielded), consistent with Zimmerman-Traxler TS theory.
7. **No published ML baseline exists** for aldol stereochemistry prediction — this is the first comprehensive benchmark with 37 models.

## Class Distribution (Evans Subset)

| Class | Ca | Cb | Stereo | Count | % |
|-------|----|----|--------|-------|---|
| C0 | 0 | 0 | syn | 753 | 41.3% |
| C1 | 0 | 1 | anti | 188 | 10.3% |
| C2 | 1 | 0 | anti | 135 | 7.4% |
| C3 | 1 | 1 | syn | 746 | 40.9% |
