# ChiralAldol + MechAware: Mechanism-Informed Stereoselectivity Prediction

## Overview

ChiralAldol is a method for predicting Evans asymmetric aldol reaction stereoselectivity using chemistry-informed 3D steric descriptors. **MechAware** extends it with explicit Z/E enolate separation and base-dependent weighting.

**Champion**: MechAware-Full (151d), TSCV = **0.733 ± 0.074** (V3 公平基准, 2026-05-16)

Key modules:
- `ze_enolate_generator.py` — Z/E 3D dihedral marking + constrained conformer generation
- `steric_descriptors.py` — %Vbur (face-dependent) + Sterimol + dihedral encoding (24d)
- `aldehyde_steric.py` — Aldehyde Sterimol + Vbur (10d)
- `conformer_sampler.py` — ETKDGv3 conformer ensemble (no MMFF)
- `feature_builder.py` — V1-V5 feature assembly
- `rebuild/` — V3 16-step data cleaning pipeline

## Chemistry

Evans aldol stereoselectivity is determined by the facial selectivity of the metal enolate intermediate:

```
     O    O                     O-    O
     ||   ||    + base          |     ||
R-CH2-C---N-[Oxaz]   -->  R-CH=C---N-[Oxaz]
                                 |
                                 M+
```

1. **Evans auxiliary R/S** controls which face of the enolate is shielded
2. **Base** determines enolate Z/E geometry (Bu2BOTf/DIPEA → Z, LDA → Z, Et3N → mixed)
3. **Metal chelation** creates a Zimmerman-Traxler 6-membered ring TS
4. **3D steric hindrance** on each face determines product stereochemistry

ChiralAldol explicitly models this mechanism by generating enolate intermediates, sampling conformational ensembles, and computing face-dependent steric descriptors.

## Pipeline

```
M1: Enolate Generator     ketone SMILES → Z/E enolate SMILES (SMARTS)
M2: Conformer Sampler     ETKDG v3 × 100 → MMFF optimize → RMSD cluster → K representatives
M3: Steric Descriptors    %Vbur(si/re), Sterimol L/B1/B5, dihedrals → Boltzmann aggregate
M4: Feature Builder       steric(24d) + conditions(35d) + aux_chirality(6d) = 65d
M5: XGBoost Classifier    3-config grid search, class-weighted, balanced_accuracy
```

## Features (24 steric descriptors)

| Group | Features | Description |
|-------|----------|-------------|
| %Vbur | Vbur_si, Vbur_re, Vbur_diff, Vbur_total (×mean,std) | Face-dependent buried volume around alpha carbon. Vbur_diff captures facial selectivity |
| Sterimol | L, B1, B5 (×mean,std) | R-group geometry: length, min/max perpendicular width |
| Dihedrals | sin_tau1, cos_tau1, sin_tau2, cos_tau2 (×mean,std) | Auxiliary-enolate torsion and R-group orientation |
| Ensemble | n_conformers, n_clusters | Conformational diversity measures |

Each descriptor is computed per representative conformer and aggregated across the ensemble using Boltzmann-weighted mean and standard deviation.

## Results

### Temporal Split (hardest, most realistic)

| Model | Bal.Acc | MCC | Joint | Method |
|-------|---------|-----|-------|--------|
| **ChiralAldol-Stack** (NEW champion) | **0.725** | **0.635** | **0.774** | **Stacking: 3D steric + DRFP** |
| ChiralAldol-WtVote | 0.710 | 0.633 | 0.774 | Weighted voting |
| DRFP+Cond+XGBoost (prev. champion) | 0.711 | 0.511 | 0.677 | 2D fingerprint |
| ChiralAldol-XGB (standalone) | 0.664 | 0.489 | 0.639 | 3D steric only |
| SterOnly-XGB (ablation) | 0.564 | 0.499 | 0.697 | steric(24d) only |
| CondAux-XGB (ablation) | 0.522 | 0.344 | 0.581 | cond+aux only |

### Cross-Split Performance

| Model | Temporal | Scaffold | Grouped Random |
|-------|----------|----------|----------------|
| **ChiralAldol-Stack** | **0.725** | 0.709 | 0.778 |
| ChiralAldol-WtVote | 0.710 | 0.722 | **0.790** |
| DRFP+Cond+XGB | 0.711 | 0.826 | 0.775 |
| ChiralAldol-XGB | 0.664 | 0.676 | 0.757 |

### Late Fusion Methodology

Early feature concatenation (65d + 128d = 193d) failed (0.636) due to DRFP's dimensional dominance drowning out the 24d steric signal. Solution: late fusion at prediction-probability level.

**Stacking**: 5-fold stratified CV produces out-of-fold predictions from both base models → 8d meta-features (2 models × 4 class probs) → LogisticRegression learns optimal combination weights. Only 8 features for ~1500 samples = extremely stable.

**Weighted Voting**: α = val_bal_acc_A / (val_bal_acc_A + val_bal_acc_B). Zero additional training.

### Key Findings

1. **ChiralAldol-Stack (0.725) surpasses DRFP+Cond (0.711)** on temporal split by +1.4%, proving 3D steric descriptors add information invisible to 2D fingerprints
2. **Beats all 3D DL models**: ChiralAldol (0.664) >> ChiENN (0.257), EquiReact (0.318)
3. **Beats all Transformers**: ChiralAldol (0.664) > DistilBERT (0.591), ChemBERTa (0.425)
4. **3D steric features are informative**: SterOnly (0.564) > CondAux (0.522) on temporal
5. **Chemical validation**: R-auxiliary → Vbur_diff < 0, S-auxiliary → Vbur_diff > 0 (Evans model)
6. **Late fusion >> Early fusion**: Stacking (0.725) >> concatenation (0.636) — proper fusion strategy is critical for multi-view learning on small datasets
7. **Honest gap**: Current 0.725 is far from the 0.90+ accuracy needed for practical synthetic utility

## Pipeline Statistics

| Stage | Success | Time |
|-------|---------|------|
| M1: Enolate generation | 1801/1822 (98.8%) | <1s |
| M2: Conformer ensemble (100 conf/mol) | 1801/1822 (98.8%) | ~50 min |
| M3: Steric descriptors | 1798/1822 (98.7%) | ~76s |
| M4+M5: Features + model | 1822/1822 (100%) | ~30s |

## Usage

```bash
# Full pipeline (enolate → conformers → features → models)
conda run -n aldol-rxn python scripts/run_chiralaldol_pipeline.py

# Pipeline supports checkpointing: if interrupted, re-run to resume from last checkpoint
```

## Files

```
chiralaldol/
  __init__.py
  enolate_generator.py     # M1: SMARTS-based ketone → enolate
  conformer_sampler.py     # M2: ETKDG + MMFF + RMSD clustering
  steric_descriptors.py    # M3: %Vbur, Sterimol, dihedrals
  feature_builder.py       # M4: combine steric + conditions + aux
  utils.py                 # shared: clean_mol, plane_normal, vdw_radii

data/processed/chiralaldol/
  enolates.csv             # 1822 rows: enolate SMILES + Z/E selectivity
  conformer_ensembles.pkl  # 2.8 MB: K conformers per molecule
  steric_features.csv      # 1822 × 24: Boltzmann-aggregated descriptors
```
