# Model Registry — AldolRxnMaster

All models evaluated for Evans asymmetric aldol 4-class stereochemistry prediction.

## Naming Convention

- **Prediction files**: `results/predictions/{category}/{key}_{split}.csv`
- **Key**: Short unique identifier (snake_case, no spaces)
- **Split names**: `tscv_fold{1-4}`, `scaffold`, `grouped_seed{42,123,456,789,1024}`, `temporal_2019`

## Categories

| Category | Description | Directory |
|----------|-------------|-----------|
| steric | Hand-crafted 3D steric + condition features | `results/predictions/steric/` |
| fp | Molecular fingerprint-based models | `results/predictions/fp/` |
| gnn | Graph neural networks | `results/predictions/gnn/` |
| meta | Meta-learning approaches | `results/predictions/meta/` |
| baseline | Simple baselines, condition-only, majority | `results/predictions/baseline/` |

---

## Full Registry

### Category: steric (Hand-crafted 3D Features)

| Key | Full Name | Dim | Status | Best TSCV | Script |
|-----|-----------|-----|--------|-----------|--------|
| cv1_xgb | ChiralAldol-V1-XGB | 65d | deprecated | — | run_chiralaldol.py |
| cv2_xgb | ChiralAldol-V2-XGB | 75d | **active** | 0.726 | run_chiralaldol.py |
| cv2_lgbm | ChiralAldol-V2-LGBM | 75d | active | — | run_all_models.py |
| cv2_et | ChiralAldol-V2-ExtraTrees | 75d | active | — | run_all_models.py |
| cv2_rf | ChiralAldol-V2-RandomForest | 75d | active | — | run_all_models.py |
| cv2_stack | ChiralAldol-V2-Stacking | 75d | active | — | run_chiralaldol.py |
| cv2_vote | ChiralAldol-V2-WtVote | 75d | active | — | run_chiralaldol.py |
| cv3_xgb | ChiralAldol-V3-XGB | 87d | deprecated | — | run_chiralaldol.py |
| cv3b_xgb | ChiralAldol-V3b-XGB | 80d | deprecated | — | run_chiralaldol.py |
| cv4_xgb | ChiralAldol-V4-XGB | 79d | deprecated | — | run_qts_pipeline.py |
| cv5_xgb | ChiralAldol-V5-XGB | 87d | deprecated | — | run_v5_pipeline.py |
| cv5_lgbm | ChiralAldol-V5-LGBM | 87d | deprecated | — | run_v5_pipeline.py |
| ma_bw | MechAware-BW | 72d | **active** | 0.692 | run_mechaware.py |
| ma_ze | MechAware-ZE | 120d | **active** | — | run_mechaware.py |
| ma_full | MechAware-Full | 144d | **active** | 0.704 | run_mechaware.py |
| steronly_xgb | StericOnly-XGB | 24d | active | — | run_all_models.py |

### Category: fp (Fingerprint-Based)

| Key | Full Name | Dim | Status | Best TSCV | Script |
|-----|-----------|-----|--------|-----------|--------|
| drfp_xgb | DRFP-XGB | 2048d | ⚠️ LEAKAGE | ~~0.849~~ | run_all_models.py |
| drfp_cond_xgb | DRFP+Cond-XGB | ~2083d | ⚠️ LEAKAGE | ~~0.872~~ | run_all_models.py |
| drfp_aux_cond_xgb | DRFP+Aux+Cond-XGB | ~2089d | ⚠️ LEAKAGE | — | run_all_models.py |
| drfp_lgbm | DRFP-LGBM | 2048d | ⚠️ LEAKAGE | — | run_all_models.py |
| rxnfp_xgb | RXNFP-XGB | 256d | active | — | run_all_models.py |
| rxnfp_mlp | RXNFP-MLP | 256d | deprecated | — | run_all_models.py |
| morgan_xgb | Morgan-XGB | 2048d | active | — | run_precompute_fps.py |

### Category: gnn (Graph Neural Networks)

| Key | Full Name | Dim | Status | Best TSCV | Script |
|-----|-----------|-----|--------|-----------|--------|
| mpnn_film | MPNN+FiLM | — | deprecated | 0.497 | run_gnn_benchmark.py |
| mpnn_concat | MPNN+Concat | — | deprecated | — | run_gnn_benchmark.py |
| mpnn_inject | MPNN+Inject | — | deprecated | — | run_gnn_benchmark.py |
| gat_film | GAT-MultiView+FiLM | — | deprecated | — | run_gnn_benchmark.py |
| gat_concat | GAT-MultiView+Concat | — | deprecated | — | run_gnn_benchmark.py |
| equi_concat | Equiformer+Concat | — | deprecated | — | run_gnn_benchmark.py |
| equi_film | Equiformer+FiLM | — | deprecated | — | run_gnn_benchmark.py |
| schnet_film | SchNet-3D+FiLM | — | deprecated | — | run_gnn_benchmark.py |
| schnet_concat | SchNet-3D+Concat | — | deprecated | — | run_gnn_benchmark.py |
| chienn | ChiENN-Product | — | deprecated | — | run_chienn_product.py |
| chemprop | ChemProp-Aldol | — | deprecated | — | run_chemprop.py |
| chemahnet | ChemAHNet-Aldol | — | deprecated | — | run_chemahnet.py |

### Category: meta (Meta-Learning)

| Key | Full Name | Dim | Status | Best TSCV | Script |
|-----|-----------|-----|--------|-----------|--------|
| protonet | ProtoNet | — | active | — | run_protonet.py |

### Category: baseline (Baselines)

| Key | Full Name | Dim | Status | Best TSCV | Script |
|-----|-----------|-----|--------|-----------|--------|
| cond_xgb | CondOnly-XGB | 35d | active | — | run_all_models.py |
| condaux_xgb | CondAux-XGB | 41d | active | — | run_all_models.py |
| knn_1 | 1-NN | — | active | — | run_all_models.py |
| knn_5 | 5-NN | — | active | — | run_all_models.py |
| majority | MajorityClass | — | active | — | run_all_models.py |

---

## Status Definitions

| Status | Meaning |
|--------|---------|
| **active** | Currently maintained, included in benchmarks |
| deprecated | Superseded or failed, kept for reference only |
| experimental | Under development, results not finalized |

## Key Findings

- **Champion (TSCV)**: cv2_xgb (0.726 on fair comparison, 1551 rows)
- **Champion (Scaffold)**: ma_full (0.777 on fair comparison)
- **Champion (Temporal 2019+)**: ma_full (0.796 on fair comparison)
- **Leakage confirmed**: Old scaffold 0.826 was inflated by ~7% due to group_id bug
- **GNN verdict**: All GNN models ≤ 0.50 — insufficient data for end-to-end learning
