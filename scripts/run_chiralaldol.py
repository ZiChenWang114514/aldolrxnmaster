#!/usr/bin/env python
# DEPRECATED: Use scripts/run_chiralaldol_pipeline.py instead.
# This file is the Phase 9 standalone version kept for reference only.
# The pipeline script includes all stages (enolates → conformers → steric → V1/V2/V3/V3b training).
"""ChiralAldol: Chemistry-informed 3D steric descriptors for Evans aldol prediction.

Novel method pipeline:
  1. Enolate generation (ketone → Z/E enolate)
  2. Conformer ensemble sampling (ETKDG + MMFF + RMSD clustering)
  3. 3D steric descriptors (%Vbur, Sterimol, dihedrals)
  4. Feature integration (steric + conditions + aux chirality)
  5. XGBoost prediction with 3-config grid search

Models:
  - ChiralAldol-XGB: 3D steric + conditions + aux → XGBoost
  - SterOnly-XGB: 3D steric only → XGBoost (ablation)
  - CondAux-XGB: conditions + aux only → XGBoost (ablation, no 3D)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci
from chiralaldol.feature_builder import build_chiralaldol_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"
for d in [RESULTS_DIR / "predictions", RESULTS_DIR / "tables"]:
    d.mkdir(parents=True, exist_ok=True)


def load_split(split_name):
    with open(SPLIT_DIR / f"{split_name}.json") as f:
        sp = json.load(f)
    return np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])


def train_xgb(X_tr, y_tr, X_val, y_val):
    """Train XGBoost with 3-config grid search (consistent with project standard)."""
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15,
         "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({
            "objective": "multi:softprob", "num_class": 4,
            "tree_method": "hist", "random_state": 42,
            "n_jobs": 4, "verbosity": 0,
            "gamma": 0.1, "reg_lambda": 1.0,
        })
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def evaluate_and_save(name, y_test, y_pred, y_prob, test_idx, split_name):
    """Evaluate and save predictions (standard project format)."""
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  {name}: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint={metrics['joint_accuracy']:.4f}, "
                f"F1m={metrics['f1_macro']:.4f}")

    out = pd.DataFrame({"idx": test_idx, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out[f"prob_{c}"] = y_prob[:, c]
    out.to_csv(RESULTS_DIR / "predictions" / f"{name}_{split_name}.csv", index=False)

    return metrics, ci


def run_split(split_name, X_full, feature_names):
    """Run all ChiralAldol models on one split."""
    logger.info(f"\n{'='*60}\n  ChiralAldol — {split_name}\n{'='*60}")

    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)
    tr, va, te = load_split(split_name)

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    logger.info(f"Train classes: {np.bincount(y[tr], minlength=4)}")
    logger.info(f"Test classes:  {np.bincount(y[te], minlength=4)}")

    # Identify feature groups by name
    steric_mask = np.array([n.endswith("_mean") or n.endswith("_std") or
                            n in ("n_conformers", "n_clusters")
                            for n in feature_names])
    cond_mask = np.array([not steric_mask[i] and not n.startswith("aux_")
                          for i, n in enumerate(feature_names)])
    aux_mask = np.array([n.startswith("aux_") for n in feature_names])

    # ---- Model 1: ChiralAldol-XGB (full) ----
    logger.info("\n--- ChiralAldol-XGB ---")
    X = X_full
    m = train_xgb(X[tr], y[tr], X[va], y[va])
    yp = m.predict(X[te])
    pp = m.predict_proba(X[te])
    evaluate_and_save("chiralaldol_xgboost", y[te], yp, pp, te, split_name)

    # ---- Model 2: SterOnly-XGB (ablation: only 3D steric) ----
    logger.info("\n--- SterOnly-XGB (ablation) ---")
    X_ster = X_full[:, steric_mask]
    m = train_xgb(X_ster[tr], y[tr], X_ster[va], y[va])
    yp = m.predict(X_ster[te])
    pp = m.predict_proba(X_ster[te])
    evaluate_and_save("chiralaldol_steronly_xgboost", y[te], yp, pp, te, split_name)

    # ---- Model 3: CondAux-XGB (ablation: no 3D, just cond+aux) ----
    logger.info("\n--- CondAux-XGB (ablation: no 3D) ---")
    X_condaux = X_full[:, cond_mask | aux_mask]
    m = train_xgb(X_condaux[tr], y[tr], X_condaux[va], y[va])
    yp = m.predict(X_condaux[te])
    pp = m.predict_proba(X_condaux[te])
    evaluate_and_save("chiralaldol_condaux_xgboost", y[te], yp, pp, te, split_name)


if __name__ == "__main__":
    logger.info("Building ChiralAldol features...")
    X, feature_names = build_chiralaldol_features(PROJECT)
    logger.info(f"Feature matrix: {X.shape}, names: {len(feature_names)}")

    for split in ["evans_temporal", "evans_scaffold", "evans_grouped_random_seed42"]:
        run_split(split, X, feature_names)

    logger.info("\nDone! Now run: conda run -n aldol-rxn python scripts/rebuild_comparison.py")
