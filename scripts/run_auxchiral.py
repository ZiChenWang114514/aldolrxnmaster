#!/usr/bin/env python
"""AuxChiral models: Evans auxiliary R/S chirality + reaction conditions.

A chemistry-prior-driven approach that explicitly encodes the Evans
auxiliary's R/S configuration as the primary feature. This is what a
human chemist uses to predict stereochemical outcome.

Models:
  1. AuxChiral-XGB:        aux(6d) + conditions(35d) = 41d
  2. AuxChiral+Ald-XGB:    41d + aldehyde descriptors(17d) + ald FP SVD(32d) = 90d
  3. AuxChiral-LGBM:       aux(6d) + conditions(35d) = 41d (LightGBM)
  4. CondOnly-XGB:         conditions(35d) only — ablation: no aux
  5. AuxNoBase-XGB:        aux(6d) + conditions_no_base(27d) = 33d — ablation: no base
"""

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"
for d in [RESULTS_DIR / "predictions", RESULTS_DIR / "tables"]:
    d.mkdir(parents=True, exist_ok=True)

# Base feature columns (to identify for ablation)
BASE_COLS = [c for c in pd.read_csv(FEAT_DIR / "reaction_conditions.csv", nrows=0).columns
             if c.startswith("base_")]


def load_features():
    """Load all feature sources needed by AuxChiral models."""
    aux = pd.read_csv(FEAT_DIR / "auxchiral_features.csv")
    cond = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    desc = pd.read_csv(FEAT_DIR / "rdkit_descriptors.csv")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")

    # Aldehyde descriptors only (17 cols)
    ald_cols = [c for c in desc.columns if c.startswith("aldehyde_")]
    ald_desc = desc[ald_cols].values.astype(np.float32)

    # Aldehyde Morgan FP for SVD
    fps = np.load(FEAT_DIR / "morgan_fps.npz")
    ald_fp = fps["aldehyde"].astype(np.float32)

    return aux, cond, ald_desc, ald_fp, labels


def load_split(split_name):
    with open(SPLIT_DIR / f"{split_name}.json") as f:
        sp = json.load(f)
    return np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])


def train_xgb(X_tr, y_tr, X_val, y_val):
    """Train XGBoost with 3-config grid search (same as run_all_models.py)."""
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multi:softprob", "num_class": 4, "tree_method": "hist",
                    "random_state": 42, "n_jobs": -1, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def train_lgbm(X_tr, y_tr, X_val, y_val):
    """Train LightGBM with 3-config grid search."""
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 47, "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15, "num_leaves": 23, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multiclass", "num_class": 4, "class_weight": "balanced",
                    "random_state": 42, "n_jobs": -1, "verbose": -1, "min_child_samples": 10})
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X_tr, y_tr)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def evaluate_and_save(name, y_test, y_pred, y_prob, test_idx, split_name):
    """Evaluate model and save predictions (standard format)."""
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  {name}: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint={metrics['joint_accuracy']:.4f}, "
                f"F1m={metrics['f1_macro']:.4f}")

    # Save predictions CSV (standard format per CLAUDE.md)
    out = pd.DataFrame({
        "idx": test_idx,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    for c in range(4):
        out[f"prob_{c}"] = y_prob[:, c]
    out.to_csv(RESULTS_DIR / "predictions" / f"{name}_{split_name}.csv", index=False)

    return metrics, ci


def run_split(split_name):
    """Run all AuxChiral models on one split."""
    logger.info(f"\n{'='*60}\n  AuxChiral — {split_name}\n{'='*60}")

    aux, cond, ald_desc, ald_fp, labels = load_features()
    y = labels["label_joint"].values.astype(int)
    tr, va, te = load_split(split_name)

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    logger.info(f"Train classes: {np.bincount(y[tr], minlength=4)}")
    logger.info(f"Test classes:  {np.bincount(y[te], minlength=4)}")

    # Build feature matrices
    X_aux = aux.values.astype(np.float32)                   # (n, 6)
    X_cond = cond.values.astype(np.float32)                 # (n, 35)
    X_auxcond = np.hstack([X_aux, X_cond])                  # (n, 41)

    # Aldehyde features: descriptors + SVD-reduced FP
    svd = TruncatedSVD(n_components=32, random_state=42)
    svd.fit(ald_fp[tr])
    ald_fp_svd = svd.transform(ald_fp)                      # (n, 32)
    X_ald = np.hstack([ald_desc, ald_fp_svd])               # (n, 49)

    X_auxcond_ald = np.hstack([X_auxcond, X_ald])           # (n, 90)

    # Conditions only (no aux) — ablation
    X_cond_only = X_cond                                    # (n, 35)

    # AuxChiral no base — ablation
    cond_nobase_cols = [c for c in cond.columns if c not in BASE_COLS]
    X_cond_nobase = cond[cond_nobase_cols].values.astype(np.float32)
    X_aux_nobase = np.hstack([X_aux, X_cond_nobase])        # (n, 6+27)

    # Replace NaN/inf
    for X in [X_auxcond, X_auxcond_ald, X_cond_only, X_aux_nobase]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Model 1: AuxChiral-XGB ----
    logger.info("\n--- AuxChiral-XGB ---")
    m = train_xgb(X_auxcond[tr], y[tr], X_auxcond[va], y[va])
    yp = m.predict(X_auxcond[te])
    pp = m.predict_proba(X_auxcond[te])
    evaluate_and_save("auxchiral_xgboost", y[te], yp, pp, te, split_name)

    # ---- Model 2: AuxChiral+Ald-XGB ----
    logger.info("\n--- AuxChiral+Ald-XGB ---")
    m = train_xgb(X_auxcond_ald[tr], y[tr], X_auxcond_ald[va], y[va])
    yp = m.predict(X_auxcond_ald[te])
    pp = m.predict_proba(X_auxcond_ald[te])
    evaluate_and_save("auxchiral_ald_xgboost", y[te], yp, pp, te, split_name)

    # ---- Model 3: AuxChiral-LGBM ----
    logger.info("\n--- AuxChiral-LGBM ---")
    m = train_lgbm(X_auxcond[tr], y[tr], X_auxcond[va], y[va])
    yp = m.predict(X_auxcond[te])
    pp = m.predict_proba(X_auxcond[te])
    evaluate_and_save("auxchiral_lgbm", y[te], yp, pp, te, split_name)

    # ---- Model 4: CondOnly-XGB (ablation: no aux) ----
    logger.info("\n--- CondOnly-XGB (ablation) ---")
    m = train_xgb(X_cond_only[tr], y[tr], X_cond_only[va], y[va])
    yp = m.predict(X_cond_only[te])
    pp = m.predict_proba(X_cond_only[te])
    evaluate_and_save("auxchiral_noaux_xgboost", y[te], yp, pp, te, split_name)

    # ---- Model 5: AuxNoBase-XGB (ablation: no base) ----
    logger.info("\n--- AuxNoBase-XGB (ablation) ---")
    m = train_xgb(X_aux_nobase[tr], y[tr], X_aux_nobase[va], y[va])
    yp = m.predict(X_aux_nobase[te])
    pp = m.predict_proba(X_aux_nobase[te])
    evaluate_and_save("auxchiral_nobase_xgboost", y[te], yp, pp, te, split_name)


if __name__ == "__main__":
    for split in ["evans_temporal", "evans_scaffold", "evans_grouped_random_seed42"]:
        run_split(split)
    logger.info("\nDone! Now run: conda run -n aldol-rxn python scripts/rebuild_comparison.py")
