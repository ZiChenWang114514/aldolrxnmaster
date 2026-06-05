#!/usr/bin/env python3
"""C1: Auxiliary-aware independent modeling — Evans/Crimmins/Oppolzer each get their own TSCV.

Literature motivation: [7] Betinol 2023 — generality evaluation framework.
Chemical space audit showed Evans=0.771, Crimmins=0.453, Oppolzer=0.371 in unified model.

Usage:
    conda run -n aldol-rxn python scripts/run_aux_models.py
"""

import json
import logging
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from chiralaldol.config import CLEAN_DIR, OPTUNA_DIR, RESULTS_DIR
from chiralaldol.data_io import prepare_Xy

CLEAN_CSV = CLEAN_DIR / "substrate_aldol_clean.csv"
OUT_DIR = RESULTS_DIR / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aux_models")


def load_optuna_params():
    with open(OPTUNA_DIR / "xgb_optuna_best.json") as f:
        xgb_p = json.load(f)["best_params"]
    with open(OPTUNA_DIR / "et_optuna_best.json") as f:
        et_p = json.load(f)["best_params"]
    return xgb_p, et_p


def run_tscv_for_subset(X, y, n_folds, model_type, params, subset_name):
    """Run time-series CV for a subset."""
    n = len(y)
    if n < 20:
        logger.warning(f"  {subset_name}: too few samples ({n}), skipping")
        return None

    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_scores = []
    all_cm = np.zeros((4, 4), dtype=int)

    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        if len(te_idx) < 3:
            continue

        if model_type == "xgb":
            p = dict(params)
            p.update({"objective": "multi:softprob", "num_class": 4,
                       "tree_method": "hist", "random_state": 42, "n_jobs": 8, "verbosity": 0})
            sw = compute_sample_weight("balanced", y[tr_idx])
            model = xgb.XGBClassifier(**p)
            model.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        else:
            p = dict(params)
            p.update({"class_weight": "balanced", "random_state": 42, "n_jobs": 8})
            model = ExtraTreesClassifier(**p)
            model.fit(X[tr_idx], y[tr_idx])

        y_pred = model.predict(X[te_idx])
        acc = balanced_accuracy_score(y[te_idx], y_pred)
        fold_scores.append(acc)

        cm = confusion_matrix(y[te_idx], y_pred, labels=[0, 1, 2, 3])
        all_cm += cm

    if not fold_scores:
        return None

    return {
        "subset": subset_name,
        "n_samples": n,
        "n_folds": len(fold_scores),
        "model": model_type,
        "tscv_mean": round(np.mean(fold_scores), 4),
        "tscv_std": round(np.std(fold_scores), 4),
        "fold_scores": [round(s, 4) for s in fold_scores],
        "confusion_matrix": all_cm.tolist(),
    }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Auxiliary-Aware Independent Modeling")
    logger.info("=" * 60)

    X, y, valid_mask, _ = prepare_Xy()
    meta = pd.read_csv(CLEAN_CSV)

    xgb_params, et_params = load_optuna_params()

    # Define subsets
    subsets = [
        ("Evans", meta["auxiliary_type"] == "evans", 4),
        ("Crimmins_thione", meta["auxiliary_type"] == "crimmins_thione", 3),
        ("Crimmins_oxathione", meta["auxiliary_type"] == "crimmins_oxathione", 3),
        ("Oppolzer", meta["auxiliary_type"] == "oppolzer", 3),
        ("All", np.ones(len(meta), dtype=bool), 4),
    ]

    all_results = []

    for subset_name, mask, n_folds in subsets:
        sub_mask = mask.values if hasattr(mask, 'values') else mask
        sub_valid = sub_mask & valid_mask
        sub_idx = np.where(sub_valid)[0]

        if len(sub_idx) < 20:
            logger.info(f"\n{subset_name}: {len(sub_idx)} samples (too few, skipping)")
            continue

        X_sub = X[sub_idx]
        y_sub = y[sub_idx]

        logger.info(f"\n{'='*40}")
        logger.info(f"{subset_name}: {len(sub_idx)} samples, {n_folds}-fold TSCV")
        logger.info(f"  Class dist: {dict(zip(*np.unique(y_sub, return_counts=True)))}")

        # Run both XGB and ET
        for model_type, params in [("xgb", xgb_params), ("et", et_params)]:
            result = run_tscv_for_subset(X_sub, y_sub, n_folds, model_type, params, subset_name)
            if result:
                all_results.append(result)
                logger.info(f"  {model_type}: TSCV = {result['tscv_mean']:.4f} ± {result['tscv_std']:.4f}")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([{
        "subset": r["subset"],
        "n_samples": r["n_samples"],
        "model": r["model"],
        "n_folds": r["n_folds"],
        "tscv_mean": r["tscv_mean"],
        "tscv_std": r["tscv_std"],
    } for r in all_results])
    results_df.to_csv(OUT_DIR / "benchmark_v4_per_auxiliary.csv", index=False)

    # Save detailed results with confusion matrices
    with open(OUT_DIR / "per_auxiliary_detail.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PER-AUXILIARY BENCHMARK SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['subset']:25s} ({r['model']:3s}): TSCV = {r['tscv_mean']:.4f} ± {r['tscv_std']:.4f}  (n={r['n_samples']})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results: {OUT_DIR / 'benchmark_v4_per_auxiliary.csv'}")


if __name__ == "__main__":
    main()
