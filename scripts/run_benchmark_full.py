#!/usr/bin/env python3
"""V5 Full Benchmark: includes MechAware-Full/BW + original models.

Requires run_mechaware.py to have completed first.

Usage:
    conda run -n aldol-rxn python scripts/run_benchmark_full.py
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, N_CLASSES, PRED_DIR, RESULTS_DIR, VALID_AUXILIARIES
from chiralaldol.data_io import (
    load_mechaware_bw,
    load_mechaware_full,
    load_splits,
    prepare_Xy,
    save_predictions,
)
from chiralaldol.model_trainers import (
    MajorityClassifier,
    train_et,
    train_lgbm,
    train_rf,
    train_xgb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_v5_full")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V5 Full Benchmark (with MechAware)")
    logger.info("=" * 70)

    # Load all feature matrices
    X_v4, y, _valid_mask, feat_names = prepare_Xy()
    n = len(y)

    feat_matrices = {"v4_84d": X_v4}

    # steric-only (34d)
    steric_path = FEAT_DIR / "steric_features.csv"
    if steric_path.exists():
        steric = pd.read_csv(steric_path).values.astype(np.float32)
        np.nan_to_num(steric, copy=False)
        feat_matrices["steric_34d"] = steric

    # conditions + aux (54d): condition_features.csv is 2434 rows, filter to 2427 valid
    cond_path = CLEAN_DIR / "condition_features.csv"
    clean_path = CLEAN_DIR / "substrate_aldol_clean.csv"
    if cond_path.exists() and clean_path.exists():
        cond_df = pd.read_csv(cond_path)
        clean = pd.read_csv(clean_path, usecols=["auxiliary_type", "n_defined_stereocenters"])
        valid_rows = clean["auxiliary_type"].isin(VALID_AUXILIARIES)
        cond = cond_df.loc[valid_rows].values.astype(np.float32)
        np.nan_to_num(cond, copy=False)
        clean = clean.loc[valid_rows].reset_index(drop=True)
        # borneol_ester has 0 rows in V5; exclude to avoid all-zero column
        aux_types = [a for a in VALID_AUXILIARIES if a != "borneol_ester"]
        aux = np.column_stack([
            *[(clean["auxiliary_type"] == a).astype(int).values for a in aux_types],
            clean["n_defined_stereocenters"].fillna(2).values,
        ]).astype(np.float32)
        condaux = np.hstack([cond, aux])
        feat_matrices["condaux_54d"] = condaux
        feat_matrices["cond_44d"] = cond

    # MechAware features
    X_ma_full = load_mechaware_full(feat_names=feat_names)
    X_ma_bw = load_mechaware_bw(feat_names=feat_names)
    if X_ma_full is not None:
        feat_matrices["ma_full"] = X_ma_full
        logger.info(f"MechAware-Full: {X_ma_full.shape[1]}d")
    if X_ma_bw is not None:
        feat_matrices["ma_bw"] = X_ma_bw
        logger.info(f"MechAware-BW: {X_ma_bw.shape[1]}d")

    logger.info(f"Data: {n} rows, {len(np.unique(y[y >= 0]))} classes")
    logger.info(f"Label dist: {dict(zip(*np.unique(y[y >= 0], return_counts=True)))}")

    splits = load_splits()
    logger.info(f"Loaded {len(splits)} splits")

    # Model registry
    MODELS = {
        # MechAware (new)
        "ma_full_xgb":  ("mechaware", "ma_full", train_xgb),
        "ma_full_lgbm": ("mechaware", "ma_full", train_lgbm),
        "ma_bw_xgb":    ("mechaware", "ma_bw", train_xgb),
        # Original steric
        "full_xgb":     ("steric", "v4_84d", train_xgb),
        "full_lgbm":    ("steric", "v4_84d", train_lgbm),
        "full_rf":      ("steric", "v4_84d", train_rf),
        "full_et":      ("steric", "v4_84d", train_et),
        "steronly_xgb": ("steric", "steric_34d", train_xgb),
        # Baseline
        "condaux_xgb":  ("baseline", "condaux_54d", train_xgb),
        "cond_xgb":     ("baseline", "cond_44d", train_xgb),
        "majority":     ("baseline", "v4_84d", lambda Xtr, ytr, Xv, yv: MajorityClassifier().fit(Xtr, ytr)),
    }

    all_results = []

    for model_key, (category, feat_key, trainer) in MODELS.items():
        if feat_key not in feat_matrices:
            logger.warning(f"  Skipping {model_key}: {feat_key} not available")
            continue

        X = feat_matrices[feat_key]
        logger.info(f"\n--- {model_key} ({category}, {X.shape[1]}d) ---")

        out_dir = PRED_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            tr = np.array(split_data["train"], dtype=int)
            va = np.array(split_data.get("val", []), dtype=int)
            te = np.array(split_data["test"], dtype=int)

            if len(va) == 0:
                va = tr[-max(1, len(tr) // 10):]
                tr = tr[:-len(va)]
            if len(tr) < 10 or len(te) < 3:
                continue

            model = trainer(X[tr], y[tr], X[va], y[va])
            y_pred = model.predict(X[te])
            y_prob = model.predict_proba(X[te]) if hasattr(model, "predict_proba") else np.eye(N_CLASSES)[y_pred.astype(int)]

            bal_acc = balanced_accuracy_score(y[te], y_pred)
            mcc = matthews_corrcoef(y[te], y_pred)

            save_predictions(out_dir / f"{model_key}_{split_name}.csv",
                            te, y[te], y_pred, y_prob, N_CLASSES)

            all_results.append({
                "model": model_key, "category": category, "feat_key": feat_key,
                "dims": X.shape[1], "split": split_name,
                "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                "n_train": len(tr), "n_test": len(te),
            })

        mr = [r for r in all_results if r["model"] == model_key]
        if mr:
            tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
            grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
            if tscv: logger.info(f"  TSCV: {np.mean(tscv):.3f} ± {np.std(tscv):.3f}")
            if grouped: logger.info(f"  Grouped: {np.mean(grouped):.3f} ± {np.std(grouped):.3f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    table_path = RESULTS_DIR / "tables" / "benchmark_v5_full.csv"
    results_df.to_csv(table_path, index=False)

    # Print summary
    print("\n" + "=" * 90)
    print("V4 FULL BENCHMARK SUMMARY")
    print("=" * 90)
    summary = []
    for mk in dict.fromkeys(r["model"] for r in all_results):
        mr = [r for r in all_results if r["model"] == mk]
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        summary.append({
            "model": mk, "dims": mr[0]["dims"],
            "TSCV": f"{np.mean(tscv):.3f}±{np.std(tscv):.3f}" if tscv else "---",
            "Scaffold": f"{scaffold[0]:.3f}" if scaffold else "---",
            "Grouped": f"{np.mean(grouped):.3f}±{np.std(grouped):.3f}" if grouped else "---",
        })
    print(pd.DataFrame(summary).to_string(index=False))
    print("=" * 90)
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
