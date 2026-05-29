#!/usr/bin/env python3
"""V4 Full Benchmark: includes MechAware-Full/BW + original models.

Requires run_mechaware_v4.py to have completed first.

Usage:
    conda run -n aldol-rxn python scripts/run_benchmark_v4_full.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import FEAT_DIR, N_CLASSES, PRED_DIR, RESULTS_DIR
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
logger = logging.getLogger("benchmark_v4_full")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V4 Full Benchmark (with MechAware)")
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

    # conditions + aux (50d)
    PROJECT = Path(__file__).resolve().parent.parent
    cond_path = PROJECT / "data" / "clean_v4" / "condition_features.csv"
    clean_path = PROJECT / "data" / "clean_v4" / "substrate_aldol_clean.csv"
    if cond_path.exists() and clean_path.exists():
        cond = pd.read_csv(cond_path).values.astype(np.float32)
        aux_types = ["evans", "crimmins_thione", "crimmins_oxathione", "other_auxiliary", "myers"]
        clean = pd.read_csv(clean_path, usecols=["auxiliary_type", "n_defined_stereocenters"])
        aux = np.column_stack([
            *[(clean["auxiliary_type"] == a).astype(int).values for a in aux_types],
            clean["n_defined_stereocenters"].fillna(2).values,
        ]).astype(np.float32)
        condaux = np.hstack([cond, aux])
        feat_matrices["condaux_50d"] = condaux
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
        "ma_full_xgb":  ("steric", "ma_full", train_xgb),
        "ma_full_lgbm": ("steric", "ma_full", train_lgbm),
        "ma_bw_xgb":    ("steric", "ma_bw", train_xgb),
        # Original steric
        "full_xgb":     ("steric", "v4_84d", train_xgb),
        "full_lgbm":    ("steric", "v4_84d", train_lgbm),
        "full_rf":      ("steric", "v4_84d", train_rf),
        "full_et":      ("steric", "v4_84d", train_et),
        "steronly_xgb": ("steric", "steric_34d", train_xgb),
        # Baseline
        "condaux_xgb":  ("baseline", "condaux_50d", train_xgb),
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
    table_path = RESULTS_DIR / "tables" / "benchmark_v4_full.csv"
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
