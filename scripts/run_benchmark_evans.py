#!/usr/bin/env python3
"""V4 Evans-only Benchmark: 11 models x 10 splits on Evans subset (1654 rows).

Filters all existing splits_v4/ JSON files to Evans-only rows, then runs the
same model suite as run_all_models_v4.py. Allows fair comparison of Evans-only
performance vs. full-dataset benchmark (benchmark_v4.csv).

Usage:
    conda run -n aldol-rxn python scripts/run_evans_benchmark_v4.py 2>&1 | tee logs/benchmark_v4_evans.log
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import CLEAN_DIR, PRED_DIR, RESULTS_DIR
from chiralaldol.data_io import load_mechaware_bw, load_mechaware_full, load_splits, prepare_Xy, save_predictions
from chiralaldol.feature_registry import select_features
from chiralaldol.model_trainers import (
    MajorityClassifier,
    train_et,
    train_lgbm,
    train_rf,
    train_xgb,
)

SUBSET = "evans"  # auxiliary_type value to filter on

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evans_benchmark_v4")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V4 Evans-only Benchmark")
    logger.info("=" * 70)

    X_all, y_raw, valid_mask, feat_names = prepare_Xy()

    # Evans-specific mask
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    evans_mask = (clean["auxiliary_type"] == SUBSET).values
    combined_mask = valid_mask & evans_mask

    n_total = len(y_raw)
    n_evans = evans_mask.sum()
    n_valid = valid_mask.sum()
    n_combined = combined_mask.sum()
    logger.info(f"Total rows: {n_total} | Evans: {n_evans} | Valid label: {n_valid} | Combined: {n_combined}")

    logger.info("Target: label_joint")
    logger.info(f"Subset: {SUBSET} ({n_combined} rows)")

    y = np.where(combined_mask, y_raw, -1).astype(int)
    y_valid = y[combined_mask]
    n_classes = len(np.unique(y_valid))
    logger.info(f"Classes: {n_classes}, dist: {dict(zip(*np.unique(y_valid, return_counts=True)))}")

    splits = load_splits()
    logger.info(f"Loaded {len(splits)} splits")

    # Pre-load MechAware feature matrices
    X_ma_bw = load_mechaware_bw(feat_names=feat_names)
    X_ma_full = load_mechaware_full(feat_names=feat_names)
    if X_ma_bw is None:
        X_ma_bw = X_all
    if X_ma_full is None:
        X_ma_full = X_all

    MODELS = {
        "v4b_full_xgb":        ("v4b",       lambda X, y, fn: X,                                           train_xgb),
        "v4b_full_lgbm":       ("v4b",       lambda X, y, fn: X,                                           train_lgbm),
        "v4b_full_rf":         ("v4b",       lambda X, y, fn: X,                                           train_rf),
        "v4b_full_et":         ("v4b",       lambda X, y, fn: X,                                           train_et),
        "v4b_condaux_xgb":     ("v4b",       lambda X, y, fn: select_features(X, fn, include=["conditions", "auxiliary", "chirality", "rgroup"]), train_xgb),
        "v4b_chiral_only_xgb": ("ablation",  lambda X, y, fn: select_features(X, fn, include="chirality"), train_xgb),
        "v4b_no_chiral_xgb":   ("ablation",  lambda X, y, fn: select_features(X, fn, exclude=["chirality", "rgroup", "chiralenv", "aldpri"]), train_xgb),
        "ma_full_xgb":         ("mechaware", lambda X, y, fn: X_ma_full,                                   train_xgb),
        "ma_bw_xgb":           ("mechaware", lambda X, y, fn: X_ma_bw,                                     train_xgb),
        "steronly_xgb":        ("steric",    lambda X, y, fn: select_features(X, fn, include="steric"),     train_xgb),
        "cond_xgb":            ("baseline",  lambda X, y, fn: select_features(X, fn, include="conditions"), train_xgb),
        "majority":            ("baseline",  lambda X, y, fn: X,                                           lambda Xtr, ytr, Xv, yv: MajorityClassifier().fit(Xtr, ytr)),
    }

    all_results = []

    for model_key, (category, feat_loader, trainer) in MODELS.items():
        logger.info(f"\n--- {model_key} ({category}) ---")

        X = feat_loader(X_all, y, feat_names)
        logger.info(f"  Features: {X.shape[1]}d")

        out_dir = PRED_DIR / f"evans_{category}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            tr_raw = np.array(split_data["train"], dtype=int)
            tr = tr_raw[combined_mask[tr_raw]]
            va_raw = np.array(split_data.get("val", []), dtype=int)
            va = va_raw[combined_mask[va_raw]] if len(va_raw) > 0 else np.array([], dtype=int)
            te_raw = np.array(split_data["test"], dtype=int)
            te = te_raw[combined_mask[te_raw]]

            if len(va) == 0:
                va = tr[-max(1, len(tr) // 10):]
                tr = tr[:-len(va)]

            if len(tr) < 10 or len(te) < 3:
                logger.warning(f"  Skipping {split_name}: train={len(tr)}, test={len(te)}")
                continue

            model = trainer(X[tr], y[tr], X[va], y[va])

            y_pred = model.predict(X[te])
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X[te])
            else:
                y_prob = np.zeros((len(te), n_classes))
                for i, p in enumerate(y_pred):
                    y_prob[i, int(p)] = 1.0

            bal_acc = balanced_accuracy_score(y[te], y_pred)
            mcc = matthews_corrcoef(y[te], y_pred)

            save_predictions(out_dir / f"{model_key}_{split_name}.csv",
                            te, y[te], y_pred, y_prob, n_classes)

            all_results.append({
                "model": model_key, "category": category, "split": split_name,
                "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                "n_train": len(tr), "n_test": len(te),
            })

        model_results = [r for r in all_results if r["model"] == model_key]
        if model_results:
            tscv = [r["bal_acc"] for r in model_results if "tscv" in r["split"]]
            grouped = [r["bal_acc"] for r in model_results if "grouped" in r["split"]]
            scaffold = [r["bal_acc"] for r in model_results if "scaffold" in r["split"]]
            if tscv:
                logger.info(f"  TSCV: {np.mean(tscv):.3f} ± {np.std(tscv):.3f}")
            if grouped:
                logger.info(f"  Grouped: {np.mean(grouped):.3f} ± {np.std(grouped):.3f}")
            if scaffold:
                logger.info(f"  Scaffold: {scaffold[0]:.3f}")

    results_df = pd.DataFrame(all_results)
    table_path = RESULTS_DIR / "tables" / "benchmark_v4_evans.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path, index=False)
    logger.info(f"\nSaved results to {table_path}")

    print("\n" + "=" * 80)
    print("V4 EVANS-ONLY BENCHMARK SUMMARY")
    print("=" * 80)
    summary_rows = []
    for model_key in MODELS:
        mr = [r for r in all_results if r["model"] == model_key]
        if not mr:
            continue
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        cat = mr[0]["category"]
        summary_rows.append({
            "model": model_key,
            "category": cat,
            "TSCV": f"{np.mean(tscv):.3f}±{np.std(tscv):.3f}" if tscv else "---",
            "Scaffold": f"{scaffold[0]:.3f}" if scaffold else "---",
            "Grouped": f"{np.mean(grouped):.3f}±{np.std(grouped):.3f}" if grouped else "---",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
