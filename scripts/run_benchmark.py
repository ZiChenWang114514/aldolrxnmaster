#!/usr/bin/env python3
"""V5 Model Benchmark: 11 active models x 10 splits (no DRFP -- confirmed leakage).

Usage:
    conda run -n aldol-rxn python scripts/run_benchmark.py
"""

import argparse
import logging
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, top_k_accuracy_score

from chiralaldol.config import PRED_DIR, RESULTS_DIR, TARGET_LABEL
from chiralaldol.data_io import load_mechaware_bw, load_mechaware_full, load_splits, prepare_Xy, save_predictions
from chiralaldol.feature_registry import select_features
from chiralaldol.model_trainers import (
    MajorityClassifier,
    train_et,
    train_lgbm,
    train_rf,
    train_xgb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_v4")


def main(target=None, tag=None):
    t0 = time.time()
    target = target or TARGET_LABEL
    tag = tag or "v4"
    logger.info("=" * 70)
    logger.info(f"V4 Model Benchmark  (target={target}, tag={tag})")
    logger.info("=" * 70)

    X_all, y_raw, valid_mask, feat_names = prepare_Xy(target_label=target)
    logger.info(f"Target: {target}")
    logger.info(f"Data: {len(y_raw)} total rows, {valid_mask.sum()} valid, {X_all.shape[1]} features")

    y = np.where(valid_mask, y_raw, -1).astype(int)
    y_valid = y[valid_mask]
    n_classes = len(np.unique(y_valid))
    logger.info(f"Classes: {n_classes}, dist: {dict(zip(*np.unique(y_valid, return_counts=True)))}")

    # Retarget XGB's num_class without touching global N_CLASSES (enables 2-class experiments)
    _xgb = partial(train_xgb, n_classes=n_classes)
    splits = load_splits()
    logger.info(f"Loaded {len(splits)} splits")

    # Pre-load MechAware feature matrices
    X_ma_bw = load_mechaware_bw(feat_names=feat_names)
    X_ma_full = load_mechaware_full(feat_names=feat_names)
    if X_ma_bw is None:
        X_ma_bw = X_all
    if X_ma_full is None:
        X_ma_full = X_all

    # Model registry: (category, feature_loader, trainer)
    MODELS = {
        # === V4b: with chirality + rgroup + chiralenv features ===
        "v4b_full_xgb":        ("v4b",       lambda X, y, fn: X,                                           _xgb),
        "v4b_full_lgbm":       ("v4b",       lambda X, y, fn: X,                                           train_lgbm),
        "v4b_full_rf":         ("v4b",       lambda X, y, fn: X,                                           train_rf),
        "v4b_full_et":         ("v4b",       lambda X, y, fn: X,                                           train_et),
        "v4b_condaux_xgb":     ("v4b",       lambda X, y, fn: select_features(X, fn, include=["conditions", "auxiliary", "chirality", "rgroup"]), _xgb),
        # === ablation ===
        "v4b_chiral_only_xgb": ("ablation",  lambda X, y, fn: select_features(X, fn, include="chirality"), _xgb),
        "v4b_no_chiral_xgb":   ("ablation",  lambda X, y, fn: select_features(X, fn, exclude=["chirality", "rgroup", "chiralenv", "aldpri"]), _xgb),
        # === MechAware + V4b features ===
        "ma_full_xgb":         ("mechaware", lambda X, y, fn: X_ma_full,                                   _xgb),
        "ma_bw_xgb":           ("mechaware", lambda X, y, fn: X_ma_bw,                                     _xgb),
        # === original V4 baselines ===
        "steronly_xgb":        ("steric",    lambda X, y, fn: select_features(X, fn, include="steric"),     _xgb),
        "cond_xgb":            ("baseline",  lambda X, y, fn: select_features(X, fn, include="conditions"), _xgb),
        "majority":            ("baseline",  lambda X, y, fn: X,                                           lambda Xtr, ytr, Xv, yv: MajorityClassifier().fit(Xtr, ytr)),
    }

    all_results = []

    for model_key, (category, feat_loader, trainer) in MODELS.items():
        logger.info(f"\n--- {model_key} ({category}) ---")

        X = feat_loader(X_all, y, feat_names)
        logger.info(f"  Features: {X.shape[1]}d")

        out_dir = PRED_DIR / category if tag == "v4" else PRED_DIR / category / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            tr_raw = np.array(split_data["train"], dtype=int)
            tr = tr_raw[valid_mask[tr_raw]]
            va_raw = np.array(split_data.get("val", []), dtype=int)
            va = va_raw[valid_mask[va_raw]] if len(va_raw) > 0 else np.array([], dtype=int)
            te_raw = np.array(split_data["test"], dtype=int)
            te = te_raw[valid_mask[te_raw]]

            if len(va) == 0:
                va = tr[-max(1, len(tr) // 10):]
                tr = tr[:-len(va)]

            if len(tr) < 10 or len(te) < 3:
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

            # top-2 accuracy (4-class only; degenerate for 2-class)
            if n_classes > 2:
                top2 = top_k_accuracy_score(y[te], y_prob, k=2, labels=list(range(n_classes)))
            else:
                top2 = np.nan
            # pair accuracy: project 4-class CIP prediction onto the Ca==Cb axis
            if n_classes == 4:
                pair_true = (y[te] // 2 == y[te] % 2).astype(int)
                pair_pred = (y_pred // 2 == y_pred % 2).astype(int)
                pair_acc = balanced_accuracy_score(pair_true, pair_pred)
            else:
                pair_acc = np.nan

            save_predictions(out_dir / f"{model_key}_{split_name}.csv",
                            te, y[te], y_pred, y_prob, n_classes)

            all_results.append({
                "model": model_key, "category": category, "split": split_name,
                "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                "top2": round(top2, 4) if not np.isnan(top2) else np.nan,
                "pair_acc": round(pair_acc, 4) if not np.isnan(pair_acc) else np.nan,
                "n_train": len(tr), "n_test": len(te),
            })

        # Log summary for this model
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

    # Save results table
    results_df = pd.DataFrame(all_results)
    table_path = RESULTS_DIR / "tables" / f"benchmark_{tag}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path, index=False)
    logger.info(f"\nSaved results to {table_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("V4 BENCHMARK SUMMARY")
    print("=" * 80)
    summary_rows = []
    for model_key in MODELS:
        mr = [r for r in all_results if r["model"] == model_key]
        if not mr:
            continue
        tscv_r = [r for r in mr if "tscv" in r["split"]]
        tscv = [r["bal_acc"] for r in tscv_r]
        tscv_w = [r["n_test"] for r in tscv_r]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        # n_test-weighted TSCV mean (fixes fold4 n=75 over-weighting)
        tscv_wmean = np.average(tscv, weights=tscv_w) if tscv else float("nan")
        top2 = [r["top2"] for r in tscv_r if not pd.isna(r["top2"])]
        pair = [r["pair_acc"] for r in tscv_r if not pd.isna(r["pair_acc"])]
        cat = mr[0]["category"]
        summary_rows.append({
            "model": model_key,
            "category": cat,
            "TSCV_wmean": f"{tscv_wmean:.3f}" if tscv else "---",
            "TSCV_mean": f"{np.mean(tscv):.3f}±{np.std(tscv):.3f}" if tscv else "---",
            "top2": f"{np.mean(top2):.3f}" if top2 else "---",
            "pair_acc": f"{np.mean(pair):.3f}" if pair else "---",
            "Scaffold": f"{scaffold[0]:.3f}" if scaffold else "---",
            "Grouped": f"{np.mean(grouped):.3f}±{np.std(grouped):.3f}" if grouped else "---",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                    help="label column in labels.csv (default: config.TARGET_LABEL)")
    ap.add_argument("--tag", default=None,
                    help="output tag; isolates table/pred dir (default: v4)")
    args = ap.parse_args()
    main(target=args.target, tag=args.tag)
