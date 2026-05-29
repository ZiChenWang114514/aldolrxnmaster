#!/usr/bin/env python3
"""Run Optuna-tuned models on ALL splits (TSCV + Scaffold + Grouped).

Applies the best hyperparameters found by run_optuna_v4.py to the full
benchmark suite, producing results comparable to run_all_models_v4.py.

Usage:
    conda run -n aldol-rxn python scripts/run_optuna_benchmark.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import N_CLASSES, N_JOBS, OPTUNA_DIR, PRED_DIR, RESULTS_DIR
from chiralaldol.data_io import load_mechaware_bw, load_splits, prepare_Xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optuna_bench")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Optuna-Tuned Full Benchmark")
    logger.info("=" * 70)

    X_v4b, y, valid_mask, feat_names = prepare_Xy()
    X_ma = load_mechaware_bw(feat_names=feat_names)
    if X_ma is None:
        X_ma = X_v4b
    splits = load_splits()

    # Load Optuna best params
    with open(OPTUNA_DIR / "et_optuna_best.json") as f:
        et_params = json.load(f)["best_params"]
    with open(OPTUNA_DIR / "xgb_optuna_best.json") as f:
        xgb_params = json.load(f)["best_params"]
    with open(OPTUNA_DIR / "ma_bw_xgb_optuna_best.json") as f:
        ma_params = json.load(f)["best_params"]

    # Model definitions
    models = {
        "et_optuna": {
            "X": X_v4b,
            "category": "optuna",
            "train_fn": lambda Xtr, ytr, Xva, yva: _train_et(Xtr, ytr, et_params),
        },
        "xgb_optuna": {
            "X": X_v4b,
            "category": "optuna",
            "train_fn": lambda Xtr, ytr, Xva, yva: _train_xgb(Xtr, ytr, xgb_params),
        },
        "ma_bw_xgb_optuna": {
            "X": X_ma,
            "category": "optuna",
            "train_fn": lambda Xtr, ytr, Xva, yva: _train_xgb(Xtr, ytr, ma_params),
        },
    }

    all_results = []
    out_dir = PRED_DIR / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, mdef in models.items():
        logger.info(f"\n--- {model_key} ---")
        X = mdef["X"]

        for split_name, split_data in sorted(splits.items()):
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

            model = mdef["train_fn"](X[tr], y[tr], X[va], y[va])
            y_pred = model.predict(X[te])
            y_prob = model.predict_proba(X[te]) if hasattr(model, "predict_proba") else None

            bal_acc = balanced_accuracy_score(y[te], y_pred)
            mcc = matthews_corrcoef(y[te], y_pred)

            # Save predictions
            out = pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": y_pred})
            if y_prob is not None:
                for c in range(min(N_CLASSES, y_prob.shape[1])):
                    out[f"prob_{c}"] = y_prob[:, c]
            out.to_csv(out_dir / f"{model_key}_{split_name}.csv", index=False)

            all_results.append({
                "model": model_key, "category": "optuna", "split": split_name,
                "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                "n_train": len(tr), "n_test": len(te),
            })

        # Summary
        mr = [r for r in all_results if r["model"] == model_key]
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        if tscv:
            logger.info(f"  TSCV: {np.mean(tscv):.4f} ± {np.std(tscv):.4f}")
        if scaffold:
            logger.info(f"  Scaffold: {scaffold[0]:.4f}")
        if grouped:
            logger.info(f"  Grouped: {np.mean(grouped):.4f} ± {np.std(grouped):.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    table_path = RESULTS_DIR / "tables" / "benchmark_v4_optuna.csv"
    results_df.to_csv(table_path, index=False)

    # Summary table
    print("\n" + "=" * 80)
    print("OPTUNA-TUNED BENCHMARK SUMMARY")
    print("=" * 80)

    baselines = {
        "v4b_full_et (default)": {"TSCV": 0.624, "Scaffold": 0.613, "Grouped": 0.738},
        "ma_bw_xgb (default)":   {"TSCV": 0.604, "Scaffold": 0.607, "Grouped": 0.752},
        "v4b_full_xgb (default)":{"TSCV": 0.602, "Scaffold": 0.589, "Grouped": 0.747},
    }

    for model_key in models:
        mr = [r for r in all_results if r["model"] == model_key]
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]

        t_mean = np.mean(tscv) if tscv else 0
        g_mean = np.mean(grouped) if grouped else 0
        s_val = scaffold[0] if scaffold else 0

        print(f"\n  {model_key}:")
        print(f"    TSCV:     {t_mean:.4f} ± {np.std(tscv):.4f}" if tscv else "")
        print(f"    Scaffold: {s_val:.4f}" if scaffold else "")
        print(f"    Grouped:  {g_mean:.4f} ± {np.std(grouped):.4f}" if grouped else "")

    print("\n  --- Baselines ---")
    for name, vals in baselines.items():
        print(f"  {name}: TSCV={vals['TSCV']}, Scaffold={vals['Scaffold']}, Grouped={vals['Grouped']}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results: {table_path}")


def _train_et(X_tr, y_tr, params):
    p = dict(params)
    p.update({"class_weight": "balanced", "random_state": 42, "n_jobs": N_JOBS})
    m = ExtraTreesClassifier(**p)
    m.fit(X_tr, y_tr)
    return m


def _train_xgb(X_tr, y_tr, params):
    p = dict(params)
    p.update({"objective": "multi:softprob", "num_class": N_CLASSES,
              "tree_method": "hist", "random_state": 42, "n_jobs": N_JOBS, "verbosity": 0})
    sw = compute_sample_weight("balanced", y_tr)
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


if __name__ == "__main__":
    main()
