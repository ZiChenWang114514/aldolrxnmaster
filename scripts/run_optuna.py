#!/usr/bin/env python3
"""A1: Optuna hyperparameter search for ExtraTrees and XGBoost on TSCV.

Literature motivation: [13] Baczewska 2024 (Angew. Chem.) — Optuna 300-step
search significantly improved NN on ~1000 catalyst reactions.

Usage:
    conda run -n aldol-rxn python scripts/run_optuna_v4.py
    conda run -n aldol-rxn python scripts/run_optuna_v4.py --model xgb --n-trials 300
    conda run -n aldol-rxn python scripts/run_optuna_v4.py --model et --n-trials 300
    conda run -n aldol-rxn python scripts/run_optuna_v4.py --model all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import OPTUNA_DIR, SPLITS_DIR
from chiralaldol.data_io import load_mechaware_bw, prepare_Xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optuna_v4")

# Suppress Optuna's verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════ DATA LOADING ═══════════════════════════

def load_data():
    """Load V4 features, labels, and TSCV splits."""
    X, y, valid_mask, _ = prepare_Xy()

    # Load TSCV splits only (4 folds)
    tscv_splits = []
    for i in range(1, 5):
        fpath = SPLITS_DIR / f"tscv_fold{i}.json"
        if fpath.exists():
            with open(fpath) as f:
                tscv_splits.append(json.load(f))

    logger.info(f"Data: {X.shape[0]} rows × {X.shape[1]}d features, {len(tscv_splits)} TSCV folds")
    return X, y, valid_mask, tscv_splits


def eval_tscv(model_fn, X, y, valid_mask, splits):
    """Evaluate a model factory across all TSCV folds, return mean balanced accuracy."""
    scores = []
    for split in splits:
        tr_raw = np.array(split["train"], dtype=int)
        tr = tr_raw[valid_mask[tr_raw]]
        te_raw = np.array(split["test"], dtype=int)
        te = te_raw[valid_mask[te_raw]]

        if len(tr) < 10 or len(te) < 3:
            continue

        # Use last 10% of train as validation for early stopping
        va = tr[-max(1, len(tr) // 10):]
        tr_sub = tr[:-len(va)]

        model = model_fn(X[tr_sub], y[tr_sub], X[va], y[va])
        y_pred = model.predict(X[te])
        scores.append(balanced_accuracy_score(y[te], y_pred))

    return np.mean(scores) if scores else 0.0


# ═══════════════════════════ OPTUNA OBJECTIVES ═══════════════════════════

def create_et_objective(X, y, valid_mask, splits):
    """Create Optuna objective for ExtraTrees."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20, 30, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 8,
        }

        def model_fn(X_tr, y_tr, X_va, y_va):
            m = ExtraTreesClassifier(**params)
            m.fit(X_tr, y_tr)
            return m

        return eval_tscv(model_fn, X, y, valid_mask, splits)

    return objective


def create_xgb_objective(X, y, valid_mask, splits):
    """Create Optuna objective for XGBoost."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "multi:softprob",
            "num_class": 4,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 8,
            "verbosity": 0,
        }

        def model_fn(X_tr, y_tr, X_va, y_va):
            sw = compute_sample_weight("balanced", y_tr)
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=sw)
            return m

        return eval_tscv(model_fn, X, y, valid_mask, splits)

    return objective


# ═══════════════════════════ MECHAWARE VARIANTS ═══════════════════════════

def create_ma_bw_xgb_objective(X_ma, y, valid_mask, splits):
    """Optuna objective for MechAware-BW + XGBoost."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "multi:softprob",
            "num_class": 4,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 8,
            "verbosity": 0,
        }

        def model_fn(X_tr, y_tr, X_va, y_va):
            sw = compute_sample_weight("balanced", y_tr)
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=sw)
            return m

        return eval_tscv(model_fn, X_ma, y, valid_mask, splits)

    return objective


# ═══════════════════════════ REPORTING ═══════════════════════════

def save_results(study, model_name, X, y, valid_mask, splits, model_fn_factory):
    """Save best params, retrain with best params, report per-fold scores."""
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    # Save study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OPTUNA_DIR / f"{model_name}_trials.csv", index=False)

    best = study.best_params
    best_val = study.best_value

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} — Best TSCV: {best_val:.4f}")
    logger.info(f"  Best params: {json.dumps(best, indent=2, default=str)}")
    logger.info(f"{'='*60}")

    # Retrain and get per-fold breakdown
    model_fn = model_fn_factory(best)
    fold_scores = []
    for i, split in enumerate(splits):
        tr_raw = np.array(split["train"], dtype=int)
        tr = tr_raw[valid_mask[tr_raw]]
        te_raw = np.array(split["test"], dtype=int)
        te = te_raw[valid_mask[te_raw]]
        va = tr[-max(1, len(tr) // 10):]
        tr_sub = tr[:-len(va)]

        m = model_fn(X[tr_sub], y[tr_sub], X[va], y[va])
        score = balanced_accuracy_score(y[te], m.predict(X[te]))
        fold_scores.append(score)
        logger.info(f"  Fold {i+1}: {score:.4f}")

    logger.info(f"  Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save best params JSON
    result = {
        "model": model_name,
        "best_tscv": round(best_val, 4),
        "fold_scores": [round(s, 4) for s in fold_scores],
        "mean": round(np.mean(fold_scores), 4),
        "std": round(np.std(fold_scores), 4),
        "best_params": best,
        "n_trials": len(study.trials),
    }
    with open(OPTUNA_DIR / f"{model_name}_best.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for V4 models")
    parser.add_argument("--model", choices=["et", "xgb", "ma_bw_xgb", "all"], default="all",
                        help="Which model to optimize")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of Optuna trials per model")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"Optuna V4 Hyperparameter Search (n_trials={args.n_trials})")
    logger.info("=" * 70)

    X, y, valid_mask, splits = load_data()
    results = []

    # --- ExtraTrees ---
    if args.model in ("et", "all"):
        logger.info("\n>>> ExtraTrees optimization...")
        study_et = optuna.create_study(direction="maximize", study_name="et_v4b")
        study_et.optimize(create_et_objective(X, y, valid_mask, splits),
                         n_trials=args.n_trials, show_progress_bar=True)

        def et_factory(params):
            def fn(X_tr, y_tr, X_va, y_va):
                p = dict(params)
                p.update({"class_weight": "balanced", "random_state": 42, "n_jobs": 8})
                m = ExtraTreesClassifier(**p)
                m.fit(X_tr, y_tr)
                return m
            return fn

        results.append(save_results(study_et, "et_optuna", X, y, valid_mask, splits, et_factory))

    # --- XGBoost ---
    if args.model in ("xgb", "all"):
        logger.info("\n>>> XGBoost optimization...")
        study_xgb = optuna.create_study(direction="maximize", study_name="xgb_v4b")
        study_xgb.optimize(create_xgb_objective(X, y, valid_mask, splits),
                          n_trials=args.n_trials, show_progress_bar=True)

        def xgb_factory(params):
            def fn(X_tr, y_tr, X_va, y_va):
                p = dict(params)
                p.update({"objective": "multi:softprob", "num_class": 4,
                          "tree_method": "hist", "random_state": 42, "n_jobs": 8, "verbosity": 0})
                sw = compute_sample_weight("balanced", y_tr)
                m = xgb.XGBClassifier(**p)
                m.fit(X_tr, y_tr, sample_weight=sw)
                return m
            return fn

        results.append(save_results(study_xgb, "xgb_optuna", X, y, valid_mask, splits, xgb_factory))

    # --- MechAware BW + XGBoost ---
    if args.model in ("ma_bw_xgb", "all"):
        X_ma = load_mechaware_bw()
        if X_ma is not None:
            logger.info("\n>>> MechAware-BW + XGBoost optimization...")
            study_ma = optuna.create_study(direction="maximize", study_name="ma_bw_xgb_v4b")
            study_ma.optimize(create_ma_bw_xgb_objective(X_ma, y, valid_mask, splits),
                             n_trials=args.n_trials, show_progress_bar=True)

            def ma_xgb_factory(params):
                def fn(X_tr, y_tr, X_va, y_va):
                    p = dict(params)
                    p.update({"objective": "multi:softprob", "num_class": 4,
                              "tree_method": "hist", "random_state": 42, "n_jobs": 8, "verbosity": 0})
                    sw = compute_sample_weight("balanced", y_tr)
                    m = xgb.XGBClassifier(**p)
                    m.fit(X_tr, y_tr, sample_weight=sw)
                    return m
                return fn

            results.append(save_results(study_ma, "ma_bw_xgb_optuna", X_ma, y, valid_mask, splits, ma_xgb_factory))

    # --- Final summary ---
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 70)

    # Baseline comparison
    baselines = {
        "v4b_full_et (default)": 0.624,
        "v4b_full_xgb (default)": 0.602,
        "ma_bw_xgb (default)": 0.604,
    }

    for r in results:
        name = r["model"]
        baseline_key = {
            "et_optuna": "v4b_full_et (default)",
            "xgb_optuna": "v4b_full_xgb (default)",
            "ma_bw_xgb_optuna": "ma_bw_xgb (default)",
        }.get(name, "")
        baseline = baselines.get(baseline_key, 0)
        delta = r["mean"] - baseline
        print(f"  {name}: TSCV = {r['mean']:.4f} ± {r['std']:.4f}  (Δ = {delta:+.4f} vs default)")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {OPTUNA_DIR}/")


if __name__ == "__main__":
    main()
