#!/usr/bin/env python3
"""Phase A4+A5: Time-series CV + V2-XGB retrain on cleaned data.

Implements 4-fold temporal CV and retrains V2-XGB (75d) on the cleaned
1801-row dataset (evans_v2_clean.csv).

Usage:
    conda run -n aldol-rxn python scripts/run_timeseries_cv.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHIRALALDOL_DIR = PROJECT / "data" / "processed" / "chiralaldol"
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"


# ---------------------------------------------------------------------------
# XGBoost training (same as existing pipeline)
# ---------------------------------------------------------------------------

def train_xgb(X_tr, y_tr, X_val, y_val):
    """Train XGBoost with 3-config grid search on validation set."""
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
            "n_jobs": 2, "verbosity": 0,
            "gamma": 0.1, "reg_lambda": 1.0,
        })
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


# ---------------------------------------------------------------------------
# Time-series CV folds
# ---------------------------------------------------------------------------

TIMESERIES_FOLDS = [
    {"name": "fold1", "train_cutoff": 2013, "test_start": 2014, "test_end": 2015},
    {"name": "fold2", "train_cutoff": 2015, "test_start": 2016, "test_end": 2017},
    {"name": "fold3", "train_cutoff": 2017, "test_start": 2018, "test_end": 2019},
    {"name": "fold4", "train_cutoff": 2019, "test_start": 2020, "test_end": 2099},
]


def make_timeseries_split(df, fold):
    """Create train/val/test indices for a time-series fold.

    train: Year <= train_cutoff (use last 10% as val)
    test:  test_start <= Year <= test_end
    """
    group_years = df.groupby("group_id")["Year"].min()

    train_groups = set(group_years[group_years <= fold["train_cutoff"]].index)
    test_groups = set(group_years[
        (group_years >= fold["test_start"]) &
        (group_years <= fold["test_end"])
    ].index)

    train_all_idx = df[df["group_id"].isin(train_groups)].index.tolist()
    test_idx = df[df["group_id"].isin(test_groups)].index.tolist()

    if not train_all_idx or not test_idx:
        return None, None, None

    # Split train into train/val (90/10 by groups)
    rng = np.random.RandomState(42)
    train_group_list = sorted(train_groups)
    rng.shuffle(train_group_list)
    n_val_groups = max(1, int(len(train_group_list) * 0.1))
    val_groups = set(train_group_list[:n_val_groups])
    pure_train_groups = set(train_group_list[n_val_groups:])

    train_idx = df[df["group_id"].isin(pure_train_groups)].index.tolist()
    val_idx = df[df["group_id"].isin(val_groups)].index.tolist()

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Phase A4+A5: Time-series CV + V2-XGB Retrain")
    logger.info("=" * 60)

    # Load cleaned data
    df = pd.read_csv(PROJECT / "data" / "processed" / "evans_v2_clean.csv")
    logger.info(f"Loaded cleaned data: {len(df)} rows")

    # Load V2 features (75d)
    from chiralaldol.feature_builder import build_chiralaldol_v2_features
    X, feature_names = build_chiralaldol_v2_features(PROJECT)
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    assert len(X) == len(df) == len(y), f"Size mismatch: X={len(X)}, df={len(df)}, y={len(y)}"
    logger.info(f"Features: {X.shape} ({len(feature_names)}d)")

    # ==================================================================
    # Part 1: Time-series CV (4 folds)
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TIME-SERIES CV (4 folds)")
    logger.info("=" * 60)

    fold_results = []

    for fold in TIMESERIES_FOLDS:
        logger.info(f"\n--- {fold['name']}: train≤{fold['train_cutoff']}, "
                    f"test {fold['test_start']}-{fold['test_end']} ---")

        tr, va, te = make_timeseries_split(df, fold)
        if tr is None:
            logger.warning(f"  Skipping {fold['name']}: insufficient data")
            continue

        logger.info(f"  Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")

        # Class distribution in test
        y_test = y[te]
        test_dist = {int(c): int((y_test == c).sum()) for c in range(4)}
        logger.info(f"  Test class dist: {test_dist}")

        # Train
        model = train_xgb(X[tr], y[tr], X[va], y[va])
        y_pred = model.predict(X[te])
        y_prob = model.predict_proba(X[te])

        # Evaluate
        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        bal_acc = metrics["balanced_accuracy"]
        mcc = metrics["mcc"]

        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
        per_class = []
        for c in range(4):
            if cm[c].sum() > 0:
                per_class.append(cm[c, c] / cm[c].sum())
            else:
                per_class.append(float("nan"))

        logger.info(f"  bal_acc={bal_acc:.4f}, MCC={mcc:.4f}")
        logger.info(f"  Per-class acc: C0={per_class[0]:.3f}, C1={per_class[1]:.3f}, "
                    f"C2={per_class[2]:.3f}, C3={per_class[3]:.3f}")

        fold_results.append({
            "fold": fold["name"],
            "train_cutoff": fold["train_cutoff"],
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "bal_acc": bal_acc,
            "mcc": mcc,
            "joint_acc": metrics["joint_accuracy"],
            "per_class_acc": per_class,
            "test_dist": test_dist,
        })

        # Save predictions
        pred_dir = RESULTS_DIR / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"idx": te, "y_true": y_test, "y_pred": y_pred})
        for c in range(4):
            out[f"prob_{c}"] = y_prob[:, c]
        out.to_csv(pred_dir / f"chiralaldol_v2_xgboost_evans_tscv_{fold['name']}.csv", index=False)

    # Summary
    if fold_results:
        accs = [r["bal_acc"] for r in fold_results]
        mccs = [r["mcc"] for r in fold_results]
        logger.info(f"\n  TSCV Mean bal_acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        logger.info(f"  TSCV Mean MCC:     {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")

        # Save TSCV results
        tscv_path = RESULTS_DIR / "tables" / "tscv_results.json"
        tscv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tscv_path, "w") as f:
            json.dump(fold_results, f, indent=2, default=str)
        logger.info(f"  Saved TSCV results to {tscv_path}")

    # ==================================================================
    # Part 2: Original temporal split (for comparison with 0.783 baseline)
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ORIGINAL TEMPORAL SPLIT (comparison with 0.783)")
    logger.info("=" * 60)

    split_path = SPLIT_DIR / "evans_temporal.json"
    with open(split_path) as f:
        split = json.load(f)

    tr, va, te = split["train"], split["val"], split["test"]
    logger.info(f"  Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")

    model = train_xgb(X[tr], y[tr], X[va], y[va])
    y_pred = model.predict(X[te])
    y_prob = model.predict_proba(X[te])
    y_test = y[te]

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    bal_acc = metrics["balanced_accuracy"]
    mcc = metrics["mcc"]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    per_class = []
    for c in range(4):
        if cm[c].sum() > 0:
            per_class.append(cm[c, c] / cm[c].sum())
        else:
            per_class.append(float("nan"))

    logger.info(f"  V2-XGB (clean data) bal_acc = {bal_acc:.4f}")
    logger.info(f"  V2-XGB (old 1822)   bal_acc = 0.7829")
    logger.info(f"  Delta: {bal_acc - 0.7829:+.4f}")
    logger.info(f"  MCC={mcc:.4f}, joint_acc={metrics['joint_accuracy']:.4f}")
    logger.info(f"  Per-class: C0={per_class[0]:.3f}, C1={per_class[1]:.3f}, "
                f"C2={per_class[2]:.3f}, C3={per_class[3]:.3f}")

    # Save predictions
    pred_dir = RESULTS_DIR / "predictions"
    out = pd.DataFrame({"idx": te, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out[f"prob_{c}"] = y_prob[:, c]
    out.to_csv(pred_dir / "chiralaldol_v2_clean_xgboost_evans_temporal.csv", index=False)

    # ==================================================================
    # Part 3: Scaffold + Grouped Random splits
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("OTHER SPLITS (scaffold + grouped random)")
    logger.info("=" * 60)

    for split_name in ["scaffold", "grouped_random_seed42"]:
        sp_path = SPLIT_DIR / f"evans_{split_name}.json"
        with open(sp_path) as f:
            sp = json.load(f)

        tr, va, te = sp["train"], sp["val"], sp["test"]
        model = train_xgb(X[tr], y[tr], X[va], y[va])
        y_pred = model.predict(X[te])
        y_prob = model.predict_proba(X[te])
        y_test = y[te]

        m = compute_all_metrics(y_test, y_pred, y_prob)
        logger.info(f"  {split_name}: bal_acc={m['balanced_accuracy']:.4f}, "
                    f"MCC={m['mcc']:.4f}, joint={m['joint_accuracy']:.4f}")

        out = pd.DataFrame({"idx": te, "y_true": y_test, "y_pred": y_pred})
        for c in range(4):
            out[f"prob_{c}"] = y_prob[:, c]
        out.to_csv(pred_dir / f"chiralaldol_v2_clean_xgboost_evans_{split_name}.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
