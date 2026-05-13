#!/usr/bin/env python
"""SHAP feature importance + error analysis for ChiralAldol.

Outputs:
  - SHAP summary (text): top-20 feature importance ranking
  - Error analysis: hard cases, complementarity between ChiralAldol and DRFP
"""

import json
import logging
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(H:%M:%S)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
CHIRALALDOL_DIR = PROJECT / "data" / "processed" / "chiralaldol"
OUT_DIR = Path(__file__).resolve().parent
PRED_DIR = PROJECT / "results" / "predictions"


def load_data():
    """Load all features and labels."""
    from sklearn.decomposition import TruncatedSVD

    steric_df = pd.read_csv(CHIRALALDOL_DIR / "steric_features.csv")
    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    aux_df = pd.read_csv(FEAT_DIR / "auxchiral_features.csv")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    # ChiralAldol features (65d)
    X_full = np.hstack([steric_df.values, cond_df.values, aux_df.values]).astype(np.float32)
    feature_names = list(steric_df.columns) + list(cond_df.columns) + list(aux_df.columns)
    np.nan_to_num(X_full, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Temporal split
    with open(SPLIT_DIR / "evans_temporal.json") as f:
        sp = json.load(f)
    tr, va, te = np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])

    return X_full, y, feature_names, tr, va, te, labels


def train_model(X_tr, y_tr, X_val, y_val):
    """Train XGBoost (same config as pipeline)."""
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
        cfg.update({"objective": "multi:softprob", "num_class": 4,
                    "tree_method": "hist", "random_state": 42,
                    "n_jobs": 4, "verbosity": 0,
                    "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def run_shap_analysis(model, X_test, feature_names, y_test):
    """SHAP TreeExplainer analysis."""
    logger.info("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values is list of 4 arrays (one per class), each (n_test, n_features)
    # Compute mean absolute SHAP per feature across all classes
    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_abs = np.abs(shap_values).mean(axis=2) if shap_values.ndim == 3 else np.abs(shap_values)

    mean_importance = shap_abs.mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_importance,
    }).sort_values("mean_abs_shap", ascending=False)

    logger.info("\n=== SHAP Feature Importance (Top-20) ===")
    for i, row in importance_df.head(20).iterrows():
        rank = list(importance_df.index).index(i) + 1
        logger.info(f"  {rank:2d}. {row['feature']:30s}  SHAP={row['mean_abs_shap']:.4f}")

    # Check Vbur_diff ranking
    vbur_diff_rank = list(importance_df["feature"]).index("Vbur_diff_mean") + 1
    logger.info(f"\n%Vbur_diff_mean rank: #{vbur_diff_rank}")

    importance_df.to_csv(OUT_DIR / "shap_importance.csv", index=False)
    logger.info(f"Saved to {OUT_DIR / 'shap_importance.csv'}")

    return importance_df, shap_values


def run_error_analysis(y_test, test_idx, labels_df):
    """Analyze prediction errors across models."""
    logger.info("\n=== Error Analysis (Temporal Split, 155 test samples) ===")

    # Load predictions from key models
    models_to_compare = {
        "ChiralAldol-XGB": "chiralaldol_xgboost",
        "ChiralAldol-Stack": "chiralaldol_stacking",
        "ChiralAldol-WtVote": "chiralaldol_weighted_vote",
        "DRFP+Cond": "drfp+cond+xgboost",
        "ProtoNet": "protonet",
        "Chemprop+Cond": "chemprop_cond",
        "DistilBERT": "distilbert_rxn",
    }

    preds = {}
    for display_name, file_prefix in models_to_compare.items():
        csv_path = PRED_DIR / f"{file_prefix}_evans_temporal.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            preds[display_name] = df["y_pred"].values
        else:
            logger.warning(f"  Missing: {csv_path}")

    if not preds:
        logger.error("No prediction files found!")
        return

    n_test = len(y_test)

    # Correctness matrix
    correct = {name: (pred == y_test).astype(int) for name, pred in preds.items()}

    # Per-model accuracy
    logger.info("\nPer-model accuracy on temporal test:")
    for name in preds:
        acc = correct[name].sum() / n_test
        logger.info(f"  {name:25s}: {acc:.3f} ({correct[name].sum()}/{n_test})")

    # Complementarity: ChiralAldol-Stack vs DRFP+Cond
    if "ChiralAldol-Stack" in correct and "DRFP+Cond" in correct:
        c_stack = correct["ChiralAldol-Stack"]
        c_drfp = correct["DRFP+Cond"]

        both_right = ((c_stack == 1) & (c_drfp == 1)).sum()
        stack_only = ((c_stack == 1) & (c_drfp == 0)).sum()
        drfp_only = ((c_stack == 0) & (c_drfp == 1)).sum()
        both_wrong = ((c_stack == 0) & (c_drfp == 0)).sum()

        logger.info(f"\n--- Complementarity: ChiralAldol-Stack vs DRFP+Cond ---")
        logger.info(f"  Both correct:       {both_right:3d} ({100*both_right/n_test:.1f}%)")
        logger.info(f"  Stack ONLY correct: {stack_only:3d} ({100*stack_only/n_test:.1f}%)")
        logger.info(f"  DRFP ONLY correct:  {drfp_only:3d} ({100*drfp_only/n_test:.1f}%)")
        logger.info(f"  Both wrong:         {both_wrong:3d} ({100*both_wrong/n_test:.1f}%)")

    # Hard cases: wrong by ALL models
    all_correct = np.ones(n_test, dtype=int)
    for name in correct:
        all_correct &= correct[name]
    all_wrong = np.ones(n_test, dtype=int)
    for name in correct:
        all_wrong &= (1 - correct[name])

    n_hard = all_wrong.sum()
    logger.info(f"\n--- Hard Cases (wrong by ALL {len(preds)} models) ---")
    logger.info(f"  {n_hard}/{n_test} ({100*n_hard/n_test:.1f}%) samples are universally mispredicted")

    if n_hard > 0:
        hard_idx = test_idx[all_wrong == 1]
        hard_df = labels_df.iloc[hard_idx][["label_joint", "label_Ca", "label_Cb", "label_SA"]].copy()
        hard_df["global_idx"] = hard_idx
        logger.info(f"  Class distribution of hard cases:")
        for c in range(4):
            n_c = (hard_df["label_joint"] == c).sum()
            logger.info(f"    C{c}: {n_c} ({100*n_c/n_hard:.1f}%)")

        hard_df.to_csv(OUT_DIR / "hard_cases.csv", index=False)
        logger.info(f"  Saved {n_hard} hard cases to {OUT_DIR / 'hard_cases.csv'}")

    # Per-class error rates for ChiralAldol-Stack
    if "ChiralAldol-Stack" in preds:
        logger.info(f"\n--- Per-class accuracy: ChiralAldol-Stack ---")
        for c in range(4):
            mask = y_test == c
            if mask.sum() > 0:
                acc = correct["ChiralAldol-Stack"][mask].mean()
                logger.info(f"  C{c}: {acc:.3f} ({correct['ChiralAldol-Stack'][mask].sum()}/{mask.sum()})")


if __name__ == "__main__":
    X_full, y, feature_names, tr, va, te, labels_df = load_data()

    logger.info(f"Features: {len(feature_names)}, Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")

    # Train ChiralAldol-XGB for SHAP
    logger.info("\nTraining ChiralAldol-XGB for SHAP analysis...")
    model = train_model(X_full[tr], y[tr], X_full[va], y[va])
    y_pred = model.predict(X_full[te])
    bal_acc = balanced_accuracy_score(y[te], y_pred)
    logger.info(f"ChiralAldol-XGB bal_acc on test: {bal_acc:.4f}")

    # SHAP
    importance_df, shap_values = run_shap_analysis(model, X_full[te], feature_names, y[te])

    # Error analysis
    run_error_analysis(y[te], te, labels_df)

    logger.info("\nDone!")
