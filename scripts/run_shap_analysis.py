#!/usr/bin/env python3
"""D1: SHAP feature importance analysis for the champion model.

Literature motivation: [16] Sharpless AD + [23] C-H DNN — all published papers
report SHAP; this is a reviewer-required analysis.

Usage:
    conda run -n aldol-rxn python scripts/run_shap_analysis.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import OPTUNA_DIR, RESULTS_DIR, SPLITS_DIR
from chiralaldol.data_io import prepare_Xy

OUT_DIR = RESULTS_DIR / "shap"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shap_analysis")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SHAP Feature Importance Analysis")
    logger.info("=" * 60)

    # Load data
    X, y, valid_mask, feat_names = prepare_Xy()

    logger.info(f"Features: {X.shape[1]}d, samples: {valid_mask.sum()}")

    # Load Optuna best params for champion (ma_bw_xgb uses MechAware features,
    # but for SHAP interpretability we use v4b features directly with xgb_optuna params)
    with open(OPTUNA_DIR / "xgb_optuna_best.json") as f:
        xgb_params = json.load(f)["best_params"]

    xgb_params.update({
        "objective": "multi:softprob", "num_class": 4,
        "tree_method": "hist", "random_state": 42, "n_jobs": 8, "verbosity": 0,
    })

    # Load TSCV splits
    tscv_splits = []
    for i in range(1, 5):
        with open(SPLITS_DIR / f"tscv_fold{i}.json") as f:
            tscv_splits.append(json.load(f))

    # Collect SHAP values across all TSCV folds
    all_shap = []
    all_X_test = []
    all_y_test = []
    all_y_pred = []

    for i, split in enumerate(tscv_splits):
        tr_raw = np.array(split["train"], dtype=int)
        tr = tr_raw[valid_mask[tr_raw]]
        te_raw = np.array(split["test"], dtype=int)
        te = te_raw[valid_mask[te_raw]]

        va = tr[-max(1, len(tr) // 10):]
        tr_sub = tr[:-len(va)]

        sw = compute_sample_weight("balanced", y[tr_sub])
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X[tr_sub], y[tr_sub], sample_weight=sw)

        y_pred = model.predict(X[te])
        acc = balanced_accuracy_score(y[te], y_pred)
        logger.info(f"  Fold {i+1}: bal_acc={acc:.4f}, n_test={len(te)}")

        # TreeSHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[te])

        # Normalize format: XGBoost may return (n_test, n_features, n_classes) 3D array
        # or list of n_classes arrays each (n_test, n_features)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # shape (n_test, n_features, n_classes) → list of 4 arrays
            sv_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
        elif isinstance(shap_values, list):
            sv_list = shap_values
        else:
            # Single array fallback
            sv_list = [shap_values]

        all_shap.append(sv_list)
        all_X_test.append(X[te])
        all_y_test.extend(y[te].tolist())
        all_y_pred.extend(y_pred.tolist())

    # Concatenate across folds
    n_classes = len(all_shap[0])
    merged_shap = []
    for c in range(n_classes):
        merged_shap.append(np.vstack([sv[c] for sv in all_shap]))
    X_test_all = np.vstack(all_X_test)

    logger.info(f"\nTotal test samples: {len(all_y_test)}")
    logger.info(f"SHAP shape per class: {merged_shap[0].shape}")

    # === Global importance: mean(|SHAP|) across all classes ===
    global_importance = np.zeros(X.shape[1])
    for c in range(n_classes):
        global_importance += np.abs(merged_shap[c]).mean(axis=0)
    global_importance /= n_classes

    importance_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": global_importance,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(OUT_DIR / "shap_importance_global.csv", index=False)

    logger.info("\n=== Global Top-20 Features ===")
    for _, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']:45s}  {row['mean_abs_shap']:.6f}")

    # === Per-class importance ===
    class_names = ["RR (0)", "RS (1)", "SR (2)", "SS (3)"]
    per_class_rows = []
    for c in range(n_classes):
        class_imp = np.abs(merged_shap[c]).mean(axis=0)
        for j, fname in enumerate(feat_names):
            per_class_rows.append({
                "class": class_names[c],
                "class_id": c,
                "feature": fname,
                "mean_abs_shap": class_imp[j],
            })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(OUT_DIR / "shap_per_class.csv", index=False)

    # Print per-class top-10
    for c in range(n_classes):
        sub = per_class_df[per_class_df["class_id"] == c].sort_values("mean_abs_shap", ascending=False)
        logger.info(f"\n=== {class_names[c]} Top-10 ===")
        for _, row in sub.head(10).iterrows():
            logger.info(f"  {row['feature']:45s}  {row['mean_abs_shap']:.6f}")

    # === Feature group summary ===
    groups = {
        "steric": [f for f in feat_names if any(f.startswith(p) for p in
                   ("B1_", "B5_", "L_", "Vbur_", "sin_tau", "cos_tau", "n_conformers", "n_clusters", "ald_L", "ald_B1", "ald_B5", "ald_Vbur", "ald_n_"))],
        "conditions": [f for f in feat_names if f.startswith("feat_")],
        "auxiliary": [f for f in feat_names if f.startswith("aux_") and not f.startswith("aux_rg_") and f != "aux_oppolzer"],
        "chirality": [f for f in feat_names if f.startswith("chiral_") and not f.startswith("chiralenv_")],
        "rgroup": [f for f in feat_names if f.startswith("aux_rg_") or f == "aux_oppolzer"],
        "chiralenv": [f for f in feat_names if f.startswith("chiralenv_")],
        "aldpri": [f for f in feat_names if f.startswith("ald_pri_")],
        "delta_chiral": [f for f in feat_names if f.startswith("delta_chiral_")],
        "chiral_det": [f for f in feat_names if f.startswith("chiral_det_")],
    }

    logger.info("\n=== Feature Group Importance ===")
    group_summary = []
    for gname, gcols in groups.items():
        if not gcols:
            continue
        idx = [feat_names.index(c) for c in gcols if c in feat_names]
        g_imp = global_importance[idx].sum()
        group_summary.append({"group": gname, "n_features": len(idx), "total_importance": round(g_imp, 6)})
        logger.info(f"  {gname:20s}: {len(idx):3d}d, importance={g_imp:.6f}")

    pd.DataFrame(group_summary).to_csv(OUT_DIR / "shap_group_summary.csv", index=False)

    # Save raw SHAP values for visualization
    np.savez_compressed(OUT_DIR / "shap_values_raw.npz",
                        shap_class0=merged_shap[0], shap_class1=merged_shap[1],
                        shap_class2=merged_shap[2], shap_class3=merged_shap[3],
                        X_test=X_test_all, y_test=np.array(all_y_test),
                        y_pred=np.array(all_y_pred), feature_names=np.array(feat_names))

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")
    logger.info(f"Results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
