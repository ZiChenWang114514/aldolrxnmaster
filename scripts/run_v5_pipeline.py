#!/usr/bin/env python
"""Phase V5: Cross-term interaction features + multi-model ensemble.

V5 = V2 (75d) + 12d cross-term/Z-E/derived features = 87d total.
Cross-terms capture enolate×aldehyde steric interaction (r=0.25-0.34 with label).

Models trained:
  - V5-XGB: XGBoost with extended 6-config grid
  - V5-LGBM: LightGBM
  - V5-ET: ExtraTrees
  - V5-Stack: 5-fold OOF stacking (V5-XGB + V5-LGBM + V5-ET + DRFP+Cond-XGB)
  - V5s-XGB: Feature-selected XGBoost (fallback if V5-XGB < V2)

Usage:
    conda run -n aldol-rxn python scripts/run_v5_pipeline.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ── Training functions ────────────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, X_val, y_val):
    """XGBoost with extended 6-config grid search."""
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15,
         "subsample": 0.9, "colsample_bytree": 0.8},
        {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.08,
         "subsample": 0.85, "colsample_bytree": 0.65},
        {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.03,
         "subsample": 0.75, "colsample_bytree": 0.5},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.12,
         "subsample": 0.85, "colsample_bytree": 0.75},
    ]
    best_m, best_acc = None, 0.0
    for cfg in configs:
        cfg.update({
            "objective": "multi:softprob", "num_class": 4,
            "tree_method": "hist", "random_state": 42,
            "n_jobs": 4, "verbosity": 0,
            "gamma": 0.1, "reg_lambda": 1.0,
        })
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def train_lgbm(X_tr, y_tr, X_val, y_val):
    """LightGBM with 3-config grid search."""
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 10},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6, "min_child_samples": 15},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15,
         "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 5},
    ]
    best_m, best_acc = None, 0.0
    for cfg in configs:
        cfg.update({
            "objective": "multiclass", "num_class": 4,
            "class_weight": "balanced", "random_state": 42,
            "n_jobs": 4, "verbose": -1,
        })
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def train_et(X_tr, y_tr, X_val, y_val):
    """ExtraTrees with 3-config grid search."""
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 3},
        {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2},
    ]
    best_m, best_acc = None, 0.0
    for cfg in configs:
        cfg.update({
            "class_weight": "balanced", "random_state": 42, "n_jobs": 4,
        })
        m = ExtraTreesClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def eval_save(name, y_test, y_pred, y_prob, test_idx, split_name):
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    logger.info(f"  {name} [{split_name}]: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}")
    out = pd.DataFrame({"idx": test_idx, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out[f"prob_{c}"] = y_prob[:, c]
    out.to_csv(PRED_DIR / f"{name}_{split_name}.csv", index=False)
    return metrics


def load_split(split_name):
    with open(SPLIT_DIR / f"{split_name}.json") as f:
        sp = json.load(f)
    return np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── Build V5 features ─────────────────────────────────────────────────
    logger.info("Building V5 features (V2 75d + 12d cross-terms)...")
    from chiralaldol.feature_builder import build_chiralaldol_v5_features
    X_v5, names_v5 = build_chiralaldol_v5_features(PROJECT)
    logger.info(f"V5 feature matrix: {X_v5.shape}, {len(names_v5)} names")

    # ── Load labels ───────────────────────────────────────────────────────
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    # ── Load DRFP+Cond for stacking base model B ─────────────────────────
    drfp_data = np.load(FEAT_DIR / "drfp_fps.npz")
    X_drfp_raw = drfp_data[list(drfp_data.keys())[0]].astype(np.float32)
    svd = TruncatedSVD(n_components=128, random_state=42)
    X_drfp = svd.fit_transform(X_drfp_raw).astype(np.float32)
    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    X_drfp_cond = np.hstack([X_drfp, cond_df.values.astype(np.float32)])
    logger.info(f"DRFP+Cond: {X_drfp_cond.shape}")

    splits = ["evans_temporal", "evans_scaffold", "evans_grouped_random_seed42"]
    v5_xgb_temporal = None

    for split_name in splits:
        logger.info(f"\n{'='*60}\n  V5 — {split_name}\n{'='*60}")
        tr, va, te = load_split(split_name)

        # ── V5-XGB ────────────────────────────────────────────────────
        logger.info("--- V5-XGB ---")
        m_xgb = train_xgb(X_v5[tr], y[tr], X_v5[va], y[va])
        met = eval_save("chiralaldol_v5_xgboost",
                        y[te], m_xgb.predict(X_v5[te]),
                        m_xgb.predict_proba(X_v5[te]), te, split_name)
        if split_name == "evans_temporal":
            v5_xgb_temporal = met["balanced_accuracy"]

        # ── V5-LGBM ──────────────────────────────────────────────────
        logger.info("--- V5-LGBM ---")
        m_lgbm = train_lgbm(X_v5[tr], y[tr], X_v5[va], y[va])
        eval_save("chiralaldol_v5_lgbm",
                  y[te], m_lgbm.predict(X_v5[te]),
                  m_lgbm.predict_proba(X_v5[te]), te, split_name)

        # ── V5-ET ─────────────────────────────────────────────────────
        logger.info("--- V5-ET ---")
        m_et = train_et(X_v5[tr], y[tr], X_v5[va], y[va])
        eval_save("chiralaldol_v5_et",
                  y[te], m_et.predict(X_v5[te]),
                  m_et.predict_proba(X_v5[te]), te, split_name)

        # ── V5-Stack (5-fold OOF) ─────────────────────────────────────
        logger.info("--- V5-Stack ---")
        train_all = np.concatenate([tr, va])
        n_base = 4  # V5-XGB, V5-LGBM, V5-ET, DRFP+Cond-XGB
        oof_meta = np.zeros((len(train_all), n_base * 4), dtype=np.float32)
        test_meta = np.zeros((len(te), n_base * 4), dtype=np.float32)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_i, (ktr, kva) in enumerate(kfold.split(train_all, y[train_all])):
            ktr_idx = train_all[ktr]
            kva_idx = train_all[kva]

            # Base A: V5-XGB
            m_a = train_xgb(X_v5[ktr_idx], y[ktr_idx], X_v5[kva_idx], y[kva_idx])
            oof_meta[kva, 0:4] = m_a.predict_proba(X_v5[kva_idx])

            # Base B: V5-LGBM
            m_b = train_lgbm(X_v5[ktr_idx], y[ktr_idx], X_v5[kva_idx], y[kva_idx])
            oof_meta[kva, 4:8] = m_b.predict_proba(X_v5[kva_idx])

            # Base C: V5-ET
            m_c = train_et(X_v5[ktr_idx], y[ktr_idx], X_v5[kva_idx], y[kva_idx])
            oof_meta[kva, 8:12] = m_c.predict_proba(X_v5[kva_idx])

            # Base D: DRFP+Cond-XGB
            m_d = train_xgb(X_drfp_cond[ktr_idx], y[ktr_idx],
                            X_drfp_cond[kva_idx], y[kva_idx])
            oof_meta[kva, 12:16] = m_d.predict_proba(X_drfp_cond[kva_idx])

        # Full-train base models for test prediction
        m_a_full = train_xgb(X_v5[train_all], y[train_all], X_v5[va], y[va])
        test_meta[:, 0:4] = m_a_full.predict_proba(X_v5[te])
        m_b_full = train_lgbm(X_v5[train_all], y[train_all], X_v5[va], y[va])
        test_meta[:, 4:8] = m_b_full.predict_proba(X_v5[te])
        m_c_full = train_et(X_v5[train_all], y[train_all], X_v5[va], y[va])
        test_meta[:, 8:12] = m_c_full.predict_proba(X_v5[te])
        m_d_full = train_xgb(X_drfp_cond[train_all], y[train_all],
                             X_drfp_cond[va], y[va])
        test_meta[:, 12:16] = m_d_full.predict_proba(X_drfp_cond[te])

        # Meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                  random_state=42)
        meta.fit(oof_meta, y[train_all])
        stack_pred = meta.predict(test_meta)
        stack_prob = meta.predict_proba(test_meta)
        eval_save("chiralaldol_v5_stacking",
                  y[te], stack_pred, stack_prob, te, split_name)

    # ── Feature selection fallback ────────────────────────────────────────
    if v5_xgb_temporal is not None and v5_xgb_temporal < 0.783:
        logger.info(f"\nV5-XGB temporal ({v5_xgb_temporal:.4f}) < V2 (0.783). "
                    "Running feature selection fallback (V5s)...")
        tr, va, te = load_split("evans_temporal")
        m_full = train_xgb(X_v5[tr], y[tr], X_v5[va], y[va])
        importances = m_full.feature_importances_
        keep_mask = importances >= 0.005
        n_keep = keep_mask.sum()
        logger.info(f"  Keeping {n_keep}/{len(importances)} features (importance >= 0.005)")

        X_sel = X_v5[:, keep_mask]
        for split_name in splits:
            tr, va, te = load_split(split_name)
            m_sel = train_xgb(X_sel[tr], y[tr], X_sel[va], y[va])
            eval_save("chiralaldol_v5s_xgboost",
                      y[te], m_sel.predict(X_sel[te]),
                      m_sel.predict_proba(X_sel[te]), te, split_name)

    # ── Rebuild comparison ────────────────────────────────────────────────
    logger.info("\nRebuilding comparison tables...")
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "rebuild_comparison.py")],
        check=True,
    )
    logger.info("Done! Check results/tables/ for updated comparison.")


if __name__ == "__main__":
    main()
