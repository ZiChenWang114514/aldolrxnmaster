#!/usr/bin/env python3
"""A3: Ensemble Stacking — combine ET + XGB + MechAware-BW-XGB via OOF meta-learner.

Literature motivation: [10] Chung 2024 (Sci. Rep.) — GMM-guided composite model
(SVR+RF+LASSO) achieved R²=0.936 on 342 CPA reactions, significantly outperforming
single models.

Strategy:
  Level-0: v4b_full_et (128d) + v4b_full_xgb (128d) + ma_bw_xgb (156d)
  OOF predictions → 3 models × 4 classes = 12d meta-features
  Level-1: LogisticRegression or LightGBM meta-learner

Usage:
    conda run -n aldol-rxn python scripts/run_stacking_v4.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

PROJECT = Path(__file__).resolve().parent.parent
FEAT_DIR = PROJECT / "data" / "features_v4"
SPLITS_DIR = PROJECT / "data" / "splits_v4"
RESULTS_DIR = PROJECT / "results" / "stacking"
PRED_DIR = PROJECT / "results" / "predictions_v4" / "stacking"

TARGET_LABEL = "label_joint"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stacking_v4")


# ═══════════════════════════ DATA LOADING ═══════════════════════════

def load_data():
    """Load V4 features, MechAware BW, labels, all splits."""
    X_df = pd.read_csv(FEAT_DIR / "v4_features.csv")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    feat_names = list(X_df.columns)

    valid_mask = labels[TARGET_LABEL].notna().values
    y_full = labels[TARGET_LABEL].values
    X_v4b = X_df.values.astype(np.float32)
    np.nan_to_num(X_v4b, copy=False)
    y = np.where(valid_mask, y_full, -1).astype(int)

    # MechAware BW
    bw_path = FEAT_DIR / "v4_mechaware_bw.csv"
    X_bw = pd.read_csv(bw_path).values.astype(np.float32) if bw_path.exists() else None
    if X_bw is not None:
        np.nan_to_num(X_bw, copy=False)
        # Append chirality/rgroup/chiralenv/aldpri from V4b
        new_idx = [i for i, c in enumerate(feat_names)
                   if c.startswith(("chiral_", "aux_rg_", "aux_oppolzer", "chiralenv_", "ald_pri_"))]
        X_ma = np.hstack([X_bw, X_v4b[:, new_idx]])
    else:
        X_ma = X_v4b  # fallback

    # Load all splits
    splits = {}
    for f in sorted(SPLITS_DIR.glob("*.json")):
        with open(f) as fp:
            splits[f.stem] = json.load(fp)

    return X_v4b, X_ma, y, valid_mask, feat_names, splits


# ═══════════════════════════ LEVEL-0 TRAINERS ═══════════════════════════

def train_et(X_tr, y_tr):
    m = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42,
                              n_jobs=8, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_xgb_model(X_tr, y_tr):
    sw = compute_sample_weight("balanced", y_tr)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.7, gamma=0.1,
                           reg_lambda=1.0, objective="multi:softprob", num_class=4,
                           tree_method="hist", random_state=42, n_jobs=8, verbosity=0)
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


# ═══════════════════════════ OOF GENERATION ═══════════════════════════

def generate_oof(X, y, valid_mask, tscv_splits, trainer_fn, model_name):
    """Generate out-of-fold predictions for TSCV splits.

    Returns:
        oof_probs: (n_samples, 4) array of OOF probability predictions
        oof_mask: boolean mask of samples that have OOF predictions
    """
    n = len(y)
    oof_probs = np.full((n, 4), np.nan, dtype=np.float64)

    for i, split in enumerate(tscv_splits):
        tr_raw = np.array(split["train"], dtype=int)
        tr = tr_raw[valid_mask[tr_raw]]
        te_raw = np.array(split["test"], dtype=int)
        te = te_raw[valid_mask[te_raw]]

        if len(tr) < 10 or len(te) < 3:
            continue

        model = trainer_fn(X[tr], y[tr])
        probs = model.predict_proba(X[te])

        # Ensure 4-class probabilities
        if probs.shape[1] < 4:
            full_probs = np.zeros((len(te), 4))
            full_probs[:, :probs.shape[1]] = probs
            probs = full_probs

        oof_probs[te] = probs
        logger.info(f"  {model_name} fold {i+1}: {len(te)} OOF predictions")

    oof_mask = ~np.isnan(oof_probs[:, 0])
    logger.info(f"  {model_name} total OOF: {oof_mask.sum()} / {valid_mask.sum()}")
    return oof_probs, oof_mask


# ═══════════════════════════ LEVEL-1 META-LEARNER ═══════════════════════════

def train_level1(oof_meta, y, oof_mask, valid_mask, tscv_splits):
    """Train and evaluate Level-1 meta-learner on TSCV splits.

    The meta-learner operates on OOF probabilities from Level-0 models.
    We evaluate it on the same TSCV splits using a nested approach:
    for each TSCV fold, train Level-1 on OOF from OTHER folds, predict on THIS fold.
    """
    results = []

    for i, split in enumerate(tscv_splits):
        te_raw = np.array(split["test"], dtype=int)
        te = te_raw[valid_mask[te_raw]]
        te_with_oof = te[oof_mask[te]]

        # Training indices: all OOF samples NOT in this fold's test set
        te_set = set(te)
        tr_meta = np.array([j for j in range(len(y))
                            if oof_mask[j] and valid_mask[j] and j not in te_set], dtype=int)

        if len(tr_meta) < 10 or len(te_with_oof) < 3:
            continue

        X_tr_meta = oof_meta[tr_meta]
        y_tr_meta = y[tr_meta]
        X_te_meta = oof_meta[te_with_oof]
        y_te = y[te_with_oof]

        # LogisticRegression meta-learner
        lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=42, C=1.0)
        lr.fit(X_tr_meta, y_tr_meta)
        y_pred_lr = lr.predict(X_te_meta)
        score_lr = balanced_accuracy_score(y_te, y_pred_lr)

        # LightGBM meta-learner
        try:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                   random_state=42, verbose=-1)
            sw = compute_sample_weight("balanced", y_tr_meta)
            lgbm.fit(X_tr_meta, y_tr_meta, sample_weight=sw)
            y_pred_lgbm = lgbm.predict(X_te_meta)
            score_lgbm = balanced_accuracy_score(y_te, y_pred_lgbm)
        except ImportError:
            score_lgbm = 0.0

        results.append({
            "fold": i + 1,
            "n_test": len(te_with_oof),
            "lr_score": score_lr,
            "lgbm_score": score_lgbm,
        })

        logger.info(f"  Fold {i+1}: LR={score_lr:.4f}, LGBM={score_lgbm:.4f}")

    return results


# ═══════════════════════════ FULL STACKING ON ALL SPLITS ═══════════════════════════

def run_stacking_on_split(X_v4b, X_ma, y, valid_mask, split_name, split_data):
    """Run full stacking pipeline on a single split (train→test)."""
    tr_raw = np.array(split_data["train"], dtype=int)
    tr = tr_raw[valid_mask[tr_raw]]
    te_raw = np.array(split_data["test"], dtype=int)
    te = te_raw[valid_mask[te_raw]]

    if len(tr) < 10 or len(te) < 3:
        return None

    # Split train into inner-train and inner-val for OOF
    np.random.seed(42)
    perm = np.random.permutation(len(tr))
    n_val = max(1, len(tr) // 5)  # 20% for OOF
    inner_val_idx = tr[perm[:n_val]]
    inner_tr_idx = tr[perm[n_val:]]

    # Level-0: train on inner-train, predict inner-val (OOF) and test
    l0_models = {
        "et": (X_v4b, train_et),
        "xgb": (X_v4b, train_xgb_model),
        "ma_bw_xgb": (X_ma, train_xgb_model),
    }

    oof_val_list = []
    test_pred_list = []

    for name, (X_feat, trainer) in l0_models.items():
        model = trainer(X_feat[inner_tr_idx], y[inner_tr_idx])
        oof_val_list.append(model.predict_proba(X_feat[inner_val_idx]))
        test_pred_list.append(model.predict_proba(X_feat[te]))

    oof_val_meta = np.hstack(oof_val_list)  # (n_val, 12)
    test_meta = np.hstack(test_pred_list)    # (n_test, 12)

    # Level-1: LogisticRegression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=1.0)
    lr.fit(oof_val_meta, y[inner_val_idx])
    y_pred = lr.predict(test_meta)
    y_prob = lr.predict_proba(test_meta)

    score = balanced_accuracy_score(y[te], y_pred)
    return {"split": split_name, "bal_acc": score, "n_test": len(te),
            "y_true": y[te], "y_pred": y_pred, "y_prob": y_prob, "test_idx": te}


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V4 Ensemble Stacking")
    logger.info("=" * 70)

    X_v4b, X_ma, y, valid_mask, feat_names, splits = load_data()
    logger.info(f"V4b features: {X_v4b.shape[1]}d, MechAware: {X_ma.shape[1]}d")
    logger.info(f"Valid samples: {valid_mask.sum()}, splits: {len(splits)}")

    # === Part 1: TSCV OOF-based analysis ===
    logger.info("\n--- Phase 1: TSCV OOF Generation ---")
    tscv_splits = [splits[k] for k in sorted(splits) if "tscv" in k]

    # Generate OOF predictions for each Level-0 model
    oof_et, mask_et = generate_oof(X_v4b, y, valid_mask, tscv_splits, train_et, "ET")
    oof_xgb, mask_xgb = generate_oof(X_v4b, y, valid_mask, tscv_splits, train_xgb_model, "XGB")
    oof_ma, mask_ma = generate_oof(X_ma, y, valid_mask, tscv_splits, train_xgb_model, "MA-BW-XGB")

    # Combine into meta-features (12d)
    oof_combined = np.full((len(y), 12), np.nan)
    combined_mask = mask_et & mask_xgb & mask_ma
    oof_combined[combined_mask] = np.hstack([
        oof_et[combined_mask],
        oof_xgb[combined_mask],
        oof_ma[combined_mask],
    ])

    logger.info(f"\nCombined OOF mask: {combined_mask.sum()} samples have all 3 predictions")

    # Evaluate Level-1 meta-learner on TSCV
    logger.info("\n--- Phase 2: Level-1 Meta-Learner (TSCV) ---")
    meta_results = train_level1(oof_combined, y, combined_mask, valid_mask, tscv_splits)

    if meta_results:
        lr_scores = [r["lr_score"] for r in meta_results]
        lgbm_scores = [r["lgbm_score"] for r in meta_results]
        logger.info(f"\n  Level-1 LR:   TSCV = {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
        logger.info(f"  Level-1 LGBM: TSCV = {np.mean(lgbm_scores):.4f} ± {np.std(lgbm_scores):.4f}")

    # === Part 2: Full stacking on all splits ===
    logger.info("\n--- Phase 3: Full Stacking on All Splits ---")
    all_results = []
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in sorted(splits.items()):
        result = run_stacking_on_split(X_v4b, X_ma, y, valid_mask, split_name, split_data)
        if result is None:
            continue

        # Save prediction CSV
        out = pd.DataFrame({
            "idx": result["test_idx"],
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
        })
        for c in range(4):
            out[f"prob_{c}"] = result["y_prob"][:, c] if c < result["y_prob"].shape[1] else 0.0
        out.to_csv(PRED_DIR / f"stacking_lr_{split_name}.csv", index=False)

        all_results.append({
            "split": split_name,
            "bal_acc": round(result["bal_acc"], 4),
            "n_test": result["n_test"],
        })
        logger.info(f"  {split_name}: {result['bal_acc']:.4f} (n={result['n_test']})")

    # === Summary ===
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "stacking_results.csv", index=False)

    tscv = [r["bal_acc"] for r in all_results if "tscv" in r["split"]]
    grouped = [r["bal_acc"] for r in all_results if "grouped" in r["split"]]
    scaffold = [r["bal_acc"] for r in all_results if "scaffold" in r["split"]]

    print("\n" + "=" * 70)
    print("STACKING RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Stacking (ET+XGB+MA-BW → LR):")
    if tscv:
        print(f"    TSCV:     {np.mean(tscv):.4f} ± {np.std(tscv):.4f}")
    if scaffold:
        print(f"    Scaffold: {scaffold[0]:.4f}")
    if grouped:
        print(f"    Grouped:  {np.mean(grouped):.4f} ± {np.std(grouped):.4f}")

    print(f"\n  Baselines (default hyperparams):")
    print(f"    v4b_full_et:  TSCV=0.624, Scaffold=0.613, Grouped=0.738")
    print(f"    ma_bw_xgb:    TSCV=0.604, Scaffold=0.607, Grouped=0.752")
    print(f"    v4b_full_xgb: TSCV=0.602, Scaffold=0.589, Grouped=0.747")

    if tscv:
        delta = np.mean(tscv) - 0.624
        print(f"\n  Δ TSCV vs ET champion: {delta:+.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}/stacking_results.csv")
    print(f"  Predictions: {PRED_DIR}/")


if __name__ == "__main__":
    main()
