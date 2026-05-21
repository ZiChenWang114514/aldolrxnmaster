#!/usr/bin/env python3
"""MechAware Steps M4-M5: Feature assembly + XGBoost TSCV training.

Assembles three MechAware feature versions:
  - MechAware-Full (161d): Ketone(24) + Z(24) + E(24) + BW(24) + w(2) + Ald(10) + Cond(44) + Aux(9)
  - MechAware-ZE  (137d): Z(24) + E(24) + BW(24) + w(2) + Ald(10) + Cond(44) + Aux(9)
  - MechAware-BW   (89d): BW(24) + w(2) + Ald(10) + Cond(44) + Aux(9)

Usage:
    conda run -n aldol-rxn python scripts/run_mechaware_model.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci
from chiralaldol.ze_enolate_generator import get_ze_weights
from chiralaldol.rebuild.constants import BASE_CATEGORIES, ACTIVATOR_CATEGORIES

MECHAWARE_DIR = PROJECT_DIR / "data" / "v3" / "mechaware"
V3_DIR = PROJECT_DIR / "data" / "v3"
RESULTS_DIR = PROJECT_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(MECHAWARE_DIR / "mechaware_model.log", mode="w", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("mechaware")


def train_xgb(X_tr, y_tr, X_val, y_val):
    """3-config grid search XGBoost (same as ChiralAldol pipeline)."""
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
            "n_jobs": 1, "verbosity": 0,
            "gamma": 0.1, "reg_lambda": 1.0,
        })
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def _infer_base_and_activator(row: pd.Series) -> tuple[str, str]:
    """Infer dominant base and activator from condition one-hot columns."""
    # Base: find the one-hot that is 1
    base = "no_base"
    for cat in BASE_CATEGORIES:
        col = f"base_{cat}"
        if col in row.index and row[col] > 0.5:
            base = cat
            break

    # Activator: find the one-hot that is 1
    activator = ""
    for cat in ACTIVATOR_CATEGORIES:
        col = f"act_{cat}"
        if col in row.index and row[col] > 0.5:
            activator = cat
            break

    return base, activator


def main():
    logger.info("=" * 60)
    logger.info("MechAware Model — Feature Assembly + Training")
    logger.info("=" * 60)
    t0 = time.time()

    # ── Load MechAware steric features ──
    ketone_path = MECHAWARE_DIR / "ketone_steric.csv"
    z_path = MECHAWARE_DIR / "z_enolate_steric.csv"
    e_path = MECHAWARE_DIR / "e_enolate_steric.csv"

    for p in [ketone_path, z_path, e_path]:
        if not p.exists():
            logger.error(f"Missing: {p}. Run scripts/run_ze_conformers.py first!")
            sys.exit(1)

    ket_df = pd.read_csv(ketone_path)
    z_df = pd.read_csv(z_path)
    e_df = pd.read_csv(e_path)
    logger.info(f"Loaded steric: ketone={ket_df.shape}, Z={z_df.shape}, E={e_df.shape}")

    # ── Load V3 features (conditions, auxiliary, labels, aldehyde) ──
    cond_path = V3_DIR / "features" / "condition_features.csv"
    labels_path = V3_DIR / "features" / "labels.csv"
    ald_path = V3_DIR / "features" / "aldehyde_steric_features.csv"

    # V3 conditions
    cond_df = pd.read_csv(cond_path) if cond_path.exists() else None
    labels_df = pd.read_csv(labels_path) if labels_path.exists() else None

    # V3 aldehyde steric (if available from V3 run with conformers)
    if ald_path.exists():
        ald_df = pd.read_csv(ald_path)
        logger.info(f"Loaded V3 aldehyde steric: {ald_df.shape}")
    else:
        ald_df = None
        logger.warning("V3 aldehyde steric not found — will be excluded")

    # V3 interim for auxiliary
    aux_interim = V3_DIR / "interim" / "06_aux_chirality.csv"
    if aux_interim.exists():
        aux_full = pd.read_csv(aux_interim, usecols=["original_index", "aux_C4_cip", "aux_rgroup_type", "aux_mw"])
    else:
        aux_full = None

    # ── Merge on original_index ──
    merged = ket_df[["original_index"]].copy()

    # Steric features (drop original_index from feature dfs before concat)
    ket_feat_cols = [c for c in ket_df.columns if c != "original_index"]
    z_feat_cols = [c for c in z_df.columns if c != "original_index"]
    e_feat_cols = [c for c in e_df.columns if c != "original_index"]

    for col in ket_feat_cols:
        merged[col] = ket_df[col].values
    for col in z_feat_cols:
        merged[col] = z_df[col].values
    for col in e_feat_cols:
        merged[col] = e_df[col].values

    # Merge conditions
    if cond_df is not None:
        merged = merged.merge(cond_df, on="original_index", how="inner")
    cond_cols = [c for c in merged.columns if c.startswith("base_") or c.startswith("metal_")
                 or c.startswith("solvent_") or c.startswith("act_") or c.startswith("has_")]

    # Merge labels
    if labels_df is not None:
        merged = merged.merge(labels_df, on="original_index", how="inner")

    # Merge aldehyde steric (may not have original_index column)
    if ald_df is not None:
        ald_feat_cols = [c for c in ald_df.columns if c != "original_index"]
        if "original_index" in ald_df.columns:
            merged = merged.merge(ald_df, on="original_index", how="inner")
        else:
            # V3 step11 saves without original_index — skip merge, features will come from V3 features
            logger.warning("  Aldehyde steric has no original_index — skipping merge (using V3 features if available)")
            ald_feat_cols = []
    else:
        ald_feat_cols = []

    # Merge auxiliary
    if aux_full is not None:
        merged = merged.merge(aux_full, on="original_index", how="inner")
        merged["aux_config_R"] = merged["aux_C4_cip"].map({"R": 1.0, "S": 0.0}).fillna(-1.0)
        aux_cols = ["aux_config_R", "aux_mw"]
    else:
        aux_cols = []

    n_rows = len(merged)
    logger.info(f"Merged dataset: {n_rows} rows")

    if "label_joint" not in merged.columns:
        logger.error("label_joint not found — labels merge failed")
        sys.exit(1)

    # ── Compute base-weighted (BW) features ──
    logger.info("Computing base-weighted enolate features...")
    bw_cols = []
    w_z_list = []
    w_e_list = []

    for _, row in merged.iterrows():
        base, activator = _infer_base_and_activator(row)
        w_z, w_e = get_ze_weights(base, activator)
        w_z_list.append(w_z)
        w_e_list.append(w_e)

    merged["w_Z"] = w_z_list
    merged["w_E"] = w_e_list

    # BW = w_Z * Z_desc + w_E * E_desc
    from chiralaldol.steric_descriptors import STERIC_DESC_NAMES
    for name in STERIC_DESC_NAMES:
        z_col = f"z_{name}"
        e_col = f"e_{name}"
        bw_col = f"bw_{name}"
        if z_col in merged.columns and e_col in merged.columns:
            merged[bw_col] = merged["w_Z"] * merged[z_col] + merged["w_E"] * merged[e_col]
            bw_cols.append(bw_col)

    logger.info(f"  BW features: {len(bw_cols)}d")

    # ── Assemble feature versions ──
    weight_cols = ["w_Z", "w_E"]

    feat_versions = {
        "MechAware-Full": ket_feat_cols + z_feat_cols + e_feat_cols + bw_cols + weight_cols + ald_feat_cols + cond_cols + aux_cols,
        "MechAware-ZE": z_feat_cols + e_feat_cols + bw_cols + weight_cols + ald_feat_cols + cond_cols + aux_cols,
        "MechAware-BW": bw_cols + weight_cols + ald_feat_cols + cond_cols + aux_cols,
    }

    for name, cols in feat_versions.items():
        actual = [c for c in cols if c in merged.columns]
        feat_versions[name] = actual
        logger.info(f"  {name}: {len(actual)}d")

    # ── Handle NaN ──
    all_feat_cols = set()
    for cols in feat_versions.values():
        all_feat_cols.update(cols)
    all_feat_cols = sorted(all_feat_cols)

    nan_count = merged[all_feat_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"  {nan_count} NaN values — filling with 0")
        merged[all_feat_cols] = merged[all_feat_cols].fillna(0.0)

    # ── Load V3 splits ──
    splits_dir = V3_DIR / "splits"
    tscv_splits = {}
    for fold_num in range(1, 5):
        sp = splits_dir / f"evans_tscv_fold{fold_num}.json"
        if sp.exists():
            with open(sp) as f:
                tscv_splits[f"fold{fold_num}"] = json.load(f)

    if not tscv_splits:
        logger.error("No TSCV splits found in V3!")
        sys.exit(1)

    logger.info(f"Loaded {len(tscv_splits)} TSCV folds")

    # ── Remap split indices ──
    # V3 splits use position indices into the V3 Evans subset.
    # Load V3 interim to reconstruct the exact row order.
    v3_interim_path = V3_DIR / "interim" / "09_conditions.csv"
    if not v3_interim_path.exists():
        logger.error(f"V3 interim not found: {v3_interim_path}")
        sys.exit(1)
    v3_interim = pd.read_csv(v3_interim_path, usecols=["original_index", "Reaction_Class"])
    v3_evans_df = v3_interim[v3_interim["Reaction_Class"] == "EvansAux"].reset_index(drop=True)
    v3_oi_list = v3_evans_df["original_index"].tolist()
    logger.info(f"V3 Evans row order: {len(v3_oi_list)} rows")

    # Build mapping: V3_evans_position → merged_position
    merged_oi = merged["original_index"].tolist()
    merged_oi_to_idx = {oi: i for i, oi in enumerate(merged_oi)}

    # ── Train and evaluate ──
    logger.info("=" * 60)
    logger.info("Step M5: TSCV Training & Evaluation")
    logger.info("=" * 60)

    y = merged["label_joint"].values
    results_all = {}
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for version_name, feat_cols in feat_versions.items():
        logger.info(f"\n--- {version_name} ({len(feat_cols)}d) ---")
        X = merged[feat_cols].values.astype(np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        fold_accs = []
        for fold_name, split in tscv_splits.items():
            # Remap V3 indices → merged indices
            train_v3 = split["train"]
            val_v3 = split.get("val", [])
            test_v3 = split["test"]

            def remap(v3_indices):
                result = []
                for v3_idx in v3_indices:
                    if v3_idx < len(v3_oi_list):
                        oi = v3_oi_list[v3_idx]
                        if oi in merged_oi_to_idx:
                            result.append(merged_oi_to_idx[oi])
                return np.array(result, dtype=int)

            tr_idx = remap(train_v3)
            va_idx = remap(val_v3) if val_v3 else tr_idx[-len(tr_idx)//10:]
            te_idx = remap(test_v3)

            if len(tr_idx) == 0 or len(te_idx) == 0:
                logger.warning(f"  {fold_name}: empty split after remap, skipping")
                continue

            # Ensure val is not empty
            if len(va_idx) == 0:
                va_idx = tr_idx[-max(1, len(tr_idx)//10):]
                tr_idx = tr_idx[:-len(va_idx)]

            model = train_xgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])
            y_pred = model.predict(X[te_idx])
            y_prob = model.predict_proba(X[te_idx])
            bal_acc = balanced_accuracy_score(y[te_idx], y_pred)
            fold_accs.append(bal_acc)

            # Save predictions
            out = pd.DataFrame({
                "idx": te_idx, "y_true": y[te_idx], "y_pred": y_pred,
            })
            for c in range(4):
                out[f"prob_{c}"] = y_prob[:, c]
            csv_name = f"mechaware_{version_name.lower().replace('-','_')}_evans_tscv_{fold_name}.csv"
            out.to_csv(pred_dir / csv_name, index=False)

            logger.info(f"  {fold_name}: bal_acc={bal_acc:.4f} (train={len(tr_idx)}, test={len(te_idx)})")

        if fold_accs:
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            results_all[version_name] = {
                "tscv_mean": float(mean_acc),
                "tscv_std": float(std_acc),
                "n_features": len(feat_cols),
                "folds": {f"fold{i+1}": float(a) for i, a in enumerate(fold_accs)},
            }
            logger.info(f"  TSCV mean: {mean_acc:.4f} ± {std_acc:.4f}")

    # ── Summary comparison ──
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  V2 baseline (75d):     TSCV = 0.682 ± 0.044")
    for name, res in results_all.items():
        logger.info(f"  {name} ({res['n_features']}d): TSCV = {res['tscv_mean']:.4f} ± {res['tscv_std']:.4f}")

    # Save results
    results_path = MECHAWARE_DIR / "mechaware_results.json"
    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    elapsed = time.time() - t0
    logger.info(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
