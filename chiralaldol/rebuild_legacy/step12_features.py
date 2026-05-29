"""Step 12: Feature integration + split-aware normalization utilities.

Combines all feature blocks into a single matrix (~85d).
NaN audit: any row with NaN in feature matrix → DELETE.
Provides normalize_split() for split-aware normalization.
"""

import json
import logging

import numpy as np
import pandas as pd

from .constants import AUX_RGROUP_TYPES

logger = logging.getLogger(__name__)


def normalize_split(X: np.ndarray, train_idx: np.ndarray,
                    continuous_mask: np.ndarray) -> np.ndarray:
    """Split-aware normalization: compute stats from train only, apply to all.

    Args:
        X: feature matrix (n_samples, n_features)
        train_idx: indices of training samples
        continuous_mask: boolean mask, True for continuous features to normalize

    Returns: normalized feature matrix
    """
    X_norm = X.copy()
    if len(train_idx) == 0:
        return X_norm

    for j in range(X.shape[1]):
        if continuous_mask[j]:
            mean = np.nanmean(X[train_idx, j])
            std = np.nanstd(X[train_idx, j])
            if std > 1e-8:
                X_norm[:, j] = (X[:, j] - mean) / std
            else:
                X_norm[:, j] = 0.0
    return X_norm


def run(context: dict) -> dict:
    """Integrate all features and enforce strict NaN deletion."""
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    n_start = len(df)
    logger.info(f"Step 12: Feature integration for {n_start} rows")

    # ── Collect feature columns ──

    # 1. Enolate steric (24d)
    enolate_cols = context.get("enolate_steric_cols", [])
    # 2. Aldehyde steric (10d)
    aldehyde_cols = context.get("aldehyde_steric_cols", [])
    # 3. Conditions (44d)
    condition_cols = context.get("condition_feature_cols", [])
    # 4. Auxiliary features
    # aux_config_R (1d from CIP)
    if "aux_C4_cip" in df.columns:
        df["aux_config_R"] = df["aux_C4_cip"].map({"R": 1.0, "S": 0.0}).fillna(-1.0)
    else:
        logger.warning("  aux_C4_cip not found, setting aux_config_R = -1")
        df["aux_config_R"] = -1.0
    # aux_rgroup one-hot
    if "aux_rgroup_type" in df.columns:
        for rt in AUX_RGROUP_TYPES:
            df[f"aux_rg_{rt}"] = (df["aux_rgroup_type"] == rt).astype(float)
    else:
        for rt in AUX_RGROUP_TYPES:
            df[f"aux_rg_{rt}"] = 0.0
    # aux_n_stereocenters (should exist from step 3; skip if missing)
    if "n_defined_stereocenters" not in df.columns:
        logger.warning("  n_defined_stereocenters missing — excluding from features")

    aux_cols = ["aux_config_R"] + [f"aux_rg_{rt}" for rt in AUX_RGROUP_TYPES]
    if "n_defined_stereocenters" in df.columns:
        aux_cols.append("n_defined_stereocenters")

    all_feature_cols = enolate_cols + aldehyde_cols + condition_cols + aux_cols
    logger.info(f"  Feature blocks: enolate={len(enolate_cols)}d, "
                f"aldehyde={len(aldehyde_cols)}d, "
                f"conditions={len(condition_cols)}d, "
                f"auxiliary={len(aux_cols)}d")
    logger.info(f"  Total features: {len(all_feature_cols)}d")

    # Check which columns exist
    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"  Missing {len(missing_cols)} feature columns: {missing_cols[:5]}...")
        all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    # ── NaN audit ──
    feat_matrix = df[all_feature_cols].values.astype(float)
    nan_per_row = np.isnan(feat_matrix).sum(axis=1)
    nan_per_col = np.isnan(feat_matrix).sum(axis=0)

    # Log NaN statistics
    n_nan_rows = (nan_per_row > 0).sum()
    logger.info(f"  Rows with NaN: {n_nan_rows}/{n_start}")
    for j, col in enumerate(all_feature_cols):
        if nan_per_col[j] > 0:
            logger.info(f"    {col}: {int(nan_per_col[j])} NaN")

    # Delete rows with any NaN
    if n_nan_rows > 0:
        nan_mask = nan_per_row > 0
        for i in range(n_start):
            if nan_mask[i]:
                oi = df.iloc[i]["original_index"]
                if oi not in audit._deletion_reasons:
                    audit.mark_deleted_by_oi([oi], "feature_nan")
        df = df[~nan_mask].reset_index(drop=True)
        logger.info(f"  Deleted {n_nan_rows} rows with NaN features")

    # ── Build label columns ──
    # Normalize labels to 0/1
    for lbl in ["label_Ca", "label_Cb", "label_SA"]:
        if lbl in df.columns:
            vals = df[lbl].values
            if np.nanmin(vals) < 0:
                df[lbl] = ((vals + 1) / 2).astype(int)

    if "label_Ca" in df.columns and "label_Cb" in df.columns:
        df["label_joint"] = df["label_Ca"].astype(int) * 2 + df["label_Cb"].astype(int)

    # ── Build feature manifest ──
    # Identify continuous vs binary columns
    continuous_cols = set(enolate_cols + aldehyde_cols)
    # Add continuous condition features
    continuous_condition = [
        "base_pKa", "base_steric_A", "base_nucleophilicity",
        "metal_coordination_num", "metal_ionic_radius_pm", "metal_pearson_hardness",
        "solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30",
        "solvent_epsilon", "solvent_viscosity", "solvent_bp",
    ]
    continuous_cols.update(continuous_condition)
    continuous_cols.add("aux_mw")
    continuous_cols.add("n_defined_stereocenters")

    is_continuous = [col in continuous_cols for col in all_feature_cols]

    manifest = {
        "feature_names": all_feature_cols,
        "n_features": len(all_feature_cols),
        "continuous_mask": is_continuous,
        "blocks": {
            "enolate_steric": enolate_cols,
            "aldehyde_steric": aldehyde_cols,
            "conditions": condition_cols,
            "auxiliary": aux_cols,
        },
    }

    # ── Save outputs ──
    # Raw features CSV
    feat_df = df[["original_index"] + all_feature_cols]
    feat_path = context["output_dir"] / "features" / "v3_features_raw.csv"
    feat_df.to_csv(feat_path, index=False)

    # Feature manifest JSON
    manifest_path = context["output_dir"] / "features" / "feature_names.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Labels
    label_cols = [c for c in ["label_Ca", "label_Cb", "label_SA", "label_joint"] if c in df.columns]
    if label_cols:
        label_df = df[["original_index"] + label_cols]
        label_path = context["output_dir"] / "features" / "labels.csv"
        label_df.to_csv(label_path, index=False)

    n_end = len(df)
    logger.info(f"  Step 12 complete: {n_start} → {n_end} rows, {len(all_feature_cols)}d features")

    context["df"] = df
    context["all_feature_cols"] = all_feature_cols
    context["continuous_mask"] = np.array(is_continuous)
    return context
