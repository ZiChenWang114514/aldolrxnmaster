"""Unified data loading utilities for AldolRxnMaster."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEAT_DIR, SPLITS_DIR, TARGET_LABEL

logger = logging.getLogger(__name__)

# Prefixes for chirality-related features appended to MechAware
CHIRALITY_PREFIXES = (
    "chiral_", "aux_rg_", "aux_oppolzer", "chiralenv_",
    "ald_pri_", "delta_chiral_", "chiral_det_",
)


def load_features(feat_dir=None):
    """Load v4_features.csv. Returns (X_df, feature_names)."""
    feat_dir = Path(feat_dir) if feat_dir else FEAT_DIR
    X_df = pd.read_csv(feat_dir / "v4_features.csv")
    return X_df, list(X_df.columns)


def load_labels(feat_dir=None):
    """Load labels.csv."""
    feat_dir = Path(feat_dir) if feat_dir else FEAT_DIR
    return pd.read_csv(feat_dir / "labels.csv")


def load_splits(splits_dir=None):
    """Load all split JSON files. Returns {name: {train: [...], test: [...]}}."""
    splits_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
    splits = {}
    for f in sorted(splits_dir.glob("*.json")):
        with open(f) as fp:
            splits[f.stem] = json.load(fp)
    return splits


def prepare_Xy(target_label=None, feat_dir=None):
    """Load features + labels, compute valid mask.

    Returns (X, y, valid_mask, feat_names) where:
        X: float32 array with NaN filled to 0
        y: int array with -1 for invalid rows
        valid_mask: boolean array
        feat_names: list of column names
    """
    target_label = target_label or TARGET_LABEL
    X_df, feat_names = load_features(feat_dir)
    labels = load_labels(feat_dir)

    if target_label not in labels.columns:
        raise ValueError(f"Target '{target_label}' not in labels.csv. "
                         f"Available: {labels.columns.tolist()}")

    valid_mask = labels[target_label].notna().values
    y_full = labels[target_label].values
    X = X_df.values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    y = np.where(valid_mask, y_full, -1).astype(int)

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.info(f"Filtering {n_invalid} rows with NaN in {target_label}")

    return X, y, valid_mask, feat_names


def load_mechaware_bw(feat_dir=None, feat_names=None):
    """Load MechAware BW features + chirality columns from v4_features.

    Returns X_ma (float32 array) or None if file missing.
    """
    feat_dir = Path(feat_dir) if feat_dir else FEAT_DIR

    bw_path = feat_dir / "v4_mechaware_bw.csv"
    if not bw_path.exists():
        logger.warning("MechAware BW not found at %s", bw_path)
        return None

    X_bw = pd.read_csv(bw_path).values.astype(np.float32)
    np.nan_to_num(X_bw, copy=False)

    # Append chirality features from v4_features
    if feat_names is None:
        X_df, feat_names = load_features(feat_dir)
        X_base = X_df.values.astype(np.float32)
        np.nan_to_num(X_base, copy=False)
    else:
        X_df, _ = load_features(feat_dir)
        X_base = X_df.values.astype(np.float32)
        np.nan_to_num(X_base, copy=False)

    # Check row count match (MechAware may be stale after data filtering)
    if X_bw.shape[0] != X_base.shape[0]:
        logger.warning("MechAware BW rows (%d) != features rows (%d), skipping",
                        X_bw.shape[0], X_base.shape[0])
        return None

    new_idx = [i for i, c in enumerate(feat_names)
               if c.startswith(CHIRALITY_PREFIXES)]
    X_new = X_base[:, new_idx]
    return np.hstack([X_bw, X_new])


def load_mechaware_full(feat_dir=None, feat_names=None):
    """Load MechAware Full features + chirality columns from v4_features.

    Returns X_ma (float32 array) or None if file missing.
    """
    feat_dir = Path(feat_dir) if feat_dir else FEAT_DIR

    full_path = feat_dir / "v4_mechaware_full.csv"
    if not full_path.exists():
        logger.warning("MechAware Full not found at %s", full_path)
        return None

    X_full = pd.read_csv(full_path).values.astype(np.float32)
    np.nan_to_num(X_full, copy=False)

    if feat_names is None:
        X_df, feat_names = load_features(feat_dir)
        X_base = X_df.values.astype(np.float32)
        np.nan_to_num(X_base, copy=False)
    else:
        X_df, _ = load_features(feat_dir)
        X_base = X_df.values.astype(np.float32)
        np.nan_to_num(X_base, copy=False)

    if X_full.shape[0] != X_base.shape[0]:
        logger.warning("MechAware Full rows (%d) != features rows (%d), skipping",
                        X_full.shape[0], X_base.shape[0])
        return None

    new_idx = [i for i, c in enumerate(feat_names)
               if c.startswith(CHIRALITY_PREFIXES)]
    X_new = X_base[:, new_idx]
    return np.hstack([X_full, X_new])
