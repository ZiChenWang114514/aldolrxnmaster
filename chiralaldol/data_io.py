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


def save_predictions(path, test_idx, y_true, y_pred, y_prob=None, n_classes=4):
    """Save predictions to standard CSV format (idx, y_true, y_pred, prob_0..N)."""
    out = pd.DataFrame({"idx": test_idx, "y_true": y_true, "y_pred": y_pred})
    if y_prob is not None:
        for c in range(min(n_classes, y_prob.shape[1])):
            out[f"prob_{c}"] = y_prob[:, c]
    out.to_csv(path, index=False)


def _load_mechaware(feat_dir, csv_name, feat_names):
    """Load MechAware CSV + append chirality columns from v4_features."""
    feat_dir = Path(feat_dir) if feat_dir else FEAT_DIR
    path = feat_dir / csv_name
    if not path.exists():
        logger.warning("Not found: %s", path)
        return None

    X_mech = pd.read_csv(path).values.astype(np.float32)
    np.nan_to_num(X_mech, copy=False)

    X_df, feat_names = load_features(feat_dir)
    X_base = X_df.values.astype(np.float32)
    np.nan_to_num(X_base, copy=False)

    if X_mech.shape[0] != X_base.shape[0]:
        logger.warning("Row mismatch: %s has %d rows vs features %d, skipping",
                       csv_name, X_mech.shape[0], X_base.shape[0])
        return None

    chir_idx = [i for i, c in enumerate(feat_names)
                if c.startswith(CHIRALITY_PREFIXES)]
    return np.hstack([X_mech, X_base[:, chir_idx]])


def load_mechaware_bw(feat_dir=None, feat_names=None):
    """Load MechAware BW features + chirality columns. Returns array or None."""
    return _load_mechaware(feat_dir, "v4_mechaware_bw.csv", feat_names)


def load_mechaware_full(feat_dir=None, feat_names=None):
    """Load MechAware Full features + chirality columns. Returns array or None."""
    return _load_mechaware(feat_dir, "v4_mechaware_full.csv", feat_names)
