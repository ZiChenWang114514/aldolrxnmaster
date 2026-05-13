"""M4: Feature Builder — Combine 3D steric descriptors with reaction conditions.

Builds the unified ChiralAldol feature matrix:
  - 3D steric descriptors (from M3): ~24d
  - Reaction conditions (existing): 35d
  - Auxiliary chirality (existing): 6d
  Total: ~65d (ChiralAldol-Core)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .steric_descriptors import (
    STERIC_DESC_NAMES,
    compute_ensemble_descriptors,
    find_reactive_center,
)

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)


def compute_all_steric_features(project_dir: Path) -> pd.DataFrame:
    """Compute steric features for all 1822 molecules using precomputed ensembles.

    Loads conformer_ensembles.pkl and enolates.csv, computes descriptors,
    saves to steric_features.csv.
    """
    chiralaldol_dir = project_dir / "data" / "processed" / "chiralaldol"

    # Load precomputed data
    enolates = pd.read_csv(chiralaldol_dir / "enolates.csv")
    with open(chiralaldol_dir / "conformer_ensembles.pkl", "rb") as f:
        ensembles = pickle.load(f)

    n = len(enolates)
    rows = []
    n_ok = 0

    for i in range(n):
        smi = str(enolates["enolate_smiles"].iloc[i])
        ens = ensembles.get(i)

        if ens is None:
            rows.append({k: 0.0 for k in STERIC_DESC_NAMES})
            continue

        desc = compute_ensemble_descriptors(smi, ens)
        if desc is None:
            rows.append({k: 0.0 for k in STERIC_DESC_NAMES})
            continue

        row = {}
        for k in STERIC_DESC_NAMES:
            row[k] = desc.get(k, 0.0)
        rows.append(row)
        n_ok += 1

    logger.info(f"Steric features: {n_ok}/{n} computed successfully")

    df = pd.DataFrame(rows)
    # Ensure column order
    df = df[STERIC_DESC_NAMES]
    return df


def build_chiralaldol_features(project_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Build the full ChiralAldol feature matrix.

    Combines:
      1. 3D steric descriptors (M3)
      2. Reaction conditions (existing)
      3. Auxiliary chirality (existing)

    Returns (X, feature_names) where X is (1822, d) float32 array.
    """
    feat_dir = project_dir / "data" / "processed" / "features"
    chiralaldol_dir = project_dir / "data" / "processed" / "chiralaldol"

    # 1. Load or compute steric features
    steric_path = chiralaldol_dir / "steric_features.csv"
    if steric_path.exists():
        steric_df = pd.read_csv(steric_path)
        logger.info(f"Loaded steric features: {steric_df.shape}")
    else:
        logger.info("Computing steric features (first time)...")
        steric_df = compute_all_steric_features(project_dir)
        steric_df.to_csv(steric_path, index=False)
        logger.info(f"Saved steric features to {steric_path}")

    # 2. Load reaction conditions (35d)
    cond_df = pd.read_csv(feat_dir / "reaction_conditions.csv")
    logger.info(f"Loaded reaction conditions: {cond_df.shape}")

    # 3. Load auxiliary chirality (6d)
    aux_df = pd.read_csv(feat_dir / "auxchiral_features.csv")
    logger.info(f"Loaded auxiliary chirality: {aux_df.shape}")

    # Validate alignment
    assert len(steric_df) == len(cond_df) == len(aux_df), \
        f"Row count mismatch: steric={len(steric_df)}, cond={len(cond_df)}, aux={len(aux_df)}"

    # Combine
    steric_cols = list(steric_df.columns)
    cond_cols = list(cond_df.columns)
    aux_cols = list(aux_df.columns)

    X_steric = steric_df.values.astype(np.float32)
    X_cond = cond_df.values.astype(np.float32)
    X_aux = aux_df.values.astype(np.float32)

    X = np.hstack([X_steric, X_cond, X_aux])
    feature_names = steric_cols + cond_cols + aux_cols

    # Replace NaN/inf
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"ChiralAldol features: {X.shape} "
                f"(steric={X_steric.shape[1]}, cond={X_cond.shape[1]}, aux={X_aux.shape[1]})")

    return X, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    project = Path("/data2/zcwang/aldolrxnmaster")
    X, names = build_chiralaldol_features(project)
    print(f"Feature matrix: {X.shape}")
    print(f"Feature names ({len(names)}): {names}")
    print(f"NaN count: {np.isnan(X).sum()}")
