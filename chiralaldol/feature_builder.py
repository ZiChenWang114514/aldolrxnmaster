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


def build_chiralaldol_v2_features(project_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Build the ChiralAldol V2 feature matrix (75d).

    Combines:
      1. Enolate 3D steric descriptors (24d, M3)
      2. Aldehyde 3D steric descriptors (10d, M3b) — NEW
      3. Reaction conditions (35d)
      4. Auxiliary chirality (6d)

    Returns (X, feature_names) where X is (1822, 75) float32 array.
    """
    from .aldehyde_steric import ALDEHYDE_STERIC_DESC_NAMES

    feat_dir = project_dir / "data" / "processed" / "features"
    chiralaldol_dir = project_dir / "data" / "processed" / "chiralaldol"

    # 1. Enolate steric features (24d)
    steric_path = chiralaldol_dir / "steric_features.csv"
    steric_df = pd.read_csv(steric_path)
    logger.info(f"Loaded enolate steric features: {steric_df.shape}")

    # 2. Aldehyde steric features (10d)
    ald_path = chiralaldol_dir / "aldehyde_steric_features.csv"
    if not ald_path.exists():
        raise FileNotFoundError(
            f"Aldehyde steric features not found: {ald_path}. "
            "Run stage3b_aldehyde_features() first."
        )
    ald_df = pd.read_csv(ald_path)
    logger.info(f"Loaded aldehyde steric features: {ald_df.shape}")

    # 3. Reaction conditions (35d)
    cond_df = pd.read_csv(feat_dir / "reaction_conditions.csv")
    logger.info(f"Loaded reaction conditions: {cond_df.shape}")

    # 4. Auxiliary chirality (6d)
    aux_df = pd.read_csv(feat_dir / "auxchiral_features.csv")
    logger.info(f"Loaded auxiliary chirality: {aux_df.shape}")

    assert len(steric_df) == len(ald_df) == len(cond_df) == len(aux_df), (
        f"Row count mismatch: enolate={len(steric_df)}, ald={len(ald_df)}, "
        f"cond={len(cond_df)}, aux={len(aux_df)}"
    )

    steric_cols = list(steric_df.columns)
    ald_cols = list(ald_df.columns)
    cond_cols = list(cond_df.columns)
    aux_cols = list(aux_df.columns)

    X_steric = steric_df.values.astype(np.float32)
    X_ald = ald_df.values.astype(np.float32)
    X_cond = cond_df.values.astype(np.float32)
    X_aux = aux_df.values.astype(np.float32)

    X = np.hstack([X_steric, X_ald, X_cond, X_aux])
    feature_names = steric_cols + ald_cols + cond_cols + aux_cols

    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        f"ChiralAldol V2 features: {X.shape} "
        f"(enolate_steric={X_steric.shape[1]}, ald_steric={X_ald.shape[1]}, "
        f"cond={X_cond.shape[1]}, aux={X_aux.shape[1]})"
    )
    return X, feature_names


def build_chiralaldol_v3_features(project_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Build the ChiralAldol V3 feature matrix (87d).

    Combines:
      1. Enolate 3D steric descriptors (24d, M3)
      2. Aldehyde 3D steric descriptors (10d, M3b)
      3. GFN2-xTB electronic descriptors (12d, B1) — NEW
      4. Reaction conditions (35d)
      5. Auxiliary chirality (6d)

    Returns (X, feature_names) where X is (1822, 87) float32 array.
    """
    from .xtb_descriptors import XTB_FEATURE_NAMES

    feat_dir = project_dir / "data" / "processed" / "features"
    chiralaldol_dir = project_dir / "data" / "processed" / "chiralaldol"

    # 1. Enolate steric (24d)
    steric_df = pd.read_csv(chiralaldol_dir / "steric_features.csv")

    # 2. Aldehyde steric (10d)
    ald_path = chiralaldol_dir / "aldehyde_steric_features.csv"
    if not ald_path.exists():
        raise FileNotFoundError(f"Run stage3b first: {ald_path}")
    ald_df = pd.read_csv(ald_path)

    # 3. xTB electronic (12d)
    xtb_path = chiralaldol_dir / "xtb_electronic_features.csv"
    if not xtb_path.exists():
        raise FileNotFoundError(f"Run stage3c first: {xtb_path}")
    xtb_df = pd.read_csv(xtb_path)

    # 4. Conditions (35d)
    cond_df = pd.read_csv(feat_dir / "reaction_conditions.csv")

    # 5. Auxiliary chirality (6d)
    aux_df = pd.read_csv(feat_dir / "auxchiral_features.csv")

    n = len(steric_df)
    assert all(len(df) == n for df in [ald_df, xtb_df, cond_df, aux_df])

    X_steric = steric_df.values.astype(np.float32)
    X_ald = ald_df.values.astype(np.float32)
    X_xtb = xtb_df.values.astype(np.float32)
    X_cond = cond_df.values.astype(np.float32)
    X_aux = aux_df.values.astype(np.float32)

    X = np.hstack([X_steric, X_ald, X_xtb, X_cond, X_aux])
    feature_names = (list(steric_df.columns) + list(ald_df.columns)
                     + XTB_FEATURE_NAMES + list(cond_df.columns) + list(aux_df.columns))

    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        f"ChiralAldol V3 features: {X.shape} "
        f"(enolate_steric={X_steric.shape[1]}, ald_steric={X_ald.shape[1]}, "
        f"xtb={X_xtb.shape[1]}, cond={X_cond.shape[1]}, aux={X_aux.shape[1]})"
    )
    return X, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    project = Path("/data2/zcwang/aldolrxnmaster")
    X, names = build_chiralaldol_features(project)
    print(f"Feature matrix: {X.shape}")
    print(f"Feature names ({len(names)}): {names}")
    print(f"NaN count: {np.isnan(X).sum()}")
