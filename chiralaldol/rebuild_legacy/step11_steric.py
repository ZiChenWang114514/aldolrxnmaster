"""Step 11: 3D steric features — enolate (24d) + aldehyde (10d) = 34d.

Reuses core computation from chiralaldol/steric_descriptors.py and aldehyde_steric.py.
Key fix: conformer ensembles store AddHs mol + full coords, but steric functions
expect no-H mol + heavy-atom-only coords. This module handles the mapping.
"""

import logging
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

ENOLATE_STERIC_NAMES = [
    "Vbur_si_mean", "Vbur_si_std", "Vbur_re_mean", "Vbur_re_std",
    "Vbur_diff_mean", "Vbur_diff_std", "Vbur_total_mean", "Vbur_total_std",
    "L_mean", "L_std", "B1_mean", "B1_std", "B5_mean", "B5_std",
    "sin_tau1_mean", "sin_tau1_std", "cos_tau1_mean", "cos_tau1_std",
    "sin_tau2_mean", "sin_tau2_std", "cos_tau2_mean", "cos_tau2_std",
    "n_conformers", "n_clusters",
]

ALDEHYDE_STERIC_NAMES = [
    "ald_L_mean", "ald_L_std", "ald_B1_mean", "ald_B1_std",
    "ald_B5_mean", "ald_B5_std", "ald_Vbur_total_mean", "ald_Vbur_total_std",
    "ald_n_conformers", "ald_n_clusters",
]


def _get_heavy_atom_indices(mol_with_h):
    """Get indices of heavy atoms (non-H) in an AddHs mol."""
    return [i for i in range(mol_with_h.GetNumAtoms())
            if mol_with_h.GetAtomWithIdx(i).GetAtomicNum() > 1]


def _extract_heavy_coords(mol_with_h, coords_with_h):
    """Extract heavy-atom-only coordinates from full (with-H) coordinates."""
    heavy_idx = _get_heavy_atom_indices(mol_with_h)
    if coords_with_h is None or len(coords_with_h) < max(heavy_idx) + 1:
        return None
    return coords_with_h[heavy_idx]


def _compute_enolate_steric(ensemble: dict) -> dict | None:
    """Compute 24d enolate steric descriptors from conformer ensemble."""
    reps = ensemble.get("representatives", [])
    mol_h = ensemble.get("mol")
    if not reps or mol_h is None:
        return None

    mol_no_h = Chem.RemoveHs(mol_h)

    from chiralaldol.steric_descriptors import (
        find_reactive_center, compute_single_conformer_descriptors,
    )

    center = find_reactive_center(mol_no_h)
    if center is None:
        return None

    all_desc = []
    weights = []

    for conf_id, energy, weight, coords_full in reps:
        if coords_full is None:
            continue
        coords_heavy = _extract_heavy_coords(mol_h, coords_full)
        if coords_heavy is None or len(coords_heavy) != mol_no_h.GetNumAtoms():
            continue
        desc = compute_single_conformer_descriptors(mol_no_h, coords_heavy, center)
        if desc is not None:
            all_desc.append(desc)
            weights.append(weight)

    if not all_desc:
        return None

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    result = {}
    for key in sorted(all_desc[0].keys()):
        values = np.array([d[key] for d in all_desc])
        wmean = np.average(values, weights=weights)
        wstd = np.sqrt(np.average((values - wmean) ** 2, weights=weights))
        result[f"{key}_mean"] = float(wmean)
        result[f"{key}_std"] = float(wstd)

    result["n_conformers"] = len(all_desc)
    result["n_clusters"] = ensemble.get("n_clusters", len(all_desc))
    return result


def _compute_aldehyde_steric(ensemble: dict) -> dict | None:
    """Compute 10d aldehyde steric descriptors from conformer ensemble."""
    reps = ensemble.get("representatives", [])
    mol_h = ensemble.get("mol")
    if not reps or mol_h is None:
        return None

    mol_no_h = Chem.RemoveHs(mol_h)

    from chiralaldol.aldehyde_steric import (
        find_aldehyde_center, find_aldehyde_R_atoms,
        compute_aldehyde_single_conformer,
    )

    center = find_aldehyde_center(mol_no_h)
    if center is None:
        return None

    carbonyl_idx, o_idx = center
    r_idxs = find_aldehyde_R_atoms(mol_no_h, carbonyl_idx, o_idx)

    all_desc = []
    weights = []

    for conf_id, energy, weight, coords_full in reps:
        if coords_full is None:
            continue
        coords_heavy = _extract_heavy_coords(mol_h, coords_full)
        if coords_heavy is None or len(coords_heavy) != mol_no_h.GetNumAtoms():
            continue
        desc = compute_aldehyde_single_conformer(mol_no_h, coords_heavy,
                                                  carbonyl_idx, o_idx, r_idxs)
        if desc is not None:
            all_desc.append(desc)
            weights.append(weight)

    if not all_desc:
        return None

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    result = {}
    for key in ["L", "B1", "B5", "Vbur_total"]:
        values = np.array([d[key] for d in all_desc])
        wmean = np.average(values, weights=weights)
        wstd = np.sqrt(np.average((values - wmean) ** 2, weights=weights))
        result[f"ald_{key}_mean"] = float(wmean)
        result[f"ald_{key}_std"] = float(wstd)

    result["ald_n_conformers"] = len(all_desc)
    result["ald_n_clusters"] = ensemble.get("n_clusters", len(all_desc))
    return result


def run(context: dict) -> dict:
    """Compute 3D steric features for all rows with conformers."""
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    conformer_ensembles = context.get("conformer_ensembles", {})
    n_start = len(df)
    logger.info(f"Step 11: 3D steric features for {n_start} rows")

    sys.path.insert(0, str(context["project_dir"]))

    enolate_feats = []
    aldehyde_feats = []
    steric_failed = []

    for i, row in df.iterrows():
        oi = row["original_index"]
        ens = conformer_ensembles.get(oi)

        if ens is None:
            enolate_feats.append(None)
            aldehyde_feats.append(None)
            steric_failed.append(True)
            continue

        enolate_result = _compute_enolate_steric(ens.get("enolate", {}))
        aldehyde_result = _compute_aldehyde_steric(ens.get("aldehyde", {}))

        if enolate_result is None or aldehyde_result is None:
            enolate_feats.append(None)
            aldehyde_feats.append(None)
            steric_failed.append(True)
        else:
            enolate_feats.append(enolate_result)
            aldehyde_feats.append(aldehyde_result)
            steric_failed.append(False)

        if (i + 1) % 200 == 0:
            n_ok = sum(1 for f in steric_failed if not f)
            logger.info(f"  Steric [{i+1}/{n_start}]: {n_ok} ok")

    n_failed = sum(steric_failed)
    n_ok = n_start - n_failed
    logger.info(f"  Steric features: {n_ok}/{n_start} computed, {n_failed} failed")

    if n_failed > 0:
        fail_mask = pd.Series(steric_failed)
        for idx in df.index[fail_mask]:
            oi = df.at[idx, "original_index"]
            if oi not in audit._deletion_reasons:
                audit.mark_deleted_by_oi([oi], "steric_computation_failed")
        df = df[~fail_mask.values].reset_index(drop=True)
        enolate_feats = [f for f, failed in zip(enolate_feats, steric_failed) if not failed]
        aldehyde_feats = [f for f, failed in zip(aldehyde_feats, steric_failed) if not failed]
        logger.info(f"  Deleted {n_failed} rows with steric failure")

    if len(df) == 0:
        raise RuntimeError("Step 11: All rows failed steric computation — cannot proceed")

    # Build feature DataFrames
    enolate_df = pd.DataFrame(enolate_feats)
    aldehyde_df = pd.DataFrame(aldehyde_feats)

    # Reorder columns to match expected names
    for col in ENOLATE_STERIC_NAMES:
        if col not in enolate_df.columns:
            enolate_df[col] = 0.0
    enolate_df = enolate_df[ENOLATE_STERIC_NAMES]

    for col in ALDEHYDE_STERIC_NAMES:
        if col not in aldehyde_df.columns:
            aldehyde_df[col] = 0.0
    aldehyde_df = aldehyde_df[ALDEHYDE_STERIC_NAMES]

    # Save
    enolate_path = context["output_dir"] / "features" / "steric_features.csv"
    aldehyde_path = context["output_dir"] / "features" / "aldehyde_steric_features.csv"
    enolate_df.to_csv(enolate_path, index=False)
    aldehyde_df.to_csv(aldehyde_path, index=False)

    for col in ENOLATE_STERIC_NAMES:
        df[col] = enolate_df[col].values
    for col in ALDEHYDE_STERIC_NAMES:
        df[col] = aldehyde_df[col].values

    n_end = len(df)
    logger.info(f"  Step 11 complete: {n_start} → {n_end} rows, 34d steric features")

    context["df"] = df
    context["enolate_steric_cols"] = ENOLATE_STERIC_NAMES
    context["aldehyde_steric_cols"] = ALDEHYDE_STERIC_NAMES
    return context
