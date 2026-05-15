#!/usr/bin/env python3
"""Phase A: Data audit + cleaning for AldolRxnMaster.

Performs:
  A1. Fix chirality_valid bug (was using Product_ instead of Raw_Product_Smiles)
  A2. Delete 21 rows with missing Ketone/Aldehyde
  A3. Fill missing solvents via metal/reagent inference
  A4. Output quality_audit.csv and evans_v2_clean.csv
  A5. Re-align all feature CSVs and regenerate splits

Usage:
    conda run -n aldol-rxn python scripts/run_data_audit.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from chiralaldol.solvent_lookup import fill_missing_solvents
from src.aldolrxnmaster.data.split import temporal_split, scaffold_split, grouped_random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_defined_stereocenters(smiles: str) -> int:
    """Count R/S-assigned stereocenters from SMILES."""
    if pd.isna(smiles) or not str(smiles).strip():
        return -1
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return -1
    try:
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return len([c for c in chiral if c[1] in ("R", "S")])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Phase A: Data Audit + Cleaning")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    csv_path = PROJECT_DIR / "data" / "processed" / "evans_clean.csv"
    df = pd.read_csv(csv_path)
    n_orig = len(df)
    logger.info(f"Loaded {csv_path.name}: {n_orig} rows, {df.shape[1]} columns")

    audit_log = []
    audit_log.append(f"Original dataset: {n_orig} rows")

    # -----------------------------------------------------------------------
    # A1: Fix chirality_valid using Raw_Product_Smiles
    # -----------------------------------------------------------------------
    logger.info("\n--- A1: Fixing chirality_valid (using Raw_Product_Smiles) ---")

    df["n_stereocenters_raw"] = df["Raw_Product_Smiles"].apply(count_defined_stereocenters)
    df["chirality_valid_fixed"] = df["n_stereocenters_raw"] >= 2

    n_fixed_valid = df["chirality_valid_fixed"].sum()
    n_old_valid = df["chirality_valid"].sum()

    audit_log.append(f"\nA1: Chirality valid fix")
    audit_log.append(f"  Old chirality_valid (Product_ column): {n_old_valid}/{n_orig}")
    audit_log.append(f"  New chirality_valid (Raw_Product_Smiles): {n_fixed_valid}/{n_orig}")
    audit_log.append(f"  Recovered: {n_fixed_valid - n_old_valid} rows")

    logger.info(f"  Old valid: {n_old_valid}, New valid: {n_fixed_valid} "
                f"(+{n_fixed_valid - n_old_valid} recovered)")

    # Update the column
    df["chirality_valid"] = df["chirality_valid_fixed"]
    df["n_stereocenters"] = df["n_stereocenters_raw"]
    df.drop(columns=["n_stereocenters_raw", "chirality_valid_fixed"], inplace=True)

    # -----------------------------------------------------------------------
    # A2: Delete rows with missing Ketone/Aldehyde
    # -----------------------------------------------------------------------
    logger.info("\n--- A2: Removing rows with missing Ketone/Aldehyde ---")

    missing_mask = df["Ketone"].isna() | df["Aldehyde"].isna()
    n_missing = missing_mask.sum()
    missing_indices = df[missing_mask].index.tolist()

    audit_log.append(f"\nA2: Missing molecule removal")
    audit_log.append(f"  Rows with missing Ketone/Aldehyde: {n_missing}")
    audit_log.append(f"  Indices: {missing_indices}")

    logger.info(f"  Deleting {n_missing} rows with missing molecules")

    # Keep valid rows
    valid_mask = ~missing_mask
    df_clean = df[valid_mask].reset_index(drop=True)
    n_clean = len(df_clean)

    audit_log.append(f"  After removal: {n_clean} rows")

    # -----------------------------------------------------------------------
    # A3: Fill missing solvents
    # -----------------------------------------------------------------------
    logger.info("\n--- A3: Filling missing solvents ---")

    n_unknown_before = (df_clean["solvent_known"] == False).sum()
    df_clean = fill_missing_solvents(df_clean)
    n_unknown_after = (df_clean["solvent_known"] == False).sum()
    n_inferred = (df_clean["solvent_inferred"] == True).sum()

    audit_log.append(f"\nA3: Solvent inference")
    audit_log.append(f"  Unknown before: {n_unknown_before}")
    audit_log.append(f"  Inferred: {n_inferred}")
    audit_log.append(f"  Still unknown: {n_unknown_after}")
    audit_log.append(f"  Fill rate: {(1 - n_unknown_after/n_unknown_before)*100:.1f}%")

    logger.info(f"  Before: {n_unknown_before} unknown → After: {n_unknown_after} "
                f"({n_inferred} inferred)")

    # -----------------------------------------------------------------------
    # Quality report
    # -----------------------------------------------------------------------
    logger.info("\n--- Quality Report ---")

    # Class distribution
    label_dist = df_clean["label_joint"].value_counts().sort_index()
    audit_log.append(f"\nClass distribution (clean):")
    for cls, cnt in label_dist.items():
        pct = cnt / n_clean * 100
        audit_log.append(f"  Class {int(cls)}: {cnt} ({pct:.1f}%)")

    # Year distribution
    year_dist = df_clean["Year"].describe()
    audit_log.append(f"\nYear range: {int(year_dist['min'])}-{int(year_dist['max'])}")

    # Solvent coverage
    solvent_known_pct = df_clean["solvent_known"].mean() * 100
    audit_log.append(f"Solvent known: {solvent_known_pct:.1f}%")

    # Chirality coverage
    chiral_valid_pct = df_clean["chirality_valid"].mean() * 100
    audit_log.append(f"Chirality valid: {chiral_valid_pct:.1f}%")

    logger.info(f"  Clean dataset: {n_clean} rows")
    logger.info(f"  Classes: {label_dist.to_dict()}")
    logger.info(f"  Solvent known: {solvent_known_pct:.1f}%")
    logger.info(f"  Chirality valid: {chiral_valid_pct:.1f}%")

    # -----------------------------------------------------------------------
    # Save quality audit
    # -----------------------------------------------------------------------
    audit_dir = PROJECT_DIR / "data" / "processed"

    # Save quality audit CSV (row-level)
    audit_csv = audit_dir / "quality_audit.csv"
    audit_df = df_clean[["Reaction_ID", "Year", "label_joint",
                          "chirality_valid", "n_stereocenters",
                          "solvent_known", "solvent_inferred"]].copy()
    audit_df.to_csv(audit_csv, index=False)
    logger.info(f"  Saved {audit_csv}")

    # Save audit log
    audit_log_path = audit_dir / "quality_audit_log.txt"
    with open(audit_log_path, "w") as f:
        f.write("\n".join(audit_log))
    logger.info(f"  Saved {audit_log_path}")

    # -----------------------------------------------------------------------
    # Save cleaned dataset
    # -----------------------------------------------------------------------
    clean_path = audit_dir / "evans_v2_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    logger.info(f"  Saved {clean_path}: {n_clean} rows")

    # -----------------------------------------------------------------------
    # Re-align feature CSVs (remove deleted rows)
    # -----------------------------------------------------------------------
    logger.info("\n--- Re-aligning feature CSVs ---")

    chiralaldol_dir = audit_dir / "chiralaldol"
    feat_dir = audit_dir / "features"

    feature_files = [
        chiralaldol_dir / "steric_features.csv",
        chiralaldol_dir / "aldehyde_steric_features.csv",
        chiralaldol_dir / "enolates.csv",
        feat_dir / "reaction_conditions.csv",
        feat_dir / "auxchiral_features.csv",
    ]

    for fpath in feature_files:
        if fpath.exists():
            feat_df = pd.read_csv(fpath)
            if len(feat_df) == n_orig:
                feat_clean = feat_df[valid_mask.values].reset_index(drop=True)
                # Backup original
                backup = fpath.with_suffix(".csv.bak_v1")
                if not backup.exists():
                    feat_df.to_csv(backup, index=False)
                feat_clean.to_csv(fpath, index=False)
                logger.info(f"  {fpath.name}: {n_orig} → {len(feat_clean)} rows (backed up to .bak_v1)")
            else:
                logger.warning(f"  {fpath.name}: unexpected row count {len(feat_df)}, skipping")
        else:
            logger.warning(f"  {fpath.name}: not found, skipping")

    # Also handle labels.csv
    labels_path = feat_dir / "labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        if len(labels_df) == n_orig:
            backup = labels_path.with_suffix(".csv.bak_v1")
            if not backup.exists():
                labels_df.to_csv(backup, index=False)
            labels_clean = labels_df[valid_mask.values].reset_index(drop=True)
            labels_clean.to_csv(labels_path, index=False)
            logger.info(f"  labels.csv: {n_orig} → {len(labels_clean)} rows")

    # Handle reaction_smiles.csv
    rxn_smi_path = feat_dir / "reaction_smiles.csv"
    if rxn_smi_path.exists():
        rxn_df = pd.read_csv(rxn_smi_path)
        if len(rxn_df) == n_orig:
            backup = rxn_smi_path.with_suffix(".csv.bak_v1")
            if not backup.exists():
                rxn_df.to_csv(backup, index=False)
            rxn_clean = rxn_df[valid_mask.values].reset_index(drop=True)
            rxn_clean.to_csv(rxn_smi_path, index=False)
            logger.info(f"  reaction_smiles.csv: {n_orig} → {len(rxn_clean)} rows")

    # Handle fingerprint npz files
    fp_dir = audit_dir / "fingerprints"
    for fp_name in ["drfp_fps.npz", "rxnfp_fps.npz"]:
        fp_path = fp_dir / fp_name
        if fp_path.exists():
            data = np.load(fp_path)
            fps = data["fps"]
            if fps.shape[0] == n_orig:
                backup = fp_path.with_suffix(".npz.bak_v1")
                if not backup.exists():
                    np.savez_compressed(backup, fps=fps)
                fps_clean = fps[valid_mask.values]
                np.savez_compressed(fp_path, fps=fps_clean)
                logger.info(f"  {fp_name}: {fps.shape[0]} → {fps_clean.shape[0]} rows")

    # Handle tabular_features.npz
    tab_path = feat_dir / "tabular_features.npz"
    if tab_path.exists():
        data = np.load(tab_path, allow_pickle=True)
        # Check what keys are in it
        for key in data.files:
            arr = data[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 1 and arr.shape[0] == n_orig:
                logger.info(f"  tabular_features.npz[{key}]: {arr.shape[0]} → {arr.shape[0] - n_missing}")

    # -----------------------------------------------------------------------
    # Update reaction_conditions.csv with new solvent values
    # -----------------------------------------------------------------------
    logger.info("\n--- Updating reaction_conditions.csv with inferred solvents ---")

    cond_path = feat_dir / "reaction_conditions.csv"
    if cond_path.exists():
        cond_df = pd.read_csv(cond_path)
        # Find solvent columns
        solvent_cols = [c for c in cond_df.columns if c.startswith("solvent_")]
        if solvent_cols:
            # Update from cleaned df
            for col in ["solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30", "solvent_known"]:
                if col in cond_df.columns and col in df_clean.columns:
                    cond_df[col] = df_clean[col].values
            cond_df.to_csv(cond_path, index=False)
            logger.info(f"  Updated solvent parameters in reaction_conditions.csv")

    # -----------------------------------------------------------------------
    # Regenerate splits for cleaned data
    # -----------------------------------------------------------------------
    logger.info("\n--- Regenerating splits ---")

    splits_dir = audit_dir / "splits"

    # Evans temporal
    split_temporal = temporal_split(df_clean)
    with open(splits_dir / "evans_temporal.json", "w") as f:
        json.dump(split_temporal, f, indent=2)
    logger.info(f"  evans_temporal: train={len(split_temporal['train'])}, "
                f"val={len(split_temporal['val'])}, test={len(split_temporal['test'])}")

    # Evans scaffold
    split_scaffold = scaffold_split(df_clean)
    with open(splits_dir / "evans_scaffold.json", "w") as f:
        json.dump(split_scaffold, f, indent=2)
    logger.info(f"  evans_scaffold: train={len(split_scaffold['train'])}, "
                f"val={len(split_scaffold['val'])}, test={len(split_scaffold['test'])}")

    # Evans grouped random (5 seeds)
    for seed in [42, 123, 456, 789, 1024]:
        split_gr = grouped_random_split(df_clean, seed=seed)
        with open(splits_dir / f"evans_grouped_random_seed{seed}.json", "w") as f:
            json.dump(split_gr, f, indent=2)

    logger.info(f"  Regenerated 7 split files for {n_clean}-row dataset")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Original:    {n_orig} rows")
    logger.info(f"  Removed:     {n_missing} (missing Ketone/Aldehyde)")
    logger.info(f"  Clean:       {n_clean} rows")
    logger.info(f"  Chirality:   {chiral_valid_pct:.1f}% valid (was {n_old_valid/n_orig*100:.1f}%)")
    logger.info(f"  Solvent:     {solvent_known_pct:.1f}% known (was {(n_orig-n_unknown_before)/n_orig*100:.1f}%)")
    logger.info(f"  Output:      {clean_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
