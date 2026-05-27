"""Step 12: Final audit report and output."""

import logging
from pathlib import Path

import pandas as pd

from .audit import AuditTracker
from .constants import CLEAN_V4_DIR, AUDIT_DIR

logger = logging.getLogger("rebuild_v4.step12")

# Core columns for the clean output
CLEAN_COLUMNS = [
    # Identifiers
    "Reaction ID", "_orig_idx",
    # SMILES
    "Reaction",
    "canonical_main_product_smiles", "main_product_smiles",
    "canonical_ketone_smiles", "ketone_smiles",
    "canonical_aldehyde_smiles", "aldehyde_smiles",
    # Auxiliary
    "auxiliary_type",
    # Stereo
    "n_defined_stereocenters",
    "ca_atom_idx", "cb_atom_idx",
    "mapping_confidence",
    # Labels
    "label_Ca", "label_Cb", "label_SA", "label_joint",
    "label_confidence", "label_source",
    # 3D dihedral-based syn/anti (from step08b)
    "label_syn_anti_3d", "dihedral_oh_cb_ca_co", "conformer_energy", "synanti_confidence",
    # Conditions (raw)
    "temperature_c", "time_h", "pressure_torr", "yield_pct",
    "solvent_name", "metal", "base_type", "activator_type",
    # Optical
    "ee_value", "dr_ratio", "optical_syn_anti",
    # Dedup
    "substrate_key", "group_id", "rxn_hash",
    # Reaxys raw (for reference)
    "Reagent", "Catalyst", "Solvent (Reaction Details)",
    "References",
]


def run(df: pd.DataFrame, feat_df: pd.DataFrame, audit: AuditTracker) -> None:
    """Write final outputs and audit reports."""
    logger.info("Step 12: Writing outputs and audit reports...")

    # Ensure output directories exist
    CLEAN_V4_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Clean CSV: all substrate-controlled aldol ---
    out_cols = [c for c in CLEAN_COLUMNS if c in df.columns]
    clean_df = df[out_cols].copy()

    out_path = CLEAN_V4_DIR / "substrate_aldol_clean.csv"
    clean_df.to_csv(out_path, index=False)
    logger.info(f"  Wrote {len(clean_df)} rows to {out_path}")

    # --- Evans subset (backward compatibility) ---
    evans_mask = df["auxiliary_type"] == "evans"
    evans_df = clean_df[evans_mask]
    evans_path = CLEAN_V4_DIR / "evans_clean.csv"
    evans_df.to_csv(evans_path, index=False)
    logger.info(f"  Wrote {len(evans_df)} Evans rows to {evans_path}")

    # --- Labels ---
    label_cols = ["label_Ca", "label_Cb", "label_SA", "label_joint",
                  "label_confidence", "label_source",
                  "label_syn_anti_3d", "dihedral_oh_cb_ca_co", "synanti_confidence"]
    labels = df[[c for c in label_cols if c in df.columns]].copy()
    labels.to_csv(CLEAN_V4_DIR / "labels.csv", index=False)

    # --- Condition features ---
    feat_df.to_csv(CLEAN_V4_DIR / "condition_features.csv", index=False)
    logger.info(f"  Wrote {feat_df.shape[1]}-d condition features")

    # --- Audit reports ---
    summary = audit.summary_df()
    summary.to_csv(AUDIT_DIR / "step_summary.csv", index=False)
    logger.info(f"  Wrote step summary to {AUDIT_DIR / 'step_summary.csv'}")

    row_audit = audit.row_audit_df()
    row_audit.to_csv(AUDIT_DIR / "row_audit.csv", index=False)
    logger.info(f"  Wrote {len(row_audit)} row-level audit entries")

    # --- Print summary ---
    audit.print_summary()

    # --- Distribution summary ---
    print("\n--- AUXILIARY TYPE DISTRIBUTION ---")
    print(df["auxiliary_type"].value_counts().to_string())

    print("\n--- LABEL DISTRIBUTION (4-class) ---")
    print(df["label_joint"].value_counts().sort_index().to_string())

    print(f"\n--- TOTAL CLEAN ROWS: {len(df)} ---")
    print(f"  Evans: {evans_mask.sum()}")
    print(f"  Non-Evans: {(~evans_mask).sum()}")
    print(f"  Unique substrate pairs: {df['group_id'].nunique()}")
