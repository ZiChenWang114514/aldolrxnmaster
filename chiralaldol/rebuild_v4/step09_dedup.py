"""Step 09: Deduplication and group_id assignment."""

import hashlib
import logging

import pandas as pd

from .audit import AuditTracker
from .utils import canonical_smiles

logger = logging.getLogger("rebuild_v4.step09")


def _make_substrate_key(ketone: str, aldehyde: str) -> str:
    """Role-aware substrate key: ketone||aldehyde (order preserved)."""
    k = canonical_smiles(str(ketone)) if pd.notna(ketone) else ""
    a = canonical_smiles(str(aldehyde)) if pd.notna(aldehyde) else ""
    return f"{k or ''}||{a or ''}"


def _make_rxn_hash(row: pd.Series, condition_cols: list[str]) -> str:
    """Hash reaction + conditions + labels for exact duplicate detection."""
    parts = [
        str(row.get("substrate_key", "")),
        str(row.get("label_Ca", "")),
        str(row.get("label_Cb", "")),
    ]
    for col in condition_cols:
        parts.append(str(row.get(col, "")))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Deduplicate and assign group_id for split integrity."""
    logger.info("Step 09: Deduplication and group_id assignment...")
    n_start = len(df)

    # Use canonical SMILES columns if available
    ketone_col = "canonical_ketone_smiles" if "canonical_ketone_smiles" in df.columns else "ketone_smiles"
    aldehyde_col = "canonical_aldehyde_smiles" if "canonical_aldehyde_smiles" in df.columns else "aldehyde_smiles"

    # --- Substrate key ---
    df["substrate_key"] = [
        _make_substrate_key(row[ketone_col], row[aldehyde_col])
        for _, row in df.iterrows()
    ]

    # --- Group ID (same substrate pair -> same group) ---
    unique_keys = df["substrate_key"].unique()
    key_to_gid = {k: i for i, k in enumerate(sorted(unique_keys))}
    df["group_id"] = df["substrate_key"].map(key_to_gid)
    logger.info(f"  Assigned {len(unique_keys)} group IDs")

    # --- Exact dedup ---
    condition_cols = []
    for col in ["Reagent", "Catalyst", "Solvent (Reaction Details)",
                 "Temperature (Reaction Details) [C]"]:
        if col in df.columns:
            condition_cols.append(col)

    df["rxn_hash"] = df.apply(lambda r: _make_rxn_hash(r, condition_cols), axis=1)

    dup_mask = df.duplicated(subset="rxn_hash", keep="first")
    n_dups = dup_mask.sum()
    audit.record_drop("09_dedup", df.loc[dup_mask, "_orig_idx"], "exact_duplicate")
    df = df[~dup_mask].reset_index(drop=True)

    logger.info(f"  Removed {n_dups} exact duplicates")
    audit.record_step("09_dedup", len(df))
    logger.info(f"  Step 09 complete: {n_start} -> {len(df)} rows")
    return df
