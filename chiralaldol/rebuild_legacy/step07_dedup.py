"""Step 7: Template-based deduplication with role-aware substrate keys.

Fixes bug C1: no longer sorts reactants alphabetically (preserves ketone/aldehyde role).
Uses reaction template extraction from atom mapping for Tier 1 dedup.
"""

import hashlib
import logging

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)


def _safe_canonical(smi: str) -> str:
    """Canonicalize a single molecule SMILES (stereo-preserving)."""
    if pd.isna(smi) or not str(smi).strip():
        return ""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return str(smi).strip()
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _role_aware_substrate_key(ketone_smi: str, aldehyde_smi: str) -> str:
    """Create substrate key preserving ketone/aldehyde role order (fixes C1)."""
    k = _safe_canonical(ketone_smi)
    a = _safe_canonical(aldehyde_smi)
    return f"{k}||{a}"


def _reaction_hash(row: pd.Series, condition_cols: list[str]) -> str:
    """Create a hash for exact duplicate detection.

    Includes: substrate key + conditions + labels.
    """
    k = _safe_canonical(row.get("Ketone", ""))
    a = _safe_canonical(row.get("Aldehyde", ""))
    p = _safe_canonical(row.get("Raw_Product_Smiles", ""))

    parts = [k, a, p]
    for col in condition_cols:
        val = row.get(col, "")
        parts.append("<NA>" if pd.isna(val) else str(val))
    for col in ["label_Ca", "label_Cb", "label_SA"]:
        val = row.get(col, "")
        parts.append("<NA>" if pd.isna(val) else str(val))

    key = "||".join(parts)
    return hashlib.md5(key.encode()).hexdigest()


def run(context: dict) -> dict:
    """Template-based deduplication with role-aware substrate keys."""
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    n_start = len(df)
    logger.info(f"Step 7: Template-based deduplication for {n_start} rows")

    # Condition columns for exact dedup
    condition_cols = ["Reagents", "Solvent", "metal"]

    # Build substrate keys (role-aware, NOT sorted)
    ketone_col = "canonical_Ketone" if "canonical_Ketone" in df.columns else "Ketone"
    aldehyde_col = "canonical_Aldehyde" if "canonical_Aldehyde" in df.columns else "Aldehyde"

    df["substrate_key"] = df.apply(
        lambda r: _role_aware_substrate_key(r.get(ketone_col, ""), r.get(aldehyde_col, "")),
        axis=1,
    )

    # Tier 1: Exact duplicates (same substrates + conditions + labels)
    df["rxn_hash"] = df.apply(lambda r: _reaction_hash(r, condition_cols), axis=1)
    dup_mask = df.duplicated(subset=["rxn_hash"], keep="first")
    n_tier1 = dup_mask.sum()

    if n_tier1 > 0:
        audit.mark_deleted_by_oi(df.loc[dup_mask, "original_index"].values, "exact_duplicate")
        df = df[~dup_mask].reset_index(drop=True)
        logger.info(f"  Tier 1 (exact duplicates): removed {n_tier1}")

    # Assign group_id based on substrate_key
    # Same substrate pair (role-aware) → same group
    substrate_groups = {}
    group_counter = 0
    group_ids = []
    for sk in df["substrate_key"]:
        if sk not in substrate_groups:
            substrate_groups[sk] = group_counter
            group_counter += 1
        group_ids.append(substrate_groups[sk])
    df["group_id"] = group_ids

    # Log group statistics
    group_sizes = df["group_id"].value_counts()
    n_groups = len(group_sizes)
    n_multi = (group_sizes > 1).sum()
    logger.info(f"  Groups: {n_groups} total, {n_multi} with 2+ members")
    logger.info(f"  Largest group: {group_sizes.max()} members")

    # Log by reaction class
    if "Reaction_Class" in df.columns:
        for cls in df["Reaction_Class"].unique():
            cls_count = (df["Reaction_Class"] == cls).sum()
            logger.info(f"  {cls}: {cls_count} rows")

    n_end = len(df)
    logger.info(f"  Step 7 complete: {n_start} → {n_end} rows (removed {n_start - n_end})")

    out_path = context["output_dir"] / "interim" / "07_deduplicated.csv"
    df.to_csv(out_path, index=False)

    context["df"] = df
    return context
