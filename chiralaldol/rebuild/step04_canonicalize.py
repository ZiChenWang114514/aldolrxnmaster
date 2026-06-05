"""Step 04: Canonical SMILES with stereo preservation."""

import logging

import pandas as pd

from .audit import AuditTracker
from .utils import safe_mol, canonical_smiles, count_defined_stereocenters

logger = logging.getLogger("rebuild_v4.step04")

SMILES_COLS = ["main_product_smiles", "ketone_smiles", "aldehyde_smiles"]


def _canonicalize_with_stereo_check(smiles: str) -> tuple[str, bool, bool]:
    """Canonicalize SMILES, check stereo preservation.

    Returns (canonical_smi, parseable, stereo_ok).
    If canonicalization loses stereocenters, returns original SMILES.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return smiles, False, False

    mol_orig = safe_mol(smiles)
    if mol_orig is None:
        return smiles, False, False

    n_orig = count_defined_stereocenters(mol_orig)
    canon = canonical_smiles(smiles, isomeric=True)
    if canon is None:
        return smiles, True, True  # parseable but can't canonicalize -> keep original

    mol_canon = safe_mol(canon)
    if mol_canon is None:
        return smiles, True, False

    n_canon = count_defined_stereocenters(mol_canon)
    if n_canon < n_orig:
        # Canonicalization lost stereocenters -> keep original
        return smiles, True, False
    return canon, True, True


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Canonicalize SMILES columns with stereo preservation."""
    logger.info("Step 04: Canonicalizing SMILES with stereo preservation...")
    n_start = len(df)

    for col in SMILES_COLS:
        if col not in df.columns:
            continue
        results = df[col].apply(_canonicalize_with_stereo_check)
        df[f"canonical_{col}"] = results.apply(lambda x: x[0])
        df[f"{col}_parseable"] = results.apply(lambda x: x[1])
        df[f"{col}_stereo_ok"] = results.apply(lambda x: x[2])

    # Drop rows where main product is completely unparseable
    prod_col = "main_product_smiles"
    if f"{prod_col}_parseable" in df.columns:
        bad = ~df[f"{prod_col}_parseable"]
        audit.record_drop("04_canonicalize", df.loc[bad, "_orig_idx"], "product_unparseable")
        df = df[~bad].reset_index(drop=True)

    audit.record_step("04_canonicalize", len(df))
    logger.info(f"  Step 04 complete: {n_start} -> {len(df)} rows")
    return df
