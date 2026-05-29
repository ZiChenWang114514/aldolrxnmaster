"""Step 01: Load raw Reaxys CSV and filter to aldol reactions.

Two-pass filter:
  Pass 1 (keyword): Reaction Type / Named Reaction contains 'aldol' etc.
  Pass 2 (structure): Reaction SMILES contains chiral auxiliary SMARTS patterns.
  Union of both passes is kept.
"""

import logging

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from .constants import RAW_CSV, REAXYS_COLS, AUXILIARY_SMARTS
from .audit import AuditTracker
from .utils import safe_mol

logger = logging.getLogger("rebuild_v4.step01")

# Pre-compile auxiliary SMARTS for structural filtering
_AUX_PATS = {k: Chem.MolFromSmarts(v) for k, v in AUXILIARY_SMARTS.items()}


def _has_auxiliary_in_reactants(rxn_smi: str) -> bool:
    """Check if any reactant in the reaction SMILES matches an auxiliary pattern."""
    if not isinstance(rxn_smi, str) or ">>" not in rxn_smi:
        return False
    reactant_str = rxn_smi.split(">>")[0]
    for r_smi in reactant_str.split("."):
        mol = safe_mol(r_smi.strip())
        if mol is None:
            continue
        for pat in _AUX_PATS.values():
            if pat and mol.HasSubstructMatch(pat):
                return True
    return False


def run(audit: AuditTracker) -> pd.DataFrame:
    """Load data.csv, filter to aldol-type reactions with preparation data."""
    logger.info(f"Loading {RAW_CSV} ...")
    # Only read columns that exist
    all_cols = pd.read_csv(RAW_CSV, nrows=0).columns.tolist()
    use_cols = [c for c in REAXYS_COLS if c in all_cols]
    df = pd.read_csv(RAW_CSV, usecols=use_cols, low_memory=False)
    df["_orig_idx"] = range(len(df))
    logger.info(f"  Loaded {len(df)} rows, {df['Reaction ID'].nunique()} unique IDs")

    n_start = len(df)

    # --- Filter 1: Must have Reaction SMILES with >> ---
    has_smiles = (
        df["Reaction"].notna()
        & (df["Reaction"].str.strip() != "")
        & df["Reaction"].str.contains(">>", na=False)
    )
    drop_no_smi = df.index[~has_smiles]
    audit.record_drop("01_load_filter", df.loc[drop_no_smi, "_orig_idx"], "missing_or_invalid_smiles")
    df = df[has_smiles].copy()
    logger.info(f"  After SMILES filter: {len(df)} rows")

    # --- Filter 2: Record Type must contain "preparation" ---
    if "Record Type" in df.columns:
        has_prep = df["Record Type"].fillna("").str.lower().str.contains("preparation", na=False)
        drop_no_prep = df.index[~has_prep]
        audit.record_drop("01_load_filter", df.loc[drop_no_prep, "_orig_idx"], "no_preparation_record")
        df = df[has_prep].copy()
        logger.info(f"  After preparation filter: {len(df)} rows")

    # --- Filter 3: Two-pass aldol/auxiliary filter ---
    # Pass 1: Keyword-based
    rxn_type = df["Reaction Type"].fillna("").str.lower()
    named_rxn = df["Named Reaction"].fillna("").str.lower() if "Named Reaction" in df.columns else pd.Series("", index=df.index)
    other_cond = df["Other Conditions"].fillna("").str.lower() if "Other Conditions" in df.columns else pd.Series("", index=df.index)

    aldol_keywords = ["aldol", "claisen-schmidt", "claisen schmidt", "evans"]
    is_aldol_keyword = pd.Series(False, index=df.index)
    for kw in aldol_keywords:
        is_aldol_keyword |= rxn_type.str.contains(kw, na=False)
        is_aldol_keyword |= named_rxn.str.contains(kw, na=False)

    n_keyword = is_aldol_keyword.sum()
    logger.info(f"  Pass 1 (keyword): {n_keyword} rows match aldol keywords")

    # Pass 2: Structural — check reactants for auxiliary SMARTS
    # Only check rows NOT already matched by keywords (for efficiency)
    remaining = df[~is_aldol_keyword].copy()
    logger.info(f"  Pass 2 (structure): checking {len(remaining)} unmatched rows for auxiliary SMARTS...")

    # Suppress RDKit warnings during bulk SMILES parsing
    rdlog = RDLogger.logger()
    rdlog.setLevel(RDLogger.ERROR)

    has_aux = remaining["Reaction"].apply(_has_auxiliary_in_reactants)

    rdlog.setLevel(RDLogger.WARNING)

    is_aux_structure = pd.Series(False, index=df.index)
    is_aux_structure.loc[remaining.index[has_aux]] = True

    n_structural = is_aux_structure.sum()
    logger.info(f"  Pass 2 (structure): {n_structural} additional rows match auxiliary SMARTS")

    # Union of both passes
    is_relevant = is_aldol_keyword | is_aux_structure
    drop_irrelevant = df.index[~is_relevant]
    audit.record_drop("01_load_filter", df.loc[drop_irrelevant, "_orig_idx"], "not_aldol_or_auxiliary")
    df = df[is_relevant].copy()
    logger.info(f"  After combined filter: {len(df)} rows ({n_keyword} keyword + {n_structural} structural)")

    df = df.reset_index(drop=True)
    audit.record_step("01_load_filter", len(df))
    logger.info(f"  Step 01 complete: {n_start} -> {len(df)} rows")
    return df
