"""Step 03: Detect chiral auxiliary type and exclude chiral catalysis."""

import logging

import pandas as pd
from rdkit import Chem

from .audit import AuditTracker
from .constants import (
    AUXILIARY_SMARTS,
    GENERIC_AUXILIARY_SMARTS,
    CHIRAL_CATALYST_KEYWORDS,
    CHIRAL_CATALYSIS_NAMED_REACTIONS,
    YNAMIDE_EXCLUDE_SMARTS,
)
from .utils import safe_mol

logger = logging.getLogger("rebuild_v4.step03")

# Pre-compile SMARTS
_AUX_PATS = {k: Chem.MolFromSmarts(v) for k, v in AUXILIARY_SMARTS.items()}
_GENERIC_PATS = [Chem.MolFromSmarts(s) for s in GENERIC_AUXILIARY_SMARTS]
_YNAMIDE_PAT = Chem.MolFromSmarts(YNAMIDE_EXCLUDE_SMARTS)

# Cyclic N-acyl auxiliaries: safe to check both ketone and product sides
_CYCLIC_TYPES = frozenset({"evans", "crimmins_thione", "crimmins_oxathione", "oppolzer", "super_quat"})

# Ester/acyclic auxiliaries: detect on reactant side only (like Myers)
_ESTER_AUX_TYPES = {"menthyl_ester", "borneol_ester", "abiko", "oxazoline"}
_REACTANT_ONLY_TYPES = {"myers"} | _ESTER_AUX_TYPES


def _detect_auxiliary(ketone_smi: str, product_smi: str, rxn_smi: str) -> str:
    """Detect auxiliary type from ketone, product, and full reaction SMILES.

    For cyclic auxiliaries (Evans/Crimmins/Oppolzer/SuperQuat): check both ketone and product.
    For acyclic auxiliaries (Myers): check ONLY the reactant side to avoid false positives
    (aldol products with PhCH(OH) from benzaldehyde would match otherwise).

    Returns auxiliary type string or 'none'.
    """
    # Cyclic N-acyl auxiliaries: safe to check both ketone and product
    for smi in [ketone_smi, product_smi]:
        mol = safe_mol(smi)
        if mol is None:
            continue
        for aux_type, pat in _AUX_PATS.items():
            if aux_type not in _CYCLIC_TYPES:
                continue
            if pat and mol.HasSubstructMatch(pat):
                return aux_type

    # Acyclic/ester auxiliaries: check ONLY reactant side
    # (Myers, Abiko, menthyl_ester, borneol_ester, oxazoline)
    ketone_mol = safe_mol(ketone_smi)
    if ketone_mol is not None:
        for aux_type, pat in _AUX_PATS.items():
            if aux_type not in _REACTANT_ONLY_TYPES:
                continue
            if pat and ketone_mol.HasSubstructMatch(pat):
                return aux_type

    # Also check all reactants from the full reaction SMILES
    if isinstance(rxn_smi, str) and ">>" in rxn_smi:
        reactant_str = rxn_smi.split(">>")[0]
        for r_smi in reactant_str.split("."):
            mol = safe_mol(r_smi.strip())
            if mol is None:
                continue
            for aux_type, pat in _AUX_PATS.items():
                if aux_type not in _REACTANT_ONLY_TYPES:
                    continue
                if pat and mol.HasSubstructMatch(pat):
                    return aux_type

    # Generic patterns: check both sides
    for smi in [ketone_smi, product_smi]:
        mol = safe_mol(smi)
        if mol is None:
            continue
        for pat in _GENERIC_PATS:
            if pat and mol.HasSubstructMatch(pat):
                # Exclude ynamide (keteniminium mechanism)
                if _YNAMIDE_PAT and mol.HasSubstructMatch(_YNAMIDE_PAT):
                    return "ynamide_excluded"
                return "other_auxiliary"

    return "none"


def _has_chiral_catalyst(catalyst_str: str, reagent_str: str, named_rxn_str: str) -> bool:
    """Check if reaction uses a chiral catalyst (should be excluded).

    NOTE: Pass named_rxn_str="" when calling on rows that already have a detected
    structural auxiliary, to avoid over-exclusion of Mukaiyama-named substrate-
    controlled reactions.
    """
    combined = " ".join([
        str(catalyst_str) if pd.notna(catalyst_str) else "",
        str(reagent_str) if pd.notna(reagent_str) else "",
    ]).lower()

    for kw in CHIRAL_CATALYST_KEYWORDS:
        if kw in combined:
            return True

    named = str(named_rxn_str).lower() if pd.notna(named_rxn_str) else ""
    for kw in CHIRAL_CATALYSIS_NAMED_REACTIONS:
        if kw in named:
            return True

    return False


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Detect auxiliary types and filter to substrate-controlled aldol only."""
    logger.info("Step 03: Detecting chiral auxiliaries and excluding chiral catalysis...")
    n_start = len(df)

    # Detect auxiliary type
    aux_types = []
    for _, row in df.iterrows():
        ketone = row.get("ketone_smiles", "")
        product = row.get("main_product_smiles", "")
        rxn = row.get("Reaction", "")
        aux_types.append(_detect_auxiliary(
            str(ketone) if pd.notna(ketone) else "",
            str(product) if pd.notna(product) else "",
            str(rxn) if pd.notna(rxn) else "",
        ))
    df["auxiliary_type"] = aux_types

    # Log distribution before filtering
    logger.info(f"  Auxiliary type distribution (before filtering):")
    for atype, count in df["auxiliary_type"].value_counts().items():
        logger.info(f"    {atype}: {count}")

    # --- Filter 1: Must have a chiral auxiliary ---
    no_aux = df["auxiliary_type"] == "none"
    audit.record_drop("03_auxiliary_detect", df.loc[no_aux, "_orig_idx"], "no_chiral_auxiliary")
    df = df[~no_aux].reset_index(drop=True)
    logger.info(f"  After auxiliary filter: {len(df)} rows")

    # --- Filter 1b (V5): Exclude ynamide reactions (keteniminium mechanism) ---
    is_ynamide = df["auxiliary_type"] == "ynamide_excluded"
    if is_ynamide.sum() > 0:
        audit.record_drop("03_auxiliary_detect", df.loc[is_ynamide, "_orig_idx"], "ynamide_excluded")
        df = df[~is_ynamide].reset_index(drop=True)
        logger.info(f"  After ynamide exclusion: {len(df)} rows")

    # --- Filter 2: Exclude chiral catalysis ---
    catalyst_col = "Catalyst" if "Catalyst" in df.columns else None
    reagent_col = "Reagent" if "Reagent" in df.columns else None
    named_col = "Named Reaction" if "Named Reaction" in df.columns else None

    is_chiral_cat = df.apply(
        lambda row: _has_chiral_catalyst(
            row.get(catalyst_col, "") if catalyst_col else "",
            row.get(reagent_col, "") if reagent_col else "",
            "",  # 已有辅基结构证据；不再用命名反应排除（防止辅基控制的 Mukaiyama aldol 被误删）
        ),
        axis=1,
    )
    audit.record_drop("03_auxiliary_detect", df.loc[is_chiral_cat, "_orig_idx"], "chiral_catalysis_excluded")
    df = df[~is_chiral_cat].reset_index(drop=True)
    logger.info(f"  After chiral catalysis exclusion: {len(df)} rows")

    # Log final distribution
    logger.info(f"  Final auxiliary type distribution:")
    for atype, count in df["auxiliary_type"].value_counts().items():
        logger.info(f"    {atype}: {count}")

    audit.record_step("03_auxiliary_detect", len(df))
    logger.info(f"  Step 03 complete: {n_start} -> {len(df)} rows")
    return df
