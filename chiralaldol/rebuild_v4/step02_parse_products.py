"""Step 02: Parse reaction SMILES, identify main aldol product, extract ketone/aldehyde."""

import logging
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from .audit import AuditTracker
from .constants import (
    ALDOL_PRODUCT_SMARTS_AUX,
    ALDOL_PRODUCT_SMARTS_GENERIC,
    ALDEHYDE_SMARTS,
    DEHYDRATION_SMARTS,
    AUXILIARY_SMARTS,
)
from .utils import safe_mol, split_reaction_smiles, mol_weight

logger = logging.getLogger("rebuild_v4.step02")

# Pre-compile SMARTS
_PAT_AUX = Chem.MolFromSmarts(ALDOL_PRODUCT_SMARTS_AUX)
_PAT_GENERIC = Chem.MolFromSmarts(ALDOL_PRODUCT_SMARTS_GENERIC)
_PAT_DEHYDRATION = Chem.MolFromSmarts(DEHYDRATION_SMARTS)
_PAT_ALDEHYDE = Chem.MolFromSmarts(ALDEHYDE_SMARTS)
_AUX_PATS = {k: Chem.MolFromSmarts(v) for k, v in AUXILIARY_SMARTS.items()}


def _identify_main_product(
    product_smiles_list: list[str],
    reactant_smiles_list: list[str],
    yield_val: Optional[float],
) -> Optional[str]:
    """Multi-layer strategy to identify the main aldol product.

    Priority:
      Layer 1: beta-hydroxy carbonyl substructure match (auxiliary-type first, then generic)
      Layer 2: MW conservation (product MW ~ sum(reactant MW))
      Layer 3: Auxiliary substructure preserved in product
      Layer 4: Yield-aware fallback / max MW
    """
    candidates = []
    for smi in product_smiles_list:
        mol = safe_mol(smi)
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)

        # Check for dehydration product (exclude)
        if _PAT_DEHYDRATION and mol.HasSubstructMatch(_PAT_DEHYDRATION):
            # Check if it also has beta-OH (some products have both motifs)
            if not (_PAT_AUX and mol.HasSubstructMatch(_PAT_AUX)):
                if not (_PAT_GENERIC and mol.HasSubstructMatch(_PAT_GENERIC)):
                    continue

        has_aux_product = _PAT_AUX and mol.HasSubstructMatch(_PAT_AUX)
        has_generic_product = _PAT_GENERIC and mol.HasSubstructMatch(_PAT_GENERIC)
        has_auxiliary = any(pat and mol.HasSubstructMatch(pat) for pat in _AUX_PATS.values())

        candidates.append({
            "smiles": smi,
            "mol": mol,
            "mw": mw,
            "has_aux_product": has_aux_product,
            "has_generic_product": has_generic_product,
            "has_auxiliary": has_auxiliary,
        })

    if not candidates:
        return None

    # Layer 1: auxiliary-type beta-hydroxy match
    aux_matches = [c for c in candidates if c["has_aux_product"]]
    if len(aux_matches) == 1:
        return aux_matches[0]["smiles"]
    if len(aux_matches) > 1:
        # Multiple matches: pick by MW (closest to expected)
        return max(aux_matches, key=lambda c: c["mw"])["smiles"]

    # Layer 1b: generic beta-hydroxy match
    generic_matches = [c for c in candidates if c["has_generic_product"]]
    if len(generic_matches) == 1:
        return generic_matches[0]["smiles"]
    if len(generic_matches) > 1:
        return max(generic_matches, key=lambda c: c["mw"])["smiles"]

    # Layer 2: MW conservation
    reactant_mw = sum(mol_weight(s) for s in reactant_smiles_list)
    if reactant_mw > 0:
        # Aldol product MW should be close to reactant sum (within 20%)
        mw_matches = [c for c in candidates
                      if abs(c["mw"] - reactant_mw) / reactant_mw < 0.20]
        if mw_matches:
            return max(mw_matches, key=lambda c: c["mw"])["smiles"]

    # Layer 3: Product contains auxiliary substructure
    aux_preserved = [c for c in candidates if c["has_auxiliary"]]
    if aux_preserved:
        return max(aux_preserved, key=lambda c: c["mw"])["smiles"]

    # Layer 4: Fallback to largest MW product
    return max(candidates, key=lambda c: c["mw"])["smiles"]


def _classify_reactants(
    reactant_smiles_list: list[str],
) -> tuple[Optional[str], Optional[str]]:
    """Classify reactants into ketone/acyl-auxiliary and aldehyde.

    Returns (ketone_smiles, aldehyde_smiles).
    """
    if len(reactant_smiles_list) < 2:
        return None, None

    aldehydes = []
    ketones = []

    for smi in reactant_smiles_list:
        mol = safe_mol(smi)
        if mol is None:
            continue
        # Check if it's an aldehyde
        if _PAT_ALDEHYDE and mol.HasSubstructMatch(_PAT_ALDEHYDE):
            aldehydes.append(smi)
        else:
            ketones.append(smi)

    # If no clear aldehyde found, try by MW (aldehyde is usually smaller)
    if not aldehydes and len(reactant_smiles_list) >= 2:
        mw_pairs = [(smi, mol_weight(smi)) for smi in reactant_smiles_list]
        mw_pairs.sort(key=lambda x: x[1])
        aldehydes = [mw_pairs[0][0]]
        ketones = [p[0] for p in mw_pairs[1:]]

    # If multiple aldehydes, pick the simpler one (lower MW)
    aldehyde = min(aldehydes, key=lambda s: mol_weight(s)) if aldehydes else None

    # Ketone: pick the one with auxiliary substructure, or largest MW
    if ketones:
        aux_ketones = [s for s in ketones
                       if any(pat and safe_mol(s) and safe_mol(s).HasSubstructMatch(pat)
                              for pat in _AUX_PATS.values())]
        ketone = aux_ketones[0] if aux_ketones else max(ketones, key=lambda s: mol_weight(s))
    else:
        ketone = None

    return ketone, aldehyde


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Parse reaction SMILES, identify main product and classify reactants."""
    logger.info("Step 02: Parsing reaction SMILES and identifying main products...")
    n_start = len(df)

    main_products = []
    ketone_list = []
    aldehyde_list = []
    product_sources = []

    for i, row in df.iterrows():
        rxn_smi = row["Reaction"]
        reactants, products = split_reaction_smiles(rxn_smi)

        if not products:
            main_products.append(None)
            ketone_list.append(None)
            aldehyde_list.append(None)
            product_sources.append("none")
            continue

        yield_val = row.get("Yield (numerical)")
        main_prod = _identify_main_product(products, reactants, yield_val)
        main_products.append(main_prod)
        product_sources.append("reaction_smiles")

        ketone, aldehyde = _classify_reactants(reactants)
        ketone_list.append(ketone)
        aldehyde_list.append(aldehyde)

    df["main_product_smiles"] = main_products
    df["ketone_smiles"] = ketone_list
    df["aldehyde_smiles"] = aldehyde_list
    df["product_source"] = product_sources

    # Drop rows where main product could not be identified
    no_product = df["main_product_smiles"].isna()
    audit.record_drop("02_parse_products", df.loc[no_product, "_orig_idx"], "no_main_product")
    df = df[~no_product].reset_index(drop=True)

    # Drop rows where product cannot be parsed by RDKit
    bad_parse = df["main_product_smiles"].apply(lambda s: safe_mol(s) is None)
    audit.record_drop("02_parse_products", df.loc[bad_parse, "_orig_idx"], "product_unparseable")
    df = df[~bad_parse].reset_index(drop=True)

    audit.record_step("02_parse_products", len(df))
    logger.info(f"  Step 02 complete: {n_start} -> {len(df)} rows")
    return df
