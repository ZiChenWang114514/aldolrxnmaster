"""Step 6: Auxiliary chirality extraction via SMARTS substructure matching.

Precisely locate C4 of the Evans oxazolidinone ring and extract its CIP R/S code.
Also classify the R-group type (benzyl, isopropyl, phenyl, etc.).
"""

import logging

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# SMARTS patterns for Evans oxazolidinone C4
# Pattern: C4 is in a 5-membered ring: C4-O-C(=O)-N-C4, with C4 bearing a substituent
OXAZ_PATTERNS = [
    # Standard: [C:1] bonded to (ring-N, ring-CH2-O, substituent, H)
    "[C:1]1([*])COC(=O)N1",
    # Variant without explicit H
    "[C:1]1COC(=O)N1[*]",
    # Broader: any 5-ring with O-C(=O)-N
    "[C:1]1[CH2]OC(=O)N1",
]

# R-group classification SMARTS
RGROUP_SMARTS = {
    "benzyl": "[CH2]c1ccccc1",       # -CH2-Ph
    "isopropyl": "[CH]([CH3])[CH3]",  # -CH(CH3)2
    "phenyl": "c1ccccc1",             # directly bonded Ph (no CH2)
    "tert_butyl": "[C]([CH3])([CH3])[CH3]",
    "methyl": "[CH3]",                # lone methyl
    "indanyl": "C1CCc2ccccc21",       # indane ring
}


def _extract_aux_chirality(ketone_smi: str) -> dict:
    """Extract auxiliary chirality from ketone SMILES.

    Returns dict with: aux_C4_cip, aux_rgroup_type, aux_smarts_match, aux_mw
    """
    result = {
        "aux_C4_cip": None,
        "aux_rgroup_type": "other",
        "aux_smarts_match": False,
        "aux_mw": 0.0,
    }

    if pd.isna(ketone_smi) or not str(ketone_smi).strip():
        return result

    mol = Chem.MolFromSmiles(str(ketone_smi))
    if mol is None:
        return result

    result["aux_mw"] = Descriptors.MolWt(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Try each SMARTS pattern
    c4_idx = None
    for patt_str in OXAZ_PATTERNS:
        patt = Chem.MolFromSmarts(patt_str)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            c4_idx = matches[0][0]  # First atom in pattern = C4
            result["aux_smarts_match"] = True
            break

    if c4_idx is None:
        # Fallback: use first defined stereocenter (legacy method)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        defined = [(idx, cip) for idx, cip in centers if cip in ("R", "S")]
        if defined:
            result["aux_C4_cip"] = defined[0][1]
        return result

    # Get CIP at C4
    atom = mol.GetAtomWithIdx(c4_idx)
    cip = atom.GetPropsAsDict().get("_CIPCode", None)
    result["aux_C4_cip"] = cip

    # Classify R-group: examine neighbors of C4 that are NOT in the ring
    ring_info = mol.GetRingInfo()
    c4_rings = ring_info.AtomRingSizes(c4_idx) if hasattr(ring_info, 'AtomRingSizes') else []

    # Get substituent atoms (neighbors not in the 5-membered ring with C4)
    c4_ring_atoms = set()
    for ring in ring_info.AtomRings():
        if c4_idx in ring:
            c4_ring_atoms.update(ring)

    substituent_root = None
    for nb in atom.GetNeighbors():
        if nb.GetIdx() not in c4_ring_atoms:
            substituent_root = nb.GetIdx()
            break

    if substituent_root is not None:
        # Try R-group SMARTS patterns
        for rgroup_name, rg_smarts in RGROUP_SMARTS.items():
            rg_patt = Chem.MolFromSmarts(rg_smarts)
            if rg_patt is None:
                continue
            rg_matches = mol.GetSubstructMatches(rg_patt)
            for match in rg_matches:
                if substituent_root in match:
                    result["aux_rgroup_type"] = rgroup_name
                    return result

    return result


def run(context: dict) -> dict:
    """Extract auxiliary chirality for all rows."""
    df: pd.DataFrame = context["df"].copy()
    n = len(df)
    logger.info(f"Step 6: Auxiliary chirality extraction for {n} rows")

    ketone_col = "canonical_Ketone" if "canonical_Ketone" in df.columns else "Ketone"

    aux_results = df[ketone_col].apply(_extract_aux_chirality)
    df["aux_C4_cip"] = [r["aux_C4_cip"] for r in aux_results]
    df["aux_rgroup_type"] = [r["aux_rgroup_type"] for r in aux_results]
    df["aux_smarts_match"] = [r["aux_smarts_match"] for r in aux_results]
    df["aux_mw"] = [r["aux_mw"] for r in aux_results]

    # Statistics
    n_match = df["aux_smarts_match"].sum()
    n_cip = df["aux_C4_cip"].notna().sum()
    logger.info(f"  SMARTS match: {n_match}/{n} ({100*n_match/max(n,1):.1f}%)")
    logger.info(f"  CIP extracted: {n_cip}/{n}")

    if n_cip > 0:
        cip_dist = df["aux_C4_cip"].value_counts().to_dict()
        logger.info(f"  CIP distribution: {cip_dist}")

    rgroup_dist = df["aux_rgroup_type"].value_counts().to_dict()
    logger.info(f"  R-group types: {rgroup_dist}")

    # By Reaction_Class
    if "Reaction_Class" in df.columns:
        for cls in ["EvansAux"]:
            cls_mask = df["Reaction_Class"] == cls
            if cls_mask.sum() > 0:
                cls_match = df.loc[cls_mask, "aux_smarts_match"].sum()
                logger.info(f"  {cls}: {cls_match}/{cls_mask.sum()} SMARTS match")

    out_path = context["output_dir"] / "interim" / "06_aux_chirality.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Step 6 complete: {n} rows (no deletions)")

    context["df"] = df
    return context
