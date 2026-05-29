"""Step 2: SMILES strict isomeric canonicalization with stereo preservation check."""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

SMILES_COLS = ["Raw_Product_Smiles", "Ketone", "Aldehyde"]


def _count_stereocenters(mol) -> int:
    """Count defined stereocenters in an RDKit mol object."""
    if mol is None:
        return -1
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return sum(1 for _, cip in centers if cip in ("R", "S"))


def _safe_canonical(smi: str) -> tuple[str | None, bool, bool]:
    """Canonicalize SMILES with stereo preservation check.

    Returns: (canonical_smi, parseable, stereo_preserved)
    """
    if pd.isna(smi) or not str(smi).strip():
        return None, False, False

    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, False, False

    n_stereo_orig = _count_stereocenters(mol)

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

    # Re-parse to verify stereo preservation
    mol2 = Chem.MolFromSmiles(canonical)
    if mol2 is None:
        return str(smi), True, False  # Keep original if re-parse fails

    n_stereo_canon = _count_stereocenters(mol2)
    stereo_ok = (n_stereo_canon >= n_stereo_orig) or n_stereo_orig == 0

    if not stereo_ok:
        return str(smi), True, False  # Keep original SMILES if stereo lost

    return canonical, True, True


def run(context: dict) -> dict:
    """Canonicalize all SMILES columns with strict stereo preservation."""
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    n_start = len(df)
    logger.info(f"Step 2: Canonicalizing SMILES for {n_start} rows")

    for col in SMILES_COLS:
        if col not in df.columns:
            logger.warning(f"  Column {col} not found, skipping")
            continue

        canonical_col = f"canonical_{col}"
        parse_col = f"{col}_parseable"
        stereo_col = f"{col}_stereo_ok"

        results = df[col].apply(_safe_canonical)
        df[canonical_col] = [r[0] for r in results]
        df[parse_col] = [r[1] for r in results]
        df[stereo_col] = [r[2] for r in results]

        n_parse = df[parse_col].sum()
        n_stereo = df[stereo_col].sum()
        n_lost = n_parse - n_stereo
        logger.info(f"  {col}: {n_parse}/{n_start} parseable, {n_lost} stereo lost (kept original)")

    # Also canonicalize Raw_Reaction_Smiles
    if "Raw_Reaction_Smiles" in df.columns:
        canonical_rxn = []
        rxn_parseable = []
        for smi in df["Raw_Reaction_Smiles"]:
            if pd.isna(smi) or not str(smi).strip():
                canonical_rxn.append(None)
                rxn_parseable.append(False)
                continue
            smi_str = str(smi)
            if ">>" not in smi_str:
                canonical_rxn.append(smi_str)
                rxn_parseable.append(True)
                continue
            parts = smi_str.split(">>")
            reactants_str = parts[0]
            products_str = parts[1] if len(parts) > 1 else ""
            # Canonicalize each component but do NOT sort reactants (preserve role)
            r_mols = [Chem.MolFromSmiles(s) for s in reactants_str.split(".")]
            p_mols = [Chem.MolFromSmiles(s) for s in products_str.split(".")]
            r_can = []
            for m in r_mols:
                if m is not None:
                    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
                    r_can.append(Chem.MolToSmiles(m, isomericSmiles=True))
                else:
                    r_can.append("")
            p_can = []
            for m in p_mols:
                if m is not None:
                    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
                    p_can.append(Chem.MolToSmiles(m, isomericSmiles=True))
                else:
                    p_can.append("")
            canonical_rxn.append(".".join(r_can) + ">>" + ".".join(p_can))
            rxn_parseable.append(True)
        df["canonical_Raw_Reaction_Smiles"] = canonical_rxn
        df["Raw_Reaction_Smiles_parseable"] = rxn_parseable

    # Delete rows where product is unparseable
    product_parse_col = "Raw_Product_Smiles_parseable"
    if product_parse_col in df.columns:
        bad_mask = ~df[product_parse_col]
        n_bad = bad_mask.sum()
        if n_bad > 0:
            audit.mark_deleted_by_oi(df.loc[bad_mask, "original_index"].values, "unparseable_product")
            df = df[~bad_mask].reset_index(drop=True)
            logger.info(f"  Deleted {n_bad} rows with unparseable product SMILES")

    n_end = len(df)
    logger.info(f"  Step 2 complete: {n_start} → {n_end} rows")

    out_path = context["output_dir"] / "interim" / "02_canonical.csv"
    df.to_csv(out_path, index=False)

    context["df"] = df
    return context
