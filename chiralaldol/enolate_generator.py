"""M1: Enolate Generator — Convert Evans auxiliary ketone to enolate intermediate.

Chemistry:
  Evans N-acyl oxazolidinone + base → metal enolate (Z or E)
  The alpha-CH2 adjacent to C(=O)-N is deprotonated:
    R-CH2-C(=O)-N-[Oxaz]  →  R-CH=C(O⁻)-N-[Oxaz]

  Z/E selectivity is controlled by the base:
    Bu2BOTf/DIPEA → >98% Z-enolate (standard Evans conditions)
    LDA           → ~95% Z-enolate
    TiCl4/DIPEA   → Z-enolate (chelation control)
    Et3N          → lower selectivity, mix of Z/E

  We generate the enolate without Z/E specification in SMILES and let
  the conformer generator produce both geometries. The base-dependent
  weighting is applied at the descriptor aggregation stage.
"""

import ast
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .utils import clean_mol

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# SMARTS: alpha-C (sp3, with H's) bonded to acyl C(=O) bonded to amide N
# This specifically matches the N-acyl chain, NOT the ring C(=O)-O
# X3 covers terminal CH2 (1 heavy neighbor), X4 covers substituted CH/CH2
ACYL_ALPHA_SMARTS = Chem.MolFromSmarts("[CH2,CH;X3,X4:1]-[CX3:2](=[OX1:3])-[NX3:4]")

# Base classification for Z/E selectivity
# Z-dominant: essentially all Evans standard conditions
Z_DOMINANT = {"DIPEA", "LDA", "LiHMDS", "NaHMDS", "KHMDS"}
# Bases where selectivity is lower or E may compete
MIXED_ZE = {"Et3N", "other_base"}


def ketone_to_enolate(smiles: str) -> tuple[str | None, str]:
    """Convert a single Evans ketone SMILES to enolate SMILES.

    Returns:
        (enolate_smiles, status): status is 'success', 'no_match', or error description
    """
    mol = clean_mol(smiles)
    if mol is None:
        return None, "parse_fail"

    matches = mol.GetSubstructMatches(ACYL_ALPHA_SMARTS)
    if not matches:
        return None, "no_match"

    # Use first match (Evans aux should have exactly one acyl alpha position)
    alpha_idx, carbonyl_c_idx, carbonyl_o_idx, n_idx = matches[0]

    try:
        rwmol = Chem.RWMol(mol)

        # Change alpha_C — carbonyl_C bond: single → double
        bond_cc = rwmol.GetBondBetweenAtoms(alpha_idx, carbonyl_c_idx)
        bond_cc.SetBondType(Chem.BondType.DOUBLE)

        # Change carbonyl_C=O bond: double → single
        bond_co = rwmol.GetBondBetweenAtoms(carbonyl_c_idx, carbonyl_o_idx)
        bond_co.SetBondType(Chem.BondType.SINGLE)

        # Set formal charge on O to -1 (enolate oxygen)
        rwmol.GetAtomWithIdx(carbonyl_o_idx).SetFormalCharge(-1)

        # Fix explicit Hs on alpha carbon: sp3→sp2 loses one H
        alpha_atom = rwmol.GetAtomWithIdx(alpha_idx)
        expl_h = alpha_atom.GetNumExplicitHs()
        if expl_h > 0:
            alpha_atom.SetNumExplicitHs(expl_h - 1)
        alpha_atom.SetNoImplicit(False)

        # Sanitize — RDKit recalculates implicit Hs from valence
        Chem.SanitizeMol(rwmol)
        enolate_mol = rwmol.GetMol()

        # Validate: check the new double bond exists and O has charge
        bond_check = enolate_mol.GetBondBetweenAtoms(alpha_idx, carbonyl_c_idx)
        if bond_check.GetBondType() != Chem.BondType.DOUBLE:
            return None, "bond_type_error"

        enolate_smi = Chem.MolToSmiles(enolate_mol)

        # Verify round-trip
        check = Chem.MolFromSmiles(enolate_smi)
        if check is None:
            return None, "roundtrip_fail"

        return enolate_smi, "success"

    except Exception as e:
        return None, f"error: {e}"


def classify_ze_selectivity(base_str: str) -> str:
    """Classify Z/E selectivity based on base identity.

    Returns: 'Z_dominant', 'mixed', or 'unknown'
    """
    if not base_str or str(base_str) == "nan":
        return "unknown"

    base = str(base_str).strip()

    # Exact match against known categories (case-insensitive)
    for z_base in Z_DOMINANT:
        if base.upper() == z_base.upper():
            return "Z_dominant"
    for m_base in MIXED_ZE:
        if base.upper() == m_base.upper():
            return "mixed"

    return "unknown"


def get_dominant_base(reagents_str: str, base_cols: dict) -> str:
    """Extract dominant base from reaction conditions.

    Args:
        reagents_str: raw reagent string from dataset
        base_cols: dict of base column values from reaction_conditions.csv

    Returns: base name string
    """
    # Check one-hot encoded base columns
    for base_name, value in base_cols.items():
        if value > 0 and base_name != "no_base":
            return base_name.replace("base_", "")
    return "unknown"


def generate_all_enolates(project_dir: Path) -> pd.DataFrame:
    """Generate enolates for all 1822 Evans reactions.

    Returns DataFrame with columns:
        idx, ketone_smiles, enolate_smiles, status, ze_selectivity
    """
    processed = project_dir / "data" / "processed"
    df = pd.read_csv(processed / "evans_clean.csv")
    cond = pd.read_csv(processed / "features" / "reaction_conditions.csv")

    # Identify base columns
    base_cols = [c for c in cond.columns if c.startswith("base_")]

    n = len(df)
    results = []
    n_success = 0
    n_fallback = 0

    for i in range(n):
        ketone_smi = str(df["Ketone"].iloc[i])
        enolate_smi, status = ketone_to_enolate(ketone_smi)

        # Determine Z/E selectivity from base encoding
        base_dict = {c: cond[c].iloc[i] for c in base_cols}
        dominant_base = get_dominant_base(None, base_dict)
        ze_sel = classify_ze_selectivity(dominant_base)

        if enolate_smi is None:
            # Fallback: use original ketone (cleaned)
            mol = clean_mol(ketone_smi)
            fallback_smi = Chem.MolToSmiles(mol) if mol else ""
            results.append({
                "idx": i,
                "ketone_smiles": ketone_smi,
                "enolate_smiles": fallback_smi,
                "status": f"fallback({status})",
                "ze_selectivity": ze_sel,
                "is_enolate": False,
            })
            n_fallback += 1
        else:
            results.append({
                "idx": i,
                "ketone_smiles": ketone_smi,
                "enolate_smiles": enolate_smi,
                "status": status,
                "ze_selectivity": ze_sel,
                "is_enolate": True,
            })
            n_success += 1

    logger.info(f"Enolate generation: {n_success}/{n} success, {n_fallback} fallback")

    result_df = pd.DataFrame(results)
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    project = Path("/data2/zcwang/aldolrxnmaster")
    out = generate_all_enolates(project)
    out_path = project / "data" / "processed" / "chiralaldol" / "enolates.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out)} enolates to {out_path}")
    logger.info(f"Status counts:\n{out['status'].value_counts()}")
