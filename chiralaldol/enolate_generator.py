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

import logging

from rdkit import Chem, RDLogger

from .utils import ACYL_ALPHA_SMARTS, clean_mol

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

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


