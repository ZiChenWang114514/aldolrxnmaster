"""StereoEnumerator: Generate all 4 stereoisomeric aldol products by flipping Ca/Cb chiral centers.

For each Evans aldol reaction, the product has exactly 2 new stereocenters (Ca, Cb).
This module enumerates all 4 possible (Ca_config, Cb_config) combinations:
  candidate 0: (R, R)
  candidate 1: (R, S)
  candidate 2: (S, R)
  candidate 3: (S, S)

Ground truth is identified by matching against the experimentally observed product.
"""

import logging
from itertools import product as iterproduct

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

logger = logging.getLogger(__name__)

# SMARTS patterns for Evans aldol product core
# :1 = Cb (OH-bearing carbon), :2 = Ca (adjacent to C(=O)N)
EVANS_PRODUCT_SMARTS = [
    "[C:1]([OH1])([#6])[C:2]([#6])[C](=O)[N]",
    "[C:1]([OH1])[C:2][C](=O)[N]",
]


def _find_ca_cb_atoms(mol):
    """Find Ca and Cb atom indices in an aldol product molecule.

    Returns:
        tuple (ca_idx, cb_idx) or None if not found.
        Ca = alpha-carbon (adjacent to C(=O)N)
        Cb = beta-carbon (OH-bearing)
    """
    for smarts_str in EVANS_PRODUCT_SMARTS:
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue
        # Build map: atom_map_num -> SMARTS atom index
        map_to_smarts_idx = {}
        for atom in pattern.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                map_to_smarts_idx[map_num] = atom.GetIdx()

        matches = mol.GetSubstructMatches(pattern)
        if matches and 1 in map_to_smarts_idx and 2 in map_to_smarts_idx:
            # :1 = Cb (OH-bearing), :2 = Ca (adjacent to C(=O)N)
            cb_idx = matches[0][map_to_smarts_idx[1]]
            ca_idx = matches[0][map_to_smarts_idx[2]]
            return ca_idx, cb_idx
    return None


def _find_ca_cb_by_atom_map(mol, mapped_product_smi):
    """Find Ca/Cb using atom mapping from the mapped product SMILES.

    In the dataset, Ca is the carbon that forms the new C-C bond on the ketone side,
    and Cb is the carbon that forms the new C-C bond on the aldehyde side (has OH).

    Falls back to SMARTS if atom mapping is unavailable.
    """
    # Try SMARTS-based identification first (more reliable)
    result = _find_ca_cb_atoms(mol)
    if result is not None:
        return result

    # If SMARTS fails, try atom-map based approach
    if mapped_product_smi and pd.notna(mapped_product_smi):
        mapped_mol = Chem.MolFromSmiles(mapped_product_smi)
        if mapped_mol is not None:
            result = _find_ca_cb_atoms(mapped_mol)
            if result is not None:
                return result

    return None


def _get_cip_code(mol, atom_idx):
    """Get CIP R/S code for a specific atom."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.HasProp("_CIPCode"):
        return atom.GetProp("_CIPCode")
    return None


def _set_chirality(mol, atom_idx, target_cip):
    """Set the chirality of an atom to a target CIP code (R or S).

    Strategy: check current CIP. If it matches target, do nothing.
    If it differs, invert the chiral tag.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atom = mol.GetAtomWithIdx(atom_idx)
    current_cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None

    if current_cip == target_cip:
        return True  # Already correct

    # Invert chirality
    current_tag = atom.GetChiralTag()
    if current_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif current_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    else:
        # No chiral tag set — try to set one
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    # Verify
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    new_cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None

    if new_cip == target_cip:
        return True

    # If still wrong after inversion, try the other tag
    if new_cip != target_cip:
        atom = mol.GetAtomWithIdx(atom_idx)
        current_tag = atom.GetChiralTag()
        if current_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        else:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        new_cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None

    return new_cip == target_cip


def _try_recover_cip_from_mapped(mapped_product_smi, ca_idx_in_canon, cb_idx_in_canon,
                                    canonical_mol):
    """Try to recover missing CIP codes by using atom-mapped product SMILES.

    The mapped SMILES often has explicit stereo even when canonical loses it.
    We match atoms by SMARTS pattern in the mapped mol.
    """
    if not mapped_product_smi or not isinstance(mapped_product_smi, str):
        return None, None

    mapped_mol = Chem.MolFromSmiles(mapped_product_smi)
    if mapped_mol is None:
        return None, None

    result = _find_ca_cb_atoms(mapped_mol)
    if result is None:
        return None, None

    ca_idx_m, cb_idx_m = result
    Chem.AssignStereochemistry(mapped_mol, cleanIt=True, force=True)
    ca_atom = mapped_mol.GetAtomWithIdx(ca_idx_m)
    cb_atom = mapped_mol.GetAtomWithIdx(cb_idx_m)
    ca_cip = ca_atom.GetProp("_CIPCode") if ca_atom.HasProp("_CIPCode") else None
    cb_cip = cb_atom.GetProp("_CIPCode") if cb_atom.HasProp("_CIPCode") else None
    return ca_cip, cb_cip


def enumerate_stereoisomers(product_smi, mapped_product_smi=None):
    """Enumerate all 4 stereoisomers of an aldol product by flipping Ca/Cb configurations.

    Args:
        product_smi: Canonical product SMILES (with defined stereo at Ca, Cb)
        mapped_product_smi: Optional atom-mapped product SMILES for fallback identification

    Returns:
        dict with keys:
            'candidates': list of 4 SMILES (ordered by candidate_id: RR, RS, SR, SS)
            'ca_idx': atom index of Ca
            'cb_idx': atom index of Cb
            'true_candidate_id': which candidate matches the input (ground truth)
            'success': bool
        or None on failure
    """
    mol = Chem.MolFromSmiles(product_smi)
    if mol is None:
        return None

    # Find Ca and Cb
    result = _find_ca_cb_by_atom_map(mol, mapped_product_smi)
    if result is None:
        return None
    ca_idx, cb_idx = result

    # Get current CIP codes
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    ca_atom = mol.GetAtomWithIdx(ca_idx)
    cb_atom = mol.GetAtomWithIdx(cb_idx)
    orig_ca_cip = ca_atom.GetProp("_CIPCode") if ca_atom.HasProp("_CIPCode") else None
    orig_cb_cip = cb_atom.GetProp("_CIPCode") if cb_atom.HasProp("_CIPCode") else None

    # If CIP missing from canonical, try mapped product
    if orig_ca_cip is None or orig_cb_cip is None:
        map_ca_cip, map_cb_cip = _try_recover_cip_from_mapped(
            mapped_product_smi, ca_idx, cb_idx, mol
        )
        if orig_ca_cip is None and map_ca_cip is not None:
            orig_ca_cip = map_ca_cip
            # Also set the chiral tag on the canonical mol so enumeration works
            _set_chirality(mol, ca_idx, orig_ca_cip)
        if orig_cb_cip is None and map_cb_cip is not None:
            orig_cb_cip = map_cb_cip
            _set_chirality(mol, cb_idx, orig_cb_cip)
        # Update product_smi to reflect recovered stereo
        if orig_ca_cip is not None and orig_cb_cip is not None:
            product_smi = Chem.MolToSmiles(mol)

    if orig_ca_cip is None or orig_cb_cip is None:
        return None

    # Enumerate all 4 combinations
    configs = [("R", "R"), ("R", "S"), ("S", "R"), ("S", "S")]
    candidates = []
    true_candidate_id = None

    for cand_id, (ca_config, cb_config) in enumerate(configs):
        # Work on a fresh copy each time
        cand_mol = Chem.RWMol(Chem.MolFromSmiles(product_smi))
        if cand_mol is None:
            return None

        # Set Ca chirality
        ok_ca = _set_chirality(cand_mol, ca_idx, ca_config)
        # Set Cb chirality
        ok_cb = _set_chirality(cand_mol, cb_idx, cb_config)

        if not (ok_ca and ok_cb):
            # Fallback: try sanitize and retry
            try:
                Chem.SanitizeMol(cand_mol)
                ok_ca = _set_chirality(cand_mol, ca_idx, ca_config)
                ok_cb = _set_chirality(cand_mol, cb_idx, cb_config)
            except Exception:
                pass

            if not (ok_ca and ok_cb):
                return None

        cand_smi = Chem.MolToSmiles(cand_mol)
        candidates.append(cand_smi)

        # Check if this matches the original
        if ca_config == orig_ca_cip and cb_config == orig_cb_cip:
            true_candidate_id = cand_id

    # Verify all 4 are distinct
    unique_smiles = set(candidates)
    if len(unique_smiles) < 4:
        # Some candidates collapsed — likely a symmetry issue
        logger.warning(
            f"Only {len(unique_smiles)} unique candidates for {product_smi}"
        )
        # Still proceed if we can identify the true product
        if true_candidate_id is None:
            return None

    if true_candidate_id is None:
        # Try to match by canonicalization
        canon_orig = Chem.MolToSmiles(Chem.MolFromSmiles(product_smi))
        for i, cand in enumerate(candidates):
            if Chem.MolToSmiles(Chem.MolFromSmiles(cand)) == canon_orig:
                true_candidate_id = i
                break

    if true_candidate_id is None:
        return None

    return {
        "candidates": candidates,
        "ca_idx": ca_idx,
        "cb_idx": cb_idx,
        "true_candidate_id": true_candidate_id,
        "orig_ca_cip": orig_ca_cip,
        "orig_cb_cip": orig_cb_cip,
        "success": True,
    }


def enumerate_dataset(df, product_col="canonical_Raw_Product_Smiles",
                      mapped_col="Mapped_Product"):
    """Enumerate all 4 stereoisomers for every reaction in the dataset.

    Args:
        df: DataFrame with product SMILES column
        product_col: column name for canonical product SMILES
        mapped_col: column name for atom-mapped product SMILES

    Returns:
        DataFrame with columns:
            reaction_id, candidate_id, candidate_smiles, is_true_product,
            ca_config, cb_config, orig_ca_cip, orig_cb_cip
    """
    configs = [("R", "R"), ("R", "S"), ("S", "R"), ("S", "S")]
    rows = []
    n_success = 0
    n_fail = 0

    for idx, row in df.iterrows():
        product_smi = row[product_col]
        mapped_smi = row.get(mapped_col, None)

        result = enumerate_stereoisomers(product_smi, mapped_smi)

        if result is None:
            n_fail += 1
            logger.warning(f"Failed to enumerate reaction idx={idx}: {product_smi[:50]}")
            continue

        n_success += 1
        for cand_id, cand_smi in enumerate(result["candidates"]):
            ca_config, cb_config = configs[cand_id]
            rows.append({
                "reaction_id": idx,
                "candidate_id": cand_id,
                "candidate_smiles": cand_smi,
                "is_true_product": int(cand_id == result["true_candidate_id"]),
                "ca_config": ca_config,
                "cb_config": cb_config,
                "orig_ca_cip": result["orig_ca_cip"],
                "orig_cb_cip": result["orig_cb_cip"],
            })

    logger.info(f"Enumeration complete: {n_success} success, {n_fail} failed "
                f"({n_fail/(n_success+n_fail)*100:.1f}% failure rate)")

    candidates_df = pd.DataFrame(rows)
    return candidates_df


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Example Evans aldol product
    test_smi = "CCCCCCC[C@@H](O)[C@H](C)C(=O)N1C(=O)OC[C@@H]1Cc1ccccc1"
    result = enumerate_stereoisomers(test_smi)
    if result:
        print(f"Success! True candidate: {result['true_candidate_id']}")
        for i, smi in enumerate(result['candidates']):
            marker = " <-- TRUE" if i == result['true_candidate_id'] else ""
            print(f"  Candidate {i} ({['RR','RS','SR','SS'][i]}): {smi}{marker}")
    else:
        print("Failed to enumerate")
