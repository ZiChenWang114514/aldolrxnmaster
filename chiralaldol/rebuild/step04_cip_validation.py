"""Step 4: CIP label cross-validation.

Identify Ca (alpha-carbon) and Cb (OH-bearing carbon) in the product via
atom mapping, extract their CIP R/S codes, and compare with dataset labels.
Delete rows where CIP extraction disagrees with labels.
"""

import logging
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)


def _extract_atom_map_dict(mapped_smi: str) -> dict[int, int]:
    """Parse atom-mapped SMILES → {atom_map_num: atom_idx}."""
    if pd.isna(mapped_smi) or not str(mapped_smi).strip():
        return {}
    mol = Chem.MolFromSmiles(str(mapped_smi), sanitize=False)
    if mol is None:
        return {}
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return {}
    result = {}
    for atom in mol.GetAtoms():
        mn = atom.GetAtomMapNum()
        if mn > 0:
            result[mn] = atom.GetIdx()
    return result


def _find_new_cc_bond(mapped_rxn: str) -> tuple[int, int] | None:
    """Find the new C-C bond formed in the reaction using atom maps.

    Returns (map_num_Ca, map_num_Cb) where Cb has the new OH.
    """
    if pd.isna(mapped_rxn) or ">>" not in str(mapped_rxn):
        return None

    parts = str(mapped_rxn).split(">>")
    reactant_smi = parts[0]
    product_smi = parts[1]

    # Build bond sets for reactants and products
    def bond_set(smi):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return set()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return set()
        bonds = set()
        for bond in mol.GetBonds():
            a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
            a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
            if a1 > 0 and a2 > 0:
                bonds.add((min(a1, a2), max(a1, a2)))
        return bonds

    r_bonds = bond_set(reactant_smi)
    p_bonds = bond_set(product_smi)
    new_bonds = p_bonds - r_bonds

    # Find new C-C bonds
    prod_mol = Chem.MolFromSmiles(product_smi, sanitize=False)
    if prod_mol is None:
        return None
    try:
        Chem.SanitizeMol(prod_mol)
    except Exception:
        return None

    map_to_idx = {}
    for atom in prod_mol.GetAtoms():
        mn = atom.GetAtomMapNum()
        if mn > 0:
            map_to_idx[mn] = atom.GetIdx()

    cc_bonds = []
    for m1, m2 in new_bonds:
        if m1 in map_to_idx and m2 in map_to_idx:
            a1 = prod_mol.GetAtomWithIdx(map_to_idx[m1])
            a2 = prod_mol.GetAtomWithIdx(map_to_idx[m2])
            if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
                cc_bonds.append((m1, m2))

    if len(cc_bonds) != 1:
        return None

    m1, m2 = cc_bonds[0]
    # Determine which is Cb (has OH neighbor) and which is Ca
    idx1 = map_to_idx[m1]
    idx2 = map_to_idx[m2]

    def has_oh_neighbor(mol, idx):
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 8:
                # Check if this O has an H (i.e., OH)
                if nb.GetTotalNumHs() > 0:
                    return True
        return False

    if has_oh_neighbor(prod_mol, idx1):
        return (m2, m1)  # (Ca_map, Cb_map) — Cb has OH
    elif has_oh_neighbor(prod_mol, idx2):
        return (m1, m2)
    else:
        return None


def _get_cip_at_map(mapped_smi: str, map_num: int) -> str | None:
    """Get CIP code (R/S) for the atom with given map number."""
    if pd.isna(mapped_smi):
        return None
    mol = Chem.MolFromSmiles(str(mapped_smi), sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetPropsAsDict().get("_CIPCode", None)
    return None


def _smarts_fallback_cip(product_smi: str) -> tuple[str | None, str | None]:
    """Fallback: use SMARTS to find Ca/Cb in product without atom mapping.

    Evans aldol product core: [C:1](OH)(R)[C:2](R')C(=O)N
    :1 = Cb (OH-bearing), :2 = Ca (next to C(=O)N)
    """
    if pd.isna(product_smi):
        return None, None
    mol = Chem.MolFromSmiles(str(product_smi))
    if mol is None:
        return None, None

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Try multiple SMARTS patterns for the beta-hydroxy-N-acyl fragment
    patterns = [
        "[C:1]([OH1])([#6])[C:2]([#6])[C](=O)[N]",
        "[C:1]([OH1])[C:2][C](=O)[N]",
    ]
    for patt_str in patterns:
        patt = Chem.MolFromSmarts(patt_str)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            cb_idx, ca_idx = matches[0][0], matches[0][1]
            cb_atom = mol.GetAtomWithIdx(cb_idx)
            ca_atom = mol.GetAtomWithIdx(ca_idx)
            cip_cb = cb_atom.GetPropsAsDict().get("_CIPCode", None)
            cip_ca = ca_atom.GetPropsAsDict().get("_CIPCode", None)
            return cip_ca, cip_cb

    return None, None


def run(context: dict) -> dict:
    """Validate labels using SA consistency and CIP relative configuration.

    Key insight from literature: CIP R/S is substrate-dependent — the same
    syn diastereomer can be (2R,3S) for one substrate but (2S,3R) for another.
    Therefore we CANNOT do a global Ca=0↔R mapping.

    Instead we validate:
    1. SA consistency: label_SA should match the relative config (Ca same/diff as Cb)
       within each substrate class.
    2. CIP extraction is recorded for audit but NOT used for deletion.
    3. Rows with missing critical SMILES (Ketone/Aldehyde both NaN) are deleted.
    """
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    n_start = len(df)
    logger.info(f"Step 4: Label validation for {n_start} rows")

    # ── Part A: Extract CIP codes (informational, not for deletion) ──
    cip_ca_list = []
    cip_cb_list = []
    method_list = []

    for _, row in df.iterrows():
        mapped_rxn = row.get("Mapped_Reaction")
        mapped_prod = row.get("Mapped_Product")
        product_smi = row.get("canonical_Raw_Product_Smiles", row.get("Raw_Product_Smiles"))

        cip_ca, cip_cb = None, None
        method = "none"

        # Method 1: atom mapping
        if pd.notna(mapped_rxn) and pd.notna(mapped_prod):
            bond_info = _find_new_cc_bond(str(mapped_rxn))
            if bond_info is not None:
                ca_map, cb_map = bond_info
                cip_ca = _get_cip_at_map(str(mapped_prod), ca_map)
                cip_cb = _get_cip_at_map(str(mapped_prod), cb_map)
                if cip_ca and cip_cb:
                    method = "atom_mapping"

        # Method 2: SMARTS fallback
        if cip_ca is None or cip_cb is None:
            cip_ca_f, cip_cb_f = _smarts_fallback_cip(product_smi)
            if cip_ca_f and cip_cb_f:
                cip_ca = cip_ca_f
                cip_cb = cip_cb_f
                method = "smarts_fallback"

        cip_ca_list.append(cip_ca)
        cip_cb_list.append(cip_cb)
        method_list.append(method)

    df["cip_Ca_extracted"] = cip_ca_list
    df["cip_Cb_extracted"] = cip_cb_list
    df["cip_method"] = method_list

    n_extracted = sum(1 for m in method_list if m != "none")
    n_atmap = sum(1 for m in method_list if m == "atom_mapping")
    n_smarts = sum(1 for m in method_list if m == "smarts_fallback")
    logger.info(f"  CIP extracted: {n_extracted}/{n_start} ({n_atmap} atom_mapping, {n_smarts} smarts_fallback)")

    # ── Part B: SA consistency check ──
    # Normalize labels to 0/1
    has_labels = df["label_Ca"].notna() & df["label_Cb"].notna() & df["label_SA"].notna()
    n_with_labels = has_labels.sum()
    logger.info(f"  Rows with all labels: {n_with_labels}/{n_start}")

    if n_with_labels > 0:
        sub = df[has_labels].copy()
        ca_vals = sub["label_Ca"].values
        cb_vals = sub["label_Cb"].values
        sa_vals = sub["label_SA"].values

        # Normalize to 0/1 (handle NaN safely)
        if np.any(np.isnan(ca_vals)) or np.any(np.isnan(cb_vals)) or np.any(np.isnan(sa_vals)):
            logger.warning("  NaN found in labels; dropping NaN rows from SA check")
            valid = ~(np.isnan(ca_vals) | np.isnan(cb_vals) | np.isnan(sa_vals))
            ca_vals, cb_vals, sa_vals = ca_vals[valid], cb_vals[valid], sa_vals[valid]
            sub = sub[valid]
        if len(ca_vals) == 0:
            logger.warning("  No valid labels after NaN removal")
        elif np.nanmin(ca_vals) < 0:
            ca_01 = ((ca_vals + 1) / 2).astype(int)
            cb_01 = ((cb_vals + 1) / 2).astype(int)
            sa_01 = ((sa_vals + 1) / 2).astype(int)
        else:
            ca_01 = ca_vals.astype(int)
            cb_01 = cb_vals.astype(int)
            sa_01 = sa_vals.astype(int)

        # Check: does SA=1 (syn) correspond to Ca==Cb or Ca!=Cb?
        same_config = (ca_01 == cb_01).astype(int)
        # Count agreement
        sa1_same = sum(1 for s, sc in zip(sa_01, same_config) if s == 1 and sc == 1)
        sa1_diff = sum(1 for s, sc in zip(sa_01, same_config) if s == 1 and sc == 0)
        sa0_same = sum(1 for s, sc in zip(sa_01, same_config) if s == 0 and sc == 1)
        sa0_diff = sum(1 for s, sc in zip(sa_01, same_config) if s == 0 and sc == 0)

        logger.info(f"  SA consistency: SA=1,Ca==Cb:{sa1_same} SA=1,Ca!=Cb:{sa1_diff} SA=0,Ca==Cb:{sa0_same} SA=0,Ca!=Cb:{sa0_diff}")

        # Determine SA convention: SA=1 means syn = Ca==Cb? or Ca!=Cb?
        if sa1_same + sa0_diff > sa1_diff + sa0_same:
            sa_convention = "syn_is_same"  # SA=1 means Ca==Cb
            n_consistent = sa1_same + sa0_diff
        else:
            sa_convention = "syn_is_diff"  # SA=1 means Ca!=Cb
            n_consistent = sa1_diff + sa0_same

        n_inconsistent = n_with_labels - n_consistent
        logger.info(f"  SA convention: {sa_convention} ({n_consistent} consistent, {n_inconsistent} inconsistent)")
        context["sa_convention"] = sa_convention

        # Find inconsistent rows and delete them
        if sa_convention == "syn_is_same":
            inconsistent = (sa_01 != same_config)
        else:
            inconsistent = (sa_01 == same_config)

        # Map back to df indices
        inconsistent_indices = sub.index[inconsistent]
        n_sa_bad = len(inconsistent_indices)

        if n_sa_bad > 0:
            audit.mark_deleted_by_oi(
                df.loc[inconsistent_indices, "original_index"].values, "sa_label_inconsistent")
            df = df.drop(inconsistent_indices).reset_index(drop=True)
            logger.info(f"  Deleted {n_sa_bad} rows with SA inconsistency")

    # ── Part C: Delete rows with missing critical SMILES ──
    ketone_col = "canonical_Ketone" if "canonical_Ketone" in df.columns else "Ketone"
    aldehyde_col = "canonical_Aldehyde" if "canonical_Aldehyde" in df.columns else "Aldehyde"

    missing_mol = df[ketone_col].isna() & df[aldehyde_col].isna()
    n_missing = missing_mol.sum()
    if n_missing > 0:
        audit.mark_deleted_by_oi(
            df.loc[missing_mol, "original_index"].values, "missing_ketone_and_aldehyde")
        df = df[~missing_mol].reset_index(drop=True)
        logger.info(f"  Deleted {n_missing} rows with both Ketone and Aldehyde missing")

    n_end = len(df)
    logger.info(f"  Step 4 complete: {n_start} → {n_end} rows")

    out_path = context["output_dir"] / "interim" / "04_cip_validated.csv"
    df.to_csv(out_path, index=False)

    context["df"] = df
    return context
