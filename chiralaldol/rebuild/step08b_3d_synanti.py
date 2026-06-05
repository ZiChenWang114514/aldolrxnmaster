"""Step 08b: Determine true syn/anti via 3D conformer dihedral analysis.

CIP R/S codes are UNRELIABLE for syn/anti determination (~52% accuracy,
equivalent to a coin flip). The root cause is that CIP priorities depend
on substituent identity — the same syn product can be (R,S) on one
substrate but (S,R) on another.

This step computes the OH-Cb-Ca-C(=O) dihedral angle from a 3D conformer
(ETKDGv3 + MMFF optimization) to directly measure the spatial relationship
between the two new stereocenters. |θ| < 90° → syn, |θ| ≥ 90° → anti.
"""

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from .audit import AuditTracker

logger = logging.getLogger("rebuild_v4.step08b")


def _find_oh_neighbor(mol, cb_idx: int) -> int | None:
    """Find the oxygen neighbor of Cb (free OH or protected O)."""
    cb_atom = mol.GetAtomWithIdx(cb_idx)
    for nb in cb_atom.GetNeighbors():
        if nb.GetAtomicNum() == 8:
            return nb.GetIdx()
    return None


def _find_co_neighbor(mol, ca_idx: int, cb_idx: int) -> int | None:
    """Find the C=O carbon adjacent to Ca (excluding Cb).

    Matches any carbon neighbor of Ca that has a double bond to oxygen,
    which works for all auxiliary types (Evans C(=O)-N, Oppolzer C(=O)-N-SO2, etc.).
    """
    ca_atom = mol.GetAtomWithIdx(ca_idx)
    for nb in ca_atom.GetNeighbors():
        if nb.GetIdx() == cb_idx:
            continue
        if nb.GetAtomicNum() != 6:
            continue
        for nb2 in nb.GetNeighbors():
            if nb2.GetAtomicNum() == 8:
                bond = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                if bond and bond.GetBondTypeAsDouble() == 2.0:
                    return nb.GetIdx()
    return None


def _compute_synanti_3d(product_smi: str, ca_idx: int, cb_idx: int) -> dict:
    """Generate a 3D conformer and compute the OH-Cb-Ca-C(=O) dihedral.

    Returns dict with dihedral (degrees), energy (kcal/mol MMFF),
    is_syn (bool), and confidence (0-1).
    """
    result = {"dihedral": None, "energy": None, "is_syn": None, "confidence": None}

    try:
        mol = Chem.MolFromSmiles(str(product_smi))
        if mol is None:
            return result
        mol = Chem.AddHs(mol)

        ca_idx_h = int(ca_idx)
        cb_idx_h = int(cb_idx)

        # Locate reference atoms
        oh_idx = _find_oh_neighbor(mol, cb_idx_h)
        co_idx = _find_co_neighbor(mol, ca_idx_h, cb_idx_h)
        if oh_idx is None or co_idx is None:
            return result

        # Generate 3D conformer (ETKDGv3)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            # Retry without fixed seed
            params.randomSeed = -1
            cid = AllChem.EmbedMolecule(mol, params)
            if cid < 0:
                return result

        # MMFF optimization
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        # MMFF energy
        energy = None
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            if ff:
                energy = ff.CalcEnergy()

        # Dihedral: OH - Cb - Ca - C(=O)
        dihedral = AllChem.GetDihedralDeg(
            mol.GetConformer(), oh_idx, cb_idx_h, ca_idx_h, co_idx
        )

        is_syn = abs(dihedral) < 90.0
        confidence = abs(abs(dihedral) - 90.0) / 90.0  # 0 = at boundary, 1 = very sure

        result["dihedral"] = round(dihedral, 2)
        result["energy"] = round(energy, 4) if energy is not None else None
        result["is_syn"] = is_syn
        result["confidence"] = round(confidence, 4)
    except Exception as e:
        logger.debug(f"3D syn/anti failed for {product_smi[:40]}...: {e}")

    return result


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Compute 3D dihedral-based syn/anti labels for all rows."""
    logger.info("Step 08b: 3D dihedral-based syn/anti determination...")
    n_start = len(df)

    prod_col = (
        "canonical_main_product_smiles"
        if "canonical_main_product_smiles" in df.columns
        else "main_product_smiles"
    )

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Processing {i}/{len(df)}...")

        ca = row.get("ca_atom_idx")
        cb = row.get("cb_atom_idx")

        if pd.isna(ca) or pd.isna(cb):
            results.append(
                {"dihedral": None, "energy": None, "is_syn": None, "confidence": None}
            )
            continue

        r = _compute_synanti_3d(row[prod_col], int(ca), int(cb))
        results.append(r)

    res_df = pd.DataFrame(results)
    df["dihedral_oh_cb_ca_co"] = res_df["dihedral"].values
    df["conformer_energy"] = res_df["energy"].values
    df["label_syn_anti_3d"] = (
        res_df["is_syn"]
        .apply(lambda x: int(x) if x is not None else None)
        .values
    )
    df["synanti_confidence"] = res_df["confidence"].values

    n_ok = df["label_syn_anti_3d"].notna().sum()
    n_syn = int(df["label_syn_anti_3d"].eq(1).sum())
    n_anti = int(df["label_syn_anti_3d"].eq(0).sum())
    n_fail = len(df) - n_ok
    logger.info(
        f"  3D syn/anti computed: {n_ok}/{len(df)} rows "
        f"(syn={n_syn}, anti={n_anti}, failed={n_fail})"
    )

    # No rows dropped in this step
    audit.record_step("08b_3d_synanti", len(df))
    logger.info(f"  Step 08b complete: {n_start} -> {len(df)} rows (no drops)")
    return df
