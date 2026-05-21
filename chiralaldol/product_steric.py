"""Product-level steric descriptors for StereoRank candidates (48d: Ca 24d + Cb 24d).

Uses the same professional algorithms as steric_descriptors.py (grid-based %Vbur,
Sterimol L/B1/B5, dihedral sin/cos encoding) but applied to PRODUCT molecules
at both stereocenters (Ca and Cb).

Different stereoisomers have different conformational preferences → different
steric profiles → genuine feature differentiation for ranking.
"""

import logging
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger

from .conformer_sampler import generate_conformer_ensemble
from .steric_descriptors import (
    compute_single_conformer_descriptors,
    find_rgroup_atoms,
)
from .utils import clean_mol

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# Product SMARTS: :1=Cb (OH-bearing), :2=Ca (adjacent to C(=O)N)
PRODUCT_SMARTS_LIST = [
    "[C:1]([OH1])([#6])[C:2]([#6])[C](=O)[N]",
    "[C:1]([OH1])[C:2][C](=O)[N]",
]


def find_product_centers(mol):
    """Find Ca and Cb centers in a product molecule, returning two 4-tuples.

    Each tuple is (center_idx, scaffold_idx, plane_atom_1, plane_atom_2)
    compatible with compute_single_conformer_descriptors(mol, coords, center).

    For Ca: center = (Ca, carbonyl_C, carbonyl_O, N)
        - plane defined by Ca-carbonyl_C-O (same as enolate)
        - R-group = substituents on Ca away from carbonyl

    For Cb: center = (Cb, Ca, OH_O, aldehyde_R_C)
        - plane defined by Cb-Ca-OH
        - R-group = substituents on Cb away from Ca (aldehyde side)

    Returns:
        (ca_center_tuple, cb_center_tuple) or None if detection fails.
    """
    # Find Ca and Cb via SMARTS
    ca_idx = cb_idx = None
    for smarts_str in PRODUCT_SMARTS_LIST:
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue
        map_to_smarts_idx = {}
        for atom in pattern.GetAtoms():
            mn = atom.GetAtomMapNum()
            if mn > 0:
                map_to_smarts_idx[mn] = atom.GetIdx()
        matches = mol.GetSubstructMatches(pattern)
        if matches and 1 in map_to_smarts_idx and 2 in map_to_smarts_idx:
            cb_idx = matches[0][map_to_smarts_idx[1]]
            ca_idx = matches[0][map_to_smarts_idx[2]]
            break

    if ca_idx is None or cb_idx is None:
        return None

    # --- Ca center: find carbonyl_C, carbonyl_O, N ---
    carbonyl_c = None
    carbonyl_o = None
    n_idx = None

    for nb in mol.GetAtomWithIdx(ca_idx).GetNeighbors():
        if nb.GetIdx() == cb_idx:
            continue
        if nb.GetAtomicNum() == 6:
            # Check if this C has a double-bond O and a bonded N
            has_dbl_o = False
            has_n = False
            for nb2 in nb.GetNeighbors():
                if nb2.GetAtomicNum() == 8:
                    bond = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                    if bond and bond.GetBondTypeAsDouble() >= 1.5:
                        has_dbl_o = True
                        carbonyl_o = nb2.GetIdx()
                elif nb2.GetAtomicNum() == 7:
                    has_n = True
                    n_idx = nb2.GetIdx()
            if has_dbl_o and has_n:
                carbonyl_c = nb.GetIdx()
                break

    if carbonyl_c is None or carbonyl_o is None or n_idx is None:
        return None

    ca_center = (ca_idx, carbonyl_c, carbonyl_o, n_idx)

    # --- Cb center: find OH oxygen, aldehyde R first carbon ---
    oh_idx = None
    ald_r_idx = None

    for nb in mol.GetAtomWithIdx(cb_idx).GetNeighbors():
        if nb.GetIdx() == ca_idx:
            continue
        if nb.GetAtomicNum() == 8:
            # OH: single-bonded O (not C=O)
            bond = mol.GetBondBetweenAtoms(cb_idx, nb.GetIdx())
            if bond and bond.GetBondTypeAsDouble() < 1.5:
                oh_idx = nb.GetIdx()
        elif nb.GetAtomicNum() == 6:
            # First carbon of aldehyde R-group
            if ald_r_idx is None:
                ald_r_idx = nb.GetIdx()

    if oh_idx is None:
        return None
    # Fallback: if no aldehyde R carbon found, use Ca as the 4th atom
    if ald_r_idx is None:
        ald_r_idx = ca_idx

    cb_center = (cb_idx, ca_idx, oh_idx, ald_r_idx)

    return ca_center, cb_center


def compute_product_ensemble_descriptors(smiles, n_confs=100):
    """Compute 48d product steric descriptors (Ca 24d + Cb 24d).

    Args:
        smiles: Product SMILES with defined stereochemistry
        n_confs: Number of conformers for ensemble generation

    Returns:
        dict with 48 descriptor values (prefixed prod_ca_ and prod_cb_),
        or None on failure.
    """
    mol = clean_mol(smiles)
    if mol is None:
        return None

    # Find product centers
    centers = find_product_centers(mol)
    if centers is None:
        return None
    ca_center, cb_center = centers

    # Generate conformer ensemble (full pipeline: ETKDG + MMFF + clustering)
    ensemble = generate_conformer_ensemble(smiles, n_confs=n_confs)
    if ensemble is None:
        return None

    representatives = ensemble["representatives"]
    if not representatives:
        return None

    # Compute descriptors for each representative at BOTH centers
    all_desc_ca = []
    all_desc_cb = []
    weights = []

    for conf_id, energy, weight, coords in representatives:
        if coords is None:
            continue
        if len(coords) != mol.GetNumAtoms():
            continue

        desc_ca = compute_single_conformer_descriptors(mol, coords, ca_center)
        desc_cb = compute_single_conformer_descriptors(mol, coords, cb_center)

        if desc_ca is not None and desc_cb is not None:
            all_desc_ca.append(desc_ca)
            all_desc_cb.append(desc_cb)
            weights.append(weight)

    if not all_desc_ca:
        return None

    # Boltzmann-weighted aggregation
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    result = {}

    # Ca descriptors (24d)
    for key in sorted(all_desc_ca[0].keys()):
        values = np.array([d[key] for d in all_desc_ca])
        wmean = np.average(values, weights=weights)
        wstd = np.sqrt(np.average((values - wmean) ** 2, weights=weights))
        result[f"prod_ca_{key}_mean"] = float(wmean)
        result[f"prod_ca_{key}_std"] = float(wstd)
    result["prod_ca_n_conformers"] = len(all_desc_ca)
    result["prod_ca_n_clusters"] = ensemble["n_clusters"]

    # Cb descriptors (24d)
    for key in sorted(all_desc_cb[0].keys()):
        values = np.array([d[key] for d in all_desc_cb])
        wmean = np.average(values, weights=weights)
        wstd = np.sqrt(np.average((values - wmean) ** 2, weights=weights))
        result[f"prod_cb_{key}_mean"] = float(wmean)
        result[f"prod_cb_{key}_std"] = float(wstd)
    result["prod_cb_n_conformers"] = len(all_desc_cb)
    result["prod_cb_n_clusters"] = ensemble["n_clusters"]

    return result


def _compute_single_worker(args):
    """Worker for parallel computation."""
    idx, smiles, n_confs = args
    try:
        result = compute_product_ensemble_descriptors(smiles, n_confs=n_confs)
        return idx, result
    except Exception as e:
        logger.debug(f"Failed idx={idx}: {e}")
        return idx, None


def compute_all_candidates_48d(candidates_df, n_confs=100, n_workers=8):
    """Compute 48d product descriptors for all candidates in parallel.

    Args:
        candidates_df: DataFrame with 'candidate_smiles' column
        n_confs: Conformers per molecule
        n_workers: Parallel workers

    Returns:
        DataFrame with 48 product descriptor columns added
    """
    tasks = [(idx, row["candidate_smiles"], n_confs)
             for idx, row in candidates_df.iterrows()]

    logger.info(f"Computing 48d product steric for {len(tasks)} candidates "
                f"({n_workers} workers, {n_confs} confs/mol)...")

    results = {}
    with Pool(n_workers) as pool:
        for i, (idx, desc) in enumerate(pool.imap_unordered(_compute_single_worker, tasks, chunksize=5)):
            results[idx] = desc
            if (i + 1) % 200 == 0:
                n_ok = sum(1 for v in results.values() if v is not None)
                logger.info(f"  Progress: {i+1}/{len(tasks)} ({n_ok} success)")

    n_ok = sum(1 for v in results.values() if v is not None)
    n_fail = len(results) - n_ok
    logger.info(f"Done: {n_ok} success, {n_fail} failed ({n_fail/(n_ok+n_fail)*100:.1f}%)")

    # Get all feature column names from first successful result
    sample_result = next((v for v in results.values() if v is not None), None)
    if sample_result is None:
        logger.error("All computations failed!")
        return candidates_df

    feat_cols = sorted(sample_result.keys())

    # Fill into DataFrame
    for col in feat_cols:
        candidates_df[col] = candidates_df.index.map(
            lambda idx: results.get(idx, {}).get(col, np.nan) if results.get(idx) else np.nan
        )

    return candidates_df
