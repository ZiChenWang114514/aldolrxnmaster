"""M3b: Aldehyde 3D Steric Descriptors for Zimmerman-Traxler TS modeling.

Computes Sterimol (L/B1/B5) and total %Vbur for the aldehyde R-group,
using the same conformer ensemble sampling and Boltzmann weighting as M3.

Physical motivation:
  In the Zimmerman-Traxler transition state, the aldehyde R-group strongly
  prefers the pseudo-equatorial position to avoid 1,3-diaxial interactions
  with the Evans oxazolidinone auxiliary. The steric bulk of R (captured by
  Sterimol B5 and Vbur_total) determines how strongly this equatorial preference
  drives the observed syn-selectivity.

Features (10d per molecule):
  ald_L_mean/std        : Sterimol L along CHO bond axis
  ald_B1_mean/std       : Sterimol B1 (minimum perpendicular width)
  ald_B5_mean/std       : Sterimol B5 (maximum perpendicular width)
  ald_Vbur_total_mean/std : total %Vbur around the carbonyl C (sphere r=3.5Å)
  ald_n_conformers      : number of valid conformers used
  ald_n_clusters        : number of conformer clusters
"""

import logging

import numpy as np
from rdkit import Chem, RDLogger

from .steric_descriptors import compute_buried_volume, compute_sterimol

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# SMARTS: aldehyde carbon — degree 3, exactly 1H, double-bonded to O with degree 1
# Matches R-CHO: carbonyl_C=:1, O=:2
ALDEHYDE_SMARTS = Chem.MolFromSmarts("[CX3;H1:1](=[OX1:2])")

ALDEHYDE_STERIC_DESC_NAMES = [
    "ald_L_mean", "ald_L_std",
    "ald_B1_mean", "ald_B1_std",
    "ald_B5_mean", "ald_B5_std",
    "ald_Vbur_total_mean", "ald_Vbur_total_std",
    "ald_n_conformers", "ald_n_clusters",
]


def strip_atom_map(smiles: str) -> str | None:
    """Remove atom mapping numbers from a SMILES string and return canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def find_aldehyde_center(mol: Chem.Mol) -> tuple[int, int] | None:
    """Find (carbonyl_C_idx, O_idx) for the aldehyde CHO group.

    Returns the first match of ALDEHYDE_SMARTS, or None if not found.
    """
    if ALDEHYDE_SMARTS is None:
        return None
    matches = mol.GetSubstructMatches(ALDEHYDE_SMARTS)
    if matches:
        return matches[0]  # (carbonyl_C_idx, O_idx)
    return None


def find_aldehyde_R_atoms(mol: Chem.Mol, carbonyl_idx: int, o_idx: int) -> list[int]:
    """BFS to find R-group atoms attached to the aldehyde carbonyl carbon.

    Excludes the oxygen and all hydrogens. Returns list of heavy-atom indices.
    For formaldehyde (HCHO) this returns an empty list.
    """
    visited = {carbonyl_idx, o_idx}
    queue = []

    for neighbor in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx not in visited and neighbor.GetAtomicNum() != 1:
            queue.append(nidx)
            visited.add(nidx)

    rgroup = []
    while queue:
        current = queue.pop(0)
        rgroup.append(current)
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx not in visited and neighbor.GetAtomicNum() != 1:
                visited.add(nidx)
                queue.append(nidx)

    return rgroup


def compute_aldehyde_single_conformer(
    mol: Chem.Mol,
    coords: np.ndarray,
    carbonyl_idx: int,
    o_idx: int,
    r_idxs: list[int],
) -> dict | None:
    """Compute aldehyde steric descriptors for a single conformer.

    Args:
        mol: RDKit Mol (no explicit H, i.e. after Chem.MolFromSmiles)
        coords: (n_heavy_atoms, 3) heavy-atom coordinates
        carbonyl_idx: index of aldehyde carbonyl carbon
        o_idx: index of aldehyde oxygen
        r_idxs: heavy-atom indices of the R-group

    Returns dict with L, B1, B5, Vbur_total, or None on failure.
    """
    n_atoms = mol.GetNumAtoms()
    if coords is None or len(coords) != n_atoms:
        return None
    if carbonyl_idx >= n_atoms or o_idx >= n_atoms:
        return None

    atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)]

    # Sterimol of R-group (attach point = carbonyl C)
    sterimol = compute_sterimol(coords, atomic_nums, carbonyl_idx, r_idxs)

    # Total buried volume around carbonyl C (normal = C→O bond direction)
    co_vec = coords[o_idx] - coords[carbonyl_idx]
    norm_len = np.linalg.norm(co_vec)
    normal = co_vec / norm_len if norm_len > 1e-6 else np.array([0.0, 0.0, 1.0])
    vbur = compute_buried_volume(
        coords, atomic_nums, carbonyl_idx, normal, radius=3.5, grid_spacing=0.25
    )

    return {
        "L": sterimol["L"],
        "B1": sterimol["B1"],
        "B5": sterimol["B5"],
        "Vbur_total": vbur["Vbur_total"],
    }


def compute_aldehyde_ensemble_descriptors(smiles: str, ensemble: dict) -> dict | None:
    """Boltzmann-aggregate aldehyde steric descriptors over a conformer ensemble.

    Args:
        smiles: clean (atom-map-free) canonical SMILES of the aldehyde
        ensemble: output from generate_conformer_ensemble()

    Returns 10d dict (ALDEHYDE_STERIC_DESC_NAMES), or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    center = find_aldehyde_center(mol)
    if center is None:
        logger.debug(f"No aldehyde SMARTS match: {smiles}")
        return None

    carbonyl_idx, o_idx = center
    r_idxs = find_aldehyde_R_atoms(mol, carbonyl_idx, o_idx)

    representatives = ensemble.get("representatives", [])
    if not representatives:
        return None

    all_desc, weights = [], []
    for _conf_id, _energy, weight, coords in representatives:
        if coords is None:
            continue
        desc = compute_aldehyde_single_conformer(mol, coords, carbonyl_idx, o_idx, r_idxs)
        if desc is not None:
            all_desc.append(desc)
            weights.append(weight)

    if not all_desc:
        return None

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    result = {}
    for key in ["L", "B1", "B5", "Vbur_total"]:
        values = np.array([d[key] for d in all_desc])
        wmean = np.average(values, weights=weights)
        wstd = np.sqrt(np.average((values - wmean) ** 2, weights=weights))
        result[f"ald_{key}_mean"] = float(wmean)
        result[f"ald_{key}_std"] = float(wstd)

    result["ald_n_conformers"] = len(all_desc)
    result["ald_n_clusters"] = ensemble.get("n_clusters", 1)

    return result
