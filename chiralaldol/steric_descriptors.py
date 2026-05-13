"""M3: 3D Steric Descriptors — Face-dependent buried volume, Sterimol, dihedrals.

For each representative conformer:
  A. %Vbur (si-face / re-face) — buried volume around reactive center
  B. Sterimol L/B1/B5 — R-group geometry
  C. Key dihedral angles (sin/cos encoded)

Then Boltzmann-weighted aggregation across conformers → fixed-length vector.
"""

import logging

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

from .utils import clean_mol, get_plane_normal, get_vdw_radius

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# SMARTS to identify the enolate reactive center
# Matches C=C-C(=O or [O-])-N pattern (the enolate C=C adjacent to amide)
ENOLATE_CENTER_SMARTS = Chem.MolFromSmarts("[CH,CH2,C:1]=[CX3:2](-[OX1,OX2:3])-[NX3:4]")
# Fallback for ketone (not enolized): alpha-CH bonded to C(=O)-N
KETONE_CENTER_SMARTS = Chem.MolFromSmarts("[CH2,CH;X3,X4:1]-[CX3:2](=[OX1:3])-[NX3:4]")


def find_reactive_center(mol: Chem.Mol) -> tuple[int, int, int, int] | None:
    """Find the reactive center atoms: (alpha_C, carbonyl_C, O, N).

    Tries enolate pattern first, then ketone pattern as fallback.
    """
    for pattern in [ENOLATE_CENTER_SMARTS, KETONE_CENTER_SMARTS]:
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            return matches[0]
    return None


def compute_buried_volume(
    coords: np.ndarray,
    atomic_nums: list[int],
    center_idx: int,
    normal: np.ndarray,
    radius: float = 3.5,
    grid_spacing: float = 0.25,
) -> dict:
    """Compute face-dependent %Vbur around a center atom.

    Divides a sphere of given radius into upper (si-face) and lower (re-face)
    hemispheres relative to the plane defined by the normal vector.
    Computes what fraction of each hemisphere is occupied by van der Waals spheres
    of neighboring atoms.

    Returns dict with %Vbur_si, %Vbur_re, %Vbur_diff, %Vbur_total.
    """
    center = coords[center_idx]
    n_atoms = len(coords)

    # Generate grid points in sphere
    n_grid = int(2 * radius / grid_spacing) + 1
    lin = np.linspace(-radius, radius, n_grid)
    xx, yy, zz = np.meshgrid(lin, lin, lin)
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Filter to sphere
    dist_from_center = np.linalg.norm(grid, axis=1)
    in_sphere = dist_from_center <= radius
    grid_in = grid[in_sphere]  # relative to center
    n_pts = len(grid_in)
    if n_pts == 0:
        return {"Vbur_si": 0.0, "Vbur_re": 0.0, "Vbur_diff": 0.0, "Vbur_total": 0.0}

    # Classify points as si-face (>0) or re-face (<0) by dot product with normal
    face_sign = grid_in @ normal  # positive = si-face
    si_mask = face_sign > 0
    re_mask = face_sign <= 0

    n_si = si_mask.sum()
    n_re = re_mask.sum()

    # Check which grid points are inside any atom's vdW sphere
    occupied = np.zeros(n_pts, dtype=bool)
    for j in range(n_atoms):
        if j == center_idx:
            continue
        r_vdw = get_vdw_radius(atomic_nums[j])
        delta = grid_in - (coords[j] - center)
        dist_sq = (delta ** 2).sum(axis=1)
        occupied |= (dist_sq <= r_vdw ** 2)

    # Compute buried fractions
    occ_si = (occupied & si_mask).sum()
    occ_re = (occupied & re_mask).sum()

    vbur_si = occ_si / max(n_si, 1) * 100.0
    vbur_re = occ_re / max(n_re, 1) * 100.0
    vbur_total = occupied.sum() / max(n_pts, 1) * 100.0
    vbur_diff = vbur_si - vbur_re

    return {
        "Vbur_si": vbur_si,
        "Vbur_re": vbur_re,
        "Vbur_diff": vbur_diff,
        "Vbur_total": vbur_total,
    }


def compute_sterimol(
    coords: np.ndarray,
    atomic_nums: list[int],
    attach_idx: int,
    rgroup_idxs: list[int],
) -> dict:
    """Compute Sterimol L, B1, B5 for an R-group.

    L: length along the attachment axis (from attach atom to furthest atom)
    B1: minimum perpendicular extent
    B5: maximum perpendicular extent

    Args:
        coords: (n_atoms, 3) array
        atomic_nums: list of atomic numbers
        attach_idx: atom index where R-group connects to scaffold
        rgroup_idxs: atom indices belonging to the R-group
    """
    if not rgroup_idxs:
        return {"L": 0.0, "B1": 0.0, "B5": 0.0}

    origin = coords[attach_idx]
    # Filter out-of-bounds indices (safety)
    rgroup_idxs = [i for i in rgroup_idxs if i < len(coords)]
    if not rgroup_idxs:
        return {"L": 0.0, "B1": 0.0, "B5": 0.0}
    rg_coords = coords[rgroup_idxs] - origin
    rg_nums = [atomic_nums[i] for i in rgroup_idxs]

    if len(rg_coords) == 0:
        return {"L": 0.0, "B1": 0.0, "B5": 0.0}

    # Determine principal axis (direction to center of mass of R-group)
    com = rg_coords.mean(axis=0)
    com_norm = np.linalg.norm(com)
    if com_norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = com / com_norm

    # L: max projection along axis + vdW radius of furthest atom
    projections = rg_coords @ axis
    max_proj_idx = np.argmax(projections)
    L = projections[max_proj_idx] + get_vdw_radius(rg_nums[max_proj_idx])

    # Perpendicular distances (for B1, B5)
    perp_vecs = rg_coords - np.outer(projections, axis)
    perp_dists = np.linalg.norm(perp_vecs, axis=1)

    # Add vdW radii for B values
    perp_with_vdw = perp_dists + np.array([get_vdw_radius(z) for z in rg_nums])

    B1 = perp_with_vdw.min() if len(perp_with_vdw) > 0 else 0.0
    B5 = perp_with_vdw.max() if len(perp_with_vdw) > 0 else 0.0

    return {"L": float(L), "B1": float(B1), "B5": float(B5)}


def compute_dihedrals(
    coords: np.ndarray,
    alpha_idx: int,
    carbonyl_idx: int,
    o_idx: int,
    n_idx: int,
    mol: Chem.Mol,
) -> dict:
    """Compute key dihedral angles encoded as sin/cos.

    tau1: O-C_carbonyl-C_alpha-[next_neighbor] (enolate torsion)
    tau2: C_alpha-C_carbonyl-N-[ring_neighbor] (auxiliary orientation)
    """
    result = {}

    def _dihedral(p1, p2, p3, p4):
        """Compute dihedral angle in radians."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-8 or n2_norm < 1e-8:
            return 0.0
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return np.arctan2(y, x)

    # tau1: O-C_carbonyl-C_alpha-neighbor
    # Find a neighbor of alpha that is NOT the carbonyl C
    alpha_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(alpha_idx).GetNeighbors()
                       if n.GetIdx() != carbonyl_idx and n.GetAtomicNum() != 1]
    if alpha_neighbors:
        tau1 = _dihedral(coords[o_idx], coords[carbonyl_idx],
                         coords[alpha_idx], coords[alpha_neighbors[0]])
        result["sin_tau1"] = np.sin(tau1)
        result["cos_tau1"] = np.cos(tau1)
    else:
        result["sin_tau1"] = 0.0
        result["cos_tau1"] = 1.0

    # tau2: alpha-carbonyl-N-ring_neighbor
    n_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                   if n.GetIdx() != carbonyl_idx and n.GetAtomicNum() != 1]
    if n_neighbors:
        tau2 = _dihedral(coords[alpha_idx], coords[carbonyl_idx],
                         coords[n_idx], coords[n_neighbors[0]])
        result["sin_tau2"] = np.sin(tau2)
        result["cos_tau2"] = np.cos(tau2)
    else:
        result["sin_tau2"] = 0.0
        result["cos_tau2"] = 1.0

    return result


def find_rgroup_atoms(mol: Chem.Mol, alpha_idx: int, carbonyl_idx: int) -> list[int]:
    """Find atoms belonging to the R-group on the alpha carbon.

    R-group = everything connected to alpha_C except the carbonyl_C side.
    Uses BFS from alpha, excluding carbonyl direction.
    """
    visited = {alpha_idx, carbonyl_idx}
    queue = []

    for neighbor in mol.GetAtomWithIdx(alpha_idx).GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx != carbonyl_idx and neighbor.GetAtomicNum() != 1:
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


def compute_single_conformer_descriptors(
    mol: Chem.Mol,
    coords: np.ndarray,
    center: tuple[int, int, int, int],
) -> dict | None:
    """Compute all steric descriptors for one conformer.

    Args:
        mol: RDKit Mol (no Hs)
        coords: (n_atoms, 3) heavy-atom coordinates
        center: (alpha_idx, carbonyl_c_idx, o_idx, n_idx)

    Returns dict of descriptor values, or None if computation fails.
    """
    alpha_idx, carbonyl_idx, o_idx, n_idx = center
    n_atoms = mol.GetNumAtoms()

    if coords is None or len(coords) < max(center) + 1:
        return None
    if len(coords) != n_atoms:
        return None  # atom count mismatch between mol and conformer coords

    atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)]

    # Define the reactive plane normal
    # Use alpha_C, carbonyl_C, O to define the plane
    try:
        normal = get_plane_normal(coords, alpha_idx, carbonyl_idx, o_idx)
    except Exception:
        normal = np.array([0.0, 0.0, 1.0])

    desc = {}

    # A. Buried volume (%Vbur)
    vbur = compute_buried_volume(coords, atomic_nums, alpha_idx, normal,
                                 radius=3.5, grid_spacing=0.25)
    desc.update(vbur)

    # B. Sterimol of R-group
    rgroup_idxs = find_rgroup_atoms(mol, alpha_idx, carbonyl_idx)
    sterimol = compute_sterimol(coords, atomic_nums, alpha_idx, rgroup_idxs)
    desc.update(sterimol)

    # C. Dihedral angles
    dihedrals = compute_dihedrals(coords, alpha_idx, carbonyl_idx, o_idx, n_idx, mol)
    desc.update(dihedrals)

    return desc


def compute_ensemble_descriptors(
    smiles: str,
    ensemble: dict,
) -> dict | None:
    """Compute Boltzmann-aggregated steric descriptors for a conformer ensemble.

    Args:
        smiles: SMILES of the molecule (enolate or ketone)
        ensemble: output from generate_conformer_ensemble()

    Returns dict of aggregated descriptors (mean + std), or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Find reactive center
    center = find_reactive_center(mol)
    if center is None:
        return None

    representatives = ensemble["representatives"]
    if not representatives:
        return None

    # Compute descriptors for each representative conformer
    all_desc = []
    weights = []
    for conf_id, energy, weight, coords in representatives:
        if coords is None:
            continue
        desc = compute_single_conformer_descriptors(mol, coords, center)
        if desc is not None:
            all_desc.append(desc)
            weights.append(weight)

    if not all_desc:
        return None

    # Normalize weights
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    # Get all descriptor keys
    desc_keys = sorted(all_desc[0].keys())

    # Boltzmann-weighted aggregation: mean + std
    result = {}
    for key in desc_keys:
        values = np.array([d[key] for d in all_desc])
        wmean = np.average(values, weights=weights)
        wstd = np.sqrt(np.average((values - wmean) ** 2, weights=weights))
        result[f"{key}_mean"] = wmean
        result[f"{key}_std"] = wstd

    # Add metadata
    result["n_conformers"] = len(all_desc)
    result["n_clusters"] = ensemble["n_clusters"]

    return result


# Descriptor column names (for consistent ordering)
STERIC_DESC_NAMES = [
    "Vbur_si_mean", "Vbur_si_std",
    "Vbur_re_mean", "Vbur_re_std",
    "Vbur_diff_mean", "Vbur_diff_std",
    "Vbur_total_mean", "Vbur_total_std",
    "L_mean", "L_std",
    "B1_mean", "B1_std",
    "B5_mean", "B5_std",
    "sin_tau1_mean", "sin_tau1_std",
    "cos_tau1_mean", "cos_tau1_std",
    "sin_tau2_mean", "sin_tau2_std",
    "cos_tau2_mean", "cos_tau2_std",
    "n_conformers", "n_clusters",
]
