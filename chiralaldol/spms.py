"""SPMS: Spherical Projection of Molecular Stereostructure.

Computes directional steric maps around key atoms by projecting
van der Waals surfaces onto a spherical mesh. Based on the concept
from SEMG-MIGNN (Li et al., Nature Communications 2023).

Each SPMS matrix is a (n_theta × n_phi) grid where:
  - theta: polar angle (0 → π), rows
  - phi: azimuthal angle (0 → 2π), columns
  - value: distance from mesh point to nearest vdW surface
  - smaller value = more sterically blocked direction
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# van der Waals radii (Å) from Bondi/RDKit
VDW_RADII = {
    1: 1.20, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52,
    9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75,
    22: 1.87, 35: 1.85, 53: 1.98,
}

# SMARTS for target atoms
ALPHA_C_SMARTS = Chem.MolFromSmarts("[CH2,CH;X3,X4:1]-[CX3:2](=[OX1:3])-[NX3:4]")
ALDEHYDE_C_SMARTS = Chem.MolFromSmarts("[CX3H1:1](=[OX1])")


def compute_spms(coords, atomic_nums, center_idx,
                 sphere_radius=10.0, n_theta=10, n_phi=20):
    """Compute SPMS steric map for one atom.

    Args:
        coords: (n_atoms, 3) atomic coordinates
        atomic_nums: (n_atoms,) atomic numbers
        center_idx: index of target atom
        sphere_radius: radius of projection sphere (Å)
        n_theta: polar angle resolution
        n_phi: azimuthal angle resolution

    Returns:
        (n_theta, n_phi) float32 matrix of distances to nearest vdW surface
    """
    center = coords[center_idx]
    other_mask = np.arange(len(coords)) != center_idx
    other_coords = coords[other_mask]  # (n-1, 3)
    other_radii = np.array([VDW_RADII.get(int(z), 1.70)
                            for z in atomic_nums[other_mask]])  # (n-1,)

    # Build spherical mesh
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')  # (n_theta, n_phi)

    mesh_pts = np.stack([
        sphere_radius * np.sin(T) * np.cos(P) + center[0],
        sphere_radius * np.sin(T) * np.sin(P) + center[1],
        sphere_radius * np.cos(T) + center[2],
    ], axis=-1)  # (n_theta, n_phi, 3)

    # Vectorized distance: (n_theta, n_phi, n_other)
    diff = mesh_pts[:, :, np.newaxis, :] - other_coords[np.newaxis, np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # (n_theta, n_phi, n_other)

    # Penetration depth = distance - vdW radius
    penetration = dists - other_radii[np.newaxis, np.newaxis, :]  # (n_theta, n_phi, n_other)
    steric_map = penetration.min(axis=-1).astype(np.float32)  # (n_theta, n_phi)

    return steric_map


def standardize_orientation(coords, center_idx):
    """Standardize molecular orientation for reproducible SPMS.

    1. Center target atom at origin
    2. Nearest bonded neighbor aligned to +z axis
    3. Second nearest neighbor placed in yz plane

    Returns: (n_atoms, 3) rotated coordinates
    """
    coords = coords.copy().astype(np.float64)
    center = coords[center_idx].copy()
    coords -= center  # translate

    # Find nearest atom
    dists = np.linalg.norm(coords, axis=1)
    dists[center_idx] = np.inf
    nearest_idx = np.argmin(dists)

    # Rotate nearest to +z
    v = coords[nearest_idx]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        return coords.astype(np.float32)
    v = v / v_norm
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(v, z):
        R1 = np.eye(3)
    elif np.allclose(v, -z):
        R1 = np.diag([1.0, -1.0, -1.0])
    else:
        axis = np.cross(v, z)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v, z), -1, 1))
        R1 = _rotation_matrix(axis, angle)

    coords = (R1 @ coords.T).T

    # Find second nearest and rotate to yz plane
    dists[nearest_idx] = np.inf
    second_idx = np.argmin(dists)
    w = coords[second_idx]
    # Project to xy plane, rotate so x=0
    angle_xy = np.arctan2(w[0], w[1])
    R2 = _rotation_matrix(np.array([0.0, 0.0, 1.0]), -angle_xy)
    coords = (R2 @ coords.T).T

    return coords.astype(np.float32)


def _rotation_matrix(axis, angle):
    """Rodrigues' rotation formula."""
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ])


def find_alpha_carbon(mol):
    """Find α-carbon index in ketone/enolate precursor.

    Returns atom index or -1 if not found.
    """
    match = mol.GetSubstructMatch(ALPHA_C_SMARTS)
    if match:
        return match[0]  # :1 tagged atom
    return -1


def find_aldehyde_carbon(mol):
    """Find aldehyde carbon index.

    Returns atom index or -1 if not found.
    """
    match = mol.GetSubstructMatch(ALDEHYDE_C_SMARTS)
    if match:
        return match[0]
    return -1


def compute_spms_for_conformer(mol, coords, target_idx,
                               sphere_radius=10.0, n_theta=10, n_phi=20):
    """Compute SPMS for one conformer at a target atom.

    Args:
        mol: RDKit Mol (with Hs)
        coords: (n_atoms, 3) coordinates for this conformer
        target_idx: atom index to compute SPMS at
        sphere_radius, n_theta, n_phi: SPMS parameters

    Returns:
        (n_theta, n_phi) steric map, or None if target not found
    """
    if target_idx < 0 or target_idx >= len(coords):
        return None

    atomic_nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

    # Standardize orientation
    std_coords = standardize_orientation(coords, target_idx)

    return compute_spms(std_coords, atomic_nums, target_idx,
                        sphere_radius, n_theta, n_phi)


def compute_spms_ensemble(mol, representatives, target_idx,
                          sphere_radius=10.0, n_theta=10, n_phi=20):
    """Compute Boltzmann-weighted SPMS over conformer ensemble.

    Args:
        mol: RDKit Mol
        representatives: list of (cid, energy, weight, coords) tuples
        target_idx: atom index

    Returns:
        (n_theta, n_phi) Boltzmann-weighted average SPMS, or None
    """
    if target_idx < 0:
        return None

    maps = []
    weights = []
    for _, _, weight, coords in representatives:
        smap = compute_spms_for_conformer(
            mol, coords, target_idx, sphere_radius, n_theta, n_phi)
        if smap is not None:
            maps.append(smap)
            weights.append(weight)

    if not maps:
        return None

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    result = sum(w * m for w, m in zip(weights, maps))
    return result.astype(np.float32)
