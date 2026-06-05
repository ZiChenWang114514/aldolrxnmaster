"""Face Steric Map: directional steric maps for ZT ring faces.

Computes 2D steric maps for the si-face and re-face of a molecular plane,
capturing which regions above/below a reference plane are blocked.

Two modes:
1. From conformer coords + reference plane (for tree models, any auxiliary)
2. From ZT ring 3D coords (for ZT-GNN, Evans-only)
"""

import numpy as np

from .utils import VDW_RADII
from .spms import ALPHA_C_SMARTS


def compute_ring_normal(ring_coords):
    """Compute normal vector of a ring from atom coordinates.

    Uses SVD to find best-fit plane for potentially non-planar rings.
    """
    center = ring_coords.mean(axis=0)
    centered = ring_coords - center
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # smallest singular value → normal direction
    return normal / np.linalg.norm(normal)


def _build_face_grid(center, normal, u_axis, grid_size, grid_extent):
    """Build 2D grid points on one face of a plane."""
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    ticks = np.linspace(-grid_extent, grid_extent, grid_size)
    U, V = np.meshgrid(ticks, ticks, indexing='ij')  # (grid_size, grid_size)

    # Grid points at distance=grid_extent above the plane
    grid_pts = (center[np.newaxis, np.newaxis, :]
                + U[:, :, np.newaxis] * u_axis[np.newaxis, np.newaxis, :]
                + V[:, :, np.newaxis] * v_axis[np.newaxis, np.newaxis, :]
                + grid_extent * normal[np.newaxis, np.newaxis, :])

    return grid_pts  # (grid_size, grid_size, 3)


def compute_face_maps(coords, atomic_nums, center, normal,
                      grid_size=10, grid_extent=5.0):
    """Compute steric maps for both faces of a plane.

    Args:
        coords: (n_atoms, 3) all atom coordinates
        atomic_nums: (n_atoms,) atomic numbers
        center: (3,) center of the reference plane
        normal: (3,) normal vector of the plane
        grid_size: resolution of each face map
        grid_extent: extent of the grid in Å

    Returns:
        si_map: (grid_size, grid_size) steric map for +normal face
        re_map: (grid_size, grid_size) steric map for -normal face
    """
    radii = np.array([VDW_RADII.get(int(z), 1.70) for z in atomic_nums])

    # Choose u-axis perpendicular to normal
    if abs(normal[2]) < 0.9:
        u_axis = np.cross(normal, [0, 0, 1])
    else:
        u_axis = np.cross(normal, [1, 0, 0])
    u_axis /= np.linalg.norm(u_axis)

    maps = []
    for sign in [1.0, -1.0]:  # si-face (+n), re-face (-n)
        n_dir = sign * normal
        grid_pts = _build_face_grid(center, n_dir, u_axis, grid_size, grid_extent)

        # Distance from each grid point to each atom's vdW surface
        diff = grid_pts[:, :, np.newaxis, :] - coords[np.newaxis, np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)  # (grid_size, grid_size, n_atoms)
        penetration = dists - radii[np.newaxis, np.newaxis, :]
        face_map = penetration.min(axis=-1).astype(np.float32)
        maps.append(face_map)

    return maps[0], maps[1]  # si_map, re_map


def compute_face_maps_from_conformer(mol, coords, grid_size=10, grid_extent=5.0):
    """Compute face steric maps using α-carbon plane as reference.

    The reference plane is defined by:
    - center: α-carbon position
    - normal: cross product of (C_carbonyl - C_alpha) × (neighbor - C_alpha)

    Returns:
        (si_map, re_map) each (grid_size, grid_size), or (None, None)
    """
    # Find α-carbon and its neighbors
    match = mol.GetSubstructMatch(ALPHA_C_SMARTS)
    if not match:
        return None, None

    alpha_idx = match[0]      # C_alpha (:1)
    carbonyl_idx = match[1]   # C_carbonyl (:2)

    center = coords[alpha_idx]
    atomic_nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

    # Build reference plane from C_alpha neighbors
    v1 = coords[carbonyl_idx] - center
    v1 /= np.linalg.norm(v1) + 1e-8

    # Find another neighbor of alpha carbon for the plane
    alpha_atom = mol.GetAtomWithIdx(alpha_idx)
    nbr_indices = [n.GetIdx() for n in alpha_atom.GetNeighbors()
                   if n.GetIdx() != carbonyl_idx]
    if not nbr_indices:
        return None, None

    v2 = coords[nbr_indices[0]] - center
    v2 /= np.linalg.norm(v2) + 1e-8

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None, None
    normal /= norm

    return compute_face_maps(coords, atomic_nums, center, normal,
                             grid_size, grid_extent)


def compute_face_maps_ensemble(mol, representatives, grid_size=10, grid_extent=5.0):
    """Boltzmann-weighted face maps over conformer ensemble.

    Returns:
        (si_map, re_map) each (grid_size, grid_size), or (None, None)
    """
    si_maps, re_maps, weights = [], [], []

    for _, _, weight, coords in representatives:
        si, re = compute_face_maps_from_conformer(mol, coords, grid_size, grid_extent)
        if si is not None:
            si_maps.append(si)
            re_maps.append(re)
            weights.append(weight)

    if not si_maps:
        return None, None

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    si_avg = sum(w * m for w, m in zip(weights, si_maps)).astype(np.float32)
    re_avg = sum(w * m for w, m in zip(weights, re_maps)).astype(np.float32)
    return si_avg, re_avg


def extract_face_map_features(si_map, re_map):
    """Extract statistical features from face maps for tree models.

    Returns: dict of feature name → value (32 features total)
    """
    feats = {}
    for name, fmap in [("si", si_map), ("re", re_map)]:
        feats[f"face_{name}_mean"] = float(fmap.mean())
        feats[f"face_{name}_std"] = float(fmap.std())
        feats[f"face_{name}_min"] = float(fmap.min())
        feats[f"face_{name}_max"] = float(fmap.max())
        feats[f"face_{name}_q25"] = float(np.percentile(fmap, 25))
        feats[f"face_{name}_q75"] = float(np.percentile(fmap, 75))
        # Quadrant analysis (which quadrant is most blocked)
        h, w = fmap.shape
        feats[f"face_{name}_tl"] = float(fmap[:h//2, :w//2].mean())
        feats[f"face_{name}_tr"] = float(fmap[:h//2, w//2:].mean())
        feats[f"face_{name}_bl"] = float(fmap[h//2:, :w//2].mean())
        feats[f"face_{name}_br"] = float(fmap[h//2:, w//2:].mean())

    # Cross-face features (most informative for selectivity)
    feats["face_diff_mean"] = feats["face_si_mean"] - feats["face_re_mean"]
    feats["face_diff_min"] = feats["face_si_min"] - feats["face_re_min"]
    feats["face_ratio"] = (feats["face_si_mean"] /
                           (feats["face_re_mean"] + 1e-8))
    feats["face_asymmetry"] = abs(feats["face_diff_mean"])

    return feats  # 24 features
