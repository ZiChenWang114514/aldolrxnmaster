"""Shared utilities for ChiralAldol pipeline."""

import numpy as np
from rdkit import Chem


# SMARTS: alpha-C (sp3, with H's) bonded to acyl C(=O) bonded to amide N
# Matches N-acyl chain in Evans / Crimmins / Oppolzer auxiliaries
ACYL_ALPHA_SMARTS = Chem.MolFromSmarts("[CH2,CH;X3,X4:1]-[CX3:2](=[OX1:3])-[NX3:4]")


def clean_mol(smiles: str) -> Chem.Mol | None:
    """Parse SMILES, remove atom mappings, return canonical Mol."""
    if not smiles or str(smiles).strip() == "" or str(smiles) == "nan":
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


def get_plane_normal(coords: np.ndarray, idx_a: int, idx_b: int, idx_c: int) -> np.ndarray:
    """Compute plane normal from three atom indices.

    Returns unit normal vector n = (b-a) x (c-a), normalized.
    """
    v1 = coords[idx_b] - coords[idx_a]
    v2 = coords[idx_c] - coords[idx_a]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


# Van der Waals radii (Bondi, 1964) in Angstroms
VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
    5: 1.92,   # B
    14: 2.10,  # Si
}


def get_vdw_radius(atomic_num: int) -> float:
    """Get van der Waals radius for an element."""
    return VDW_RADII.get(atomic_num, 1.70)


def wmean(pairs):
    """Weighted mean of (value, weight) pairs. Returns NaN if empty."""
    if not pairs:
        return float("nan")
    a = np.array([p[0] for p in pairs])
    w = np.array([p[1] for p in pairs])
    return float(np.average(a, weights=w))
