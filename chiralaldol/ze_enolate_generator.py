"""Z/E Enolate Generator V2 — 3D dihedral marking + coordMap constrained embedding.

Algorithm (3D marking method):
  1. ketone → enolate mol (reuse ACYL_ALPHA_SMARTS)
  2. Embed 1 seed conformer (ETKDGv3)
  3. Find alpha_nb—alpha—cc—O dihedral atoms
  4. For Z: SetDihedralDeg → 0°, then constrained EmbedMultipleConfs(100)
  5. For E: SetDihedralDeg → 180°, then constrained EmbedMultipleConfs(100)
  6. For Ketone: plain EmbedMultipleConfs(100)
  7. No MMFF optimization — use single-point energy for Boltzmann weighting

Z/E weights by base/activator:
  Bu2BOTf/DIPEA → 98% Z + 2% E
  LDA/LiHMDS   → 95% Z + 5% E
  Et3N          → 70% Z + 30% E
  default       → 50% Z + 50% E
"""

import copy
import logging

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

from .utils import ACYL_ALPHA_SMARTS, clean_mol

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

N_CONFS = 100
RMSD_CUTOFF = 1.0
MAX_CLUSTERS = 8
RT_298K = 0.5922

# Z/E weights
ZE_WEIGHTS_ACTIVATOR = {
    "Bu2BOTf": (0.98, 0.02), "Chx2BCl": (0.98, 0.02),
    "Ipc2BCl": (0.98, 0.02), "9BBN_OTf": (0.98, 0.02),
    "TiCl4": (0.90, 0.10), "Sn_OTf2": (0.85, 0.15),
    "MgCl2": (0.80, 0.20), "BF3_OEt2": (0.70, 0.30),
}
ZE_WEIGHTS_BASE = {
    "DIPEA": (0.98, 0.02), "LDA": (0.95, 0.05),
    "LiHMDS": (0.95, 0.05), "NaHMDS": (0.95, 0.05),
    "KHMDS": (0.95, 0.05), "Et3N": (0.70, 0.30),
    "other_base": (0.50, 0.50), "no_base": (0.50, 0.50),
}


def get_ze_weights(base: str, activator: str = "") -> tuple[float, float]:
    if activator and activator in ZE_WEIGHTS_ACTIVATOR:
        return ZE_WEIGHTS_ACTIVATOR[activator]
    if base in ZE_WEIGHTS_BASE:
        return ZE_WEIGHTS_BASE[base]
    return (0.50, 0.50)


def _single_point_energy(mol, conf_id: int) -> float:
    """MMFF single-point energy (no optimization)."""
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return 0.0
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return 0.0
        return ff.CalcEnergy()
    except Exception:
        return 0.0


def _cluster_and_weight(mol, valid_cids, energies) -> list[tuple]:
    """Greedy RMSD clustering + Boltzmann weighting."""
    if not valid_cids:
        return []

    order = np.argsort(energies)
    sorted_cids = [valid_cids[i] for i in order]
    sorted_energies = [energies[i] for i in order]

    reps = []
    for cid, energy in zip(sorted_cids, sorted_energies):
        is_new = True
        for rep_cid, _, _, _ in reps:
            try:
                rmsd = rdMolAlign.GetBestRMS(mol, mol, prbId=cid, refId=rep_cid)
                if rmsd < RMSD_CUTOFF:
                    is_new = False
                    break
            except Exception:
                continue
        if is_new:
            coords = mol.GetConformer(cid).GetPositions()
            reps.append((cid, energy, 1.0, coords))
            if len(reps) >= MAX_CLUSTERS:
                break

    if len(reps) > 1:
        e_min = reps[0][1]
        weights = [np.exp(-(e - e_min) / RT_298K) for _, e, _, _ in reps]
        w_sum = sum(weights)
        reps = [(cid, e, w / w_sum, coords) for (cid, e, _, coords), w in zip(reps, weights)]

    return reps


def _embed_plain(mol, n_confs=N_CONFS) -> dict | None:
    """Plain ETKDGv3 embedding (no MMFF), for ketone or aldehyde."""
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if len(cids) == 0:
            return None

    energies = [_single_point_energy(mol, cid) for cid in range(mol.GetNumConformers())]
    valid = [(i, e) for i, e in enumerate(energies) if e < 1e8]
    if not valid:
        valid = [(i, 0.0) for i in range(mol.GetNumConformers())]

    reps = _cluster_and_weight(mol, [v[0] for v in valid], [v[1] for v in valid])
    if not reps:
        return None

    return {"mol": mol, "representatives": reps,
            "n_conformers": len(valid), "n_clusters": len(reps)}


def _embed_constrained_ze(enolate_h, alpha_nb, alpha_idx, cc_idx, o_idx,
                           target_dihedral: float) -> dict | None:
    """Embed 100 conformers with Z or E geometry constrained via coordMap.

    1. Embed 1 seed conformer
    2. SetDihedralDeg to target (0° for Z, 180° for E)
    3. Use seed atom positions as coordMap for constrained multi-conf embedding
    """
    mol = copy.deepcopy(enolate_h)

    # Step 1: embed seed
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    seed_cid = AllChem.EmbedMolecule(mol, params)
    if seed_cid == -1:
        params.useRandomCoords = True
        seed_cid = AllChem.EmbedMolecule(mol, params)
        if seed_cid == -1:
            return None

    # Step 2: force dihedral
    conf = mol.GetConformer(seed_cid)
    rdMolTransforms.SetDihedralDeg(conf, alpha_nb, alpha_idx, cc_idx, o_idx, target_dihedral)

    # Step 3: build coordMap from key atoms (fix C=C and O positions)
    from rdkit.Geometry import Point3D
    coord_map = {}
    for idx in [alpha_idx, cc_idx, o_idx]:
        pos = conf.GetAtomPosition(idx)
        coord_map[idx] = Point3D(pos.x, pos.y, pos.z)

    # Remove seed conformer, embed with coordMap constraint
    # Note: EmbedMultipleConfs with coordMap uses legacy API (not params object)
    mol.RemoveAllConformers()
    cids = AllChem.EmbedMultipleConfs(
        mol, numConfs=N_CONFS, coordMap=coord_map,
        randomSeed=42, numThreads=1,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
        enforceChirality=True,
    )
    if len(cids) == 0:
        # Fallback: embed without coordMap, then filter by dihedral
        mol.RemoveAllConformers()
        params3 = AllChem.ETKDGv3()
        params3.randomSeed = 42
        params3.numThreads = 0
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS * 2, params=params3)
        if len(cids) == 0:
            return None

    # Filter: only keep conformers where dihedral is near target
    target_range = 90.0  # ±90° from target
    good_cids = []
    good_energies = []
    for cid in range(mol.GetNumConformers()):
        d = rdMolTransforms.GetDihedralDeg(mol.GetConformer(cid),
                                            alpha_nb, alpha_idx, cc_idx, o_idx)
        diff = abs(d - target_dihedral)
        if diff > 180:
            diff = 360 - diff
        if diff < target_range:
            e = _single_point_energy(mol, cid)
            good_cids.append(cid)
            good_energies.append(e)

    if not good_cids:
        return None

    reps = _cluster_and_weight(mol, good_cids, good_energies)
    if not reps:
        return None

    return {"mol": mol, "representatives": reps,
            "n_conformers": len(good_cids), "n_clusters": len(reps)}


def _make_enolate_h(smiles: str):
    """Convert ketone → enolate mol (AddHs) + key atom indices."""
    mol = clean_mol(smiles)
    if mol is None:
        return None

    matches = mol.GetSubstructMatches(ACYL_ALPHA_SMARTS)
    if not matches:
        return None

    alpha_idx, cc_idx, o_idx, n_idx = matches[0]

    try:
        rwmol = Chem.RWMol(mol)
        rwmol.GetBondBetweenAtoms(alpha_idx, cc_idx).SetBondType(Chem.BondType.DOUBLE)
        rwmol.GetBondBetweenAtoms(cc_idx, o_idx).SetBondType(Chem.BondType.SINGLE)
        rwmol.GetAtomWithIdx(o_idx).SetFormalCharge(-1)
        alpha = rwmol.GetAtomWithIdx(alpha_idx)
        if alpha.GetNumExplicitHs() > 0:
            alpha.SetNumExplicitHs(alpha.GetNumExplicitHs() - 1)
        alpha.SetNoImplicit(False)
        Chem.SanitizeMol(rwmol)
        enolate = rwmol.GetMol()
    except Exception:
        return None

    enolate_h = Chem.AddHs(enolate)

    # Find heavy-atom neighbor of alpha (NOT cc) for dihedral
    alpha_nb = None
    for nb in enolate_h.GetAtomWithIdx(alpha_idx).GetNeighbors():
        if nb.GetIdx() != cc_idx and nb.GetAtomicNum() > 1:
            alpha_nb = nb.GetIdx()
            break
    if alpha_nb is None:
        return None

    return enolate_h, alpha_idx, cc_idx, o_idx, alpha_nb


def generate_ze_conformers(ketone_smi: str) -> dict | None:
    """Generate Ketone + Z-enolate + E-enolate conformer ensembles.

    Returns dict with keys: ketone, z_enolate, e_enolate (each an ensemble dict).
    """
    if pd.isna(ketone_smi) or not str(ketone_smi).strip():
        return None

    result = {}

    # Ketone conformers
    ketone_mol = clean_mol(ketone_smi)
    if ketone_mol is None:
        return None
    ketone_h = Chem.AddHs(ketone_mol)
    k_ens = _embed_plain(ketone_h)
    if k_ens is None:
        return None
    result["ketone"] = k_ens

    # Enolate setup
    enolate_data = _make_enolate_h(ketone_smi)
    if enolate_data is None:
        return None
    enolate_h, alpha_idx, cc_idx, o_idx, alpha_nb = enolate_data

    # Z-enolate (dihedral ≈ 0°)
    z_ens = _embed_constrained_ze(enolate_h, alpha_nb, alpha_idx, cc_idx, o_idx,
                                   target_dihedral=0.0)
    result["z_enolate"] = z_ens

    # E-enolate (dihedral ≈ 180°)
    e_ens = _embed_constrained_ze(enolate_h, alpha_nb, alpha_idx, cc_idx, o_idx,
                                   target_dihedral=180.0)
    result["e_enolate"] = e_ens

    return result
