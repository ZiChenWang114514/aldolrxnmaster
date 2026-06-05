"""M2: Conformer Ensemble Sampler — Multi-conformer ETKDG + MMFF + RMSD clustering.

Pipeline:
  1. Generate N conformers per molecule (ETKDG v3)
  2. Optimize with MMFF94s force field
  3. Filter high-energy conformers (>10 kcal/mol above minimum)
  4. RMSD-based hierarchical clustering → K representative conformers
  5. Boltzmann weight each representative (T=298K)
"""

import logging

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# Constants
RT_298K = 0.5922  # kcal/mol at 298K (R * T)
ENERGY_CUTOFF = 10.0  # kcal/mol above minimum
DEFAULT_N_CONFS = 100
DEFAULT_RMSD_CUTOFF = 1.0  # Angstroms for clustering
DEFAULT_MAX_K = 8  # maximum number of clusters
DEFAULT_N_THREADS = 8  # higher throughput (~4-6 cores)


def generate_conformer_ensemble(
    smiles: str,
    n_confs: int = DEFAULT_N_CONFS,
    rmsd_cutoff: float = DEFAULT_RMSD_CUTOFF,
    max_k: int = DEFAULT_MAX_K,
    n_threads: int = DEFAULT_N_THREADS,
    seed: int = 42,
) -> dict | None:
    """Generate a conformer ensemble for a single molecule.

    Returns dict with:
        mol: RDKit Mol with all valid conformers
        representatives: list of (conf_id, energy, weight, coords)
        n_total: total conformers generated
        n_valid: conformers after energy filter
        n_clusters: number of clusters
    Or None if generation fails.
    """
    if not smiles or str(smiles).strip() == "" or str(smiles) == "nan":
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Step 1: Generate multiple conformers with ETKDG v3
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = n_threads  # limit CPU usage
    params.pruneRmsThresh = 0.1  # light pre-pruning

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        return None

    # Step 2: MMFF optimization + energy collection
    energies = {}
    for cid in cids:
        try:
            result = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                continue
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                energies[cid] = ff.CalcEnergy()
        except Exception:
            continue

    if not energies:
        return None

    # Step 3: Filter high-energy conformers
    e_min = min(energies.values())
    valid_cids = [cid for cid, e in energies.items() if (e - e_min) <= ENERGY_CUTOFF]

    if len(valid_cids) < 2:
        # Only one valid conformer — return it directly
        cid = valid_cids[0]
        mol_noh = Chem.RemoveHs(mol)
        coords = _extract_coords(mol_noh, cid if cid in [c.GetId() for c in mol_noh.GetConformers()] else 0)
        if coords is None:
            coords = _extract_heavy_coords(mol, cid)
        return {
            "representatives": [(cid, energies[cid], 1.0, coords)],
            "n_total": len(cids),
            "n_valid": 1,
            "n_clusters": 1,
        }

    # Step 4: Compute pairwise RMSD matrix (heavy atoms only)
    # Optimized: pre-align to reference + numpy vectorized RMSD
    n_valid = len(valid_cids)

    # Pre-align all conformers to the first one (N alignments, not N²)
    ref_cid = valid_cids[0]
    for cid in valid_cids[1:]:
        try:
            rdMolAlign.AlignMol(mol, mol, prbCid=cid, refCid=ref_cid)
        except Exception:
            pass

    # Extract heavy-atom coordinates for all valid conformers
    heavy_idxs = [i for i in range(mol.GetNumAtoms())
                  if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    coords_all = np.zeros((n_valid, len(heavy_idxs), 3), dtype=np.float64)
    for ci, cid in enumerate(valid_cids):
        conf = mol.GetConformer(cid)
        for hi, ai in enumerate(heavy_idxs):
            pos = conf.GetAtomPosition(ai)
            coords_all[ci, hi] = [pos.x, pos.y, pos.z]

    # Vectorized pairwise RMSD: O(n²) but in numpy, not Python loops
    diff = coords_all[:, None] - coords_all[None, :]  # (n, n, atoms, 3)
    rmsd_matrix = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=-1))  # (n, n)

    # Step 5: Hierarchical clustering
    condensed = squareform(rmsd_matrix)
    if len(condensed) == 0:
        # fallback for very few conformers
        cid = valid_cids[0]
        coords = _extract_heavy_coords(mol, cid)
        return {
            "representatives": [(cid, energies[cid], 1.0, coords)],
            "n_total": len(cids),
            "n_valid": n_valid,
            "n_clusters": 1,
        }

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=rmsd_cutoff, criterion="distance")
    n_clusters = min(len(set(labels)), max_k)

    # If too many clusters, re-cluster with larger cutoff
    if len(set(labels)) > max_k:
        labels = fcluster(Z, t=max_k, criterion="maxclust")
        n_clusters = max_k

    # Step 6: Select representative from each cluster (lowest energy)
    cluster_reps = []
    for k in range(1, n_clusters + 1):
        members = [i for i, l in enumerate(labels) if l == k]
        if not members:
            continue
        best_i = min(members, key=lambda i: energies[valid_cids[i]])
        cluster_reps.append((valid_cids[best_i], energies[valid_cids[best_i]]))

    # Step 7: Boltzmann weights
    rep_energies = np.array([e for _, e in cluster_reps])
    rep_energies_rel = rep_energies - rep_energies.min()
    boltzmann = np.exp(-rep_energies_rel / RT_298K)
    weights = boltzmann / boltzmann.sum()

    # Extract heavy-atom coordinates for each representative
    representatives = []
    for (cid, energy), w in zip(cluster_reps, weights):
        coords = _extract_heavy_coords(mol, cid)
        representatives.append((cid, energy, w, coords))

    return {
        "representatives": representatives,
        "n_total": len(cids),
        "n_valid": n_valid,
        "n_clusters": n_clusters,
    }


def _extract_heavy_coords(mol_with_h: Chem.Mol, conf_id: int) -> np.ndarray | None:
    """Extract heavy-atom coordinates from a mol with Hs."""
    try:
        conf = mol_with_h.GetConformer(conf_id)
        coords = []
        for i in range(mol_with_h.GetNumAtoms()):
            if mol_with_h.GetAtomWithIdx(i).GetAtomicNum() != 1:
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
        return np.array(coords, dtype=np.float32)
    except Exception:
        return None


def _extract_coords(mol: Chem.Mol, conf_id: int) -> np.ndarray | None:
    """Extract coordinates from a mol (no H's expected)."""
    try:
        conf = mol.GetConformer(conf_id)
        return np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
    except Exception:
        return None


