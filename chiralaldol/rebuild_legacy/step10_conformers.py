"""Step 10: Fast conformer generation — ETKDGv3 only (no MMFF) + multiprocessing.

Key changes from V3.1:
  - Skip MMFF optimization entirely (ETKDGv3 has CSD torsion knowledge)
  - Use MMFF single-point energy for Boltzmann weighting (no optimization)
  - 100 conformers per molecule
  - multiprocessing.Pool(8) for parallel processing
  - Deduplication by SMILES
  - 30s timeout per molecule
"""

import logging
import pickle
import signal
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

N_CONFS = 100
RMSD_CUTOFF = 1.0
MAX_CLUSTERS = 8
RT_298K = 0.5922
TIMEOUT_SEC = 30
N_WORKERS = 8


def _single_point_energy(mol, conf_id: int) -> float:
    """Compute MMFF single-point energy (no optimization) for Boltzmann weighting."""
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


def _generate_one(smi: str) -> dict | None:
    """Generate conformer ensemble for one molecule. Called in worker process."""
    if not smi or smi == "nan":
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # ETKDGv3 embedding only — no MMFF optimization
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params)
    if len(cids) == 0:
        # Fallback: random coords
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params)
        if len(cids) == 0:
            return None

    # Single-point energies for Boltzmann weighting
    energies = []
    valid_cids = []
    for cid in range(mol.GetNumConformers()):
        e = _single_point_energy(mol, cid)
        if e < 1e8:
            energies.append(e)
            valid_cids.append(cid)

    if not valid_cids:
        # Use all conformers with uniform weight if energy fails
        valid_cids = list(range(mol.GetNumConformers()))
        energies = [0.0] * len(valid_cids)

    # Greedy RMSD clustering
    order = np.argsort(energies)
    sorted_cids = [valid_cids[i] for i in order]
    sorted_energies = [energies[i] for i in order]

    representatives = []
    for cid, energy in zip(sorted_cids, sorted_energies):
        is_new = True
        for rep_cid, _, _, _ in representatives:
            try:
                rmsd = rdMolAlign.GetBestRMS(mol, mol, prbId=cid, refId=rep_cid)
                if rmsd < RMSD_CUTOFF:
                    is_new = False
                    break
            except Exception:
                continue
        if is_new:
            coords = mol.GetConformer(cid).GetPositions()
            representatives.append((cid, energy, 1.0, coords))
            if len(representatives) >= MAX_CLUSTERS:
                break

    if not representatives:
        return None

    # Boltzmann weights
    if len(representatives) > 1:
        e_min = representatives[0][1]
        weights = [np.exp(-(e - e_min) / RT_298K) for _, e, _, _ in representatives]
        w_sum = sum(weights)
        representatives = [
            (cid, e, w / w_sum, coords)
            for (cid, e, _, coords), w in zip(representatives, weights)
        ]

    return {
        "mol": mol,
        "representatives": representatives,
        "n_conformers": len(valid_cids),
        "n_clusters": len(representatives),
    }


def _worker_fn(args):
    """Worker function for multiprocessing."""
    smi, timeout = args
    # Set alarm for timeout
    old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError))
    signal.alarm(timeout)
    try:
        result = _generate_one(smi)
    except (TimeoutError, Exception):
        result = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result


def run(context: dict) -> dict:
    """Generate conformer ensembles: no MMFF, 100 confs, 8 workers."""
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    n_start = len(df)
    logger.info(f"Step 10: Fast conformer generation for {n_start} rows (no MMFF, {N_CONFS} confs, {N_WORKERS} workers)")

    ketone_col = "canonical_Ketone" if "canonical_Ketone" in df.columns else "Ketone"
    aldehyde_col = "canonical_Aldehyde" if "canonical_Aldehyde" in df.columns else "Aldehyde"

    # Enolate generation
    try:
        import sys
        sys.path.insert(0, str(context["project_dir"]))
        from chiralaldol.enolate_generator import ketone_to_enolate
        has_enolate = True
    except ImportError:
        has_enolate = False

    # Deduplicate
    unique_ketones = [str(s) for s in df[ketone_col].dropna().unique()]
    unique_aldehydes = [str(s) for s in df[aldehyde_col].dropna().unique()]

    # Build enolate SMILES
    enolate_map = {}
    for ksmi in unique_ketones:
        if has_enolate:
            try:
                result = ketone_to_enolate(ksmi)
                esmi = result[0] if isinstance(result, tuple) else result
                enolate_map[ksmi] = esmi if esmi and esmi != "parse_fail" else ksmi
            except Exception:
                enolate_map[ksmi] = ksmi
        else:
            enolate_map[ksmi] = ksmi

    unique_enolates = list(set(enolate_map.values()))
    logger.info(f"  Unique: {len(unique_enolates)} enolates, {len(unique_aldehydes)} aldehydes")

    # Parallel conformer generation
    logger.info(f"  Generating enolate conformers ({len(unique_enolates)} unique)...")
    with Pool(N_WORKERS) as pool:
        enolate_results = pool.map(_worker_fn, [(s, TIMEOUT_SEC) for s in unique_enolates])
    enolate_cache = dict(zip(unique_enolates, enolate_results))
    n_ok = sum(1 for v in enolate_results if v is not None)
    logger.info(f"  Enolates: {n_ok}/{len(unique_enolates)} success")

    logger.info(f"  Generating aldehyde conformers ({len(unique_aldehydes)} unique)...")
    with Pool(N_WORKERS) as pool:
        aldehyde_results = pool.map(_worker_fn, [(s, TIMEOUT_SEC) for s in unique_aldehydes])
    aldehyde_cache = dict(zip(unique_aldehydes, aldehyde_results))
    n_ok = sum(1 for v in aldehyde_results if v is not None)
    logger.info(f"  Aldehydes: {n_ok}/{len(unique_aldehydes)} success")

    # Map back to rows
    conformer_ensembles = {}
    status_list = []

    for _, row in df.iterrows():
        oi = row["original_index"]
        ksmi = row.get(ketone_col)
        asmi = row.get(aldehyde_col)

        enolate_ens = None
        aldehyde_ens = None

        if pd.notna(ksmi):
            esmi = enolate_map.get(str(ksmi), str(ksmi))
            enolate_ens = enolate_cache.get(esmi)
        if pd.notna(asmi):
            aldehyde_ens = aldehyde_cache.get(str(asmi))

        if enolate_ens is None or aldehyde_ens is None:
            status_list.append("failed")
            audit.mark_deleted_by_oi([oi], "conformer_generation_failed")
        else:
            status_list.append("success")
            conformer_ensembles[oi] = {"enolate": enolate_ens, "aldehyde": aldehyde_ens}

    df["conformer_status"] = status_list
    n_failed = sum(1 for s in status_list if s == "failed")
    logger.info(f"  Results: {n_start - n_failed} success, {n_failed} failed")

    if n_failed > 0:
        df = df[df["conformer_status"] == "success"].reset_index(drop=True)
        logger.info(f"  Deleted {n_failed} rows")

    if len(df) == 0:
        raise RuntimeError("Step 10: All conformer generations failed")

    conf_path = context["output_dir"] / "interim" / "10_conformers.pkl"
    with open(conf_path, "wb") as f:
        pickle.dump(conformer_ensembles, f)
    logger.info(f"  Saved {len(conformer_ensembles)} ensembles")

    context["df"] = df
    context["conformer_ensembles"] = conformer_ensembles
    logger.info(f"  Step 10 complete: {n_start} → {len(df)} rows")
    return context
