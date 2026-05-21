#!/usr/bin/env python3
"""MechAware: Z/E/Ketone conformer generation + steric features.

Fast: ETKDGv3 only (no MMFF opt), 100 confs, 8 workers, 3D dihedral marking for Z/E.
"""

import logging
import pickle
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from chiralaldol.ze_enolate_generator import generate_ze_conformers, N_CONFS
from chiralaldol.steric_descriptors import (
    STERIC_DESC_NAMES, find_reactive_center,
    compute_single_conformer_descriptors,
)

MECHAWARE_DIR = PROJECT_DIR / "data" / "v3" / "mechaware"
N_WORKERS = 8
TIMEOUT_SEC = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(MECHAWARE_DIR / "ze_conformers.log", mode="w", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("ze_conformers")


def _worker_ze(args):
    """Worker: generate Z/E/Ketone conformers for one ketone SMILES."""
    ksmi, timeout = args
    old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError))
    signal.alarm(timeout)
    try:
        result = generate_ze_conformers(ksmi)
    except (TimeoutError, Exception):
        result = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result


def _get_heavy_indices(mol_h):
    return [i for i in range(mol_h.GetNumAtoms()) if mol_h.GetAtomWithIdx(i).GetAtomicNum() > 1]


def _compute_steric(mol_h, ensemble: dict) -> dict | None:
    """Compute 24d steric from ensemble, handling H-atom coord mapping."""
    reps = ensemble.get("representatives", [])
    if not reps or mol_h is None:
        return None

    mol_no_h = Chem.RemoveHs(mol_h)
    center = find_reactive_center(mol_no_h)
    if center is None:
        return None

    heavy_idx = _get_heavy_indices(mol_h)
    if len(heavy_idx) != mol_no_h.GetNumAtoms():
        return None

    all_desc, weights = [], []
    for _, energy, weight, coords_full in reps:
        if coords_full is None:
            continue
        coords_heavy = coords_full[heavy_idx]
        desc = compute_single_conformer_descriptors(mol_no_h, coords_heavy, center)
        if desc is not None:
            all_desc.append(desc)
            weights.append(weight)

    if not all_desc:
        return None

    weights = np.array(weights) / sum(weights)
    result = {}
    for key in sorted(all_desc[0].keys()):
        vals = np.array([d[key] for d in all_desc])
        wmean = np.average(vals, weights=weights)
        wstd = np.sqrt(np.average((vals - wmean) ** 2, weights=weights))
        result[f"{key}_mean"] = float(wmean)
        result[f"{key}_std"] = float(wstd)
    result["n_conformers"] = len(all_desc)
    result["n_clusters"] = ensemble.get("n_clusters", len(all_desc))
    return result


def main():
    MECHAWARE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load Evans data
    dedup_path = PROJECT_DIR / "data" / "v3" / "interim" / "07_deduplicated.csv"
    if not dedup_path.exists():
        logger.error(f"Missing {dedup_path}")
        sys.exit(1)

    df = pd.read_csv(dedup_path)
    evans_df = df[df["Reaction_Class"] == "EvansAux"].reset_index(drop=True)
    n = len(evans_df)
    kcol = "canonical_Ketone" if "canonical_Ketone" in evans_df.columns else "Ketone"
    logger.info(f"Loaded {n} Evans reactions")

    # Deduplicate ketones
    unique_ketones = [str(s) for s in evans_df[kcol].dropna().unique()]
    logger.info(f"Unique ketones: {len(unique_ketones)}")

    # ── Step M2: Parallel Z/E/Ketone conformer generation ──
    logger.info(f"Generating Z/E/Ketone conformers ({N_CONFS} confs, no MMFF, {N_WORKERS} workers)...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(_worker_ze, [(s, TIMEOUT_SEC) for s in unique_ketones])

    ketone_cache = dict(zip(unique_ketones, results))
    n_ok = sum(1 for v in results if v is not None)
    n_z = sum(1 for v in results if v is not None and v.get("z_enolate") is not None)
    n_e = sum(1 for v in results if v is not None and v.get("e_enolate") is not None)
    logger.info(f"  Success: {n_ok}/{len(unique_ketones)} ketone, {n_z} Z, {n_e} E")

    elapsed_m2 = time.time() - t0
    logger.info(f"  M2 done in {elapsed_m2:.0f}s")

    # Map back to rows
    all_confs = {}
    for idx, row in evans_df.iterrows():
        oi = row.get("original_index", idx)
        ksmi = row.get(kcol)
        if pd.notna(ksmi) and str(ksmi) in ketone_cache:
            result = ketone_cache[str(ksmi)]
            if result is not None:
                all_confs[oi] = result

    # Save conformers
    with open(MECHAWARE_DIR / "ze_conformers.pkl", "wb") as f:
        pickle.dump(all_confs, f)

    # ── Step M3: Compute steric features ──
    logger.info("Computing 3×24d steric features...")
    t1 = time.time()

    ket_feats, z_feats, e_feats, valid_ois = [], [], [], []

    for oi, ens in all_confs.items():
        k_desc = _compute_steric(ens["ketone"]["mol"], ens["ketone"]) if ens.get("ketone") else None
        z_desc = _compute_steric(ens["z_enolate"]["mol"], ens["z_enolate"]) if ens.get("z_enolate") else None
        e_desc = _compute_steric(ens["e_enolate"]["mol"], ens["e_enolate"]) if ens.get("e_enolate") else None

        if k_desc is None:
            continue
        if z_desc is None:
            z_desc = {k: 0.0 for k in STERIC_DESC_NAMES}
        if e_desc is None:
            e_desc = {k: 0.0 for k in STERIC_DESC_NAMES}

        ket_feats.append(k_desc)
        z_feats.append(z_desc)
        e_feats.append(e_desc)
        valid_ois.append(oi)

    elapsed_m3 = time.time() - t1
    logger.info(f"  Steric: {len(valid_ois)}/{len(all_confs)} success ({elapsed_m3:.0f}s)")

    # Save CSVs with prefixed column names
    def save_csv(feats, prefix, filename):
        rows = [{f"{prefix}_{n}": feat.get(n, 0.0) for n in STERIC_DESC_NAMES} for feat in feats]
        out = pd.DataFrame(rows)
        out.insert(0, "original_index", valid_ois)
        out.to_csv(MECHAWARE_DIR / filename, index=False)
        logger.info(f"  Saved {filename} ({len(out)} rows × {len(out.columns)-1}d)")

    save_csv(ket_feats, "ket", "ketone_steric.csv")
    save_csv(z_feats, "z", "z_enolate_steric.csv")
    save_csv(e_feats, "e", "e_enolate_steric.csv")

    total = time.time() - t0
    logger.info(f"\nDone! {len(valid_ois)} rows, total {total:.0f}s")


if __name__ == "__main__":
    main()
