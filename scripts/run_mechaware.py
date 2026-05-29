#!/usr/bin/env python3
"""V4 MechAware: Z/E enolate separation + BW weighting + full feature integration.

Generates 156d MechAware-Full features for V4 data:
  - Ketone steric 24d
  - Z-enolate steric 24d
  - E-enolate steric 24d
  - BW-weighted steric 24d (base-dependent Z/E weighting)
  - Z/E weights 2d
  - Aldehyde steric 10d
  - Conditions 44d
  - Auxiliary 6d
  = 158d total

Usage:
    conda run -n aldol-rxn python scripts/run_mechaware_v4.py
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import CLEAN_DIR, FEAT_DIR
from chiralaldol.steric_descriptors import (
    STERIC_DESC_NAMES,
    compute_single_conformer_descriptors,
    find_reactive_center,
)
from chiralaldol.ze_enolate_generator import generate_ze_conformers, get_ze_weights

CLEAN_CSV = CLEAN_DIR / "substrate_aldol_clean.csv"
MA_DIR = FEAT_DIR / "mechaware"
CONF_CACHE = MA_DIR / "ze_conformers_v4.pkl"

N_WORKERS = 8
TIMEOUT_SEC = 60


def _worker_ze(args):
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


def _compute_steric(mol_h, ensemble):
    """Compute 24d steric from ensemble."""
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
    MA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("V4 MechAware Feature Engineering")
    print("=" * 60)

    # Load V4 data
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df)} rows")

    kcol = "canonical_ketone_smiles"
    unique_ketones = [str(s) for s in df[kcol].dropna().unique() if str(s).strip()]
    print(f"Unique ketones: {len(unique_ketones)}")

    # --- Step 1: Z/E/Ketone conformer generation ---
    if CONF_CACHE.exists():
        print(f"Loading cached Z/E conformers from {CONF_CACHE}")
        with open(CONF_CACHE, "rb") as f:
            ketone_cache = pickle.load(f)
    else:
        print(f"Generating Z/E/Ketone conformers ({N_WORKERS} workers, {TIMEOUT_SEC}s timeout)...")
        with Pool(N_WORKERS) as pool:
            results = pool.map(_worker_ze, [(s, TIMEOUT_SEC) for s in unique_ketones])
        ketone_cache = {smi: res for smi, res in zip(unique_ketones, results)}

        with open(CONF_CACHE, "wb") as f:
            pickle.dump(ketone_cache, f)

    n_ok = sum(1 for v in ketone_cache.values() if v is not None)
    n_z = sum(1 for v in ketone_cache.values() if v is not None and v.get("z_enolate") is not None)
    n_e = sum(1 for v in ketone_cache.values() if v is not None and v.get("e_enolate") is not None)
    print(f"  Ketone: {n_ok}/{len(unique_ketones)}, Z: {n_z}, E: {n_e}")

    # --- Step 2: Compute 3×24d steric features ---
    print("Computing ketone/Z/E steric features...")

    ket_rows, z_rows, e_rows = [], [], []
    null_steric = {f"{n}_mean": np.nan for n in sorted(STERIC_DESC_NAMES)}
    null_steric.update({f"{n}_std": np.nan for n in sorted(STERIC_DESC_NAMES)})
    null_steric["n_conformers"] = 0
    null_steric["n_clusters"] = 0

    for _, row in df.iterrows():
        ksmi = row.get(kcol)
        ens = ketone_cache.get(str(ksmi)) if pd.notna(ksmi) else None

        if ens is not None and ens.get("ketone"):
            k_desc = _compute_steric(ens["ketone"]["mol"], ens["ketone"])
        else:
            k_desc = None

        if ens is not None and ens.get("z_enolate"):
            z_desc = _compute_steric(ens["z_enolate"]["mol"], ens["z_enolate"])
        else:
            z_desc = None

        if ens is not None and ens.get("e_enolate"):
            e_desc = _compute_steric(ens["e_enolate"]["mol"], ens["e_enolate"])
        else:
            e_desc = None

        ket_rows.append(k_desc if k_desc else null_steric.copy())
        z_rows.append(z_desc if z_desc else null_steric.copy())
        e_rows.append(e_desc if e_desc else null_steric.copy())

    # Prefix columns
    ket_df = pd.DataFrame(ket_rows).rename(columns=lambda c: f"ket_{c}")
    z_df = pd.DataFrame(z_rows).rename(columns=lambda c: f"z_{c}")
    e_df = pd.DataFrame(e_rows).rename(columns=lambda c: f"e_{c}")

    n_ket_ok = ket_df.dropna(subset=["ket_n_conformers"]).shape[0]
    n_z_ok = z_df[z_df["z_n_conformers"] > 0].shape[0]
    n_e_ok = e_df[e_df["e_n_conformers"] > 0].shape[0]
    print(f"  Ketone steric: {n_ket_ok}/{len(df)}")
    print(f"  Z-enolate steric: {n_z_ok}/{len(df)}")
    print(f"  E-enolate steric: {n_e_ok}/{len(df)}")

    # Save individual CSVs
    ket_df.to_csv(MA_DIR / "ketone_steric.csv", index=False)
    z_df.to_csv(MA_DIR / "z_enolate_steric.csv", index=False)
    e_df.to_csv(MA_DIR / "e_enolate_steric.csv", index=False)

    # --- Step 3: BW-weighted steric ---
    print("Computing BW-weighted steric features...")

    # Get base/activator from conditions
    bw_rows = []
    w_z_list, w_e_list = [], []

    for i, row in df.iterrows():
        base = row.get("base_type", "no_base")
        activator = row.get("activator_type", "")
        wz, we = get_ze_weights(str(base), str(activator))
        w_z_list.append(wz)
        w_e_list.append(we)

        z_row = z_df.iloc[i]
        e_row = e_df.iloc[i]

        bw = {}
        for name in sorted(STERIC_DESC_NAMES):
            for suffix in ["_mean", "_std"]:
                col = f"{name}{suffix}"
                zv = z_row.get(f"z_{col}", 0)
                ev = e_row.get(f"e_{col}", 0)
                if pd.isna(zv): zv = 0
                if pd.isna(ev): ev = 0
                bw[f"bw_{col}"] = wz * zv + we * ev
        bw["bw_n_conformers"] = 0
        bw["bw_n_clusters"] = 0
        bw_rows.append(bw)

    bw_df = pd.DataFrame(bw_rows)
    w_df = pd.DataFrame({"w_Z": w_z_list, "w_E": w_e_list})

    # --- Step 4: Integrate full MechAware feature matrix ---
    print("Integrating MechAware-Full features...")

    # Load existing V4 features
    steric_v4 = pd.read_csv(FEAT_DIR / "steric_features.csv")  # 34d (enolate 24d + aldehyde 10d)
    cond_df = pd.read_csv(CLEAN_DIR / "condition_features.csv")  # 44d

    # Auxiliary one-hot
    aux_types = ["evans", "crimmins_thione", "crimmins_oxathione", "other_auxiliary", "myers"]
    aux_df = pd.DataFrame({f"aux_{a}": (df["auxiliary_type"] == a).astype(int) for a in aux_types})
    aux_df["n_defined_stereocenters"] = df["n_defined_stereocenters"].fillna(2)

    # Aldehyde steric (10d) from v4 steric
    ald_cols = [c for c in steric_v4.columns if c.startswith("ald_")]
    ald_df = steric_v4[ald_cols].copy()

    # MechAware-Full: ket(24) + z(24) + e(24) + bw(24) + w(2) + ald(10) + cond(44) + aux(6) = 158d
    ma_full = pd.concat([
        ket_df.reset_index(drop=True),   # 24d (+ n_conformers, n_clusters = 26)
        z_df.reset_index(drop=True),     # 26
        e_df.reset_index(drop=True),     # 26
        bw_df.reset_index(drop=True),    # 26
        w_df.reset_index(drop=True),     # 2
        ald_df.reset_index(drop=True),   # 10
        cond_df.reset_index(drop=True),  # 44
        aux_df.reset_index(drop=True),   # 6
    ], axis=1)

    # Fill NaN with column median
    for col in ma_full.columns:
        if ma_full[col].isna().any():
            med = ma_full[col].median()
            ma_full[col] = ma_full[col].fillna(med if pd.notna(med) else 0.0)

    print(f"MechAware-Full: {ma_full.shape[0]} rows × {ma_full.shape[1]}d")
    print(f"NaN remaining: {ma_full.isna().sum().sum()}")

    ma_full.to_csv(FEAT_DIR / "v4_mechaware_full.csv", index=False)

    # Also save a BW-only version: bw(24+2) + ald(10) + cond(44) + aux(6) = 86d (+ n_conf/clust)
    ma_bw = pd.concat([
        bw_df.reset_index(drop=True),
        w_df.reset_index(drop=True),
        ald_df.reset_index(drop=True),
        cond_df.reset_index(drop=True),
        aux_df.reset_index(drop=True),
    ], axis=1)
    for col in ma_bw.columns:
        if ma_bw[col].isna().any():
            med = ma_bw[col].median()
            ma_bw[col] = ma_bw[col].fillna(med if pd.notna(med) else 0.0)

    ma_bw.to_csv(FEAT_DIR / "v4_mechaware_bw.csv", index=False)
    print(f"MechAware-BW: {ma_bw.shape[0]} rows × {ma_bw.shape[1]}d")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
