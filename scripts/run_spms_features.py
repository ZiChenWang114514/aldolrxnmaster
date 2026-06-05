#!/usr/bin/env python3
"""Compute SPMS directional steric features and compress for downstream models.

Phase A: Compute SPMS matrices at α-carbon (ketone) and aldehyde carbon
Phase B: Compress to flat vectors via Conv2D autoencoder → append to 154d features

Usage:
    conda run -n aldol-rxn python scripts/run_spms_features.py
    conda run -n aldol-rxn python scripts/run_spms_features.py --phase A
    conda run -n aldol-rxn python scripts/run_spms_features.py --phase B
"""

import argparse
import logging
import pickle
import time

import numpy as np
import pandas as pd

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, SPMS_DIR, VALID_AUXILIARIES
from chiralaldol.data_io import load_features
from chiralaldol.spms import (
    compute_spms_ensemble,
    find_alpha_carbon,
    find_aldehyde_carbon,
)
from chiralaldol.face_steric_map import (
    compute_face_maps_ensemble,
    extract_face_map_features,
)

CLEAN_CSV = CLEAN_DIR / "substrate_aldol_clean.csv"
CONF_DIR = FEAT_DIR / "conformers"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spms_features")


def phase_a_compute_spms():
    """Compute SPMS matrices for all molecules."""
    logger.info("=" * 60)
    logger.info("Phase A: Computing SPMS features")
    logger.info("=" * 60)

    SPMS_DIR.mkdir(parents=True, exist_ok=True)

    # Load conformer ensembles
    with open(CONF_DIR / "ketone_conformers.pkl", "rb") as f:
        ketone_confs = pickle.load(f)
    with open(CONF_DIR / "aldehyde_conformers.pkl", "rb") as f:
        aldehyde_confs = pickle.load(f)

    logger.info(f"Loaded {len(ketone_confs)} ketone, {len(aldehyde_confs)} aldehyde ensembles")

    # Compute SPMS for ketones (α-carbon)
    ketone_spms = {}
    n_found, n_total = 0, 0
    for smi, ens in ketone_confs.items():
        n_total += 1
        mol = ens["mol"]
        idx = find_alpha_carbon(mol)
        if idx >= 0:
            smap = compute_spms_ensemble(mol, ens["representatives"], idx)
            if smap is not None:
                ketone_spms[smi] = smap
                n_found += 1

    logger.info(f"Ketone α-carbon SPMS: {n_found}/{n_total} ({100*n_found/n_total:.1f}%)")

    # Compute SPMS for aldehydes (aldehyde carbon)
    aldehyde_spms = {}
    n_found, n_total = 0, 0
    for smi, ens in aldehyde_confs.items():
        n_total += 1
        mol = ens["mol"]
        idx = find_aldehyde_carbon(mol)
        if idx >= 0:
            smap = compute_spms_ensemble(mol, ens["representatives"], idx)
            if smap is not None:
                aldehyde_spms[smi] = smap
                n_found += 1

    logger.info(f"Aldehyde carbon SPMS: {n_found}/{n_total} ({100*n_found/n_total:.1f}%)")

    # Save
    with open(SPMS_DIR / "ketone_spms.pkl", "wb") as f:
        pickle.dump(ketone_spms, f)
    with open(SPMS_DIR / "aldehyde_spms.pkl", "wb") as f:
        pickle.dump(aldehyde_spms, f)

    logger.info(f"Saved SPMS to {SPMS_DIR}")
    return ketone_spms, aldehyde_spms


def phase_b_compress(method="autoencoder", latent_dim=16):
    """Compress SPMS matrices and append to feature matrix."""
    logger.info("=" * 60)
    logger.info(f"Phase B: Compressing SPMS ({method}, dim={latent_dim})")
    logger.info("=" * 60)

    # Load SPMS
    with open(SPMS_DIR / "ketone_spms.pkl", "rb") as f:
        ketone_spms = pickle.load(f)
    with open(SPMS_DIR / "aldehyde_spms.pkl", "rb") as f:
        aldehyde_spms = pickle.load(f)

    # Load clean data to map SMILES → row indices
    df_full = pd.read_csv(CLEAN_CSV)
    df = df_full[df_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    n_rows = len(df)
    logger.info(f"Dataset: {n_rows} rows")

    # Build per-row SPMS arrays
    ketone_col = "canonical_ketone_smiles"
    aldehyde_col = "canonical_aldehyde_smiles"

    # Default SPMS: zeros (for molecules without match)
    default_spms = np.zeros((10, 20), dtype=np.float32)

    ketone_maps = []
    aldehyde_maps = []
    ket_hit, ald_hit = 0, 0

    for i in range(n_rows):
        ksmi = df.iloc[i].get(ketone_col, "")
        asmi = df.iloc[i].get(aldehyde_col, "")

        km = ketone_spms.get(ksmi, default_spms)
        am = aldehyde_spms.get(asmi, default_spms)

        if ksmi in ketone_spms:
            ket_hit += 1
        if asmi in aldehyde_spms:
            ald_hit += 1

        ketone_maps.append(km)
        aldehyde_maps.append(am)

    logger.info(f"SPMS coverage: ketone {ket_hit}/{n_rows} ({100*ket_hit/n_rows:.1f}%), "
                f"aldehyde {ald_hit}/{n_rows} ({100*ald_hit/n_rows:.1f}%)")

    # Stack: (N, 2, 10, 20)
    spms_arrays = np.stack([
        np.array(ketone_maps),     # channel 0: ketone α-carbon
        np.array(aldehyde_maps),   # channel 1: aldehyde carbon
    ], axis=1)  # (N, 2, 10, 20)

    logger.info(f"SPMS array shape: {spms_arrays.shape}")

    # Compress
    if method == "autoencoder":
        from chiralaldol.spms_compressor import train_autoencoder
        _, latent, _, _ = train_autoencoder(
            spms_arrays, n_channels=2, latent_dim=latent_dim,
            epochs=100, lr=1e-3, batch_size=64)
    elif method == "pca":
        from chiralaldol.spms_compressor import compress_spms_pca
        latent, _ = compress_spms_pca(spms_arrays, n_components=latent_dim)
    elif method == "stats":
        # Statistical summaries: per-channel mean/std/min/max by row and column
        N = spms_arrays.shape[0]
        feats = []
        for ch in range(2):
            arr = spms_arrays[:, ch]  # (N, 10, 20)
            feats.extend([
                arr.mean(axis=(1, 2)),  # global mean
                arr.std(axis=(1, 2)),   # global std
                arr.min(axis=(1, 2)),   # global min
                arr.max(axis=(1, 2)),   # global max
                arr.mean(axis=2).mean(axis=1),  # row-mean
                arr.mean(axis=1).mean(axis=1),  # col-mean
                arr.mean(axis=2).std(axis=1),   # row-variation
                arr.mean(axis=1).std(axis=1),   # col-variation
            ])
        latent = np.column_stack(feats)  # (N, 16)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Compressed to {latent.shape}")

    # Create feature column names
    feat_names = [f"spms_{method}_{i}" for i in range(latent.shape[1])]

    # Load existing features and append
    existing, _ = load_features()
    if len(existing) != n_rows:
        logger.warning(f"Feature rows ({len(existing)}) != data rows ({n_rows})")
        return

    # Drop old SPMS columns if they exist
    old_spms_cols = [c for c in existing.columns if c.startswith("spms_")]
    if old_spms_cols:
        existing = existing.drop(columns=old_spms_cols)

    spms_df = pd.DataFrame(latent, columns=feat_names)
    combined = pd.concat([existing, spms_df], axis=1)

    out_path = FEAT_DIR / "v5_features_spms.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved {combined.shape[1]}d features to {out_path}")

    # Also save raw SPMS arrays for GNN use
    np.save(SPMS_DIR / "spms_arrays.npy", spms_arrays)
    np.save(SPMS_DIR / "spms_latent.npy", latent)
    logger.info(f"Saved raw arrays to {SPMS_DIR}")

    return latent


def phase_c_face_maps():
    """Compute face steric maps for all ketone molecules."""
    logger.info("=" * 60)
    logger.info("Phase C: Computing Face Steric Maps")
    logger.info("=" * 60)

    SPMS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONF_DIR / "ketone_conformers.pkl", "rb") as f:
        ketone_confs = pickle.load(f)

    logger.info(f"Computing face maps for {len(ketone_confs)} ketones...")

    face_map_data = {}
    n_found = 0
    for smi, ens in ketone_confs.items():
        mol = ens["mol"]
        si, re = compute_face_maps_ensemble(mol, ens["representatives"])
        if si is not None:
            face_map_data[smi] = {"si": si, "re": re}
            n_found += 1

    logger.info(f"Face maps computed: {n_found}/{len(ketone_confs)} ({100*n_found/len(ketone_confs):.1f}%)")

    with open(SPMS_DIR / "face_maps.pkl", "wb") as f:
        pickle.dump(face_map_data, f)

    # Extract flat features for tree models
    df_full = pd.read_csv(CLEAN_CSV)
    df = df_full[df_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)

    rows = []
    for i in range(len(df)):
        ksmi = df.iloc[i].get("canonical_ketone_smiles", "")
        if ksmi in face_map_data:
            feats = extract_face_map_features(
                face_map_data[ksmi]["si"], face_map_data[ksmi]["re"])
        else:
            feats = {k: 0.0 for k in extract_face_map_features(
                np.zeros((10, 10)), np.zeros((10, 10))).keys()}
        rows.append(feats)

    face_df = pd.DataFrame(rows)
    face_df.to_csv(SPMS_DIR / "face_map_features.csv", index=False)
    logger.info(f"Saved {face_df.shape[1]}d face map features to {SPMS_DIR}")

    return face_map_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["A", "B", "C", "all"])
    parser.add_argument("--method", default="autoencoder",
                        choices=["autoencoder", "pca", "stats"])
    parser.add_argument("--latent-dim", type=int, default=16)
    args = parser.parse_args()

    t0 = time.time()

    if args.phase in ("A", "all"):
        phase_a_compute_spms()

    if args.phase in ("B", "all"):
        phase_b_compress(method=args.method, latent_dim=args.latent_dim)

    if args.phase in ("C", "all"):
        phase_c_face_maps()

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
