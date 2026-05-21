#!/usr/bin/env python
"""Generate 3D conformers for all molecules in Evans dataset.

Uses RDKit ETKDG + MMFF force field optimization.
Generates conformers for Ketone, Aldehyde, and Product separately.
Output: data/processed/conformers/ with SDF files and coordinate arrays.

Shared infrastructure for M3 (EquiReact), M4 (ChiENN), M6 (GCPNet).
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
PROCESSED_DIR = PROJECT / "data" / "processed"
CONF_DIR = PROCESSED_DIR / "conformers"
CONF_DIR.mkdir(parents=True, exist_ok=True)


def generate_conformer(smiles: str, n_confs: int = 10, max_attempts: int = 50, seed: int = 42):
    """Generate lowest-energy 3D conformer for a SMILES string.

    Returns:
        mol: RDKit Mol with 3D coordinates (or None if failed)
        coords: numpy array (n_atoms, 3) or None
        energy: MMFF energy or None
    """
    if pd.isna(smiles) or not str(smiles).strip():
        return None, None, None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None, None

    mol = Chem.AddHs(mol)

    # Generate multiple conformers with ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0  # use all threads

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        # Fallback: try with less strict parameters
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        return None, None, None

    # MMFF optimization and select lowest energy
    best_energy = float("inf")
    best_cid = cids[0]

    for cid in cids:
        try:
            result = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
            if ff is not None:
                energy = ff.CalcEnergy()
                if energy < best_energy:
                    best_energy = energy
                    best_cid = cid
        except Exception:
            continue

    # Extract coordinates (heavy atoms only)
    conf = mol.GetConformer(best_cid)
    mol_no_h = Chem.RemoveHs(mol)
    # Re-embed if needed to get heavy-atom-only conformer
    try:
        conf_no_h = mol_no_h.GetConformer()
    except Exception:
        # Map heavy atom positions from H-included conformer
        coords = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() != 1:  # skip H
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
        return mol_no_h, np.array(coords, dtype=np.float32), best_energy

    coords = np.array([[conf_no_h.GetAtomPosition(i).x,
                         conf_no_h.GetAtomPosition(i).y,
                         conf_no_h.GetAtomPosition(i).z]
                        for i in range(mol_no_h.GetNumAtoms())], dtype=np.float32)

    return mol_no_h, coords, best_energy


def process_column(df, col_name, n_confs=10):
    """Generate conformers for all SMILES in a column."""
    logger.info(f"Generating conformers for {col_name} ({len(df)} molecules)...")
    t0 = time.time()

    mols = []
    coords_list = []
    energies = []
    n_fail = 0

    for i, smi in enumerate(df[col_name]):
        mol, coords, energy = generate_conformer(str(smi), n_confs=n_confs)
        mols.append(mol)
        coords_list.append(coords)
        energies.append(energy)

        if mol is None:
            n_fail += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  {col_name}: {i+1}/{len(df)} ({n_fail} failures)")

    logger.info(f"  {col_name}: done in {time.time()-t0:.1f}s, failures={n_fail}/{len(df)}")
    return mols, coords_list, energies


def main():
    # Load Evans clean data
    df = pd.read_csv(PROCESSED_DIR / "evans_clean.csv")
    n = len(df)
    logger.info(f"Processing {n} Evans reactions")

    product_col = "Product_" if "Product_" in df.columns else "Raw_Product_Smiles"

    results = {}
    for col_name, smi_col in [("ketone", "Ketone"), ("aldehyde", "Aldehyde"), ("product", product_col)]:
        mols, coords_list, energies = process_column(df, smi_col, n_confs=10)
        results[col_name] = {
            "mols": mols,
            "coords": coords_list,
            "energies": energies,
        }

    # Save as pickle (contains RDKit mol objects)
    with open(CONF_DIR / "conformers.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save coordinates as separate npz for easy loading
    # Variable-length arrays stored as object arrays
    for role in ["ketone", "aldehyde", "product"]:
        coords = results[role]["coords"]
        n_valid = sum(1 for c in coords if c is not None)
        logger.info(f"  {role}: {n_valid}/{n} valid conformers")

    logger.info(f"Saved to {CONF_DIR / 'conformers.pkl'}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
