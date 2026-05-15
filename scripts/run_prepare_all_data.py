#!/usr/bin/env python3
"""Phase B2: Clean non-Evans data and merge into all_clean.csv.

Creates a unified dataset of Evans + non-Evans reactions for transfer learning.

Usage:
    conda run -n aldol-rxn python scripts/run_prepare_all_data.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from chiralaldol.solvent_lookup import fill_missing_solvents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def count_stereo(smi):
    if pd.isna(smi):
        return -1
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return -1
    try:
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return len([c for c in chiral if c[1] in ("R", "S")])
    except Exception:
        return -1


def main():
    logger.info("=" * 60)
    logger.info("Phase B2: Prepare all_clean.csv")
    logger.info("=" * 60)

    # Load fully processed dataset (from existing pipeline)
    df = pd.read_csv(PROJECT / "data" / "interim" / "05_imputed.csv")
    logger.info(f"Loaded imputed dataset: {len(df)} rows")

    # Fix chirality using Raw_Product_Smiles
    logger.info("Fixing chirality_valid using Raw_Product_Smiles...")
    df["n_stereo_raw"] = df["Raw_Product_Smiles"].apply(count_stereo)
    df["chirality_valid"] = df["n_stereo_raw"] >= 2
    df["n_stereocenters"] = df["n_stereo_raw"]
    df.drop(columns=["n_stereo_raw"], inplace=True)

    # Remove missing molecules
    missing_mask = df["Ketone"].isna() | df["Aldehyde"].isna()
    n_missing = missing_mask.sum()
    logger.info(f"Removing {n_missing} rows with missing Ketone/Aldehyde")
    df_clean = df[~missing_mask].reset_index(drop=True)

    # Fill missing solvents
    df_clean = fill_missing_solvents(df_clean)

    # Split into Evans and non-Evans
    evans = df_clean[df_clean["Reaction_Class"] == "EvansAux"].copy()
    non_evans = df_clean[df_clean["Reaction_Class"] != "EvansAux"].copy()

    logger.info(f"\nFinal dataset: {len(df_clean)} rows")
    logger.info(f"  Evans: {len(evans)}")
    logger.info(f"  Non-Evans: {len(non_evans)}")
    logger.info(f"    AsymmetricDouble: {(non_evans['Reaction_Class'] == 'AsymmetricDouble').sum()}")
    logger.info(f"    AsymmetricSingle: {(non_evans['Reaction_Class'] == 'AsymmetricSingle').sum()}")
    logger.info(f"    OppolzerAux: {(non_evans['Reaction_Class'] == 'OppolzerAux').sum()}")

    # Label distribution
    logger.info(f"\nLabel distribution (all):")
    for cls_name, sub in [("Evans", evans), ("Non-Evans", non_evans), ("All", df_clean)]:
        dist = sub["label_joint"].value_counts().sort_index()
        logger.info(f"  {cls_name} ({len(sub)}): " +
                    ", ".join(f"C{int(k)}={v}" for k, v in dist.items()))

    # Solvent coverage
    evans_solv = evans["solvent_known"].mean() * 100
    ne_solv = non_evans["solvent_known"].mean() * 100
    logger.info(f"\nSolvent known: Evans={evans_solv:.1f}%, Non-Evans={ne_solv:.1f}%")

    # Add reaction_class encoding for multi-task learning
    class_map = {"EvansAux": 0, "AsymmetricDouble": 1, "AsymmetricSingle": 2, "OppolzerAux": 3}
    df_clean["reaction_class_id"] = df_clean["Reaction_Class"].map(class_map)

    # Save
    out_path = PROJECT / "data" / "processed" / "all_clean.csv"
    df_clean.to_csv(out_path, index=False)
    logger.info(f"\nSaved {out_path}: {len(df_clean)} rows")

    # Also save non-Evans separately for conformer generation
    ne_path = PROJECT / "data" / "processed" / "non_evans_clean.csv"
    non_evans.to_csv(ne_path, index=False)
    logger.info(f"Saved {ne_path}: {len(non_evans)} rows")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
