"""Compute 48d professional product steric features for all StereoRank candidates.

Uses full pipeline: 100 conformers + MMFF optimization + RMSD clustering + Boltzmann weighting.
Computes face-dependent %Vbur, Sterimol L/B1/B5, dihedrals at BOTH Ca and Cb centers.

Output: data/v3/stereorank/candidates_48d_feats.csv
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from chiralaldol.product_steric import compute_all_candidates_48d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "v3" / "stereorank" / "product_48d.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
STEREORANK_DIR = PROJECT_DIR / "data" / "v3" / "stereorank"


def main():
    logger.info("=" * 60)
    logger.info("StereoRank: Computing 48d product steric features")
    logger.info("=" * 60)

    # Load candidates
    candidates_df = pd.read_csv(STEREORANK_DIR / "candidates.csv")
    logger.info(f"Loaded {len(candidates_df)} candidates ({candidates_df['reaction_id'].nunique()} reactions)")

    # Compute 48d features
    t0 = time.time()
    result_df = compute_all_candidates_48d(candidates_df, n_confs=100, n_workers=8)
    elapsed = time.time() - t0

    logger.info(f"Total computation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Check success rate
    prod_cols = [c for c in result_df.columns if c.startswith("prod_")]
    n_ok = result_df[prod_cols[0]].notna().sum() if prod_cols else 0
    logger.info(f"Success: {n_ok}/{len(result_df)} ({n_ok/len(result_df)*100:.1f}%)")
    logger.info(f"Feature columns: {len(prod_cols)}")

    # Save
    output_path = STEREORANK_DIR / "candidates_48d_feats.csv"
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
