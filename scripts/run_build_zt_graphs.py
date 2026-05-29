#!/usr/bin/env python3
"""Precompute Zimmerman-Traxler transition state graphs for all reactions.

Usage:
    conda run -n aldol-rxn python scripts/run_build_zt_graphs.py
"""

import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import CLEAN_DIR, FEAT_DIR
from chiralaldol.zt_graph_builder import build_zt_graphs_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_zt_graphs")

OUT_DIR = FEAT_DIR / "zt_graphs"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    logger.info(f"Loaded {len(df)} reactions")

    # Build for Evans subset
    evans_mask = df["auxiliary_type"] == "evans"
    evans_df = df[evans_mask].reset_index(drop=True)
    evans_orig_idx = df.index[evans_mask].tolist()

    logger.info(f"Building ZT graphs for {len(evans_df)} Evans reactions...")
    graphs = build_zt_graphs_batch(evans_df)

    # Statistics
    from collections import Counter
    statuses = Counter(g.status for g in graphs)
    logger.info(f"Status: {statuses}")
    logger.info(f"Success rate: {statuses['success']}/{len(evans_df)} "
                f"({statuses['success']/len(evans_df)*100:.1f}%)")

    # Save graphs
    out = {
        "graphs": graphs,
        "orig_indices": evans_orig_idx,
        "auxiliary_type": "evans",
        "n_total": len(graphs),
        "n_success": statuses["success"],
    }
    out_path = OUT_DIR / "evans_zt_graphs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    logger.info(f"Saved to {out_path}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
