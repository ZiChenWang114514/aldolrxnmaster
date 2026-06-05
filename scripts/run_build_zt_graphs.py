#!/usr/bin/env python3
"""Precompute Zimmerman-Traxler transition state graphs for all reactions.

Supports two modes:
  --single-ts: Build one ZT graph per reaction (v1 mode, 20d nodes / 5d edges)
  --multi-ts:  Build 4 TS graphs per reaction (default, 28d nodes / 8d edges)

Usage:
    conda run -n aldol-rxn python scripts/run_build_zt_graphs.py
    conda run -n aldol-rxn python scripts/run_build_zt_graphs.py --single-ts
"""

import argparse
import logging
import pickle
import time
from collections import Counter

import pandas as pd

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, VALID_AUXILIARIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_zt_graphs")

OUT_DIR = FEAT_DIR / "zt_graphs"
ZE_CACHE_PATH = FEAT_DIR / "mechaware" / "ze_conformers_v4.pkl"
ALD_CACHE_PATH = FEAT_DIR / "conformers" / "aldehyde_conformers.pkl"


def build_single_ts(df):
    """Build one ZT graph per reaction (legacy v1 mode)."""
    from chiralaldol.zt_3d_coords import add_3d_coords_batch
    from chiralaldol.zt_graph_builder import build_zt_graphs_batch

    graphs = build_zt_graphs_batch(df)
    statuses = Counter(g.status for g in graphs)
    logger.info(f"Single-TS status: {statuses}")

    add_3d_coords_batch(graphs)

    out = {
        "graphs": graphs,
        "n_total": len(graphs),
        "n_success": statuses.get("success", 0),
    }
    return out, "evans_zt_graphs.pkl"


def build_multi_ts(df):
    """Build 4 TS graphs per reaction (multi-TS mode)."""
    from chiralaldol.zt_graph_builder import build_multi_ts_graphs_batch

    # Load conformer caches
    ze_cache, ald_cache = None, None
    if ZE_CACHE_PATH.exists():
        with open(ZE_CACHE_PATH, "rb") as f:
            ze_cache = pickle.load(f)
        logger.info(f"Loaded ze_conformers cache: {len(ze_cache)} entries")
    if ALD_CACHE_PATH.exists():
        with open(ALD_CACHE_PATH, "rb") as f:
            ald_cache = pickle.load(f)
        logger.info(f"Loaded aldehyde conformers cache: {len(ald_cache)} entries")

    ts_sets = build_multi_ts_graphs_batch(df, ze_cache=ze_cache, ald_cache=ald_cache)

    # Statistics
    n_full_success = sum(
        1 for ts in ts_sets if all(g.status == "success" for g in ts.graphs)
    )
    n_partial = sum(
        1 for ts in ts_sets if any(g.status == "success" for g in ts.graphs)
    )
    logger.info(f"Multi-TS: {n_full_success}/{len(ts_sets)} fully successful, "
                f"{n_partial}/{len(ts_sets)} partially successful")

    out = {
        "graph_sets": ts_sets,
        "n_total": len(ts_sets),
        "n_full_success": n_full_success,
        "n_partial_success": n_partial,
    }
    return out, "evans_multi_ts_graphs.pkl"


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single-ts", action="store_true", help="Build v1 single-TS graphs")
    group.add_argument("--multi-ts", action="store_true", default=True,
                       help="Build 4-TS multi-TS graphs (default)")
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    df = df_full[df_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    logger.info(f"Loaded {len(df_full)} total, {len(df)} valid")

    evans_mask = df["auxiliary_type"] == "evans"
    evans_df = df[evans_mask].reset_index(drop=True)
    evans_orig_idx = df.index[evans_mask].tolist()
    logger.info(f"Evans: {len(evans_df)} reactions")

    if args.single_ts:
        data, fname = build_single_ts(evans_df)
    else:
        data, fname = build_multi_ts(evans_df)

    data["orig_indices"] = evans_orig_idx
    data["auxiliary_type"] = "evans"

    out_path = OUT_DIR / fname
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved to {out_path}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
