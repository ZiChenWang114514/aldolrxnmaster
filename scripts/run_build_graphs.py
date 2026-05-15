#!/usr/bin/env python3
"""Phase B4: Pre-compute graph representations for GNN training.

Builds 4 types of PyG Data objects for all Evans reactions:
  1. Reaction diff graphs (atom-mapped)
  2. Multi-view graphs (reactant + product separate)
  3. 3D spatial graphs (from conformer coordinates)
  4. TS approximation graphs (bond change annotation)

Usage:
    conda run -n aldol-rxn python scripts/run_build_graphs.py
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
os.environ["OMP_NUM_THREADS"] = "4"

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from chiralaldol.gnn.graph_builder import (
    build_diff_graph,
    build_multiview_graph,
    build_3d_graph,
    build_ts_graph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Phase B4: Build Graph Representations")
    logger.info("=" * 60)

    # Load cleaned Evans data
    df = pd.read_csv(PROJECT / "data" / "processed" / "evans_v2_clean.csv")
    n = len(df)
    logger.info(f"Loaded {n} Evans reactions")

    # Load condition features (35d)
    cond = pd.read_csv(PROJECT / "data" / "processed" / "features" / "reaction_conditions.csv")
    cond_arr = cond.values.astype(np.float32)

    # Load labels
    labels = pd.read_csv(PROJECT / "data" / "processed" / "features" / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    # Load conformer ensembles for 3D graphs
    conf_path = PROJECT / "data" / "processed" / "chiralaldol" / "conformer_ensembles.pkl"
    if conf_path.exists():
        with open(conf_path, "rb") as f:
            conformers = pickle.load(f)
        logger.info(f"Loaded conformer ensembles: {len(conformers)} entries")
    else:
        conformers = {}
        logger.warning("No conformer ensembles found — 3D graphs will be skipped")

    out_dir = PROJECT / "data" / "processed" / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build Graph Type 1: Reaction Diff ----
    logger.info("\n--- Building Reaction Diff Graphs ---")
    diff_graphs = []
    n_diff_ok = 0
    for i in range(n):
        g = build_diff_graph(df.iloc[i]["Mapped_Reaction"], y[i], cond_arr[i])
        if g is not None:
            n_diff_ok += 1
        diff_graphs.append(g)

    torch.save(diff_graphs, out_dir / "diff_graphs.pt")
    logger.info(f"  Saved {n_diff_ok}/{n} diff graphs")

    # ---- Build Graph Type 2: Multi-View ----
    logger.info("\n--- Building Multi-View Graphs ---")
    mv_graphs = []
    n_mv_ok = 0
    for i in range(n):
        row = df.iloc[i]
        # Use raw reaction SMILES (reactants.products)
        rxn = row["Raw_Reaction_Smiles"]
        parts = str(rxn).split(">>")
        if len(parts) == 2:
            g = build_multiview_graph(parts[0], parts[1], y[i], cond_arr[i])
        else:
            g = None
        if g is not None:
            n_mv_ok += 1
        mv_graphs.append(g)

    torch.save(mv_graphs, out_dir / "multiview_graphs.pt")
    logger.info(f"  Saved {n_mv_ok}/{n} multi-view graphs")

    # ---- Build Graph Type 3: 3D Spatial ----
    logger.info("\n--- Building 3D Spatial Graphs ---")
    spatial_graphs = []
    n_3d_ok = 0

    # Build index mapping: old (1822) → new (1801) indices
    # The 21 deleted rows were removed by the audit script
    df_orig = pd.read_csv(PROJECT / "data" / "processed" / "evans_clean.csv")
    deleted_indices = set(df_orig[df_orig["Ketone"].isna()].index.tolist())
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(df_orig)):
        if old_idx not in deleted_indices:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # Also need enolate SMILES to parse mol for node features
    enolates = pd.read_csv(PROJECT / "data" / "processed" / "chiralaldol" / "enolates.csv")

    for i in range(n):
        # Find corresponding old index for this new index
        old_idx = None
        for oi, ni in old_to_new.items():
            if ni == i:
                old_idx = oi
                break

        ens = conformers.get(old_idx) if old_idx is not None else None
        if ens is None or "representatives" not in ens or len(ens["representatives"]) == 0:
            spatial_graphs.append(None)
            continue

        # Get lowest-energy conformer: tuple (conf_id, energy, weight, coords)
        reps = ens["representatives"]
        best_rep = min(reps, key=lambda r: r[1])  # lowest energy
        coords = best_rep[3]  # (N_atoms, 3) array

        # Get mol from enolate SMILES
        enolate_smi = enolates.iloc[i]["enolate_smiles"] if i < len(enolates) else None
        if pd.isna(enolate_smi):
            spatial_graphs.append(None)
            continue

        mol = Chem.MolFromSmiles(str(enolate_smi))
        if mol is None or mol.GetNumAtoms() != len(coords):
            spatial_graphs.append(None)
            continue

        g = build_3d_graph(mol, coords, y[i], cond_arr[i], cutoff=5.0)
        if g is not None:
            n_3d_ok += 1
        spatial_graphs.append(g)

    torch.save(spatial_graphs, out_dir / "spatial_3d_graphs.pt")
    logger.info(f"  Saved {n_3d_ok}/{n} 3D spatial graphs")

    # ---- Build Graph Type 4: TS Approximation ----
    logger.info("\n--- Building TS Approximation Graphs ---")
    ts_graphs = []
    n_ts_ok = 0
    for i in range(n):
        g = build_ts_graph(df.iloc[i]["Mapped_Reaction"], y[i], cond_arr[i])
        if g is not None:
            n_ts_ok += 1
        ts_graphs.append(g)

    torch.save(ts_graphs, out_dir / "ts_approx_graphs.pt")
    logger.info(f"  Saved {n_ts_ok}/{n} TS approx graphs")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Diff graphs:     {n_diff_ok}/{n} ({n_diff_ok/n*100:.1f}%)")
    logger.info(f"  Multi-view:      {n_mv_ok}/{n} ({n_mv_ok/n*100:.1f}%)")
    logger.info(f"  3D spatial:      {n_3d_ok}/{n} ({n_3d_ok/n*100:.1f}%)")
    logger.info(f"  TS approx:       {n_ts_ok}/{n} ({n_ts_ok/n*100:.1f}%)")
    logger.info(f"  Output dir:      {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
