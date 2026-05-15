#!/usr/bin/env python3
"""Phase C: GNN Benchmark — 4 architectures × 3 fusion modes.

Stage 1 (coarse): Fixed hyperparams, all 12 combinations on temporal split
Stage 2 (fine):   Optuna tuning on top-3 combinations

Usage:
    conda run -n aldol-rxn python scripts/run_gnn_benchmark.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from chiralaldol.gnn.mpnn_diff import MPNNDiff
from chiralaldol.gnn.gat_multiview import GATMultiView
from chiralaldol.gnn.schnet_3d import SchNet3D
from chiralaldol.gnn.equiformer import SimpleEquiformer
from chiralaldol.gnn.graph_builder import ATOM_FEAT_DIM, BOND_FEAT_DIM
from chiralaldol.gnn.trainer import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GRAPH_DIR = PROJECT / "data" / "processed" / "graphs"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed hyperparams for Stage 1 (coarse screening)
COARSE_CONFIG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "batch_size": 32,
    "label_smoothing": 0.1,
}

# Model configs: (graph_type, architecture_class, fusion_mode)
MODEL_CONFIGS = [
    # G1: MPNN Diff × 3 fusions
    ("diff", "mpnn", "concat"),
    ("diff", "mpnn", "film"),
    ("diff", "mpnn", "inject"),
    # G2: GAT Multi-View × 3 fusions
    ("multiview", "gat", "concat"),
    ("multiview", "gat", "film"),
    ("multiview", "gat", "inject"),
    # G3: Equiformer × 3 fusions
    ("spatial_3d", "equiformer", "concat"),
    ("spatial_3d", "equiformer", "film"),
    ("spatial_3d", "equiformer", "inject"),
    # G4: SchNet × 3 fusions
    ("spatial_3d", "schnet", "concat"),
    ("spatial_3d", "schnet", "film"),
    ("spatial_3d", "schnet", "inject"),
]


def load_graphs(graph_type: str) -> list:
    """Load pre-computed graphs."""
    path = GRAPH_DIR / f"{graph_type}_graphs.pt"
    if not path.exists():
        logger.error(f"Graph file not found: {path}")
        return []
    return torch.load(path, weights_only=False)


def create_model(arch: str, fusion: str, graph_type: str,
                 hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.3) -> torch.nn.Module:
    """Instantiate a GNN model."""
    condition_dim = 35
    num_classes = 4

    if arch == "mpnn":
        # Diff graph: node_dim = ATOM_FEAT_DIM + 1 (reaction center flag)
        # Edge dim = BOND_FEAT_DIM + 1 (new bond flag)
        return MPNNDiff(
            node_input_dim=ATOM_FEAT_DIM + 1,
            edge_input_dim=BOND_FEAT_DIM + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            condition_dim=condition_dim,
            fusion=fusion,
            dropout=dropout,
        )
    elif arch == "gat":
        return GATMultiView(
            node_input_dim=ATOM_FEAT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            condition_dim=condition_dim,
            fusion=fusion,
            dropout=dropout,
        )
    elif arch == "equiformer":
        return SimpleEquiformer(
            node_input_dim=ATOM_FEAT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            condition_dim=condition_dim,
            fusion=fusion,
            dropout=dropout,
        )
    elif arch == "schnet":
        return SchNet3D(
            node_input_dim=ATOM_FEAT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            condition_dim=condition_dim,
            fusion=fusion,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def run_coarse_screening():
    """Stage 1: Run all 12 combinations with fixed hyperparams on temporal split."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Coarse Screening (12 combinations)")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)

    # Load temporal split
    with open(SPLIT_DIR / "evans_temporal.json") as f:
        split = json.load(f)
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    # Pre-load all graph types
    graph_cache = {}
    for gt in set(cfg[0] for cfg in MODEL_CONFIGS):
        graphs = load_graphs(gt)
        if graphs:
            graph_cache[gt] = graphs
            logger.info(f"Loaded {gt} graphs: {len(graphs)} "
                        f"(non-None: {sum(1 for g in graphs if g is not None)})")

    results = []

    for graph_type, arch, fusion in MODEL_CONFIGS:
        name = f"{arch}_{fusion}_{graph_type}"
        logger.info(f"\n{'='*40}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*40}")

        graphs = graph_cache.get(graph_type, [])
        if not graphs:
            logger.warning(f"  No graphs for {graph_type} — skipping")
            results.append({"name": name, "bal_acc": 0.0, "error": "no_graphs"})
            continue

        # Split graphs
        train_g = [graphs[i] for i in train_idx if i < len(graphs)]
        val_g = [graphs[i] for i in val_idx if i < len(graphs)]
        test_g = [graphs[i] for i in test_idx if i < len(graphs)]

        # Create model
        model = create_model(arch, fusion, graph_type)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model params: {n_params:,}")

        # Train
        try:
            result = train_and_evaluate(
                model, train_g, val_g, test_g, COARSE_CONFIG, DEVICE
            )
            result["name"] = name
            result["arch"] = arch
            result["fusion"] = fusion
            result["graph_type"] = graph_type
            result["n_params"] = n_params

            logger.info(f"  ★ {name}: bal_acc={result['bal_acc']:.4f} "
                        f"(val={result.get('best_val_acc', 0):.4f}, "
                        f"epochs={result.get('epochs_trained', 0)})")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            result = {"name": name, "bal_acc": 0.0, "error": str(e)}

        results.append(result)

    # Save results
    results_table = []
    for r in results:
        results_table.append({
            "name": r.get("name", "?"),
            "bal_acc": r.get("bal_acc", 0.0),
            "best_val_acc": r.get("best_val_acc", 0.0),
            "n_params": r.get("n_params", 0),
            "epochs": r.get("epochs_trained", 0),
            "error": r.get("error", ""),
        })

    results_df = pd.DataFrame(results_table).sort_values("bal_acc", ascending=False)
    out_path = RESULTS_DIR / "tables" / "gnn_coarse_screening.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("COARSE SCREENING RESULTS")
    logger.info("=" * 60)
    for _, row in results_df.iterrows():
        marker = "★" if row["bal_acc"] > 0.5 else " "
        logger.info(f"  {marker} {row['name']:40s} bal_acc={row['bal_acc']:.4f}")

    logger.info(f"\nSaved to {out_path}")

    # Save predictions for top model
    best = results_df.iloc[0]
    best_result = [r for r in results if r.get("name") == best["name"]][0]
    if "y_true" in best_result:
        pred_dir = RESULTS_DIR / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({
            "idx": test_idx[:len(best_result["y_true"])],
            "y_true": best_result["y_true"],
            "y_pred": best_result["y_pred"],
        })
        for c in range(4):
            pred_df[f"prob_{c}"] = best_result["y_prob"][:, c]
        pred_df.to_csv(pred_dir / f"gnn_best_{best['name']}_evans_temporal.csv", index=False)

    return results_df


def main():
    results = run_coarse_screening()
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
