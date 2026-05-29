#!/usr/bin/env python3
"""Phase B2: ChiENN on ZT Transition State Graphs.

Uses the ChiENN (Chiral Edge Neural Network, Gaiński 2023) architecture
with ZT graphs converted to edge graphs via to_edge_graph + circle_index.

ChiENN encodes chirality through spatially-ordered neighbor message passing
on the edge graph — the ordering is derived from 3D coordinates.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_chienn.py
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_chienn.py --epochs 80
"""

import argparse
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data, Batch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add ChiENN to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external" / "ChiENN"))

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, PRED_DIR, RESULTS_DIR, SPLITS_DIR
from chiralaldol.data_io import prepare_Xy, load_splits, save_predictions

# ChiENN imports
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from chienn.data.edge_graph.collate_circle_index import collate_circle_index
from chienn.model.chienn_model import ChiENNModel

ZT_GRAPHS_PATH = FEAT_DIR / "zt_graphs" / "evans_zt_graphs.pkl"
OUT_PRED_DIR = PRED_DIR / "zt_chienn"
TABLE_DIR = RESULTS_DIR / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zt_chienn")

K_NEIGHBORS = 3


def zt_graph_to_chienn_data(zt_graph, label=None):
    """Convert ZT graph to ChiENN-compatible edge graph with circle_index.

    Steps:
    1. Build PyG Data from ZT graph (x, edge_index, edge_attr, pos)
    2. Apply to_edge_graph() to convert to edge graph
    3. circle_index is computed automatically
    """
    if zt_graph.status != "success" or not hasattr(zt_graph, "pos") or zt_graph.pos is None:
        return None

    x = torch.tensor(zt_graph.node_features, dtype=torch.float)
    edge_index = torch.tensor(zt_graph.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(zt_graph.edge_features, dtype=torch.float)
    pos = torch.tensor(zt_graph.pos, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    try:
        # Convert to edge graph + compute circle_index
        edge_data = to_edge_graph(data)
        edge_data.pos = None  # Remove pos after circle_index computed

        if label is not None:
            edge_data.y = torch.tensor([label], dtype=torch.long)

        return edge_data
    except Exception as e:
        return None


class ChiENNClassifier(nn.Module):
    """ChiENN model adapted for 4-class classification on ZT graphs."""

    def __init__(self, in_node_dim, hidden_dim=128, n_classes=4, n_layers=3,
                 k_neighbors=3, dropout=0.1):
        super().__init__()
        self.chienn = ChiENNModel(
            k_neighbors=k_neighbors,
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_dim=n_classes,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, batch):
        return self.chienn(batch)


def collate_chienn_batch(data_list):
    """Custom collate function for ChiENN batches."""
    batch = Batch.from_data_list(data_list, exclude_keys=["circle_index"])
    batch.circle_index = collate_circle_index(data_list, K_NEIGHBORS)
    return batch


def load_evans_data():
    """Load Evans ZT graphs with 3D coords."""
    meta = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    _, y_all, valid_mask, _ = prepare_Xy()
    splits = load_splits()

    evans_mask = (meta["auxiliary_type"] == "evans").values
    combined_mask = valid_mask & evans_mask
    y = np.where(combined_mask, y_all, -1).astype(int)

    with open(ZT_GRAPHS_PATH, "rb") as f:
        zt_data = pickle.load(f)

    return zt_data["graphs"], zt_data["orig_indices"], y, combined_mask, splits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        p = F.softmax(out, dim=1)
        preds.append(p.argmax(1).cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())
        probs.append(p.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    return balanced_accuracy_score(y_true, y_pred), y_pred, y_true, y_prob


def run_split(graphs, orig_indices, y, combined_mask, split_data,
              device, epochs=80, lr=1e-3, hidden_dim=128):
    """Train ChiENN on one split."""
    tr_raw = np.array(split_data["train"], dtype=int)
    tr_full = tr_raw[combined_mask[tr_raw]]
    te_raw = np.array(split_data["test"], dtype=int)
    te_full = te_raw[combined_mask[te_raw]]

    va = tr_full[-max(1, len(tr_full) // 10):]
    tr = tr_full[:-len(va)]

    if len(tr) < 10 or len(te_full) < 3:
        return None

    orig_to_zt = {orig: i for i, orig in enumerate(orig_indices)}

    def make_data_list(indices):
        data_list, valid_idx = [], []
        for idx in indices:
            if idx not in orig_to_zt:
                continue
            g = graphs[orig_to_zt[idx]]
            d = zt_graph_to_chienn_data(g, label=int(y[idx]))
            if d is not None:
                data_list.append(d)
                valid_idx.append(idx)
        return data_list, valid_idx

    tr_data, tr_idx = make_data_list(tr)
    va_data, va_idx = make_data_list(va)
    te_data, te_idx = make_data_list(te_full)

    if len(tr_data) < 10 or len(te_data) < 3:
        return None

    # Determine input dim from first data point
    in_dim = tr_data[0].x.shape[1]

    from torch.utils.data import DataLoader as TorchDL

    tr_loader = TorchDL(tr_data, batch_size=32, shuffle=True, collate_fn=collate_chienn_batch)
    va_loader = TorchDL(va_data, batch_size=64, shuffle=False, collate_fn=collate_chienn_batch)
    te_loader = TorchDL(te_data, batch_size=64, shuffle=False, collate_fn=collate_chienn_batch)

    # Class weights
    y_train = np.array([d.y.item() for d in tr_data])
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight = torch.zeros(4, device=device)
    for c, w in zip(classes, cw):
        weight[c] = w
    criterion = nn.CrossEntropyLoss(weight=weight)

    model = ChiENNClassifier(
        in_node_dim=in_dim, hidden_dim=hidden_dim, n_classes=4,
        n_layers=3, k_neighbors=K_NEIGHBORS, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_va_acc, best_state, patience, no_improve = 0, None, 15, 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        va_acc, _, _, _ = evaluate(model, va_loader, device)
        scheduler.step()

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    te_acc, y_pred, y_true, y_prob = evaluate(model, te_loader, device)

    return {
        "bal_acc": round(te_acc, 4),
        "n_train": len(tr_data),
        "n_test": len(te_data),
        "y_pred": y_pred, "y_true": y_true, "y_prob": y_prob,
        "test_idx": np.array(te_idx),
        "best_va_acc": round(best_va_acc, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--splits", default="tscv", choices=["tscv", "grouped", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"B2: ChiENN on ZT Graphs (epochs={args.epochs}, device={device})")
    logger.info("=" * 60)

    graphs, orig_indices, y, combined_mask, splits = load_evans_data()
    logger.info(f"Evans: {combined_mask.sum()} valid, {len(graphs)} ZT graphs")

    # Test edge graph conversion on first graph
    test_g = next(g for g in graphs if g.status == "success")
    test_d = zt_graph_to_chienn_data(test_g, label=0)
    if test_d is None:
        logger.error("Edge graph conversion failed on test graph!")
        return
    logger.info(f"Edge graph test: {test_d.x.shape[0]} edge-nodes, "
                f"in_dim={test_d.x.shape[1]}, circle_index={len(test_d.circle_index)}")

    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.splits == "tscv":
        splits = {k: v for k, v in splits.items() if "tscv" in k}
    elif args.splits == "grouped":
        splits = {k: v for k, v in splits.items() if "grouped" in k}

    all_results = []

    for split_name, split_data in sorted(splits.items()):
        logger.info(f"\n  --- {split_name} ---")
        result = run_split(
            graphs, orig_indices, y, combined_mask, split_data,
            device, epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden,
        )
        if result is None:
            logger.warning(f"  Skipped")
            continue

        logger.info(f"  bal_acc={result['bal_acc']:.4f} (va={result['best_va_acc']:.4f})")

        save_predictions(OUT_PRED_DIR / f"chienn_{split_name}.csv",
                        result["test_idx"], result["y_true"],
                        result["y_pred"], result["y_prob"])

        all_results.append({
            "model": "ChiENN", "split": split_name,
            "bal_acc": result["bal_acc"], "n_train": result["n_train"], "n_test": result["n_test"],
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLE_DIR / "benchmark_zt_chienn_evans.csv", index=False)

    print("\n" + "=" * 70)
    print("B2: ChiENN EVANS-ONLY BENCHMARK")
    print("=" * 70)
    tscv = [r["bal_acc"] for r in all_results if "tscv" in r["split"]]
    grouped = [r["bal_acc"] for r in all_results if "grouped" in r["split"]]
    if tscv:
        print(f"  ChiENN TSCV:    {np.mean(tscv):.4f} ± {np.std(tscv):.4f}")
    if grouped:
        print(f"  ChiENN Grouped: {np.mean(grouped):.4f} ± {np.std(grouped):.4f}")
    print(f"\n  --- Baselines ---")
    print(f"  Tree (Evans ET):        TSCV=0.710")
    print(f"  Chemprop+153d+ZT:       TSCV=0.809")
    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
