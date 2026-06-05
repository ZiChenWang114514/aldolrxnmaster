#!/usr/bin/env python3
"""Phase B2-B4: ZT-GNN Benchmark — 3 GNN architectures on ZT transition state graphs.

Models:
  1. ZT-GIN: Graph Isomorphism Network (powerful WL-equivalent baseline)
  2. ZT-GAT: Graph Attention Network v2 (attention over ZT ring interactions)
  3. ZT-Chiral: Chirality-aware GNN (type-specific message passing + ring-aware readout)

All models operate directly on ZT graphs (not SMILES). Evans-only subset.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py --model ZT-GIN --splits tscv
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py --model all --epochs 100
"""

import argparse
import json
import logging
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, PRED_DIR, RESULTS_DIR, SPLITS_DIR
from chiralaldol.data_io import prepare_Xy, load_splits, save_predictions
from chiralaldol.gnn.zt_dataset import build_pyg_dataset
from chiralaldol.gnn.zt_models import ZT_MODELS

ZT_GRAPHS_PATH = FEAT_DIR / "zt_graphs" / "evans_zt_graphs.pkl"
OUT_PRED_DIR = PRED_DIR / "zt_gnn"
TABLE_DIR = RESULTS_DIR / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zt_gnn")


def load_evans_data():
    """Load Evans ZT graphs + labels + splits."""
    from chiralaldol.config import VALID_AUXILIARIES
    meta_full = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    meta = meta_full[meta_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    X_153d, y_all, valid_mask, feat_names = prepare_Xy()
    splits = load_splits()

    evans_mask = (meta["auxiliary_type"] == "evans").values
    combined_mask = valid_mask & evans_mask

    # Load ZT graphs
    with open(ZT_GRAPHS_PATH, "rb") as f:
        zt_data = pickle.load(f)
    graphs = zt_data["graphs"]
    orig_indices = zt_data["orig_indices"]

    # Map: orig_idx → (graph, label, 153d features)
    y = np.where(combined_mask, y_all, -1).astype(int)

    return graphs, orig_indices, y, combined_mask, splits, X_153d


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.y.view(-1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return bal_acc, y_pred, y_true, y_prob


def run_split(model_cls, model_kwargs, graphs, orig_indices, y, combined_mask,
              split_data, X_153d, device, epochs=80, lr=1e-3, use_global_feat=False):
    """Train and evaluate one GNN on one split."""
    tr_raw = np.array(split_data["train"], dtype=int)
    tr_full = tr_raw[combined_mask[tr_raw]]
    te_raw = np.array(split_data["test"], dtype=int)
    te_full = te_raw[combined_mask[te_raw]]

    # Validation split
    va = tr_full[-max(1, len(tr_full) // 10):]
    tr = tr_full[:-len(va)]

    if len(tr) < 10 or len(te_full) < 3:
        return None

    # Map full-dataset indices to Evans ZT graph indices
    orig_to_zt = {orig: i for i, orig in enumerate(orig_indices)}

    def make_data_list(indices):
        data_list = []
        valid_idx = []
        for idx in indices:
            if idx in orig_to_zt:
                zt_i = orig_to_zt[idx]
                g = graphs[zt_i]
                if g.status != "success":
                    continue
                from chiralaldol.gnn.zt_dataset import zt_graph_to_pyg
                xf = X_153d[idx] if use_global_feat else None
                d = zt_graph_to_pyg(g, label=int(y[idx]), extra_features=xf)
                if d is not None:
                    data_list.append(d)
                    valid_idx.append(idx)
        return data_list, valid_idx

    tr_data, tr_idx = make_data_list(tr)
    va_data, va_idx = make_data_list(va)
    te_data, te_idx = make_data_list(te_full)

    if len(tr_data) < 10 or len(te_data) < 3:
        return None

    tr_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=64, shuffle=False)
    te_loader = DataLoader(te_data, batch_size=64, shuffle=False)

    # Class weights
    y_train = np.array([d.y.item() for d in tr_data])
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight = torch.zeros(4, device=device)
    for c, w in zip(classes, cw):
        weight[c] = w
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Model
    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_va_acc = 0
    best_state = None
    patience = 15
    no_improve = 0

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

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    te_acc, y_pred, y_true, y_prob = evaluate(model, te_loader, device)

    return {
        "bal_acc": round(te_acc, 4),
        "n_train": len(tr_data),
        "n_test": len(te_data),
        "y_pred": y_pred,
        "y_true": y_true,
        "y_prob": y_prob,
        "test_idx": np.array(te_idx),
        "best_va_acc": round(best_va_acc, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(ZT_MODELS.keys()) + ["all"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--splits", default="tscv", choices=["tscv", "grouped", "all"])
    parser.add_argument("--use-global-feat", action="store_true",
                        help="Append 153d global features to graph readout")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"ZT-GNN Benchmark (epochs={args.epochs}, device={device})")
    logger.info("=" * 60)

    graphs, orig_indices, y, combined_mask, splits, X_153d = load_evans_data()
    logger.info(f"Evans: {combined_mask.sum()} valid, {len(graphs)} ZT graphs")

    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Filter splits
    if args.splits == "tscv":
        splits = {k: v for k, v in splits.items() if "tscv" in k}
    elif args.splits == "grouped":
        splits = {k: v for k, v in splits.items() if "grouped" in k}

    # Models to run
    model_names = list(ZT_MODELS.keys()) if args.model == "all" else [args.model]

    global_feat_dim = X_153d.shape[1] if args.use_global_feat else 0

    all_results = []

    for model_name in model_names:
        model_cls = ZT_MODELS[model_name]

        # Model-specific kwargs
        if model_name == "ZT-GCPNet":
            model_kwargs = {
                "node_dim": 20, "edge_dim": 5,
                "hidden_scalar": 64, "hidden_vector": 16,
                "n_layers": args.layers, "n_classes": 4,
                "dropout": 0.2, "global_feat_dim": global_feat_dim,
            }
        elif model_name == "ZT-ChiDeK":
            model_kwargs = {
                "node_dim": 20, "edge_dim": 5, "hidden_dim": args.hidden,
                "n_backbone_layers": 3, "n_chiral_layers": 2,
                "n_classes": 4, "dropout": 0.2, "global_feat_dim": global_feat_dim,
            }
        elif model_name in ("ZT-ComENet", "ZT-Hybrid"):
            model_kwargs = {
                "node_dim": 20, "edge_dim": 5, "hidden_dim": args.hidden,
                "n_layers": args.layers, "n_classes": 4,
                "dropout": 0.2, "global_feat_dim": global_feat_dim,
                "n_rbf": 16,
            }
            if model_name == "ZT-Hybrid":
                model_kwargs["spms_dim"] = 16
        else:
            model_kwargs = {
                "node_dim": 20, "edge_dim": 5, "hidden_dim": args.hidden,
                "n_layers": args.layers, "n_classes": 4,
                "dropout": 0.2, "global_feat_dim": global_feat_dim,
            }
            if model_name == "ZT-GAT":
                model_kwargs["n_heads"] = 4

        logger.info(f"\n{'='*40}\n  Model: {model_name}\n{'='*40}")

        for split_name, split_data in sorted(splits.items()):
            result = run_split(
                model_cls, model_kwargs, graphs, orig_indices, y, combined_mask,
                split_data, X_153d, device,
                epochs=args.epochs, lr=args.lr,
                use_global_feat=args.use_global_feat,
            )

            if result is None:
                logger.warning(f"  {split_name}: skipped")
                continue

            logger.info(f"  {split_name}: bal_acc={result['bal_acc']:.4f} "
                        f"(va={result['best_va_acc']:.4f}, n_test={result['n_test']})")

            fname = f"{model_name}_{split_name}.csv".replace(" ", "_")
            save_predictions(OUT_PRED_DIR / fname,
                            result["test_idx"], result["y_true"],
                            result["y_pred"], result["y_prob"])

            all_results.append({
                "model": model_name,
                "split": split_name,
                "bal_acc": result["bal_acc"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            })

    # Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLE_DIR / "benchmark_zt_gnn_evans.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("ZT-GNN EVANS-ONLY BENCHMARK")
    print("=" * 70)

    for model_name in model_names:
        mr = [r for r in all_results if r["model"] == model_name]
        if not mr:
            continue
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        print(f"\n  {model_name}:")
        if tscv:
            print(f"    TSCV:     {np.mean(tscv):.4f} ± {np.std(tscv):.4f}")
        if scaffold:
            print(f"    Scaffold: {scaffold[0]:.4f}")
        if grouped:
            print(f"    Grouped:  {np.mean(grouped):.4f} ± {np.std(grouped):.4f}")

    print(f"\n  --- Baselines ---")
    print(f"    Tree (Evans ET):     TSCV=0.710")
    print(f"    Chemprop+153d+ZT:    TSCV=0.809")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
