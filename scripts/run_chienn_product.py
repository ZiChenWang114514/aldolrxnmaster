#!/usr/bin/env python
"""ChiENN product encoder for aldol stereochemistry prediction.

Uses ChiENN (chirality-aware GNN) to encode product molecules,
then concatenates reaction conditions for 4-class classification.

Architecture:
  Product SMILES → 3D conformer → edge graph with circle_index → ChiENN encoder
  → product embedding (128-dim) → concat conditions (14-dim) → MLP → 4-class
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "external" / "ChiENN"))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"


def build_product_graphs(product_smiles_list):
    """Convert product SMILES to ChiENN graph data objects."""
    from chienn.data import smiles_to_data_with_circle_index

    graphs = []
    failed = 0
    for i, smi in enumerate(product_smiles_list):
        try:
            data = smiles_to_data_with_circle_index(str(smi))
            if data is not None and data.x is not None:
                graphs.append(data)
            else:
                graphs.append(None)
                failed += 1
        except Exception:
            graphs.append(None)
            failed += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  Graph construction: {i+1}/{len(product_smiles_list)} ({failed} failures)")

    logger.info(f"  Total: {len(graphs)} graphs, {failed} failures")
    return graphs


class ChiENNProductClassifier(nn.Module):
    """ChiENN encoder for product + conditions → 4-class."""

    def __init__(self, in_node_dim=93, hidden_dim=128, cond_dim=14, n_layers=3, k_neighbors=3, dropout=0.2):
        super().__init__()
        from chienn.model.chienn_model import ChiENNModel

        # ChiENN encoder (output = hidden_dim)
        self.encoder = ChiENNModel(
            k_neighbors=k_neighbors,
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,  # output embedding, not final prediction
            n_layers=n_layers,
            dropout=dropout,
        )
        # Replace output_layer with identity to get embeddings
        self.encoder.output_layer = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, batch, conditions):
        # Get product embedding from ChiENN
        emb = self.encoder(batch)  # (B, hidden_dim)
        # Concat conditions
        fused = torch.cat([emb, conditions], dim=1)
        return self.classifier(fused)


def run_split(split_name, epochs=30):
    from chienn.data import collate_with_circle_index

    logger.info(f"\n{'='*60}\n  ChiENN-Product: {split_name}\n{'='*60}")

    # Load data
    evans_df = pd.read_csv(PROJECT / "data" / "processed" / "evans_clean.csv")
    product_col = "Product_" if "Product_" in evans_df.columns else "Raw_Product_Smiles"
    product_smiles = evans_df[product_col].values

    labels_df = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels_df["label_joint"].values.astype(int)

    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    X_cond = cond_df.values.astype(np.float32)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)
    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")

    # Build graphs
    logger.info("Building product graphs...")
    all_graphs = build_product_graphs(product_smiles)

    # Filter valid indices
    valid_mask = [g is not None for g in all_graphs]
    logger.info(f"Valid graphs: {sum(valid_mask)}/{len(valid_mask)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_neighbors = 3

    # Build datasets as lists of (graph, cond, label) for each split
    def make_batch_data(indices):
        graphs, conds, labels = [], [], []
        for i in indices:
            if all_graphs[i] is not None:
                graphs.append(all_graphs[i])
                conds.append(X_cond[i])
                labels.append(y[i])
        return graphs, np.array(conds, dtype=np.float32), np.array(labels, dtype=np.int64)

    tr_graphs, tr_conds, tr_labels = make_batch_data(tr)
    va_graphs, va_conds, va_labels = make_batch_data(va)
    te_graphs, te_conds, te_labels = make_batch_data(te)

    logger.info(f"Valid: train={len(tr_graphs)}, val={len(va_graphs)}, test={len(te_graphs)}")

    # Determine in_node_dim from first graph
    in_node_dim = tr_graphs[0].x.shape[1]
    cond_dim = tr_conds.shape[1]
    logger.info(f"in_node_dim={in_node_dim}, cond_dim={cond_dim}")

    # Model
    model = ChiENNProductClassifier(
        in_node_dim=in_node_dim, hidden_dim=128, cond_dim=cond_dim,
        n_layers=3, k_neighbors=k_neighbors, dropout=0.2,
    ).to(device)

    cw = compute_class_weight("balanced", classes=np.arange(4), y=tr_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 32
    best_acc, best_state, patience = 0, None, 0

    for ep in range(epochs):
        model.train()
        # Shuffle training data
        perm = np.random.permutation(len(tr_graphs))

        for start in range(0, len(tr_graphs), batch_size):
            idx = perm[start:start + batch_size]
            batch_graphs = [tr_graphs[i] for i in idx]
            batch_conds = torch.tensor(tr_conds[idx], dtype=torch.float32).to(device)
            batch_labels = torch.tensor(tr_labels[idx], dtype=torch.long).to(device)

            batch = collate_with_circle_index(batch_graphs, k_neighbors).to(device)
            optimizer.zero_grad()
            logits = model(batch, batch_conds)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        va_preds = []
        with torch.no_grad():
            for start in range(0, len(va_graphs), batch_size):
                batch_graphs = va_graphs[start:start + batch_size]
                batch_conds = torch.tensor(va_conds[start:start + batch_size], dtype=torch.float32).to(device)
                batch = collate_with_circle_index(batch_graphs, k_neighbors).to(device)
                logits = model(batch, batch_conds)
                va_preds.extend(logits.argmax(1).cpu().numpy())

        acc = balanced_accuracy_score(va_labels[:len(va_preds)], np.array(va_preds))
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (ep + 1) % 5 == 0:
            logger.info(f"    ep{ep+1}: val_bacc={acc:.4f}")
        if patience >= 8:
            logger.info(f"    Early stop at ep{ep+1}")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()

    # Predict test
    te_preds, te_probs = [], []
    with torch.no_grad():
        for start in range(0, len(te_graphs), batch_size):
            batch_graphs = te_graphs[start:start + batch_size]
            batch_conds = torch.tensor(te_conds[start:start + batch_size], dtype=torch.float32).to(device)
            batch = collate_with_circle_index(batch_graphs, k_neighbors).to(device)
            logits = model(batch, batch_conds)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            te_preds.extend(logits.argmax(1).cpu().numpy())
            te_probs.extend(probs)

    y_pred = np.array(te_preds)
    y_prob = np.array(te_probs)
    y_test = te_labels[:len(y_pred)]

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  ChiENN-Product: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint_acc={metrics['joint_accuracy']:.4f}")

    # Save (use original test indices for valid graphs only)
    valid_te_idx = [i for i in te if all_graphs[i] is not None][:len(y_pred)]
    out_df = pd.DataFrame({"idx": valid_te_idx, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out_df[f"prob_{c}"] = y_prob[:, c]
    out_df.to_csv(RESULTS_DIR / "predictions" / f"chienn_product_{split_name}.csv", index=False)

    return metrics, ci


if __name__ == "__main__":
    run_split("evans_temporal")
    run_split("evans_scaffold")
    run_split("evans_grouped_random_seed42")
    logger.info("\nDone! Now run: python scripts/rebuild_comparison.py")
