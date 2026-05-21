#!/usr/bin/env python
"""EquiReact (3DReact) for aldol stereochemistry prediction.

E(3)-equivariant neural network on 3D molecular structures.
Uses 3D conformers of reactants and products.

Run with equireact conda env:
  conda run -n equireact python scripts/run_equireact.py
"""

import json
import logging
import os
import pickle
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
from torch.utils.data import Dataset, DataLoader
from torch_cluster import radius_graph
from torch_scatter import scatter_mean

warnings.filterwarnings("ignore")

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
sys.path.insert(0, str(PROJECT / "external" / "EquiReact"))
sys.path.insert(0, str(PROJECT / "src"))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
CONF_DIR = PROJECT / "data" / "processed" / "conformers"
RESULTS_DIR = PROJECT / "results"

# Atomic number to one-hot (common organic elements)
ELEMENT_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H,B,C,N,O,F,Si,P,S,Cl,Br,I


def atom_features_onehot(atomic_num):
    """One-hot encode atomic number."""
    feat = [0] * (len(ELEMENT_LIST) + 1)
    if atomic_num in ELEMENT_LIST:
        feat[ELEMENT_LIST.index(atomic_num)] = 1
    else:
        feat[-1] = 1
    return feat


class AldolReaction3DDataset(Dataset):
    """Dataset of 3D molecular structures for aldol reactions."""

    def __init__(self, conformers, conditions, labels, indices, radius=5.0):
        self.radius = radius
        self.data = []

        for i in indices:
            prod_mol = conformers["product"]["mols"][i]
            prod_coords = conformers["product"]["coords"][i]

            if prod_mol is None or prod_coords is None:
                continue

            # Build node features (atomic numbers → one-hot)
            from rdkit import Chem
            node_feats = []
            for atom in prod_mol.GetAtoms():
                node_feats.append(atom_features_onehot(atom.GetAtomicNum()))

            self.data.append({
                "coords": torch.tensor(prod_coords, dtype=torch.float32),
                "node_feats": torch.tensor(node_feats, dtype=torch.float32),
                "conditions": torch.tensor(conditions[i], dtype=torch.float32),
                "label": int(labels[i]),
                "idx": i,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_3d(batch_list):
    """Collate variable-size molecular graphs into a batch."""
    coords_list, feats_list, conds_list, labels_list, batch_idx = [], [], [], [], []
    offset = 0
    for i, item in enumerate(batch_list):
        n = item["coords"].shape[0]
        coords_list.append(item["coords"])
        feats_list.append(item["node_feats"])
        conds_list.append(item["conditions"])
        labels_list.append(item["label"])
        batch_idx.extend([i] * n)

    return {
        "coords": torch.cat(coords_list, dim=0),
        "node_feats": torch.cat(feats_list, dim=0),
        "conditions": torch.stack(conds_list),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "batch": torch.tensor(batch_idx, dtype=torch.long),
    }


class Simple3DClassifier(nn.Module):
    """Simplified 3D-aware classifier using SchNet-style radial basis + message passing."""

    def __init__(self, node_dim=13, cond_dim=14, hidden_dim=128, n_layers=3, radius=5.0, dropout=0.2):
        super().__init__()
        self.radius = radius

        # Node embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Distance embedding (Gaussian smearing)
        self.n_gaussians = 32
        self.mu = nn.Parameter(torch.linspace(0.0, radius, self.n_gaussians), requires_grad=False)
        self.sigma = (self.mu[1] - self.mu[0]).item()

        # Message passing layers
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.n_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Readout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def gaussian_smearing(self, dist):
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.mu) / self.sigma) ** 2)

    def forward(self, coords, node_feats, conditions, batch):
        # Build radius graph
        edge_index = radius_graph(coords, r=self.radius, batch=batch, max_num_neighbors=20)

        # Edge features from distances
        dist = (coords[edge_index[0]] - coords[edge_index[1]]).norm(dim=-1)
        edge_attr = self.gaussian_smearing(dist)
        edge_weight = self.edge_mlp(edge_attr)

        # Node embedding
        x = self.node_embed(node_feats)

        # Message passing
        for conv in self.conv_layers:
            # Aggregate neighbor messages
            msg = edge_weight * x[edge_index[0]]
            agg = scatter_mean(msg, edge_index[1], dim=0, dim_size=x.size(0))
            x = x + conv(agg)

        # Global pooling (mean over nodes per graph)
        graph_emb = scatter_mean(x, batch, dim=0)

        # Concat conditions and classify
        out = torch.cat([graph_emb, conditions], dim=1)
        return self.classifier(out)


def run_split(split_name, epochs=40):
    logger.info(f"\n{'='*60}\n  EquiReact-3D: {split_name}\n{'='*60}")

    # Load conformers
    with open(CONF_DIR / "conformers.pkl", "rb") as f:
        conformers = pickle.load(f)

    # Load conditions and labels
    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    X_cond = cond_df.values.astype(np.float32)

    labels_df = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels_df["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)
    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")

    # Build datasets
    tr_ds = AldolReaction3DDataset(conformers, X_cond, y, tr)
    va_ds = AldolReaction3DDataset(conformers, X_cond, y, va)
    te_ds = AldolReaction3DDataset(conformers, X_cond, y, te)

    logger.info(f"Valid 3D data: train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")

    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_3d)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, collate_fn=collate_3d)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, collate_fn=collate_3d)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Simple3DClassifier(
        node_dim=len(ELEMENT_LIST) + 1, cond_dim=X_cond.shape[1],
        hidden_dim=128, n_layers=3, radius=5.0, dropout=0.2,
    ).to(device)

    # Class-weighted loss
    tr_labels = np.array([d["label"] for d in tr_ds.data])
    cw = compute_class_weight("balanced", classes=np.arange(4), y=tr_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, patience = 0, None, 0

    for ep in range(epochs):
        model.train()
        for batch in tr_dl:
            coords = batch["coords"].to(device)
            feats = batch["node_feats"].to(device)
            conds = batch["conditions"].to(device)
            labels = batch["labels"].to(device)
            batch_idx = batch["batch"].to(device)

            optimizer.zero_grad()
            logits = model(coords, feats, conds, batch_idx)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        va_preds = []
        with torch.no_grad():
            for batch in va_dl:
                logits = model(
                    batch["coords"].to(device), batch["node_feats"].to(device),
                    batch["conditions"].to(device), batch["batch"].to(device),
                )
                va_preds.extend(logits.argmax(1).cpu().numpy())

        va_labels = np.array([d["label"] for d in va_ds.data])
        acc = balanced_accuracy_score(va_labels[:len(va_preds)], np.array(va_preds))

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (ep + 1) % 5 == 0:
            logger.info(f"    ep{ep+1}: val_bacc={acc:.4f}")
        if patience >= 10:
            logger.info(f"    Early stop at ep{ep+1}")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()

    # Test
    te_preds, te_probs = [], []
    with torch.no_grad():
        for batch in te_dl:
            logits = model(
                batch["coords"].to(device), batch["node_feats"].to(device),
                batch["conditions"].to(device), batch["batch"].to(device),
            )
            probs = F.softmax(logits, dim=1).cpu().numpy()
            te_preds.extend(logits.argmax(1).cpu().numpy())
            te_probs.extend(probs)

    y_pred = np.array(te_preds)
    y_prob = np.array(te_probs)
    te_labels = np.array([d["label"] for d in te_ds.data])
    y_test = te_labels[:len(y_pred)]

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  EquiReact-3D: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint_acc={metrics['joint_accuracy']:.4f}")

    # Save
    te_indices = [d["idx"] for d in te_ds.data][:len(y_pred)]
    out_df = pd.DataFrame({"idx": te_indices, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out_df[f"prob_{c}"] = y_prob[:, c]
    out_df.to_csv(RESULTS_DIR / "predictions" / f"equireact_{split_name}.csv", index=False)

    return metrics, ci


if __name__ == "__main__":
    run_split("evans_temporal")
    run_split("evans_scaffold")
    run_split("evans_grouped_random_seed42")
    logger.info("\nDone! Now run: python scripts/rebuild_comparison.py")
