#!/usr/bin/env python
"""Prototypical Networks for 4-class aldol stereochemistry prediction.

Meta-learning approach: learn an embedding space where class prototypes
are well-separated. Especially good for imbalanced small datasets
(anti classes C1=10.3%, C2=7.4%).

Based on: Snell et al. (NeurIPS 2017) + Nature Communications 2025 (asymmetric catalysis).
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"


class ProtoEncoder(nn.Module):
    """Encoder for prototypical network: DRFP → embedding space."""
    def __init__(self, input_dim=2048, hidden_dim=512, embed_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_prototypes(embeddings, labels, n_classes=4):
    """Compute class prototypes as mean embedding per class."""
    prototypes = torch.zeros(n_classes, embeddings.size(1), device=embeddings.device)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(0)
    return prototypes


def proto_loss(query_emb, query_labels, prototypes):
    """Prototypical network loss: negative log-softmax of distances."""
    # Euclidean distance to each prototype
    dists = torch.cdist(query_emb, prototypes)  # (N, n_classes)
    log_probs = F.log_softmax(-dists, dim=1)  # closer = higher prob
    loss = F.nll_loss(log_probs, query_labels)
    return loss, -dists  # return logits for prediction


def episodic_train(encoder, X_train, y_train, X_val, y_val,
                   n_episodes=2000, n_support=5, n_query=10,
                   lr=1e-3, device="cuda:0", patience=200):
    """Train encoder with episodic prototypical learning."""
    encoder = encoder.to(device)
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

    classes = torch.unique(y_tr)
    n_classes = len(classes)

    # Build per-class index lists
    class_indices = {c.item(): (y_tr == c).nonzero(as_tuple=True)[0] for c in classes}

    best_acc, best_state, wait = 0, None, 0

    for ep in range(n_episodes):
        encoder.train()

        # Sample support and query from training set
        support_idx, query_idx = [], []
        for c in classes:
            idx = class_indices[c.item()]
            n_available = len(idx)
            # Adjust n_support/n_query if class is too small
            ns = min(n_support, n_available // 2)
            nq = min(n_query, n_available - ns)
            if ns < 1 or nq < 1:
                ns, nq = 1, min(1, n_available - 1)
            perm = torch.randperm(n_available)[:ns + nq]
            support_idx.append(idx[perm[:ns]])
            query_idx.append(idx[perm[ns:ns + nq]])

        support_idx = torch.cat(support_idx)
        query_idx = torch.cat(query_idx)

        # Forward pass
        support_emb = encoder(X_tr[support_idx])
        query_emb = encoder(X_tr[query_idx])

        # Compute prototypes from support
        prototypes = compute_prototypes(support_emb, y_tr[support_idx], n_classes)

        # Loss on query
        loss, _ = proto_loss(query_emb, y_tr[query_idx], prototypes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation every 50 episodes
        if (ep + 1) % 50 == 0:
            encoder.eval()
            with torch.no_grad():
                # Use ALL training data as support for validation
                all_emb = encoder(X_tr)
                prototypes = compute_prototypes(all_emb, y_tr, n_classes)
                val_emb = encoder(X_va)
                dists = torch.cdist(val_emb, prototypes)
                val_pred = (-dists).argmax(1).cpu().numpy()
            acc = balanced_accuracy_score(y_va.cpu().numpy(), val_pred)

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                wait = 0
            else:
                wait += 50

            if (ep + 1) % 200 == 0:
                logger.info(f"    ep{ep+1}: loss={loss.item():.4f}, val_bacc={acc:.4f}")

            if wait >= patience:
                logger.info(f"    Early stop at ep{ep+1}")
                break

    encoder.load_state_dict(best_state)
    encoder.to(device)
    return encoder


def predict_protonet(encoder, X_train, y_train, X_test, device="cuda:0"):
    """Predict using full training set as support."""
    encoder.eval()
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        all_emb = encoder(X_tr)
        prototypes = compute_prototypes(all_emb, y_tr, 4)
        test_emb = encoder(X_te)
        dists = torch.cdist(test_emb, prototypes)
        logits = -dists
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    return preds, probs


def run_split(split_name):
    logger.info(f"\n{'='*60}\n  ProtoNet: {split_name}\n{'='*60}")

    # Load DRFP features
    X_drfp = np.load(FEAT_DIR / "drfp_fps.npz")["X"].astype(np.float32)

    # Load conditions and concat
    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    X_cond = cond_df.values.astype(np.float32)
    X = np.hstack([X_drfp, X_cond])  # DRFP + conditions

    labels_df = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels_df["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)
    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    logger.info(f"Features: {X.shape[1]}-dim (DRFP 2048 + Cond {X_cond.shape[1]})")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    encoder = ProtoEncoder(input_dim=X.shape[1], hidden_dim=512, embed_dim=128, dropout=0.2)

    encoder = episodic_train(
        encoder, X[tr], y[tr], X[va], y[va],
        n_episodes=3000, n_support=5, n_query=10,
        lr=1e-3, device=device, patience=300,
    )

    y_pred, y_prob = predict_protonet(encoder, X[tr], y[tr], X[te], device)
    y_test = y[te]

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  ProtoNet: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint_acc={metrics['joint_accuracy']:.4f}")

    # Save predictions
    out_df = pd.DataFrame({"idx": te, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out_df[f"prob_{c}"] = y_prob[:, c]
    out_df.to_csv(RESULTS_DIR / "predictions" / f"protonet_{split_name}.csv", index=False)

    return metrics, ci


if __name__ == "__main__":
    run_split("evans_temporal")
    run_split("evans_scaffold")
    run_split("evans_grouped_random_seed42")
    logger.info("\nDone! Now run: python scripts/rebuild_comparison.py")
