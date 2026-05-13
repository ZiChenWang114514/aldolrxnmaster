#!/usr/bin/env python
"""ChemAHNet-style chemistry-informed DL for aldol stereochemistry prediction.

Inspired by: Nature Computational Science 2025/2026 — ChemAHNet for asymmetric hydrogenation.
Adapted for aldol: three modules encoding reaction structure + conditions + stereo-awareness.

Architecture:
  Module 1: DRFP reaction encoder (2048 → 256)
  Module 2: Conditions encoder (14 → 64) with metal/solvent attention
  Module 3: Stereo-aware fusion with cross-attention between structure and conditions
  Output: 4-class classification (Ca×Cb joint label)
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


class ReactionEncoder(nn.Module):
    """Module 1: Encode DRFP reaction fingerprint."""
    def __init__(self, input_dim=2048, hidden_dim=512, out_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ConditionsEncoder(nn.Module):
    """Module 2: Encode reaction conditions with metal/solvent attention."""
    def __init__(self, n_metal, n_solvent, out_dim=64):
        super().__init__()
        # Metal embedding (one-hot → learned)
        self.metal_embed = nn.Sequential(
            nn.Linear(n_metal, 32),
            nn.GELU(),
        )
        # Solvent embedding (continuous Kamlet-Taft → learned)
        self.solvent_embed = nn.Sequential(
            nn.Linear(n_solvent, 32),
            nn.GELU(),
        )
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x_metal, x_solvent):
        m = self.metal_embed(x_metal)
        s = self.solvent_embed(x_solvent)
        return self.fusion(torch.cat([m, s], dim=1))


class StereoFusion(nn.Module):
    """Module 3: Cross-attention fusion between structure and conditions."""
    def __init__(self, struct_dim=256, cond_dim=64, hidden_dim=128, n_heads=4, dropout=0.2):
        super().__init__()
        self.total_dim = struct_dim + cond_dim

        # Cross-attention: conditions attend to structure
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.proj_struct = nn.Linear(struct_dim, hidden_dim)
        self.proj_cond = nn.Linear(cond_dim, hidden_dim)

        # Final fusion
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, struct_emb, cond_emb):
        # Project to common dimension
        s = self.proj_struct(struct_emb).unsqueeze(1)  # (B, 1, H)
        c = self.proj_cond(cond_emb).unsqueeze(1)  # (B, 1, H)

        # Cross-attention: query=conditions, key/value=structure
        attn_out, _ = self.cross_attn(c, s, s)  # (B, 1, H)
        attn_out = attn_out.squeeze(1)  # (B, H)

        # Concat attended + structure projection
        fused = torch.cat([s.squeeze(1), attn_out], dim=1)  # (B, 2H)
        return self.classifier(fused)


class ChemAHNetAldol(nn.Module):
    """Full ChemAHNet-style model for aldol stereochemistry."""
    def __init__(self, drfp_dim=2048, n_metal=9, n_solvent=5, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.rxn_encoder = ReactionEncoder(drfp_dim, 512, 256, dropout)
        self.cond_encoder = ConditionsEncoder(n_metal, n_solvent, 64)
        self.stereo_fusion = StereoFusion(256, 64, hidden_dim, n_heads=4, dropout=dropout)

    def forward(self, x_drfp, x_metal, x_solvent):
        struct_emb = self.rxn_encoder(x_drfp)
        cond_emb = self.cond_encoder(x_metal, x_solvent)
        logits = self.stereo_fusion(struct_emb, cond_emb)
        return logits


def run_split(split_name, epochs=50):
    logger.info(f"\n{'='*60}\n  ChemAHNet-Aldol: {split_name}\n{'='*60}")

    # Load features
    X_drfp = np.load(FEAT_DIR / "drfp_fps.npz")["X"].astype(np.float32)
    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")

    # Split conditions into metal, solvent, and reagent features
    metal_cols = [c for c in cond_df.columns if c.startswith("metal_")]
    solvent_cols = [c for c in cond_df.columns if c.startswith("solvent_")]
    reagent_cols = [c for c in cond_df.columns if c.startswith(("base_", "activator_", "has_", "reagent_"))]
    X_metal = cond_df[metal_cols].values.astype(np.float32)
    # Solvent + reagent combined as "non-metal conditions"
    X_solvent = cond_df[solvent_cols + reagent_cols].values.astype(np.float32)

    labels_df = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels_df["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)
    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    logger.info(f"DRFP: {X_drfp.shape[1]}, Metal: {X_metal.shape[1]}, Solvent+Reagent: {X_solvent.shape[1]}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tensors
    drfp_tr = torch.tensor(X_drfp[tr], dtype=torch.float32)
    drfp_va = torch.tensor(X_drfp[va], dtype=torch.float32)
    drfp_te = torch.tensor(X_drfp[te], dtype=torch.float32)
    metal_tr = torch.tensor(X_metal[tr], dtype=torch.float32)
    metal_va = torch.tensor(X_metal[va], dtype=torch.float32)
    metal_te = torch.tensor(X_metal[te], dtype=torch.float32)
    solv_tr = torch.tensor(X_solvent[tr], dtype=torch.float32)
    solv_va = torch.tensor(X_solvent[va], dtype=torch.float32)
    solv_te = torch.tensor(X_solvent[te], dtype=torch.float32)
    y_tr_t = torch.tensor(y[tr], dtype=torch.long)
    y_va_t = torch.tensor(y[va], dtype=torch.long)

    tr_ds = TensorDataset(drfp_tr, metal_tr, solv_tr, y_tr_t)
    va_ds = TensorDataset(drfp_va, metal_va, solv_va, y_va_t)
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=128)

    # Model
    model = ChemAHNetAldol(
        drfp_dim=X_drfp.shape[1],
        n_metal=X_metal.shape[1],
        n_solvent=X_solvent.shape[1],
        hidden_dim=128,
        dropout=0.2,
    ).to(device)

    # Class-weighted loss
    cw = compute_class_weight("balanced", classes=np.arange(4), y=y[tr])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, patience = 0, None, 0
    for ep in range(epochs):
        model.train()
        for drfp, metal, solv, lab in tr_dl:
            drfp, metal, solv, lab = drfp.to(device), metal.to(device), solv.to(device), lab.to(device)
            optimizer.zero_grad()
            logits = model(drfp, metal, solv)
            loss = criterion(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        preds = []
        with torch.no_grad():
            for drfp, metal, solv, _ in va_dl:
                logits = model(drfp.to(device), metal.to(device), solv.to(device))
                preds.extend(logits.argmax(1).cpu().numpy())
        acc = balanced_accuracy_score(y[va], np.array(preds))

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (ep + 1) % 10 == 0:
            logger.info(f"    ep{ep+1}: val_bacc={acc:.4f}")
        if patience >= 10:
            logger.info(f"    Early stop at ep{ep+1}")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()

    # Predict test
    with torch.no_grad():
        logits = model(drfp_te.to(device), metal_te.to(device), solv_te.to(device))
        y_prob = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(1).cpu().numpy()

    y_test = y[te]
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  ChemAHNet-Aldol: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint_acc={metrics['joint_accuracy']:.4f}")

    # Save
    out_df = pd.DataFrame({"idx": te, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out_df[f"prob_{c}"] = y_prob[:, c]
    out_df.to_csv(RESULTS_DIR / "predictions" / f"chemahnet_aldol_{split_name}.csv", index=False)

    return metrics, ci


if __name__ == "__main__":
    run_split("evans_temporal")
    run_split("evans_scaffold")
    run_split("evans_grouped_random_seed42")
    logger.info("\nDone! Now run: python scripts/rebuild_comparison.py")
