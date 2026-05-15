#!/usr/bin/env python3
"""Phase D: Transfer Learning — 4 strategies on best GNN architecture.

Strategies:
  T0. No-transfer baseline (Evans only)
  T1. Pretrain-Finetune (all 4258 → freeze → Evans fine-tune)
  T2. Multi-task (shared encoder + dual classification heads)
  T3. Curriculum (2-class syn/anti on all → 4-class on Evans)

Also includes a hybrid approach:
  H1. GNN encoder → extract embeddings → XGBoost classifier

Uses MPNN+FiLM (best from coarse screening) as the base architecture.

Usage:
    conda run -n aldol-rxn python scripts/run_transfer_learning.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

os.environ["OMP_NUM_THREADS"] = "4"

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from chiralaldol.gnn.mpnn_diff import MPNNDiff
from chiralaldol.gnn.graph_builder import (
    ATOM_FEAT_DIM, BOND_FEAT_DIM, build_diff_graph,
)
from chiralaldol.gnn.trainer import (
    LabelSmoothingCrossEntropy, compute_class_weights,
    train_epoch, evaluate, train_and_evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT / "results"


# ========================================================================
# Build graphs for non-Evans data (for transfer learning)
# ========================================================================

def build_all_diff_graphs():
    """Build diff graphs for ALL 4258 reactions (Evans + non-Evans)."""
    cache_path = PROJECT / "data" / "processed" / "graphs" / "all_diff_graphs.pt"
    if cache_path.exists():
        graphs = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded cached all_diff_graphs: {len(graphs)}")
        return graphs

    logger.info("Building diff graphs for all 4258 reactions...")
    df = pd.read_csv(PROJECT / "data" / "processed" / "all_clean.csv")

    # Build condition features for all reactions
    # Use Evans condition columns if available, else build minimal
    evans_cond = pd.read_csv(PROJECT / "data" / "processed" / "features" / "reaction_conditions.csv")
    condition_dim = evans_cond.shape[1]

    # For non-Evans, we need to build condition features
    # Simplified: use the same metal/solvent encoding
    from chiralaldol.solvent_lookup import get_kamlet_taft, METAL_DEFAULT_SOLVENT

    graphs = []
    n_ok = 0

    for i, row in df.iterrows():
        label = int(row["label_joint"])
        mapped_rxn = row.get("Mapped_Reaction")

        if pd.isna(mapped_rxn):
            graphs.append(None)
            continue

        # Build condition vector (simplified: metal + solvent)
        cond_vec = np.zeros(condition_dim, dtype=np.float32)

        # Metal one-hot (first 9 columns: B, Cu, Li, Mg, Sn, Ti, Zn, Zr, none)
        metal_map = {"B": 0, "Cu": 1, "Li": 2, "Mg": 3, "Sn": 4, "Ti": 5, "Zn": 6, "Zr": 7}
        metal = str(row.get("metal", ""))
        if metal in metal_map:
            cond_vec[metal_map[metal]] = 1.0
        else:
            cond_vec[8] = 1.0  # "none"

        # Solvent (indices 9-13: alpha, beta, pi_star, ET30, known)
        for j, col in enumerate(["solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30"]):
            if col in df.columns and pd.notna(row.get(col)):
                cond_vec[9 + j] = float(row[col])
        if row.get("solvent_known", False):
            cond_vec[13] = 1.0

        g = build_diff_graph(mapped_rxn, label, cond_vec)
        if g is not None:
            # Also store reaction_class for multi-task
            class_map = {"EvansAux": 0, "AsymmetricDouble": 1, "AsymmetricSingle": 2, "OppolzerAux": 3}
            g.reaction_class = torch.tensor([class_map.get(row["Reaction_Class"], 0)], dtype=torch.long)
            n_ok += 1
        graphs.append(g)

        if (i + 1) % 1000 == 0:
            logger.info(f"  {i+1}/{len(df)} graphs built ({n_ok} ok)")

    torch.save(graphs, cache_path)
    logger.info(f"Saved {n_ok}/{len(df)} diff graphs to {cache_path}")
    return graphs


# ========================================================================
# Transfer Learning Strategies
# ========================================================================

def create_mpnn(hidden_dim=128, num_layers=3, dropout=0.3, fusion="film"):
    """Create MPNN model (best architecture from coarse screening)."""
    return MPNNDiff(
        node_input_dim=ATOM_FEAT_DIM + 1,
        edge_input_dim=BOND_FEAT_DIM + 1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=4,
        condition_dim=35,
        fusion=fusion,
        dropout=dropout,
    )


def strategy_t0_no_transfer(evans_graphs, split, config):
    """T0: No-transfer baseline — Evans only."""
    logger.info("\n" + "=" * 50)
    logger.info("T0: No-Transfer Baseline (Evans only)")
    logger.info("=" * 50)

    model = create_mpnn()
    train_g = [evans_graphs[i] for i in split["train"]]
    val_g = [evans_graphs[i] for i in split["val"]]
    test_g = [evans_graphs[i] for i in split["test"]]

    return train_and_evaluate(model, train_g, val_g, test_g, config, DEVICE)


def strategy_t1_pretrain_finetune(all_graphs, evans_graphs, evans_indices, split, config):
    """T1: Pretrain on all 4258 → freeze backbone → finetune on Evans."""
    logger.info("\n" + "=" * 50)
    logger.info("T1: Pretrain-Finetune")
    logger.info("=" * 50)

    # Phase 1: Pretrain on ALL data
    model = create_mpnn()
    valid_all = [g for g in all_graphs if g is not None]
    n_all = len(valid_all)

    # Split all data: 90% train, 10% val
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_all)
    n_train = int(n_all * 0.9)
    pretrain_train = [valid_all[i] for i in perm[:n_train]]
    pretrain_val = [valid_all[i] for i in perm[n_train:]]

    logger.info(f"  Pretrain: {len(pretrain_train)} train, {len(pretrain_val)} val")

    # Pretrain
    pretrain_config = {**config, "epochs": 50, "patience": 10}
    pretrain_loader = DataLoader(pretrain_train, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(pretrain_val, batch_size=config.get("batch_size", 32))

    model = model.to(DEVICE)
    labels = np.array([g.y.item() for g in pretrain_train])
    weights = compute_class_weights(labels).to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights)
    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=1e-4)

    best_val = 0
    best_state = None
    for epoch in range(1, 51):
        train_epoch(model, pretrain_loader, optimizer, criterion, DEVICE)
        val_acc, _, _, _ = evaluate(model, val_loader, DEVICE)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            logger.info(f"  Pretrain epoch {epoch}: val_acc={val_acc:.4f} (best={best_val:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    logger.info(f"  Pretrain done: best_val={best_val:.4f}")

    # Phase 2: Freeze backbone, finetune classifier on Evans
    # Freeze conv layers
    for name, param in model.named_parameters():
        if "classifier" not in name and "film" not in name:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Finetune: {n_trainable} trainable params (classifier + FiLM)")

    train_g = [evans_graphs[i] for i in split["train"]]
    val_g = [evans_graphs[i] for i in split["val"]]
    test_g = [evans_graphs[i] for i in split["test"]]

    finetune_config = {**config, "epochs": 50, "patience": 15, "lr": 1e-4}
    return train_and_evaluate(model, train_g, val_g, test_g, finetune_config, DEVICE)


def strategy_t2_multitask(all_graphs, evans_graphs, split, config):
    """T2: Multi-task — shared encoder + dual heads (stereo + reaction_class)."""
    logger.info("\n" + "=" * 50)
    logger.info("T2: Multi-Task Joint Training")
    logger.info("=" * 50)

    model = create_mpnn()
    model = model.to(DEVICE)

    # Add second classification head for reaction class
    reaction_class_head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 4),  # 4 reaction classes
    ).to(DEVICE)

    # Combine all parameters
    all_params = list(model.parameters()) + list(reaction_class_head.parameters())
    optimizer = AdamW(all_params, lr=config.get("lr", 1e-3), weight_decay=1e-4)

    # Prepare data: Evans + non-Evans together
    valid_all = [g for g in all_graphs if g is not None]
    valid_evans_train = [evans_graphs[i] for i in split["train"] if evans_graphs[i] is not None]
    valid_evans_val = [evans_graphs[i] for i in split["val"] if evans_graphs[i] is not None]
    valid_evans_test = [evans_graphs[i] for i in split["test"] if evans_graphs[i] is not None]

    # Mix Evans and non-Evans for training
    rng = np.random.RandomState(42)
    non_evans = [g for g in valid_all if not any(
        torch.equal(g.x, eg.x) for eg in valid_evans_train[:1])]  # simplified check
    train_mix = valid_evans_train + [g for g in valid_all if g not in valid_evans_train]

    labels_mix = np.array([g.y.item() for g in train_mix])
    weights = compute_class_weights(labels_mix).to(DEVICE)
    criterion_stereo = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights)
    criterion_class = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_mix, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(valid_evans_val, batch_size=config.get("batch_size", 32))
    test_loader = DataLoader(valid_evans_test, batch_size=config.get("batch_size", 32))

    best_val = 0
    best_state = None
    best_head_state = None

    for epoch in range(1, config.get("epochs", 100) + 1):
        model.train()
        reaction_class_head.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            # Stereo prediction (main task)
            logits_stereo = model(batch)
            loss_stereo = criterion_stereo(logits_stereo, batch.y.squeeze())

            # Reaction class prediction (auxiliary task)
            # Extract graph embedding from model's readout
            x = batch.x
            h = model.node_embed(x)
            for conv, norm in zip(model.convs, model.norms):
                h_new = conv(h, batch.edge_index, batch.edge_attr)
                h_new = norm(h_new)
                h_new = model.act(h_new)
                h = h + h_new
            graph_emb = model.readout(h, batch.batch)
            logits_class = reaction_class_head(graph_emb)

            if hasattr(batch, "reaction_class"):
                loss_class = criterion_class(logits_class, batch.reaction_class.squeeze())
            else:
                loss_class = torch.tensor(0.0, device=DEVICE)

            loss = loss_stereo + 0.3 * loss_class
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on Evans val
        val_acc, _, _, _ = evaluate(model, val_loader, DEVICE)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}: val_acc={val_acc:.4f} (best={best_val:.4f})")

        if epoch - 1 > 20 and val_acc < best_val - 0.05:
            break

    if best_state:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, DEVICE)
    return {
        "bal_acc": test_acc,
        "best_val_acc": best_val,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def strategy_t3_curriculum(all_graphs, evans_graphs, split, config):
    """T3: Curriculum — 2-class (syn/anti) on all → 4-class on Evans."""
    logger.info("\n" + "=" * 50)
    logger.info("T3: Curriculum Learning (2-class → 4-class)")
    logger.info("=" * 50)

    # Phase 1: 2-class syn/anti on ALL data
    model = create_mpnn(num_layers=3, hidden_dim=128)

    # Modify for 2-class output
    model.classifier = nn.Sequential(
        nn.Linear(128 + 35, 128),  # concat fusion
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),  # 2 classes: syn vs anti
    )
    model.fusion_type = "concat"

    valid_all = [g for g in all_graphs if g is not None]

    # Convert 4-class to 2-class: syn (0,3) → 0, anti (1,2) → 1
    for g in valid_all:
        orig_label = g.y.item()
        g.y_4class = g.y.clone()
        g.y = torch.tensor([0 if orig_label in (0, 3) else 1], dtype=torch.long)

    n_all = len(valid_all)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_all)
    n_train = int(n_all * 0.9)

    pretrain_train = [valid_all[i] for i in perm[:n_train]]
    pretrain_val = [valid_all[i] for i in perm[n_train:]]

    logger.info(f"  Phase 1 (2-class): {len(pretrain_train)} train, {len(pretrain_val)} val")

    model = model.to(DEVICE)
    labels_2c = np.array([g.y.item() for g in pretrain_train])
    weights_2c = compute_class_weights(labels_2c, 2).to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights_2c)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pretrain_loader = DataLoader(pretrain_train, batch_size=32, shuffle=True)
    val_loader_2c = DataLoader(pretrain_val, batch_size=32)

    best_val = 0
    best_state = None
    for epoch in range(1, 31):
        train_epoch(model, pretrain_loader, optimizer, criterion, DEVICE)
        val_acc, _, _, _ = evaluate(model, val_loader_2c, DEVICE)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            logger.info(f"  2-class epoch {epoch}: val={val_acc:.4f} (best={best_val:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    logger.info(f"  Phase 1 done: 2-class val={best_val:.4f}")

    # Restore 4-class labels
    for g in valid_all:
        g.y = g.y_4class

    # Phase 2: Replace classifier with 4-class head, finetune on Evans
    model.classifier = nn.Sequential(
        nn.Linear(128 + 35, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 4),  # 4 classes
    ).to(DEVICE)

    train_g = [evans_graphs[i] for i in split["train"]]
    val_g = [evans_graphs[i] for i in split["val"]]
    test_g = [evans_graphs[i] for i in split["test"]]

    finetune_config = {**config, "epochs": 80, "patience": 20, "lr": 5e-4}
    return train_and_evaluate(model, train_g, val_g, test_g, finetune_config, DEVICE)


def strategy_h1_hybrid(evans_graphs, split, config):
    """H1: Hybrid — GNN encoder → XGBoost classifier.

    Train MPNN on Evans, extract graph embeddings, then use XGBoost
    on embeddings + original 75d features.
    """
    logger.info("\n" + "=" * 50)
    logger.info("H1: Hybrid (GNN embeddings + XGBoost)")
    logger.info("=" * 50)

    import xgboost as xgb
    from chiralaldol.feature_builder import build_chiralaldol_v2_features

    # First train MPNN
    model = create_mpnn()
    train_g = [evans_graphs[i] for i in split["train"]]
    val_g = [evans_graphs[i] for i in split["val"]]
    test_g = [evans_graphs[i] for i in split["test"]]

    result = train_and_evaluate(model, train_g, val_g, test_g, config, DEVICE)
    logger.info(f"  GNN-only: bal_acc={result['bal_acc']:.4f}")

    # Extract embeddings from trained model
    model = model.to(DEVICE)
    model.eval()

    def extract_embeddings(graphs):
        loader = DataLoader([g for g in graphs if g is not None], batch_size=64)
        embs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                x = batch.x
                h = model.node_embed(x)
                for i, (conv, norm) in enumerate(zip(model.convs, model.norms)):
                    h_new = conv(h, batch.edge_index, batch.edge_attr)
                    h_new = norm(h_new)
                    h_new = model.act(h_new)
                    h_new = model.dropout(h_new)
                    h = h + h_new
                    if model.fusion_type == "film":
                        cond = batch.condition.squeeze(1) if batch.condition.dim() == 3 else batch.condition
                        h = model.film_layers[i](h, cond[batch.batch])
                graph_emb = model.readout(h, batch.batch)
                embs.append(graph_emb.cpu().numpy())
        return np.concatenate(embs, axis=0)

    emb_train = extract_embeddings(train_g)
    emb_val = extract_embeddings(val_g)
    emb_test = extract_embeddings(test_g)

    logger.info(f"  Embeddings: train={emb_train.shape}, val={emb_val.shape}, test={emb_test.shape}")

    # Load V2 features
    X_v2, _ = build_chiralaldol_v2_features(PROJECT)
    X_train = np.hstack([X_v2[split["train"]], emb_train])
    X_val = np.hstack([X_v2[split["val"]], emb_val])
    X_test = np.hstack([X_v2[split["test"]], emb_test])

    labels = pd.read_csv(PROJECT / "data" / "processed" / "features" / "labels.csv")
    y = labels["label_joint"].values.astype(int)
    y_train = y[split["train"]]
    y_val = y[split["val"]]
    y_test = y[split["test"]]

    logger.info(f"  Hybrid features: {X_train.shape[1]}d (75d V2 + {emb_train.shape[1]}d GNN)")

    # Train XGBoost on hybrid features
    sw = compute_sample_weight("balanced", y_train)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multi:softprob", "num_class": 4,
                    "tree_method": "hist", "random_state": 42,
                    "n_jobs": 2, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_train, y_train, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m

    test_acc = balanced_accuracy_score(y_test, best_m.predict(X_test))
    y_pred = best_m.predict(X_test)
    y_prob = best_m.predict_proba(X_test)

    logger.info(f"  Hybrid XGBoost: val={best_acc:.4f}, test bal_acc={test_acc:.4f}")

    return {
        "bal_acc": test_acc,
        "best_val_acc": best_acc,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ========================================================================
# Main
# ========================================================================

def main():
    logger.info("=" * 60)
    logger.info("Phase D: Transfer Learning Experiments")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)

    # Load Evans graphs
    evans_graphs = torch.load(
        PROJECT / "data" / "processed" / "graphs" / "diff_graphs.pt",
        weights_only=False,
    )
    logger.info(f"Evans diff graphs: {len(evans_graphs)}")

    # Build/load ALL diff graphs
    all_graphs = build_all_diff_graphs()

    # Load temporal split
    with open(PROJECT / "data" / "processed" / "splits" / "evans_temporal.json") as f:
        split = json.load(f)

    config = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 20,
        "batch_size": 32,
        "label_smoothing": 0.1,
    }

    results = {}

    # T0: No-transfer
    r = strategy_t0_no_transfer(evans_graphs, split, config)
    results["T0_no_transfer"] = r
    logger.info(f"  ★ T0: {r['bal_acc']:.4f}")

    # T1: Pretrain-Finetune
    evans_indices = set(range(len(evans_graphs)))
    r = strategy_t1_pretrain_finetune(all_graphs, evans_graphs, evans_indices, split, config)
    results["T1_pretrain_finetune"] = r
    logger.info(f"  ★ T1: {r['bal_acc']:.4f}")

    # T2: Multi-task
    r = strategy_t2_multitask(all_graphs, evans_graphs, split, config)
    results["T2_multitask"] = r
    logger.info(f"  ★ T2: {r['bal_acc']:.4f}")

    # T3: Curriculum
    r = strategy_t3_curriculum(all_graphs, evans_graphs, split, config)
    results["T3_curriculum"] = r
    logger.info(f"  ★ T3: {r['bal_acc']:.4f}")

    # H1: Hybrid
    r = strategy_h1_hybrid(evans_graphs, split, config)
    results["H1_hybrid"] = r
    logger.info(f"  ★ H1: {r['bal_acc']:.4f}")

    # Save summary
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFER LEARNING RESULTS")
    logger.info("=" * 60)

    summary = []
    for name, r in results.items():
        acc = r.get("bal_acc", 0.0)
        logger.info(f"  {name:30s}: bal_acc={acc:.4f}")
        summary.append({"strategy": name, "bal_acc": acc,
                        "best_val_acc": r.get("best_val_acc", 0.0)})

    summary_df = pd.DataFrame(summary).sort_values("bal_acc", ascending=False)
    out_path = RESULTS_DIR / "tables" / "transfer_learning_results.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved to {out_path}")

    # Save predictions for best model
    best_name = summary_df.iloc[0]["strategy"]
    best_result = results[best_name]
    if "y_true" in best_result:
        pred_dir = RESULTS_DIR / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        test_idx = split["test"][:len(best_result["y_true"])]
        pred_df = pd.DataFrame({
            "idx": test_idx,
            "y_true": best_result["y_true"],
            "y_pred": best_result["y_pred"],
        })
        for c in range(4):
            pred_df[f"prob_{c}"] = best_result["y_prob"][:, c]
        pred_df.to_csv(pred_dir / f"transfer_best_{best_name}_evans_temporal.csv", index=False)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
