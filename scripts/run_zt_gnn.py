#!/usr/bin/env python3
"""ZT-GNN Benchmark — single-TS models + Multi-TS Attention Scorer.

Models:
  Single-TS (v1): ZT-GIN, ZT-GAT, ZT-Chiral, ZT-ChiDeK, ZT-GCPNet, ZT-ComENet, ZT-Hybrid
  Multi-TS:       MultiTS (4-TS attention scorer with EnhancedChiralMP)

Usage:
    # Multi-TS with global features + Z/E prior + aux loss (recommended)
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py \
        --model MultiTS --use-global-feat --use-ze-prior --use-aux-loss

    # Multi-TS ablation: no global features
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py \
        --model MultiTS --no-global-feat

    # Single-TS baseline (v1 models)
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py \
        --model ZT-Chiral --use-global-feat

    # Run all models
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_gnn.py --model all
"""

import argparse
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

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, PRED_DIR, RESULTS_DIR
from chiralaldol.data_io import prepare_Xy, load_splits, save_predictions
from chiralaldol.gnn.zt_models import ZT_MODELS

SINGLE_TS_PATH = FEAT_DIR / "zt_graphs" / "evans_zt_graphs.pkl"
MULTI_TS_PATH = FEAT_DIR / "zt_graphs" / "evans_multi_ts_graphs.pkl"
OUT_PRED_DIR = PRED_DIR / "zt_gnn"
TABLE_DIR = RESULTS_DIR / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zt_gnn")


# ═══════════════════════════ Data Loading ═══════════════════════════


def load_evans_data():
    """Load Evans labels + splits + 156d features."""
    from chiralaldol.config import VALID_AUXILIARIES
    meta = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    meta = meta[meta["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    X_feat, y_all, valid_mask, feat_names = prepare_Xy()
    splits = load_splits()

    evans_mask = (meta["auxiliary_type"] == "evans").values
    combined_mask = valid_mask & evans_mask

    return meta, y_all, combined_mask, splits, X_feat


def load_single_ts_graphs():
    """Load pre-computed single-TS graphs."""
    with open(SINGLE_TS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["graphs"], data["orig_indices"]


def load_multi_ts_graphs():
    """Load pre-computed multi-TS graph sets."""
    with open(MULTI_TS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["graph_sets"], data["orig_indices"]


# ═══════════════════════════ Single-TS Training ═══════════════════════════


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
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_labels.append(batch.y.view(-1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return balanced_accuracy_score(y_true, y_pred), y_pred, y_true, y_prob


def run_single_ts_split(model_cls, model_kwargs, graphs, orig_indices, y,
                        combined_mask, split_data, X_feat, device,
                        epochs=80, lr=1e-3, use_global_feat=False):
    """Train single-TS model on one split."""
    tr_raw = np.array(split_data["train"], dtype=int)
    te_raw = np.array(split_data["test"], dtype=int)
    tr_full = tr_raw[combined_mask[tr_raw]]
    te_full = te_raw[combined_mask[te_raw]]

    va = tr_full[-max(1, len(tr_full) // 10):]
    tr = tr_full[:-len(va)]
    if len(tr) < 10 or len(te_full) < 3:
        return None

    orig_to_zt = {orig: i for i, orig in enumerate(orig_indices)}

    def make_data_list(indices):
        from chiralaldol.gnn.zt_dataset import zt_graph_to_pyg
        data_list, valid_idx = [], []
        for idx in indices:
            if idx not in orig_to_zt:
                continue
            g = graphs[orig_to_zt[idx]]
            if g.status != "success":
                continue
            xf = X_feat[idx] if use_global_feat else None
            d = zt_graph_to_pyg(g, label=int(y[idx]), extra_features=xf)
            if d is not None:
                data_list.append(d)
                valid_idx.append(idx)
        return data_list, valid_idx

    tr_data, _ = make_data_list(tr)
    va_data, _ = make_data_list(va)
    te_data, te_idx = make_data_list(te_full)

    if len(tr_data) < 10 or len(te_data) < 3:
        return None

    return _train_loop(
        model_cls, model_kwargs, tr_data, va_data, te_data, te_idx,
        device, epochs, lr, loader_cls="standard",
    )


# ═══════════════════════════ Multi-TS Training ═══════════════════════════


def run_multi_ts_split(model_kwargs, ts_sets, orig_indices, y,
                       combined_mask, split_data, X_feat, device,
                       epochs=100, lr=1e-3, use_global_feat=False,
                       use_aux_loss=False, aux_loss_weight=0.1,
                       syn_anti_labels=None):
    """Train MultiTSAttentionScorer on one split."""
    from chiralaldol.gnn.zt_dataset import multi_ts_to_pyg_list, MultiTSDataLoader
    from chiralaldol.gnn.zt_models import MultiTSAttentionScorer

    tr_raw = np.array(split_data["train"], dtype=int)
    te_raw = np.array(split_data["test"], dtype=int)
    tr_full = tr_raw[combined_mask[tr_raw]]
    te_full = te_raw[combined_mask[te_raw]]

    va = tr_full[-max(1, len(tr_full) // 10):]
    tr = tr_full[:-len(va)]
    if len(tr) < 10 or len(te_full) < 3:
        return None

    orig_to_zt = {orig: i for i, orig in enumerate(orig_indices)}

    def make_rxn_data(indices):
        rxn_list, valid_idx = [], []
        for idx in indices:
            if idx not in orig_to_zt:
                continue
            zt_i = orig_to_zt[idx]
            ts_set = ts_sets[zt_i]
            if not any(g.status == "success" for g in ts_set.graphs):
                continue
            xf = X_feat[idx] if use_global_feat else None
            data_4 = multi_ts_to_pyg_list(ts_set, label=int(y[idx]), extra_features=xf)
            rxn_list.append(data_4)
            valid_idx.append(idx)
        return rxn_list, valid_idx

    tr_rxn, _ = make_rxn_data(tr)
    va_rxn, _ = make_rxn_data(va)
    te_rxn, te_idx = make_rxn_data(te_full)

    if len(tr_rxn) < 10 or len(te_rxn) < 3:
        return None

    # Create model
    model = MultiTSAttentionScorer(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights
    y_train = np.array([rxn[0].y.item() for rxn in tr_rxn])
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight = torch.zeros(4, device=device)
    for c, w in zip(classes, cw):
        weight[c] = w
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Syn/anti labels for aux loss
    sa_labels = None
    if use_aux_loss and syn_anti_labels is not None:
        sa_labels = syn_anti_labels

    tr_loader = MultiTSDataLoader(tr_rxn, batch_size=32, shuffle=True)
    va_loader = MultiTSDataLoader(va_rxn, batch_size=64, shuffle=False)
    te_loader = MultiTSDataLoader(te_rxn, batch_size=64, shuffle=False)

    best_va_acc, best_state = 0, None
    patience, no_improve = 20, 0

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            rxn_y = batch.y.view(-1)[::4]  # one label per reaction (4 TS per rxn)
            loss = criterion(logits, rxn_y)
            # Auxiliary loss: encourage syn TS attention for syn reactions
            if use_aux_loss and epoch >= 10:
                alpha, rxn_b, ts_t = model.get_ts_attention()
                warmup = min(1.0, (epoch - 10) / 10.0)
                aux_l = _compute_aux_loss(alpha, rxn_b, ts_t, rxn_y)
                if aux_l is not None:
                    loss = loss + aux_loss_weight * warmup * aux_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.n_reactions
            n += batch.n_reactions
        scheduler.step()

        # Validate
        model.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                logits = model(batch)
                va_preds.append(logits.argmax(dim=1).cpu().numpy())
                va_labels.append(batch.y.view(-1)[::4].cpu().numpy())
        va_pred = np.concatenate(va_preds)
        va_true = np.concatenate(va_labels)
        va_acc = balanced_accuracy_score(va_true, va_pred)

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

    # Test
    model.eval()
    te_preds, te_labels, te_probs = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            te_preds.append(probs.argmax(dim=1).cpu().numpy())
            te_labels.append(batch.y.view(-1)[::4].cpu().numpy())
            te_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(te_preds)
    y_true = np.concatenate(te_labels)
    y_prob = np.concatenate(te_probs)
    te_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "bal_acc": round(te_acc, 4),
        "n_train": len(tr_rxn),
        "n_test": len(te_rxn),
        "y_pred": y_pred,
        "y_true": y_true,
        "y_prob": y_prob,
        "test_idx": np.array(te_idx),
        "best_va_acc": round(best_va_acc, 4),
    }


def _compute_aux_loss(alpha, rxn_batch, ts_types, y_rxn):
    """Auxiliary loss: encourage syn-TS attention for syn-label reactions.

    syn labels: y ∈ {0, 3} (RR, SS) → syn TS should have higher attention
    anti labels: y ∈ {1, 2} (RS, SR) → anti TS should have higher attention
    """
    n_rxn = y_rxn.size(0)
    losses = []

    for r in range(n_rxn):
        mask = (rxn_batch == r)
        a = alpha[mask]
        tt = ts_types[mask]
        if len(a) != 4:
            continue

        # syn TSs = 0 (Z-syn), 2 (E-syn); anti TSs = 1 (Z-anti), 3 (E-anti)
        syn_mask = (tt == 0) | (tt == 2)
        alpha_syn = a[syn_mask].sum()

        # Target: syn label → high syn attention, anti label → low syn attention
        label = y_rxn[r].item()
        target = 1.0 if label in (0, 3) else 0.0  # syn products → syn TS
        target_t = torch.tensor(target, device=alpha.device)
        losses.append(F.binary_cross_entropy(alpha_syn.clamp(1e-7, 1 - 1e-7), target_t))

    if not losses:
        return None
    return torch.stack(losses).mean()


# ═══════════════════════════ Shared Training Loop ═══════════════════════════


def _train_loop(model_cls, model_kwargs, tr_data, va_data, te_data, te_idx,
                device, epochs, lr, loader_cls="standard"):
    """Standard training loop for single-TS models."""
    tr_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=64)
    te_loader = DataLoader(te_data, batch_size=64)

    y_train = np.array([d.y.item() for d in tr_data])
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight = torch.zeros(4, device=device)
    for c, w in zip(classes, cw):
        weight[c] = w
    criterion = nn.CrossEntropyLoss(weight=weight)

    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_va_acc, best_state = 0, None
    patience, no_improve = 15, 0

    for epoch in range(epochs):
        train_one_epoch(model, tr_loader, optimizer, criterion, device)
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
        "y_pred": y_pred,
        "y_true": y_true,
        "y_prob": y_prob,
        "test_idx": np.array(te_idx),
        "best_va_acc": round(best_va_acc, 4),
    }


# ═══════════════════════════ Main ═══════════════════════════


SINGLE_TS_MODELS = [k for k in ZT_MODELS if k != "MultiTS"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MultiTS",
                        choices=list(ZT_MODELS.keys()) + ["all", "all-single"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--splits", default="tscv", choices=["tscv", "grouped", "all"])
    parser.add_argument("--use-global-feat", action="store_true")
    parser.add_argument("--no-global-feat", action="store_true")
    parser.add_argument("--use-ze-prior", action="store_true", default=False)
    parser.add_argument("--no-ze-prior", dest="use_ze_prior", action="store_false")
    parser.add_argument("--use-pairwise", action="store_true", default=True)
    parser.add_argument("--no-pairwise", dest="use_pairwise", action="store_false")
    parser.add_argument("--use-aux-loss", action="store_true")
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_global = args.use_global_feat and not args.no_global_feat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"ZT-GNN Benchmark (model={args.model}, epochs={args.epochs}, device={device})")
    logger.info(f"  global_feat={use_global}, ze_prior={args.use_ze_prior}, aux_loss={args.use_aux_loss}")
    logger.info("=" * 60)

    meta, y_all, combined_mask, splits, X_feat = load_evans_data()
    logger.info(f"Evans: {combined_mask.sum()} valid reactions")

    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Filter splits
    if args.splits == "tscv":
        splits = {k: v for k, v in splits.items() if "tscv" in k}
    elif args.splits == "grouped":
        splits = {k: v for k, v in splits.items() if "grouped" in k}

    # Determine models to run
    if args.model == "all":
        model_names = list(ZT_MODELS.keys())
    elif args.model == "all-single":
        model_names = SINGLE_TS_MODELS
    else:
        model_names = [args.model]

    global_feat_dim = X_feat.shape[1] if use_global else 0

    # Load graphs (lazily)
    single_ts_data = None
    multi_ts_data = None

    all_results = []

    for model_name in model_names:
        logger.info(f"\n{'='*40}\n  Model: {model_name}\n{'='*40}")

        for split_name, split_data in sorted(splits.items()):
            torch.manual_seed(args.seed)

            if model_name == "MultiTS":
                if multi_ts_data is None:
                    multi_ts_data = load_multi_ts_graphs()
                ts_sets, orig_indices = multi_ts_data

                result = run_multi_ts_split(
                    model_kwargs={
                        "node_dim": 28, "edge_dim": 8,
                        "hidden_dim": args.hidden, "n_layers": args.layers,
                        "n_classes": 4, "dropout": 0.3,
                        "global_feat_dim": global_feat_dim,
                        "use_ze_prior": args.use_ze_prior,
                        "use_pairwise": args.use_pairwise,
                    },
                    ts_sets=ts_sets, orig_indices=orig_indices,
                    y=y_all, combined_mask=combined_mask,
                    split_data=split_data, X_feat=X_feat, device=device,
                    epochs=args.epochs, lr=args.lr,
                    use_global_feat=use_global,
                    use_aux_loss=args.use_aux_loss,
                    aux_loss_weight=args.aux_loss_weight,
                )
            else:
                if single_ts_data is None:
                    single_ts_data = load_single_ts_graphs()
                graphs, orig_indices = single_ts_data

                # Model kwargs
                if model_name == "ZT-GCPNet":
                    mkw = {"node_dim": 20, "edge_dim": 5,
                           "hidden_scalar": 64, "hidden_vector": 16,
                           "n_layers": args.layers, "n_classes": 4,
                           "dropout": 0.2, "global_feat_dim": global_feat_dim}
                elif model_name == "ZT-ChiDeK":
                    mkw = {"node_dim": 20, "edge_dim": 5, "hidden_dim": 128,
                           "n_backbone_layers": 3, "n_chiral_layers": 2,
                           "n_classes": 4, "dropout": 0.2,
                           "global_feat_dim": global_feat_dim}
                elif model_name in ("ZT-ComENet", "ZT-Hybrid"):
                    mkw = {"node_dim": 20, "edge_dim": 5, "hidden_dim": 128,
                           "n_layers": args.layers, "n_classes": 4,
                           "dropout": 0.2, "global_feat_dim": global_feat_dim,
                           "n_rbf": 16}
                    if model_name == "ZT-Hybrid":
                        mkw["spms_dim"] = 16
                else:
                    mkw = {"node_dim": 20, "edge_dim": 5, "hidden_dim": 128,
                           "n_layers": args.layers, "n_classes": 4,
                           "dropout": 0.2, "global_feat_dim": global_feat_dim}
                    if model_name == "ZT-GAT":
                        mkw["n_heads"] = 4

                result = run_single_ts_split(
                    ZT_MODELS[model_name], mkw,
                    graphs, orig_indices, y_all, combined_mask,
                    split_data, X_feat, device,
                    epochs=args.epochs, lr=args.lr,
                    use_global_feat=use_global,
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

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLE_DIR / "benchmark_zt_gnn_evans.csv", index=False)

    print("\n" + "=" * 70)
    print("ZT-GNN EVANS-ONLY BENCHMARK")
    print("=" * 70)
    for mn in model_names:
        mr = [r for r in all_results if r["model"] == mn]
        if not mr:
            continue
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        print(f"\n  {mn}:")
        if tscv:
            print(f"    TSCV:    {np.mean(tscv):.4f} ± {np.std(tscv):.4f}")
        if grouped:
            print(f"    Grouped: {np.mean(grouped):.4f} ± {np.std(grouped):.4f}")

    print(f"\n  --- Baselines ---")
    print(f"    Tree (Evans ET):     TSCV=0.710")
    print(f"    ZT-Chiral+feat (v1): TSCV=0.818")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
