#!/usr/bin/env python3
"""Run remaining SPMS benchmark experiments:
  1. ZT-GNN (ZT-GIN) + SPMS as global features
  2. ZT-GNN (ZT-GIN) + Face Map as global features
  3. Chemprop + SPMS features (baseline)
  4. Chemprop + SPMS stats
  5. Chemprop + Face Map
  6. Comprehensive comparison + per-auxiliary analysis

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_spms_remaining.py
"""

import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import (
    CLEAN_DIR, FEAT_DIR, PRED_DIR, RESULTS_DIR, SPMS_DIR,
    VALID_AUXILIARIES,
)
from chiralaldol.data_io import load_splits, prepare_Xy, save_predictions
from chiralaldol.gnn.zt_dataset import zt_graph_to_pyg
from chiralaldol.gnn.zt_models import ZTGIN
from chiralaldol.spms_compressor import extract_spms_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spms_remaining")

ZT_GRAPHS_PATH = FEAT_DIR / "zt_graphs" / "evans_zt_graphs.pkl"
OUT_DIR = PRED_DIR / "spms"
TABLE_DIR = RESULTS_DIR / "tables"
N_CLASSES = 4


# ════════════════════ ZT-GNN Experiments ════════════════════

def load_evans_data_with_spms(feat_type="spms"):
    """Load Evans ZT graphs + augmented global features."""
    meta_full = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    meta = meta_full[meta_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    _, y_all, valid_mask, _ = prepare_Xy()

    # Load base features
    X_base = pd.read_csv(FEAT_DIR / "v4_features.csv").values.astype(np.float32)
    np.nan_to_num(X_base, copy=False)

    # Augment with SPMS or Face Map
    if feat_type == "spms":
        spms_arrays = np.load(SPMS_DIR / "spms_arrays.npy")
        X_extra, _ = extract_spms_stats(spms_arrays)
    elif feat_type == "face_map":
        face_df = pd.read_csv(SPMS_DIR / "face_map_features.csv")
        X_extra = face_df.values.astype(np.float32)
        np.nan_to_num(X_extra, copy=False)
    else:
        X_extra = np.zeros((len(X_base), 0), dtype=np.float32)

    X_augmented = np.hstack([X_base, X_extra])

    splits = load_splits()
    evans_mask = (meta["auxiliary_type"] == "evans").values
    combined_mask = valid_mask & evans_mask

    with open(ZT_GRAPHS_PATH, "rb") as f:
        zt_data = pickle.load(f)
    graphs = zt_data["graphs"]
    orig_indices = zt_data["orig_indices"]
    y = np.where(combined_mask, y_all, -1).astype(int)

    return graphs, orig_indices, y, combined_mask, splits, X_augmented


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
        preds.append(p.argmax(dim=1).cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())
        probs.append(p.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    return balanced_accuracy_score(y_true, y_pred), y_pred, y_true, y_prob


def run_zt_gnn_experiment(feat_type, device, epochs=80):
    """Run ZT-GIN with augmented global features."""
    logger.info(f"\n{'='*50}\n  ZT-GIN + {feat_type} global features\n{'='*50}")

    graphs, orig_indices, y, combined_mask, splits, X_feat = \
        load_evans_data_with_spms(feat_type)

    orig_to_zt = {orig: i for i, orig in enumerate(orig_indices)}
    results = []

    for split_name, split_data in sorted(splits.items()):
        tr_raw = np.array(split_data["train"], dtype=int)
        tr_full = tr_raw[combined_mask[tr_raw]]
        te_raw = np.array(split_data["test"], dtype=int)
        te_full = te_raw[combined_mask[te_raw]]

        va = tr_full[-max(1, len(tr_full) // 10):]
        tr = tr_full[:-len(va)]

        if len(tr) < 10 or len(te_full) < 3:
            continue

        def make_data_list(indices):
            data_list, valid_idx = [], []
            for idx in indices:
                if idx in orig_to_zt:
                    g = graphs[orig_to_zt[idx]]
                    if g.status != "success":
                        continue
                    d = zt_graph_to_pyg(g, label=int(y[idx]),
                                        extra_features=X_feat[idx])
                    if d is not None:
                        data_list.append(d)
                        valid_idx.append(idx)
            return data_list, valid_idx

        tr_data, _ = make_data_list(tr)
        va_data, _ = make_data_list(va)
        te_data, te_idx = make_data_list(te_full)

        if len(tr_data) < 10 or len(te_data) < 3:
            continue

        tr_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_data, batch_size=64)
        te_loader = DataLoader(te_data, batch_size=64)

        # Class weights
        y_train = np.array([d.y.item() for d in tr_data])
        classes = np.unique(y_train)
        if len(classes) < 2:
            continue
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        weight = torch.zeros(4, device=device)
        for c, w in zip(classes, cw):
            weight[c] = w
        criterion = nn.CrossEntropyLoss(weight=weight)

        model = ZTGIN(node_dim=20, edge_dim=5, hidden_dim=128, n_layers=4,
                      n_classes=4, dropout=0.2,
                      global_feat_dim=X_feat.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_va, best_state, no_improve = 0, None, 0
        for epoch in range(epochs):
            train_one_epoch(model, tr_loader, optimizer, criterion, device)
            va_acc, _, _, _ = evaluate(model, va_loader, device)
            scheduler.step()
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 15:
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        te_acc, y_pred, y_true, y_prob = evaluate(model, te_loader, device)

        save_predictions(OUT_DIR / f"ztgin_{feat_type}_{split_name}.csv",
                        np.array(te_idx), y_true, y_pred, y_prob)

        results.append({
            "model": f"ZT-GIN+{feat_type}", "split": split_name,
            "bal_acc": round(te_acc, 4), "n_train": len(tr_data), "n_test": len(te_data),
        })
        logger.info(f"  {split_name}: {te_acc:.4f} (va={best_va:.4f})")

    return results


# ════════════════════ Chemprop Experiments ════════════════════

def run_chemprop_experiment(feat_type, device):
    """Run Chemprop v2 (Lightning-based) with augmented features.

    Reuses the same API as scripts/run_chemprop.py: MoleculeDatapoint(mol=, y=, x_d=).
    """
    logger.info(f"\n{'='*50}\n  Chemprop + {feat_type}\n{'='*50}")

    try:
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, MulticlassClassificationFFN, NormAggregation
        import lightning as L
    except ImportError as e:
        logger.warning(f"chemprop/lightning not available: {e}")
        return []

    from rdkit import Chem

    meta_full = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    meta = meta_full[meta_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    _, y_all, valid_mask, _ = prepare_Xy()
    y = np.where(valid_mask, y_all, -1).astype(int)
    splits = load_splits()

    # Load features
    X_base = pd.read_csv(FEAT_DIR / "v4_features.csv").values.astype(np.float32)
    np.nan_to_num(X_base, copy=False)

    if feat_type == "baseline":
        X_feat = X_base
    elif feat_type == "spms":
        spms_arrays = np.load(SPMS_DIR / "spms_arrays.npy")
        stats, _ = extract_spms_stats(spms_arrays)
        X_feat = np.hstack([X_base, stats])
    elif feat_type == "face_map":
        face_df = pd.read_csv(SPMS_DIR / "face_map_features.csv")
        X_extra = face_df.values.astype(np.float32)
        np.nan_to_num(X_extra, copy=False)
        X_feat = np.hstack([X_base, X_extra])
    else:
        X_feat = X_base

    def build_dps(indices):
        dps, valid_idx = [], []
        for i in indices:
            ket = str(meta.iloc[i].get("canonical_ketone_smiles", ""))
            ald = str(meta.iloc[i].get("canonical_aldehyde_smiles", ""))
            if not ket or ket == "nan":
                continue
            combined = f"{ket}.{ald}" if ald and ald != "nan" else ket
            mol = Chem.MolFromSmiles(combined)
            if mol is None:
                mol = Chem.MolFromSmiles(ket)
                if mol is None:
                    continue
            dp = MoleculeDatapoint(mol=mol, y=np.array([y[i]]), x_d=X_feat[i])
            dps.append(dp)
            valid_idx.append(i)
        return dps, np.array(valid_idx)

    results = []
    grouped_splits = {k: v for k, v in splits.items() if "grouped" in k}

    for split_name, split_data in sorted(grouped_splits.items()):
        tr_raw = np.array(split_data["train"], dtype=int)
        tr = tr_raw[valid_mask[tr_raw]]
        te_raw = np.array(split_data["test"], dtype=int)
        te = te_raw[valid_mask[te_raw]]
        va = tr[-max(1, len(tr) // 10):]
        tr = tr[:-len(va)]

        if len(tr) < 10 or len(te) < 3:
            continue

        try:
            tr_dps, _ = build_dps(tr)
            va_dps, _ = build_dps(va)
            te_dps, te_vi = build_dps(te)

            if len(tr_dps) < 10 or len(te_dps) < 3:
                continue

            tr_dl = build_dataloader(MoleculeDataset(tr_dps), batch_size=32, shuffle=True, num_workers=0)
            va_dl = build_dataloader(MoleculeDataset(va_dps), batch_size=64, shuffle=False, num_workers=0)
            te_dl = build_dataloader(MoleculeDataset(te_dps), batch_size=64, shuffle=False, num_workers=0)

            d_xd = X_feat.shape[1]
            mp = BondMessagePassing(d_v=72, d_e=14, depth=3, d_h=300, dropout=0.1)
            agg = NormAggregation()
            ffn = MulticlassClassificationFFN(
                n_classes=4, input_dim=300 + d_xd, hidden_dim=300, n_layers=2, dropout=0.1)

            model = MPNN(
                message_passing=mp, agg=agg, predictor=ffn,
                batch_norm=True, warmup_epochs=2,
                init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
            )

            trainer = L.Trainer(
                max_epochs=30, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1, enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
            )
            trainer.fit(model, tr_dl, va_dl)

            preds = trainer.predict(model, te_dl)
            all_logits = torch.cat(preds, dim=0).squeeze(1)
            y_prob = torch.softmax(all_logits, dim=1).numpy()
            y_pred = y_prob.argmax(axis=1)
            y_true = y[te_vi[:len(y_pred)]]

            te_acc = balanced_accuracy_score(y_true, y_pred)
            save_predictions(OUT_DIR / f"chemprop_{feat_type}_{split_name}.csv",
                            te_vi[:len(y_pred)], y_true, y_pred, y_prob)

            results.append({
                "model": f"Chemprop+{feat_type}", "split": split_name,
                "bal_acc": round(te_acc, 4),
                "n_train": len(tr_dps), "n_test": len(te_dps),
            })
            logger.info(f"  {split_name}: {te_acc:.4f}")

        except Exception as e:
            logger.warning(f"  {split_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


# ════════════════════ Comprehensive Summary ════════════════════

def generate_summary(all_results):
    """Generate comprehensive comparison table."""
    df = pd.DataFrame(all_results)
    if df.empty:
        logger.warning("No results to summarize")
        return df

    print("\n" + "=" * 90)
    print("COMPREHENSIVE SPMS BENCHMARK — ALL METHODS × ALL MODELS")
    print("=" * 90)

    models = sorted(df["model"].unique())
    print(f"\n  {'Model':<30s} {'TSCV':>12s}  {'Grouped':>12s}  {'Scaffold':>10s}")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}  {'-'*10}")

    for model in models:
        sub = df[df["model"] == model]
        tscv = sub[sub["split"].str.contains("tscv")]["bal_acc"]
        grouped = sub[sub["split"].str.contains("grouped")]["bal_acc"]
        scaffold = sub[sub["split"].str.contains("scaffold")]["bal_acc"]

        tscv_str = f"{tscv.mean():.4f}±{tscv.std():.3f}" if len(tscv) > 0 else "—"
        grp_str = f"{grouped.mean():.4f}±{grouped.std():.3f}" if len(grouped) > 0 else "—"
        scf_str = f"{scaffold.mean():.4f}" if len(scaffold) > 0 else "—"
        print(f"  {model:<30s} {tscv_str:>12s}  {grp_str:>12s}  {scf_str:>10s}")

    return df


def per_auxiliary_analysis(all_results):
    """Per-auxiliary breakdown for key models."""
    pred_dir = OUT_DIR
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    clean = clean[clean["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)

    print("\n" + "=" * 90)
    print("PER-AUXILIARY ANALYSIS (Grouped splits)")
    print("=" * 90)

    # Check available prediction files
    for prefix in ["baseline_et", "spms_stats_et", "face_map_et", "spms_face_et",
                    "ztgin_spms", "ztgin_face_map",
                    "chemprop_baseline", "chemprop_spms", "chemprop_face_map"]:
        files = sorted(pred_dir.glob(f"{prefix}_grouped_*.csv"))
        if not files:
            continue

        print(f"\n  {prefix}:")
        for aux in VALID_AUXILIARIES:
            aux_idx = set(clean[clean["auxiliary_type"] == aux].index.tolist())
            accs = []
            for f in files:
                df = pd.read_csv(f)
                mask = df["idx"].isin(aux_idx)
                if mask.sum() >= 3:
                    accs.append(balanced_accuracy_score(df[mask]["y_true"], df[mask]["y_pred"]))
            if accs:
                print(f"    {aux:25s}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ─── ZT-GNN experiments (already completed, load from predictions) ───
    for feat_type in ["spms", "face_map"]:
        pred_files = sorted(OUT_DIR.glob(f"ztgin_{feat_type}_*.csv"))
        if pred_files:
            logger.info(f"Loading existing ZT-GIN+{feat_type} results ({len(pred_files)} files)")
            for f in pred_files:
                df = pd.read_csv(f)
                split_name = f.stem.replace(f"ztgin_{feat_type}_", "")
                ba = balanced_accuracy_score(df["y_true"], df["y_pred"])
                all_results.append({
                    "model": f"ZT-GIN+{feat_type}", "split": split_name,
                    "bal_acc": round(ba, 4), "n_train": 0, "n_test": len(df),
                })
        else:
            results = run_zt_gnn_experiment(feat_type, device, epochs=80)
            all_results.extend(results)

    # ─── Chemprop experiments ───
    for feat_type in ["baseline", "spms", "face_map"]:
        results = run_chemprop_experiment(feat_type, device)
        all_results.extend(results)

    # ─── Load existing tree results for comparison ───
    existing = TABLE_DIR / "benchmark_spms.csv"
    if existing.exists():
        tree_df = pd.read_csv(existing)
        for _, row in tree_df.iterrows():
            all_results.append({
                "model": f"{row['model']}+{row['features']}",
                "split": row["split"],
                "bal_acc": row["bal_acc"],
                "n_train": row.get("n_train", 0),
                "n_test": row.get("n_test", 0),
            })

    # ─── Save + summarize ───
    results_df = generate_summary(all_results)
    results_df.to_csv(TABLE_DIR / "benchmark_spms_full.csv", index=False)

    per_auxiliary_analysis(all_results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
