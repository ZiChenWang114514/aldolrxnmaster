#!/usr/bin/env python3
"""Phase B1: Chemprop MPNN + ZT-TS Features on Evans-only subset.

Tests whether Zimmerman-Traxler transition state features improve MPNN
prediction on Evans aldol reactions.

Three modes:
  1. smiles_only: Chemprop MPNN on ketone.aldehyde SMILES
  2. smiles_153d: Chemprop + 153d handcrafted features (existing baseline)
  3. smiles_zt: Chemprop + 32d ZT-TS features
  4. smiles_153d_zt: Chemprop + 153d + 32d ZT features (185d total)

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_chemprop.py
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_zt_chemprop.py --epochs 80
"""

import argparse
import logging
import os
import pickle
import time
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from sklearn.metrics import balanced_accuracy_score

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, PRED_DIR, RESULTS_DIR
from chiralaldol.data_io import prepare_Xy, load_splits, save_predictions
from chiralaldol.zt_features import extract_zt_features_batch, ZT_FEATURE_DIM

CLEAN_CSV = CLEAN_DIR / "substrate_aldol_clean.csv"
ZT_GRAPHS_PATH = FEAT_DIR / "zt_graphs" / "evans_zt_graphs.pkl"
OUT_PRED_DIR = PRED_DIR / "zt_chemprop"
TABLE_DIR = RESULTS_DIR / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zt_chemprop")


def load_data():
    """Load Evans-only data with ZT features."""
    from chiralaldol.config import VALID_AUXILIARIES
    # Filter to VALID_AUXILIARIES (2434 → 2427) to align with v5_features.csv and splits
    meta_full = pd.read_csv(CLEAN_CSV)
    meta = meta_full[meta_full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)

    X_153d, y, valid_mask, feat_names = prepare_Xy()
    splits = load_splits()

    # Evans mask in 2427-space
    evans_mask = (meta["auxiliary_type"] == "evans").values
    combined_mask = valid_mask & evans_mask

    # Load ZT features (orig_indices are in 2427-space, set by run_build_zt_graphs.py)
    with open(ZT_GRAPHS_PATH, "rb") as f:
        zt_data = pickle.load(f)
    zt_graphs = zt_data["graphs"]
    zt_orig_idx = zt_data["orig_indices"]
    X_zt_evans = extract_zt_features_batch(zt_graphs)

    # Align ZT features to 2427-space dataset
    X_zt = np.zeros((len(meta), ZT_FEATURE_DIM), dtype=np.float32)
    for i, orig_i in enumerate(zt_orig_idx):
        X_zt[orig_i] = X_zt_evans[i]

    # Combined features (both 2427 rows)
    X_185d = np.hstack([X_153d, X_zt])  # 2427 × 185 ✓

    logger.info(f"Data: {len(meta)} total, Evans={evans_mask.sum()}, "
                f"valid Evans={combined_mask.sum()}")
    logger.info(f"Features: 153d base + {ZT_FEATURE_DIM}d ZT = {X_185d.shape[1]}d")

    return meta, X_153d, X_zt, X_185d, y, combined_mask, splits


def build_datapoints(meta, y, indices, X_features=None):
    """Build Chemprop MoleculeDatapoints."""
    from chemprop.data import MoleculeDatapoint

    datapoints = []
    valid_indices = []

    for i in indices:
        ket_smi = str(meta.iloc[i].get("canonical_ketone_smiles", ""))
        ald_smi = str(meta.iloc[i].get("canonical_aldehyde_smiles", ""))

        if not ket_smi or ket_smi == "nan" or not ald_smi or ald_smi == "nan":
            continue

        combined_smi = f"{ket_smi}.{ald_smi}"
        mol = Chem.MolFromSmiles(combined_smi)
        if mol is None:
            mol = Chem.MolFromSmiles(ket_smi)
            if mol is None:
                continue

        x_d = X_features[i] if X_features is not None else None
        dp = MoleculeDatapoint(mol=mol, y=np.array([y[i]]), x_d=x_d)
        datapoints.append(dp)
        valid_indices.append(i)

    return datapoints, np.array(valid_indices)


def train_and_evaluate(meta, y, combined_mask, X_features, split_name, split_data,
                       mode_label="", epochs=50):
    """Train Chemprop on one split, return metrics."""
    from chemprop.data import MoleculeDataset, build_dataloader
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, MulticlassClassificationFFN, NormAggregation

    tr_raw = np.array(split_data["train"], dtype=int)
    tr = tr_raw[combined_mask[tr_raw]]
    te_raw = np.array(split_data["test"], dtype=int)
    te = te_raw[combined_mask[te_raw]]

    va = tr[-max(1, len(tr) // 10):]
    tr = tr[:-len(va)]

    if len(tr) < 10 or len(te) < 3:
        return None

    tr_dps, tr_valid = build_datapoints(meta, y, tr, X_features)
    va_dps, va_valid = build_datapoints(meta, y, va, X_features)
    te_dps, te_valid = build_datapoints(meta, y, te, X_features)

    if len(tr_dps) < 10 or len(te_dps) < 3:
        return None

    tr_ds = MoleculeDataset(tr_dps)
    va_ds = MoleculeDataset(va_dps)
    te_ds = MoleculeDataset(te_dps)

    tr_dl = build_dataloader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    va_dl = build_dataloader(va_ds, batch_size=64, shuffle=False, num_workers=0)
    te_dl = build_dataloader(te_ds, batch_size=64, shuffle=False, num_workers=0)

    d_xd = X_features.shape[1] if X_features is not None else 0
    mp = BondMessagePassing(d_v=72, d_e=14, depth=3, d_h=300, dropout=0.1)
    agg = NormAggregation()
    ffn = MulticlassClassificationFFN(
        n_classes=4, input_dim=300 + d_xd, hidden_dim=300, n_layers=2, dropout=0.1
    )

    model = MPNN(
        message_passing=mp, agg=agg, predictor=ffn, batch_norm=True,
        warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, tr_dl, va_dl)

    preds = trainer.predict(model, te_dl)
    all_logits = torch.cat(preds, dim=0).squeeze(1)
    y_prob = torch.softmax(all_logits, dim=1).numpy()
    y_pred = y_prob.argmax(axis=1)
    y_test = y[te_valid[:len(y_pred)]]

    bal_acc = balanced_accuracy_score(y_test, y_pred)

    return {
        "split": split_name,
        "mode": mode_label,
        "bal_acc": round(bal_acc, 4),
        "n_train": len(tr_dps),
        "n_test": len(te_dps),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "test_idx": te_valid[:len(y_pred)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--splits", default="tscv",
                        choices=["tscv", "grouped", "all"],
                        help="Which splits to evaluate")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"ZT-Chemprop: Evans-only ({args.epochs} epochs)")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    meta, X_153d, X_zt, X_185d, y, combined_mask, splits = load_data()

    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Filter splits
    if args.splits == "tscv":
        splits = {k: v for k, v in splits.items() if "tscv" in k}
    elif args.splits == "grouped":
        splits = {k: v for k, v in splits.items() if "grouped" in k}

    # 4 modes to compare
    modes = {
        "SMILES-only": None,
        "SMILES+153d": X_153d,
        "SMILES+ZT32d": X_zt,
        "SMILES+153d+ZT32d": X_185d,
    }

    all_results = []

    for mode_label, X_feat in modes.items():
        logger.info(f"\n{'='*40}\n  Mode: {mode_label} (d_xd={X_feat.shape[1] if X_feat is not None else 0})\n{'='*40}")

        for split_name, split_data in sorted(splits.items()):
            result = train_and_evaluate(
                meta, y, combined_mask, X_feat, split_name, split_data,
                mode_label=mode_label, epochs=args.epochs,
            )
            if result is None:
                continue

            logger.info(f"  {split_name}: {result['bal_acc']:.4f}")

            fname = f"zt_chemprop_{mode_label.replace('+', '_')}_{split_name}.csv"
            save_predictions(OUT_PRED_DIR / fname,
                            result["test_idx"], result["y_test"],
                            result["y_pred"], result["y_prob"])

            all_results.append({
                "model": mode_label,
                "split": split_name,
                "bal_acc": result["bal_acc"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            })

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLE_DIR / "benchmark_zt_chemprop_evans.csv", index=False)

    print("\n" + "=" * 70)
    print("ZT-CHEMPROP EVANS-ONLY BENCHMARK")
    print("=" * 70)

    for mode_label in modes:
        mr = [r for r in all_results if r["model"] == mode_label]
        if not mr:
            continue
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        print(f"\n  {mode_label}:")
        if tscv:
            print(f"    TSCV:     {np.mean(tscv):.4f} ± {np.std(tscv):.4f}")
        if scaffold:
            print(f"    Scaffold: {scaffold[0]:.4f}")
        if grouped:
            print(f"    Grouped:  {np.mean(grouped):.4f} ± {np.std(grouped):.4f}")

    print(f"\n  --- Tree Baseline (Evans-only ET) ---")
    print(f"    TSCV: 0.710")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
