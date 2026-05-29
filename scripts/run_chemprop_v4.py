#!/usr/bin/env python3
"""Chemprop v2 MPNN baseline on V4d data — fair comparison with tree models.

Uses ketone+aldehyde SMILES as multi-component input (no product leakage).
Runs on all V4d splits (TSCV + scaffold + grouped) for direct comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_chemprop_v4.py
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_chemprop_v4.py --epochs 80
    CUDA_VISIBLE_DEVICES=0 conda run -n aldol-rxn python scripts/run_chemprop_v4.py --mode features_only
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L
from rdkit import Chem, RDLogger
from sklearn.metrics import balanced_accuracy_score

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT = Path(__file__).resolve().parent.parent
FEAT_DIR = PROJECT / "data" / "features_v4"
SPLITS_DIR = PROJECT / "data" / "splits_v4"
CLEAN_CSV = PROJECT / "data" / "clean_v4" / "substrate_aldol_clean.csv"
PRED_DIR = PROJECT / "results" / "predictions_v4" / "chemprop"
TABLE_DIR = PROJECT / "results" / "tables"

TARGET_LABEL = "label_joint"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chemprop_v4")


def load_data():
    """Load V4d clean data, features, labels, and splits."""
    meta = pd.read_csv(CLEAN_CSV)
    X_df = pd.read_csv(FEAT_DIR / "v4_features.csv")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")

    valid_mask = labels[TARGET_LABEL].notna().values
    y = np.where(valid_mask, labels[TARGET_LABEL].values, -1).astype(int)
    X_features = X_df.values.astype(np.float32)
    np.nan_to_num(X_features, copy=False)

    splits = {}
    for f in sorted(SPLITS_DIR.glob("*.json")):
        with open(f) as fp:
            splits[f.stem] = json.load(fp)

    return meta, X_features, y, valid_mask, splits


def build_datapoints(meta, y, indices, X_features=None):
    """Build Chemprop MoleculeDatapoints from ketone+aldehyde SMILES."""
    from chemprop.data import MoleculeDatapoint

    datapoints = []
    valid_indices = []

    for i in indices:
        ket_smi = str(meta.iloc[i].get("canonical_ketone_smiles", ""))
        ald_smi = str(meta.iloc[i].get("canonical_aldehyde_smiles", ""))

        if not ket_smi or ket_smi == "nan" or not ald_smi or ald_smi == "nan":
            continue

        # Multi-component SMILES: ketone.aldehyde
        combined_smi = f"{ket_smi}.{ald_smi}"
        mol = Chem.MolFromSmiles(combined_smi)
        if mol is None:
            # Fallback: try ketone only
            mol = Chem.MolFromSmiles(ket_smi)
            if mol is None:
                continue

        x_d = X_features[i] if X_features is not None else None
        dp = MoleculeDatapoint(mol=mol, y=np.array([y[i]]), x_d=x_d)
        datapoints.append(dp)
        valid_indices.append(i)

    return datapoints, np.array(valid_indices)


def train_and_evaluate(meta, y, valid_mask, X_features, split_name, split_data,
                       mode="smiles_only", epochs=50):
    """Train Chemprop on one split, return metrics."""
    from chemprop.data import MoleculeDataset, build_dataloader
    from chemprop.nn import BondMessagePassing, MulticlassClassificationFFN, NormAggregation
    from chemprop.models import MPNN

    tr_raw = np.array(split_data["train"], dtype=int)
    tr = tr_raw[valid_mask[tr_raw]]
    te_raw = np.array(split_data["test"], dtype=int)
    te = te_raw[valid_mask[te_raw]]

    # Use last 10% of train as val
    va = tr[-max(1, len(tr) // 10):]
    tr = tr[:-len(va)]

    if len(tr) < 10 or len(te) < 3:
        return None

    # Build datapoints
    use_features = X_features if mode == "smiles_features" else None

    tr_dps, tr_valid = build_datapoints(meta, y, tr, use_features)
    va_dps, va_valid = build_datapoints(meta, y, va, use_features)
    te_dps, te_valid = build_datapoints(meta, y, te, use_features)

    if len(tr_dps) < 10 or len(te_dps) < 3:
        return None

    # Datasets and dataloaders
    tr_ds = MoleculeDataset(tr_dps)
    va_ds = MoleculeDataset(va_dps)
    te_ds = MoleculeDataset(te_dps)

    tr_dl = build_dataloader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    va_dl = build_dataloader(va_ds, batch_size=64, shuffle=False, num_workers=0)
    te_dl = build_dataloader(te_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model
    d_xd = X_features.shape[1] if use_features is not None else 0
    mp = BondMessagePassing(d_v=72, d_e=14, depth=3, d_h=300, dropout=0.1)
    agg = NormAggregation()
    ffn = MulticlassClassificationFFN(
        n_classes=4, input_dim=300 + d_xd, hidden_dim=300, n_layers=2, dropout=0.1
    )

    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=2,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )

    # Train
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

    # Predict
    preds = trainer.predict(model, te_dl)
    all_logits = torch.cat(preds, dim=0).squeeze(1)  # (N, 4)
    y_prob = torch.softmax(all_logits, dim=1).numpy()
    y_pred = y_prob.argmax(axis=1)
    y_test = y[te_valid[:len(y_pred)]]

    bal_acc = balanced_accuracy_score(y_test, y_pred)

    return {
        "split": split_name,
        "mode": mode,
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
    parser.add_argument("--mode", choices=["smiles_only", "smiles_features", "both"], default="both")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"Chemprop v2 MPNN Baseline (epochs={args.epochs})")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    meta, X_features, y, valid_mask, splits = load_data()
    logger.info(f"Data: {len(meta)} rows, {X_features.shape[1]}d features, {len(splits)} splits")

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    modes = []
    if args.mode in ("smiles_only", "both"):
        modes.append("smiles_only")
    if args.mode in ("smiles_features", "both"):
        modes.append("smiles_features")

    all_results = []

    for mode in modes:
        mode_label = "Chemprop" if mode == "smiles_only" else "Chemprop+Features"
        logger.info(f"\n{'='*40}\n  Mode: {mode_label}\n{'='*40}")

        for split_name, split_data in sorted(splits.items()):
            logger.info(f"\n  --- {split_name} ---")

            result = train_and_evaluate(
                meta, y, valid_mask, X_features, split_name, split_data,
                mode=mode, epochs=args.epochs,
            )

            if result is None:
                logger.warning(f"  Skipped {split_name} (insufficient data)")
                continue

            logger.info(f"  {mode_label}: bal_acc={result['bal_acc']:.4f}")

            # Save predictions
            fname = f"chemprop_{mode}_{split_name}.csv"
            out = pd.DataFrame({
                "idx": result["test_idx"],
                "y_true": result["y_test"],
                "y_pred": result["y_pred"],
            })
            for c in range(4):
                out[f"prob_{c}"] = result["y_prob"][:, c]
            out.to_csv(PRED_DIR / fname, index=False)

            all_results.append({
                "model": mode_label,
                "split": split_name,
                "bal_acc": result["bal_acc"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            })

    # Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLE_DIR / "benchmark_v4_chemprop.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("CHEMPROP V2 BENCHMARK SUMMARY")
    print("=" * 70)

    for mode_label in ["Chemprop", "Chemprop+Features"]:
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

    print(f"\n  --- Tree Model Baselines (153d Optuna) ---")
    print(f"  ma_bw_xgb_optuna: TSCV=0.669, Scaffold=0.597, Grouped=0.746")
    print(f"  et_optuna:        TSCV=0.646, Scaffold=0.610, Grouped=0.758")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results: {TABLE_DIR / 'benchmark_v4_chemprop.csv'}")


if __name__ == "__main__":
    main()
