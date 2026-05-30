#!/usr/bin/env python3
"""SPMS Directional Steric Feature Benchmark.

Compares baseline 154d vs SPMS-augmented features across Tree/GNN/Chemprop.

Usage:
    conda run -n aldol-rxn python scripts/run_spms_benchmark.py
    conda run -n aldol-rxn python scripts/run_spms_benchmark.py --method spms --model xgb
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import FEAT_DIR, PRED_DIR, RESULTS_DIR, SPMS_DIR
from chiralaldol.data_io import load_labels, load_splits, save_predictions
from chiralaldol.model_trainers import train_xgb, train_et, train_rf
from chiralaldol.spms_compressor import extract_spms_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spms_benchmark")

TARGET_LABEL = "label_joint"
N_CLASSES = 4
OUT_DIR = PRED_DIR / "spms"
TABLE_DIR = RESULTS_DIR / "tables"


def load_feature_set(name):
    """Load a feature set by name."""
    if name == "baseline":
        df = pd.read_csv(FEAT_DIR / "v4_features.csv")
    elif name == "spms_ae":
        df = pd.read_csv(FEAT_DIR / "v4_features_spms.csv")
    elif name == "spms_pca":
        # Load baseline + PCA-compressed SPMS
        base = pd.read_csv(FEAT_DIR / "v4_features.csv")
        latent = np.load(SPMS_DIR / "spms_latent.npy")  # may be PCA
        spms_cols = [f"spms_pca_{i}" for i in range(latent.shape[1])]
        spms_df = pd.DataFrame(latent, columns=spms_cols)
        df = pd.concat([base, spms_df], axis=1)
    elif name == "spms_stats":
        spms_arrays = np.load(SPMS_DIR / "spms_arrays.npy")
        base = pd.read_csv(FEAT_DIR / "v4_features.csv")
        stats, stat_names = extract_spms_stats(spms_arrays)
        spms_df = pd.DataFrame(stats, columns=stat_names)
        df = pd.concat([base, spms_df], axis=1)
    elif name == "spms_only":
        latent = np.load(SPMS_DIR / "spms_latent.npy")
        cols = [f"spms_{i}" for i in range(latent.shape[1])]
        df = pd.DataFrame(latent, columns=cols)
    elif name == "face_map":
        base = pd.read_csv(FEAT_DIR / "v4_features.csv")
        face = pd.read_csv(SPMS_DIR / "face_map_features.csv")
        df = pd.concat([base, face], axis=1)
    elif name == "spms_face":
        base = pd.read_csv(FEAT_DIR / "v4_features.csv")
        spms_arrays = np.load(SPMS_DIR / "spms_arrays.npy")
        stats, stat_names = extract_spms_stats(spms_arrays)
        spms_df = pd.DataFrame(stats, columns=stat_names)
        face = pd.read_csv(SPMS_DIR / "face_map_features.csv")
        df = pd.concat([base, spms_df, face], axis=1)
    else:
        raise ValueError(f"Unknown feature set: {name}")

    X = df.values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    return X, list(df.columns)


def run_benchmark(feature_sets, models, splits):
    """Run benchmark across feature sets × models × splits."""
    labels = load_labels()
    valid_mask = labels[TARGET_LABEL].notna().values
    y_full = labels[TARGET_LABEL].values
    y = np.where(valid_mask, y_full, -1).astype(int)

    all_results = []

    for feat_name in feature_sets:
        X, _ = load_feature_set(feat_name)
        logger.info(f"\n{'='*50}\n  Feature set: {feat_name} ({X.shape[1]}d)\n{'='*50}")

        for model_name, trainer in models.items():
            for split_name, split_data in sorted(splits.items()):
                tr_raw = np.array(split_data["train"], dtype=int)
                tr = tr_raw[valid_mask[tr_raw]]
                va_raw = np.array(split_data.get("val", []), dtype=int)
                va = va_raw[valid_mask[va_raw]] if len(va_raw) > 0 else np.array([], dtype=int)
                te_raw = np.array(split_data["test"], dtype=int)
                te = te_raw[valid_mask[te_raw]]

                if len(va) == 0:
                    va = tr[-max(1, len(tr) // 10):]
                    tr = tr[:-len(va)]

                if len(tr) < 10 or len(te) < 3:
                    continue

                model = trainer(X[tr], y[tr], X[va], y[va])
                y_pred = model.predict(X[te])
                y_prob = (model.predict_proba(X[te])
                          if hasattr(model, "predict_proba")
                          else np.eye(N_CLASSES)[y_pred.astype(int)])

                bal_acc = balanced_accuracy_score(y[te], y_pred)
                mcc = matthews_corrcoef(y[te], y_pred)

                save_predictions(
                    OUT_DIR / f"{feat_name}_{model_name}_{split_name}.csv",
                    te, y[te], y_pred, y_prob, N_CLASSES)

                all_results.append({
                    "features": feat_name, "dims": X.shape[1],
                    "model": model_name, "split": split_name,
                    "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                    "n_train": len(tr), "n_test": len(te),
                })

    return pd.DataFrame(all_results)


def print_summary(results_df):
    """Print comparison summary."""
    print("\n" + "=" * 80)
    print("SPMS DIRECTIONAL STERIC FEATURES BENCHMARK")
    print("=" * 80)

    for model in sorted(results_df["model"].unique()):
        print(f"\n  Model: {model}")
        print(f"  {'Features':<25s} {'Dims':>5s}  {'TSCV':>10s}  {'Grouped':>10s}  {'Scaffold':>10s}")
        print(f"  {'-'*25} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

        for feat in results_df["features"].unique():
            sub = results_df[(results_df["model"] == model) &
                             (results_df["features"] == feat)]
            dims = sub["dims"].iloc[0]

            tscv = sub[sub["split"].str.contains("tscv")]["bal_acc"]
            grouped = sub[sub["split"].str.contains("grouped")]["bal_acc"]
            scaffold = sub[sub["split"].str.contains("scaffold")]["bal_acc"]

            tscv_str = f"{tscv.mean():.4f}±{tscv.std():.3f}" if len(tscv) > 0 else "—"
            grp_str = f"{grouped.mean():.4f}±{grouped.std():.3f}" if len(grouped) > 0 else "—"
            scf_str = f"{scaffold.mean():.4f}" if len(scaffold) > 0 else "—"

            print(f"  {feat:<25s} {dims:5d}  {tscv_str:>10s}  {grp_str:>10s}  {scf_str:>10s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+",
                        default=["baseline", "spms_ae", "spms_stats"],
                        choices=["baseline", "spms_ae", "spms_pca",
                                 "spms_stats", "spms_only",
                                 "face_map", "spms_face"])
    parser.add_argument("--models", nargs="+", default=["xgb", "et"],
                        choices=["xgb", "et", "rf"])
    parser.add_argument("--splits", default="all",
                        choices=["tscv", "grouped", "scaffold", "all"])
    args = parser.parse_args()

    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    if args.splits == "tscv":
        splits = {k: v for k, v in splits.items() if "tscv" in k}
    elif args.splits == "grouped":
        splits = {k: v for k, v in splits.items() if "grouped" in k}
    elif args.splits == "scaffold":
        splits = {k: v for k, v in splits.items() if "scaffold" in k}

    model_map = {"xgb": train_xgb, "et": train_et, "rf": train_rf}
    models = {k: model_map[k] for k in args.models}

    results_df = run_benchmark(args.features, models, splits)
    results_df.to_csv(TABLE_DIR / "benchmark_spms.csv", index=False)

    print_summary(results_df)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
