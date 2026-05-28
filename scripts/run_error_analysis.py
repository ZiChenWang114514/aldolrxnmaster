#!/usr/bin/env python3
"""D2: Error analysis — confusion matrices, high-confidence errors, label error candidates.

Usage:
    conda run -n aldol-rxn python scripts/run_error_analysis.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

PROJECT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT / "results" / "predictions_v4" / "optuna"
CLEAN_CSV = PROJECT / "data" / "clean_v4" / "substrate_aldol_clean.csv"
OUT_DIR = PROJECT / "results" / "analysis"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("error_analysis")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Error Analysis")
    logger.info("=" * 60)

    meta = pd.read_csv(CLEAN_CSV)

    # Load all TSCV predictions for champion model
    preds = []
    for i in range(1, 5):
        path = PRED_DIR / f"ma_bw_xgb_optuna_tscv_fold{i}.csv"
        if not path.exists():
            # Fallback to xgb_optuna if ma_bw not available
            path = PRED_DIR / f"xgb_optuna_tscv_fold{i}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["fold"] = i
            preds.append(df)

    if not preds:
        logger.error("No TSCV predictions found!")
        return

    pred_all = pd.concat(preds, ignore_index=True)
    logger.info(f"Total predictions: {len(pred_all)} across {len(preds)} folds")

    y_true = pred_all["y_true"].values.astype(int)
    y_pred = pred_all["y_pred"].values.astype(int)
    test_idx = pred_all["idx"].values.astype(int)

    # === 1. Global confusion matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=["true_RR", "true_RS", "true_SR", "true_SS"],
                          columns=["pred_RR", "pred_RS", "pred_SR", "pred_SS"])
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    logger.info(f"\nGlobal balanced accuracy: {bal_acc:.4f}")
    logger.info(f"Confusion matrix:\n{cm_df}")

    # Per-class recall
    for c in range(4):
        mask = y_true == c
        if mask.sum() > 0:
            recall = (y_pred[mask] == c).sum() / mask.sum()
            logger.info(f"  Class {c} recall: {recall:.3f} ({(y_pred[mask] == c).sum()}/{mask.sum()})")

    # Most common error pairs
    errors = [(y_true[i], y_pred[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    error_counts = Counter(errors)
    logger.info(f"\nMost common errors (true→pred):")
    for (t, p), cnt in error_counts.most_common(6):
        class_names = ["RR", "RS", "SR", "SS"]
        logger.info(f"  {class_names[t]}→{class_names[p]}: {cnt}")

    # === 2. Per-auxiliary confusion ===
    test_aux = meta.iloc[test_idx]["auxiliary_type"].values
    aux_results = []

    for aux in sorted(set(test_aux)):
        mask = test_aux == aux
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        acc = balanced_accuracy_score(yt, yp)

        per_class_recall = {}
        for c in range(4):
            cmask = yt == c
            if cmask.sum() > 0:
                per_class_recall[f"recall_class{c}"] = round((yp[cmask] == c).sum() / cmask.sum(), 3)
            else:
                per_class_recall[f"recall_class{c}"] = None

        aux_results.append({
            "auxiliary": aux,
            "n_test": mask.sum(),
            "bal_acc": round(acc, 4),
            **per_class_recall,
        })
        logger.info(f"  {aux:20s}: n={mask.sum():5d}, bal_acc={acc:.3f}")

    pd.DataFrame(aux_results).to_csv(OUT_DIR / "per_auxiliary_recall.csv", index=False)

    # === 3. High-confidence errors ===
    prob_cols = [c for c in pred_all.columns if c.startswith("prob_")]
    if prob_cols:
        prob_matrix = pred_all[prob_cols].values
        max_prob = prob_matrix.max(axis=1)

        high_conf_wrong = (max_prob > 0.8) & (y_true != y_pred)
        n_hc = high_conf_wrong.sum()
        logger.info(f"\nHigh-confidence errors (prob>0.8 but wrong): {n_hc}/{len(y_true)}")

        if n_hc > 0:
            hc_df = pred_all[high_conf_wrong].copy()
            hc_df["max_prob"] = max_prob[high_conf_wrong]
            hc_df["auxiliary_type"] = test_aux[high_conf_wrong]
            if "canonical_ketone_smiles" in meta.columns:
                hc_df["ketone_smiles"] = meta.iloc[hc_df["idx"].values]["canonical_ketone_smiles"].values
            if "canonical_aldehyde_smiles" in meta.columns:
                hc_df["aldehyde_smiles"] = meta.iloc[hc_df["idx"].values]["canonical_aldehyde_smiles"].values

            hc_df = hc_df.sort_values("max_prob", ascending=False)
            hc_df.to_csv(OUT_DIR / "high_confidence_errors.csv", index=False)
            logger.info(f"  Saved {n_hc} high-confidence errors")

            # Show top 5
            for _, row in hc_df.head(5).iterrows():
                cn = ["RR", "RS", "SR", "SS"]
                logger.info(f"    idx={row['idx']}: true={cn[int(row['y_true'])]}, "
                           f"pred={cn[int(row['y_pred'])]}, prob={row['max_prob']:.3f}, "
                           f"aux={row['auxiliary_type']}")

    # === 4. Label error candidates (consistently wrong across folds) ===
    # Find samples that appear in multiple folds' test sets and are always wrong
    idx_errors = {}  # idx → [list of (fold, true, pred)]
    for _, row in pred_all.iterrows():
        idx = int(row["idx"])
        if row["y_true"] != row["y_pred"]:
            if idx not in idx_errors:
                idx_errors[idx] = []
            idx_errors[idx].append({
                "fold": int(row["fold"]),
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"]),
            })

    # Samples wrong every time they appear (most likely label errors)
    idx_appearances = pred_all.groupby("idx").size()
    candidates = []
    for idx, errs in idx_errors.items():
        n_appear = idx_appearances.get(idx, 1)
        if len(errs) >= n_appear and n_appear >= 1:
            candidates.append({
                "idx": idx,
                "n_wrong": len(errs),
                "n_appearances": n_appear,
                "y_true": errs[0]["y_true"],
                "predicted_as": Counter([e["y_pred"] for e in errs]).most_common(1)[0][0],
                "auxiliary_type": meta.iloc[idx]["auxiliary_type"] if idx < len(meta) else "?",
            })

    # Add SMILES for review
    for c in candidates:
        idx = c["idx"]
        if idx < len(meta):
            if "canonical_ketone_smiles" in meta.columns:
                c["ketone_smiles"] = meta.iloc[idx]["canonical_ketone_smiles"]

    cand_df = pd.DataFrame(candidates).sort_values("n_wrong", ascending=False)
    cand_df.to_csv(OUT_DIR / "label_error_candidates.csv", index=False)
    logger.info(f"\nLabel error candidates (always wrong): {len(candidates)}")

    logger.info(f"\nResults saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
