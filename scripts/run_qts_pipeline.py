#!/usr/bin/env python
"""Phase C1: qTS Pipeline — compute quasi-TS features and train ChiralAldolV4-XGB.

Stages:
  1. Extract aldehyde SMILES from reaction SMILES
  2. Run 4×1822 = 7288 xTB single-points on ZT TS scaffolds
  3. Save qts_features.csv to data/processed/chiralaldol/
  4. Train ChiralAldolV4-XGB (V2 75d + 4d qTS = 79d) on all 3 splits
  5. Rebuild comparison tables

Usage:
    conda run -n aldol-rxn python scripts/run_qts_pipeline.py [--workers N] [--skip-compute]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci
from chiralaldol.qts_builder import (
    QTS_FEATURE_NAMES,
    compute_qts_features_batch,
)

RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
CHIRALALDOL_DIR = PROJECT / "data" / "processed" / "chiralaldol"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"
CHECKPOINT = CHIRALALDOL_DIR / "qts_checkpoint.pkl"
QTS_FEATURES_CSV = CHIRALALDOL_DIR / "qts_features.csv"


# ── Aldehyde extraction ───────────────────────────────────────────────────────

_ALD_SMARTS = Chem.MolFromSmarts("[CX3;H1](=[OX1])")


def extract_aldehyde_smiles(rxn_smi: str) -> str | None:
    """Extract aldehyde SMILES from reaction SMILES (reactant side)."""
    reactant_block = rxn_smi.split(">>")[0]
    for frag in reactant_block.split("."):
        mol = Chem.MolFromSmiles(frag)
        if mol and mol.HasSubstructMatch(_ALD_SMARTS):
            return frag
    return None


# ── XGBoost training ──────────────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, X_val, y_val):
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15,
         "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0.0
    for cfg in configs:
        cfg.update({
            "objective": "multi:softprob", "num_class": 4,
            "tree_method": "hist", "random_state": 42,
            "n_jobs": 4, "verbosity": 0,
            "gamma": 0.1, "reg_lambda": 1.0,
        })
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def eval_save(name, y_test, y_pred, y_prob, test_idx, split_name):
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    logger.info(f"  {name} [{split_name}]: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}")
    out = pd.DataFrame({"idx": test_idx, "y_true": y_test, "y_pred": y_pred})
    for c in range(4):
        out[f"prob_{c}"] = y_prob[:, c]
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_dir / f"{name}_{split_name}.csv", index=False)
    return metrics


def load_split(split_name):
    with open(SPLIT_DIR / f"{split_name}.json") as f:
        sp = json.load(f)
    return np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-compute", action="store_true",
                        help="Skip xTB computation, load existing qts_features.csv")
    args = parser.parse_args()

    CHIRALALDOL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Extract aldehyde SMILES ──────────────────────────────────
    logger.info("Stage 1: Extracting aldehyde SMILES...")
    rxn_df = pd.read_csv(FEAT_DIR / "reaction_smiles.csv")
    aldehydes_smi = [extract_aldehyde_smiles(smi) for smi in rxn_df["rxn_smiles_clean"]]
    n_found = sum(x is not None for x in aldehydes_smi)
    logger.info(f"  Found: {n_found}/{len(aldehydes_smi)} aldehyde SMILES")

    # ── Stage 2: Compute qTS features ─────────────────────────────────────
    if args.skip_compute and QTS_FEATURES_CSV.exists():
        logger.info("Stage 2: Loading existing qts_features.csv (--skip-compute)")
        qts_df = pd.read_csv(QTS_FEATURES_CSV)
    else:
        logger.info(f"Stage 2: Computing qTS features ({args.workers} workers)...")
        enolates_df = pd.read_csv(CHIRALALDOL_DIR / "enolates.csv")
        qts_df = compute_qts_features_batch(
            enolates_df,
            aldehydes_smi,
            checkpoint_path=CHECKPOINT,
            n_workers=args.workers,
        )
        qts_df.to_csv(QTS_FEATURES_CSV, index=False)
        logger.info(f"  Saved to {QTS_FEATURES_CSV}")

    n_valid = qts_df.notna().all(axis=1).sum()
    logger.info(f"  qTS valid: {n_valid}/{len(qts_df)}")
    logger.info(f"  NaN rate per feature:\n{qts_df.isna().mean().to_string()}")

    # ── Stage 3: Build V4 feature matrix (V2 75d + qTS 4d = 79d) ──────────
    logger.info("Stage 3: Building V4 feature matrix...")
    from chiralaldol.feature_builder import build_chiralaldol_v2_features
    X_v2, names_v2 = build_chiralaldol_v2_features(PROJECT)
    logger.info(f"  V2 base: {X_v2.shape}")

    X_qts = qts_df.values.astype(np.float32)
    np.nan_to_num(X_qts, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    X_v4 = np.hstack([X_v2, X_qts])
    names_v4 = names_v2 + QTS_FEATURE_NAMES
    logger.info(f"  V4: {X_v4.shape} ({X_v2.shape[1]}d V2 + {X_qts.shape[1]}d qTS)")

    # ── Stage 4: Train and evaluate ────────────────────────────────────────
    logger.info("Stage 4: Training ChiralAldolV4-XGB...")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    splits = ["evans_temporal", "evans_scaffold", "evans_grouped_random_seed42"]
    for split_name in splits:
        tr, va, te = load_split(split_name)
        m = train_xgb(X_v4[tr], y[tr], X_v4[va], y[va])
        eval_save("chiralaldol_v4_xgboost",
                  y[te], m.predict(X_v4[te]), m.predict_proba(X_v4[te]),
                  te, split_name)

    # ── Stage 5: Rebuild comparison ────────────────────────────────────────
    logger.info("Stage 5: Rebuilding comparison tables...")
    import subprocess
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "rebuild_comparison.py")],
        check=True,
    )
    logger.info("Done! Check results/tables/ for updated comparison.")


if __name__ == "__main__":
    main()
