#!/usr/bin/env python
"""Train T5Chem classification model for 4-class stereochemistry prediction.

Uses T5Chem's native classification head (T5ForProperty) with SimpleTokenizer.
Trains from scratch (num_layers=4, d_model=256) since MolT5's SentencePiece
tokenizer is incompatible with T5Chem's SimpleTokenizer.

Usage:
    conda activate aldol-rxn
    python scripts/run_t5chem_classification.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "t5chem"))

from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"
for d in [RESULTS_DIR / "predictions", RESULTS_DIR / "tables"]:
    d.mkdir(parents=True, exist_ok=True)


def prepare_data_files(rxn_smiles, labels, train_idx, val_idx, test_idx, data_dir):
    """Write T5Chem-format data files: {split}.source and {split}.target."""
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(data_dir / f"{name}.source", "w") as f:
            for i in idx:
                f.write(rxn_smiles[i] + "\n")
        with open(data_dir / f"{name}.target", "w") as f:
            for i in idx:
                f.write(f"{labels[i]}.0\n")


def run_t5chem_split(split_name):
    """Train and evaluate T5Chem on one split."""
    logger.info(f"\n{'='*60}\n  T5Chem SPLIT: {split_name}\n{'='*60}")

    # Load data
    rxn_df = pd.read_csv(FEAT_DIR / "reaction_smiles.csv")
    rxn_smiles = rxn_df["rxn_smiles_clean"].values

    labels_df = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels_df["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)
    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    logger.info(f"Train classes: {np.bincount(y[tr], minlength=4)}")

    # Prepare temp data directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="t5chem_aldol_"))
    data_dir = tmp_dir / "data"
    output_dir = tmp_dir / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    prepare_data_files(rxn_smiles, y, tr, va, te, data_dir)
    logger.info(f"Data files written to {data_dir}")

    # Train T5Chem
    from t5chem.run_trainer import train

    args = Namespace(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        task_type="classification",
        run_name=f"aldol_{split_name}",
        pretrain="",
        vocab="",
        tokenizer="simple",
        random_seed=42,
        num_epoch=50,
        eval_steps=50,
        log_step=50,
        batch_size=32,
        init_lr=5e-4,
        num_classes=4,
    )

    logger.info("Starting T5Chem training...")
    train(args)
    logger.info("T5Chem training complete.")

    # Run prediction on test set
    from t5chem.run_prediction import predict

    pred_args = Namespace(
        data_dir=str(data_dir),
        model_dir=str(output_dir),
        prediction=str(tmp_dir / "predictions.csv"),
        prefix="",
        num_beams=1,
        num_preds=1,
        batch_size=64,
    )

    predict(pred_args)

    # Read predictions and compute metrics
    pred_df = pd.read_csv(tmp_dir / "predictions.csv")
    y_test = y[te]
    y_pred = pred_df["prediction_1"].values.astype(int)

    metrics = compute_all_metrics(y_test, y_pred)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  T5Chem-Clf: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint_acc={metrics['joint_accuracy']:.4f}, "
                f"F1m={metrics['f1_macro']:.4f}")

    # Save predictions in same format as run_all_models.py
    # T5Chem classification outputs class indices, no probabilities
    out_df = pd.DataFrame({
        "idx": te,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    # Create uniform probability (no softmax available from T5Chem native prediction)
    prob = np.zeros((len(te), 4), dtype=np.float32)
    for i, p in enumerate(y_pred):
        prob[i, p] = 1.0
    for c in range(4):
        out_df[f"prob_{c}"] = prob[:, c]
    out_df.to_csv(RESULTS_DIR / "predictions" / f"t5chem_clf_{split_name}.csv", index=False)

    # Append to comparison table
    append_to_comparison(split_name, "T5Chem-Clf", metrics, ci)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return metrics, ci


def append_to_comparison(split_name, model_name, metrics, ci):
    """Append T5Chem results to the comparison CSV."""
    comp_path = RESULTS_DIR / "tables" / f"comparison_{split_name}.csv"

    row = {
        "Model": model_name,
        "Bal.Acc": f"{metrics['balanced_accuracy']:.4f}",
        "MCC": f"{metrics['mcc']:.4f}",
        "Joint": f"{metrics['joint_accuracy']:.4f}",
        "F1m": f"{metrics['f1_macro']:.4f}",
        "Ca": f"{metrics['ca_accuracy']:.4f}",
        "Cb": f"{metrics['cb_accuracy']:.4f}",
        "SA": f"{metrics['sa_accuracy']:.4f}",
        "F1_C0": f"{metrics['f1_class0']:.3f}",
        "F1_C1": f"{metrics['f1_class1']:.3f}",
        "F1_C2": f"{metrics['f1_class2']:.3f}",
        "F1_C3": f"{metrics['f1_class3']:.3f}",
    }
    if "balanced_accuracy" in ci:
        ba = ci["balanced_accuracy"]
        row["Bal.Acc 95%CI"] = f"[{ba['ci_lo']:.3f},{ba['ci_hi']:.3f}]"

    if comp_path.exists():
        df = pd.read_csv(comp_path)
        # Remove existing T5Chem row if present
        df = df[df["Model"] != model_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(comp_path, index=False)
    logger.info(f"  Results appended to {comp_path}")


if __name__ == "__main__":
    run_t5chem_split("evans_temporal")
    run_t5chem_split("evans_scaffold")
    run_t5chem_split("evans_grouped_random_seed42")
    logger.info("\nDone! T5Chem results in results/tables/ and results/predictions/")
