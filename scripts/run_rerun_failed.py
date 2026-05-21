#!/usr/bin/env python
"""Re-run only the failed/fixed models: DistilBERT, RoBERTa, MolT5.

Does NOT re-run successful models (DRFP, RXNFP, ChemBERTa, baselines).
After this, run rebuild_comparison.py to regenerate unified tables.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"

# Import model functions from run_all_models
sys.path.insert(0, str(PROJECT / "scripts"))
from run_all_models import (
    load_rxn_smiles,
    train_transformer,
    predict_tf,
    train_molt5,
    predict_molt5,
    save_preds,
    evaluate_model,
)


def load_split(split_name):
    """Load labels and split indices."""
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)

    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    return {
        "y_train": y[tr], "y_val": y[va], "y_test": y[te],
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "labels_df": labels,
    }


def run_split(split_name):
    logger.info(f"\n{'='*60}\n  RERUN: {split_name}\n{'='*60}")
    data = load_split(split_name)
    rxn_smi = load_rxn_smiles()

    logger.info(f"Train={len(data['y_train'])}, Val={len(data['y_val'])}, Test={len(data['y_test'])}")

    # --- DistilBERT ---
    logger.info("\n--- DistilBERT-Rxn ---")
    try:
        tm, tt, td = train_transformer(data, rxn_smi, "distilbert-base-uncased", 15)
        ts = rxn_smi[data["test_idx"]]
        y_pred, y_prob = predict_tf(tm, tt, td, ts)
        r = evaluate_model("DistilBERT-Rxn", data["y_test"], y_pred, y_prob)
        save_preds("distilbert_rxn", split_name, data["labels_df"], data["test_idx"], y_pred, y_prob)
        del tm
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"DistilBERT failed: {e}")

    # --- RoBERTa ---
    logger.info("\n--- RoBERTa-Rxn ---")
    try:
        tm, tt, td = train_transformer(data, rxn_smi, "roberta-base", 15)
        ts = rxn_smi[data["test_idx"]]
        y_pred, y_prob = predict_tf(tm, tt, td, ts)
        r = evaluate_model("RoBERTa-Rxn", data["y_test"], y_pred, y_prob)
        save_preds("roberta_rxn", split_name, data["labels_df"], data["test_idx"], y_pred, y_prob)
        del tm
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"RoBERTa failed: {e}")

    # --- MolT5-base (fixed: freeze 6 not 10) ---
    logger.info("\n--- MolT5-base (fixed) ---")
    try:
        tm, tt, td = train_molt5(data, rxn_smi, epochs=15)
        ts = rxn_smi[data["test_idx"]]
        y_pred, y_prob = predict_molt5(tm, tt, td, ts)
        r = evaluate_model("MolT5-base", data["y_test"], y_pred, y_prob)
        save_preds("molt5_base", split_name, data["labels_df"], data["test_idx"], y_pred, y_prob)
        del tm
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"MolT5 failed: {e}")


if __name__ == "__main__":
    run_split("evans_temporal")
    run_split("evans_scaffold")
    run_split("evans_grouped_random_seed42")
    logger.info("\nDone! Now run: python scripts/rebuild_comparison.py")
