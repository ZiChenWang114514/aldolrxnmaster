#!/usr/bin/env python
"""Train Chemprop v2 MPNN on Evans aldol reactions — reaction SMILES classification.

Models:
  - Chemprop: reaction SMILES → MPNN → 4-class
  - Chemprop+Cond: reaction SMILES + 14-dim conditions → MPNN → 4-class

Uses Chemprop v2 Python API with ReactionDatapoint + CondensedGraphOfReactionFeaturizer.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L
from rdkit import Chem
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
for d in [RESULTS_DIR / "predictions", RESULTS_DIR / "tables"]:
    d.mkdir(parents=True, exist_ok=True)


def parse_rxn_smiles(rxn_smi: str):
    """Parse 'reactants>>product' into (reactant_mol, product_mol)."""
    parts = rxn_smi.split(">>")
    if len(parts) != 2:
        return None, None
    rct_mol = Chem.MolFromSmiles(parts[0])
    pdt_mol = Chem.MolFromSmiles(parts[1])
    return rct_mol, pdt_mol


def load_data(split_name, with_conditions=False):
    """Load reaction data for a split."""
    rxn_df = pd.read_csv(FEAT_DIR / "reaction_smiles.csv")
    rxn_smiles = rxn_df["rxn_smiles_clean"].values

    labels_df = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels_df["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)
    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    # Load conditions if needed
    conditions = None
    if with_conditions:
        cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
        conditions = cond_df.values.astype(np.float32)

    return rxn_smiles, y, tr, va, te, conditions, labels_df


def build_datapoints(rxn_smiles, y, indices, conditions=None):
    """Build Chemprop ReactionDatapoints."""
    from chemprop.data import ReactionDatapoint

    datapoints = []
    for i in indices:
        rct, pdt = parse_rxn_smiles(str(rxn_smiles[i]))
        if rct is None or pdt is None:
            continue
        x_d = conditions[i] if conditions is not None else None
        dp = ReactionDatapoint(
            rct=rct, pdt=pdt,
            y=np.array([y[i]]),
            x_d=x_d,
        )
        datapoints.append(dp)
    return datapoints


def train_chemprop(split_name, with_conditions=False, epochs=30):
    """Train and evaluate Chemprop on one split."""
    from chemprop.data import ReactionDataset, build_dataloader
    from chemprop.featurizers import CondensedGraphOfReactionFeaturizer, RxnMode
    from chemprop.nn import BondMessagePassing, MulticlassClassificationFFN, NormAggregation
    from chemprop.nn.transforms import ScaleTransform
    from chemprop.models import MPNN

    model_name = "Chemprop+Cond" if with_conditions else "Chemprop"
    logger.info(f"\n{'='*60}\n  {model_name}: {split_name}\n{'='*60}")

    rxn_smiles, y, tr, va, te, conditions, labels_df = load_data(split_name, with_conditions)

    logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    logger.info(f"Train classes: {np.bincount(y[tr], minlength=4)}")

    # Build datapoints
    tr_dps = build_datapoints(rxn_smiles, y, tr, conditions)
    va_dps = build_datapoints(rxn_smiles, y, va, conditions)
    te_dps = build_datapoints(rxn_smiles, y, te, conditions)

    logger.info(f"Valid datapoints: train={len(tr_dps)}, val={len(va_dps)}, test={len(te_dps)}")

    # Featurizer
    featurizer = CondensedGraphOfReactionFeaturizer(mode_=RxnMode.REAC_DIFF)

    # Datasets
    tr_ds = ReactionDataset(tr_dps, featurizer=featurizer)
    va_ds = ReactionDataset(va_dps, featurizer=featurizer)
    te_ds = ReactionDataset(te_dps, featurizer=featurizer)

    tr_dl = build_dataloader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    va_dl = build_dataloader(va_ds, batch_size=64, shuffle=False, num_workers=0)
    te_dl = build_dataloader(te_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model components
    # CGR featurizer for reactions produces d_v=106, d_e=28 (not default 72/14)
    # x_d (molecule-level extra features) are concatenated after aggregation, before FFN
    d_xd = conditions.shape[1] if conditions is not None else 0
    mp = BondMessagePassing(d_v=106, d_e=28, depth=3, d_h=300, dropout=0.1)
    agg = NormAggregation()
    ffn = MulticlassClassificationFFN(n_classes=4, input_dim=300 + d_xd, hidden_dim=300, n_layers=2, dropout=0.1)

    # Build MPNN
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
    # preds is a list of tensors, each (batch, n_tasks, n_classes)
    all_probs = torch.cat(preds, dim=0).squeeze(1).numpy()  # (N, 4)
    y_pred = all_probs.argmax(axis=1)
    y_prob = torch.softmax(torch.tensor(all_probs), dim=1).numpy()
    y_test = y[te[:len(y_pred)]]  # handle potential dropped datapoints

    # Evaluate
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)

    logger.info(f"  {model_name}: bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, joint_acc={metrics['joint_accuracy']:.4f}")

    # Save predictions
    fname = "chemprop_cond" if with_conditions else "chemprop"
    out_df = pd.DataFrame({
        "idx": te[:len(y_pred)],
        "y_true": y_test,
        "y_pred": y_pred,
    })
    for c in range(4):
        out_df[f"prob_{c}"] = y_prob[:, c]
    out_df.to_csv(RESULTS_DIR / "predictions" / f"{fname}_{split_name}.csv", index=False)

    return metrics, ci


if __name__ == "__main__":
    splits = ["evans_temporal", "evans_scaffold", "evans_grouped_random_seed42"]

    for split in splits:
        # M1: Chemprop (reaction SMILES only)
        train_chemprop(split, with_conditions=False, epochs=30)

        # M2: Chemprop + Conditions
        train_chemprop(split, with_conditions=True, epochs=30)

    logger.info("\nDone! Now run: python scripts/rebuild_comparison.py")
