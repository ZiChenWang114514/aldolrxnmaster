#!/usr/bin/env python
"""Rebuild comparison tables from all prediction CSVs.

Scans results/predictions/ for all *_{split_name}.csv files,
recomputes metrics using the same evaluation functions,
and generates unified comparison tables.

This ensures ALL models are fairly compared even if they
were trained in different runs.
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
PRED_DIR = PROJECT / "results" / "predictions"
TABLE_DIR = PROJECT / "results" / "tables"

# Model name mapping: filename prefix → display name
NAME_MAP = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "xgboost_fullfp": "XGBoost-FullFP",
    "rf": "RF",
    "1_nn": "1-NN",
    "5_nn": "5-NN",
    "morgan_mlp": "Morgan-MLP",
    "drfp+xgboost": "DRFP+XGBoost",
    "drfp+lightgbm": "DRFP+LightGBM",
    "drfp+cond+xgboost": "DRFP+Cond+XGBoost",
    "rxnfp+xgboost": "RXNFP+XGBoost",
    "rxnfp+lightgbm": "RXNFP+LightGBM",
    "rxnfp+mlp": "RXNFP+MLP",
    "distilbert_rxn": "DistilBERT-Rxn",
    "roberta_rxn": "RoBERTa-Rxn",
    "chemberta_77m": "ChemBERTa-77M",
    "molt5_base": "MolT5-base",
    "t5chem_clf": "T5Chem-Clf",
    "chemprop": "Chemprop",
    "chemprop_cond": "Chemprop+Cond",
    "protonet": "ProtoNet",
    "chemahnet_aldol": "ChemAHNet-Aldol",
    "chienn_product": "ChiENN-Product",
    "equireact": "EquiReact",
    "gcpnet": "GCPNet",
    "drfp_aux_cond_xgboost": "DRFP+Aux+Cond-XGB",
    "auxchiral_xgboost": "AuxChiral-XGB",
    "auxchiral_ald_xgboost": "AuxChiral+Ald-XGB",
    "auxchiral_lgbm": "AuxChiral-LGBM",
    "auxchiral_noaux_xgboost": "CondOnly-XGB",
    "auxchiral_nobase_xgboost": "AuxNoBase-XGB",
    "majorityclass": "MajorityClass",
    "random": "Random",
    "chiralaldol_xgboost": "ChiralAldol-XGB",
    "chiralaldol_steronly_xgboost": "SterOnly-XGB",
    "chiralaldol_condaux_xgboost": "CondAux-XGB",
    "chiralaldol_drfp_xgboost": "ChiralAldol+DRFP-XGB",
    "chiralaldol_weighted_vote": "ChiralAldol-WtVote",
    "chiralaldol_stacking": "ChiralAldol-Stack",
    # Phase 11-A1: aldehyde steric descriptors added (75d = 24+10+35+6)
    "chiralaldol_v2_xgboost": "ChiralAldolV2-XGB",
    "chiralaldol_v2_stacking": "ChiralAldolV2-Stack",
}

# Display order
MODEL_ORDER = [
    "ChiralAldolV2-Stack", "ChiralAldolV2-XGB",
    "ChiralAldol-Stack", "ChiralAldol-WtVote", "ChiralAldol+DRFP-XGB", "ChiralAldol-XGB", "SterOnly-XGB", "CondAux-XGB",
    "DRFP+Cond+XGBoost", "DRFP+Aux+Cond-XGB",
    "AuxChiral-XGB", "AuxChiral+Ald-XGB", "AuxChiral-LGBM",
    "CondOnly-XGB", "AuxNoBase-XGB",
    "DRFP+XGBoost", "DRFP+LightGBM",
    "Chemprop+Cond", "Chemprop", "ProtoNet", "ChemAHNet-Aldol",
    "DistilBERT-Rxn", "RoBERTa-Rxn", "ChemBERTa-77M", "MolT5-base", "T5Chem-Clf",
    "ChiENN-Product", "EquiReact", "GCPNet",
    "Morgan-MLP", "XGBoost-FullFP", "XGBoost", "LightGBM", "RF",
    "RXNFP+XGBoost", "RXNFP+LightGBM", "RXNFP+MLP",
    "1-NN", "5-NN",
    "MajorityClass", "Random",
]


def rebuild_split(split_name):
    """Rebuild comparison table for one split."""
    logger.info(f"\n{'='*60}\n  Rebuilding: {split_name}\n{'='*60}")

    # Find all prediction files for this split
    pattern = f"*_{split_name}.csv"
    pred_files = sorted(PRED_DIR.glob(pattern))
    logger.info(f"Found {len(pred_files)} prediction files")

    results = []
    for f in pred_files:
        # Extract model name from filename
        model_key = f.stem.replace(f"_{split_name}", "")

        # Map to display name
        display_name = NAME_MAP.get(model_key, model_key)

        try:
            df = pd.read_csv(f)
            y_true = df["y_true"].values.astype(int)
            y_pred = df["y_pred"].values.astype(int)

            # Get probabilities if available
            prob_cols = [c for c in df.columns if c.startswith("prob_")]
            y_prob = df[prob_cols].values if prob_cols else None

            # Compute metrics
            m = compute_all_metrics(y_true, y_pred, y_prob)
            ci = compute_metrics_with_ci(y_true, y_pred, n_boot=500)

            results.append({
                "name": display_name,
                "metrics": m,
                "ci": ci,
            })
            logger.info(f"  {display_name}: bal_acc={m['balanced_accuracy']:.4f}, MCC={m['mcc']:.4f}")

        except Exception as e:
            logger.error(f"  Failed to process {f.name}: {e}")

    if not results:
        logger.warning(f"No results for {split_name}")
        return

    # Sort by MODEL_ORDER
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    results.sort(key=lambda r: order_map.get(r["name"], 999))

    # Build comparison table
    rows = []
    for r in results:
        m = r["metrics"]
        ci = r.get("ci", {})
        row = {
            "Model": r["name"],
            "Bal.Acc": f"{m['balanced_accuracy']:.4f}",
            "MCC": f"{m['mcc']:.4f}",
            "Joint": f"{m['joint_accuracy']:.4f}",
            "F1m": f"{m['f1_macro']:.4f}",
            "Ca": f"{m['ca_accuracy']:.4f}",
            "Cb": f"{m['cb_accuracy']:.4f}",
            "SA": f"{m['sa_accuracy']:.4f}",
            "F1_C0": f"{m['f1_class0']:.3f}",
            "F1_C1": f"{m['f1_class1']:.3f}",
            "F1_C2": f"{m['f1_class2']:.3f}",
            "F1_C3": f"{m['f1_class3']:.3f}",
        }
        if "balanced_accuracy" in ci:
            ba = ci["balanced_accuracy"]
            row["Bal.Acc 95%CI"] = f"[{ba['ci_lo']:.3f},{ba['ci_hi']:.3f}]"
        rows.append(row)

    rdf = pd.DataFrame(rows)
    print(f"\n{rdf.to_string(index=False)}")
    rdf.to_csv(TABLE_DIR / f"comparison_{split_name}.csv", index=False)
    logger.info(f"Saved: comparison_{split_name}.csv ({len(rows)} models)")

    # Save full metrics JSON
    full_data = []
    for r in results:
        full_data.append({
            "name": r["name"],
            "metrics": {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
            "cm": r["metrics"].get("confusion_matrix"),
            "ci": r.get("ci"),
        })
    with open(TABLE_DIR / f"full_{split_name}.json", "w") as f:
        json.dump(full_data, f, indent=2, default=str)


if __name__ == "__main__":
    rebuild_split("evans_temporal")
    rebuild_split("evans_scaffold")
    rebuild_split("evans_grouped_random_seed42")
    logger.info("\nDone! All comparison tables rebuilt.")
