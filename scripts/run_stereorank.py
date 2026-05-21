"""StereoRank: Pairwise Ranking for Stereoselectivity Prediction.

Trains XGBRanker on enumerated aldol product candidates using TSCV evaluation.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from chiralaldol.pairwise_builder import (
    build_candidate_features,
    build_ranking_groups,
    split_by_reactions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "v3"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_DIR / "results" / "predictions" / "steric"
STEREORANK_DIR = DATA_DIR / "stereorank"


def load_data():
    """Load candidates and reaction features."""
    # Load candidates
    candidates_df = pd.read_csv(STEREORANK_DIR / "candidates.csv")
    logger.info(f"Loaded {len(candidates_df)} candidate rows "
                f"({candidates_df['reaction_id'].nunique()} reactions)")

    # Load main dataset for features
    main_df = pd.read_csv(DATA_DIR / "evans_clean_20260515.csv")

    # Feature columns (from V3 pipeline — steric + conditions + auxiliary)
    steric_cols = [c for c in main_df.columns if c.startswith(("Vbur_", "L_", "B1_", "B5_",
                                                                "sin_tau", "cos_tau",
                                                                "n_conformers", "n_clusters"))]
    ald_cols = [c for c in main_df.columns if c.startswith("ald_")]
    cond_cols = [c for c in main_df.columns if c.startswith(("base_", "metal_", "solvent_",
                                                              "act_", "has_"))]
    aux_cols = [c for c in main_df.columns if c.startswith("aux_")]

    feature_cols = steric_cols + ald_cols + cond_cols + aux_cols
    # Remove non-numeric
    feature_cols = [c for c in feature_cols if main_df[c].dtype in (np.float64, np.int64, float, int)]

    logger.info(f"Using {len(feature_cols)} reaction-level features")

    return candidates_df, main_df, feature_cols


def train_and_evaluate_fold(train_data, test_data, fold_name, hyperparams=None):
    """Train XGBRanker on one fold and evaluate."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return None

    if hyperparams is None:
        hyperparams = {
            "objective": "rank:pairwise",
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    model = xgb.XGBRanker(**hyperparams)

    # Train
    model.fit(
        train_data["X"],
        train_data["y"],
        group=train_data["groups"],
    )

    # Predict scores for test candidates
    scores = model.predict(test_data["X"])

    # For each reaction group, select the candidate with highest score
    test_rxn_ids = test_data["reaction_ids"]
    test_cand_ids = test_data["candidate_ids"]
    test_labels = test_data["y"]

    # Group predictions by reaction
    unique_rxns = np.unique(test_rxn_ids)
    y_true = []
    y_pred = []

    for rxn_id in unique_rxns:
        mask = test_rxn_ids == rxn_id
        rxn_scores = scores[mask]
        rxn_cands = test_cand_ids[mask]
        rxn_labels = test_labels[mask]

        # True label = candidate_id of true product
        true_idx = np.where(rxn_labels == 1)[0]
        if len(true_idx) == 0:
            continue
        true_cand = rxn_cands[true_idx[0]]

        # Predicted = candidate with highest score
        pred_cand = rxn_cands[np.argmax(rxn_scores)]

        y_true.append(int(true_cand))
        y_pred.append(int(pred_cand))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Top-1 accuracy (= fraction where highest-scored candidate is the true product)
    top1_acc = np.mean(y_true == y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # MRR (Mean Reciprocal Rank)
    mrr_values = []
    for rxn_id in unique_rxns:
        mask = test_rxn_ids == rxn_id
        rxn_scores = scores[mask]
        rxn_labels = test_labels[mask]
        # Rank (descending score)
        rank_order = np.argsort(-rxn_scores)
        true_pos = np.where(rxn_labels == 1)[0]
        if len(true_pos) > 0:
            rank_of_true = np.where(rank_order == true_pos[0])[0][0] + 1
            mrr_values.append(1.0 / rank_of_true)
    mrr = np.mean(mrr_values) if mrr_values else 0.0

    logger.info(f"  {fold_name}: top1_acc={top1_acc:.4f}, bal_acc={bal_acc:.4f}, MRR={mrr:.4f}")

    return {
        "fold": fold_name,
        "top1_acc": top1_acc,
        "bal_acc": bal_acc,
        "mrr": mrr,
        "n_test": len(unique_rxns),
        "model": model,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_tscv(candidates_df, main_df, feature_cols):
    """Run Temporal Split Cross-Validation (4-fold)."""
    logger.info("=" * 60)
    logger.info("StereoRank — TSCV 4-fold evaluation")
    logger.info("=" * 60)

    # Build candidate features
    cand_feats = build_candidate_features(candidates_df, main_df, feature_cols)
    logger.info(f"Candidate features shape: {cand_feats.shape}")

    # Get all feature columns for ranking
    feat_cols_for_rank = [c for c in cand_feats.columns
                          if c not in ("reaction_id", "candidate_id", "is_true_product")]

    # Build ranking groups
    ranking_data = build_ranking_groups(cand_feats, feat_cols_for_rank)
    logger.info(f"Ranking data: X={ranking_data['X'].shape}, "
                f"groups={len(ranking_data['groups'])} reactions")

    # Available reaction IDs in our candidates
    available_rxns = set(candidates_df["reaction_id"].unique())

    # Load TSCV splits
    fold_results = []
    for fold_num in range(1, 5):
        split_file = SPLITS_DIR / f"evans_tscv_fold{fold_num}.json"
        if not split_file.exists():
            # Try alternative naming
            split_file = SPLITS_DIR / f"tscv_fold{fold_num}.json"
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue

        with open(split_file) as f:
            split = json.load(f)

        train_ids = [i for i in split["train"] if i in available_rxns]
        test_ids = [i for i in split["test"] if i in available_rxns]

        if not train_ids or not test_ids:
            logger.warning(f"Fold {fold_num}: empty after filtering available reactions")
            continue

        logger.info(f"Fold {fold_num}: train={len(train_ids)}, test={len(test_ids)}")

        train_data, test_data = split_by_reactions(ranking_data, train_ids, test_ids)
        result = train_and_evaluate_fold(train_data, test_data, f"tscv_fold{fold_num}")

        if result is not None:
            fold_results.append(result)

    if not fold_results:
        logger.error("No fold results!")
        return None

    # Summary
    bal_accs = [r["bal_acc"] for r in fold_results]
    top1_accs = [r["top1_acc"] for r in fold_results]
    mrrs = [r["mrr"] for r in fold_results]

    logger.info("=" * 60)
    logger.info(f"TSCV Summary ({len(fold_results)} folds):")
    logger.info(f"  Balanced Accuracy: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    logger.info(f"  Top-1 Accuracy:    {np.mean(top1_accs):.4f} ± {np.std(top1_accs):.4f}")
    logger.info(f"  MRR:               {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
    logger.info("=" * 60)

    # Compare with baseline
    logger.info("\nBaseline comparison:")
    logger.info(f"  MechAware-Full TSCV:  0.733 ± 0.074")
    logger.info(f"  StereoRank TSCV:      {np.mean(bal_accs):.3f} ± {np.std(bal_accs):.3f}")
    diff = np.mean(bal_accs) - 0.733
    logger.info(f"  Δ = {diff:+.3f} ({'IMPROVEMENT' if diff > 0 else 'needs work'})")

    # Save results
    results_summary = {
        "model": "StereoRank-v1",
        "method": "XGBRanker_pairwise",
        "n_features": len(feat_cols_for_rank),
        "n_reactions": len(available_rxns),
        "tscv_mean_bal_acc": float(np.mean(bal_accs)),
        "tscv_std_bal_acc": float(np.std(bal_accs)),
        "tscv_mean_top1": float(np.mean(top1_accs)),
        "tscv_mean_mrr": float(np.mean(mrrs)),
        "folds": {r["fold"]: {"bal_acc": r["bal_acc"], "top1": r["top1_acc"], "mrr": r["mrr"]}
                  for r in fold_results},
    }

    results_file = STEREORANK_DIR / "stereorank_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return results_summary


def run_classification_baseline(candidates_df, main_df, feature_cols):
    """Run XGBoost 4-class classification on same data for direct comparison (Ablation A1)."""
    try:
        import xgboost as xgb
    except ImportError:
        return None

    logger.info("\n" + "=" * 60)
    logger.info("Ablation A1: XGBoost 4-class Classification (same features)")
    logger.info("=" * 60)

    available_rxns = set(candidates_df["reaction_id"].unique())

    # Build features (reaction-level only, no candidate encoding)
    X = main_df.loc[main_df.index.isin(available_rxns), feature_cols].values
    y = main_df.loc[main_df.index.isin(available_rxns), "label_joint"].values
    indices = main_df.loc[main_df.index.isin(available_rxns)].index.values

    fold_accs = []
    for fold_num in range(1, 5):
        split_file = SPLITS_DIR / f"evans_tscv_fold{fold_num}.json"
        if not split_file.exists():
            split_file = SPLITS_DIR / f"tscv_fold{fold_num}.json"
        if not split_file.exists():
            continue

        with open(split_file) as f:
            split = json.load(f)

        train_ids = [i for i in split["train"] if i in available_rxns]
        test_ids = [i for i in split["test"] if i in available_rxns]

        train_mask = np.isin(indices, train_ids)
        test_mask = np.isin(indices, test_ids)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, eval_metric="mlogloss",
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        fold_accs.append(bal_acc)
        logger.info(f"  Fold {fold_num}: bal_acc={bal_acc:.4f}")

    if fold_accs:
        logger.info(f"  Classification TSCV: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    return {"mean": np.mean(fold_accs), "std": np.std(fold_accs)} if fold_accs else None


def main():
    candidates_df, main_df, feature_cols = load_data()

    # Main experiment: StereoRank
    stereorank_results = run_tscv(candidates_df, main_df, feature_cols)

    # Ablation A1: Classification baseline on same data subset
    classification_results = run_classification_baseline(candidates_df, main_df, feature_cols)

    if stereorank_results and classification_results:
        diff = stereorank_results["tscv_mean_bal_acc"] - classification_results["mean"]
        logger.info(f"\n{'='*60}")
        logger.info(f"PARADIGM COMPARISON (same data, same features):")
        logger.info(f"  Classification: {classification_results['mean']:.4f}")
        logger.info(f"  StereoRank:     {stereorank_results['tscv_mean_bal_acc']:.4f}")
        logger.info(f"  Δ (Ranking - Classification) = {diff:+.4f}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
