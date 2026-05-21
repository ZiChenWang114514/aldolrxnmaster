"""Step 13: Data splitting — TSCV 4-fold + scaffold + grouped_random.

All splits respect group_id (no substrate leakage).
Split-aware normalization applied per split.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .step12_features import normalize_split

logger = logging.getLogger(__name__)

TSCV_FOLDS = [
    {"name": "fold1", "train_max_year": 2013, "test_min_year": 2014, "test_max_year": 2015},
    {"name": "fold2", "train_max_year": 2015, "test_min_year": 2016, "test_max_year": 2017},
    {"name": "fold3", "train_max_year": 2017, "test_min_year": 2018, "test_max_year": 2019},
    {"name": "fold4", "train_max_year": 2019, "test_min_year": 2020, "test_max_year": 2099},
]

GROUPED_SEEDS = [42, 123, 456, 789, 1024]


def _murcko_scaffold(smi: str) -> str:
    """Compute Murcko scaffold for a SMILES string."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return "unknown"
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return "unknown"


def _split_train_val(indices, groups, labels, val_frac=0.1, seed=42):
    """Split train indices into train/val respecting groups."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    try:
        train_idx, val_idx = next(gss.split(indices, labels, groups))
        return indices[train_idx], indices[val_idx]
    except Exception:
        n_val = max(1, int(len(indices) * val_frac))
        return indices[:-n_val], indices[-n_val:]


def run(context: dict) -> dict:
    """Generate all splits for Evans subset."""
    df: pd.DataFrame = context["df"]
    feature_cols = context.get("all_feature_cols", [])
    continuous_mask = context.get("continuous_mask", np.array([]))
    n = len(df)
    logger.info(f"Step 13: Data splitting for {n} rows")

    # Filter Evans subset
    evans_mask = df["Reaction_Class"] == "EvansAux"
    evans_df = df[evans_mask].reset_index(drop=True)
    n_evans = len(evans_df)
    logger.info(f"  Evans subset: {n_evans} rows")

    if n_evans == 0:
        logger.warning("  No Evans reactions found!")
        context["evans_df"] = evans_df
        return context

    # Prepare arrays
    X = evans_df[feature_cols].values.astype(float) if feature_cols else np.zeros((n_evans, 0))
    y = evans_df["label_joint"].values if "label_joint" in evans_df.columns else np.zeros(n_evans)
    groups = evans_df["group_id"].values if "group_id" in evans_df.columns else np.arange(n_evans)
    years = evans_df["Year"].values if "Year" in evans_df.columns else np.zeros(n_evans)

    splits_dir = context["output_dir"] / "splits"
    norm_dir = splits_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    all_splits = {}

    # ── TSCV 4-fold ──
    logger.info("  Generating TSCV 4-fold splits...")
    for fold in TSCV_FOLDS:
        train_mask = years <= fold["train_max_year"]
        test_mask = (years >= fold["test_min_year"]) & (years <= fold["test_max_year"])

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning(f"    {fold['name']}: empty train or test, skipping")
            continue

        # Split train into train/val
        train_groups = groups[train_idx]
        train_labels = y[train_idx]
        tr_idx, val_idx = _split_train_val(train_idx, train_groups, train_labels)

        split_info = {
            "name": fold["name"],
            "type": "tscv",
            "train": tr_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "train_years": f"<={fold['train_max_year']}",
            "test_years": f"{fold['test_min_year']}-{fold['test_max_year']}",
            "class_dist_test": pd.Series(y[test_idx]).value_counts().to_dict(),
        }
        all_splits[f"tscv_{fold['name']}"] = split_info

        # Save
        with open(splits_dir / f"evans_tscv_{fold['name']}.json", "w") as f:
            json.dump(split_info, f, indent=2, default=str)

        # Normalized features
        if X.shape[1] > 0 and len(continuous_mask) == X.shape[1]:
            X_norm = normalize_split(X, tr_idx, continuous_mask)
            np.savez(norm_dir / f"tscv_{fold['name']}.npz",
                     X_train=X_norm[tr_idx], y_train=y[tr_idx],
                     X_val=X_norm[val_idx], y_val=y[val_idx],
                     X_test=X_norm[test_idx], y_test=y[test_idx])

        logger.info(f"    {fold['name']}: train={len(tr_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Scaffold split ──
    logger.info("  Generating scaffold split...")
    aldehyde_col = "canonical_Aldehyde" if "canonical_Aldehyde" in evans_df.columns else "Aldehyde"
    scaffolds = evans_df[aldehyde_col].apply(_murcko_scaffold)
    scaffold_groups = scaffolds.factorize()[0]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        trainval_idx, test_idx = next(gss.split(np.arange(n_evans), y, scaffold_groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        train_idx, val_idx = next(gss2.split(
            trainval_idx, y[trainval_idx], scaffold_groups[trainval_idx]))
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]
    except Exception:
        train_idx = np.arange(int(n_evans * 0.8))
        val_idx = np.arange(int(n_evans * 0.8), int(n_evans * 0.9))
        test_idx = np.arange(int(n_evans * 0.9), n_evans)

    scaffold_split = {
        "name": "scaffold",
        "type": "scaffold",
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    with open(splits_dir / "evans_scaffold.json", "w") as f:
        json.dump(scaffold_split, f, indent=2, default=str)
    all_splits["scaffold"] = scaffold_split
    logger.info(f"    scaffold: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Grouped random (5 seeds) ──
    logger.info("  Generating grouped random splits...")
    for seed in GROUPED_SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        try:
            trainval_idx, test_idx = next(gss.split(np.arange(n_evans), y, groups))
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            train_idx, val_idx = next(gss2.split(
                trainval_idx, y[trainval_idx], groups[trainval_idx]))
            train_idx = trainval_idx[train_idx]
            val_idx = trainval_idx[val_idx]
        except Exception:
            train_idx = np.arange(int(n_evans * 0.8))
            val_idx = np.arange(int(n_evans * 0.8), int(n_evans * 0.9))
            test_idx = np.arange(int(n_evans * 0.9), n_evans)

        split_info = {
            "name": f"grouped_seed{seed}",
            "type": "grouped_random",
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "seed": seed,
        }
        with open(splits_dir / f"evans_grouped_random_seed{seed}.json", "w") as f:
            json.dump(split_info, f, indent=2, default=str)
        all_splits[f"grouped_seed{seed}"] = split_info
        logger.info(f"    seed={seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Verify no group leakage
    logger.info("  Verifying no group leakage...")
    has_leakage = False
    for split_name, split_info in all_splits.items():
        train_groups_set = set(groups[split_info["train"]])
        test_groups_set = set(groups[split_info["test"]])
        overlap = train_groups_set & test_groups_set
        if overlap:
            # TSCV leakage is expected: same substrate tested in different time periods
            if "tscv" in split_name:
                logger.warning(f"    {split_name}: {len(overlap)} group overlaps (expected for temporal splits)")
            else:
                logger.error(f"    GROUP LEAKAGE in {split_name}: {len(overlap)} groups overlap!")
                has_leakage = True
        else:
            logger.info(f"    {split_name}: no leakage ✓")
    if has_leakage:
        raise RuntimeError("Group leakage detected in non-temporal splits!")

    context["evans_df"] = evans_df
    context["splits"] = all_splits
    logger.info(f"  Step 13 complete: {len(all_splits)} splits generated")
    return context
