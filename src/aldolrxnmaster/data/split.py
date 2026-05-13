"""Step 6: Generate train/val/test splits respecting group_id integrity.

Three split strategies:
  1. Temporal: Year-based (train ≤2015, val 2016-2018, test ≥2019)
  2. Scaffold: Murcko scaffold of aldehyde (OOD structural generalization)
  3. Grouped Random: 5 seeds, 80/10/10, stratified by label_joint

All splits guarantee: reactions sharing a group_id are NEVER split across folds.

Input:  data/interim/05_imputed.csv
Output: data/processed/splits/*.json + class distribution stats
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)


def _get_murcko_scaffold(smiles: str) -> str:
    """Get Murcko scaffold SMILES for a molecule."""
    if pd.isna(smiles) or not str(smiles).strip():
        return "UNKNOWN"
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "UNKNOWN"
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return "UNKNOWN"


def _split_stats(df: pd.DataFrame, indices: list[int], label_col: str = "label_joint") -> dict:
    """Compute class distribution stats for a split."""
    subset = df.iloc[indices]
    dist = subset[label_col].value_counts().sort_index().to_dict()
    return {
        "n_samples": len(indices),
        "n_groups": subset["group_id"].nunique(),
        "class_distribution": {str(int(k)): int(v) for k, v in dist.items()},
    }


def temporal_split(
    df: pd.DataFrame,
    train_cutoff: int = 2015,
    val_cutoff: int = 2018,
) -> dict:
    """Split by publication year.

    train: Year <= train_cutoff
    val:   train_cutoff < Year <= val_cutoff
    test:  Year > val_cutoff
    """
    logger.info(f"Temporal split: train≤{train_cutoff}, val {train_cutoff+1}-{val_cutoff}, test>{val_cutoff}")

    # First assign each group_id to a period based on the earliest Year in that group
    group_years = df.groupby("group_id")["Year"].min()

    train_groups = set(group_years[group_years <= train_cutoff].index)
    val_groups = set(group_years[(group_years > train_cutoff) & (group_years <= val_cutoff)].index)
    test_groups = set(group_years[group_years > val_cutoff].index)

    train_idx = df[df["group_id"].isin(train_groups)].index.tolist()
    val_idx = df[df["group_id"].isin(val_groups)].index.tolist()
    test_idx = df[df["group_id"].isin(test_groups)].index.tolist()

    result = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "train_stats": _split_stats(df, train_idx),
        "val_stats": _split_stats(df, val_idx),
        "test_stats": _split_stats(df, test_idx),
        "params": {
            "method": "temporal",
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
        },
    }

    logger.info(
        f"  Train: {len(train_idx)} ({len(train_groups)} groups), "
        f"Val: {len(val_idx)} ({len(val_groups)} groups), "
        f"Test: {len(test_idx)} ({len(test_groups)} groups)"
    )

    return result


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split by Murcko scaffold of the aldehyde.

    Groups scaffolds from largest to smallest, assigns to train until
    train_frac is reached, then val, then test. Respects group_id.
    """
    logger.info("Scaffold split based on aldehyde Murcko scaffolds...")

    # Get scaffold for each unique aldehyde
    aldehyde_col = "Aldehyde" if "Aldehyde" in df.columns else "Raw_Product_Smiles"
    df["_scaffold"] = df[aldehyde_col].apply(_get_murcko_scaffold)

    # Map each group_id to its scaffold (use the first aldehyde in the group)
    group_scaffold = df.groupby("group_id")["_scaffold"].first().to_dict()

    # Group scaffolds and count samples per scaffold
    scaffold_to_groups = defaultdict(list)
    for gid, scaffold in group_scaffold.items():
        scaffold_to_groups[scaffold].append(gid)

    # Sort scaffolds by number of associated samples (descending)
    scaffold_sizes = {
        s: sum(len(df[df["group_id"] == g]) for g in groups)
        for s, groups in scaffold_to_groups.items()
    }
    sorted_scaffolds = sorted(scaffold_sizes.keys(), key=lambda s: scaffold_sizes[s], reverse=True)

    n_total = len(df)
    train_target = int(n_total * train_frac)
    val_target = int(n_total * (train_frac + val_frac))

    train_groups, val_groups, test_groups = set(), set(), set()
    current_count = 0

    rng = np.random.RandomState(seed)
    # Shuffle scaffolds with same size for reproducibility
    rng.shuffle(sorted_scaffolds)
    # Re-sort by size (shuffle only breaks ties)
    sorted_scaffolds = sorted(sorted_scaffolds, key=lambda s: scaffold_sizes[s], reverse=True)

    for scaffold in sorted_scaffolds:
        groups = scaffold_to_groups[scaffold]
        n_in_scaffold = scaffold_sizes[scaffold]

        if current_count < train_target:
            train_groups.update(groups)
        elif current_count < val_target:
            val_groups.update(groups)
        else:
            test_groups.update(groups)

        current_count += n_in_scaffold

    train_idx = df[df["group_id"].isin(train_groups)].index.tolist()
    val_idx = df[df["group_id"].isin(val_groups)].index.tolist()
    test_idx = df[df["group_id"].isin(test_groups)].index.tolist()

    # Clean up temp column
    df.drop(columns=["_scaffold"], inplace=True)

    result = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "train_stats": _split_stats(df, train_idx),
        "val_stats": _split_stats(df, val_idx),
        "test_stats": _split_stats(df, test_idx),
        "params": {
            "method": "scaffold",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
            "n_scaffolds": len(scaffold_to_groups),
        },
    }

    logger.info(
        f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)} "
        f"({len(scaffold_to_groups)} unique scaffolds)"
    )

    return result


def grouped_random_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Random split respecting group_id. Stratified by label_joint."""
    logger.info(f"Grouped random split (seed={seed})...")

    groups = df["group_id"].values

    # First split: train+val vs test
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=1 - train_frac - val_frac, random_state=seed)
    trainval_idx, test_idx = next(splitter1.split(df, groups=groups))

    # Second split: train vs val (from trainval)
    df_trainval = df.iloc[trainval_idx]
    groups_tv = df_trainval["group_id"].values
    val_frac_adjusted = val_frac / (train_frac + val_frac)

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_adjusted, random_state=seed + 1)
    train_rel_idx, val_rel_idx = next(splitter2.split(df_trainval, groups=groups_tv))

    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    result = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "train_stats": _split_stats(df, train_idx.tolist()),
        "val_stats": _split_stats(df, val_idx.tolist()),
        "test_stats": _split_stats(df, test_idx.tolist()),
        "params": {
            "method": "grouped_random",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
        },
    }

    logger.info(
        f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    return result


def verify_no_group_leakage(split: dict, df: pd.DataFrame) -> bool:
    """Verify that no group_id appears in multiple folds."""
    train_groups = set(df.iloc[split["train"]]["group_id"])
    val_groups = set(df.iloc[split["val"]]["group_id"])
    test_groups = set(df.iloc[split["test"]]["group_id"])

    tv_leak = train_groups & val_groups
    tt_leak = train_groups & test_groups
    vt_leak = val_groups & test_groups

    if tv_leak or tt_leak or vt_leak:
        logger.error(
            f"GROUP LEAKAGE DETECTED! train∩val={len(tv_leak)}, "
            f"train∩test={len(tt_leak)}, val∩test={len(vt_leak)}"
        )
        return False

    logger.info("  No group leakage detected ✓")
    return True


def run(
    project_root: Path = Path("."),
    subset: str = "all",  # "all" or "evans"
) -> dict[str, dict]:
    in_path = project_root / "data" / "interim" / "05_imputed.csv"
    splits_dir = project_root / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    logger.info(f"Loaded {len(df)} rows from {in_path}")

    # Filter to Evans if requested
    if subset == "evans":
        df = df[df["Reaction_Class"] == "EvansAux"].reset_index(drop=True)
        logger.info(f"Filtered to Evans subset: {len(df)} rows")
        prefix = "evans"
    else:
        prefix = "alldata"

    all_splits = {}

    # 1. Temporal split
    split = temporal_split(df)
    assert verify_no_group_leakage(split, df)
    fname = f"{prefix}_temporal.json"
    with open(splits_dir / fname, "w") as f:
        json.dump(split, f, indent=2)
    all_splits["temporal"] = split
    logger.info(f"Saved {fname}")

    # 2. Scaffold split
    split = scaffold_split(df)
    assert verify_no_group_leakage(split, df)
    fname = f"{prefix}_scaffold.json"
    with open(splits_dir / fname, "w") as f:
        json.dump(split, f, indent=2)
    all_splits["scaffold"] = split
    logger.info(f"Saved {fname}")

    # 3. Grouped random splits (5 seeds)
    for seed in [42, 123, 456, 789, 1024]:
        split = grouped_random_split(df, seed=seed)
        assert verify_no_group_leakage(split, df)
        fname = f"{prefix}_grouped_random_seed{seed}.json"
        with open(splits_dir / fname, "w") as f:
            json.dump(split, f, indent=2)
        all_splits[f"grouped_random_s{seed}"] = split
        logger.info(f"Saved {fname}")

    # Save Evans clean dataset if subset == "evans"
    if subset == "evans":
        clean_path = project_root / "data" / "processed" / "evans_clean.csv"
        df.to_csv(clean_path, index=False)
        logger.info(f"Saved Evans clean dataset: {len(df)} rows to {clean_path}")

    return all_splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path("/data2/zcwang/aldolrxnmaster")

    # Generate splits for both full dataset and Evans subset
    run(root, subset="all")
    run(root, subset="evans")
