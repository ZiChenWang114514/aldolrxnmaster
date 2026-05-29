#!/usr/bin/env python3
"""V4 Data Splitting: TSCV + Scaffold + Grouped Random.

Usage:
    conda run -n aldol-rxn python scripts/run_splits_v4.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import CLEAN_DIR, SPLITS_DIR

CLEAN_CSV = CLEAN_DIR / "substrate_aldol_clean.csv"


def extract_year(ref_str: str) -> int:
    """Extract publication year from Reaxys References field."""
    if not isinstance(ref_str, str):
        return 2010  # default
    years = re.findall(r"\((\d{4})\)", ref_str)
    if years:
        return max(int(y) for y in years)
    years = re.findall(r";\s*(\d{4})\s*;", ref_str)
    if years:
        return max(int(y) for y in years)
    return 2010


def get_murcko_scaffold(smi: str) -> str:
    """Get Murcko scaffold from SMILES."""
    if not isinstance(smi, str) or not smi.strip():
        return "unknown"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "unknown"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return "unknown"


def generate_tscv_splits(df: pd.DataFrame, n_folds: int = 4):
    """Generate expanding-window temporal CV splits.

    Each fold uses all data before a cutoff year for training,
    a small window for validation, and remaining for testing.
    Cutoffs chosen so test sets are roughly balanced.
    """
    years = df["year"].values
    # Use fixed cutoff years for reproducible, balanced splits
    cutoffs = [
        {"train_max": 2005, "val_max": 2009, "test_min": 2010},
        {"train_max": 2009, "val_max": 2013, "test_min": 2014},
        {"train_max": 2013, "val_max": 2017, "test_min": 2018},
        {"train_max": 2017, "val_max": 2020, "test_min": 2021},
    ]

    splits = []
    for fold, c in enumerate(cutoffs):
        train_idx = [i for i, y in enumerate(years) if y <= c["train_max"]]
        val_idx = [i for i, y in enumerate(years) if c["train_max"] < y <= c["val_max"]]
        test_idx = [i for i, y in enumerate(years) if y >= c["test_min"]]

        if not train_idx or not val_idx or not test_idx:
            continue

        splits.append({
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        })
        print(f"  Fold {fold + 1}: train={len(train_idx)} (≤{c['train_max']}), "
              f"val={len(val_idx)} ({c['train_max']+1}-{c['val_max']}), "
              f"test={len(test_idx)} (≥{c['test_min']})")

    return splits


def generate_scaffold_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.15):
    """Generate scaffold-based split using aldehyde Murcko scaffolds."""
    scaffolds = df["canonical_aldehyde_smiles"].apply(get_murcko_scaffold)
    groups = pd.Categorical(scaffolds).codes

    # First split: test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    trainval_idx, test_idx = next(gss.split(df, groups=groups))

    # Second split: val from trainval
    val_frac = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    sub_groups = groups[trainval_idx]
    train_sub, val_sub = next(gss2.split(trainval_idx, groups=sub_groups))

    train_idx = trainval_idx[train_sub].tolist()
    val_idx = trainval_idx[val_sub].tolist()
    test_idx = test_idx.tolist()

    print(f"  Scaffold: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def generate_grouped_splits(df: pd.DataFrame, seeds: list[int]):
    """Generate grouped random splits preserving substrate pair integrity."""
    groups = df["group_id"].values
    splits = {}

    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval_idx, test_idx = next(gss.split(df, groups=groups))

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        sub_groups = groups[trainval_idx]
        train_sub, val_sub = next(gss2.split(trainval_idx, groups=sub_groups))

        train_idx = trainval_idx[train_sub].tolist()
        val_idx = trainval_idx[val_sub].tolist()
        test_idx = test_idx.tolist()

        splits[seed] = {"train": train_idx, "val": val_idx, "test": test_idx}
        print(f"  Grouped seed={seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return splits


def _to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def verify_splits(df, all_splits: dict):
    """Verify split integrity."""
    n = len(df)
    groups = df["group_id"].values
    errors = []

    for name, split in all_splits.items():
        train, val, test = set(split["train"]), set(split["val"]), set(split["test"])

        # No overlap
        if train & val:
            errors.append(f"{name}: train/val overlap ({len(train & val)})")
        if train & test:
            errors.append(f"{name}: train/test overlap ({len(train & test)})")
        if val & test:
            errors.append(f"{name}: val/test overlap ({len(val & test)})")

        # Full coverage
        total = train | val | test
        if len(total) != n:
            errors.append(f"{name}: missing {n - len(total)} rows")

        # Group integrity (no group split across train/test)
        train_groups = set(groups[list(train)])
        test_groups = set(groups[list(test)])
        leaked = train_groups & test_groups
        if leaked:
            errors.append(f"{name}: {len(leaked)} group_ids leaked between train/test")

    if errors:
        print("  SPLIT ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  ✓ All splits verified: no overlap, full coverage, no group leakage")


def main():
    print("=" * 60)
    print("V4 DATA SPLITTING")
    print("=" * 60)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df)} rows")

    # Extract year
    df["year"] = df["References"].apply(extract_year)
    print(f"Year range: {df['year'].min()}-{df['year'].max()}")

    all_splits = {}

    # --- TSCV ---
    print("\n--- TSCV (4-fold temporal) ---")
    tscv_splits = generate_tscv_splits(df, n_folds=4)
    for i, split in enumerate(tscv_splits):
        name = f"tscv_fold{i + 1}"
        all_splits[name] = split
        with open(SPLITS_DIR / f"{name}.json", "w") as f:
            json.dump(_to_native(split), f)

    # --- Scaffold ---
    print("\n--- Scaffold ---")
    scaffold_split = generate_scaffold_split(df)
    all_splits["scaffold"] = scaffold_split
    with open(SPLITS_DIR / "scaffold.json", "w") as f:
        json.dump(_to_native(scaffold_split), f)

    # --- Grouped ---
    print("\n--- Grouped Random ---")
    seeds = [42, 123, 456, 789, 1024]
    grouped_splits = generate_grouped_splits(df, seeds)
    for seed, split in grouped_splits.items():
        name = f"grouped_seed{seed}"
        all_splits[name] = split
        with open(SPLITS_DIR / f"{name}.json", "w") as f:
            json.dump(_to_native(split), f)

    # --- Verify ---
    print("\n--- Verification ---")
    verify_splits(df, all_splits)

    print(f"\nSaved {len(all_splits)} split files to {SPLITS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
