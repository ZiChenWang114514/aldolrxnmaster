"""Step 1: Raw data loading + initial audit."""

import logging
from pathlib import Path

import pandas as pd

from .logger import AuditTracker

logger = logging.getLogger(__name__)


def run(context: dict) -> dict:
    """Load alldata.csv, initialize audit tracker, log statistics."""
    raw_path: Path = context["raw_dir"] / "alldata.csv"
    logger.info(f"Step 1: Loading {raw_path}")

    df = pd.read_csv(raw_path)
    n = len(df)
    logger.info(f"  Loaded {n} rows, {len(df.columns)} columns")
    logger.info(f"  Columns: {list(df.columns)}")

    # Add original index
    df["original_index"] = range(n)

    # Log missing values
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            logger.info(f"  Missing {col}: {missing[col]} ({100*missing[col]/n:.1f}%)")

    # Log class distribution
    if "Reaction_Class" in df.columns:
        logger.info(f"  Reaction_Class distribution:")
        for cls, cnt in df["Reaction_Class"].value_counts().items():
            logger.info(f"    {cls}: {cnt} ({100*cnt/n:.1f}%)")

    # Log label distribution
    for lbl in ["label_Ca", "label_Cb", "label_SA"]:
        if lbl in df.columns:
            dist = df[lbl].value_counts().to_dict()
            logger.info(f"  {lbl}: {dist}")

    # Log year range
    if "Year" in df.columns:
        logger.info(f"  Year range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")

    # Initialize audit tracker
    audit = AuditTracker(n)
    for col in ["Reaction_ID", "Reaction_Class", "Year"]:
        if col in df.columns:
            audit.add_column(col, df[col].values)

    # Record per-row SMILES availability
    for col in ["Raw_Product_Smiles", "Ketone", "Aldehyde", "Mapped_Reaction"]:
        audit.add_column(f"has_{col}", df[col].notna().values)

    # Save interim
    out_path = context["output_dir"] / "interim" / "01_raw_loaded.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved to {out_path}")

    context["df"] = df
    context["audit"] = audit
    return context
