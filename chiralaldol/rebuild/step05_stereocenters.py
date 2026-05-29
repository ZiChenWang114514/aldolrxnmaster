"""Step 05: Filter by stereocenter count (>=2 defined R/S in product)."""

import logging

import pandas as pd

from .audit import AuditTracker
from .utils import safe_mol, count_defined_stereocenters

logger = logging.getLogger("rebuild_v4.step05")

MIN_STEREOCENTERS = 2


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Keep only reactions whose main product has >= 2 defined stereocenters."""
    logger.info("Step 05: Filtering by stereocenter count...")
    n_start = len(df)

    # Use canonical product if available, else raw
    prod_col = "canonical_main_product_smiles" if "canonical_main_product_smiles" in df.columns else "main_product_smiles"

    df["n_defined_stereocenters"] = df[prod_col].apply(
        lambda s: count_defined_stereocenters(safe_mol(s)) if isinstance(s, str) else 0
    )

    # Log distribution
    dist = df["n_defined_stereocenters"].value_counts().sort_index()
    logger.info(f"  Stereocenter distribution:")
    for n, count in dist.items():
        logger.info(f"    {n} centers: {count} rows")

    # Filter
    insufficient = df["n_defined_stereocenters"] < MIN_STEREOCENTERS
    audit.record_drop("05_stereocenters", df.loc[insufficient, "_orig_idx"],
                       f"stereocenters<{MIN_STEREOCENTERS}")
    df = df[~insufficient].reset_index(drop=True)

    audit.record_step("05_stereocenters", len(df))
    logger.info(f"  Step 05 complete: {n_start} -> {len(df)} rows")
    return df
