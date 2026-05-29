"""Step 3: Stereocenter validation — strict deletion of products with <2 defined stereocenters."""

import logging

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)


def _count_defined_stereocenters(smi: str) -> int:
    """Count defined (R/S) stereocenters in a SMILES string."""
    if pd.isna(smi) or not str(smi).strip():
        return -1
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return -1
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return sum(1 for _, cip in centers if cip in ("R", "S"))


def run(context: dict) -> dict:
    """Validate stereocenter count, delete rows with <2 defined stereocenters."""
    df: pd.DataFrame = context["df"].copy()
    audit = context["audit"]
    n_start = len(df)
    logger.info(f"Step 3: Stereocenter validation for {n_start} rows")

    # Use canonical product if available, else raw
    product_col = "canonical_Raw_Product_Smiles"
    if product_col not in df.columns:
        product_col = "Raw_Product_Smiles"

    df["n_defined_stereocenters"] = df[product_col].apply(_count_defined_stereocenters)

    # Log distribution
    dist = df["n_defined_stereocenters"].value_counts().sort_index()
    logger.info(f"  Stereocenter distribution:")
    for n_sc, cnt in dist.items():
        logger.info(f"    {n_sc} centers: {cnt} ({100*cnt/n_start:.1f}%)")

    # Log by Reaction_Class
    if "Reaction_Class" in df.columns:
        for cls in df["Reaction_Class"].unique():
            cls_mask = df["Reaction_Class"] == cls
            cls_below2 = (df.loc[cls_mask, "n_defined_stereocenters"] < 2).sum()
            cls_total = cls_mask.sum()
            logger.info(f"    {cls}: {cls_below2}/{cls_total} with <2 stereocenters ({100*cls_below2/max(cls_total,1):.1f}%)")

    # Record in audit
    # Map back to original indices for audit tracking
    for i, row in df.iterrows():
        oi = row["original_index"]
        mask_oi = audit.df["original_index"] == oi
        if mask_oi.any():
            audit.df.loc[mask_oi, "n_defined_stereocenters"] = row["n_defined_stereocenters"]

    # Strict deletion: <2 stereocenters
    bad_mask = df["n_defined_stereocenters"] < 2
    n_bad = bad_mask.sum()
    if n_bad > 0:
        audit.mark_deleted_by_oi(df.loc[bad_mask, "original_index"].values, "insufficient_stereocenters")
        df = df[~bad_mask].reset_index(drop=True)
        logger.info(f"  Deleted {n_bad} rows with <2 defined stereocenters")

    n_end = len(df)
    logger.info(f"  Step 3 complete: {n_start} → {n_end} rows")

    out_path = context["output_dir"] / "interim" / "03_stereo_validated.csv"
    df.to_csv(out_path, index=False)

    context["df"] = df
    return context
