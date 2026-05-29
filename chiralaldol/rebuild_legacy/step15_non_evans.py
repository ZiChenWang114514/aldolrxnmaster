"""Step 15: Non-Evans parallel processing — save as separate dataset."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run(context: dict) -> dict:
    """Filter and save non-Evans reactions as independent dataset."""
    df: pd.DataFrame = context["df"]
    date_suffix = context.get("date_suffix", "20260515")
    n = len(df)
    logger.info(f"Step 15: Non-Evans processing from {n} total rows")

    non_evans_mask = df["Reaction_Class"] != "EvansAux"
    non_evans_df = df[non_evans_mask].reset_index(drop=True)
    n_ne = len(non_evans_df)
    logger.info(f"  Non-Evans subset: {n_ne} rows")

    if n_ne > 0:
        # Log class distribution
        if "Reaction_Class" in non_evans_df.columns:
            dist = non_evans_df["Reaction_Class"].value_counts()
            for cls, cnt in dist.items():
                logger.info(f"    {cls}: {cnt}")

        # Save
        out_path = context["output_dir"] / f"non_evans_clean_{date_suffix}.csv"
        non_evans_df.to_csv(out_path, index=False)
        logger.info(f"  Saved to {out_path}")

    context["non_evans_df"] = non_evans_df
    logger.info(f"  Step 15 complete: {n_ne} non-Evans rows saved")
    return context
