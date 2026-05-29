"""Step 14: Row-level audit report — every row from original 4751 documented."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run(context: dict) -> dict:
    """Build and save comprehensive row-level audit report."""
    audit = context["audit"]
    df = context["df"]
    n_original = len(audit.df)
    logger.info(f"Step 14: Building audit report for {n_original} original rows")

    # Get kept original indices
    kept_oi = set(df["original_index"].values)
    evans_mask = df["Reaction_Class"] == "EvansAux"
    evans_oi = set(df.loc[evans_mask, "original_index"].values)

    # Finalize audit
    audit_df = audit.finalize(kept_oi, evans_oi)

    # Summary statistics
    summary = {
        "total_input": n_original,
        "total_kept": len(kept_oi),
        "total_deleted": n_original - len(kept_oi),
        "evans_v3": (audit_df["final_set"] == "evans_v3").sum(),
        "non_evans_v3": (audit_df["final_set"] == "non_evans_v3").sum(),
        "deleted": (audit_df["final_set"] == "deleted").sum(),
    }

    # Deletion reasons breakdown
    deleted_mask = audit_df["final_set"] == "deleted"
    if deleted_mask.sum() > 0:
        reason_counts = audit_df.loc[deleted_mask, "deletion_reason"].value_counts().to_dict()
        summary["deletion_reasons"] = reason_counts
    else:
        summary["deletion_reasons"] = {}

    # Per-step waterfall
    waterfall = {}
    reasons_ordered = [
        "unparseable_product", "insufficient_stereocenters",
        "cip_label_mismatch", "exact_duplicate",
        "conformer_generation_failed", "steric_computation_failed",
        "feature_nan",
    ]
    for reason in reasons_ordered:
        cnt = (audit_df["deletion_reason"] == reason).sum()
        if cnt > 0:
            waterfall[reason] = int(cnt)
    summary["deletion_waterfall"] = waterfall

    # Save audit CSV
    audit_dir: Path = context["output_dir"] / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_path = audit_dir / "row_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    logger.info(f"  Saved row-level audit to {audit_path}")

    # Save summary JSON
    summary_json_path = audit_dir / "summary_report.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save summary text
    summary_txt_path = audit_dir / "summary_report.txt"
    with open(summary_txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("V3 Data Rebuild — Audit Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total input rows: {summary['total_input']}\n")
        f.write(f"Total kept:       {summary['total_kept']}\n")
        f.write(f"  Evans V3:       {summary['evans_v3']}\n")
        f.write(f"  Non-Evans V3:   {summary['non_evans_v3']}\n")
        f.write(f"Total deleted:    {summary['deleted']}\n\n")
        f.write("Deletion waterfall:\n")
        cumulative = summary["total_input"]
        for reason, cnt in waterfall.items():
            cumulative -= cnt
            f.write(f"  {reason}: -{cnt} (remaining: {cumulative})\n")
        f.write("\n")
        f.write("Deletion reasons:\n")
        for reason, cnt in summary.get("deletion_reasons", {}).items():
            f.write(f"  {reason}: {cnt}\n")

    logger.info(f"  Summary: {summary['evans_v3']} Evans + {summary['non_evans_v3']} non-Evans kept")
    logger.info(f"  Deleted: {summary['deleted']} rows")
    for reason, cnt in waterfall.items():
        logger.info(f"    {reason}: {cnt}")

    logger.info(f"  Step 14 complete")
    context["audit_summary"] = summary
    return context
