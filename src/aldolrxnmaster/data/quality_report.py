"""Step 7: Generate comprehensive data quality report.

Produces both a human-readable text report and a machine-readable JSON
summarizing the entire cleaning pipeline.

Input:  data/raw/*.csv + data/interim/*.csv + data/processed/splits/*.json
Output: data/quality_report/report.txt + report.json
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run(project_root: Path = Path(".")) -> dict:
    raw_dir = project_root / "data" / "raw"
    interim_dir = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    report_dir = project_root / "data" / "quality_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    lines = ["=" * 70, "  AldolRxnMaster — Data Quality Report", "=" * 70, ""]

    # --- Raw data summary ---
    lines.append("## 1. RAW DATA SUMMARY")
    raw_alldata = pd.read_csv(raw_dir / "alldata.csv")
    report["raw"] = {
        "alldata_rows": len(raw_alldata),
        "alldata_columns": len(raw_alldata.columns),
        "reaction_classes": raw_alldata["Reaction_Class"].value_counts().to_dict(),
        "year_range": [int(raw_alldata["Year"].min()), int(raw_alldata["Year"].max())],
    }
    lines.append(f"  alldata.csv: {len(raw_alldata)} rows, {len(raw_alldata.columns)} columns")
    lines.append(f"  Year range: {int(raw_alldata['Year'].min())} - {int(raw_alldata['Year'].max())}")
    lines.append(f"  Reaction classes:")
    for cls, cnt in raw_alldata["Reaction_Class"].value_counts().items():
        lines.append(f"    {cls}: {cnt}")

    missing = raw_alldata.isnull().sum()
    lines.append(f"  Missing values:")
    for col, n in missing[missing > 0].items():
        lines.append(f"    {col}: {n} ({n/len(raw_alldata):.1%})")
    lines.append("")

    # --- Pipeline step summaries ---
    steps = [
        ("01_consolidated.csv", "01_consolidation_log.txt", "CONSOLIDATION"),
        ("02_deduplicated.csv", "02_dedup_log.txt", "DEDUPLICATION"),
        ("03_validated.csv", "03_validation_log.txt", "SMILES VALIDATION"),
        ("04_labels_unified.csv", "04_label_unification_log.txt", "LABEL UNIFICATION"),
        ("05_imputed.csv", "05_imputation_log.txt", "IMPUTATION"),
    ]

    for csv_name, log_name, title in steps:
        csv_path = interim_dir / csv_name
        log_path = interim_dir / log_name

        if not csv_path.exists():
            lines.append(f"## {title}: SKIPPED (file not found)")
            continue

        df = pd.read_csv(csv_path)
        lines.append(f"## {title}")
        lines.append(f"  Output: {csv_name} — {len(df)} rows, {len(df.columns)} columns")

        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")
            for log_line in log_content.split("\n"):
                lines.append(f"  {log_line}")

        report[csv_name] = {
            "rows": len(df),
            "columns": len(df.columns),
        }
        lines.append("")

    # --- Final cleaned data ---
    final_path = interim_dir / "05_imputed.csv"
    if final_path.exists():
        df_final = pd.read_csv(final_path)
        lines.append("## FINAL CLEANED DATA")
        lines.append(f"  Total rows: {len(df_final)}")

        if "label_joint" in df_final.columns:
            joint_dist = df_final["label_joint"].value_counts().sort_index()
            lines.append(f"  Joint 4-class distribution:")
            class_names = {0: "Ca=0,Cb=0", 1: "Ca=0,Cb=1", 2: "Ca=1,Cb=0", 3: "Ca=1,Cb=1"}
            for cls, cnt in joint_dist.items():
                pct = cnt / len(df_final) * 100
                lines.append(f"    Class {int(cls)} ({class_names.get(int(cls), '?')}): {cnt} ({pct:.1f}%)")

            report["final"] = {
                "rows": len(df_final),
                "joint_distribution": {str(int(k)): int(v) for k, v in joint_dist.items()},
            }

        # Evans subset
        evans = df_final[df_final["Reaction_Class"] == "EvansAux"]
        lines.append(f"\n  Evans subset: {len(evans)} rows")
        if "label_joint" in evans.columns:
            evans_dist = evans["label_joint"].value_counts().sort_index()
            for cls, cnt in evans_dist.items():
                pct = cnt / len(evans) * 100
                lines.append(f"    Class {int(cls)}: {cnt} ({pct:.1f}%)")

        # Missing value audit
        lines.append(f"\n  Remaining missing values:")
        for col in df_final.columns:
            n = df_final[col].isna().sum()
            if n > 0:
                lines.append(f"    {col}: {n} ({n/len(df_final):.1%})")

        n_label_na = df_final["label_joint"].isna().sum() if "label_joint" in df_final.columns else 0
        n_smiles_invalid = (~df_final["smiles_valid"]).sum() if "smiles_valid" in df_final.columns else 0

        lines.append(f"\n  QUALITY CHECKS:")
        lines.append(f"    Labels with NaN: {n_label_na}")
        lines.append(f"    Invalid SMILES: {n_smiles_invalid}")
        lines.append("")

    # --- Split summaries ---
    splits_dir = processed_dir / "splits"
    if splits_dir.exists():
        lines.append("## SPLIT SUMMARIES")
        split_report = {}
        for json_file in sorted(splits_dir.glob("*.json")):
            with open(json_file) as f:
                split_data = json.load(f)

            name = json_file.stem
            lines.append(f"\n  {name}:")
            for fold in ["train", "val", "test"]:
                stats = split_data.get(f"{fold}_stats", {})
                n = stats.get("n_samples", 0)
                n_groups = stats.get("n_groups", 0)
                dist = stats.get("class_distribution", {})
                dist_str = ", ".join(f"C{k}:{v}" for k, v in sorted(dist.items()))
                lines.append(f"    {fold}: {n} samples, {n_groups} groups | {dist_str}")

            split_report[name] = {
                fold: split_data.get(f"{fold}_stats", {}) for fold in ["train", "val", "test"]
            }

        report["splits"] = split_report

    lines.append("")
    lines.append("=" * 70)
    lines.append("  Report generated by AldolRxnMaster data pipeline")
    lines.append("=" * 70)

    # Save
    report_text = "\n".join(lines)
    (report_dir / "report.txt").write_text(report_text, encoding="utf-8")
    with open(report_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Quality report saved to {report_dir}")
    print(report_text)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
