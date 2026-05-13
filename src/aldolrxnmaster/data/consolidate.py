"""Step 1: Consolidate all raw CSVs into a single source of truth.

Takes alldata.csv as the primary table and enriches it with mapped SMILES
from evans_aux files where alldata has NaN. Drops dead columns, standardizes
column names, and flags label conflicts.

Input:  data/raw/*.csv
Output: data/interim/01_consolidated.csv + audit log
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/01_consolidated.csv")
LOG_PATH = Path("data/interim/01_consolidation_log.txt")

# Columns to drop (dead or redundant)
DROP_COLS = [
    "is_chiral",   # all zeros
    "Config",       # redundant with label_Ca/Cb/SA
]


def load_alldata() -> pd.DataFrame:
    """Load the master alldata.csv."""
    df = pd.read_csv(RAW_DIR / "alldata.csv")
    logger.info(f"Loaded alldata.csv: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_evans_aux() -> pd.DataFrame:
    """Load evans_aux.csv for cross-referencing mapped SMILES."""
    df = pd.read_csv(RAW_DIR / "evans_aux.csv")
    logger.info(f"Loaded evans_aux.csv: {len(df)} rows")
    return df


def load_evans_mapped() -> pd.DataFrame:
    """Load evans_aux_mapped.csv — the most complete Evans subset."""
    df = pd.read_csv(RAW_DIR / "evans_aux_mapped.csv")
    logger.info(f"Loaded evans_aux_mapped.csv: {len(df)} rows")
    return df


def drop_dead_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with no information content."""
    existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")
    return df


def fill_missing_mapped_smiles(
    alldata: pd.DataFrame,
    evans_aux: pd.DataFrame,
    evans_mapped: pd.DataFrame,
) -> pd.DataFrame:
    """Fill NaN Mapped_Reaction/Ketone/Aldehyde in alldata using Evans subsets."""
    log_lines = []

    # Build a lookup: Reaction_ID -> mapped SMILES from evans files
    # evans_mapped has the most rows; evans_aux may have additional unique entries
    lookup = {}

    for _, row in evans_mapped.iterrows():
        rid = row["ID"]
        if pd.notna(row.get("mapped_reaction")):
            lookup.setdefault(rid, {})
            lookup[rid]["mapped_reaction"] = row["mapped_reaction"]
        if pd.notna(row.get("ketone")):
            lookup[rid]["ketone"] = row["ketone"]
        if pd.notna(row.get("aldehyde")):
            lookup[rid]["aldehyde"] = row["aldehyde"]
        if pd.notna(row.get("product_")):
            lookup[rid]["mapped_product"] = row["product_"]

    for _, row in evans_aux.iterrows():
        rid = row["ID"]
        if rid not in lookup and pd.notna(row.get("mapped_reaction")):
            lookup[rid] = {
                "mapped_reaction": row["mapped_reaction"],
                "ketone": row.get("ketone"),
                "aldehyde": row.get("aldehyde"),
                "mapped_product": row.get("product_"),
            }

    filled_count = 0
    for idx, row in alldata.iterrows():
        rid = row["Reaction_ID"]
        if pd.isna(row["Mapped_Reaction"]) and rid in lookup:
            info = lookup[rid]
            if "mapped_reaction" in info and pd.notna(info["mapped_reaction"]):
                alldata.at[idx, "Mapped_Reaction"] = info["mapped_reaction"]
                filled_count += 1
                log_lines.append(
                    f"Filled Mapped_Reaction for Reaction_ID={rid} from Evans subset"
                )
            if pd.isna(row["Ketone"]) and info.get("ketone") and pd.notna(info["ketone"]):
                alldata.at[idx, "Ketone"] = info["ketone"]
            if pd.isna(row["Aldehyde"]) and info.get("aldehyde") and pd.notna(info["aldehyde"]):
                alldata.at[idx, "Aldehyde"] = info["aldehyde"]
            if (
                pd.isna(row["Mapped_Product"])
                and info.get("mapped_product")
                and pd.notna(info["mapped_product"])
            ):
                alldata.at[idx, "Mapped_Product"] = info["mapped_product"]

    logger.info(f"Filled {filled_count} missing Mapped_Reaction values from Evans subsets")
    return alldata, log_lines


def flag_label_conflicts(
    alldata: pd.DataFrame,
    evans_aux: pd.DataFrame,
) -> pd.DataFrame:
    """Check for label disagreements between alldata and evans_aux.

    The alldata uses {-1, +1} encoding; evans_aux uses {0, 1}.
    Convert evans to {-1, +1} for comparison.
    """
    # Build evans label lookup (ID -> labels in {-1,+1} space)
    evans_labels = {}
    for _, row in evans_aux.iterrows():
        rid = row["ID"]
        # evans uses {0,1} -> convert to {-1,+1}: val * 2 - 1
        ca = row["label_Ca"] * 2 - 1 if pd.notna(row["label_Ca"]) else None
        cb = row["label_Cb"] * 2 - 1 if pd.notna(row["label_Cb"]) else None
        sa = row["label_SA"] * 2 - 1 if pd.notna(row["label_SA"]) else None
        evans_labels[rid] = (ca, cb, sa)

    conflicts = []
    alldata["label_conflict"] = False

    for idx, row in alldata.iterrows():
        rid = row["Reaction_ID"]
        if rid not in evans_labels:
            continue

        e_ca, e_cb, e_sa = evans_labels[rid]
        a_ca, a_cb, a_sa = row["label_Ca"], row["label_Cb"], row["label_SA"]

        has_conflict = False
        if e_ca is not None and pd.notna(a_ca) and e_ca != a_ca:
            has_conflict = True
        if e_cb is not None and pd.notna(a_cb) and e_cb != a_cb:
            has_conflict = True
        if e_sa is not None and pd.notna(a_sa) and e_sa != a_sa:
            has_conflict = True

        if has_conflict:
            alldata.at[idx, "label_conflict"] = True
            conflicts.append(
                f"Reaction_ID={rid}: alldata=(Ca={a_ca},Cb={a_cb},SA={a_sa}) "
                f"vs evans=(Ca={e_ca},Cb={e_cb},SA={e_sa})"
            )

    logger.info(f"Found {len(conflicts)} label conflicts between alldata and evans_aux")
    return alldata, conflicts


def clean_solvent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate the two solvent columns (Solvent and solvent).

    Keep the cleaned lowercase `solvent` column (less duplicated entries).
    Rename to `solvent_clean`. Drop the raw `Solvent` column.
    """
    if "solvent" in df.columns and "Solvent" in df.columns:
        df = df.rename(columns={"solvent": "solvent_clean"})
        df = df.drop(columns=["Solvent"])
        logger.info("Consolidated solvent columns: kept 'solvent_clean', dropped 'Solvent'")
    return df


def run(project_root: Path = Path(".")) -> pd.DataFrame:
    """Execute the full consolidation pipeline."""
    global RAW_DIR, OUT_PATH, LOG_PATH
    RAW_DIR = project_root / "data" / "raw"
    OUT_PATH = project_root / "data" / "interim" / "01_consolidated.csv"
    LOG_PATH = project_root / "data" / "interim" / "01_consolidation_log.txt"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load all sources
    alldata = load_alldata()
    evans_aux = load_evans_aux()
    evans_mapped = load_evans_mapped()

    all_log_lines = []

    # Step 1: Drop dead columns
    alldata = drop_dead_columns(alldata)

    # Step 2: Fill missing mapped SMILES from Evans subsets
    alldata, fill_logs = fill_missing_mapped_smiles(alldata, evans_aux, evans_mapped)
    all_log_lines.extend(fill_logs)

    # Step 3: Flag label conflicts
    alldata, conflict_logs = flag_label_conflicts(alldata, evans_aux)
    all_log_lines.extend(conflict_logs)

    # Step 4: Clean solvent columns
    alldata = clean_solvent_columns(alldata)

    # Step 5: Standardize Year to int (some are float due to NaN in original)
    if alldata["Year"].notna().all():
        alldata["Year"] = alldata["Year"].astype(int)
    else:
        n_missing_year = alldata["Year"].isna().sum()
        logger.warning(f"{n_missing_year} rows missing Year — keeping as float")

    # Summary stats
    n_total = len(alldata)
    n_evans = (alldata["Reaction_Class"] == "EvansAux").sum()
    n_mapped_na = alldata["Mapped_Reaction"].isna().sum()
    n_ketone_na = alldata["Ketone"].isna().sum()
    n_conflicts = alldata["label_conflict"].sum()

    summary = [
        f"=== Consolidation Summary ===",
        f"Total rows: {n_total}",
        f"Evans rows: {n_evans}",
        f"Remaining NaN Mapped_Reaction: {n_mapped_na}",
        f"Remaining NaN Ketone: {n_ketone_na}",
        f"Label conflicts flagged: {n_conflicts}",
        f"Columns: {list(alldata.columns)}",
        f"Output: {OUT_PATH}",
    ]
    for line in summary:
        logger.info(line)
    all_log_lines.extend(summary)

    # Save
    alldata.to_csv(OUT_PATH, index=False)
    LOG_PATH.write_text("\n".join(all_log_lines), encoding="utf-8")

    logger.info(f"Saved consolidated data to {OUT_PATH}")
    logger.info(f"Saved audit log to {LOG_PATH}")
    return alldata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
