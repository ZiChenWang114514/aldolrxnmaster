"""Step 3: Validate all SMILES and audit chirality information.

For each row:
  - Parse all SMILES columns with RDKit
  - Extract stereocenters from products
  - Verify chirality tags are present and consistent
  - Flag rows with invalid SMILES or missing chirality

Input:  data/interim/02_deduplicated.csv
Output: data/interim/03_validated.csv + audit log
"""

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

# Suppress RDKit warnings during batch processing
RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)

# SMILES columns to validate
SMILES_COLS = [
    "Raw_Reaction_Smiles",
    "Raw_Product_Smiles",
    "Ketone",
    "Aldehyde",
    "Mapped_Reaction",
    "Mapped_Product",
    "Product_",
]


def _parse_smiles(smiles: str) -> Chem.Mol | None:
    """Safely parse a SMILES string."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def _parse_reaction_smiles(rxn: str) -> bool:
    """Check if a reaction SMILES is valid (all components parseable)."""
    if pd.isna(rxn) or not str(rxn).strip():
        return False
    parts = str(rxn).split(">>")
    if len(parts) != 2:
        return False
    for side in parts:
        for smi in side.split("."):
            if smi.strip() and _parse_smiles(smi.strip()) is None:
                return False
    return True


def _count_defined_stereocenters(smiles: str) -> int:
    """Count the number of defined (R/S assigned) stereocenters."""
    mol = _parse_smiles(smiles)
    if mol is None:
        return -1
    try:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        defined = [c for c in chiral_centers if c[1] in ("R", "S")]
        return len(defined)
    except Exception:
        return -1


def _get_stereocenters(smiles: str) -> list[tuple[int, str]]:
    """Get all stereocenters with their R/S assignments."""
    mol = _parse_smiles(smiles)
    if mol is None:
        return []
    try:
        return Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    except Exception:
        return []


def _has_chirality_tags(smiles: str) -> bool:
    """Check if SMILES contains @ or @@ tags."""
    if pd.isna(smiles):
        return False
    return "@" in str(smiles)


def validate_smiles_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate all SMILES columns and add validity flags."""
    for col in SMILES_COLS:
        if col not in df.columns:
            continue

        flag_col = f"{col}_valid"

        if col in ("Raw_Reaction_Smiles", "Mapped_Reaction"):
            # Reaction SMILES: check all components
            df[flag_col] = df[col].apply(_parse_reaction_smiles)
        else:
            # Molecular SMILES
            df[flag_col] = df[col].apply(lambda s: _parse_smiles(s) is not None)

    # Overall validity: at minimum Raw_Reaction_Smiles and Raw_Product_Smiles must be valid
    df["smiles_valid"] = (
        df.get("Raw_Reaction_Smiles_valid", True) & df.get("Raw_Product_Smiles_valid", True)
    )

    return df


def audit_chirality(df: pd.DataFrame) -> pd.DataFrame:
    """Audit chirality information in product SMILES."""
    logger.info("Auditing chirality in product SMILES...")

    # Use Product_ (canonical product) for stereo analysis, fall back to Raw_Product_Smiles
    product_col = "Product_" if "Product_" in df.columns else "Raw_Product_Smiles"

    df["n_stereocenters"] = df[product_col].apply(_count_defined_stereocenters)
    df["has_chirality_tags"] = df[product_col].apply(_has_chirality_tags)

    # A valid chirality entry needs ≥2 defined stereocenters (Ca and Cb)
    df["chirality_valid"] = (df["n_stereocenters"] >= 2) & df["has_chirality_tags"]

    # Also check if atom-mapped product preserves chirality
    if "Mapped_Product" in df.columns:
        df["mapped_has_chirality"] = df["Mapped_Product"].apply(_has_chirality_tags)
        df["chirality_preserved_in_map"] = df["has_chirality_tags"] == df["mapped_has_chirality"]

    return df


def check_atom_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Check atom mapping completeness."""
    df["has_atom_map"] = df["Complete_Match"].fillna(False)

    # For rows without Complete_Match, check if Mapped_Reaction is present
    no_complete_match = df["has_atom_map"] == False  # noqa: E712
    has_mapped = df["Mapped_Reaction"].notna() & (df["Mapped_Reaction"].str.strip() != "")
    df.loc[no_complete_match & has_mapped, "has_atom_map"] = True

    return df


def run(project_root: Path = Path(".")) -> pd.DataFrame:
    in_path = project_root / "data" / "interim" / "02_deduplicated.csv"
    out_path = project_root / "data" / "interim" / "03_validated.csv"
    log_path = project_root / "data" / "interim" / "03_validation_log.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    n_total = len(df)
    logger.info(f"Loaded {n_total} rows from {in_path}")

    log_lines = [f"=== SMILES Validation & Chirality Audit ===", f"Input rows: {n_total}", ""]

    # Step 1: Validate SMILES
    logger.info("Validating SMILES columns...")
    df = validate_smiles_columns(df)

    for col in SMILES_COLS:
        flag = f"{col}_valid"
        if flag in df.columns:
            n_valid = df[flag].sum()
            n_missing = df[col].isna().sum() if col in df.columns else 0
            log_lines.append(f"{col}: {n_valid}/{n_total} valid, {n_missing} missing")

    n_overall_valid = df["smiles_valid"].sum()
    log_lines.append(f"\nOverall SMILES valid: {n_overall_valid}/{n_total}")

    # Step 2: Chirality audit
    df = audit_chirality(df)

    n_chiral_valid = df["chirality_valid"].sum()
    n_has_tags = df["has_chirality_tags"].sum()
    stereo_dist = df["n_stereocenters"].value_counts().sort_index()

    log_lines.append(f"\n=== Chirality Audit ===")
    log_lines.append(f"Products with chirality tags: {n_has_tags}/{n_total}")
    log_lines.append(f"Products with ≥2 defined stereocenters: {n_chiral_valid}/{n_total}")
    log_lines.append(f"\nStereocenter count distribution:")
    for n_sc, count in stereo_dist.items():
        log_lines.append(f"  {n_sc} stereocenters: {count} products")

    if "chirality_preserved_in_map" in df.columns:
        n_preserved = df["chirality_preserved_in_map"].sum()
        n_with_both = (df["has_chirality_tags"] & df["Mapped_Product"].notna()).sum()
        log_lines.append(
            f"\nChirality preserved in atom mapping: {n_preserved}/{n_with_both}"
        )

    # Step 3: Atom mapping check
    df = check_atom_mapping(df)
    n_has_map = df["has_atom_map"].sum()
    log_lines.append(f"\nRows with atom mapping: {n_has_map}/{n_total}")

    # Drop per-column valid flags (keep only aggregate flags)
    per_col_flags = [f"{c}_valid" for c in SMILES_COLS if f"{c}_valid" in df.columns]
    df = df.drop(columns=per_col_flags)

    # Save
    df.to_csv(out_path, index=False)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    logger.info(f"Saved {len(df)} rows to {out_path}")
    for line in log_lines:
        logger.info(line)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
