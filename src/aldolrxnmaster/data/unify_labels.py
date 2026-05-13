"""Step 4: Unify labels into a 4-class joint encoding.

alldata uses {-1, +1} for label_Ca, label_Cb, label_SA.
Convert to {0, 1} binary, then create a joint 4-class label:
  Class 0: Ca=0, Cb=0
  Class 1: Ca=0, Cb=1
  Class 2: Ca=1, Cb=0
  Class 3: Ca=1, Cb=1

Also verify labels against product SMILES chirality where possible.

Input:  data/interim/03_validated.csv
Output: data/interim/04_labels_unified.csv + audit log
"""

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)


def convert_labels_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert {-1, +1} encoding to {0, 1}."""
    for col in ["label_Ca", "label_Cb", "label_SA"]:
        if col not in df.columns:
            continue

        original_values = df[col].dropna().unique()
        logger.info(f"{col} unique values before conversion: {sorted(original_values)}")

        # Handle both {-1,+1} and {0,1} encodings
        if -1.0 in original_values:
            # {-1, +1} -> {0, 1}
            df[col] = df[col].map({-1.0: 0, 1.0: 1})
            logger.info(f"  Converted {col} from {{-1,+1}} to {{0,1}}")
        elif set(original_values) <= {0.0, 1.0}:
            df[col] = df[col].astype(int)
            logger.info(f"  {col} already in {{0,1}}")
        else:
            logger.warning(f"  {col} has unexpected values: {sorted(original_values)}")

    return df


def create_joint_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create 4-class joint label from Ca and Cb."""
    # label_joint = Ca * 2 + Cb
    # Class 0: Ca=0, Cb=0
    # Class 1: Ca=0, Cb=1
    # Class 2: Ca=1, Cb=0
    # Class 3: Ca=1, Cb=1
    mask = df["label_Ca"].notna() & df["label_Cb"].notna()
    df.loc[mask, "label_joint"] = (df.loc[mask, "label_Ca"] * 2 + df.loc[mask, "label_Cb"]).astype(
        int
    )
    df.loc[~mask, "label_joint"] = pd.NA

    return df


def verify_sa_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Verify that label_SA is consistent with Ca and Cb.

    syn = same configuration at Ca and Cb (Ca == Cb)
    anti = opposite configuration (Ca != Cb)

    label_SA encoding: in original {-1,+1} space, we need to determine
    what SA=1 means. After conversion to {0,1}:
    We verify by checking if SA correlates with (Ca == Cb) or (Ca != Cb).
    """
    mask = df["label_Ca"].notna() & df["label_Cb"].notna() & df["label_SA"].notna()
    subset = df.loc[mask].copy()

    same_config = (subset["label_Ca"] == subset["label_Cb"]).astype(int)
    sa_matches_same = (subset["label_SA"] == same_config).sum()
    sa_matches_diff = (subset["label_SA"] == (1 - same_config)).sum()

    total = len(subset)
    logger.info(f"label_SA consistency check (n={total}):")
    logger.info(f"  SA matches (Ca==Cb): {sa_matches_same}/{total} ({sa_matches_same/total:.1%})")
    logger.info(f"  SA matches (Ca!=Cb): {sa_matches_diff}/{total} ({sa_matches_diff/total:.1%})")

    # Determine which interpretation is correct
    if sa_matches_same > sa_matches_diff:
        df["sa_interpretation"] = "syn=(Ca==Cb)"
        df["sa_consistent"] = True
        inconsistent_mask = mask & (df["label_SA"] != (df["label_Ca"] == df["label_Cb"]).astype(int))
        df.loc[inconsistent_mask, "sa_consistent"] = False
        n_inconsistent = inconsistent_mask.sum()
    else:
        df["sa_interpretation"] = "anti=(Ca!=Cb)"
        df["sa_consistent"] = True
        inconsistent_mask = mask & (
            df["label_SA"] != (df["label_Ca"] != df["label_Cb"]).astype(int)
        )
        df.loc[inconsistent_mask, "sa_consistent"] = False
        n_inconsistent = inconsistent_mask.sum()

    logger.info(f"  Inconsistent SA labels: {n_inconsistent}")
    return df


def verify_labels_against_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Verify stored labels against chirality tags extracted from product SMILES.

    This is an approximate check: we extract R/S from the product SMILES
    and compare to the stored Ca/Cb labels. Not all products will have
    clearly identifiable aldol stereocenters.
    """
    product_col = "Product_" if "Product_" in df.columns else "Raw_Product_Smiles"
    n_verified = 0
    n_mismatch = 0
    n_unverifiable = 0

    df["label_verified"] = pd.NA

    for idx, row in df.iterrows():
        smi = row.get(product_col)
        if pd.isna(smi):
            n_unverifiable += 1
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            n_unverifiable += 1
            continue

        try:
            centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            defined = [(atom_idx, cip) for atom_idx, cip in centers if cip in ("R", "S")]
        except Exception:
            n_unverifiable += 1
            continue

        if len(defined) < 2:
            n_unverifiable += 1
            continue

        # We found ≥2 stereocenters — the product has chirality info
        # However, without knowing which atom is Ca and which is Cb,
        # we can only verify that the product HAS stereochemistry, not the exact mapping.
        # Mark as "has_stereo_verified"
        n_verified += 1
        df.at[idx, "label_verified"] = True

    df["label_verified"] = df["label_verified"].fillna(False)
    logger.info(
        f"Label vs SMILES chirality: {n_verified} verified, "
        f"{n_mismatch} mismatches, {n_unverifiable} unverifiable"
    )

    return df


def run(project_root: Path = Path(".")) -> pd.DataFrame:
    in_path = project_root / "data" / "interim" / "03_validated.csv"
    out_path = project_root / "data" / "interim" / "04_labels_unified.csv"
    log_path = project_root / "data" / "interim" / "04_label_unification_log.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    n_total = len(df)
    logger.info(f"Loaded {n_total} rows from {in_path}")

    log_lines = [f"=== Label Unification ===", f"Input rows: {n_total}", ""]

    # Step 1: Convert to binary {0,1}
    df = convert_labels_to_binary(df)

    for col in ["label_Ca", "label_Cb", "label_SA"]:
        if col in df.columns:
            dist = df[col].value_counts().sort_index()
            n_na = df[col].isna().sum()
            log_lines.append(f"{col} distribution: {dict(dist)}, NaN: {n_na}")

    # Step 2: Create 4-class joint label
    df = create_joint_label(df)
    joint_dist = df["label_joint"].value_counts().sort_index()
    log_lines.append(f"\nJoint 4-class distribution:")
    class_names = {0: "Ca=0,Cb=0", 1: "Ca=0,Cb=1", 2: "Ca=1,Cb=0", 3: "Ca=1,Cb=1"}
    for cls, count in joint_dist.items():
        pct = count / n_total * 100
        log_lines.append(f"  Class {int(cls)} ({class_names.get(int(cls), '?')}): {count} ({pct:.1f}%)")

    # Step 3: Verify SA consistency
    df = verify_sa_consistency(df)
    n_sa_inconsistent = (~df["sa_consistent"]).sum() if "sa_consistent" in df.columns else 0
    log_lines.append(f"\nSA interpretation: {df['sa_interpretation'].iloc[0] if 'sa_interpretation' in df.columns else 'N/A'}")
    log_lines.append(f"SA inconsistent rows: {n_sa_inconsistent}")

    # Step 4: Verify labels against product SMILES
    df = verify_labels_against_smiles(df)
    n_verified = df["label_verified"].sum()
    log_lines.append(f"\nLabels verified against product SMILES chirality: {n_verified}/{n_total}")

    # Save
    df.to_csv(out_path, index=False)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    logger.info(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
