"""Step 2: Deduplicate reactions and assign group_id for split integrity.

Three-tier deduplication:
  Tier 1: Exact duplicates (same canonical rxn + same conditions + same labels) → remove
  Tier 2: Same reaction, different conditions → keep all, same group_id
  Tier 3: Same substrate pair, different product stereochem → keep all, same group_id

Input:  data/interim/01_consolidated.csv
Output: data/interim/02_deduplicated.csv + audit log
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_canonical(smiles: str) -> str:
    """Canonicalize a SMILES string, returning original if RDKit fails."""
    if pd.isna(smiles) or not smiles.strip():
        return ""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return smiles.strip()


def _canonical_reaction(rxn_smiles: str) -> str:
    """Canonicalize a reaction SMILES (reactants>>products)."""
    if pd.isna(rxn_smiles) or not rxn_smiles.strip():
        return ""
    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        return rxn_smiles.strip()
    reactants_str, products_str = parts
    # Canonicalize each component
    reactants = sorted([_safe_canonical(s) for s in reactants_str.split(".")])
    products = sorted([_safe_canonical(s) for s in products_str.split(".")])
    return ".".join(reactants) + ">>" + ".".join(products)


def _make_substrate_key(ketone: str, aldehyde: str) -> str:
    """Create a canonical substrate pair key for grouping."""
    k = _safe_canonical(ketone) if pd.notna(ketone) else ""
    a = _safe_canonical(aldehyde) if pd.notna(aldehyde) else ""
    # Sort to ensure (A,B) == (B,A) if roles were swapped
    return "||".join(sorted([k, a]))


def _hash_row_for_dedup(row: pd.Series) -> str:
    """Create a hash of reaction + conditions + labels for exact dedup."""
    parts = [
        str(row.get("canonical_rxn", "")),
        str(row.get("Reagents", "")),
        str(row.get("solvent_clean", "")),
        str(row.get("metal", "")),
        str(row.get("label_Ca", "")),
        str(row.get("label_Cb", "")),
        str(row.get("label_SA", "")),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def run(project_root: Path = Path(".")) -> pd.DataFrame:
    in_path = project_root / "data" / "interim" / "01_consolidated.csv"
    out_path = project_root / "data" / "interim" / "02_deduplicated.csv"
    log_path = project_root / "data" / "interim" / "02_dedup_log.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    n_before = len(df)
    logger.info(f"Loaded {n_before} rows from {in_path}")

    log_lines = []

    # --- Compute canonical keys ---
    logger.info("Computing canonical SMILES keys...")
    df["canonical_rxn"] = df["Raw_Reaction_Smiles"].apply(_canonical_reaction)
    df["substrate_key"] = df.apply(
        lambda r: _make_substrate_key(r.get("Ketone", ""), r.get("Aldehyde", "")),
        axis=1,
    )

    # --- Tier 1: Exact duplicates ---
    df["dedup_hash"] = df.apply(_hash_row_for_dedup, axis=1)
    n_exact_dupes = df.duplicated(subset=["dedup_hash"], keep="first").sum()
    df_deduped = df.drop_duplicates(subset=["dedup_hash"], keep="first").copy()
    df_deduped = df_deduped.drop(columns=["dedup_hash"])
    n_after_tier1 = len(df_deduped)

    log_lines.append(f"Tier 1 (exact duplicates): removed {n_exact_dupes} rows")
    log_lines.append(f"  Before: {n_before}, After: {n_after_tier1}")
    logger.info(f"Tier 1: removed {n_exact_dupes} exact duplicates ({n_before} -> {n_after_tier1})")

    # --- Assign group_id based on substrate pair ---
    # Reactions sharing the same substrate pair MUST be in the same split fold
    substrate_keys = df_deduped["substrate_key"].unique()
    key_to_group = {k: i for i, k in enumerate(sorted(substrate_keys))}
    df_deduped["group_id"] = df_deduped["substrate_key"].map(key_to_group)

    n_groups = df_deduped["group_id"].nunique()
    logger.info(f"Assigned {n_groups} unique group_ids based on substrate pairs")

    # --- Classify remaining into Tier 2 / Tier 3 ---
    # Count how many rows share the same canonical_rxn
    rxn_counts = df_deduped.groupby("canonical_rxn").size()
    multi_rxn = set(rxn_counts[rxn_counts > 1].index)

    # Count how many rows share the same substrate_key
    sub_counts = df_deduped.groupby("substrate_key").size()
    multi_sub = set(sub_counts[sub_counts > 1].index)

    def classify_tier(row):
        if row["canonical_rxn"] in multi_rxn:
            return 2  # Same reaction, different conditions
        elif row["substrate_key"] in multi_sub:
            return 3  # Same substrates, different products
        else:
            return 0  # Unique reaction

    df_deduped["dedup_tier"] = df_deduped.apply(classify_tier, axis=1)

    tier_counts = df_deduped["dedup_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        tier_desc = {0: "unique", 2: "same rxn diff conditions", 3: "same substrates diff products"}
        log_lines.append(f"Tier {tier} ({tier_desc.get(tier, '?')}): {count} rows")
        logger.info(f"Tier {tier}: {count} rows")

    # --- Group size distribution ---
    group_sizes = df_deduped.groupby("group_id").size()
    log_lines.append(f"\nGroup size distribution:")
    log_lines.append(f"  Singleton groups: {(group_sizes == 1).sum()}")
    log_lines.append(f"  Groups with 2+ members: {(group_sizes > 1).sum()}")
    log_lines.append(f"  Max group size: {group_sizes.max()}")
    log_lines.append(f"  Mean group size: {group_sizes.mean():.2f}")

    # --- Save ---
    # Drop internal working columns
    df_deduped = df_deduped.drop(columns=["canonical_rxn", "substrate_key"])

    df_deduped.to_csv(out_path, index=False)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    logger.info(f"Saved {len(df_deduped)} rows to {out_path}")
    return df_deduped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
