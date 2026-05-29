"""Step 8: Solvent semantic parsing from Reagents/Solvent columns.

Priority: direct (Solvent col) > reagent_parsed > metal_inferred > unknown.
Unknown solvents keep NaN (no mean filling at this stage).
"""

import ast
import logging

import pandas as pd

from .constants import SOLVENT_DB, SOLVENT_ALIASES, METAL_DEFAULT_SOLVENT

logger = logging.getLogger(__name__)

# Additional solvent names that appear in Reagents lists
REAGENT_SOLVENT_NAMES = {
    "dichloromethane", "tetrahydrofuran", "ethyl acetate", "diethyl ether",
    "toluene", "pentane", "hexane", "heptane", "acetonitrile", "methanol",
    "ethanol", "chloroform", "benzene", "acetone", "pyridine", "water",
    "dimethylformamide", "dimethyl sulfoxide", "dioxane", "1,4-dioxane",
    "2-methyltetrahydrofuran", "isopropanol", "carbon tetrachloride",
    "methyl tert-butyl ether", "1,2-dichloroethane", "nitromethane",
    "cyclopentyl methyl ether",
}


def _parse_list_str(s) -> list[str]:
    """Parse Python list string repr to list of lowercase strings."""
    if pd.isna(s) or not str(s).strip():
        return []
    try:
        result = ast.literal_eval(str(s).strip())
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if x]
        return [str(result).strip().lower()]
    except (ValueError, SyntaxError):
        return [str(s).strip().lower()]


def _normalize_solvent_name(name: str) -> str | None:
    """Normalize a solvent name to canonical form, return None if unknown."""
    name = name.lower().strip()
    # Direct match
    if name in SOLVENT_DB:
        return name
    # Alias match
    if name in SOLVENT_ALIASES:
        return SOLVENT_ALIASES[name]
    # Partial match
    for key in SOLVENT_DB:
        if key in name or name in key:
            return key
    return None


def _infer_solvent(row: pd.Series) -> tuple[str | None, str]:
    """Infer solvent from row data.

    Returns (solvent_name, source) where source is one of:
    'direct', 'reagent_parsed', 'metal_inferred', 'unknown'
    """
    # Priority 1: Direct from Solvent column
    solvent_col = row.get("solvent")  # lowercase column
    if pd.notna(solvent_col):
        names = _parse_list_str(solvent_col)
        for n in names:
            canonical = _normalize_solvent_name(n)
            if canonical:
                return canonical, "direct"

    # Also try uppercase Solvent
    solvent_col2 = row.get("Solvent")
    if pd.notna(solvent_col2):
        names = _parse_list_str(solvent_col2)
        for n in names:
            canonical = _normalize_solvent_name(n)
            if canonical:
                return canonical, "direct"

    # Priority 2: Parse Reagents for solvent names
    reagents = row.get("Reagents")
    if pd.notna(reagents):
        reagent_list = _parse_list_str(reagents)
        for r in reagent_list:
            canonical = _normalize_solvent_name(r)
            if canonical:
                return canonical, "reagent_parsed"
        # Also check if reagent text contains solvent keywords
        reagent_text = " ".join(reagent_list)
        for solvent_name in REAGENT_SOLVENT_NAMES:
            if solvent_name in reagent_text:
                canonical = _normalize_solvent_name(solvent_name)
                if canonical:
                    return canonical, "reagent_parsed"
        # Lewis acid → solvent inference from reagent text
        if any(kw in reagent_text for kw in ["butylboryl", "bbn", "boryl", "boron trifluoride"]):
            return "dichloromethane", "reagent_parsed"
        if "diethyl etherate" in reagent_text or "etherate" in reagent_text:
            return "diethyl ether", "reagent_parsed"

    # Priority 3: Metal-based inference
    metal = row.get("metal")
    if pd.notna(metal) and str(metal).strip() not in ("0", "", "nan"):
        metal_str = str(metal).strip()
        if metal_str in METAL_DEFAULT_SOLVENT:
            return METAL_DEFAULT_SOLVENT[metal_str], "metal_inferred"

    return None, "unknown"


def run(context: dict) -> dict:
    """Parse solvents from all available columns with semantic inference."""
    df: pd.DataFrame = context["df"].copy()
    n = len(df)
    logger.info(f"Step 8: Solvent semantic parsing for {n} rows")

    solvent_names = []
    solvent_sources = []

    for _, row in df.iterrows():
        name, source = _infer_solvent(row)
        solvent_names.append(name)
        solvent_sources.append(source)

    df["solvent_name"] = solvent_names
    df["solvent_source"] = solvent_sources

    # Statistics
    source_counts = pd.Series(solvent_sources).value_counts()
    for src, cnt in source_counts.items():
        logger.info(f"  {src}: {cnt} ({100*cnt/n:.1f}%)")

    n_known = sum(1 for s in solvent_names if s is not None)
    logger.info(f"  Total known: {n_known}/{n} ({100*n_known/n:.1f}%)")

    # Top solvents
    known_solvents = [s for s in solvent_names if s is not None]
    if known_solvents:
        top = pd.Series(known_solvents).value_counts().head(10)
        logger.info(f"  Top solvents: {top.to_dict()}")

    out_path = context["output_dir"] / "interim" / "08_solvent_parsed.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Step 8 complete: {n} rows (no deletions)")

    context["df"] = df
    return context
