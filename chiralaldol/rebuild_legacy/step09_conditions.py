"""Step 9: Condition feature engineering (44d total).

Base:      8d one-hot + 3d chemical properties = 11d
Metal:    10d one-hot + 3d Lewis acidity        = 13d
Solvent:   7d (KT 4d + physical 3d) + 1d flag   =  8d
Activator: 9d one-hot + 3d binary flags          = 12d
Total:                                            = 44d
"""

import ast
import logging

import numpy as np
import pandas as pd

from .constants import (
    BASE_MAP, BASE_CATEGORIES, BASE_PROPERTIES,
    ACTIVATOR_MAP, ACTIVATOR_CATEGORIES,
    METAL_CATEGORIES, METAL_PROPERTIES,
    SOLVENT_DB, SOLVENT_ALIASES, SOLVENT_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


def _parse_reagent_list(s) -> list[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    try:
        result = ast.literal_eval(str(s).strip())
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if x]
        return [str(result).strip().lower()]
    except (ValueError, SyntaxError):
        return [str(s).lower()]


def _classify_base(reagents: list[str]) -> str:
    """Classify the dominant base from reagent list."""
    for r in reagents:
        for name, category in BASE_MAP.items():
            if name in r or r in name:
                return category
    return "no_base"


def _classify_activator(reagents: list[str]) -> str:
    """Classify the activator from reagent list."""
    for r in reagents:
        for name, category in ACTIVATOR_MAP.items():
            if name in r or r in name:
                return category
    return "other_activator"


def _has_keyword(reagents: list[str], keywords: list[str]) -> bool:
    text = " ".join(reagents)
    return any(kw in text for kw in keywords)


def _get_solvent_features(solvent_name: str | None) -> dict:
    """Get 14d solvent features (13 continuous + 1 flag)."""
    result = {k: np.nan for k in SOLVENT_FEATURE_NAMES}
    result["solvent_known"] = 0.0

    if solvent_name is None or (isinstance(solvent_name, float) and pd.isna(solvent_name)):
        return result

    name = str(solvent_name).lower().strip()
    name = SOLVENT_ALIASES.get(name, name)

    if name in SOLVENT_DB:
        params = SOLVENT_DB[name]
        result["solvent_alpha"] = params["alpha"]
        result["solvent_beta"] = params["beta"]
        result["solvent_pi_star"] = params["pi_star"]
        result["solvent_ET30"] = params["ET30"]
        result["solvent_epsilon"] = params["epsilon"]
        result["solvent_viscosity"] = params["viscosity_cP"]
        result["solvent_bp"] = params["bp_C"]
        result["solvent_known"] = 1.0
        result["solvent_mw"] = params["mw"]
        result["solvent_density"] = params["density_g_mL"]
        result["solvent_refractive_index"] = params["refractive_index"]
        result["solvent_dipole"] = params["dipole_D"]
        result["solvent_molar_vol"] = params["molar_vol_cm3"]
        result["solvent_logP"] = params["logP"]

    return result


def run(context: dict) -> dict:
    """Compute all condition features (50d: 44 original + 6 new solvent descriptors)."""
    df: pd.DataFrame = context["df"].copy()
    n = len(df)
    logger.info(f"Step 9: Condition feature engineering for {n} rows")

    # Parse reagents once (handle missing column)
    if "Reagents" not in df.columns:
        logger.warning("  'Reagents' column not found — creating empty reagent lists")
        df["Reagents"] = ""
    reagent_lists = df["Reagents"].apply(_parse_reagent_list)

    # ── Base features (11d) ──
    base_classes = reagent_lists.apply(_classify_base)
    # One-hot
    for cat in BASE_CATEGORIES:
        df[f"base_{cat}"] = (base_classes == cat).astype(float)
    # Chemical properties
    for prop in ["pKa", "steric_A", "nucleophilicity"]:
        df[f"base_{prop}"] = base_classes.map(lambda b: BASE_PROPERTIES.get(b, {}).get(prop, 0.0))

    logger.info(f"  Base distribution: {base_classes.value_counts().to_dict()}")

    # ── Metal features (13d) ──
    metal_col = df.get("metal", pd.Series(dtype=str))
    metal_clean = metal_col.fillna("unknown").astype(str).str.strip()
    metal_clean = metal_clean.replace({"0": "none", "": "none", "nan": "unknown"})
    # Map unlisted metals to "unknown"
    unlisted = metal_clean[~metal_clean.isin(METAL_CATEGORIES)]
    if len(unlisted.unique()) > 0:
        logger.info(f"  Unlisted metals mapped to 'unknown': {sorted(unlisted.unique())}")
        metal_clean = metal_clean.where(metal_clean.isin(METAL_CATEGORIES), "unknown")
    # One-hot
    for cat in METAL_CATEGORIES:
        df[f"metal_{cat}"] = (metal_clean == cat).astype(float)
    # Chemical properties
    for prop in ["coordination_num", "ionic_radius_pm", "pearson_hardness"]:
        df[f"metal_{prop}"] = metal_clean.map(lambda m: METAL_PROPERTIES.get(m, {}).get(prop, 0.0))

    logger.info(f"  Metal distribution: {metal_clean.value_counts().to_dict()}")

    # ── Solvent features (8d) ──
    solvent_feats = df["solvent_name"].apply(_get_solvent_features)
    for col_name in SOLVENT_FEATURE_NAMES:
        df[col_name] = [f[col_name] for f in solvent_feats]

    n_solvent_known = df["solvent_known"].sum()
    logger.info(f"  Solvent known: {int(n_solvent_known)}/{n} ({100*n_solvent_known/n:.1f}%)")

    # ── Activator features (12d) ──
    activator_classes = reagent_lists.apply(_classify_activator)
    for cat in ACTIVATOR_CATEGORIES:
        df[f"act_{cat}"] = (activator_classes == cat).astype(float)

    # Binary flags
    df["has_oxidant"] = reagent_lists.apply(
        lambda rl: float(_has_keyword(rl, ["oxone", "oxalyl", "dess-martin", "swern", "pcc", "jones"]))
    )
    df["has_silylating"] = reagent_lists.apply(
        lambda rl: float(_has_keyword(rl, ["silyl", "tms", "tbs", "tbdms", "tips", "chloro-trimethyl-silane"]))
    )
    df["has_additive"] = reagent_lists.apply(
        lambda rl: float(_has_keyword(rl, ["4-dimethylaminopyridine", "dmap", "imidazole", "hatu"]))
    )

    logger.info(f"  Activator distribution: {activator_classes.value_counts().head(5).to_dict()}")

    # ── Collect all condition feature column names ──
    cond_cols = []
    cond_cols += [f"base_{c}" for c in BASE_CATEGORIES]
    cond_cols += ["base_pKa", "base_steric_A", "base_nucleophilicity"]
    cond_cols += [f"metal_{c}" for c in METAL_CATEGORIES]
    cond_cols += ["metal_coordination_num", "metal_ionic_radius_pm", "metal_pearson_hardness"]
    cond_cols += SOLVENT_FEATURE_NAMES
    cond_cols += [f"act_{c}" for c in ACTIVATOR_CATEGORIES]
    cond_cols += ["has_oxidant", "has_silylating", "has_additive"]

    logger.info(f"  Total condition features: {len(cond_cols)}d")
    context["condition_feature_cols"] = cond_cols

    # Save condition features separately
    cond_df = df[["original_index"] + cond_cols]
    feat_path = context["output_dir"] / "features" / "condition_features.csv"
    cond_df.to_csv(feat_path, index=False)

    out_path = context["output_dir"] / "interim" / "09_conditions.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Step 9 complete: {n} rows, {len(cond_cols)}d features saved")

    context["df"] = df
    return context
