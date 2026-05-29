"""Step 10: Extract and normalize reaction conditions from Reaxys columns."""

import logging

import pandas as pd

from .audit import AuditTracker
from .constants import (
    SOLVENT_ALIASES,
    METAL_KEYWORDS,
    BASE_MAP,
    ACTIVATOR_MAP,
)
from .utils import parse_numeric_field, parse_semicolon_list

logger = logging.getLogger("rebuild_v4.step10")


def _normalize_solvent(raw: str) -> str:
    """Normalize solvent name to canonical form."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = raw.strip().lower()
    return SOLVENT_ALIASES.get(s, s)


def _identify_metal(reagent_str: str, catalyst_str: str) -> str:
    """Identify Lewis acid metal from reagent/catalyst strings."""
    combined = " ".join([
        str(reagent_str).lower() if pd.notna(reagent_str) else "",
        str(catalyst_str).lower() if pd.notna(catalyst_str) else "",
    ])
    for metal, keywords in METAL_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return metal
    return "none" if combined.strip() else "unknown"


def _identify_base(reagent_str: str) -> str:
    """Identify base from reagent string."""
    if not isinstance(reagent_str, str) or not reagent_str.strip():
        return "no_base"
    s = reagent_str.lower()
    for name, category in BASE_MAP.items():
        if name in s:
            return category
    return "no_base"


def _identify_activator(reagent_str: str, catalyst_str: str) -> str:
    """Identify Lewis acid activator from reagent/catalyst strings."""
    combined = " ".join([
        str(reagent_str).lower() if pd.notna(reagent_str) else "",
        str(catalyst_str).lower() if pd.notna(catalyst_str) else "",
    ])
    for name, category in ACTIVATOR_MAP.items():
        if name in combined:
            return category
    return "other_activator" if combined.strip() else "other_activator"


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Extract structured conditions from Reaxys raw columns."""
    logger.info("Step 10: Extracting reaction conditions...")
    n_start = len(df)

    # --- Temperature ---
    temp_col = "Temperature (Reaction Details) [C]"
    if temp_col in df.columns:
        df["temperature_c"] = df[temp_col].apply(parse_numeric_field)
    else:
        df["temperature_c"] = None

    # --- Time ---
    time_col = "Time (Reaction Details) [h]"
    if time_col in df.columns:
        df["time_h"] = df[time_col].apply(parse_numeric_field)
    else:
        df["time_h"] = None

    # --- Pressure ---
    pres_col = "Pressure (Reaction Details) [Torr]"
    if pres_col in df.columns:
        df["pressure_torr"] = df[pres_col].apply(parse_numeric_field)
    else:
        df["pressure_torr"] = None

    # --- Yield ---
    if "Yield (numerical)" in df.columns:
        df["yield_pct"] = df["Yield (numerical)"].apply(parse_numeric_field)
    else:
        df["yield_pct"] = None

    # --- Solvent ---
    solvent_col = "Solvent (Reaction Details)"
    if solvent_col in df.columns:
        # Parse semicolon-separated list, normalize first entry
        df["solvent_raw"] = df[solvent_col].apply(
            lambda x: parse_semicolon_list(x)
        )
        df["solvent_name"] = df["solvent_raw"].apply(
            lambda lst: _normalize_solvent(lst[0]) if lst else ""
        )
    else:
        df["solvent_raw"] = [[] for _ in range(len(df))]
        df["solvent_name"] = ""

    # --- Reagent parsing ---
    reagent_col = "Reagent"
    catalyst_col = "Catalyst"

    df["metal"] = df.apply(
        lambda r: _identify_metal(
            r.get(reagent_col, ""),
            r.get(catalyst_col, ""),
        ), axis=1,
    )
    df["base_type"] = df.apply(
        lambda r: _identify_base(r.get(reagent_col, "")), axis=1,
    )
    df["activator_type"] = df.apply(
        lambda r: _identify_activator(
            r.get(reagent_col, ""),
            r.get(catalyst_col, ""),
        ), axis=1,
    )

    # Log condition coverage
    for col in ["temperature_c", "time_h", "yield_pct", "solvent_name", "metal"]:
        if col in df.columns:
            non_null = df[col].notna() & (df[col] != "") & (df[col] != "unknown")
            logger.info(f"  {col}: {non_null.sum()} / {len(df)} non-empty")

    audit.record_step("10_conditions_extract", len(df))
    logger.info(f"  Step 10 complete: {n_start} -> {len(df)} rows (no drops)")
    return df
