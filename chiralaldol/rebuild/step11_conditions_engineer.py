"""Step 11: Condition feature engineering (one-hot + physical properties)."""

import logging

import numpy as np
import pandas as pd

from .audit import AuditTracker
from .constants import (
    BASE_CATEGORIES,
    BASE_PROPERTIES,
    METAL_CATEGORIES,
    METAL_PROPERTIES,
    ACTIVATOR_CATEGORIES,
    SOLVENT_DB,
    SOLVENT_FEATURE_NAMES,
    METAL_DEFAULT_SOLVENT,
)

logger = logging.getLogger("rebuild_v4.step11")


def _one_hot(value: str, categories: list[str]) -> dict[str, int]:
    """One-hot encode a value against a category list."""
    return {f"feat_{cat}": int(value == cat) for cat in categories}


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Engineer condition features: one-hot + physical properties."""
    logger.info("Step 11: Engineering condition features...")
    n_start = len(df)

    feat_rows = []
    for _, row in df.iterrows():
        feats = {}

        # --- Base features (11d) ---
        base = row.get("base_type", "no_base")
        feats.update(_one_hot(base, BASE_CATEGORIES))
        props = BASE_PROPERTIES.get(base, BASE_PROPERTIES["no_base"])
        feats["feat_base_pKa"] = props["pKa"]
        feats["feat_base_steric_A"] = props["steric_A"]
        feats["feat_base_nucleophilicity"] = props["nucleophilicity"]

        # --- Metal features (13d) ---
        metal = row.get("metal", "unknown")
        if metal not in METAL_CATEGORIES:
            metal = "unknown"
        feats.update({f"feat_metal_{cat}": int(metal == cat) for cat in METAL_CATEGORIES})
        mprops = METAL_PROPERTIES.get(metal, METAL_PROPERTIES["unknown"])
        feats["feat_metal_coord_num"] = mprops["coordination_num"]
        feats["feat_metal_ionic_radius"] = mprops["ionic_radius_pm"]
        feats["feat_metal_hardness"] = mprops["pearson_hardness"]

        # --- Solvent features (8d) ---
        solvent_name = row.get("solvent_name", "")
        # Try solvent DB lookup
        if solvent_name in SOLVENT_DB:
            sparams = SOLVENT_DB[solvent_name]
            solvent_known = 1
        elif metal in METAL_DEFAULT_SOLVENT:
            sparams = SOLVENT_DB.get(METAL_DEFAULT_SOLVENT[metal], {})
            solvent_known = 0
        else:
            # Mean across all known solvents
            sparams = {
                k: np.mean([v[k] for v in SOLVENT_DB.values()])
                for k in ["alpha", "beta", "pi_star", "ET30", "epsilon", "viscosity_cP", "bp_C"]
            }
            solvent_known = 0

        feats["feat_solvent_alpha"] = sparams.get("alpha", 0)
        feats["feat_solvent_beta"] = sparams.get("beta", 0)
        feats["feat_solvent_pi_star"] = sparams.get("pi_star", 0)
        feats["feat_solvent_ET30"] = sparams.get("ET30", 0)
        feats["feat_solvent_epsilon"] = sparams.get("epsilon", 0)
        feats["feat_solvent_viscosity"] = sparams.get("viscosity_cP", 0)
        feats["feat_solvent_bp"] = sparams.get("bp_C", 0)
        feats["feat_solvent_known"] = solvent_known

        # --- Activator features (9d) ---
        activator = row.get("activator_type", "other_activator")
        if activator not in ACTIVATOR_CATEGORIES:
            activator = "other_activator"
        feats.update({f"feat_act_{cat}": int(activator == cat) for cat in ACTIVATOR_CATEGORIES})

        # --- Numeric conditions (3d) ---
        feats["feat_temperature_c"] = row.get("temperature_c") if pd.notna(row.get("temperature_c")) else 0.0
        feats["feat_time_h"] = row.get("time_h") if pd.notna(row.get("time_h")) else 0.0
        feats["feat_yield_pct"] = row.get("yield_pct") if pd.notna(row.get("yield_pct")) else 0.0

        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)
    logger.info(f"  Generated {feat_df.shape[1]} condition features")

    # Save feature matrix separately
    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    audit.record_step("11_conditions_engineer", len(df))
    logger.info(f"  Step 11 complete: {n_start} -> {len(df)} rows (no drops)")
    return df, feat_df
