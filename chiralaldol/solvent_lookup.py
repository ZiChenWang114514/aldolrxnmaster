"""Solvent inference and Kamlet-Taft parameter lookup for Evans aldol reactions.

Infers missing solvents from metal/reagent context using established
Evans aldol reaction conventions, then fills Kamlet-Taft parameters.
"""

import ast
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Kamlet-Taft solvent parameters: alpha (HBD), beta (HBA), pi_star (polarity), ET30
# Source: Reichardt, Chem. Rev. 1994; Marcus, Chem. Soc. Rev. 1993
KAMLET_TAFT = {
    "dichloromethane":      {"alpha": 0.13, "beta": 0.10, "pi_star": 0.82, "ET30": 40.7},
    "tetrahydrofuran":      {"alpha": 0.00, "beta": 0.55, "pi_star": 0.58, "ET30": 37.4},
    "ethyl acetate":        {"alpha": 0.00, "beta": 0.45, "pi_star": 0.55, "ET30": 38.1},
    "diethyl ether":        {"alpha": 0.00, "beta": 0.47, "pi_star": 0.27, "ET30": 34.5},
    "toluene":              {"alpha": 0.00, "beta": 0.11, "pi_star": 0.54, "ET30": 33.9},
    "pentane":              {"alpha": 0.00, "beta": 0.00, "pi_star": -0.08, "ET30": 31.1},
    "hexane":               {"alpha": 0.00, "beta": 0.00, "pi_star": -0.04, "ET30": 31.0},
    "acetonitrile":         {"alpha": 0.19, "beta": 0.40, "pi_star": 0.75, "ET30": 45.6},
    "methanol":             {"alpha": 0.98, "beta": 0.66, "pi_star": 0.60, "ET30": 55.4},
    "water":                {"alpha": 1.17, "beta": 0.47, "pi_star": 1.09, "ET30": 63.1},
    "chloroform":           {"alpha": 0.20, "beta": 0.10, "pi_star": 0.58, "ET30": 39.1},
    "1,2-dichloroethane":   {"alpha": 0.00, "beta": 0.10, "pi_star": 0.81, "ET30": 41.3},
    "dimethylformamide":    {"alpha": 0.00, "beta": 0.69, "pi_star": 0.88, "ET30": 43.2},
    "dmso":                 {"alpha": 0.00, "beta": 0.76, "pi_star": 1.00, "ET30": 45.1},
    "pyridine":             {"alpha": 0.00, "beta": 0.64, "pi_star": 0.87, "ET30": 40.5},
    "1-methyl-pyrrolidin-2-one": {"alpha": 0.00, "beta": 0.77, "pi_star": 0.92, "ET30": 42.2},
    "benzene":              {"alpha": 0.00, "beta": 0.10, "pi_star": 0.59, "ET30": 34.3},
    "difluoromethane":      {"alpha": 0.05, "beta": 0.05, "pi_star": 0.40, "ET30": 35.0},
    "acetone":              {"alpha": 0.08, "beta": 0.43, "pi_star": 0.71, "ET30": 42.2},
    "dioxane":              {"alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0},
    "carbon tetrachloride": {"alpha": 0.00, "beta": 0.10, "pi_star": 0.28, "ET30": 32.4},
}

# Solvent name aliases from the dataset
SOLVENT_ALIASES = {
    "1,2-dichloro-ethane": "1,2-dichloroethane",
    "Difluoromethane": "difluoromethane",
    "N,N-dimethylformamide": "dimethylformamide",
}

# Default solvent inference rules based on Evans aldol reaction conventions
# Metal → most common solvent in known reactions
METAL_DEFAULT_SOLVENT = {
    "B":  "dichloromethane",   # Bu2BOTf/Chx2BCl → CH2Cl2 (70% of known B)
    "Ti": "dichloromethane",   # TiCl4 → CH2Cl2 (82% of known Ti)
    "Sn": "dichloromethane",   # Sn(OTf)2 → CH2Cl2 (100% of known Sn)
    "Li": "tetrahydrofuran",   # LDA/LiHMDS → THF (86% of known Li)
    "Mg": "ethyl acetate",     # MgCl2 → EtOAc (53% of known Mg)
    "Zn": "dichloromethane",   # Zn(OTf)2 → CH2Cl2
    "Cu": "dichloromethane",   # Cu(OTf)2 → CH2Cl2
    "Zr": "dichloromethane",   # Zr(OTf)2 → CH2Cl2
}


def infer_solvent_from_context(metal: str, reagents_str: str) -> str | None:
    """Infer the most likely solvent from metal and reagent context.

    Returns solvent name (key in KAMLET_TAFT) or None if cannot infer.
    """
    # Try metal-based inference
    if pd.notna(metal) and str(metal) not in ("0", "", "nan"):
        metal_str = str(metal).strip()
        if metal_str in METAL_DEFAULT_SOLVENT:
            return METAL_DEFAULT_SOLVENT[metal_str]

    # Try reagent-based inference
    if pd.notna(reagents_str):
        try:
            reagents = ast.literal_eval(str(reagents_str))
        except (ValueError, SyntaxError):
            reagents = []

        reagent_text = " ".join(str(r).lower() for r in reagents)

        # Boron reagents → CH2Cl2
        if any(kw in reagent_text for kw in ["butylboryl", "bbn", "boron", "boryl"]):
            return "dichloromethane"
        # TiCl4 → CH2Cl2
        if "ticl" in reagent_text or "titanium" in reagent_text:
            return "dichloromethane"
        # LDA/LHMDS → THF
        if any(kw in reagent_text for kw in ["lda", "lithium diisopropylamide", "lhmds", "lihmds"]):
            return "tetrahydrofuran"

    return None


def get_kamlet_taft(solvent_name: str) -> dict:
    """Look up Kamlet-Taft parameters for a solvent name.

    Returns dict with alpha, beta, pi_star, ET30 keys.
    Falls back to zeros for unknown solvents.
    """
    name = solvent_name.lower().strip()
    name = SOLVENT_ALIASES.get(name, name)

    if name in KAMLET_TAFT:
        return KAMLET_TAFT[name]

    # Try partial match
    for key, params in KAMLET_TAFT.items():
        if key in name or name in key:
            return params

    return {"alpha": 0.0, "beta": 0.0, "pi_star": 0.0, "ET30": 0.0}


def fill_missing_solvents(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing solvent parameters using metal/reagent inference.

    Modifies solvent_alpha, solvent_beta, solvent_pi_star, solvent_ET30,
    solvent_known columns in-place.

    Returns modified dataframe with new column 'solvent_inferred' marking
    inferred rows.
    """
    df = df.copy()
    df["solvent_inferred"] = False

    unknown_mask = df["solvent_known"] == False
    n_unknown = unknown_mask.sum()
    n_inferred = 0

    for idx in df[unknown_mask].index:
        metal = df.at[idx, "metal"]
        reagents = df.at[idx, "Reagents"]

        inferred = infer_solvent_from_context(metal, reagents)
        if inferred is not None:
            params = get_kamlet_taft(inferred)
            df.at[idx, "solvent_alpha"] = params["alpha"]
            df.at[idx, "solvent_beta"] = params["beta"]
            df.at[idx, "solvent_pi_star"] = params["pi_star"]
            df.at[idx, "solvent_ET30"] = params["ET30"]
            df.at[idx, "solvent_known"] = True
            df.at[idx, "solvent_inferred"] = True
            n_inferred += 1

    n_still_unknown = n_unknown - n_inferred
    logger.info(
        f"Solvent inference: {n_inferred}/{n_unknown} filled "
        f"({n_still_unknown} still unknown)"
    )

    return df
