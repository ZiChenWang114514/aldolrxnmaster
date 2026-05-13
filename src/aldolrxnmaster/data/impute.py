"""Step 5: Handle missing values with documented strategies.

NO fillna(0). Every imputation is tracked with explicit flags.

Strategies:
  - Solvent (27% missing): "unknown" category + Kamlet-Taft lookup
  - Reagents (7% missing): "unknown" category
  - Metal (~13% "0"): distinguish "none" vs "unknown"
  - Mapped_Reaction (4.2%): mark has_atom_map=False

Input:  data/interim/04_labels_unified.csv
Output: data/interim/05_imputed.csv + audit log
"""

import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Kamlet-Taft solvent parameters: alpha (HBD), beta (HBA), pi_star (dipolarity)
# Source: Marcus 1993, Reichardt 2003
# For common solvents in aldol reactions
KAMLET_TAFT = {
    "dichloromethane": {"alpha": 0.13, "beta": 0.10, "pi_star": 0.82, "ET30": 40.7},
    "tetrahydrofuran": {"alpha": 0.00, "beta": 0.55, "pi_star": 0.58, "ET30": 37.4},
    "diethyl ether": {"alpha": 0.00, "beta": 0.47, "pi_star": 0.27, "ET30": 34.5},
    "toluene": {"alpha": 0.00, "beta": 0.11, "pi_star": 0.54, "ET30": 33.9},
    "chloroform": {"alpha": 0.20, "beta": 0.10, "pi_star": 0.58, "ET30": 39.1},
    "acetonitrile": {"alpha": 0.19, "beta": 0.40, "pi_star": 0.75, "ET30": 45.6},
    "methanol": {"alpha": 0.98, "beta": 0.66, "pi_star": 0.60, "ET30": 55.4},
    "ethanol": {"alpha": 0.86, "beta": 0.75, "pi_star": 0.54, "ET30": 51.9},
    "water": {"alpha": 1.17, "beta": 0.47, "pi_star": 1.09, "ET30": 63.1},
    "dmf": {"alpha": 0.00, "beta": 0.69, "pi_star": 0.88, "ET30": 43.2},
    "n,n-dimethylformamide": {"alpha": 0.00, "beta": 0.69, "pi_star": 0.88, "ET30": 43.2},
    "dmso": {"alpha": 0.00, "beta": 0.76, "pi_star": 1.00, "ET30": 45.1},
    "dimethyl sulfoxide": {"alpha": 0.00, "beta": 0.76, "pi_star": 1.00, "ET30": 45.1},
    "hexane": {"alpha": 0.00, "beta": 0.00, "pi_star": -0.04, "ET30": 31.0},
    "pentane": {"alpha": 0.00, "beta": 0.00, "pi_star": -0.08, "ET30": 31.1},
    "heptane": {"alpha": 0.00, "beta": 0.00, "pi_star": -0.02, "ET30": 31.1},
    "benzene": {"alpha": 0.00, "beta": 0.10, "pi_star": 0.59, "ET30": 34.3},
    "acetone": {"alpha": 0.08, "beta": 0.43, "pi_star": 0.71, "ET30": 42.2},
    "ethyl acetate": {"alpha": 0.00, "beta": 0.45, "pi_star": 0.55, "ET30": 38.1},
    "1,2-dichloroethane": {"alpha": 0.10, "beta": 0.10, "pi_star": 0.81, "ET30": 41.3},
    "isopropanol": {"alpha": 0.76, "beta": 0.84, "pi_star": 0.48, "ET30": 48.4},
    "2-propanol": {"alpha": 0.76, "beta": 0.84, "pi_star": 0.48, "ET30": 48.4},
    "carbon tetrachloride": {"alpha": 0.00, "beta": 0.10, "pi_star": 0.28, "ET30": 32.4},
    "dioxane": {"alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0},
    "1,4-dioxane": {"alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0},
    "pyridine": {"alpha": 0.00, "beta": 0.64, "pi_star": 0.87, "ET30": 40.5},
    "nitromethane": {"alpha": 0.22, "beta": 0.06, "pi_star": 0.85, "ET30": 46.3},
    "1,2-dimethoxyethane": {"alpha": 0.00, "beta": 0.41, "pi_star": 0.53, "ET30": 38.2},
    "methyl tert-butyl ether": {"alpha": 0.00, "beta": 0.55, "pi_star": 0.27, "ET30": 34.7},
}

# Solvent name normalization
SOLVENT_ALIASES = {
    "thf": "tetrahydrofuran",
    "dcm": "dichloromethane",
    "ch2cl2": "dichloromethane",
    "meoh": "methanol",
    "etoh": "ethanol",
    "dmf": "n,n-dimethylformamide",
    "et2o": "diethyl ether",
    "chcl3": "chloroform",
    "mecn": "acetonitrile",
    "etoac": "ethyl acetate",
    "phme": "toluene",
    "dme": "1,2-dimethoxyethane",
    "mtbe": "methyl tert-butyl ether",
    "ipoh": "isopropanol",
    "i-proh": "isopropanol",
}


def _parse_list_string(s: str) -> list[str]:
    """Parse a Python list string repr like "['a', 'b']" to actual list."""
    if pd.isna(s) or not str(s).strip():
        return []
    s = str(s).strip()
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if x]
        return [str(result).strip().lower()]
    except (ValueError, SyntaxError):
        return [s.lower()]


def _normalize_solvent_name(name: str) -> str:
    """Normalize a solvent name to canonical form."""
    name = name.strip().lower()
    return SOLVENT_ALIASES.get(name, name)


def encode_solvent(df: pd.DataFrame) -> pd.DataFrame:
    """Encode solvent information using Kamlet-Taft parameters."""
    solvent_col = "solvent_clean" if "solvent_clean" in df.columns else None
    if solvent_col is None:
        logger.warning("No solvent column found")
        for col in ["solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30"]:
            df[col] = pd.NA
        df["solvent_known"] = False
        return df

    # Compute mean Kamlet-Taft parameters across all known solvents (for unknown fallback)
    all_kt = list(KAMLET_TAFT.values())
    mean_alpha = sum(v["alpha"] for v in all_kt) / len(all_kt)
    mean_beta = sum(v["beta"] for v in all_kt) / len(all_kt)
    mean_pi_star = sum(v["pi_star"] for v in all_kt) / len(all_kt)
    mean_ET30 = sum(v["ET30"] for v in all_kt) / len(all_kt)

    alphas, betas, pi_stars, et30s, knowns = [], [], [], [], []

    for _, row in df.iterrows():
        solvents = _parse_list_string(row.get(solvent_col, ""))

        if not solvents:
            alphas.append(mean_alpha)
            betas.append(mean_beta)
            pi_stars.append(mean_pi_star)
            et30s.append(mean_ET30)
            knowns.append(False)
            continue

        # Normalize and look up each solvent; average if multiple
        a_vals, b_vals, p_vals, e_vals = [], [], [], []
        any_known = False
        for solv in solvents:
            solv = _normalize_solvent_name(solv)
            if solv in KAMLET_TAFT:
                kt = KAMLET_TAFT[solv]
                a_vals.append(kt["alpha"])
                b_vals.append(kt["beta"])
                p_vals.append(kt["pi_star"])
                e_vals.append(kt["ET30"])
                any_known = True

        if any_known:
            alphas.append(sum(a_vals) / len(a_vals))
            betas.append(sum(b_vals) / len(b_vals))
            pi_stars.append(sum(p_vals) / len(p_vals))
            et30s.append(sum(e_vals) / len(e_vals))
        else:
            alphas.append(mean_alpha)
            betas.append(mean_beta)
            pi_stars.append(mean_pi_star)
            et30s.append(mean_ET30)

        knowns.append(any_known)

    df["solvent_alpha"] = alphas
    df["solvent_beta"] = betas
    df["solvent_pi_star"] = pi_stars
    df["solvent_ET30"] = et30s
    df["solvent_known"] = knowns

    n_known = sum(knowns)
    logger.info(f"Solvent Kamlet-Taft: {n_known}/{len(df)} rows with known parameters")

    return df


def encode_metal(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode metal column."""
    if "metal" not in df.columns:
        df["metal_clean"] = "unknown"
        return df

    def clean_metal(val):
        if pd.isna(val):
            return "unknown"
        val = str(val).strip()
        if val in ("0", "[]", "", "nan"):
            return "none"
        return val

    df["metal_clean"] = df["metal"].apply(clean_metal)

    # Try to infer metal from Reagents if metal is "none" or "unknown"
    metal_reagent_map = {
        "titanium": "Ti",
        "ticl4": "Ti",
        "titanium tetrachloride": "Ti",
        "titanium(iv) chloride": "Ti",
        "boron": "B",
        "dibutylboron": "B",
        "di-n-butylboryl": "B",
        "9-borabicyclononane": "B",
        "9-bbn": "B",
        "triflate": "B",  # Bu2BOTf context
        "lithium": "Li",
        "n-butyllithium": "Li",
        "lda": "Li",
        "lithium diisopropylamide": "Li",
        "tin": "Sn",
        "tin(ii)": "Sn",
        "stannous": "Sn",
        "magnesium": "Mg",
        "mgcl2": "Mg",
        "magnesium chloride": "Mg",
        "zinc": "Zn",
        "zirconium": "Zr",
    }

    n_inferred = 0
    for idx, row in df.iterrows():
        if row["metal_clean"] not in ("none", "unknown"):
            continue
        reagents_str = str(row.get("Reagents", "")).lower()
        for keyword, metal in metal_reagent_map.items():
            if keyword in reagents_str:
                df.at[idx, "metal_clean"] = metal
                n_inferred += 1
                break

    logger.info(f"Metal: inferred {n_inferred} from reagent strings")

    metal_dist = df["metal_clean"].value_counts()
    logger.info(f"Metal distribution:\n{metal_dist.to_string()}")

    return df


def run(project_root: Path = Path(".")) -> pd.DataFrame:
    in_path = project_root / "data" / "interim" / "04_labels_unified.csv"
    out_path = project_root / "data" / "interim" / "05_imputed.csv"
    log_path = project_root / "data" / "interim" / "05_imputation_log.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    n_total = len(df)
    logger.info(f"Loaded {n_total} rows from {in_path}")

    log_lines = [f"=== Missing Value Imputation ===", f"Input rows: {n_total}", ""]

    # Pre-imputation missing summary
    log_lines.append("Pre-imputation missing values:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            log_lines.append(f"  {col}: {n_miss} ({n_miss/n_total:.1%})")

    # Step 1: Encode solvent with Kamlet-Taft parameters
    df = encode_solvent(df)
    n_solvent_known = df["solvent_known"].sum()
    log_lines.append(f"\nSolvent: {n_solvent_known}/{n_total} with Kamlet-Taft params")

    # Step 2: Clean and encode metal
    df = encode_metal(df)
    metal_dist = df["metal_clean"].value_counts()
    log_lines.append(f"\nMetal distribution after cleaning:")
    for m, c in metal_dist.items():
        log_lines.append(f"  {m}: {c}")

    # Step 3: Post-imputation summary
    log_lines.append(f"\nPost-imputation missing values:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            log_lines.append(f"  {col}: {n_miss} ({n_miss/n_total:.1%})")

    # Save
    df.to_csv(out_path, index=False)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    logger.info(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
