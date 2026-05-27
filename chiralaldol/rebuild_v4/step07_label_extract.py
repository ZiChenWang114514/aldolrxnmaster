"""Step 07: Extract stereochemistry labels from CIP codes and ee/dr data."""

import logging
import re
from typing import Optional

import pandas as pd
from rdkit import Chem

from .audit import AuditTracker
from .utils import safe_mol

logger = logging.getLogger("rebuild_v4.step07")


def _extract_cip_labels(product_smi: str, ca_idx, cb_idx) -> tuple[Optional[int], Optional[int]]:
    """Extract CIP R/S at Ca and Cb atom indices.

    Returns (label_Ca, label_Cb) where R=0, S=1. None if unavailable.
    """
    if pd.isna(ca_idx) or pd.isna(cb_idx):
        return None, None

    mol = safe_mol(product_smi)
    if mol is None:
        return None, None

    try:
        ca_idx = int(ca_idx)
        cb_idx = int(cb_idx)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        ca_atom = mol.GetAtomWithIdx(ca_idx)
        cb_atom = mol.GetAtomWithIdx(cb_idx)

        ca_cip = ca_atom.GetPropsAsDict().get("_CIPCode")
        cb_cip = cb_atom.GetPropsAsDict().get("_CIPCode")

        label_ca = {"R": 0, "S": 1}.get(ca_cip)
        label_cb = {"R": 0, "S": 1}.get(cb_cip)
        return label_ca, label_cb
    except Exception:
        return None, None


def _parse_optical_yield(optical_str: str) -> dict:
    """Parse Reaxys 'Yield (optical)' field for ee/dr/syn/anti info.

    Returns dict with keys: ee_value, dr_ratio, syn_anti, is_major_syn.
    """
    result = {"ee_value": None, "dr_ratio": None, "syn_anti": None, "is_major_syn": None}

    if not isinstance(optical_str, str) or not optical_str.strip():
        return result

    s = optical_str.lower().strip()

    # Extract ee value — Reaxys uses "percent" (not "%"), e.g. "99 percent ee"
    ee_match = re.search(r"(\d+\.?\d*)\s*(?:%|percent)?\s*ee", s)
    if ee_match:
        result["ee_value"] = float(ee_match.group(1))

    # Extract de value (diastereomeric excess) — same "percent" format
    de_match = re.search(r"(\d+\.?\d*)\s*(?:%|percent)?\s*de", s)
    if de_match:
        result["ee_value"] = float(de_match.group(1))  # treat de like ee for our purposes

    # Extract dr ratio (e.g., "95:5", ">20:1")
    dr_match = re.search(r"(?:dr|d\.?r\.?)\s*[=:]?\s*>?(\d+\.?\d*)\s*:\s*(\d+\.?\d*)", s)
    if dr_match:
        major = float(dr_match.group(1))
        minor = float(dr_match.group(2))
        if major + minor > 0:
            result["dr_ratio"] = major / (major + minor)
    else:
        # Try standalone ratio
        ratio_match = re.search(r"(\d+\.?\d*)\s*:\s*(\d+\.?\d*)", s)
        if ratio_match:
            major = float(ratio_match.group(1))
            minor = float(ratio_match.group(2))
            if major + minor > 0:
                result["dr_ratio"] = major / (major + minor)

    # Detect syn/anti keywords
    if "anti" in s:
        result["syn_anti"] = "anti"
        result["is_major_syn"] = False
    elif "syn" in s:
        result["syn_anti"] = "syn"
        result["is_major_syn"] = True

    # Evans syn/anti terminology
    if "evans syn" in s:
        result["syn_anti"] = "evans_syn"
        result["is_major_syn"] = True
    elif "non-evans" in s or "non evans" in s:
        result["syn_anti"] = "non_evans"
        result["is_major_syn"] = False

    return result


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Extract stereochemistry labels from CIP codes and optical yield data."""
    logger.info("Step 07: Extracting stereochemistry labels...")
    n_start = len(df)

    prod_col = "canonical_main_product_smiles" if "canonical_main_product_smiles" in df.columns else "main_product_smiles"

    # --- CIP extraction ---
    cip_results = [
        _extract_cip_labels(row[prod_col], row.get("ca_atom_idx"), row.get("cb_atom_idx"))
        for _, row in df.iterrows()
    ]
    df["cip_Ca"] = [r[0] for r in cip_results]
    df["cip_Cb"] = [r[1] for r in cip_results]

    cip_ok = df["cip_Ca"].notna() & df["cip_Cb"].notna()
    logger.info(f"  CIP extraction success: {cip_ok.sum()} / {len(df)}")

    # --- Optical yield parsing ---
    if "Yield (optical)" in df.columns:
        optical_results = df["Yield (optical)"].apply(
            lambda x: _parse_optical_yield(str(x) if pd.notna(x) else "")
        )
        df["ee_value"] = optical_results.apply(lambda x: x["ee_value"])
        df["dr_ratio"] = optical_results.apply(lambda x: x["dr_ratio"])
        df["optical_syn_anti"] = optical_results.apply(lambda x: x["syn_anti"])
        df["optical_is_major_syn"] = optical_results.apply(lambda x: x["is_major_syn"])

        optical_ok = df["optical_syn_anti"].notna()
        logger.info(f"  Optical yield parsing success: {optical_ok.sum()} / {len(df)}")
    else:
        df["ee_value"] = None
        df["dr_ratio"] = None
        df["optical_syn_anti"] = None
        df["optical_is_major_syn"] = None

    # No drops in this step
    audit.record_step("07_label_extract", len(df))
    logger.info(f"  Step 07 complete: {n_start} -> {len(df)} rows (no drops)")
    return df
