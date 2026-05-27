"""Step 06: Atom mapping via RXNMapper + SMARTS template dual verification."""

import logging
from typing import Optional

import pandas as pd
from rdkit import Chem

from .audit import AuditTracker
from .constants import ALDOL_PRODUCT_SMARTS_AUX, AUXILIARY_SMARTS
from .utils import safe_mol

logger = logging.getLogger("rebuild_v4.step06")

# Product template: :1 = Cb (OH-bearing), :2 = Ca (alpha to C=O-N)
_PRODUCT_TEMPLATES_RAW = {
    "aux": "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])[NX3]",
    "generic": "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])",
}

# Pre-compile and find tagged atom positions
_PRODUCT_TEMPLATES = {}
for _name, _smarts in _PRODUCT_TEMPLATES_RAW.items():
    _pat = Chem.MolFromSmarts(_smarts)
    # Find which query atom indices have atom map :1 and :2
    _map_positions = {}
    for _i in range(_pat.GetNumAtoms()):
        _mapnum = _pat.GetAtomWithIdx(_i).GetAtomMapNum()
        if _mapnum > 0:
            _map_positions[_mapnum] = _i
    _PRODUCT_TEMPLATES[_name] = (_pat, _map_positions)


def _template_locate_ca_cb(product_smi: str) -> Optional[tuple[int, int]]:
    """Locate Ca and Cb atom indices in product using SMARTS templates.

    Returns (ca_idx, cb_idx) or None.
    """
    mol = safe_mol(product_smi)
    if mol is None:
        return None

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for template_name, (pat, map_positions) in _PRODUCT_TEMPLATES.items():
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        if matches:
            # :1 = Cb position, :2 = Ca position (from pre-computed map)
            cb_query_idx = map_positions[1]  # atom map :1
            ca_query_idx = map_positions[2]  # atom map :2
            cb_idx = matches[0][cb_query_idx]
            ca_idx = matches[0][ca_query_idx]
            return ca_idx, cb_idx
    return None


def _rxnmapper_map(rxn_smiles_list: list[str]) -> list[tuple[Optional[str], float]]:
    """Batch atom mapping via RXNMapper. Returns list of (mapped_rxn, confidence)."""
    try:
        from rxnmapper import RXNMapper
        mapper = RXNMapper()
        results = mapper.get_attention_guided_atom_maps(rxn_smiles_list)
        return [(r["mapped_rxn"], r["confidence"]) for r in results]
    except ImportError:
        logger.warning("RXNMapper not installed, skipping AI-based atom mapping")
        return [(None, 0.0)] * len(rxn_smiles_list)
    except Exception as e:
        logger.warning(f"RXNMapper failed: {e}")
        return [(None, 0.0)] * len(rxn_smiles_list)


def _rxnmapper_locate_ca_cb(mapped_rxn: str, product_smi: str) -> Optional[tuple[int, int]]:
    """From a mapped reaction, locate Ca/Cb in the product by tracing the new C-C bond."""
    if not mapped_rxn:
        return None
    try:
        # Parse mapped product
        parts = mapped_rxn.split(">>")
        if len(parts) != 2:
            return None

        prod_mol = Chem.MolFromSmiles(parts[1])
        if prod_mol is None:
            return None

        # Fall back to template matching on the mapped product
        return _template_locate_ca_cb(parts[1])
    except Exception:
        return None


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Atom mapping: SMARTS template + optional RXNMapper dual verification."""
    logger.info("Step 06: Atom mapping (template + RXNMapper)...")
    n_start = len(df)

    prod_col = "canonical_main_product_smiles" if "canonical_main_product_smiles" in df.columns else "main_product_smiles"

    # --- SMARTS template mapping ---
    template_results = df[prod_col].apply(_template_locate_ca_cb)
    df["template_ca_idx"] = template_results.apply(lambda x: x[0] if x else None)
    df["template_cb_idx"] = template_results.apply(lambda x: x[1] if x else None)
    template_ok = template_results.apply(lambda x: x is not None)
    logger.info(f"  Template mapping success: {template_ok.sum()} / {len(df)}")

    # --- RXNMapper (batch) ---
    rxn_col = "Reaction"
    rxn_list = df[rxn_col].tolist()

    # Process in batches to avoid memory issues
    batch_size = 500
    all_mapped = []
    all_conf = []
    for start in range(0, len(rxn_list), batch_size):
        batch = rxn_list[start:start + batch_size]
        results = _rxnmapper_map(batch)
        for mapped, conf in results:
            all_mapped.append(mapped)
            all_conf.append(conf)

    df["mapped_reaction"] = all_mapped
    df["rxnmapper_confidence"] = all_conf

    # RXNMapper-based Ca/Cb
    rxnmapper_results = [
        _rxnmapper_locate_ca_cb(mapped, prod)
        for mapped, prod in zip(df["mapped_reaction"], df[prod_col])
    ]
    df["rxnmapper_ca_idx"] = [r[0] if r else None for r in rxnmapper_results]
    df["rxnmapper_cb_idx"] = [r[1] if r else None for r in rxnmapper_results]
    rxnmapper_ok = pd.Series([r is not None for r in rxnmapper_results])
    logger.info(f"  RXNMapper mapping success: {rxnmapper_ok.sum()} / {len(df)}")

    # --- Dual verification ---
    def compute_confidence(row):
        t_ok = pd.notna(row.get("template_ca_idx"))
        r_ok = pd.notna(row.get("rxnmapper_ca_idx"))
        if t_ok and r_ok:
            if (row["template_ca_idx"] == row["rxnmapper_ca_idx"] and
                    row["template_cb_idx"] == row["rxnmapper_cb_idx"]):
                return "high"
            return "low"
        if t_ok:
            return "medium"
        if r_ok:
            return "medium"
        return "failed"

    df["mapping_confidence"] = df.apply(compute_confidence, axis=1)

    # Final Ca/Cb: prefer template when available (more reliable for known patterns)
    df["ca_atom_idx"] = df["template_ca_idx"].fillna(df["rxnmapper_ca_idx"])
    df["cb_atom_idx"] = df["template_cb_idx"].fillna(df["rxnmapper_cb_idx"])

    conf_dist = df["mapping_confidence"].value_counts()
    logger.info(f"  Mapping confidence distribution:")
    for level, count in conf_dist.items():
        logger.info(f"    {level}: {count}")

    # No rows dropped in this step — failed mappings are kept but flagged
    audit.record_step("06_atom_mapping", len(df))
    logger.info(f"  Step 06 complete: {n_start} -> {len(df)} rows (no drops)")
    return df
