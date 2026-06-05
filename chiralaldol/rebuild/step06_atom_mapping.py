"""Step 06: Atom mapping via RXNMapper + SMARTS template dual verification."""

import logging

import pandas as pd
from rdkit import Chem

from .audit import AuditTracker
from .utils import safe_mol

logger = logging.getLogger("rebuild_v4.step06")

# Product template: :1 = Cb (OH-bearing), :2 = Ca (alpha to C=O)
# Order matters: aux-specific templates checked first; protected-OH added for failed mappings
_PRODUCT_TEMPLATES_RAW = {
    # Free OH, amide (Evans/Crimmins/Myers)
    "aux":            "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])[NX3]",
    # Free OH, ester (V5: menthyl/borneol/abiko chiral ester auxiliaries)
    "ester":          "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])[OX2]",
    # Oxazoline-type (V5: Meyers' oxazoline auxiliary)
    "oxaz":           "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])C1=N[CH]CO1",
    # Free OH, generic carbonyl (Oppolzer sulfonamide etc.)
    "generic":        "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])",
    # Silyl ether protection (TBS/TMS/TIPS): O-SiX4
    "aux_silyl":      "[CX4:1]([OX2][SiX4])([#6])[CX4:2]([#6])[CX3](=[OX1])[NX3]",
    "ester_silyl":    "[CX4:1]([OX2][SiX4])([#6])[CX4:2]([#6])[CX3](=[OX1])[OX2]",
    "generic_silyl":  "[CX4:1]([OX2][SiX4])([#6])[CX4:2]([#6])[CX3](=[OX1])",
    # Benzyl ether protection (OBn/PMB): O-CH2-aryl or O-CH2-alkyl
    "aux_bn":         "[CX4:1]([OX2][CH2][c,C])([#6])[CX4:2]([#6])[CX3](=[OX1])[NX3]",
    "ester_bn":       "[CX4:1]([OX2][CH2][c,C])([#6])[CX4:2]([#6])[CX3](=[OX1])[OX2]",
    "generic_bn":     "[CX4:1]([OX2][CH2][c,C])([#6])[CX4:2]([#6])[CX3](=[OX1])",
    # Acetal/MOM/THP ether: O-CX4-O (methoxymethyl, tetrahydropyranyl, etc.)
    "aux_acetal":     "[CX4:1]([OX2][CX4][OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])[NX3]",
    "ester_acetal":   "[CX4:1]([OX2][CX4][OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])[OX2]",
    "generic_acetal": "[CX4:1]([OX2][CX4][OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])",
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


def _template_locate_ca_cb(product_smi: str) -> tuple[int, int] | None:
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


def _rxnmapper_map(rxn_smiles_list: list[str]) -> list[tuple[str | None, float]]:
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


def _rxnmapper_locate_ca_cb(mapped_rxn: str, product_smi: str) -> tuple[int, int] | None:
    """From a mapped reaction, locate Ca/Cb in the product.

    Strategy:
    1. Try template matching on the mapped product SMILES (fast path, handles all OH forms).
    2. If templates all fail, use a broad β-hydroxy carbonyl pattern that matches any
       O-protected Cb adjacent to Ca with downstream carbonyl (last-resort fallback).
    """
    if not mapped_rxn:
        return None
    try:
        parts = mapped_rxn.split(">>")
        if len(parts) != 2:
            return None

        mapped_prod_smi = parts[1]

        # Step 1: template matching on the atom-mapped product SMILES
        result = _template_locate_ca_cb(mapped_prod_smi)
        if result is not None:
            return result

        # Step 2: broad fallback — any O-substituted Cb adjacent to Ca with C=O
        prod_mol = Chem.MolFromSmiles(mapped_prod_smi)
        if prod_mol is None:
            return None
        # [CX4](O-anything)(carbon) - [CX4](carbon) - C=O  — very permissive
        broad_pat = Chem.MolFromSmarts("[CX4:1]([OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])")
        if broad_pat:
            matches = prod_mol.GetSubstructMatches(broad_pat)
            if matches:
                # :1 = query position 0 (Cb), :2 = query position 3 (Ca)
                # Identify query atom positions by map number
                cb_q, ca_q = None, None
                for qi in range(broad_pat.GetNumAtoms()):
                    mn = broad_pat.GetAtomWithIdx(qi).GetAtomMapNum()
                    if mn == 1:
                        cb_q = qi
                    elif mn == 2:
                        ca_q = qi
                if cb_q is not None and ca_q is not None:
                    cb_idx = matches[0][cb_q]
                    ca_idx = matches[0][ca_q]
                    return ca_idx, cb_idx
        return None
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
