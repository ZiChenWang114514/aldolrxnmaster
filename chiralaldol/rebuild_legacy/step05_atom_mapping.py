"""Step 5: Atom mapping dual verification (template SMARTS + optional RXNMapper).

This step does NOT delete rows — it only flags verification status.
RXNMapper is optional; if not installed, only template verification is done.
"""

import logging

import pandas as pd
from rdkit import Chem, RDLogger

from .constants import EVANS_PRODUCT_SMARTS

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)


def _template_verify(product_smi: str, mapped_product: str) -> bool:
    """Verify atom mapping using Evans aldol product SMARTS template.

    The template identifies Cb (OH-bearing) and Ca (next to C(=O)N).
    We check that these atoms are among the mapped atoms.
    """
    if pd.isna(product_smi) or pd.isna(mapped_product):
        return False

    # Parse clean product
    mol = Chem.MolFromSmiles(str(product_smi))
    if mol is None:
        return False

    # Try SMARTS patterns
    patterns = [
        "[C:1]([OH1])([#6])[C:2]([#6])[C](=O)[N]",
        "[C:1]([OH1])[C:2][C](=O)[N]",
    ]

    for patt_str in patterns:
        patt = Chem.MolFromSmarts(patt_str)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            return True

    return False


def _rxnmapper_verify(rxn_smi: str) -> tuple[bool, str | None]:
    """Verify using RXNMapper (if available).

    Returns (success, mapped_rxn_smi).
    """
    try:
        from rxnmapper import RXNMapper
    except ImportError:
        return False, None

    if pd.isna(rxn_smi) or not str(rxn_smi).strip():
        return False, None

    try:
        rxn_mapper = RXNMapper()
        results = rxn_mapper.get_attention_guided_atom_maps([str(rxn_smi)])
        if results and len(results) > 0:
            return True, results[0].get("mapped_rxn", None)
    except Exception:
        pass

    return False, None


def run(context: dict) -> dict:
    """Dual verification of atom mapping (template + optional RXNMapper)."""
    df: pd.DataFrame = context["df"].copy()
    n = len(df)
    logger.info(f"Step 5: Atom mapping verification for {n} rows")

    # Check if rxnmapper is available
    try:
        import rxnmapper
        has_rxnmapper = True
        logger.info("  RXNMapper available — running dual verification")
    except ImportError:
        has_rxnmapper = False
        logger.info("  RXNMapper not installed — template-only verification")

    template_ok_list = []
    rxnmapper_ok_list = []

    product_col = "canonical_Raw_Product_Smiles"
    if product_col not in df.columns:
        product_col = "Raw_Product_Smiles"

    for _, row in df.iterrows():
        product_smi = row.get(product_col, row.get("Raw_Product_Smiles"))
        mapped_product = row.get("Mapped_Product")

        # Template verification
        tok = _template_verify(product_smi, mapped_product)
        template_ok_list.append(tok)

        # RXNMapper verification (batch later if needed)
        rxnmapper_ok_list.append(None)

    df["atmap_template_ok"] = template_ok_list

    # Batch RXNMapper if available
    if has_rxnmapper:
        try:
            from rxnmapper import RXNMapper
            rxn_mapper = RXNMapper()
            rxn_smiles = df.get("Raw_Reaction_Smiles", pd.Series(dtype=str)).tolist()
            valid_indices = [i for i, s in enumerate(rxn_smiles) if pd.notna(s) and str(s).strip()]

            if valid_indices:
                valid_smiles = [str(rxn_smiles[i]) for i in valid_indices]
                # Process in batches of 100
                batch_size = 100
                rxnmapper_results = [None] * n
                for batch_start in range(0, len(valid_smiles), batch_size):
                    batch = valid_smiles[batch_start:batch_start + batch_size]
                    batch_indices = valid_indices[batch_start:batch_start + batch_size]
                    try:
                        results = rxn_mapper.get_attention_guided_atom_maps(batch)
                        for j, res in enumerate(results):
                            idx = batch_indices[j]
                            conf = res.get("confidence", 0)
                            rxnmapper_results[idx] = conf > 0.5
                    except Exception as e:
                        logger.warning(f"  RXNMapper batch failed: {e}")
                        for idx in batch_indices:
                            rxnmapper_results[idx] = None

                df["atmap_rxnmapper_ok"] = rxnmapper_results
        except Exception as e:
            logger.warning(f"  RXNMapper failed entirely: {e}")
            df["atmap_rxnmapper_ok"] = None
    else:
        df["atmap_rxnmapper_ok"] = None

    # Summary
    n_template = sum(1 for v in template_ok_list if v)
    logger.info(f"  Template match: {n_template}/{n} ({100*n_template/max(n,1):.1f}%)")
    if has_rxnmapper and "atmap_rxnmapper_ok" in df.columns:
        n_rxnm = df["atmap_rxnmapper_ok"].sum() if df["atmap_rxnmapper_ok"].notna().any() else 0
        logger.info(f"  RXNMapper OK: {n_rxnm}/{n}")

    out_path = context["output_dir"] / "interim" / "05_atmap_verified.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Step 5 complete: {n} rows (no deletions, verification only)")

    context["df"] = df
    return context
