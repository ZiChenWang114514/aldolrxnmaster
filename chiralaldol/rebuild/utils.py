"""Shared utility functions for V4 rebuild pipeline."""

import re
import logging
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger("rebuild_v4")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_mol(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES, return None on failure (suppresses RDKit warnings)."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol
    except Exception:
        return None


def canonical_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    """Canonicalize SMILES, preserving stereochemistry. Returns None on failure."""
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def count_defined_stereocenters(mol: Chem.Mol) -> int:
    """Count stereocenters with defined (R/S) CIP codes."""
    if mol is None:
        return 0
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return sum(1 for _, cip in centers if cip in ("R", "S"))
    except Exception:
        return 0


def mol_weight(smiles: str) -> float:
    """Compute molecular weight from SMILES. Returns 0.0 on failure."""
    mol = safe_mol(smiles)
    if mol is None:
        return 0.0
    return Descriptors.MolWt(mol)


def has_substructure(smiles: str, smarts: str) -> bool:
    """Check if SMILES contains a SMARTS substructure."""
    mol = safe_mol(smiles)
    if mol is None:
        return False
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return False
    return mol.HasSubstructMatch(pat)


def split_reaction_smiles(rxn_smi: str) -> tuple[list[str], list[str]]:
    """Split reaction SMILES into (reactant_list, product_list)."""
    if not isinstance(rxn_smi, str) or ">>" not in rxn_smi:
        return [], []
    parts = rxn_smi.split(">>")
    if len(parts) != 2:
        return [], []
    reactants = [s.strip() for s in parts[0].split(".") if s.strip()]
    products = [s.strip() for s in parts[1].split(".") if s.strip()]
    return reactants, products


def parse_numeric_field(val) -> Optional[float]:
    """Parse a Reaxys numeric field that may contain ranges, semicolons, etc."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    # Handle semicolon-separated (multi-stage): take first value
    if ";" in s:
        s = s.split(";")[0].strip()
    # Handle range (e.g., "10 - 20"): take midpoint
    range_match = re.match(r"(-?\d+\.?\d*)\s*[-–]\s*(-?\d+\.?\d*)", s)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        return (lo + hi) / 2
    # Handle "Ca." prefix (approximately)
    s = re.sub(r"^[Cc]a\.?\s*", "", s)
    # Try direct float parse
    try:
        return float(s)
    except ValueError:
        return None


def parse_semicolon_list(val) -> list[str]:
    """Parse a semicolon-separated Reaxys field into cleaned list."""
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).split(";") if x.strip()]
