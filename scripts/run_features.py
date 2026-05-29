#!/usr/bin/env python3
"""V4 Feature Engineering: conformer generation + steric + chirality + auxiliary + integration.

Usage:
    conda run -n aldol-rxn python scripts/run_features_v4.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import CLEAN_DIR, FEAT_DIR

CLEAN_CSV = CLEAN_DIR / "substrate_aldol_clean.csv"
CONF_DIR = FEAT_DIR / "conformers"

from chiralaldol.rebuild_legacy.step10_conformers import TIMEOUT_SEC, _worker_fn
from chiralaldol.rebuild_legacy.step11_steric import (
    ALDEHYDE_STERIC_NAMES,
    ENOLATE_STERIC_NAMES,
    _compute_aldehyde_steric,
    _compute_enolate_steric,
)

# ═══════════════════════════ Auxiliary C4 SMARTS ═══════════════════════════
# Patterns that match the CHIRAL center (C4) as the first atom (:1).
# For Evans/Crimmins, C4 is directly matched; for Oppolzer/Myers, use fallback.
_AUX_C4_SMARTS = {
    "evans":              "[C:1]1([*])COC(=O)N1",
    "crimmins_thione":    "[C:1]1([*])CSC(=S)N1",
    "crimmins_oxathione": "[C:1]1([*])COC(=S)N1",
    "super_quat":         "[C:1]1([*])([*])COC(=O)N1",
    "oppolzer":           "[C:1]1([*])[C][C]S(=O)(=O)N1",  # 5-membered sultam ring
}
_AUX_C4_PATS = {k: Chem.MolFromSmarts(v) for k, v in _AUX_C4_SMARTS.items()}

# R-group classification SMARTS (from V3 step06_auxiliary.py)
RGROUP_SMARTS = {
    "benzyl":     "[CH2]c1ccccc1",
    "isopropyl":  "[CH]([CH3])[CH3]",
    "phenyl":     "c1ccccc1",
    "tert_butyl": "[C]([CH3])([CH3])[CH3]",
    "methyl":     "[CH3]",
    "indanyl":    "C1CCc2ccccc21",
}
_RGROUP_PATS = {k: Chem.MolFromSmarts(v) for k, v in RGROUP_SMARTS.items()}

# Chirality environment SMARTS: locate alpha carbon of the acyl group
_ALPHA_SMARTS_LIST = [
    "[CH2,CH;X3,X4:1]-[CX3:2](=[OX1])-[NX3]",   # standard ketone-N
    "[C:1][CX3](=[OX1])[NX3,OX2]",                 # fallback: broader
    "[C:1][CX3](=[OX1])",                           # last resort: any acyl
]
_ALPHA_PATS = [Chem.MolFromSmarts(s) for s in _ALPHA_SMARTS_LIST]


def generate_conformers(smiles_list: list[str], label: str) -> dict[str, dict]:
    """Generate conformer ensembles for unique SMILES, with incremental caching."""
    cache_path = CONF_DIR / f"{label}_conformers.pkl"
    ensembles = {}
    if cache_path.exists():
        print(f"  Loading cached {label} conformers from {cache_path}")
        with open(cache_path, "rb") as f:
            ensembles = pickle.load(f)

    unique_smiles = sorted(set(s for s in smiles_list if isinstance(s, str) and s.strip()))
    missing = [s for s in unique_smiles if s not in ensembles]

    if not missing:
        print(f"  {label}: all {len(unique_smiles)} SMILES cached, skipping generation")
        return ensembles

    print(f"  Generating {label} conformers for {len(missing)} new SMILES ({len(ensembles)} cached)...")

    from multiprocessing import Pool
    args = [(smi, TIMEOUT_SEC) for smi in missing]
    with Pool(8) as pool:
        results = pool.map(_worker_fn, args)

    n_new = 0
    for smi, result in zip(missing, results):
        if result is not None:
            ensembles[smi] = result
            n_new += 1

    n_ok = sum(1 for s in unique_smiles if s in ensembles)
    print(f"  {label}: {n_ok}/{len(unique_smiles)} total ({n_new} newly generated)")

    with open(cache_path, "wb") as f:
        pickle.dump(ensembles, f)
    return ensembles


def compute_steric_features(df: pd.DataFrame, ketone_ensembles: dict, aldehyde_ensembles: dict):
    """Compute 34d steric features for all rows."""
    print("Computing steric features...")

    ketone_col = "canonical_ketone_smiles"
    aldehyde_col = "canonical_aldehyde_smiles"

    enolate_rows = []
    aldehyde_rows = []

    for i, row in df.iterrows():
        # Enolate steric (from ketone conformers)
        ksmi = row.get(ketone_col)
        if isinstance(ksmi, str) and ksmi in ketone_ensembles:
            desc = _compute_enolate_steric(ketone_ensembles[ksmi])
        else:
            desc = None
        enolate_rows.append(desc if desc else {k: np.nan for k in ENOLATE_STERIC_NAMES})

        # Aldehyde steric
        asmi = row.get(aldehyde_col)
        if isinstance(asmi, str) and asmi in aldehyde_ensembles:
            desc = _compute_aldehyde_steric(aldehyde_ensembles[asmi])
        else:
            desc = None
        aldehyde_rows.append(desc if desc else {k: np.nan for k in ALDEHYDE_STERIC_NAMES})

    enolate_df = pd.DataFrame(enolate_rows)
    aldehyde_df = pd.DataFrame(aldehyde_rows)

    n_enolate_ok = enolate_df.dropna(how="all").shape[0]
    n_aldehyde_ok = aldehyde_df.dropna(how="all").shape[0]
    print(f"  Enolate steric: {n_enolate_ok}/{len(df)} ({enolate_df.shape[1]}d)")
    print(f"  Aldehyde steric: {n_aldehyde_ok}/{len(df)} ({aldehyde_df.shape[1]}d)")

    steric_df = pd.concat([enolate_df, aldehyde_df], axis=1)
    return steric_df


def compute_chirality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7d auxiliary chirality features from ketone SMILES (no product leakage).

    Features:
      chiral_dominant_sign  — sign of Σ(CIP_sign) across all stereocenters {-1,0,+1}
      chiral_primary_cip_R  — first stereocenter: R=1, S=0, unknown=-1
      chiral_n_defined      — number of defined (R/S) stereocenters
      chiral_sum_signs      — raw sum of CIP signs (R=+1, S=-1)
      chiral_aux_c4_R       — CIP at the auxiliary C4 via SMARTS {0,1,-1}
      chiral_aux_c4_match   — whether C4 SMARTS matched {0,1}
      chiral_aux_mw         — ketone molecular weight
    """
    print("Computing chirality features from ketone SMILES...")
    rows = []
    for _, row in df.iterrows():
        ksmi = row.get("canonical_ketone_smiles")
        aux_type = row.get("auxiliary_type", "")
        feat = _extract_chirality_one(ksmi, aux_type)
        rows.append(feat)

    chiral_df = pd.DataFrame(rows)
    n_with_stereo = (chiral_df["chiral_n_defined"] > 0).sum()
    n_c4_match = (chiral_df["chiral_aux_c4_match"] > 0).sum()
    print(f"  Ketones with defined stereocenters: {n_with_stereo}/{len(df)}")
    print(f"  Auxiliary C4 SMARTS matched: {n_c4_match}/{len(df)}")
    return chiral_df


def _extract_chirality_one(ketone_smi, aux_type: str) -> dict:
    """Extract chirality features for one ketone."""
    feat = {
        "chiral_dominant_sign": 0.0,
        "chiral_primary_cip_R": -1.0,
        "chiral_n_defined": 0,
        "chiral_sum_signs": 0.0,
        "chiral_aux_c4_R": -1.0,
        "chiral_aux_c4_match": 0.0,
        "chiral_aux_mw": 0.0,
    }
    if not isinstance(ketone_smi, str) or not ketone_smi.strip():
        return feat

    mol = Chem.MolFromSmiles(ketone_smi)
    if mol is None:
        return feat

    feat["chiral_aux_mw"] = Descriptors.MolWt(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Block A: global CIP summary
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    defined = [(idx, cip) for idx, cip in centers if cip in ("R", "S")]
    feat["chiral_n_defined"] = len(defined)

    if defined:
        signs = [+1 if cip == "R" else -1 for _, cip in defined]
        feat["chiral_sum_signs"] = float(sum(signs))
        feat["chiral_dominant_sign"] = float(np.sign(sum(signs)))
        first_cip = defined[0][1]
        feat["chiral_primary_cip_R"] = 1.0 if first_cip == "R" else 0.0

    # Block B: SMARTS-based C4 CIP extraction
    # Try specific C4 pattern for Evans/Crimmins types
    c4_idx = None
    for pat_name, pat in _AUX_C4_PATS.items():
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        if matches:
            c4_idx = matches[0][0]
            feat["chiral_aux_c4_match"] = 1.0
            break

    if c4_idx is not None:
        atom = mol.GetAtomWithIdx(c4_idx)
        cip = atom.GetPropsAsDict().get("_CIPCode")
        if cip == "R":
            feat["chiral_aux_c4_R"] = 1.0
        elif cip == "S":
            feat["chiral_aux_c4_R"] = 0.0
    elif defined:
        # Fallback: mark as unknown (-1) rather than using an arbitrary stereocenter.
        # Using the first-found stereocenter injects noise (wrong center picked).
        feat["chiral_aux_c4_R"] = -1.0
        feat["chiral_aux_c4_match"] = 0.0

    return feat


def compute_rgroup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7d R-group features from ketone SMILES.

    Features:
      aux_rg_benzyl, aux_rg_isopropyl, aux_rg_phenyl, aux_rg_tert_butyl,
      aux_rg_methyl, aux_rg_indanyl, aux_rg_other
    Note: aux_oppolzer moved to auxiliary one-hot (compute_auxiliary_features).
    """
    print("Computing R-group features from ketone SMILES...")
    rg_names = ["benzyl", "isopropyl", "phenyl", "tert_butyl", "methyl", "indanyl", "other"]
    rows = []
    for _, row in df.iterrows():
        ksmi = row.get("canonical_ketone_smiles")
        aux_type = row.get("auxiliary_type", "")
        feat = {f"aux_rg_{rg}": 0 for rg in rg_names}

        if not isinstance(ksmi, str) or not ksmi.strip():
            rows.append(feat)
            continue

        mol = Chem.MolFromSmiles(ksmi)
        if mol is None:
            rows.append(feat)
            continue

        # Find C4 via SMARTS
        c4_idx = None
        for pat in _AUX_C4_PATS.values():
            if pat is None:
                continue
            matches = mol.GetSubstructMatches(pat)
            if matches:
                c4_idx = matches[0][0]
                break

        if c4_idx is None:
            rows.append(feat)
            continue

        # Find substituent root: neighbor of C4 not in the same ring
        atom = mol.GetAtomWithIdx(c4_idx)
        ring_info = mol.GetRingInfo()
        c4_ring_atoms = set()
        for ring in ring_info.AtomRings():
            if c4_idx in ring:
                c4_ring_atoms.update(ring)

        substituent_root = None
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in c4_ring_atoms:
                substituent_root = nb.GetIdx()
                break

        if substituent_root is not None:
            for rg_name, rg_pat in _RGROUP_PATS.items():
                if rg_pat is None:
                    continue
                rg_matches = mol.GetSubstructMatches(rg_pat)
                for match in rg_matches:
                    if substituent_root in match:
                        feat[f"aux_rg_{rg_name}"] = 1
                        rows.append(feat)
                        break
                else:
                    continue
                break
            else:
                feat["aux_rg_other"] = 1
                rows.append(feat)
        else:
            rows.append(feat)

    rg_df = pd.DataFrame(rows)
    # Log distribution
    for col in rg_df.columns:
        n = rg_df[col].sum()
        if n > 0:
            print(f"  {col}: {n}")
    return rg_df


def compute_chirality_environment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 21d chirality environment features encoding spatial distribution
    of stereocenters near the reaction site.

    Ketone (15d): distance-layered stereocenter counts, gradients, molecular descriptors.
    Aldehyde (6d): global stereocenter summary + molecular descriptors.
    """
    print("Computing chirality environment features...")
    rows = []
    n_alpha_found = 0

    for _, row in df.iterrows():
        feat = {}

        # --- Ketone chirality environment (15d) ---
        ksmi = row.get("canonical_ketone_smiles")
        ket_feat = _compute_ketone_chirality_env(ksmi)
        feat.update(ket_feat)

        if ket_feat.get("chiralenv_ket_n_total_stereo", 0) > 0:
            n_alpha_found += 1

        # --- Aldehyde chirality environment (6d) ---
        asmi = row.get("canonical_aldehyde_smiles")
        ald_feat = _compute_aldehyde_chirality_env(asmi)
        feat.update(ald_feat)

        rows.append(feat)

    env_df = pd.DataFrame(rows)
    n_alpha_match = (env_df.get("chiralenv_ket_nearest_chiral_dist", pd.Series(dtype=float)) < 99).sum()
    print(f"  Alpha carbon found (SMARTS matched): {n_alpha_found}/{len(df)}")
    print(f"  Rows with nearby stereocenters: {n_alpha_match}/{len(df)}")
    return env_df


def _compute_ketone_chirality_env(smi) -> dict:
    """15d ketone chirality environment."""
    feat = {
        "chiralenv_ket_n_stereo_3bond": 0, "chiralenv_ket_n_stereo_4bond": 0,
        "chiralenv_ket_n_stereo_5bond": 0,
        "chiralenv_ket_chirality_gradient": 0.0,
        "chiralenv_ket_nearest_chiral_sign": 0.0,
        "chiralenv_ket_nearest_chiral_dist": 99.0,
        "chiralenv_ket_alpha_is_chiral": 0, "chiralenv_ket_alpha_chiral_sign": 0.0,
        "chiralenv_ket_chirality_product_3bond": 0.0,
        "chiralenv_ket_nearest_pair_relation": 0.0,
        "chiralenv_ket_n_total_stereo": 0,
        "chiralenv_ket_mw": 0.0, "chiralenv_ket_n_heavy": 0,
        "chiralenv_ket_frac_sp3": 0.0, "chiralenv_ket_aux_ring_size": 0,
    }
    if not isinstance(smi, str) or not smi.strip():
        return feat

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return feat

    feat["chiralenv_ket_mw"] = Descriptors.MolWt(mol)
    feat["chiralenv_ket_n_heavy"] = mol.GetNumHeavyAtoms()
    feat["chiralenv_ket_frac_sp3"] = Descriptors.FractionCSP3(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Find alpha carbon
    alpha_idx = None
    for pat in _ALPHA_PATS:
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        if matches:
            alpha_idx = matches[0][0]
            break

    if alpha_idx is None:
        return feat

    # Find all defined stereocenters
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    defined = [(idx, cip) for idx, cip in centers if cip in ("R", "S")]
    feat["chiralenv_ket_n_total_stereo"] = len(defined)

    # Check if alpha itself is chiral (bug fix: skip self in distance calc)
    for cidx, cip in defined:
        if cidx == alpha_idx:
            feat["chiralenv_ket_alpha_is_chiral"] = 1
            feat["chiralenv_ket_alpha_chiral_sign"] = 1.0 if cip == "R" else -1.0
            break

    # Distance-layered counting (≤3, ≤4, ≤5 bonds)
    gradient = 0.0
    nearest_dist = 99
    nearest_sign = 0.0
    signs_3bond = []

    for cidx, cip in defined:
        if cidx == alpha_idx:
            continue  # skip self
        try:
            path = Chem.GetShortestPath(mol, alpha_idx, cidx)
            dist = len(path) - 1
        except Exception:
            continue

        sign = 1.0 if cip == "R" else -1.0

        if dist <= 3:
            feat["chiralenv_ket_n_stereo_3bond"] += 1
            signs_3bond.append(sign)
        if dist <= 4:
            feat["chiralenv_ket_n_stereo_4bond"] += 1
        if dist <= 5:
            feat["chiralenv_ket_n_stereo_5bond"] += 1

        if dist > 0:
            gradient += sign / dist

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_sign = sign

    feat["chiralenv_ket_chirality_gradient"] = gradient
    feat["chiralenv_ket_nearest_chiral_dist"] = float(nearest_dist)
    feat["chiralenv_ket_nearest_chiral_sign"] = nearest_sign

    # Product of CIP signs within 3 bonds
    if signs_3bond:
        product = 1.0
        for s in signs_3bond:
            product *= s
        feat["chiralenv_ket_chirality_product_3bond"] = product

    # Nearest pair relation
    if len(defined) >= 2:
        non_self = [(idx, cip) for idx, cip in defined if idx != alpha_idx]
        if len(non_self) >= 2:
            s1 = 1.0 if non_self[0][1] == "R" else -1.0
            s2 = 1.0 if non_self[1][1] == "R" else -1.0
            feat["chiralenv_ket_nearest_pair_relation"] = s1 * s2

    # Smallest ring containing N (auxiliary ring size)
    ring_info = mol.GetRingInfo()
    min_n_ring = 0
    for ring in ring_info.AtomRings():
        has_n = any(mol.GetAtomWithIdx(idx).GetAtomicNum() == 7 for idx in ring)
        if has_n:
            if min_n_ring == 0 or len(ring) < min_n_ring:
                min_n_ring = len(ring)
    feat["chiralenv_ket_aux_ring_size"] = min_n_ring

    return feat


def _compute_aldehyde_chirality_env(smi) -> dict:
    """6d aldehyde chirality environment."""
    feat = {
        "chiralenv_ald_n_stereo": 0,
        "chiralenv_ald_chirality_sum": 0.0,
        "chiralenv_ald_chirality_product": 0.0,
        "chiralenv_ald_mw": 0.0,
        "chiralenv_ald_n_heavy": 0,
        "chiralenv_ald_frac_sp3": 0.0,
    }
    if not isinstance(smi, str) or not smi.strip():
        return feat

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return feat

    feat["chiralenv_ald_mw"] = Descriptors.MolWt(mol)
    feat["chiralenv_ald_n_heavy"] = mol.GetNumHeavyAtoms()
    feat["chiralenv_ald_frac_sp3"] = Descriptors.FractionCSP3(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    defined = [(idx, cip) for idx, cip in centers if cip in ("R", "S")]
    feat["chiralenv_ald_n_stereo"] = len(defined)

    if defined:
        signs = [1.0 if cip == "R" else -1.0 for _, cip in defined]
        feat["chiralenv_ald_chirality_sum"] = sum(signs)
        product = 1.0
        for s in signs:
            product *= s
        feat["chiralenv_ald_chirality_product"] = product

    return feat


def compute_aldehyde_priority_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 8d aldehyde CIP-priority proxy features.

    These help the model predict how the aldehyde substituent affects
    absolute CIP assignment at Cb (the new carbinol center).
    """
    print("Computing aldehyde CIP-priority features...")
    _ald_pat = Chem.MolFromSmarts("[CX3H1:1](=O)")  # aldehyde C
    rows = []
    for _, row in df.iterrows():
        smi = row.get("canonical_aldehyde_smiles")
        feat = _compute_ald_priority_one(smi, _ald_pat)
        rows.append(feat)
    pri_df = pd.DataFrame(rows)
    for col in pri_df.columns:
        v = pri_df[col].mean()
        if v != 0:
            print(f"  {col}: mean={v:.3f}")
    return pri_df


def _compute_ald_priority_one(smi, ald_pat) -> dict:
    feat = {
        "ald_pri_is_aromatic": 0, "ald_pri_alpha_branching": 0,
        "ald_pri_max_atomic_num": 6, "ald_pri_has_halogen": 0,
        "ald_pri_has_heteroatom": 0, "ald_pri_n_rings": 0,
        "ald_pri_chain_length": 0, "ald_pri_priority_proxy": 0.0,
    }
    if not isinstance(smi, str) or not smi.strip():
        return feat
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return feat

    feat["ald_pri_n_rings"] = mol.GetRingInfo().NumRings()

    # Find aldehyde carbon and its alpha neighbor
    matches = mol.GetSubstructMatches(ald_pat) if ald_pat else []
    if not matches:
        return feat
    ald_c_idx = matches[0][0]
    ald_c = mol.GetAtomWithIdx(ald_c_idx)

    # Alpha carbon = neighbor of aldehyde C that is not O
    alpha_idx = None
    for nb in ald_c.GetNeighbors():
        if nb.GetAtomicNum() != 8:  # not oxygen
            alpha_idx = nb.GetIdx()
            break
    if alpha_idx is None:
        return feat

    alpha = mol.GetAtomWithIdx(alpha_idx)
    feat["ald_pri_is_aromatic"] = int(alpha.GetIsAromatic())

    # Alpha branching: non-H neighbors excluding aldehyde C
    non_h_neighbors = [nb for nb in alpha.GetNeighbors() if nb.GetIdx() != ald_c_idx]
    feat["ald_pri_alpha_branching"] = len(non_h_neighbors)

    # Max atomic number among alpha neighbors
    if non_h_neighbors:
        feat["ald_pri_max_atomic_num"] = max(nb.GetAtomicNum() for nb in non_h_neighbors)

    # Halogen / heteroatom check (whole molecule)
    halogens = {9, 17, 35, 53}
    heteroatoms = {7, 8, 16}
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum in halogens:
            feat["ald_pri_has_halogen"] = 1
        if anum in heteroatoms and atom.GetIdx() != ald_c_idx:
            # Exclude the aldehyde O itself
            is_ald_o = (anum == 8 and any(
                nb.GetIdx() == ald_c_idx for nb in atom.GetNeighbors()
            ))
            if not is_ald_o:
                feat["ald_pri_has_heteroatom"] = 1

    # Chain length: longest path from alpha C (BFS, simple)
    visited = {ald_c_idx}
    queue = [(alpha_idx, 0)]
    max_depth = 0
    while queue:
        curr, depth = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        max_depth = max(max_depth, depth)
        for nb in mol.GetAtomWithIdx(curr).GetNeighbors():
            if nb.GetIdx() not in visited:
                queue.append((nb.GetIdx(), depth + 1))
    feat["ald_pri_chain_length"] = max_depth

    # CIP priority proxy: sum of Morgan ranks of alpha's neighbors
    from rdkit.Chem import rdMolDescriptors
    morgan_info = {}
    rdMolDescriptors.GetMorganFingerprint(mol, 2, bitInfo=morgan_info)
    # Use atom invariants (connectivity) as proxy
    rank_sum = 0.0
    for nb in non_h_neighbors:
        rank_sum += nb.GetAtomicNum() + 0.1 * len(list(nb.GetNeighbors()))
    feat["ald_pri_priority_proxy"] = rank_sum

    return feat


def compute_delta_chirality_features(df: pd.DataFrame) -> pd.DataFrame:
    """B1: Delta chirality descriptors (16d) — [37] Baimacheva 2025.

    Computes FP(ketone) - FP(enantiomer) to isolate chirality-sensitive bits,
    then PCA to 16d. Uses ketone SMILES only (no product leakage).
    """
    import re

    from rdkit.Chem import AllChem
    from sklearn.decomposition import PCA

    print("Computing delta chirality features...")

    def _flip_chirality(smi):
        """Flip all @/@@: @ → @@, @@ → @."""
        if smi is None or not isinstance(smi, str):
            return None
        return re.sub(r'@@|@', lambda m: '@' if m.group() == '@@' else '@@', smi)

    deltas = []
    valid_idx = []
    for i, row in df.iterrows():
        smi = row.get("canonical_ketone_smiles")
        if pd.isna(smi) or not isinstance(smi, str):
            deltas.append(np.zeros(512))
            continue

        mol = Chem.MolFromSmiles(smi)
        enan_smi = _flip_chirality(smi)
        enan_mol = Chem.MolFromSmiles(enan_smi) if enan_smi else None

        if mol is None or enan_mol is None:
            deltas.append(np.zeros(512))
            continue

        fp_orig = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512, useChirality=True)
        fp_enan = AllChem.GetMorganFingerprintAsBitVect(enan_mol, radius=3, nBits=512, useChirality=True)
        delta = np.array(fp_orig, dtype=np.float32) - np.array(fp_enan, dtype=np.float32)
        deltas.append(delta)
        if np.any(delta != 0):
            valid_idx.append(i)

    delta_matrix = np.vstack(deltas)
    n_nonzero = len(valid_idx)
    print(f"  Non-zero delta vectors: {n_nonzero}/{len(df)} ({n_nonzero/len(df)*100:.1f}%)")

    # PCA to 16d
    n_components = min(16, delta_matrix.shape[1], n_nonzero)
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components, random_state=42)
    delta_pca = pca.fit_transform(delta_matrix)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {n_components}d explained variance: {explained:.3f}")

    # Pad to 16d if needed
    if delta_pca.shape[1] < 16:
        pad = np.zeros((delta_pca.shape[0], 16 - delta_pca.shape[1]))
        delta_pca = np.hstack([delta_pca, pad])

    cols = [f"delta_chiral_{i}" for i in range(16)]
    return pd.DataFrame(delta_pca, columns=cols)


def compute_chiral_determinant(df: pd.DataFrame) -> pd.DataFrame:
    """B3: Continuous chirality determinant (3d) — [35] ChiDeK ICLR 2026.

    Computes signed tetrahedral volume for each stereocenter from 3D conformer.
    Continuous measure replaces discrete R/S encoding.
    """
    from rdkit.Chem import AllChem

    print("Computing chiral determinant features...")

    results = []
    for _, row in df.iterrows():
        smi = row.get("canonical_ketone_smiles")
        feat = {"chiral_det_mean": 0.0, "chiral_det_max": 0.0, "chiral_det_abs_sum": 0.0}

        if pd.isna(smi) or not isinstance(smi, str):
            results.append(feat)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(feat)
            continue

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            # Fallback: try without ETKDG constraints
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status != 0:
            results.append(feat)
            continue

        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        dets = []
        for atom_idx, _ in chiral_centers:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) < 4:
                continue

            # Take first 4 neighbors
            p0 = np.array(conf.GetAtomPosition(atom_idx))
            coords = [np.array(conf.GetAtomPosition(neighbors[j])) for j in range(4)]

            # Signed volume = det([v1-v0, v2-v0, v3-v0])
            v = [coords[j] - coords[0] for j in range(1, 4)]
            det_val = np.linalg.det(np.array(v))
            dets.append(det_val)

        if dets:
            feat["chiral_det_mean"] = float(np.mean(dets))
            feat["chiral_det_max"] = float(max(dets, key=abs))
            feat["chiral_det_abs_sum"] = float(np.sum(np.abs(dets)))

        results.append(feat)

    det_df = pd.DataFrame(results)
    n_nonzero = (det_df["chiral_det_abs_sum"] > 0).sum()
    print(f"  Non-zero determinant: {n_nonzero}/{len(df)} ({n_nonzero/len(df)*100:.1f}%)")
    return det_df


def compute_auxiliary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute auxiliary one-hot features (6d, unchanged for backward compat)."""
    aux_types = ["evans", "crimmins_thione", "crimmins_oxathione", "oppolzer", "other_auxiliary", "myers"]
    aux_feats = {}
    for atype in aux_types:
        col_name = f"aux_{atype}"
        aux_feats[col_name] = (df["auxiliary_type"] == atype).astype(int)

    # n_defined_stereocenters as a feature
    aux_feats["n_defined_stereocenters"] = df["n_defined_stereocenters"].fillna(2)

    return pd.DataFrame(aux_feats)


def integrate_features(
    df: pd.DataFrame,
    steric_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    chiral_df: pd.DataFrame = None,
    rgroup_df: pd.DataFrame = None,
    chiralenv_df: pd.DataFrame = None,
    aldpri_df: pd.DataFrame = None,
    delta_chiral_df: pd.DataFrame = None,
    chiral_det_df: pd.DataFrame = None,
):
    """Integrate all features into a single matrix.

    First 84 columns = steric(34) + conditions(50) + auxiliary(6) for backward compat.
    New blocks appended: chirality(7) + rgroup(8) + chiralenv(21) + aldpri(8) + delta(16) + det(3).
    """
    # Load condition features from V4 pipeline output
    cond_path = CLEAN_DIR / "condition_features.csv"
    cond_df = pd.read_csv(cond_path)
    print(f"  Conditions: {cond_df.shape[1]}d")

    # Combine: backward-compatible 84d first, then new blocks
    blocks = [
        steric_df.reset_index(drop=True),
        cond_df.reset_index(drop=True),
        aux_df.reset_index(drop=True),
    ]
    if chiral_df is not None:
        blocks.append(chiral_df.reset_index(drop=True))
        print(f"  Chirality: {chiral_df.shape[1]}d")
    if rgroup_df is not None:
        blocks.append(rgroup_df.reset_index(drop=True))
        print(f"  R-group: {rgroup_df.shape[1]}d")
    if chiralenv_df is not None:
        blocks.append(chiralenv_df.reset_index(drop=True))
        print(f"  Chirality env: {chiralenv_df.shape[1]}d")
    if aldpri_df is not None:
        blocks.append(aldpri_df.reset_index(drop=True))
        print(f"  Ald priority: {aldpri_df.shape[1]}d")
    if delta_chiral_df is not None:
        blocks.append(delta_chiral_df.reset_index(drop=True))
        print(f"  Delta chirality: {delta_chiral_df.shape[1]}d")
    if chiral_det_df is not None:
        blocks.append(chiral_det_df.reset_index(drop=True))
        print(f"  Chiral determinant: {chiral_det_df.shape[1]}d")

    feat_df = pd.concat(blocks, axis=1)
    print(f"  Combined: {feat_df.shape[1]}d before NaN handling")

    # Count NaN per row
    nan_per_row = feat_df.isna().sum(axis=1)
    print(f"  Rows with any NaN: {(nan_per_row > 0).sum()}/{len(feat_df)}")
    print(f"  NaN distribution: mean={nan_per_row.mean():.1f}, max={nan_per_row.max()}")

    # Fill NaN with column median (don't drop rows - preserve dataset size)
    for col in feat_df.columns:
        if feat_df[col].isna().any():
            median_val = feat_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            feat_df[col] = feat_df[col].fillna(median_val)

    return feat_df


def main():
    t0 = time.time()
    print("=" * 60)
    print("V4 FEATURE ENGINEERING")
    print("=" * 60)

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    CONF_DIR.mkdir(parents=True, exist_ok=True)

    # Load V4 clean data
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df)} rows from {CLEAN_CSV}")

    # --- B1: Conformer generation ---
    print("\n--- Phase B1: Conformer Generation ---")
    ketone_ensembles = generate_conformers(
        df["canonical_ketone_smiles"].dropna().tolist(), "ketone"
    )
    aldehyde_ensembles = generate_conformers(
        df["canonical_aldehyde_smiles"].dropna().tolist(), "aldehyde"
    )

    # --- B2: Steric features ---
    print("\n--- Phase B2: Steric Features ---")
    steric_df = compute_steric_features(df, ketone_ensembles, aldehyde_ensembles)
    steric_df.to_csv(FEAT_DIR / "steric_features.csv", index=False)
    print(f"  Saved steric_features.csv ({steric_df.shape})")

    # --- B3: Chirality features ---
    print("\n--- Phase B3: Auxiliary Chirality Features ---")
    chiral_df = compute_chirality_features(df)
    print(f"  Chirality features: {chiral_df.shape[1]}d")

    # --- B3b: R-group features ---
    print("\n--- Phase B3b: R-group Features ---")
    rgroup_df = compute_rgroup_features(df)
    print(f"  R-group features: {rgroup_df.shape[1]}d")

    # --- B3c: Chirality environment ---
    print("\n--- Phase B3c: Chirality Environment Features ---")
    chiralenv_df = compute_chirality_environment(df)
    print(f"  Chirality env features: {chiralenv_df.shape[1]}d")

    # --- B3d: Aldehyde CIP-priority features ---
    print("\n--- Phase B3d: Aldehyde CIP-Priority Features ---")
    aldpri_df = compute_aldehyde_priority_features(df)
    print(f"  Aldehyde priority features: {aldpri_df.shape[1]}d")

    # --- B3e: Delta chirality features ---
    print("\n--- Phase B3e: Delta Chirality Features ---")
    delta_chiral_df = compute_delta_chirality_features(df)
    print(f"  Delta chirality features: {delta_chiral_df.shape[1]}d")

    # --- B3f: Continuous chirality determinant ---
    print("\n--- Phase B3f: Continuous Chirality Determinant ---")
    chiral_det_df = compute_chiral_determinant(df)
    print(f"  Chiral determinant features: {chiral_det_df.shape[1]}d")

    # --- B4: Auxiliary features ---
    print("\n--- Phase B4: Auxiliary Features ---")
    aux_df = compute_auxiliary_features(df)
    print(f"  Auxiliary features: {aux_df.shape[1]}d")

    # --- B5: Feature integration ---
    print("\n--- Phase B5: Feature Integration ---")
    feat_df = integrate_features(
        df, steric_df, aux_df,
        chiral_df=chiral_df,
        rgroup_df=rgroup_df,
        chiralenv_df=chiralenv_df,
        aldpri_df=aldpri_df,
        delta_chiral_df=delta_chiral_df,
        chiral_det_df=chiral_det_df,
    )
    feat_df.to_csv(FEAT_DIR / "v4_features.csv", index=False)
    print(f"  Saved v4_features.csv ({feat_df.shape})")

    # Labels
    labels = df[["label_Ca", "label_Cb", "label_SA", "label_joint", "label_confidence"]].copy()
    # 3D dihedral-based syn/anti and RS-SynAnti 4-class label
    if "label_syn_anti_3d" in df.columns:
        labels["label_syn_anti_3d"] = df["label_syn_anti_3d"].values
        both_valid = labels["label_Ca"].notna() & labels["label_syn_anti_3d"].notna()
        labels["label_joint_sa"] = np.where(
            both_valid,
            labels["label_Ca"].astype("Int64") * 2 + labels["label_syn_anti_3d"].astype("Int64"),
            pd.NA,
        )
        n_valid = int(both_valid.sum())
        print(f"  label_joint_sa: {n_valid}/{len(labels)} valid ({len(labels)-n_valid} NaN)")
    labels.to_csv(FEAT_DIR / "labels.csv", index=False)

    # Feature manifest
    cond_dims = pd.read_csv(CLEAN_DIR / "condition_features.csv").shape[1]
    manifest = {
        "n_features": feat_df.shape[1],
        "n_samples": feat_df.shape[0],
        "feature_names": list(feat_df.columns),
        "steric_dims": steric_df.shape[1],
        "condition_dims": cond_dims,
        "auxiliary_dims": aux_df.shape[1],
        "chirality_dims": chiral_df.shape[1],
        "rgroup_dims": rgroup_df.shape[1],
        "chiralenv_dims": chiralenv_df.shape[1],
        "aldpri_dims": aldpri_df.shape[1],
        "delta_chiral_dims": delta_chiral_df.shape[1],
        "chiral_det_dims": chiral_det_df.shape[1],
    }
    with open(FEAT_DIR / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Final feature matrix: {feat_df.shape[0]} rows × {feat_df.shape[1]} features")
    print("Done.")


if __name__ == "__main__":
    main()
