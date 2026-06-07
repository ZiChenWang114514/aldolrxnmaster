"""B0a — explicit CIP-priority-flip decoupling feature at the carbinol carbon.

The carbinol (Cb) absolute CIP = f(physical face, CIP priority of its substituents).
At Cb the two carbon branches are R_ald (from the aldehyde) and Cα (from the enolate).
Whether R_ald outranks the Cα branch in CIP priority decides how a given physical
face maps to R vs S. The model has crude proxies (is_aromatic) but not this explicit
branch comparison — so it must entangle "physical face selection" with "CIP flip".

This computes, on the PRODUCT, a branch-priority comparison at Cb so the model can
decouple the two. Reuses the broad SMARTS from step08 (:1=Cb, :2=Cα).
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_BROAD = Chem.MolFromSmarts("[CX4:1]([OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])")
_CB_Q = _CA_Q = None
for _qi in range(_BROAD.GetNumAtoms()):
    _mn = _BROAD.GetAtomWithIdx(_qi).GetAtomMapNum()
    if _mn == 1:
        _CB_Q = _qi
    elif _mn == 2:
        _CA_Q = _qi

FEAT_NAMES = ["aldcip_ald_outranks_alpha", "aldcip_first_Z_diff",
              "aldcip_ald_branch_arom", "aldcip_match"]


def _branch_priority_seq(mol, start, blocked, max_depth=4):
    """BFS atomic-number sequence (descending) from `start`, not crossing `blocked` (=Cb)."""
    seq, seen, frontier = [], {blocked}, [(start, 0)]
    while frontier:
        nxt = []
        for a, d in frontier:
            if a in seen or d > max_depth:
                continue
            seen.add(a)
            atom = mol.GetAtomWithIdx(a)
            seq.append(atom.GetAtomicNum())
            for nb in atom.GetNeighbors():
                if nb.GetIdx() not in seen:
                    nxt.append((nb.GetIdx(), d + 1))
        frontier = nxt
    return sorted(seq, reverse=True)


def _one(prod_smi):
    feat = {k: 0.0 for k in FEAT_NAMES}
    if not isinstance(prod_smi, str) or not prod_smi.strip():
        return feat
    mol = Chem.MolFromSmiles(prod_smi)
    if mol is None:
        return feat
    matches = mol.GetSubstructMatches(_BROAD)
    if not matches:
        return feat
    pairs = {(m[_CA_Q], m[_CB_Q]) for m in matches}
    if len(pairs) > 1:
        return feat  # ambiguous
    m = matches[0]
    cb, ca = m[_CB_Q], m[_CA_Q]
    cb_atom = mol.GetAtomWithIdx(cb)
    # R_ald branch = carbon neighbor of Cb that is not Ca and not the O
    rald = None
    for nb in cb_atom.GetNeighbors():
        if nb.GetIdx() != ca and nb.GetAtomicNum() == 6:
            rald = nb.GetIdx()
            break
    if rald is None:
        return feat
    feat["aldcip_match"] = 1.0
    seq_ald = _branch_priority_seq(mol, rald, cb)
    seq_alp = _branch_priority_seq(mol, ca, cb)
    # lexicographic compare of descending atomic-number sequences (CIP-like)
    cmp = 0
    for za, zb in zip(seq_ald, seq_alp):
        if za != zb:
            cmp = 1 if za > zb else -1
            break
    if cmp == 0 and len(seq_ald) != len(seq_alp):
        cmp = 1 if len(seq_ald) > len(seq_alp) else -1
    feat["aldcip_ald_outranks_alpha"] = float(cmp)
    feat["aldcip_first_Z_diff"] = float((seq_ald[0] if seq_ald else 0) - (seq_alp[0] if seq_alp else 0))
    feat["aldcip_ald_branch_arom"] = float(mol.GetAtomWithIdx(rald).GetIsAromatic())
    return feat


def compute(product_smiles_series) -> pd.DataFrame:
    """Compute the 4d CIP-flip decoupling features for a series of product SMILES."""
    rows = [_one(s) for s in product_smiles_series]
    return pd.DataFrame(rows, columns=FEAT_NAMES)
