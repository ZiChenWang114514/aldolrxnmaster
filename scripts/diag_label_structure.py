#!/usr/bin/env python
"""A0-diag — Why did the mechanism->CIP inversion gate fail?

Three candidate explanations for the ~0.50 purity:
  (1) CIP pollution needs aldehyde priority too -> fixable by richer rule.
  (2) The labels (label_Ca/Cb or label_syn_anti_3d) are noise -> not fixable.
  (3) The aux_chir key is too coarse.

This read-only diagnostic isolates them by measuring conditional purities of
the two CIP centers separately, and the internal consistency of the syn/anti
label. Evans-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import VALID_AUXILIARIES  # noqa: E402

CLEAN = ROOT / "data" / "clean_v5" / "substrate_aldol_clean.csv"


def aux_chir(smi):
    if not isinstance(smi, str) or not smi.strip():
        return "na"
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "na"
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    try:
        c = Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)
    except Exception:
        c = Chem.FindMolChiralCenters(m, includeUnassigned=False)
    cips = sorted(x for _, x in c if x in ("R", "S"))
    return "|".join(cips) if cips else "none"


def ald_aromatic(smi):
    """Is the aldehyde carbonyl attached to an aromatic carbon?"""
    if not isinstance(smi, str) or not smi.strip():
        return -1
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return -1
    # find aldehyde C(=O)[H] / acyl carbon, check neighbor aromaticity
    patt = Chem.MolFromSmarts("[CX3H1]=O")
    mt = m.GetSubstructMatches(patt)
    if not mt:
        return -1
    c = m.GetAtomWithIdx(mt[0][0])
    for nb in c.GetNeighbors():
        if nb.GetSymbol() == "C" and nb.GetIsAromatic():
            return 1
    return 0


def wpurity(df, cols, label):
    sub = df[df[label].notna()]
    rows = []
    for _, g in sub.groupby(cols, dropna=False):
        vc = g[label].value_counts()
        rows.append((len(g), vc.iloc[0] / len(g)))
    n = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
    return float(np.average(p, weights=n)), len(sub)


def main():
    df = pd.read_csv(CLEAN)
    df = df[df["auxiliary_type"] == "evans"].copy()
    df["aux_chir"] = df["canonical_ketone_smiles"].map(aux_chir)
    df["ald_arom"] = df["canonical_aldehyde_smiles"].map(ald_aromatic)
    print(f"Evans rows: {len(df)}")
    print(f"aldehyde aromatic distribution: {df['ald_arom'].value_counts(dropna=False).to_dict()}")

    print("\n--- Q1: is the alpha-center (Ca) auxiliary-controlled? ---")
    for cols in [["aux_chir"], ["aux_chir", "ald_arom"]]:
        wp, n = wpurity(df, cols, "label_Ca")
        print(f"  purity(label_Ca | {cols}) = {wp:.4f}  (n={n})")

    print("\n--- Q2: the carbinol center (Cb) ---")
    for cols in [["aux_chir"], ["aux_chir", "label_syn_anti_3d"],
                 ["aux_chir", "ald_arom"], ["aux_chir", "label_syn_anti_3d", "ald_arom"]]:
        wp, n = wpurity(df, cols, "label_Cb")
        print(f"  purity(label_Cb | {cols}) = {wp:.4f}  (n={n})")

    print("\n--- Q3: full label_joint with aldehyde priority added ---")
    for cols in [["aux_chir", "label_syn_anti_3d"],
                 ["aux_chir", "ald_arom"],
                 ["aux_chir", "label_syn_anti_3d", "ald_arom"]]:
        wp, n = wpurity(df, cols, "label_joint")
        print(f"  purity(label_joint | {cols}) = {wp:.4f}  (n={n})")

    print("\n--- Q4: internal consistency of syn/anti labels ---")
    v = df[df["label_syn_anti_3d"].notna()].copy()
    # label_SA = int(Ca==Cb) CIP heuristic ; label_syn_anti_3d = 3D dihedral
    agree = (v["label_SA"] == v["label_syn_anti_3d"]).mean()
    print(f"  agreement(label_SA[CIP], label_syn_anti_3d[3D]) = {agree:.4f}  (n={len(v)})")
    # does syn/anti_3d correlate with the Ca==Cb pairing at all?
    v_pair = (v["label_Ca"] == v["label_Cb"]).astype(int)
    from sklearn.metrics import matthews_corrcoef
    print(f"  MCC(syn_anti_3d, Ca==Cb)  = {matthews_corrcoef(v['label_syn_anti_3d'], v_pair):.4f}")
    print(f"  MCC(label_SA,    Ca==Cb)  = {matthews_corrcoef(v['label_SA'], v_pair):.4f}  (sanity, should be ~1)")

    print("\n--- Q5: how concentrated is label_joint per aux_chir (raw)? ---")
    for k, g in df.groupby("aux_chir"):
        if len(g) < 20:
            continue
        vc = g["label_joint"].value_counts(normalize=True)
        dist = {int(i): round(p, 2) for i, p in vc.items()}
        print(f"  aux_chir={k:6s} n={len(g):4d}  label_joint dist={dist}")


if __name__ == "__main__":
    main()
