#!/usr/bin/env python
"""D0/D1 — Label denoising via broad-SMARTS cross-validation + contradiction arbitration.

D0 (analysis, default): for every row, relocate the two aldol stereocenters with
the broad SMARTS used in step08, recompute CIP on the product, and cross-check
against the stored label_Ca/label_Cb. Also classify same-substrate label
contradictions into Type A (same conditions -> noise) vs Type B (different
conditions -> legitimate stereodivergence). Emits a per-row meta table (aligned
to the 2427 feature rows) for gold-subset evaluation. Read-only w.r.t. labels.

D1 (--emit): produce data/features_v5/labels_v6.csv with corrected labels +
a `quality` column for confidence-weighted training. Only run after the D0 gate.

Reuses: step08_label_validate broad SMARTS (`[CX4:1]([OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])`,
:1=Cb carbinol, :2=Ca alpha).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import CLEAN_DIR, FEAT_DIR, VALID_AUXILIARIES  # noqa: E402

CLEAN = CLEAN_DIR / "substrate_aldol_clean.csv"
OUT_DIR = ROOT / "results" / "denoise"
PROD_COL = "canonical_main_product_smiles"

# Broad SMARTS: beta-hydroxy (or O-substituted) carbonyl with two ring/chain
# stereocenters. :1 = Cb (carbinol, O-bearing), :2 = Ca (alpha to C=O).
_BROAD = Chem.MolFromSmarts("[CX4:1]([OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])")
_CB_Q = _CA_Q = None
for _qi in range(_BROAD.GetNumAtoms()):
    _mn = _BROAD.GetAtomWithIdx(_qi).GetAtomMapNum()
    if _mn == 1:
        _CB_Q = _qi
    elif _mn == 2:
        _CA_Q = _qi


def smarts_cip(prod_smi):
    """Relocate Ca/Cb via broad SMARTS and read CIP. Returns (ca, cb, status).

    status: 'ok' (unique match), 'ambiguous' (>1 match), 'no_match', 'bad_mol'.
    ca/cb are 0(R)/1(S) or None.
    """
    if not isinstance(prod_smi, str) or not prod_smi.strip():
        return None, None, "bad_mol"
    mol = Chem.MolFromSmiles(prod_smi)
    if mol is None:
        return None, None, "bad_mol"
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    matches = mol.GetSubstructMatches(_BROAD)
    if not matches:
        return None, None, "no_match"
    if len(matches) > 1:
        # distinct (ca,cb) atom pairs? if all map to same pair, treat as unique
        pairs = {(m[_CA_Q], m[_CB_Q]) for m in matches}
        if len(pairs) > 1:
            return None, None, "ambiguous"
    m = matches[0]
    ca_cip = mol.GetAtomWithIdx(m[_CA_Q]).GetPropsAsDict().get("_CIPCode")
    cb_cip = mol.GetAtomWithIdx(m[_CB_Q]).GetPropsAsDict().get("_CIPCode")
    ca = {"R": 0, "S": 1}.get(ca_cip)
    cb = {"R": 0, "S": 1}.get(cb_cip)
    return ca, cb, "ok"


def cross_validate(df):
    """Add smarts_Ca/Cb + cip_status (agree/disagree/ambiguous/no_match/...)."""
    sca, scb, sst, status = [], [], [], []
    for prod in df[PROD_COL]:
        ca, cb, st = smarts_cip(prod)
        sca.append(ca); scb.append(cb); sst.append(st)
    df = df.copy()
    df["smarts_Ca"] = sca
    df["smarts_Cb"] = scb
    df["smarts_status"] = sst

    out = []
    for _, r in df.iterrows():
        if r["smarts_status"] != "ok" or r["smarts_Ca"] is None or r["smarts_Cb"] is None:
            out.append("no_smarts")
        elif pd.isna(r["label_Ca"]) or pd.isna(r["label_Cb"]):
            out.append("no_stored")
        elif int(r["label_Ca"]) == r["smarts_Ca"] and int(r["label_Cb"]) == r["smarts_Cb"]:
            out.append("agree")
        else:
            out.append("disagree")
    df["cip_status"] = out
    return df


def classify_contradictions(df):
    """Type A (same substrate+conditions, differing label) vs Type B (diff conditions)."""
    df = df.copy()
    sub_key = df["canonical_ketone_smiles"].astype(str) + "||" + df["canonical_aldehyde_smiles"].astype(str)
    cond_key = (sub_key + "||" + df["base_type"].astype(str) + "||" + df["activator_type"].astype(str))
    df["_sub_key"] = sub_key
    df["_cond_key"] = cond_key

    # contradiction at substrate level
    sub_n = df.groupby("_sub_key")["label_joint"].transform("nunique")
    cond_n = df.groupby("_cond_key")["label_joint"].transform("nunique")
    df["is_typeA"] = cond_n > 1            # same substrate+conditions, differing label
    df["is_contradiction"] = sub_n > 1     # same substrate, differing label
    df["is_typeB"] = df["is_contradiction"] & ~df["is_typeA"]
    return df


def report(df):
    print(f"\n{'='*72}\n  LABEL DENOISING — CROSS-VALIDATION & CONTRADICTION ANALYSIS\n{'='*72}")
    print(f"  rows: {len(df)}")

    print("\n  [broad-SMARTS cross-validation of CIP]")
    cs = df["cip_status"].value_counts()
    for k, v in cs.items():
        print(f"    {k:12s}: {v:5d}  ({v/len(df)*100:.1f}%)")
    dis = df[df["cip_status"] == "disagree"]
    print(f"  >>> cip_disagree = {len(dis)} rows (suspected silent index/CIP errors)")
    if len(dis):
        print("      by mapping_confidence:")
        for k, v in dis["mapping_confidence"].value_counts().items():
            base = (df["mapping_confidence"] == k).sum()
            print(f"        {k:8s}: {v:4d} / {base} ({v/base*100:.1f}% of that conf level)")
        # which center flips more?
        ca_flip = (dis["label_Ca"].astype("Int64") != dis["smarts_Ca"].astype("Int64")).sum()
        cb_flip = (dis["label_Cb"].astype("Int64") != dis["smarts_Cb"].astype("Int64")).sum()
        print(f"      of disagreements: Ca flips={ca_flip}, Cb flips={cb_flip}")

    print("\n  [same-substrate label contradictions]")
    print(f"    Type A (same substrate+conditions -> noise) : {int(df['is_typeA'].sum())} rows")
    print(f"    Type B (diff conditions -> legit divergence): {int(df['is_typeB'].sum())} rows")
    print(f"    total contradictory                         : {int(df['is_contradiction'].sum())} rows")

    gold = (df["mapping_confidence"] == "high") & (~df["is_typeA"]) & (df["cip_status"] == "agree")
    df["gold"] = gold
    print(f"\n  [gold (trusted) subset]")
    print(f"    high-mapconf & not-TypeA & cip_agree = {int(gold.sum())} rows "
          f"({gold.sum()/len(df)*100:.1f}%)")
    ev_gold = gold & (df["auxiliary_type"] == "evans")
    print(f"    of which Evans: {int(ev_gold.sum())}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="D1: write labels_v6.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN)
    df = cross_validate(df)
    df = classify_contradictions(df)
    df = report(df)

    # per-row audit (full 2434)
    audit_cols = ["Reaction ID", "auxiliary_type", "mapping_confidence",
                  "label_Ca", "label_Cb", "label_joint",
                  "smarts_Ca", "smarts_Cb", "smarts_status", "cip_status",
                  "is_typeA", "is_typeB", "gold"]
    audit_cols = [c for c in audit_cols if c in df.columns]
    df[audit_cols].to_csv(OUT_DIR / "crossval_analysis.csv", index=False)

    # feature-aligned meta (2427) for gold-subset evaluation in run_validation_targets
    fdf = df[df["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    meta = fdf[["cip_status", "is_typeA", "is_typeB", "gold", "smarts_Ca", "smarts_Cb",
                "mapping_confidence", "synanti_confidence"]].copy()
    meta.to_csv(FEAT_DIR / "denoise_meta.csv", index=False)
    print(f"\n  Wrote {OUT_DIR/'crossval_analysis.csv'} (2434 rows)")
    print(f"  Wrote {FEAT_DIR/'denoise_meta.csv'} (feature-aligned, {len(meta)} rows)")

    if args.emit:
        print("\n  [--emit] D1 label construction not yet implemented; run after D0 gate.")


if __name__ == "__main__":
    main()
