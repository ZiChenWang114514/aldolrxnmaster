#!/usr/bin/env python
"""D0 gate — is the alpha-axis 12% error recoverable noise or genuine chemistry?

cip_disagree=0 (broad-SMARTS confirms stored CIP), so the only actionable noise
is the 214 Type A contradictions (same substrate+conditions -> differing label).

Decisive, bias-free measurement on a FIXED gold (trusted) test set:
  For each TSCV/grouped fold, test = fold-test ∩ gold. Compare label_Ca / label_joint
  balanced accuracy when training on:
    (a) all valid rows (noisy)        vs
    (b) all valid rows minus Type A   (denoised)
  on the SAME gold test rows. Also report the gold-only ceiling (train gold->test gold).

PASS if gold label_Ca >= 0.93 AND (b) beats (a) by >= +0.02. Else the 12% is
largely irreducible -> stop, report per-axis honestly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import CLEAN_DIR, FEAT_DIR, VALID_AUXILIARIES  # noqa: E402
from chiralaldol.data_io import load_features, load_labels, load_splits  # noqa: E402
from chiralaldol.model_trainers import train_et  # noqa: E402


def evans_mask():
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv", usecols=["auxiliary_type"])
    clean = clean[clean["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    return (clean["auxiliary_type"] == "evans").values


def main():
    Xdf, _ = load_features()
    X = np.nan_to_num(Xdf.values.astype(np.float32))
    lab = load_labels()
    meta = pd.read_csv(FEAT_DIR / "denoise_meta.csv")
    assert len(meta) == len(lab) == X.shape[0], (len(meta), len(lab), X.shape[0])
    ev = evans_mask()
    typeA = meta["is_typeA"].values
    gold = meta["gold"].values
    splits = load_splits()

    # First: within Type A groups, does Ca flip or only Cb?
    full = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    fa = full[full["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    ta = fa[typeA]
    print(f"Type A rows (feature-aligned): {len(ta)}  Evans: {int((typeA & ev).sum())}")
    if len(ta):
        sub = ta["canonical_ketone_smiles"].astype(str) + "||" + ta["canonical_aldehyde_smiles"].astype(str) \
            + "||" + ta["base_type"].astype(str) + "||" + ta["activator_type"].astype(str)
        ca_var = ta.groupby(sub)["label_Ca"].transform("nunique")
        cb_var = ta.groupby(sub)["label_Cb"].transform("nunique")
        print(f"  Type A groups where Ca varies: {int((ca_var>1).sum())} rows; "
              f"Cb varies: {int((cb_var>1).sum())} rows")

    for target in ["label_Ca", "label_joint"]:
        y = lab[target].values
        valid = ~np.isnan(y)
        y = np.where(valid, y, -1).astype(int)
        print(f"\n{'='*64}\n  GATE [{target}]  (Evans, ExtraTrees, test = gold subset)\n{'='*64}")
        rows = {"noisy": [], "clean": [], "goldonly": [], "w": []}
        for name, sd in splits.items():
            if "tscv" not in name:
                continue
            base = valid & ev
            tr = np.array(sd["train"], int); te = np.array(sd["test"], int)
            te_gold = te[base[te] & gold[te]]
            if len(te_gold) < 5:
                continue
            tr_noisy = tr[base[tr]]
            tr_clean = tr[base[tr] & ~typeA[tr]]
            tr_gold = tr[base[tr] & gold[tr]]
            if len(tr_clean) < 20:
                continue
            a_noisy = balanced_accuracy_score(y[te_gold], train_et(X[tr_noisy], y[tr_noisy]).predict(X[te_gold]))
            a_clean = balanced_accuracy_score(y[te_gold], train_et(X[tr_clean], y[tr_clean]).predict(X[te_gold]))
            a_gold = balanced_accuracy_score(y[te_gold], train_et(X[tr_gold], y[tr_gold]).predict(X[te_gold]))
            rows["noisy"].append(a_noisy); rows["clean"].append(a_clean)
            rows["goldonly"].append(a_gold); rows["w"].append(len(te_gold))
            print(f"  {name}: noisy_tr={a_noisy:.4f}  clean_tr={a_clean:.4f}  gold_tr={a_gold:.4f}  n_te_gold={len(te_gold)}")
        w = np.array(rows["w"])
        if len(w):
            wn = np.average(rows["noisy"], weights=w)
            wc = np.average(rows["clean"], weights=w)
            wg = np.average(rows["goldonly"], weights=w)
            print(f"\n  TSCV_w  noisy-train  : {wn:.4f}")
            print(f"  TSCV_w  clean-train  : {wc:.4f}   (Δ vs noisy = {wc-wn:+.4f})")
            print(f"  TSCV_w  gold-train   : {wg:.4f}   (gold-only ceiling)")
            if target == "label_Ca":
                verdict = "PASS" if (wg >= 0.93 and wc - wn >= 0.02) else "FAIL"
                print(f"\n  >>> GATE: {verdict}  (gold ceiling={wg:.4f}, denoise Δ={wc-wn:+.4f})")


if __name__ == "__main__":
    main()
