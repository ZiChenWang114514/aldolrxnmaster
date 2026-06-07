#!/usr/bin/env python
"""A1 — Honest per-axis evaluation: full-test vs gold-test vs non-gold-test.

The reported 4-class numbers (0.818 Evans / 0.74 full) are deflated by ~6%
test-label noise (LESSONS L15). On a trusted "gold" test subset (high mapping
confidence ∧ ¬TypeA ∧ broad-SMARTS-CIP-agree) the model scores much higher.

This script trains ONE model (ExtraTrees) on the full data and evaluates the
SAME model on three test subsets per fold — full / gold / non-gold — for each
stereo-axis (Ca, Cb, SA, joint). Comparing gold vs non-gold separates "cleaner
labels" from a possible "easier molecules" selection effect (caveat reported).

Read-only; writes results/tables/honest_axis_eval.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import CLEAN_DIR, FEAT_DIR, RESULTS_DIR, VALID_AUXILIARIES  # noqa: E402
from chiralaldol.data_io import load_features, load_labels, load_splits  # noqa: E402
from chiralaldol.model_trainers import train_et  # noqa: E402

AXES = [("Ca", "label_Ca"), ("Cb", "label_Cb"), ("SA", "label_SA"), ("joint", "label_joint")]


def evans_mask():
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv", usecols=["auxiliary_type"])
    clean = clean[clean["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    return (clean["auxiliary_type"] == "evans").values


def wmean(pairs):
    if not pairs:
        return float("nan"), 0
    a = np.array([p[0] for p in pairs]); w = np.array([p[1] for p in pairs])
    return float(np.average(a, weights=w)), int(w.sum())


def eval_axis(X, y, base, gold, splits, kind):
    """Train on full base-train; eval same model on full/gold/nongold test per fold."""
    res = {"full": [], "gold": [], "nongold": []}
    cm_gold = None
    for name, sd in splits.items():
        if kind not in name:
            continue
        tr = np.array(sd["train"], int); te = np.array(sd["test"], int)
        tr = tr[base[tr]]
        te_full = te[base[te]]
        if len(tr) < 20 or len(te_full) < 5:
            continue
        m = train_et(X[tr], y[tr])
        pred_full = m.predict(X[te_full])
        res["full"].append((balanced_accuracy_score(y[te_full], pred_full), len(te_full)))
        te_g = te[base[te] & gold[te]]
        te_n = te[base[te] & ~gold[te]]
        if len(te_g) >= 5:
            pg = m.predict(X[te_g])
            res["gold"].append((balanced_accuracy_score(y[te_g], pg), len(te_g)))
            if y[te_g].max() <= 3:
                cm = confusion_matrix(y[te_g], pg, labels=sorted(np.unique(y[base])))
                cm_gold = cm if cm_gold is None else cm_gold + cm
        if len(te_n) >= 5:
            pn = m.predict(X[te_n])
            res["nongold"].append((balanced_accuracy_score(y[te_n], pn), len(te_n)))
    return res, cm_gold


def main():
    Xdf, _ = load_features()
    X = np.nan_to_num(Xdf.values.astype(np.float32))
    lab = load_labels()
    meta = pd.read_csv(FEAT_DIR / "denoise_meta.csv")
    gold = meta["gold"].values
    ev = evans_mask()
    splits = load_splits()

    rows = []
    for scope, smask in [("evans", ev), ("all", np.ones(len(X), bool))]:
        for axis, col in AXES:
            y = lab[col].values
            valid = ~np.isnan(y)
            base = valid & smask
            y = np.where(valid, y, -1).astype(int)
            for kind in ["tscv", "grouped"]:
                res, cm = eval_axis(X, y, base, gold, splits, kind)
                (f, nf), (g, ng), (n, nn) = wmean(res["full"]), wmean(res["gold"]), wmean(res["nongold"])
                rows.append({"scope": scope, "axis": axis, "split": kind,
                             "full": round(f, 4), "n_full": nf,
                             "gold": round(g, 4), "n_gold": ng,
                             "nongold": round(n, 4), "n_nongold": nn})
                if axis == "joint" and kind == "tscv" and scope == "evans" and cm is not None:
                    print(f"\n  [Evans joint gold-test confusion matrix]\n{cm}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("  HONEST PER-AXIS EVALUATION (ExtraTrees; same model, 3 test subsets)")
    print("=" * 84)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))

    out = RESULTS_DIR / "tables" / "honest_axis_eval.csv"
    df.to_csv(out, index=False)
    print(f"\n  Wrote {out}")
    print("\n  Reading: gold >> full quantifies test-label-noise deflation;")
    print("  gold vs nongold gap that exceeds the label-noise rate = molecule-difficulty")
    print("  selection effect (honest caveat — gold rows may be intrinsically easier).")


if __name__ == "__main__":
    main()
