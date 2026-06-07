#!/usr/bin/env python
"""B0 — carbinol-axis gate: do cheap feature/modeling levers move label_Cb / label_joint?

Levers (hstack onto the frozen 156d champion features; never mutate v5_features.csv):
  B0b-face : + real-conformer Si/Re face steric maps (24d, in spms/face_map_features.csv,
             never used in the 156d champion set)
  B0b-spms : + SPMS spherical-projection autoencoder codes (16d, in v5_features_spms.csv)
  B0b-both : + face(24) + spms(16)
  B0c-2stage: + out-of-fold Cα prediction probabilities (2d) — two-stage Cα→Cb

Evaluated on label_Cb and label_joint, on full-test and gold-test (denoise_meta gold),
Evans + all-aux, with the same TSCV fold loop as run_honest_eval. ExtraTrees.

Gate: keep a lever if gold-test Δ(Cb or joint) >= +0.01 vs the 156d baseline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import CLEAN_DIR, FEAT_DIR, VALID_AUXILIARIES  # noqa: E402
from chiralaldol.data_io import load_features, load_labels, load_splits  # noqa: E402
from chiralaldol.model_trainers import train_et  # noqa: E402


def evans_mask():
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv", usecols=["auxiliary_type"])
    clean = clean[clean["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    return (clean["auxiliary_type"] == "evans").values


def oof_ca_probs(X, lab):
    """Out-of-fold Cα prediction probabilities (2d), leakage-free via 5-fold CV-predict."""
    yca = lab["label_Ca"].values
    valid = ~np.isnan(yca)
    probs = np.zeros((len(X), 2), dtype=np.float32)
    clf = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                               class_weight="balanced")
    p = cross_val_predict(clf, X[valid], yca[valid].astype(int), cv=5,
                          method="predict_proba", n_jobs=-1)
    probs[valid] = p
    return probs


def wmean(pairs):
    if not pairs:
        return float("nan")
    a = np.array([p[0] for p in pairs]); w = np.array([p[1] for p in pairs])
    return float(np.average(a, weights=w))


def eval_variant(Xv, y, base, gold, splits):
    """Return (full_w, gold_w) TSCV balanced acc."""
    full, goldr = [], []
    for name, sd in splits.items():
        if "tscv" not in name:
            continue
        tr = np.array(sd["train"], int); te = np.array(sd["test"], int)
        tr = tr[base[tr]]; te_f = te[base[te]]
        if len(tr) < 20 or len(te_f) < 5:
            continue
        m = train_et(Xv[tr], y[tr])
        full.append((balanced_accuracy_score(y[te_f], m.predict(Xv[te_f])), len(te_f)))
        te_g = te[base[te] & gold[te]]
        if len(te_g) >= 5:
            goldr.append((balanced_accuracy_score(y[te_g], m.predict(Xv[te_g])), len(te_g)))
    return wmean(full), wmean(goldr)


def main():
    Xdf, _ = load_features()
    X = np.nan_to_num(Xdf.values.astype(np.float32))
    lab = load_labels()
    gold = pd.read_csv(FEAT_DIR / "denoise_meta.csv")["gold"].values
    ev = evans_mask()
    splits = load_splits()

    face = np.nan_to_num(pd.read_csv(FEAT_DIR / "spms" / "face_map_features.csv").values.astype(np.float32))
    spms_cols = [c for c in pd.read_csv(FEAT_DIR / "v5_features_spms.csv", nrows=1).columns if c.startswith("spms_")]
    spms = np.nan_to_num(pd.read_csv(FEAT_DIR / "v5_features_spms.csv", usecols=spms_cols).values.astype(np.float32))
    print(f"face={face.shape} spms={spms.shape}; computing OOF Cα probs...")
    ca_oof = oof_ca_probs(X, lab)

    variants = {
        "baseline(156d)": X,
        "+face(24)": np.hstack([X, face]),
        "+spms(16)": np.hstack([X, spms]),
        "+face+spms": np.hstack([X, face, spms]),
        "+CaOOF(2)": np.hstack([X, ca_oof]),
        "+face+CaOOF": np.hstack([X, face, ca_oof]),
    }

    for scope, smask in [("evans", ev), ("all", np.ones(len(X), bool))]:
        for target in ["label_Cb", "label_joint"]:
            y = lab[target].values
            valid = ~np.isnan(y)
            base = valid & smask
            y = np.where(valid, y, -1).astype(int)
            print(f"\n{'='*70}\n  [{scope}] {target}  (TSCV, full / gold)\n{'='*70}")
            base_gold = None
            for vname, Xv in variants.items():
                f, g = eval_variant(Xv, y, base, gold, splits)
                if vname.startswith("baseline"):
                    base_gold = g; base_full = f
                    print(f"    {vname:16s}  full={f:.4f}  gold={g:.4f}")
                else:
                    print(f"    {vname:16s}  full={f:.4f} (Δ{f-base_full:+.4f})  "
                          f"gold={g:.4f} (Δ{g-base_gold:+.4f})")


if __name__ == "__main__":
    main()
