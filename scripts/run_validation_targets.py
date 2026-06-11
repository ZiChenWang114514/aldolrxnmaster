#!/usr/bin/env python
"""A1-A3 — decisive validation: how learnable is each label axis?

For each target label, run TSCV (n_test-weighted) + grouped CV with a strong
fast model (ExtraTrees, the V5 champion) and a majority baseline. The lift over
majority tells us which axes carry real signal.

Hypotheses under test (from the A0 diagnostic):
  - label_Ca (alpha center)   : auxiliary-controlled -> should be HIGH
  - label_Cb (carbinol)        : the real bottleneck  -> should be LOWER
  - label_syn_anti_3d (3D geom): suspected NOISE       -> should be ~majority
  - label_SA  (Ca==Cb pair)    : CIP main-product pair
  - label_joint (4-class CIP)  : current target, ~0.74 full / 0.82 Evans

Read-only w.r.t. project data; prints a comparison table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import CLEAN_DIR, VALID_AUXILIARIES  # noqa: E402
from chiralaldol.data_io import load_splits, prepare_Xy  # noqa: E402
from chiralaldol.model_trainers import MajorityClassifier, train_et  # noqa: E402
from chiralaldol.utils import wmean  # noqa: E402

TARGETS = ["label_joint", "label_Ca", "label_Cb", "label_SA", "label_syn_anti_3d"]


def _evans_mask():
    """Boolean mask (aligned to the 2427 feature rows) selecting Evans reactions."""
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv", usecols=["auxiliary_type"])
    clean = clean[clean["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    return (clean["auxiliary_type"] == "evans").values


def eval_target(target, evans_only=False):
    X, y_raw, vmask, _ = prepare_Xy(target_label=target)
    if evans_only:
        vmask = vmask & _evans_mask()
    y = np.where(vmask, y_raw, -1).astype(int)
    splits = load_splits()

    rows = {"tscv": [], "grouped": [], "maj_tscv": []}
    for name, sd in splits.items():
        kind = "tscv" if "tscv" in name else ("grouped" if "grouped" in name else None)
        if kind is None:
            continue
        tr = np.array(sd["train"], dtype=int); tr = tr[vmask[tr]]
        te = np.array(sd["test"], dtype=int); te = te[vmask[te]]
        if len(tr) < 10 or len(te) < 3:
            continue
        m = train_et(X[tr], y[tr])
        acc = balanced_accuracy_score(y[te], m.predict(X[te]))
        maj = MajorityClassifier().fit(X[tr], y[tr])
        macc = balanced_accuracy_score(y[te], maj.predict(X[te]))
        rows[kind].append((acc, len(te)))
        if kind == "tscv":
            rows["maj_tscv"].append((macc, len(te)))

    return {
        "target": target,
        "n_classes": len(np.unique(y[vmask])),
        "n": int(vmask.sum()),
        "TSCV_w": wmean(rows["tscv"]),
        "Grouped_w": wmean(rows["grouped"]),
        "Majority": wmean(rows["maj_tscv"]),
    }


def run(evans_only):
    tag = "EVANS-ONLY" if evans_only else "ALL-AUX"
    print(f"\n{'='*78}\n  VALIDATION [{tag}]  (ExtraTrees, n_test-weighted TSCV)\n{'='*78}")
    res = [eval_target(t, evans_only) for t in TARGETS]
    df = pd.DataFrame(res)
    df["lift"] = df["TSCV_w"] - df["Majority"]
    df = df[["target", "n_classes", "n", "TSCV_w", "Grouped_w", "Majority", "lift"]]
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))
    return df


def main():
    run(evans_only=False)
    run(evans_only=True)
    print("\nReading: high TSCV_w + high lift = learnable axis; "
          "TSCV_w≈Majority (lift≈0) = noise/unlearnable.")


if __name__ == "__main__":
    main()
