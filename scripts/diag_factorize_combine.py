#!/usr/bin/env python
"""A3-combine — does (Ca, SA) factorization beat direct 4-class?

(label_Ca, label_SA) deterministically reconstructs label_joint:
    label_SA = int(Ca == Cb)  ->  Cb = Ca if SA==1 else 1-Ca
    label_joint = 2*Ca + Cb
We train two independent binary ExtraTrees heads (Ca, SA), combine to a
4-class prediction, and compare TSCV balanced accuracy to the direct 4-class
model on the SAME Evans folds. Tells us whether the CIP joint encoding is the
bottleneck (factorize -> gain) or whether it's just the product of two axes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import CLEAN_DIR, VALID_AUXILIARIES  # noqa: E402
from chiralaldol.data_io import load_features, load_labels, load_splits  # noqa: E402
from chiralaldol.model_trainers import train_et  # noqa: E402


def main():
    Xdf, _ = load_features()
    X = np.nan_to_num(Xdf.values.astype(np.float32))
    lab = load_labels()
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv", usecols=["auxiliary_type"])
    clean = clean[clean["auxiliary_type"].isin(VALID_AUXILIARIES)].reset_index(drop=True)
    evans = (clean["auxiliary_type"] == "evans").values

    ca = lab["label_Ca"].values
    sa = lab["label_SA"].values
    joint = lab["label_joint"].values
    valid = (~np.isnan(ca)) & (~np.isnan(sa)) & (~np.isnan(joint)) & evans

    splits = load_splits()
    direct, combo, w = [], [], []
    for name, sd in splits.items():
        if "tscv" not in name:
            continue
        tr = np.array(sd["train"], int); tr = tr[valid[tr]]
        te = np.array(sd["test"], int); te = te[valid[te]]
        if len(tr) < 10 or len(te) < 3:
            continue
        yj = joint.astype(int); yca = ca.astype(int); ysa = sa.astype(int)

        # direct 4-class
        md = train_et(X[tr], yj[tr])
        acc_d = balanced_accuracy_score(yj[te], md.predict(X[te]))

        # factorized: Ca head + SA head -> reconstruct joint
        mca = train_et(X[tr], yca[tr])
        msa = train_et(X[tr], ysa[tr])
        pca = mca.predict(X[te]); psa = msa.predict(X[te])
        pcb = np.where(psa == 1, pca, 1 - pca)
        pj = 2 * pca + pcb
        acc_c = balanced_accuracy_score(yj[te], pj)

        direct.append(acc_d); combo.append(acc_c); w.append(len(te))
        print(f"  {name}: direct={acc_d:.4f}  factorized(Ca,SA)={acc_c:.4f}  n={len(te)}")

    w = np.array(w)
    print(f"\n  TSCV_w direct 4-class      : {np.average(direct, weights=w):.4f}")
    print(f"  TSCV_w factorized (Ca,SA)  : {np.average(combo, weights=w):.4f}")
    print("\n  If factorized >> direct: CIP joint encoding amplifies difficulty.")
    print("  If factorized ~= direct: 4-class is just the product of two axes;")
    print("  switching label encoding will NOT unlock 0.90.")


if __name__ == "__main__":
    main()
