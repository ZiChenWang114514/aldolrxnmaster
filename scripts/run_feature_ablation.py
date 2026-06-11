#!/usr/bin/env python
"""Feature effectiveness analysis + label ablation (null-importance) — find noise features.

Motivation: SHAP / impurity importance on the noisy full-test set conflates
"feature is useful" with "feature fits label noise" (cf. LESSONS L14-L16). This
script identifies NOISE features per stereo-axis with two complementary tests,
then verifies a conservative noise set by non-destructive pruning on gold-test.

Four blocks, all per-axis (label_Ca = alpha, label_Cb = carbinol, label_joint = 4-class):

  M1  Null-importance (LABEL ABLATION).  Train ExtraTrees on real labels -> real
      impurity importance. Shuffle the label N times, retrain -> a NULL importance
      distribution per feature. A feature whose real importance is NOT above its
      null band (p_value = P(null >= real) > 0.05) is only fitting label noise.

  M2  Gold-test permutation importance.  In the TSCV fold loop, permute each
      feature column at test time and measure the balanced-accuracy drop on the
      GOLD (trusted-label) test subset. gold_drop <= 0 => feature does not help
      honest generalization. full_drop is reported alongside as a caveat: a
      feature with full_drop >> gold_drop ~ 0 is a noise-fitting fingerprint.

  M4  Leave-one-group-out (LOGO).  Drop each feature_registry group (+ an
      "unclassified" residual group) and measure the gold-test delta per axis.

  Prune verify.  NOISE = (M1 p>0.05 on ALL 3 axes) AND (M2 gold_drop<=0 on ALL 3
      axes). Drop that set IN MEMORY (never touch v5_features.csv) and re-measure
      gold-test per axis vs the 156d baseline. Pruning "passes" if gold joint
      delta >= -0.005 (flat or better) -> confirms the set is noise and the model
      can be slimmed.

Read-only w.r.t. project data. Outputs go to results/tables/.

Usage:
    conda run -n aldol-rxn python scripts/run_feature_ablation.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.config import FEAT_DIR, RESULTS_DIR  # noqa: E402
from chiralaldol.data_io import load_features, load_labels, load_splits  # noqa: E402
from chiralaldol.feature_registry import FEATURE_SUBSETS, _match_prefixes  # noqa: E402
from chiralaldol.model_trainers import train_et  # noqa: E402
from chiralaldol.utils import wmean  # noqa: E402

AXES = ["label_Ca", "label_Cb", "label_joint"]
AXIS_SHORT = {"label_Ca": "alpha(Ca)", "label_Cb": "carbinol(Cb)", "label_joint": "4class(joint)"}
N_NULL = 30          # null-importance shuffles
N_PERM_REPEAT = 5    # permutation repeats per fold
P_THRESH = 0.05      # M1 noise threshold
OUT_DIR = RESULTS_DIR / "tables"


# --------------------------------------------------------------------------- #
#  shared helpers
# --------------------------------------------------------------------------- #
def _fit_et(X, y, n_estimators=200, random_state=42):
    """Lightweight ExtraTrees for null-importance (fewer trees than champion)."""
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=None,
                             random_state=random_state, n_jobs=-1,
                             class_weight="balanced")
    m.fit(X, y)
    return m


def _tscv_folds(splits, vmask, scope):
    """Yield (tr, te) index arrays restricted to valid & scope rows, TSCV only."""
    for name, sd in splits.items():
        if "tscv" not in name:
            continue
        tr = np.array(sd["train"], int); te = np.array(sd["test"], int)
        keep = vmask & scope
        tr = tr[keep[tr]]; te = te[keep[te]]
        if len(tr) < 20 or len(te) < 5:
            continue
        yield tr, te


# --------------------------------------------------------------------------- #
#  M1 — null importance (label ablation)
# --------------------------------------------------------------------------- #
def null_importance(X, y_masked, vmask, feat_names):
    """Return DataFrame (feature, axis, real_imp, null_mean, null_p75, p_value, m1_noise)."""
    rows = []
    rng = np.random.RandomState(0)
    for axis in AXES:
        axis_y = y_masked[axis]
        v = vmask & (axis_y >= 0)
        Xv, yv = X[v], axis_y[v]
        real = _fit_et(Xv, yv, random_state=42).feature_importances_
        null = np.empty((N_NULL, X.shape[1]), dtype=np.float64)
        for i in range(N_NULL):
            ys = yv.copy(); rng.shuffle(ys)
            null[i] = _fit_et(Xv, ys, random_state=i + 1).feature_importances_
        null_mean = null.mean(axis=0)
        null_p75 = np.percentile(null, 75, axis=0)
        p_value = (null >= real[None, :]).mean(axis=0)
        for j, fname in enumerate(feat_names):
            rows.append({
                "feature": fname, "axis": axis,
                "real_imp": real[j], "null_mean": null_mean[j],
                "null_p75": null_p75[j], "p_value": p_value[j],
                "m1_noise": bool(p_value[j] > P_THRESH),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  M2 — gold-test permutation importance
# --------------------------------------------------------------------------- #
def perm_importance(X, y_masked, vmask, gold, feat_names, splits, scope):
    """Per-axis permutation drop on gold-test (and full-test caveat)."""
    rows = []
    nf = X.shape[1]
    for axis, y in y_masked.items():
        full_drop = np.zeros(nf); gold_drop = np.zeros(nf)
        full_w = 0.0; gold_w = 0.0
        for tr, te in _tscv_folds(splits, vmask, scope):
            m = train_et(X[tr], y[tr])
            g_local = gold[te]
            Xte = X[te]; yte = y[te]
            base_pred = m.predict(Xte)
            base_full = balanced_accuracy_score(yte, base_pred)
            n_f = len(te)
            has_gold = g_local.sum() >= 8 and len(np.unique(yte[g_local])) >= 2
            if has_gold:
                base_gold = balanced_accuracy_score(yte[g_local], base_pred[g_local])
                n_g = int(g_local.sum())
            for j in range(nf):
                acc_f = []; acc_g = []
                col = Xte[:, j].copy()
                for r in range(N_PERM_REPEAT):
                    rs = np.random.RandomState(1000 * r + j)
                    Xte[:, j] = rs.permutation(col)
                    pred = m.predict(Xte)
                    acc_f.append(balanced_accuracy_score(yte, pred))
                    if has_gold:
                        acc_g.append(balanced_accuracy_score(yte[g_local], pred[g_local]))
                Xte[:, j] = col
                full_drop[j] += (base_full - np.mean(acc_f)) * n_f
                if has_gold:
                    gold_drop[j] += (base_gold - np.mean(acc_g)) * n_g
            full_w += n_f
            if has_gold:
                gold_w += n_g
        full_drop /= max(full_w, 1); gold_drop /= max(gold_w, 1)
        for j, fname in enumerate(feat_names):
            rows.append({
                "feature": fname, "axis": axis,
                "gold_drop": gold_drop[j], "full_drop": full_drop[j],
                "m2_noise": bool(gold_drop[j] <= 0.0),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  M4 — leave-one-group-out
# --------------------------------------------------------------------------- #
def _group_columns(feat_names):
    """Map each registry group -> column indices; add 'unclassified' residual."""
    groups = {}
    covered = set()
    for gname, prefixes in FEATURE_SUBSETS.items():
        idx = [i for i, c in enumerate(feat_names) if _match_prefixes(c, prefixes)]
        groups[gname] = idx
        covered.update(idx)
    groups["unclassified"] = [i for i in range(len(feat_names)) if i not in covered]
    return groups


def _gold_eval_axes(X, y_masked, vmask, gold, splits, scope):
    """Per-axis (full_w, gold_w) gold-test balanced acc."""
    out = {}
    for axis, y in y_masked.items():
        full, goldr = [], []
        for tr, te in _tscv_folds(splits, vmask, scope):
            m = train_et(X[tr], y[tr])
            pred = m.predict(X[te])
            full.append((balanced_accuracy_score(y[te], pred), len(te)))
            g = gold[te]
            if g.sum() >= 8 and len(np.unique(y[te][g])) >= 2:
                goldr.append((balanced_accuracy_score(y[te][g], pred[g]), int(g.sum())))
        out[axis] = (wmean(full), wmean(goldr))
    return out


def group_ablation(X, y_masked, vmask, gold, feat_names, splits, scope):
    groups = _group_columns(feat_names)
    base = _gold_eval_axes(X, y_masked, vmask, gold, splits, scope)
    rows = []
    for gname, idx in groups.items():
        if not idx:
            continue
        keep = [i for i in range(X.shape[1]) if i not in set(idx)]
        sub = _gold_eval_axes(X[:, keep], y_masked, vmask, gold, splits, scope)
        for axis in y_masked:
            rows.append({
                "group": gname, "n_feat": len(idx), "axis": axis,
                "base_gold": base[axis][1], "ablated_gold": sub[axis][1],
                "gold_delta": sub[axis][1] - base[axis][1],
            })
    return pd.DataFrame(rows), base


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load X once; derive per-axis y from labels
    Xdf, feat_names = load_features()
    X = np.nan_to_num(Xdf.values.astype(np.float32))
    labels = load_labels()
    y_axes = {}
    vmask_axes = {}
    for axis in AXES:
        valid = labels[axis].notna().values
        y_axes[axis] = np.where(valid, labels[axis].values, -1).astype(int)
        vmask_axes[axis] = valid
    vmask = np.logical_and.reduce([vmask_axes[a] for a in AXES])
    # Single masked y dict used by all blocks (M1/M2/M4/prune)
    y_masked = {a: np.where(vmask, y_axes[a], -1).astype(int) for a in AXES}

    gold = pd.read_csv(FEAT_DIR / "denoise_meta.csv")["gold"].values.astype(bool)
    scope = np.ones(len(X), bool)
    splits = load_splits()
    print(f"X={X.shape}  common-valid={vmask.sum()}  gold={gold.sum()}  "
          f"(N_NULL={N_NULL}, N_PERM_REPEAT={N_PERM_REPEAT})")

    # --- M1 null importance (label ablation) ---
    print("\n[M1] null-importance (label ablation) ...")
    m1 = null_importance(X, y_masked, vmask, feat_names)
    m1.to_csv(OUT_DIR / "feature_null_importance.csv", index=False)
    for axis in AXES:
        sub = m1[m1.axis == axis]
        n_noise = int(sub.m1_noise.sum())
        print(f"  {AXIS_SHORT[axis]:16s}  noise(p>{P_THRESH})={n_noise}/{len(sub)}")

    # --- M2 gold-test permutation importance ---
    print("\n[M2] gold-test permutation importance ...")
    m2 = perm_importance(X, y_masked, vmask, gold, feat_names, splits, scope)
    m2.to_csv(OUT_DIR / "feature_perm_importance.csv", index=False)
    for axis in AXES:
        sub = m2[m2.axis == axis]
        n_noise = int(sub.m2_noise.sum())
        top = sub.sort_values("gold_drop", ascending=False).head(5)
        print(f"  {AXIS_SHORT[axis]:16s}  noise(gold_drop<=0)={n_noise}/{len(sub)}")
        for _, r in top.iterrows():
            print(f"      +signal {r.feature:28s} gold_drop={r.gold_drop:+.4f} "
                  f"full_drop={r.full_drop:+.4f}")

    # --- M4 group ablation ---
    print("\n[M4] leave-one-group-out (gold-test delta) ...")
    m4, base = group_ablation(X, y_masked, vmask, gold, feat_names, splits, scope)
    m4.to_csv(OUT_DIR / "feature_group_ablation.csv", index=False)
    for axis in AXES:
        sub = m4[m4.axis == axis].sort_values("gold_delta")
        print(f"  {AXIS_SHORT[axis]:16s} base_gold={base[axis][1]:.4f}  "
              f"(more negative delta = group helps more)")
        for _, r in sub.iterrows():
            print(f"      drop {r.group:14s}({int(r.n_feat):3d}d)  "
                  f"gold_delta={r.gold_delta:+.4f}")

    # --- M3 cross-axis verdict + conservative noise set ---
    print("\n[M3] cross-axis noise verdict ...")
    m1p = m1.pivot(index="feature", columns="axis", values="m1_noise")
    m2p = m2.pivot(index="feature", columns="axis", values="m2_noise")
    verdict = pd.DataFrame(index=feat_names)
    verdict["m1_all"] = m1p[AXES].all(axis=1).reindex(feat_names).values
    verdict["m2_all"] = m2p[AXES].all(axis=1).reindex(feat_names).values
    verdict["noise"] = verdict["m1_all"] & verdict["m2_all"]
    noise_feats = verdict.index[verdict["noise"]].tolist()
    verdict.reset_index(names="feature").to_csv(OUT_DIR / "noise_feature_list.csv", index=False)
    print(f"  pure-noise (M1 & M2, all 3 axes) = {len(noise_feats)} features")
    for f in noise_feats:
        print(f"      noise: {f}")

    # --- prune verification (non-destructive, reuses base from M4) ---
    print("\n[prune] non-destructive gold-test re-validation ...")
    if noise_feats:
        keep = [i for i, c in enumerate(feat_names) if c not in set(noise_feats)]
        pruned = _gold_eval_axes(X[:, keep], y_masked, vmask, gold, splits, scope)
        print(f"  baseline 156d   vs   pruned {len(keep)}d")
        for axis in AXES:
            bg = base[axis][1]; pg = pruned[axis][1]
            print(f"    {AXIS_SHORT[axis]:16s} gold: {bg:.4f} -> {pg:.4f}  "
                  f"(delta {pg - bg:+.4f})")
        jd = pruned["label_joint"][1] - base["label_joint"][1]
        print(f"  GATE: gold joint delta = {jd:+.4f}  "
              f"({'PASS (>= -0.005, set is noise)' if jd >= -0.005 else 'FAIL (set is NOT pure noise)'})")
    else:
        print("  no pure-noise features under the conservative AND criterion; "
              "156d features all contribute on >=1 axis.")

    print(f"\nDone in {time.time() - t0:.1f}s. Tables -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
