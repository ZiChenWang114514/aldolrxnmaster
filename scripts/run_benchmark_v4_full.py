#!/usr/bin/env python3
"""V4 Full Benchmark: includes MechAware-Full/BW + original models.

Requires run_mechaware_v4.py to have completed first.

Usage:
    conda run -n aldol-rxn python scripts/run_benchmark_v4_full.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

FEAT_DIR = PROJECT / "data" / "features_v4"
SPLITS_DIR = PROJECT / "data" / "splits_v4"
PRED_DIR = PROJECT / "results" / "predictions_v4"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_v4_full")


# ═══════════════════════════ TRAINERS ═══════════════════════════

def train_xgb(X_tr, y_tr, X_val, y_val):
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15,
         "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multi:softprob", "num_class": 4, "tree_method": "hist",
                    "random_state": 42, "n_jobs": 1, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def train_lgbm(X_tr, y_tr, X_val, y_val):
    from lightgbm import LGBMClassifier
    sw = compute_sample_weight("balanced", y_tr)
    m = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                        colsample_bytree=0.7, random_state=42, n_jobs=1, verbose=-1)
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


def train_rf(X_tr, y_tr, X_val, y_val):
    m = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_et(X_tr, y_tr, X_val, y_val):
    m = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42,
                              n_jobs=-1, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


class MajorityClassifier:
    def fit(self, X, y, **kw):
        self.majority = int(pd.Series(y).mode()[0])
        self.n_classes = len(np.unique(y))
    def predict(self, X):
        return np.full(len(X), self.majority)
    def predict_proba(self, X):
        p = np.zeros((len(X), self.n_classes))
        p[:, self.majority] = 1.0
        return p


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V4 Full Benchmark (with MechAware)")
    logger.info("=" * 70)

    # Load all feature matrices
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].astype(int).values
    n = len(y)

    feat_matrices = {}

    # v4_features (84d — steric + cond + aux)
    v4 = pd.read_csv(FEAT_DIR / "v4_features.csv").values.astype(np.float32)
    np.nan_to_num(v4, copy=False)
    feat_matrices["v4_84d"] = v4

    # steric-only (34d)
    steric = pd.read_csv(FEAT_DIR / "steric_features.csv").values.astype(np.float32)
    np.nan_to_num(steric, copy=False)
    feat_matrices["steric_34d"] = steric

    # conditions + aux (50d)
    cond = pd.read_csv(PROJECT / "data" / "clean_v4" / "condition_features.csv").values.astype(np.float32)
    aux_types = ["evans", "crimmins_thione", "crimmins_oxathione", "other_auxiliary", "myers"]
    clean = pd.read_csv(PROJECT / "data" / "clean_v4" / "substrate_aldol_clean.csv", usecols=["auxiliary_type", "n_defined_stereocenters"])
    aux = np.column_stack([
        *[(clean["auxiliary_type"] == a).astype(int).values for a in aux_types],
        clean["n_defined_stereocenters"].fillna(2).values,
    ]).astype(np.float32)
    condaux = np.hstack([cond, aux])
    feat_matrices["condaux_50d"] = condaux
    feat_matrices["cond_44d"] = cond

    # MechAware features
    ma_full_path = FEAT_DIR / "v4_mechaware_full.csv"
    ma_bw_path = FEAT_DIR / "v4_mechaware_bw.csv"
    if ma_full_path.exists():
        ma_full = pd.read_csv(ma_full_path).values.astype(np.float32)
        np.nan_to_num(ma_full, copy=False)
        feat_matrices["ma_full"] = ma_full
        logger.info(f"MechAware-Full: {ma_full.shape[1]}d")
    if ma_bw_path.exists():
        ma_bw = pd.read_csv(ma_bw_path).values.astype(np.float32)
        np.nan_to_num(ma_bw, copy=False)
        feat_matrices["ma_bw"] = ma_bw
        logger.info(f"MechAware-BW: {ma_bw.shape[1]}d")

    logger.info(f"Data: {n} rows, {len(np.unique(y))} classes")
    logger.info(f"Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Load splits
    splits = {}
    for f in sorted(SPLITS_DIR.glob("*.json")):
        with open(f) as fp:
            splits[f.stem] = json.load(fp)
    logger.info(f"Loaded {len(splits)} splits")

    # Model registry
    MODELS = {
        # MechAware (new)
        "ma_full_xgb":  ("steric", "ma_full", train_xgb),
        "ma_full_lgbm": ("steric", "ma_full", train_lgbm),
        "ma_bw_xgb":    ("steric", "ma_bw", train_xgb),
        # Original steric
        "full_xgb":     ("steric", "v4_84d", train_xgb),
        "full_lgbm":    ("steric", "v4_84d", train_lgbm),
        "full_rf":      ("steric", "v4_84d", train_rf),
        "full_et":      ("steric", "v4_84d", train_et),
        "steronly_xgb": ("steric", "steric_34d", train_xgb),
        # Baseline
        "condaux_xgb":  ("baseline", "condaux_50d", train_xgb),
        "cond_xgb":     ("baseline", "cond_44d", train_xgb),
        "majority":     ("baseline", "v4_84d", lambda Xtr, ytr, Xv, yv: _majority(Xtr, ytr)),
    }

    all_results = []

    for model_key, (category, feat_key, trainer) in MODELS.items():
        if feat_key not in feat_matrices:
            logger.warning(f"  Skipping {model_key}: {feat_key} not available")
            continue

        X = feat_matrices[feat_key]
        logger.info(f"\n--- {model_key} ({category}, {X.shape[1]}d) ---")

        out_dir = PRED_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            tr = np.array(split_data["train"], dtype=int)
            va = np.array(split_data.get("val", []), dtype=int)
            te = np.array(split_data["test"], dtype=int)

            if len(va) == 0:
                va = tr[-max(1, len(tr) // 10):]
                tr = tr[:-len(va)]
            if len(tr) < 10 or len(te) < 3:
                continue

            model = trainer(X[tr], y[tr], X[va], y[va])
            y_pred = model.predict(X[te])
            y_prob = model.predict_proba(X[te]) if hasattr(model, "predict_proba") else np.eye(4)[y_pred.astype(int)]

            bal_acc = balanced_accuracy_score(y[te], y_pred)
            mcc = matthews_corrcoef(y[te], y_pred)

            out = pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": y_pred})
            for c in range(min(4, y_prob.shape[1])):
                out[f"prob_{c}"] = y_prob[:, c]
            out.to_csv(out_dir / f"{model_key}_{split_name}.csv", index=False)

            all_results.append({
                "model": model_key, "category": category, "feat_key": feat_key,
                "dims": X.shape[1], "split": split_name,
                "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                "n_train": len(tr), "n_test": len(te),
            })

        mr = [r for r in all_results if r["model"] == model_key]
        if mr:
            tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
            grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
            if tscv: logger.info(f"  TSCV: {np.mean(tscv):.3f} ± {np.std(tscv):.3f}")
            if grouped: logger.info(f"  Grouped: {np.mean(grouped):.3f} ± {np.std(grouped):.3f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    table_path = PROJECT / "results" / "tables" / "benchmark_v4_full.csv"
    results_df.to_csv(table_path, index=False)

    # Print summary
    print("\n" + "=" * 90)
    print("V4 FULL BENCHMARK SUMMARY")
    print("=" * 90)
    summary = []
    for mk in dict.fromkeys(r["model"] for r in all_results):
        mr = [r for r in all_results if r["model"] == mk]
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        summary.append({
            "model": mk, "dims": mr[0]["dims"],
            "TSCV": f"{np.mean(tscv):.3f}±{np.std(tscv):.3f}" if tscv else "—",
            "Scaffold": f"{scaffold[0]:.3f}" if scaffold else "—",
            "Grouped": f"{np.mean(grouped):.3f}±{np.std(grouped):.3f}" if grouped else "—",
        })
    print(pd.DataFrame(summary).to_string(index=False))
    print("=" * 90)
    print(f"\nTotal time: {time.time()-t0:.1f}s")


def _majority(X_tr, y_tr):
    m = MajorityClassifier()
    m.fit(X_tr, y_tr)
    return m


if __name__ == "__main__":
    main()
