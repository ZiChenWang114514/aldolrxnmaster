#!/usr/bin/env python3
"""V4 Evans-only Benchmark: 11 models × 10 splits on Evans subset (1654 rows).

Filters all existing splits_v4/ JSON files to Evans-only rows, then runs the
same model suite as run_all_models_v4.py. Allows fair comparison of Evans-only
performance vs. full-dataset benchmark (benchmark_v4.csv).

Usage:
    conda run -n aldol-rxn python scripts/run_evans_benchmark_v4.py 2>&1 | tee logs/benchmark_v4_evans.log
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
CLEAN_DIR = PROJECT / "data" / "clean_v4"
SPLITS_DIR = PROJECT / "data" / "splits_v4"
PRED_DIR = PROJECT / "results" / "predictions_v4"

TARGET_LABEL = "label_joint"
SUBSET = "evans"  # auxiliary_type value to filter on

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evans_benchmark_v4")


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
                    "random_state": 42, "n_jobs": 8, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
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
                        colsample_bytree=0.7, random_state=42, n_jobs=8, verbose=-1)
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


def train_et(X_tr, y_tr, X_val, y_val):
    m = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42,
                              n_jobs=8, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_rf(X_tr, y_tr, X_val, y_val):
    m = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42,
                                n_jobs=8, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_knn(X_tr, y_tr, X_val, y_val, k=5):
    m = KNeighborsClassifier(n_neighbors=k)
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


# ═══════════════════════════ FEATURE LOADERS ═══════════════════════════

def _load_all_features():
    """Load V4 features, labels, and Evans mask.

    Returns (X, y, feature_names, combined_mask) where combined_mask is the
    intersection of label validity and auxiliary_type == 'evans'.
    """
    X_df = pd.read_csv(FEAT_DIR / "v4_features.csv")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    clean = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")

    if TARGET_LABEL not in labels.columns:
        raise ValueError(f"Target label '{TARGET_LABEL}' not in labels.csv. "
                         f"Available: {labels.columns.tolist()}")

    valid_mask = labels[TARGET_LABEL].notna().values
    evans_mask = (clean["auxiliary_type"] == SUBSET).values

    combined_mask = valid_mask & evans_mask
    n_total = len(labels)
    n_evans = evans_mask.sum()
    n_valid = valid_mask.sum()
    n_combined = combined_mask.sum()
    logger.info(f"Total rows: {n_total} | Evans: {n_evans} | Valid label: {n_valid} | Combined: {n_combined}")

    y_full = labels[TARGET_LABEL].values
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    return X, y_full, feature_names, combined_mask


def load_full(X, y, feat_names):
    return X


def load_steric_only(X, y, feat_names):
    idx = [i for i, c in enumerate(feat_names)
           if c.startswith(("Vbur_", "L_", "B1_", "B5_", "sin_tau", "cos_tau",
                            "n_conformers", "n_clusters", "ald_"))]
    return X[:, idx] if idx else X


def load_conditions_only(X, y, feat_names):
    idx = [i for i, c in enumerate(feat_names) if c.startswith("feat_")]
    return X[:, idx] if idx else X


def load_condaux_chiral(X, y, feat_names):
    idx = [i for i, c in enumerate(feat_names)
           if c.startswith(("feat_", "aux_", "chiral_", "aux_rg_"))
           or c == "n_defined_stereocenters"]
    return X[:, idx] if idx else X


def load_chiral_only(X, y, feat_names):
    idx = [i for i, c in enumerate(feat_names) if c.startswith("chiral_")]
    return X[:, idx] if idx else X


def load_no_chiral(X, y, feat_names):
    idx = [i for i, c in enumerate(feat_names)
           if not c.startswith(("chiral_", "aux_rg_", "aux_oppolzer", "chiralenv_", "ald_pri_"))]
    return X[:, idx] if idx else X


# ═══════════════════════════ MECHAWARE ═══════════════════════════

_MA_FULL = None
_MA_BW = None


def _load_mechaware():
    global _MA_FULL, _MA_BW
    if _MA_FULL is None:
        ma_path = FEAT_DIR / "v4_mechaware_full.csv"
        if ma_path.exists():
            _MA_FULL = pd.read_csv(ma_path).values.astype(np.float32)
            np.nan_to_num(_MA_FULL, copy=False)
            logger.info(f"  Loaded MechAware full: {_MA_FULL.shape}")
        bw_path = FEAT_DIR / "v4_mechaware_bw.csv"
        if bw_path.exists():
            _MA_BW = pd.read_csv(bw_path).values.astype(np.float32)
            np.nan_to_num(_MA_BW, copy=False)
            logger.info(f"  Loaded MechAware BW: {_MA_BW.shape}")


def load_ma_full_plus(X, y, feat_names):
    _load_mechaware()
    if _MA_FULL is None:
        return X
    new_idx = [i for i, c in enumerate(feat_names)
               if c.startswith(("chiral_", "aux_rg_", "aux_oppolzer", "chiralenv_", "ald_pri_"))]
    X_new = X[:, new_idx] if new_idx else np.zeros((len(X), 0), dtype=np.float32)
    return np.hstack([_MA_FULL, X_new])


def load_ma_bw_plus(X, y, feat_names):
    _load_mechaware()
    if _MA_BW is None:
        return X
    new_idx = [i for i, c in enumerate(feat_names)
               if c.startswith(("chiral_", "aux_rg_", "aux_oppolzer", "chiralenv_", "ald_pri_"))]
    X_new = X[:, new_idx] if new_idx else np.zeros((len(X), 0), dtype=np.float32)
    return np.hstack([_MA_BW, X_new])


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V4 Evans-only Benchmark")
    logger.info("=" * 70)

    X_all, y_full, feat_names, combined_mask = _load_all_features()
    logger.info(f"Target: {TARGET_LABEL}")
    logger.info(f"Subset: {SUBSET} ({combined_mask.sum()} rows)")

    y = np.where(combined_mask, y_full, -1).astype(int)
    y_valid = y[combined_mask]
    n_classes = len(np.unique(y_valid))
    logger.info(f"Classes: {n_classes}, dist: {dict(zip(*np.unique(y_valid, return_counts=True)))}")

    split_files = sorted(SPLITS_DIR.glob("*.json"))
    splits = {}
    for f in split_files:
        with open(f) as fp:
            splits[f.stem] = json.load(fp)
    logger.info(f"Loaded {len(splits)} splits")

    MODELS = {
        "v4b_full_xgb":        ("v4b", load_full, train_xgb),
        "v4b_full_lgbm":       ("v4b", load_full, train_lgbm),
        "v4b_full_rf":         ("v4b", load_full, train_rf),
        "v4b_full_et":         ("v4b", load_full, train_et),
        "v4b_condaux_xgb":     ("v4b", load_condaux_chiral, train_xgb),
        "v4b_chiral_only_xgb": ("ablation", load_chiral_only, train_xgb),
        "v4b_no_chiral_xgb":   ("ablation", load_no_chiral, train_xgb),
        "ma_full_xgb":         ("mechaware", load_ma_full_plus, train_xgb),
        "ma_bw_xgb":           ("mechaware", load_ma_bw_plus, train_xgb),
        "steronly_xgb":        ("steric", load_steric_only, train_xgb),
        "cond_xgb":            ("baseline", load_conditions_only, train_xgb),
        "majority":            ("baseline", load_full, lambda Xtr, ytr, Xv, yv: _train_majority(Xtr, ytr)),
    }

    all_results = []

    for model_key, (category, feat_loader, trainer) in MODELS.items():
        logger.info(f"\n--- {model_key} ({category}) ---")

        X = feat_loader(X_all, y, feat_names)
        logger.info(f"  Features: {X.shape[1]}d")

        out_dir = PRED_DIR / f"evans_{category}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            tr_raw = np.array(split_data["train"], dtype=int)
            tr = tr_raw[combined_mask[tr_raw]]
            va_raw = np.array(split_data.get("val", []), dtype=int)
            va = va_raw[combined_mask[va_raw]] if len(va_raw) > 0 else np.array([], dtype=int)
            te_raw = np.array(split_data["test"], dtype=int)
            te = te_raw[combined_mask[te_raw]]

            if len(va) == 0:
                va = tr[-max(1, len(tr) // 10):]
                tr = tr[:-len(va)]

            if len(tr) < 10 or len(te) < 3:
                logger.warning(f"  Skipping {split_name}: train={len(tr)}, test={len(te)}")
                continue

            model = trainer(X[tr], y[tr], X[va], y[va])

            y_pred = model.predict(X[te])
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X[te])
            else:
                y_prob = np.zeros((len(te), n_classes))
                for i, p in enumerate(y_pred):
                    y_prob[i, int(p)] = 1.0

            bal_acc = balanced_accuracy_score(y[te], y_pred)
            mcc = matthews_corrcoef(y[te], y_pred)

            out = pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": y_pred})
            for c in range(min(n_classes, y_prob.shape[1])):
                out[f"prob_{c}"] = y_prob[:, c]
            out.to_csv(out_dir / f"{model_key}_{split_name}.csv", index=False)

            all_results.append({
                "model": model_key, "category": category, "split": split_name,
                "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
                "n_train": len(tr), "n_test": len(te),
            })

        model_results = [r for r in all_results if r["model"] == model_key]
        if model_results:
            tscv = [r["bal_acc"] for r in model_results if "tscv" in r["split"]]
            grouped = [r["bal_acc"] for r in model_results if "grouped" in r["split"]]
            scaffold = [r["bal_acc"] for r in model_results if "scaffold" in r["split"]]
            if tscv:
                logger.info(f"  TSCV: {np.mean(tscv):.3f} ± {np.std(tscv):.3f}")
            if grouped:
                logger.info(f"  Grouped: {np.mean(grouped):.3f} ± {np.std(grouped):.3f}")
            if scaffold:
                logger.info(f"  Scaffold: {scaffold[0]:.3f}")

    results_df = pd.DataFrame(all_results)
    table_path = PROJECT / "results" / "tables" / "benchmark_v4_evans.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path, index=False)
    logger.info(f"\nSaved results to {table_path}")

    print("\n" + "=" * 80)
    print("V4 EVANS-ONLY BENCHMARK SUMMARY")
    print("=" * 80)
    summary_rows = []
    for model_key in MODELS:
        mr = [r for r in all_results if r["model"] == model_key]
        if not mr:
            continue
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if "scaffold" in r["split"]]
        cat = mr[0]["category"]
        summary_rows.append({
            "model": model_key,
            "category": cat,
            "TSCV": f"{np.mean(tscv):.3f}±{np.std(tscv):.3f}" if tscv else "—",
            "Scaffold": f"{scaffold[0]:.3f}" if scaffold else "—",
            "Grouped": f"{np.mean(grouped):.3f}±{np.std(grouped):.3f}" if grouped else "—",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


def _train_majority(X_tr, y_tr):
    m = MajorityClassifier()
    m.fit(X_tr, y_tr)
    return m


if __name__ == "__main__":
    main()
