#!/usr/bin/env python3
"""Unified V3 model benchmark: 15 active models × 10 splits.

Handles feature precomputation, training, evaluation, and result table generation.

Usage:
    conda run -n aldol-rxn python scripts/run_all_models_v3.py
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
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.aldolrxnmaster.evaluation.metrics import compute_all_metrics

DATA = PROJECT / "data"
FEAT = DATA / "features"
SPLITS = DATA / "splits"
PRED_DIR = PROJECT / "results" / "predictions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_v3")


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


def train_et(X_tr, y_tr, X_val, y_val):
    sw = compute_sample_weight("balanced", y_tr)
    m = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=1,
                              class_weight="balanced")
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


def train_rf(X_tr, y_tr, X_val, y_val):
    m = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=1,
                                class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_knn(X_tr, y_tr, X_val, y_val, k=5):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_tr, y_tr)
    return m


class MajorityClassifier:
    def fit(self, X, y, **kw):
        self.majority = pd.Series(y).mode()[0]
        self.n_classes = len(np.unique(y))
    def predict(self, X):
        return np.full(len(X), self.majority)
    def predict_proba(self, X):
        p = np.zeros((len(X), self.n_classes))
        p[:, self.majority] = 1.0
        return p


# ═══════════════════════════ FEATURE LOADERS ═══════════════════════════

def _load_labels():
    """Load Evans labels aligned with features."""
    labels = pd.read_csv(FEAT / "labels.csv")
    # Filter to Evans only using interim data
    interim = pd.read_csv(DATA / "interim" / "09_conditions.csv",
                          usecols=["original_index", "Reaction_Class"])
    evans_oi = set(interim[interim["Reaction_Class"] == "EvansAux"]["original_index"])
    labels = labels[labels["original_index"].isin(evans_oi)].reset_index(drop=True)
    return labels


def _load_v3_features_aligned(labels_oi):
    """Load V3 87d features aligned with labels."""
    df = pd.read_csv(FEAT / "v3_features.csv")
    df = df[df["original_index"].isin(labels_oi)].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c != "original_index"]
    return df["original_index"].values, df[feat_cols].values.astype(np.float32), feat_cols


def load_v2_style(labels_oi):
    """V2-style 75d: steric(24) + ald(10) + conditions subset(35) + aux(6)."""
    oi, X_full, cols = _load_v3_features_aligned(labels_oi)
    # V3 87d columns: steric(24) + ald(10) + conditions(44) + aux(9)
    # V2 subset: drop extended condition cols (pKa, steric_A, nucleophilicity, coordination, radius, hardness, epsilon, viscosity, bp)
    drop_suffixes = ("_pKa", "_steric_A", "_nucleophilicity", "_coordination_num",
                     "_ionic_radius_pm", "_pearson_hardness", "_epsilon", "_viscosity", "_bp")
    keep_idx = [i for i, c in enumerate(cols) if not c.endswith(drop_suffixes)]
    return oi, X_full[:, keep_idx]


def load_v3_87d(labels_oi):
    """Full V3 87d features."""
    oi, X, _ = _load_v3_features_aligned(labels_oi)
    return oi, X


def load_steric_only(labels_oi):
    """Steric-only 24d (enolate descriptors)."""
    oi, X_full, cols = _load_v3_features_aligned(labels_oi)
    steric_idx = [i for i, c in enumerate(cols)
                  if c.startswith(("Vbur_", "L_", "B1_", "B5_", "sin_tau", "cos_tau", "n_conformers", "n_clusters"))
                  and not c.startswith("ald_")]
    return oi, X_full[:, steric_idx]


def load_conditions(labels_oi):
    """Conditions-only (44d)."""
    oi, X_full, cols = _load_v3_features_aligned(labels_oi)
    cond_idx = [i for i, c in enumerate(cols) if c.startswith(("base_", "metal_", "solvent_", "act_", "has_"))]
    return oi, X_full[:, cond_idx]


def load_condaux(labels_oi):
    """Conditions + auxiliary."""
    oi, X_full, cols = _load_v3_features_aligned(labels_oi)
    idx = [i for i, c in enumerate(cols)
           if c.startswith(("base_", "metal_", "solvent_", "act_", "has_", "aux_", "n_defined"))]
    return oi, X_full[:, idx]


def load_mechaware_bw(labels_oi):
    """MechAware-BW: bw_steric(24) + w(2) + cond(44) + aux."""
    from chiralaldol.ze_enolate_generator import get_ze_weights
    from chiralaldol.steric_descriptors import STERIC_DESC_NAMES
    from chiralaldol.rebuild.constants import BASE_CATEGORIES, ACTIVATOR_CATEGORIES

    oi, X_full, cols = _load_v3_features_aligned(labels_oi)
    oi_set = set(oi)

    # Load Z/E steric
    z_df = pd.read_csv(FEAT / "mechaware" / "z_enolate_steric.csv")
    e_df = pd.read_csv(FEAT / "mechaware" / "e_enolate_steric.csv")
    z_df = z_df[z_df["original_index"].isin(oi_set)].set_index("original_index")
    e_df = e_df[e_df["original_index"].isin(oi_set)].set_index("original_index")

    # Get conditions for BW weights
    full_df = pd.DataFrame(X_full, columns=cols)
    full_df["original_index"] = oi

    bw_data = []
    for _, row in full_df.iterrows():
        row_oi = int(row["original_index"])
        base = "no_base"
        for cat in BASE_CATEGORIES:
            if row.get(f"base_{cat}", 0) > 0.5:
                base = cat; break
        act = ""
        for cat in ACTIVATOR_CATEGORIES:
            if row.get(f"act_{cat}", 0) > 0.5:
                act = cat; break
        wz, we = get_ze_weights(base, act)

        bw_row = [wz, we]
        if row_oi in z_df.index and row_oi in e_df.index:
            z_row = z_df.loc[row_oi]
            e_row = e_df.loc[row_oi]
            for name in STERIC_DESC_NAMES:
                zv = z_row.get(f"z_{name}", 0)
                ev = e_row.get(f"e_{name}", 0)
                bw_row.append(wz * zv + we * ev)
        else:
            bw_row.extend([0.0] * len(STERIC_DESC_NAMES))
        bw_data.append(bw_row)

    X_bw = np.array(bw_data, dtype=np.float32)
    # Add conditions + aux
    cond_idx = [i for i, c in enumerate(cols) if c.startswith(("base_", "metal_", "solvent_", "act_", "has_", "aux_", "n_defined"))]
    X_cond = X_full[:, cond_idx]
    X_final = np.hstack([X_bw, X_cond])
    return oi, X_final


def load_mechaware_full(labels_oi):
    """MechAware-Full: ket(24) + z(24) + e(24) + bw(24) + w(2) + cond + aux."""
    from chiralaldol.steric_descriptors import STERIC_DESC_NAMES

    oi, X_full, cols = _load_v3_features_aligned(labels_oi)
    oi_set = set(oi)

    ket_df = pd.read_csv(FEAT / "mechaware" / "ketone_steric.csv")
    z_df = pd.read_csv(FEAT / "mechaware" / "z_enolate_steric.csv")
    e_df = pd.read_csv(FEAT / "mechaware" / "e_enolate_steric.csv")

    # Filter and align
    common_oi = sorted(oi_set & set(ket_df["original_index"]) & set(z_df["original_index"]) & set(e_df["original_index"]))
    if not common_oi:
        return oi, X_full  # fallback

    ket_df = ket_df[ket_df["original_index"].isin(common_oi)].set_index("original_index").loc[common_oi]
    z_df = z_df[z_df["original_index"].isin(common_oi)].set_index("original_index").loc[common_oi]
    e_df = e_df[e_df["original_index"].isin(common_oi)].set_index("original_index").loc[common_oi]

    # Rebuild oi-aligned full df
    oi_to_idx = {o: i for i, o in enumerate(oi)}
    keep_mask = np.array([o in set(common_oi) for o in oi])
    X_full_sub = X_full[keep_mask]
    cols_sub = cols
    oi_sub = np.array(common_oi)

    ket_feat = ket_df[[c for c in ket_df.columns if c.startswith("ket_")]].values.astype(np.float32)
    z_feat = z_df[[c for c in z_df.columns if c.startswith("z_")]].values.astype(np.float32)
    e_feat = e_df[[c for c in e_df.columns if c.startswith("e_")]].values.astype(np.float32)

    # BW features
    from chiralaldol.ze_enolate_generator import get_ze_weights
    from chiralaldol.rebuild.constants import BASE_CATEGORIES, ACTIVATOR_CATEGORIES
    full_df_sub = pd.DataFrame(X_full_sub, columns=cols_sub)
    bw_feats = []
    w_feats = []
    for i, row in full_df_sub.iterrows():
        base = "no_base"
        for cat in BASE_CATEGORIES:
            if row.get(f"base_{cat}", 0) > 0.5: base = cat; break
        act = ""
        for cat in ACTIVATOR_CATEGORIES:
            if row.get(f"act_{cat}", 0) > 0.5: act = cat; break
        wz, we = get_ze_weights(base, act)
        w_feats.append([wz, we])
        bw = wz * z_feat[i] + we * e_feat[i]
        bw_feats.append(bw)

    X_bw = np.array(bw_feats, dtype=np.float32)
    X_w = np.array(w_feats, dtype=np.float32)

    cond_idx = [j for j, c in enumerate(cols_sub) if c.startswith(("base_", "metal_", "solvent_", "act_", "has_", "aux_", "n_defined"))]
    X_cond = X_full_sub[:, cond_idx]

    X_final = np.hstack([ket_feat, z_feat, e_feat, X_bw, X_w, X_cond])
    return oi_sub, X_final


def load_drfp(labels_oi):
    """DRFP 2048d fingerprints."""
    drfp_path = FEAT / "drfp_fps.npz"
    if not drfp_path.exists():
        logger.warning("DRFP not precomputed — computing now...")
        _precompute_drfp()
    data = np.load(drfp_path)
    X = data["X"]
    oi = data["original_index"]
    mask = np.isin(oi, list(labels_oi))
    return oi[mask], X[mask].astype(np.float32)


def load_drfp_cond(labels_oi):
    """DRFP + conditions."""
    oi_drfp, X_drfp = load_drfp(labels_oi)
    oi_v3, X_v3, cols = _load_v3_features_aligned(labels_oi)
    # Align
    common = sorted(set(oi_drfp) & set(oi_v3))
    drfp_map = {o: i for i, o in enumerate(oi_drfp)}
    v3_map = {o: i for i, o in enumerate(oi_v3)}
    cond_idx = [j for j, c in enumerate(cols) if c.startswith(("base_", "metal_", "solvent_", "act_", "has_"))]
    X_d = np.array([X_drfp[drfp_map[o]] for o in common], dtype=np.float32)
    X_c = np.array([X_v3[v3_map[o], cond_idx] for o in common], dtype=np.float32)
    return np.array(common), np.hstack([X_d, X_c])


def _precompute_drfp():
    """Compute DRFP for V3 Evans data."""
    sys.path.insert(0, str(PROJECT / "external" / "drfp" / "src"))
    from drfp import DrfpEncoder

    df = pd.read_csv(DATA / "clean" / "evans_clean.csv")
    rxn_col = "canonical_Raw_Reaction_Smiles" if "canonical_Raw_Reaction_Smiles" in df.columns else "Raw_Reaction_Smiles"
    smiles = df[rxn_col].fillna("").tolist()
    oi = df["original_index"].values

    fps = []
    for smi in smiles:
        try:
            fp = DrfpEncoder.encode([str(smi)], n_folded_length=2048, radius=3, rings=True)
            fps.append(fp[0])
        except Exception:
            fps.append(np.zeros(2048, dtype=np.uint8))

    X = np.array(fps, dtype=np.int8)
    np.savez(FEAT / "drfp_fps.npz", X=X, original_index=oi)
    logger.info(f"  DRFP computed: {X.shape}")


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("V3 Unified Model Benchmark")
    logger.info("=" * 70)

    # Load labels
    labels = _load_labels()
    y = labels["label_joint"].values
    labels_oi = set(labels["original_index"].values)
    n = len(labels)
    logger.info(f"Evans data: {n} rows, {len(np.unique(y))} classes")

    # Load splits
    split_files = sorted(SPLITS.glob("*.json"))
    splits = {}
    for f in split_files:
        name = f.stem
        with open(f) as fp:
            splits[name] = json.load(fp)
    logger.info(f"Loaded {len(splits)} splits")

    # Model registry
    MODELS = {
        # steric
        "cv2_xgb": ("steric", "v2_style", train_xgb),
        "cv2_lgbm": ("steric", "v2_style", train_lgbm),
        "cv2_et": ("steric", "v2_style", train_et),
        "cv2_rf": ("steric", "v2_style", train_rf),
        "ma_bw": ("steric", "mechaware_bw", train_xgb),
        "ma_full": ("steric", "mechaware_full", train_xgb),
        "steronly_xgb": ("steric", "steric_only", train_xgb),
        # fp
        "drfp_xgb": ("fp", "drfp", train_xgb),
        "drfp_cond_xgb": ("fp", "drfp_cond", train_xgb),
        # baseline
        "cond_xgb": ("baseline", "conditions", train_xgb),
        "condaux_xgb": ("baseline", "condaux", train_xgb),
        "knn_1": ("baseline", "v2_style", lambda Xtr, ytr, Xv, yv: train_knn(Xtr, ytr, Xv, yv, k=1)),
        "knn_5": ("baseline", "v2_style", lambda Xtr, ytr, Xv, yv: train_knn(Xtr, ytr, Xv, yv, k=5)),
        "majority": ("baseline", "v2_style", lambda Xtr, ytr, Xv, yv: MajorityClassifier() or MajorityClassifier()),
    }

    # Fix majority trainer
    def train_majority_fn(X_tr, y_tr, X_val, y_val):
        m = MajorityClassifier()
        m.fit(X_tr, y_tr)
        return m
    MODELS["majority"] = ("baseline", "v2_style", train_majority_fn)

    LOADERS = {
        "v2_style": load_v2_style,
        "v3_87d": load_v3_87d,
        "mechaware_bw": load_mechaware_bw,
        "mechaware_full": load_mechaware_full,
        "steric_only": load_steric_only,
        "drfp": load_drfp,
        "drfp_cond": load_drfp_cond,
        "conditions": load_conditions,
        "condaux": load_condaux,
    }

    # Run all models × all splits
    all_results = []

    for model_key, (category, feat_key, trainer) in MODELS.items():
        logger.info(f"\n--- {model_key} ({category}, {feat_key}) ---")

        # Load features
        try:
            oi, X = LOADERS[feat_key](labels_oi)
        except Exception as e:
            logger.error(f"  Feature load failed: {e}")
            continue

        np.nan_to_num(X, copy=False)

        # Build oi→index map for this feature set
        oi_to_idx = {o: i for i, o in enumerate(oi)}
        # Also need labels aligned
        label_map = dict(zip(labels["original_index"], y))
        y_aligned = np.array([label_map.get(o, -1) for o in oi])

        # Ensure output dir
        (PRED_DIR / category).mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            tr_raw = split_data["train"]
            va_raw = split_data.get("val", [])
            te_raw = split_data["test"]

            # Remap split indices (V3 Evans position → our oi position)
            # Load V3 Evans order
            interim = pd.read_csv(DATA / "interim" / "09_conditions.csv",
                                  usecols=["original_index", "Reaction_Class"])
            v3_evans = interim[interim["Reaction_Class"] == "EvansAux"].reset_index(drop=True)
            v3_oi_list = v3_evans["original_index"].tolist()

            def remap(indices):
                result = []
                for idx in indices:
                    if idx < len(v3_oi_list):
                        o = v3_oi_list[idx]
                        if o in oi_to_idx:
                            result.append(oi_to_idx[o])
                return np.array(result, dtype=int)

            tr = remap(tr_raw)
            va = remap(va_raw) if va_raw else tr[-max(1, len(tr)//10):]
            te = remap(te_raw)

            if len(tr) < 10 or len(te) < 3:
                continue
            if len(va) == 0:
                va = tr[-max(1, len(tr)//10):]
                tr = tr[:-len(va)]

            # Train
            model = trainer(X[tr], y_aligned[tr], X[va], y_aligned[va])

            # Predict
            y_pred = model.predict(X[te])
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X[te])
            else:
                y_prob = np.zeros((len(te), 4))
                for i, p in enumerate(y_pred):
                    y_prob[i, int(p)] = 1.0

            bal_acc = balanced_accuracy_score(y_aligned[te], y_pred)

            # Save prediction
            out = pd.DataFrame({"idx": te, "y_true": y_aligned[te], "y_pred": y_pred})
            for c in range(min(4, y_prob.shape[1])):
                out[f"prob_{c}"] = y_prob[:, c]
            out.to_csv(PRED_DIR / category / f"{model_key}_{split_name}.csv", index=False)

            all_results.append({
                "model": model_key, "category": category, "split": split_name,
                "bal_acc": bal_acc, "n_train": len(tr), "n_test": len(te),
            })

        # Summarize per model
        model_results = [r for r in all_results if r["model"] == model_key]
        tscv = [r["bal_acc"] for r in model_results if "tscv" in r["split"]]
        grouped = [r["bal_acc"] for r in model_results if "grouped" in r["split"]]
        scaffold = [r["bal_acc"] for r in model_results if r["split"] == "scaffold"]
        if tscv:
            s_str = f"{scaffold[0]:.4f}" if scaffold else "N/A"
            g_str = f"{np.mean(grouped):.4f}±{np.std(grouped):.4f}" if grouped else "N/A"
            logger.info(f"  TSCV: {np.mean(tscv):.4f}±{np.std(tscv):.4f} | Scaffold: {s_str} | Grouped: {g_str}")

    # Save results table
    results_df = pd.DataFrame(all_results)
    table_path = PROJECT / "results" / "tables" / "benchmark_v3_20260516.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path, index=False)
    logger.info(f"\nResults saved: {table_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("V3 BENCHMARK SUMMARY")
    print("=" * 80)
    summary = results_df.groupby("model").agg(
        tscv_mean=("bal_acc", lambda x: x[results_df.loc[x.index, "split"].str.contains("tscv")].mean()),
    ).reset_index()

    for model_key in MODELS:
        mr = [r for r in all_results if r["model"] == model_key]
        tscv = [r["bal_acc"] for r in mr if "tscv" in r["split"]]
        scaffold = [r["bal_acc"] for r in mr if r["split"] == "scaffold"]
        grouped = [r["bal_acc"] for r in mr if "grouped" in r["split"]]
        if tscv:
            t = f"{np.mean(tscv):.4f}±{np.std(tscv):.4f}"
            s = f"{scaffold[0]:.4f}" if scaffold else "N/A"
            g = f"{np.mean(grouped):.4f}" if grouped else "N/A"
            print(f"  {model_key:<16} TSCV={t}  Scaffold={s}  Grouped={g}")

    logger.info(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
