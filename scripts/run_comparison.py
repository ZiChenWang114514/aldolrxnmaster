#!/usr/bin/env python3
"""Fair Comparison: Same data, same splits, different feature sets.

Parts:
  A: Intersection comparison (4 feature sets × 4 split types)
  B: Leakage check (V2 data, old vs new group_id)
  C: Output (CSV + LaTeX + console)

Usage:
    conda run -n aldol-rxn python scripts/run_fair_comparison.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from chiralaldol.ze_enolate_generator import get_ze_weights
from chiralaldol.steric_descriptors import STERIC_DESC_NAMES
from chiralaldol.rebuild.constants import BASE_CATEGORIES, ACTIVATOR_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fair_comparison")

TSCV_FOLDS = [
    {"name": "fold1", "train_max": 2013, "test_min": 2014, "test_max": 2015},
    {"name": "fold2", "train_max": 2015, "test_min": 2016, "test_max": 2017},
    {"name": "fold3", "train_max": 2017, "test_min": 2018, "test_max": 2019},
    {"name": "fold4", "train_max": 2019, "test_min": 2020, "test_max": 2099},
]


def train_xgb(X_tr, y_tr, X_val, y_val):
    """3-config grid search XGBoost."""
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
        cfg.update({"objective": "multi:softprob", "num_class": 4,
                    "tree_method": "hist", "random_state": 42,
                    "n_jobs": 1, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def murcko_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return "unknown"
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return "unknown"


def generate_splits(df):
    """Generate all split types on a dataframe with group_id, Year, Aldehyde."""
    n = len(df)
    groups = df["group_id"].values
    years = df["Year"].values
    splits = {}

    # TSCV 4-fold
    for fold in TSCV_FOLDS:
        tr = np.where(years <= fold["train_max"])[0]
        te = np.where((years >= fold["test_min"]) & (years <= fold["test_max"]))[0]
        if len(tr) < 10 or len(te) < 5:
            continue
        va = tr[-max(1, len(tr) // 10):]
        tr = tr[:-len(va)]
        splits[f"tscv_{fold['name']}"] = {"train": tr, "val": va, "test": te}

    # Single temporal 2019+
    tr = np.where(years <= 2018)[0]
    te = np.where(years >= 2019)[0]
    if len(tr) >= 10 and len(te) >= 5:
        va = tr[-max(1, len(tr) // 10):]
        tr_sub = tr[:-len(va)]
        splits["temporal_2019"] = {"train": tr_sub, "val": va, "test": te}

    # Scaffold
    ald_col = "canonical_Aldehyde" if "canonical_Aldehyde" in df.columns else "Aldehyde"
    scaffolds = df[ald_col].apply(murcko_scaffold).factorize()[0]
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tv, te = next(gss.split(np.arange(n), groups=scaffolds))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        tr, va = next(gss2.split(tv, groups=scaffolds[tv]))
        splits["scaffold"] = {"train": tv[tr], "val": tv[va], "test": te}
    except Exception:
        pass

    # Grouped random (5 seeds)
    for seed in [42, 123, 456, 789, 1024]:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tv, te = next(gss.split(np.arange(n), groups=groups))
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            tr, va = next(gss2.split(tv, groups=groups[tv]))
            splits[f"grouped_{seed}"] = {"train": tv[tr], "val": tv[va], "test": te}
        except Exception:
            pass

    return splits


def evaluate(X, y, splits, label=""):
    """Run XGBoost on all splits, return results dict."""
    results = {}
    for split_name, sp in splits.items():
        tr, va, te = sp["train"], sp["val"], sp["test"]
        if len(tr) < 10 or len(te) < 5:
            continue
        model = train_xgb(X[tr], y[tr], X[va], y[va])
        y_pred = model.predict(X[te])
        bal_acc = balanced_accuracy_score(y[te], y_pred)
        mcc = matthews_corrcoef(y[te], y_pred)
        results[split_name] = {"bal_acc": bal_acc, "mcc": mcc, "n_test": len(te)}
    return results


# ─────────────────────── PART A ───────────────────────

def run_part_a():
    logger.info("=" * 70)
    logger.info("PART A: Intersection Fair Comparison")
    logger.info("=" * 70)

    # Load data
    v3_feats = pd.read_csv(PROJECT / "data/v3/features/v3_features_raw.csv")
    labels = pd.read_csv(PROJECT / "data/v3/features/labels.csv")
    ket = pd.read_csv(PROJECT / "data/v3/mechaware/ketone_steric.csv")
    z = pd.read_csv(PROJECT / "data/v3/mechaware/z_enolate_steric.csv")
    e = pd.read_csv(PROJECT / "data/v3/mechaware/e_enolate_steric.csv")
    interim = pd.read_csv(PROJECT / "data/v3/interim/09_conditions.csv",
                          usecols=["original_index", "Reaction_Class", "group_id", "Year",
                                   "canonical_Aldehyde", "Aldehyde"])

    # Inner merge on original_index
    merged = v3_feats.merge(labels, on="original_index")
    merged = merged.merge(ket, on="original_index")
    merged = merged.merge(z, on="original_index")
    merged = merged.merge(e, on="original_index")
    merged = merged.merge(interim[["original_index", "Reaction_Class", "group_id", "Year",
                                    "canonical_Aldehyde", "Aldehyde"]].drop_duplicates("original_index"),
                          on="original_index")

    # Filter Evans
    merged = merged[merged["Reaction_Class"] == "EvansAux"].reset_index(drop=True)
    logger.info(f"Intersection Evans rows: {len(merged)}")

    # Compute BW features
    v3_feat_cols = [c for c in v3_feats.columns if c != "original_index"]
    ket_cols = [c for c in ket.columns if c != "original_index"]
    z_cols = [c for c in z.columns if c != "original_index"]
    e_cols = [c for c in e.columns if c != "original_index"]

    w_z_list, w_e_list = [], []
    for _, row in merged.iterrows():
        base = "no_base"
        for cat in BASE_CATEGORIES:
            if row.get(f"base_{cat}", 0) > 0.5:
                base = cat
                break
        act = ""
        for cat in ACTIVATOR_CATEGORIES:
            if row.get(f"act_{cat}", 0) > 0.5:
                act = cat
                break
        wz, we = get_ze_weights(base, act)
        w_z_list.append(wz)
        w_e_list.append(we)
    merged["w_Z"] = w_z_list
    merged["w_E"] = w_e_list

    bw_cols = []
    for name in STERIC_DESC_NAMES:
        zc, ec = f"z_{name}", f"e_{name}"
        if zc in merged.columns and ec in merged.columns:
            bw_col = f"bw_{name}"
            merged[bw_col] = merged["w_Z"] * merged[zc] + merged["w_E"] * merged[ec]
            bw_cols.append(bw_col)

    # Define 4 feature sets
    # V3 steric cols (first 34 = enolate 24 + ald 10)
    steric_cols = [c for c in v3_feat_cols if c.startswith(("Vbur_", "L_", "B1_", "B5_",
                   "sin_tau", "cos_tau", "n_conformers", "n_clusters",
                   "ald_L", "ald_B1", "ald_B5", "ald_Vbur", "ald_n_"))]
    cond_cols = [c for c in v3_feat_cols if c.startswith(("base_", "metal_", "solvent_", "act_", "has_"))]
    aux_cols = [c for c in v3_feat_cols if c.startswith(("aux_rg_", "n_defined")) or c == "aux_config_R"]

    # V2-style: steric(34d) + conditions subset + aux
    v2_cond_cols = [c for c in cond_cols if not c.endswith(("_pKa", "_steric_A", "_nucleophilicity",
                    "_coordination_num", "_ionic_radius_pm", "_pearson_hardness",
                    "_epsilon", "_viscosity", "_bp"))]
    v2_style_cols = steric_cols + v2_cond_cols + aux_cols

    feature_sets = {
        "V2-style": [c for c in v2_style_cols if c in merged.columns],
        "V3": [c for c in v3_feat_cols if c in merged.columns],
        "MechAware-BW": bw_cols + ["w_Z", "w_E"] + [c for c in cond_cols if c in merged.columns] + [c for c in aux_cols if c in merged.columns],
        "MechAware-Full": ket_cols + z_cols + e_cols + bw_cols + ["w_Z", "w_E"] + [c for c in cond_cols if c in merged.columns] + [c for c in aux_cols if c in merged.columns],
    }

    for name, cols in feature_sets.items():
        logger.info(f"  {name}: {len(cols)}d")

    # Generate splits
    splits = generate_splits(merged)
    logger.info(f"  Generated {len(splits)} splits")

    # Evaluate
    y = merged["label_joint"].values
    all_results = []

    for feat_name, cols in feature_sets.items():
        X = merged[cols].values.astype(np.float32)
        np.nan_to_num(X, copy=False)
        results = evaluate(X, y, splits, feat_name)

        # Aggregate
        tscv_accs = [v["bal_acc"] for k, v in results.items() if "tscv" in k]
        grouped_accs = [v["bal_acc"] for k, v in results.items() if "grouped" in k]
        scaffold_acc = results.get("scaffold", {}).get("bal_acc", None)
        temporal_acc = results.get("temporal_2019", {}).get("bal_acc", None)

        row = {
            "Feature Set": feat_name,
            "Dim": len(cols),
            "TSCV mean": np.mean(tscv_accs) if tscv_accs else None,
            "TSCV std": np.std(tscv_accs) if tscv_accs else None,
            "Scaffold": scaffold_acc,
            "Grouped mean": np.mean(grouped_accs) if grouped_accs else None,
            "Grouped std": np.std(grouped_accs) if grouped_accs else None,
            "Temporal": temporal_acc,
        }
        all_results.append(row)
        scaf_str = f"{scaffold_acc:.4f}" if scaffold_acc else "N/A"
        logger.info(f"  {feat_name} ({len(cols)}d): TSCV={row['TSCV mean']:.4f}±{row['TSCV std']:.4f} | "
                    f"Scaffold={scaf_str} | Grouped={row['Grouped mean']:.4f}±{row['Grouped std']:.4f}")

    return pd.DataFrame(all_results)


# ──��──────────────────── PART B ───────────────────────

def run_part_b():
    logger.info("\n" + "=" * 70)
    logger.info("PART B: Leakage Check (V2 data 1801 rows)")
    logger.info("=" * 70)

    # Load V2 features
    try:
        from chiralaldol.feature_builder import build_chiralaldol_v2_features
        X_v2, v2_names = build_chiralaldol_v2_features(PROJECT)
        logger.info(f"  V2 features loaded: {X_v2.shape}")
    except Exception as e:
        logger.error(f"  Cannot load V2 features: {e}")
        return None

    np.nan_to_num(X_v2, copy=False)

    # Load V2 labels
    labels_v2 = pd.read_csv(PROJECT / "data/processed/features/labels.csv")
    y_v2 = labels_v2["label_joint"].values

    # Load V2 clean data for group_id recomputation
    v2_clean = pd.read_csv(PROJECT / "data/processed/evans_v2_clean.csv")
    years_v2 = v2_clean["Year"].values

    results_b = {}

    # ── OLD splits ──
    logger.info("  Evaluating with OLD splits...")
    old_splits_dir = PROJECT / "data/processed/splits"
    old_results = {}

    for split_file in ["evans_scaffold.json", "evans_grouped_random_seed42.json", "evans_temporal.json"]:
        path = old_splits_dir / split_file
        if not path.exists():
            continue
        with open(path) as f:
            sp = json.load(f)
        tr = np.array(sp["train"])
        va = np.array(sp.get("val", sp["train"][-len(sp["train"]) // 10:]))
        te = np.array(sp["test"])
        if len(tr) < 10 or len(te) < 5:
            continue
        model = train_xgb(X_v2[tr], y_v2[tr], X_v2[va], y_v2[va])
        acc = balanced_accuracy_score(y_v2[te], model.predict(X_v2[te]))
        name = split_file.replace("evans_", "").replace(".json", "")
        old_results[name] = acc
        logger.info(f"    OLD {name}: {acc:.4f}")

    results_b["old"] = old_results

    # ── NEW splits (role-aware group_id) ──
    logger.info("  Recomputing group_id with role-aware substrate key...")
    ketone_col = "Ketone"
    aldehyde_col = "Aldehyde"

    def safe_canonical(smi):
        if pd.isna(smi) or not str(smi).strip():
            return ""
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return str(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    new_group_map = {}
    gid_counter = 0
    new_groups = []
    for _, row in v2_clean.iterrows():
        k = safe_canonical(row.get(ketone_col, ""))
        a = safe_canonical(row.get(aldehyde_col, ""))
        key = f"{k}||{a}"
        if key not in new_group_map:
            new_group_map[key] = gid_counter
            gid_counter += 1
        new_groups.append(new_group_map[key])
    new_groups = np.array(new_groups)

    # Generate new scaffold split
    ald_col = "Aldehyde"
    scaffolds = v2_clean[ald_col].apply(murcko_scaffold).factorize()[0]

    new_results = {}
    n = len(v2_clean)

    # New scaffold
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tv, te = next(gss.split(np.arange(n), groups=scaffolds))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        tr, va = next(gss2.split(tv, groups=scaffolds[tv]))
        model = train_xgb(X_v2[tv[tr]], y_v2[tv[tr]], X_v2[tv[va]], y_v2[tv[va]])
        acc = balanced_accuracy_score(y_v2[te], model.predict(X_v2[te]))
        new_results["scaffold"] = acc
        logger.info(f"    NEW scaffold: {acc:.4f}")
    except Exception as e:
        logger.warning(f"    scaffold split failed: {e}")

    # New grouped
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tv, te = next(gss.split(np.arange(n), groups=new_groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        tr, va = next(gss2.split(tv, groups=new_groups[tv]))
        model = train_xgb(X_v2[tv[tr]], y_v2[tv[tr]], X_v2[tv[va]], y_v2[tv[va]])
        acc = balanced_accuracy_score(y_v2[te], model.predict(X_v2[te]))
        new_results["grouped_random_seed42"] = acc
        logger.info(f"    NEW grouped: {acc:.4f}")
    except Exception as e:
        logger.warning(f"    grouped split failed: {e}")

    # New temporal
    tr = np.where(years_v2 <= 2018)[0]
    te = np.where(years_v2 >= 2019)[0]
    if len(tr) >= 10 and len(te) >= 5:
        va = tr[-len(tr) // 10:]
        tr_sub = tr[:-len(va)]
        model = train_xgb(X_v2[tr_sub], y_v2[tr_sub], X_v2[va], y_v2[va])
        acc = balanced_accuracy_score(y_v2[te], model.predict(X_v2[te]))
        new_results["temporal"] = acc
        logger.info(f"    NEW temporal: {acc:.4f}")

    results_b["new"] = new_results

    # Compute deltas
    logger.info("\n  === LEAKAGE DETECTION ===")
    for split_name in ["scaffold", "grouped_random_seed42", "temporal"]:
        old_v = old_results.get(split_name, None)
        new_v = new_results.get(split_name, None)
        if old_v and new_v:
            delta = old_v - new_v
            leaked = "YES ⚠️" if delta > 0.05 else "no"
            logger.info(f"    {split_name}: OLD={old_v:.4f} NEW={new_v:.4f} Δ={delta:+.4f} Leakage={leaked}")

    return results_b


# ─────────���───────────── PART C ───────────────────────

def run_part_c(df_a, results_b):
    logger.info("\n" + "=" * 70)
    logger.info("PART C: Output")
    logger.info("=" * 70)

    # Console table
    print("\n=== FAIR COMPARISON (same rows, same splits) ===\n")
    print(f"{'Feature Set':<18} {'Dim':>4} {'TSCV mean±std':>15} {'Scaffold':>10} {'Grouped mean±std':>18} {'Temporal':>10}")
    print("-" * 80)
    for _, row in df_a.iterrows():
        tscv = f"{row['TSCV mean']:.4f}±{row['TSCV std']:.4f}" if row['TSCV mean'] else "N/A"
        scaf = f"{row['Scaffold']:.4f}" if row['Scaffold'] else "N/A"
        grp = f"{row['Grouped mean']:.4f}±{row['Grouped std']:.4f}" if row['Grouped mean'] else "N/A"
        temp = f"{row['Temporal']:.4f}" if row['Temporal'] else "N/A"
        print(f"{row['Feature Set']:<18} {row['Dim']:>4} {tscv:>15} {scaf:>10} {grp:>18} {temp:>10}")

    # Save CSV
    out_dir = PROJECT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fair_comparison_20260516.csv"
    df_a.to_csv(csv_path, index=False)
    logger.info(f"\n  Saved: {csv_path}")

    # LaTeX
    print("\n=== LaTeX Table ===\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Fair comparison of feature sets (same 1551 Evans reactions, same splits)}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Feature Set & Dim & TSCV (4-fold) & Scaffold & Grouped (5 seeds) \\")
    print(r"\midrule")
    for _, row in df_a.iterrows():
        tscv = f"${row['TSCV mean']:.3f} \\pm {row['TSCV std']:.3f}$" if row['TSCV mean'] else "---"
        scaf = f"${row['Scaffold']:.3f}$" if row['Scaffold'] else "---"
        grp = f"${row['Grouped mean']:.3f} \\pm {row['Grouped std']:.3f}$" if row['Grouped mean'] else "---"
        print(f"{row['Feature Set']} & {row['Dim']} & {tscv} & {scaf} & {grp} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Leakage results
    if results_b:
        print("\n=== LEAKAGE CHECK ===\n")
        print(f"{'Split':<20} {'Old group_id':>12} {'New group_id':>12} {'Delta':>8} {'Leakage?':>10}")
        print("-" * 65)
        for split_name in ["scaffold", "grouped_random_seed42", "temporal"]:
            old_v = results_b.get("old", {}).get(split_name)
            new_v = results_b.get("new", {}).get(split_name)
            if old_v and new_v:
                delta = old_v - new_v
                leaked = "YES ⚠️" if delta > 0.05 else "no"
                print(f"{split_name:<20} {old_v:>12.4f} {new_v:>12.4f} {delta:>+8.4f} {leaked:>10}")


def main():
    t0 = time.time()
    df_a = run_part_a()
    results_b = run_part_b()
    run_part_c(df_a, results_b)
    logger.info(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
