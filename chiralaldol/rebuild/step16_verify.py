"""Step 16: Verification suite — quality checks + XGBoost comparison + chemical sampling."""

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _quality_checks(evans_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Run automated quality checks."""
    checks = {}

    # 1. No NaN in features
    if feature_cols:
        n_nan = evans_df[feature_cols].isna().sum().sum()
        checks["no_nan_features"] = {"pass": n_nan == 0, "detail": f"{n_nan} NaN values"}
    else:
        checks["no_nan_features"] = {"pass": True, "detail": "no features to check"}

    # 2. All products have >=2 stereocenters
    if "n_defined_stereocenters" in evans_df.columns:
        n_below2 = (evans_df["n_defined_stereocenters"] < 2).sum()
        checks["stereocenters_ge2"] = {"pass": n_below2 == 0, "detail": f"{n_below2} below 2"}
    else:
        checks["stereocenters_ge2"] = {"pass": True, "detail": "column not found"}

    # 3. Class distribution not degenerate
    if "label_joint" in evans_df.columns:
        dist = evans_df["label_joint"].value_counts()
        min_frac = dist.min() / len(evans_df) if len(evans_df) > 0 else 0
        checks["class_distribution"] = {
            "pass": min_frac >= 0.03,
            "detail": f"min class frac={min_frac:.3f}, dist={dist.to_dict()}",
        }

    # 4. Year coverage
    if "Year" in evans_df.columns:
        year_range = f"{evans_df['Year'].min():.0f}-{evans_df['Year'].max():.0f}"
        checks["year_coverage"] = {"pass": True, "detail": year_range}

    # 5. Evans row count
    n_evans = len(evans_df)
    checks["evans_count"] = {"pass": n_evans >= 1500, "detail": f"{n_evans} rows"}

    return checks


def _chemical_sample(evans_df: pd.DataFrame, n_sample: int = 20, seed: int = 42) -> pd.DataFrame:
    """Random sample for manual chemical inspection."""
    cols_to_show = [
        "original_index", "Raw_Product_Smiles", "Ketone", "Aldehyde",
        "label_Ca", "label_Cb", "label_SA", "label_joint",
        "cip_Ca_extracted", "cip_Cb_extracted", "cip_match",
        "solvent_name", "solvent_source", "metal",
        "aux_C4_cip", "aux_rgroup_type",
    ]
    cols_available = [c for c in cols_to_show if c in evans_df.columns]
    sample = evans_df[cols_available].sample(n=min(n_sample, len(evans_df)), random_state=seed)
    return sample


def _train_xgboost_tscv(evans_df: pd.DataFrame, feature_cols: list[str],
                         splits: dict) -> dict:
    """Train XGBoost on TSCV folds and report balanced accuracy."""
    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        logger.warning("  XGBoost or sklearn not available, skipping model verification")
        return {"error": "xgboost/sklearn not installed"}

    if not feature_cols or "label_joint" not in evans_df.columns:
        return {"error": "no features or labels"}

    X = evans_df[feature_cols].values.astype(float)
    y = evans_df["label_joint"].values

    results = {}
    tscv_accs = []

    for split_name, split_info in splits.items():
        if not split_name.startswith("tscv_"):
            continue

        train_idx = np.array(split_info["train"])
        test_idx = np.array(split_info["test"])

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Simple XGBoost with reasonable defaults
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=1,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
        results[split_name] = float(bal_acc)
        tscv_accs.append(bal_acc)
        logger.info(f"    {split_name}: bal_acc={bal_acc:.4f}")

    if tscv_accs:
        mean_acc = np.mean(tscv_accs)
        std_acc = np.std(tscv_accs)
        results["tscv_mean"] = float(mean_acc)
        results["tscv_std"] = float(std_acc)
        logger.info(f"    TSCV mean: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"    V2 baseline: 0.682 ± 0.044")
        if mean_acc >= 0.682:
            logger.info(f"    ✓ V3 >= V2 baseline")
        else:
            logger.warning(f"    ✗ V3 < V2 baseline (delta={mean_acc - 0.682:.4f})")

    return results


def run(context: dict) -> dict:
    """Run all verification checks."""
    evans_df = context.get("evans_df", pd.DataFrame())
    feature_cols = context.get("all_feature_cols", [])
    splits = context.get("splits", {})
    verify_dir = context["output_dir"] / "verification"
    verify_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Step 16: Verification suite")

    # 1. Quality checks
    logger.info("  Running quality checks...")
    checks = _quality_checks(evans_df, feature_cols)
    all_pass = all(c["pass"] for c in checks.values())
    for name, result in checks.items():
        status = "✓" if result["pass"] else "✗"
        logger.info(f"    {status} {name}: {result['detail']}")

    with open(verify_dir / "quality_checks.json", "w") as f:
        json.dump(checks, f, indent=2, default=str)

    # 2. Chemical sample
    logger.info("  Generating chemical sample for manual inspection...")
    sample = _chemical_sample(evans_df)
    sample.to_csv(verify_dir / "chemical_sample.csv", index=False)
    logger.info(f"    Saved {len(sample)} rows to chemical_sample.csv")

    # 3. XGBoost comparison
    logger.info("  Training XGBoost for V2 comparison...")
    xgb_results = _train_xgboost_tscv(evans_df, feature_cols, splits)
    with open(verify_dir / "v2_comparison.json", "w") as f:
        json.dump(xgb_results, f, indent=2, default=str)

    # Overall summary
    logger.info(f"  Step 16 complete: all checks {'PASSED' if all_pass else 'SOME FAILED'}")
    context["verification"] = {
        "quality_checks": checks,
        "xgb_results": xgb_results,
        "all_pass": all_pass,
    }
    return context
