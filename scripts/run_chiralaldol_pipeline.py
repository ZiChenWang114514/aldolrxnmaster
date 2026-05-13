#!/usr/bin/env python
"""ChiralAldol Full Pipeline — Single script from conformers to predictions.

Stages:
  1. Enolate generation (fast, already done → skip if exists)
  2. Conformer ensemble sampling (CPU-bound, with checkpointing)
  3. Steric descriptor computation
  4. Feature integration + model training + evaluation

CPU-friendly: uses limited threads, saves progress every 100 molecules,
so it can be resumed if interrupted.
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Limit numpy/BLAS threads to avoid CPU explosion
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
CHIRALALDOL_DIR = PROJECT / "data" / "processed" / "chiralaldol"
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"

# ============================================================
#  Stage 1: Enolate generation
# ============================================================

def stage1_enolates():
    """Generate enolates (skip if already done)."""
    out_path = CHIRALALDOL_DIR / "enolates.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        logger.info(f"Stage 1 SKIP: enolates.csv exists ({len(df)} rows)")
        return df

    from chiralaldol.enolate_generator import generate_all_enolates
    df = generate_all_enolates(PROJECT)
    df.to_csv(out_path, index=False)
    logger.info(f"Stage 1 DONE: {len(df)} enolates saved")
    return df


# ============================================================
#  Stage 2: Conformer ensemble (with checkpointing)
# ============================================================

def stage2_conformers(enolates_df, n_confs=100, n_threads=2):
    """Generate conformer ensembles with checkpoint every 100 molecules."""
    from chiralaldol.conformer_sampler import generate_conformer_ensemble

    final_path = CHIRALALDOL_DIR / "conformer_ensembles.pkl"
    ckpt_path = CHIRALALDOL_DIR / "conformer_ensembles_ckpt.pkl"

    # Load checkpoint if exists
    if final_path.exists():
        with open(final_path, "rb") as f:
            results = pickle.load(f)
        n_done = sum(1 for v in results.values() if v is not None)
        logger.info(f"Stage 2 SKIP: conformer_ensembles.pkl exists ({n_done} valid)")
        return results

    results = {}
    start_idx = 0
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            results = pickle.load(f)
        start_idx = max(results.keys()) + 1 if results else 0
        logger.info(f"Stage 2 RESUME from checkpoint: {start_idx}/{len(enolates_df)} done")

    n = len(enolates_df)
    t0 = time.time()
    n_ok = sum(1 for v in results.values() if v is not None)

    for i in range(start_idx, n):
        smi = str(enolates_df["enolate_smiles"].iloc[i])
        ens = generate_conformer_ensemble(smi, n_confs=n_confs, n_threads=n_threads)
        results[i] = ens
        if ens is not None:
            n_ok += 1

        # Progress + checkpoint every 100 molecules
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            done = i - start_idx + 1
            rate = done / max(elapsed, 1)
            remaining = (n - i - 1) / max(rate, 0.01)
            logger.info(f"  [{i+1}/{n}] {n_ok} ok | "
                        f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")
            # Save checkpoint
            with open(ckpt_path, "wb") as f:
                pickle.dump(results, f)

    # Save final
    with open(final_path, "wb") as f:
        pickle.dump(results, f)
    # Remove checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info(f"Stage 2 DONE: {n_ok}/{n} valid in {time.time()-t0:.0f}s")
    return results


# ============================================================
#  Stage 3: Steric descriptor computation
# ============================================================

def stage3_steric_features(enolates_df, ensembles):
    """Compute steric descriptors for all molecules."""
    from chiralaldol.steric_descriptors import (
        STERIC_DESC_NAMES,
        compute_ensemble_descriptors,
    )

    out_path = CHIRALALDOL_DIR / "steric_features.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        logger.info(f"Stage 3 SKIP: steric_features.csv exists ({df.shape})")
        return df

    n = len(enolates_df)
    rows = []
    n_ok = 0
    t0 = time.time()

    for i in range(n):
        smi = str(enolates_df["enolate_smiles"].iloc[i])
        ens = ensembles.get(i)

        if ens is None:
            rows.append({k: 0.0 for k in STERIC_DESC_NAMES})
            continue

        desc = compute_ensemble_descriptors(smi, ens)
        if desc is None:
            rows.append({k: 0.0 for k in STERIC_DESC_NAMES})
            continue

        rows.append({k: desc.get(k, 0.0) for k in STERIC_DESC_NAMES})
        n_ok += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  Steric [{i+1}/{n}] {n_ok} ok")

    df = pd.DataFrame(rows)[STERIC_DESC_NAMES]
    df.to_csv(out_path, index=False)
    logger.info(f"Stage 3 DONE: {n_ok}/{n} computed in {time.time()-t0:.0f}s")
    return df


# ============================================================
#  Stage 3b: Aldehyde steric descriptor computation
# ============================================================

def stage3b_aldehyde_features():
    """Compute 3D steric descriptors for aldehyde R-groups.

    Loads Aldehyde SMILES from evans_clean.csv, generates conformer ensembles
    for each unique aldehyde, computes Sterimol + Vbur_total, and saves
    aldehyde_steric_features.csv (1822 × 10).
    """
    from chiralaldol.conformer_sampler import generate_conformer_ensemble
    from chiralaldol.aldehyde_steric import (
        ALDEHYDE_STERIC_DESC_NAMES,
        compute_aldehyde_ensemble_descriptors,
        strip_atom_map,
    )

    out_path = CHIRALALDOL_DIR / "aldehyde_steric_features.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        logger.info(f"Stage 3b SKIP: aldehyde_steric_features.csv exists ({df.shape})")
        return df

    ckpt_path = CHIRALALDOL_DIR / "aldehyde_ensembles_ckpt.pkl"

    # Load aldehyde SMILES (mapped → strip atom map → canonical)
    evans = pd.read_csv(PROJECT / "data" / "processed" / "evans_clean.csv")
    raw_smiles = evans["Aldehyde"].fillna("").tolist()

    # Strip atom maps; record clean SMILES per row
    clean_smiles = []
    for s in raw_smiles:
        cs = strip_atom_map(s) if s else None
        clean_smiles.append(cs)

    # Deduplicate: compute conformers only for unique valid SMILES
    unique_smiles = list({s for s in clean_smiles if s is not None})
    logger.info(f"Stage 3b: {len(unique_smiles)} unique aldehydes from {len(clean_smiles)} rows")

    # Load checkpoint if exists
    ens_cache: dict[str, dict | None] = {}
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ens_cache = pickle.load(f)
        logger.info(f"  Resumed from checkpoint: {len(ens_cache)} cached")

    # Generate conformer ensembles for missing unique aldehydes
    todo = [s for s in unique_smiles if s not in ens_cache]
    logger.info(f"  Computing {len(todo)} new conformer ensembles...")
    t0 = time.time()
    for idx, smi in enumerate(todo):
        ens = generate_conformer_ensemble(smi, n_confs=100, n_threads=4)
        ens_cache[smi] = ens
        if (idx + 1) % 100 == 0 or idx == len(todo) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1)
            remaining = (len(todo) - idx - 1) / max(rate, 0.01)
            n_ok = sum(1 for v in ens_cache.values() if v is not None)
            logger.info(f"  [{idx+1}/{len(todo)}] ok={n_ok} | "
                        f"{elapsed:.0f}s | ~{remaining:.0f}s left")
            with open(ckpt_path, "wb") as f:
                pickle.dump(ens_cache, f)

    # Compute steric descriptors per row (map unique ensemble → all rows)
    zero_row = {k: 0.0 for k in ALDEHYDE_STERIC_DESC_NAMES}
    desc_cache: dict[str, dict | None] = {}
    for smi in unique_smiles:
        ens = ens_cache.get(smi)
        desc_cache[smi] = compute_aldehyde_ensemble_descriptors(smi, ens) if ens else None

    rows = []
    n_ok = 0
    for smi in clean_smiles:
        desc = desc_cache.get(smi) if smi else None
        if desc is None:
            rows.append(zero_row.copy())
        else:
            rows.append({k: desc.get(k, 0.0) for k in ALDEHYDE_STERIC_DESC_NAMES})
            n_ok += 1

    df = pd.DataFrame(rows)[ALDEHYDE_STERIC_DESC_NAMES]
    df.to_csv(out_path, index=False)

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info(f"Stage 3b DONE: {n_ok}/{len(clean_smiles)} valid | saved {out_path}")
    return df


# ============================================================
#  Stage 4: Feature integration + model training
# ============================================================

def stage4_train_and_evaluate():
    """Build features, train models, evaluate on all splits."""
    import xgboost as xgb
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

    from sklearn.decomposition import TruncatedSVD

    # Load features
    steric_df = pd.read_csv(CHIRALALDOL_DIR / "steric_features.csv")
    cond_df = pd.read_csv(FEAT_DIR / "reaction_conditions.csv")
    aux_df = pd.read_csv(FEAT_DIR / "auxchiral_features.csv")
    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    X_steric = steric_df.values.astype(np.float32)
    X_cond = cond_df.values.astype(np.float32)
    X_aux = aux_df.values.astype(np.float32)

    # Feature matrices for different models
    X_full = np.hstack([X_steric, X_cond, X_aux])
    X_steronly = X_steric
    X_condaux = np.hstack([X_cond, X_aux])

    # Load DRFP for fusion model
    drfp_data = np.load(FEAT_DIR / "drfp_fps.npz")
    X_drfp_raw = drfp_data[list(drfp_data.keys())[0]].astype(np.float32)
    # SVD reduce DRFP 2048→128d (consistent with project standard)
    svd = TruncatedSVD(n_components=128, random_state=42)
    X_drfp = svd.fit_transform(X_drfp_raw).astype(np.float32)
    logger.info(f"DRFP SVD: {X_drfp_raw.shape[1]} → {X_drfp.shape[1]} (var={svd.explained_variance_ratio_.sum():.3f})")

    # DRFP+Cond for base model B in late fusion
    X_drfp_cond = np.hstack([X_drfp, X_cond])

    np.nan_to_num(X_full, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_steronly, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_condaux, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_drfp_cond, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Features: steric={X_steric.shape[1]}, cond={X_cond.shape[1]}, "
                f"aux={X_aux.shape[1]}, full={X_full.shape[1]}, drfp_cond={X_drfp_cond.shape[1]}")

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
            cfg.update({"objective": "multi:softprob", "num_class": 4,
                        "tree_method": "hist", "random_state": 42,
                        "n_jobs": 2, "verbosity": 0,
                        "gamma": 0.1, "reg_lambda": 1.0})
            m = xgb.XGBClassifier(**cfg)
            m.fit(X_tr, y_tr, sample_weight=sw)
            acc = balanced_accuracy_score(y_val, m.predict(X_val))
            if acc > best_acc:
                best_acc, best_m = acc, m
        return best_m

    def eval_save(name, y_test, y_pred, y_prob, test_idx, split_name):
        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        ci = compute_metrics_with_ci(y_test, y_pred, n_boot=500)
        logger.info(f"  {name}: bal_acc={metrics['balanced_accuracy']:.4f}, "
                    f"MCC={metrics['mcc']:.4f}, joint={metrics['joint_accuracy']:.4f}")
        out = pd.DataFrame({"idx": test_idx, "y_true": y_test, "y_pred": y_pred})
        for c in range(4):
            out[f"prob_{c}"] = y_prob[:, c]
        pred_dir = RESULTS_DIR / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(pred_dir / f"{name}_{split_name}.csv", index=False)
        return metrics

    # Load aldehyde steric features (Stage 3b output)
    ald_path = CHIRALALDOL_DIR / "aldehyde_steric_features.csv"
    if ald_path.exists():
        ald_df = pd.read_csv(ald_path)
        X_ald = ald_df.values.astype(np.float32)
        np.nan_to_num(X_ald, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"Loaded aldehyde steric features: {X_ald.shape}")
    else:
        logger.warning("aldehyde_steric_features.csv not found — V2 models will be skipped")
        X_ald = None

    # Run on all splits
    splits = ["evans_temporal", "evans_scaffold", "evans_grouped_random_seed42"]
    for split_name in splits:
        logger.info(f"\n{'='*60}\n  ChiralAldol — {split_name}\n{'='*60}")
        with open(SPLIT_DIR / f"{split_name}.json") as f:
            sp = json.load(f)
        tr, va, te = np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])
        logger.info(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")

        # Model 1: ChiralAldol-XGB (full)
        logger.info("\n--- ChiralAldol-XGB ---")
        m = train_xgb(X_full[tr], y[tr], X_full[va], y[va])
        eval_save("chiralaldol_xgboost", y[te], m.predict(X_full[te]),
                  m.predict_proba(X_full[te]), te, split_name)

        # Model 2: SterOnly-XGB (ablation)
        logger.info("\n--- SterOnly-XGB ---")
        m = train_xgb(X_steronly[tr], y[tr], X_steronly[va], y[va])
        eval_save("chiralaldol_steronly_xgboost", y[te], m.predict(X_steronly[te]),
                  m.predict_proba(X_steronly[te]), te, split_name)

        # Model 3: CondAux-XGB (ablation: no 3D)
        logger.info("\n--- CondAux-XGB ---")
        m = train_xgb(X_condaux[tr], y[tr], X_condaux[va], y[va])
        eval_save("chiralaldol_condaux_xgboost", y[te], m.predict(X_condaux[te]),
                  m.predict_proba(X_condaux[te]), te, split_name)

        # ---- Late Fusion Models ----
        # Base Model B: DRFP+Cond XGBoost (trained fresh on this split)
        logger.info("\n--- Base: DRFP+Cond-XGB ---")
        m_drfp = train_xgb(X_drfp_cond[tr], y[tr], X_drfp_cond[va], y[va])
        val_acc_drfp = balanced_accuracy_score(y[va], m_drfp.predict(X_drfp_cond[va]))
        logger.info(f"  DRFP+Cond val_bal_acc={val_acc_drfp:.4f}")

        # Base Model A: ChiralAldol XGBoost (already trained above, retrain for val acc)
        m_chiral = train_xgb(X_full[tr], y[tr], X_full[va], y[va])
        val_acc_chiral = balanced_accuracy_score(y[va], m_chiral.predict(X_full[va]))
        logger.info(f"  ChiralAldol val_bal_acc={val_acc_chiral:.4f}")

        # Test predictions from both base models
        prob_A_test = m_chiral.predict_proba(X_full[te])
        prob_B_test = m_drfp.predict_proba(X_drfp_cond[te])

        # Model 4: Weighted Soft Voting
        logger.info("\n--- ChiralAldol-WtVote ---")
        alpha = val_acc_chiral / (val_acc_chiral + val_acc_drfp)
        prob_vote = alpha * prob_A_test + (1 - alpha) * prob_B_test
        y_vote = prob_vote.argmax(axis=1)
        eval_save("chiralaldol_weighted_vote", y[te], y_vote, prob_vote, te, split_name)
        logger.info(f"  (alpha={alpha:.3f}, ChiralAldol weight)")

        # Model 5: Stacking with OOF predictions
        logger.info("\n--- ChiralAldol-Stack ---")
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression

        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_A = np.zeros((len(tr), 4))
        oof_B = np.zeros((len(tr), 4))

        for fold_i, (fold_tr, fold_va) in enumerate(skf.split(X_full[tr], y[tr])):
            # Fold indices are relative to tr array
            fold_tr_idx = tr[fold_tr]
            fold_va_idx = tr[fold_va]

            # Train base models on this fold
            m_a = train_xgb(X_full[fold_tr_idx], y[fold_tr_idx],
                            X_full[fold_va_idx], y[fold_va_idx])
            m_b = train_xgb(X_drfp_cond[fold_tr_idx], y[fold_tr_idx],
                            X_drfp_cond[fold_va_idx], y[fold_va_idx])

            # OOF predictions for validation fold
            oof_A[fold_va] = m_a.predict_proba(X_full[fold_va_idx])
            oof_B[fold_va] = m_b.predict_proba(X_drfp_cond[fold_va_idx])

        # Level-1: meta-learner on OOF predictions
        X_meta_train = np.hstack([oof_A, oof_B])  # (n_train, 8)
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                  class_weight="balanced")
        meta.fit(X_meta_train, y[tr])

        # Test: use full-train base models
        X_meta_test = np.hstack([prob_A_test, prob_B_test])  # (n_test, 8)
        y_stack = meta.predict(X_meta_test)
        prob_stack = meta.predict_proba(X_meta_test)
        eval_save("chiralaldol_stacking", y[te], y_stack, prob_stack, te, split_name)

        # ---- V2 Models: enolate_steric(24) + ald_steric(10) + cond(35) + aux(6) = 75d ----
        if X_ald is not None:
            logger.info("\n--- ChiralAldolV2-XGB ---")
            X_full_v2 = np.hstack([X_steric, X_ald, X_cond, X_aux])  # 75d
            np.nan_to_num(X_full_v2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Model V1: ChiralAldolV2-XGB (75d single model)
            m_v2 = train_xgb(X_full_v2[tr], y[tr], X_full_v2[va], y[va])
            prob_v2_test = m_v2.predict_proba(X_full_v2[te])
            eval_save("chiralaldol_v2_xgboost", y[te], m_v2.predict(X_full_v2[te]),
                      prob_v2_test, te, split_name)

            # Model V2: ChiralAldolV2-Stack (V2-XGB + DRFP+Cond → LogReg)
            # Reuse m_drfp (DRFP+Cond base) and prob_B_test already computed above
            logger.info("\n--- ChiralAldolV2-Stack ---")
            oof_V2 = np.zeros((len(tr), 4))
            oof_B2 = np.zeros((len(tr), 4))

            for fold_i, (fold_tr, fold_va) in enumerate(skf.split(X_full_v2[tr], y[tr])):
                fold_tr_idx = tr[fold_tr]
                fold_va_idx = tr[fold_va]
                m_v2f = train_xgb(X_full_v2[fold_tr_idx], y[fold_tr_idx],
                                  X_full_v2[fold_va_idx], y[fold_va_idx])
                m_bf = train_xgb(X_drfp_cond[fold_tr_idx], y[fold_tr_idx],
                                 X_drfp_cond[fold_va_idx], y[fold_va_idx])
                oof_V2[fold_va] = m_v2f.predict_proba(X_full_v2[fold_va_idx])
                oof_B2[fold_va] = m_bf.predict_proba(X_drfp_cond[fold_va_idx])

            X_meta_train_v2 = np.hstack([oof_V2, oof_B2])  # (n_train, 8)
            meta_v2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                         class_weight="balanced")
            meta_v2.fit(X_meta_train_v2, y[tr])

            # Test: full-train V2-XGB + existing m_drfp
            prob_v2_meta_test = np.hstack([prob_v2_test, prob_B_test])
            y_stack_v2 = meta_v2.predict(prob_v2_meta_test)
            prob_stack_v2 = meta_v2.predict_proba(prob_v2_meta_test)
            eval_save("chiralaldol_v2_stacking", y[te], y_stack_v2, prob_stack_v2, te, split_name)

    logger.info("\nStage 4 DONE! All models trained and evaluated.")


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  ChiralAldol Full Pipeline")
    logger.info("=" * 60)

    t_start = time.time()

    # Stage 1
    logger.info("\n[Stage 1] Enolate generation")
    enolates = stage1_enolates()

    # Stage 2
    logger.info("\n[Stage 2] Conformer ensemble sampling (n_confs=100, 8 threads)")
    ensembles = stage2_conformers(enolates, n_confs=100, n_threads=8)

    # Stage 3
    logger.info("\n[Stage 3] Steric descriptor computation (enolate)")
    stage3_steric_features(enolates, ensembles)

    # Stage 3b
    logger.info("\n[Stage 3b] Aldehyde steric descriptor computation")
    stage3b_aldehyde_features()

    # Stage 4
    logger.info("\n[Stage 4] Model training & evaluation (including V2 models)")
    stage4_train_and_evaluate()

    logger.info(f"\nTotal pipeline time: {time.time()-t_start:.0f}s")
    logger.info("Run 'conda run -n aldol-rxn python scripts/rebuild_comparison.py' to update tables")
