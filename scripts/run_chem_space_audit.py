#!/usr/bin/env python3
"""B5: Chemical Space Audit + Scaffold-Stratified Performance Analysis.

Literature motivation:
  [15] Kalikadien 2024 — DFT descriptors domain-external R²<0, must audit OOD
  [3] ScopeMap — CVT coverage assessment
  [7] Betinol 2023 — Generality evaluation framework

Produces:
  1. PCA visualization of 128d feature space (colored by auxiliary type)
  2. k-means clustering and per-cluster performance
  3. Scaffold-stratified performance breakdown
  4. Train→Test chemical distance analysis per TSCV fold

Usage:
    conda run -n aldol-rxn python scripts/run_chem_space_audit.py
"""

import logging
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from chiralaldol.config import CLEAN_DIR, FEAT_DIR, PRED_DIR, RESULTS_DIR
from chiralaldol.data_io import load_splits, prepare_Xy

OUT_DIR = RESULTS_DIR / "chem_space_audit"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chem_audit")


def load_data():
    """Load features, labels, metadata."""
    X, y, valid_mask, feat_names = prepare_Xy()
    meta = pd.read_csv(CLEAN_DIR / "substrate_aldol_clean.csv")
    splits = load_splits()
    return X, y, valid_mask, feat_names, meta, splits


def pca_analysis(X, y, valid_mask, meta):
    """PCA of feature space, colored by auxiliary type and label."""
    logger.info("\n=== PCA Analysis ===")
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    meta_valid = meta.iloc[np.where(valid_mask)[0]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    logger.info(f"  PCA explained variance (top 10): {[f'{v:.3f}' for v in explained]}")
    logger.info(f"  Cumulative (2D): {explained[:2].sum():.3f}")
    logger.info(f"  Cumulative (5D): {explained[:5].sum():.3f}")
    logger.info(f"  Cumulative (10D): {explained[:10].sum():.3f}")

    # Save PCA coordinates for visualization
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2],
        "label_joint": y_valid,
        "auxiliary_type": meta_valid["auxiliary_type"].values if "auxiliary_type" in meta_valid.columns else "unknown",
    })
    pca_df.to_csv(OUT_DIR / "pca_coordinates.csv", index=False)

    # Auxiliary type distribution in PCA space
    if "auxiliary_type" in meta_valid.columns:
        for aux in meta_valid["auxiliary_type"].unique():
            mask = meta_valid["auxiliary_type"].values == aux
            centroid = X_pca[mask, :2].mean(axis=0)
            spread = X_pca[mask, :2].std(axis=0)
            logger.info(f"  {aux:20s}: n={mask.sum():5d}, centroid=({centroid[0]:+.2f},{centroid[1]:+.2f}), spread=({spread[0]:.2f},{spread[1]:.2f})")

    # PCA loadings — which features drive each PC?
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        columns=["PC1", "PC2", "PC3"],
        index=[f for f in pd.read_csv(FEAT_DIR / "v4_features.csv").columns]
    )
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings_sorted = loadings.sort_values("abs_PC1", ascending=False)
    logger.info("\n  Top-10 features on PC1:")
    for _, row in loadings_sorted.head(10).iterrows():
        logger.info(f"    {row.name:40s}: PC1={row['PC1']:+.3f}")
    loadings.to_csv(OUT_DIR / "pca_loadings.csv")

    return X_pca, X_scaled, scaler, pca


def cluster_analysis(X, y, valid_mask, X_scaled, meta):
    """k-means clustering and per-cluster performance."""
    logger.info("\n=== k-Means Cluster Analysis ===")

    # Find optimal k via inertia elbow
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append((k, km.inertia_))

    logger.info("  Inertia by k:")
    for k, inertia in inertias:
        logger.info(f"    k={k}: inertia={inertia:.0f}")

    # Use k=5 as default (roughly matches auxiliary types)
    k_chosen = 5
    km = KMeans(n_clusters=k_chosen, random_state=42, n_init=10)
    X_valid_scaled = X_scaled[valid_mask[np.arange(len(X))][np.arange(valid_mask.sum())]]

    # Actually reindex
    valid_idx = np.where(valid_mask)[0]
    X_valid_scaled = X_scaled  # already filtered in pca_analysis... need to redo
    scaler = StandardScaler()
    X_valid_scaled = scaler.fit_transform(X[valid_mask])
    labels_km = km.fit_predict(X_valid_scaled)
    y_valid = y[valid_mask]

    # Per-cluster statistics
    meta_valid = meta.iloc[valid_idx]
    cluster_stats = []
    for c in range(k_chosen):
        mask = labels_km == c
        n = mask.sum()
        label_dist = Counter(y_valid[mask].tolist())
        aux_dist = Counter(meta_valid.iloc[np.where(mask)[0]]["auxiliary_type"].tolist()) if "auxiliary_type" in meta_valid.columns else {}
        dominant_aux = max(aux_dist, key=aux_dist.get) if aux_dist else "?"
        dominant_pct = aux_dist[dominant_aux] / n * 100 if aux_dist else 0

        cluster_stats.append({
            "cluster": c,
            "n": n,
            "pct": f"{n/len(y_valid)*100:.1f}%",
            "label_dist": dict(label_dist),
            "dominant_aux": f"{dominant_aux} ({dominant_pct:.0f}%)",
            "aux_breakdown": dict(aux_dist),
        })
        logger.info(f"  Cluster {c}: n={n} ({n/len(y_valid)*100:.1f}%), "
                     f"dom_aux={dominant_aux}({dominant_pct:.0f}%), "
                     f"labels={dict(label_dist)}")

    pd.DataFrame(cluster_stats).to_csv(OUT_DIR / "cluster_stats.csv", index=False)

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        "idx": valid_idx,
        "cluster": labels_km,
        "label_joint": y_valid,
    })
    cluster_df.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    return labels_km


def scaffold_performance(y, valid_mask, meta, splits):
    """Scaffold-stratified performance analysis using champion model predictions."""
    logger.info("\n=== Scaffold-Stratified Performance ===")

    # Load champion predictions (v4b_full_et on scaffold split)
    scaffold_pred_path = PRED_DIR / "v4b" / "v4b_full_et_scaffold.csv"
    if not scaffold_pred_path.exists():
        logger.warning(f"  Scaffold predictions not found: {scaffold_pred_path}")
        return

    pred_df = pd.read_csv(scaffold_pred_path)
    test_idx = pred_df["idx"].values
    y_true = pred_df["y_true"].values.astype(int)
    y_pred = pred_df["y_pred"].values.astype(int)

    # Get scaffolds for test samples
    if "scaffold" not in meta.columns:
        # Generate Murcko scaffolds
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            smiles_col = None
            for col in ["product_smiles", "smiles_product", "product_canonical"]:
                if col in meta.columns:
                    smiles_col = col
                    break
            if smiles_col is None:
                # Try ketone SMILES as proxy
                for col in ["ketone_smiles", "smiles_ketone"]:
                    if col in meta.columns:
                        smiles_col = col
                        break

            if smiles_col:
                logger.info(f"  Computing Murcko scaffolds from {smiles_col}...")
                scaffolds = []
                for smi in meta[smiles_col]:
                    try:
                        mol = Chem.MolFromSmiles(str(smi))
                        if mol:
                            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
                        else:
                            scaffolds.append("PARSE_FAIL")
                    except Exception:
                        scaffolds.append("ERROR")
                meta = meta.copy()
                meta["scaffold"] = scaffolds
            else:
                logger.warning("  No SMILES column found for scaffold computation")
                return
        except ImportError:
            logger.warning("  RDKit not available for scaffold computation")
            return

    # Per-scaffold performance
    test_scaffolds = meta.iloc[test_idx]["scaffold"].values
    scaffold_counts = Counter(test_scaffolds)

    # Group by scaffold, compute per-scaffold accuracy
    scaffold_results = []
    for scaffold, count in scaffold_counts.most_common():
        mask = test_scaffolds == scaffold
        if mask.sum() < 2:
            continue
        acc = balanced_accuracy_score(y_true[mask], y_pred[mask])
        scaffold_results.append({
            "scaffold": scaffold,
            "n_test": mask.sum(),
            "bal_acc": round(acc, 4),
            "correct": (y_true[mask] == y_pred[mask]).sum(),
            "label_dist": dict(Counter(y_true[mask].tolist())),
        })

    scaffold_df = pd.DataFrame(scaffold_results)
    scaffold_df.to_csv(OUT_DIR / "scaffold_performance.csv", index=False)

    # Summary statistics
    if len(scaffold_results) > 0:
        accs = [r["bal_acc"] for r in scaffold_results]
        ns = [r["n_test"] for r in scaffold_results]
        logger.info(f"  Total scaffolds in test: {len(scaffold_counts)}")
        logger.info(f"  Scaffolds with ≥2 samples: {len(scaffold_results)}")
        logger.info(f"  Per-scaffold bal_acc: mean={np.mean(accs):.3f}, "
                     f"median={np.median(accs):.3f}, min={np.min(accs):.3f}, max={np.max(accs):.3f}")

        # Top and bottom scaffolds
        scaffold_df_sorted = scaffold_df.sort_values("bal_acc")
        logger.info("\n  Worst 5 scaffolds:")
        for _, row in scaffold_df_sorted.head(5).iterrows():
            logger.info(f"    {row['scaffold'][:60]:60s} n={row['n_test']:3d} acc={row['bal_acc']:.3f}")
        logger.info("\n  Best 5 scaffolds:")
        for _, row in scaffold_df_sorted.tail(5).iterrows():
            logger.info(f"    {row['scaffold'][:60]:60s} n={row['n_test']:3d} acc={row['bal_acc']:.3f}")


def tscv_distance_analysis(X, y, valid_mask, splits):
    """Analyze chemical distance between train and test sets per TSCV fold."""
    logger.info("\n=== TSCV Train→Test Chemical Distance ===")
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X)

    tscv_results = []
    for split_name in sorted(splits):
        if "tscv" not in split_name:
            continue
        split = splits[split_name]
        tr = np.array(split["train"], dtype=int)
        tr = tr[valid_mask[tr]]
        te = np.array(split["test"], dtype=int)
        te = te[valid_mask[te]]

        # Compute mean pairwise distance (centroid distance as proxy)
        centroid_tr = X_all_scaled[tr].mean(axis=0)
        centroid_te = X_all_scaled[te].mean(axis=0)
        dist = np.linalg.norm(centroid_tr - centroid_te)

        # Mahalanobis-like: how many test points are within the training convex hull?
        # Simplified: compute mean distance of each test point to nearest training point
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X_all_scaled[tr])
        dists, _ = nn.kneighbors(X_all_scaled[te])
        mean_nn_dist = dists.mean()
        median_nn_dist = np.median(dists.mean(axis=1))

        # Load prediction for this fold
        pred_path = PRED_DIR / "v4b" / f"v4b_full_et_{split_name}.csv"
        bal_acc = None
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            bal_acc = balanced_accuracy_score(pred["y_true"], pred["y_pred"])

        tscv_results.append({
            "fold": split_name,
            "n_train": len(tr),
            "n_test": len(te),
            "centroid_dist": round(dist, 4),
            "mean_5nn_dist": round(mean_nn_dist, 4),
            "median_5nn_dist": round(median_nn_dist, 4),
            "bal_acc": round(bal_acc, 4) if bal_acc is not None else None,
        })
        logger.info(f"  {split_name}: centroid_dist={dist:.3f}, 5nn_dist={mean_nn_dist:.3f}, acc={bal_acc:.3f}" if bal_acc else f"  {split_name}: centroid_dist={dist:.3f}, 5nn_dist={mean_nn_dist:.3f}")

    tscv_df = pd.DataFrame(tscv_results)
    tscv_df.to_csv(OUT_DIR / "tscv_distance_analysis.csv", index=False)

    # Correlation: distance vs performance
    if all(r["bal_acc"] is not None for r in tscv_results):
        dists = [r["mean_5nn_dist"] for r in tscv_results]
        accs = [r["bal_acc"] for r in tscv_results]
        corr = np.corrcoef(dists, accs)[0, 1]
        logger.info(f"\n  Correlation (5nn_dist vs bal_acc): r = {corr:.3f}")
        logger.info(f"  {'Higher distance → lower accuracy' if corr < 0 else 'No clear distance-accuracy trend'}")


def auxiliary_performance(y, valid_mask, meta):
    """Per-auxiliary-type performance using champion predictions across all splits."""
    logger.info("\n=== Per-Auxiliary Performance ===")

    if "auxiliary_type" not in meta.columns:
        logger.warning("  No auxiliary_type column")
        return

    # Load all TSCV predictions for champion model
    all_preds = []
    for i in range(1, 5):
        pred_path = PRED_DIR / "v4b" / f"v4b_full_et_tscv_fold{i}.csv"
        if pred_path.exists():
            all_preds.append(pd.read_csv(pred_path))

    if not all_preds:
        logger.warning("  No TSCV predictions found")
        return

    pred_all = pd.concat(all_preds, ignore_index=True)
    test_idx = pred_all["idx"].values
    test_aux = meta.iloc[test_idx]["auxiliary_type"].values

    results = []
    for aux in sorted(set(test_aux)):
        mask = test_aux == aux
        if mask.sum() < 5:
            continue
        y_true = pred_all.iloc[np.where(mask)[0]]["y_true"].values.astype(int)
        y_pred = pred_all.iloc[np.where(mask)[0]]["y_pred"].values.astype(int)
        acc = balanced_accuracy_score(y_true, y_pred)
        simple_acc = (y_true == y_pred).mean()
        results.append({
            "auxiliary": aux,
            "n_test": mask.sum(),
            "bal_acc": round(acc, 4),
            "simple_acc": round(simple_acc, 4),
            "label_dist": dict(Counter(y_true.tolist())),
        })
        logger.info(f"  {aux:20s}: n={mask.sum():5d}, bal_acc={acc:.3f}, simple_acc={simple_acc:.3f}")

    pd.DataFrame(results).to_csv(OUT_DIR / "auxiliary_performance.csv", index=False)


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Chemical Space Audit — V4d")
    logger.info("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, valid_mask, feat_names, meta, splits = load_data()
    logger.info(f"Data: {X.shape[0]} rows × {X.shape[1]}d, valid: {valid_mask.sum()}")

    # 1. PCA
    X_pca, X_scaled, scaler, pca = pca_analysis(X, y, valid_mask, meta)

    # 2. Clustering
    labels_km = cluster_analysis(X, y, valid_mask, X_scaled, meta)

    # 3. Scaffold performance
    scaffold_performance(y, valid_mask, meta, splits)

    # 4. TSCV distance
    tscv_distance_analysis(X, y, valid_mask, splits)

    # 5. Per-auxiliary performance
    auxiliary_performance(y, valid_mask, meta)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Chemical Space Audit complete ({elapsed:.1f}s)")
    print(f"Results saved to: {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
