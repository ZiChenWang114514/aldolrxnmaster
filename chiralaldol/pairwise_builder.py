"""PairwiseBuilder: Construct pairwise training data for StereoRank.

For each reaction with 4 candidates, generates pairs (true_product, false_product)
with difference encoding: features = feat_true - feat_false.

Training signal: binary label (1 = first candidate is better, 0 = second is better).
"""

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_ze_weights(row):
    """Get Z/E enolate weights based on base/activator.

    Bu2BOTf/Chx2BCl/Ipc2BCl/9BBN_OTf → 98% Z
    LDA/LiHMDS/NaHMDS/KHMDS → 95% Z
    Et3N → 70% Z
    Other → 50% Z
    """
    # Check activator (boron reagents are most Z-selective)
    if row.get("act_Bu2BOTf", 0) or row.get("act_Chx2BCl", 0) or \
       row.get("act_Ipc2BCl", 0) or row.get("act_9BBN_OTf", 0):
        return 0.98, 0.02
    # Check base
    if row.get("base_LDA", 0) or row.get("base_LiHMDS", 0) or \
       row.get("base_NaHMDS", 0) or row.get("base_KHMDS", 0):
        return 0.95, 0.05
    if row.get("base_Et3N", 0):
        return 0.70, 0.30
    return 0.50, 0.50


def build_candidate_features(candidates_df, reaction_features_df, feature_cols):
    """Build candidate-specific features using mechanism-weighted steric descriptors.

    Key innovation: Each candidate corresponds to a specific Zimmerman-Traxler pathway.
    We create candidate-specific features by weighting steric descriptors by the
    mechanistic pathway probability:

    - Candidate (R,R) → Z-si: effective_steric = w_Z × Vbur_si features
    - Candidate (R,S) → Z-re: effective_steric = w_Z × Vbur_re features
    - Candidate (S,R) → E-si: effective_steric = w_E × Vbur_si features
    - Candidate (S,S) → E-re: effective_steric = w_E × Vbur_re features

    This creates genuinely different feature vectors for each candidate based on
    which TS geometry it would require.

    Args:
        candidates_df: Output of enumerate_dataset()
        reaction_features_df: DataFrame with steric/condition features
        feature_cols: Feature column names

    Returns:
        DataFrame with candidate-specific features
    """
    # Identify steric feature subgroups
    si_face_cols = [c for c in feature_cols if "Vbur_si" in c]
    re_face_cols = [c for c in feature_cols if "Vbur_re" in c]
    diff_cols = [c for c in feature_cols if "Vbur_diff" in c]
    non_face_steric = [c for c in feature_cols if c.startswith(("L_", "B1_", "B5_",
                       "sin_tau", "cos_tau")) and c in feature_cols]
    other_cols = [c for c in feature_cols if c not in si_face_cols + re_face_cols +
                  diff_cols + non_face_steric]

    rows = []
    rxn_cache = {}

    for _, cand_row in candidates_df.iterrows():
        rxn_id = cand_row["reaction_id"]
        cand_id = cand_row["candidate_id"]
        ca_config = cand_row["ca_config"]
        cb_config = cand_row["cb_config"]

        # Get reaction features (cached)
        if rxn_id not in rxn_cache:
            if rxn_id in reaction_features_df.index:
                rxn_cache[rxn_id] = reaction_features_df.loc[rxn_id]
            else:
                continue
        rxn_row = rxn_cache[rxn_id]

        # Determine Z/E weights
        w_z, w_e = _get_ze_weights(rxn_row)

        # Determine candidate's ZT pathway
        ca_R = (ca_config == "R")
        cb_R = (cb_config == "R")
        is_syn = (ca_R == cb_R)

        # Mechanistic assignment:
        # RR → Z-enolate, si-face attack (syn, favored face)
        # RS → Z-enolate, re-face attack (anti)
        # SR → E-enolate, si-face attack (anti)
        # SS → E-enolate, re-face attack (syn)
        if ca_R and cb_R:      # RR → Z-si
            w_enolate = w_z
            face = "si"
        elif ca_R and not cb_R:  # RS → Z-re
            w_enolate = w_z
            face = "re"
        elif not ca_R and cb_R:  # SR → E-si
            w_enolate = w_e
            face = "si"
        else:                    # SS → E-re
            w_enolate = w_e
            face = "re"

        feat_dict = {
            "reaction_id": rxn_id,
            "candidate_id": cand_id,
            "is_true_product": cand_row["is_true_product"],
            # Candidate identity features
            "cand_ca_R": float(ca_R),
            "cand_cb_R": float(cb_R),
            "cand_syn": float(is_syn),
            # Pathway probability (key differentiator!)
            "cand_w_enolate": w_enolate,
            # Mechanism-weighted face selectivity features
        }

        # Face-specific steric: select si or re based on candidate's pathway
        if face == "si":
            for col in si_face_cols:
                feat_dict[f"cand_face_{col}"] = float(rxn_row[col]) * w_enolate
            for col in re_face_cols:
                feat_dict[f"cand_opp_face_{col}"] = float(rxn_row[col]) * w_enolate
        else:
            for col in re_face_cols:
                feat_dict[f"cand_face_{col}"] = float(rxn_row[col]) * w_enolate
            for col in si_face_cols:
                feat_dict[f"cand_opp_face_{col}"] = float(rxn_row[col]) * w_enolate

        # Vbur difference (sign flips for re-face attack)
        for col in diff_cols:
            val = float(rxn_row[col])
            feat_dict[f"cand_{col}"] = val if face == "si" else -val
            feat_dict[f"cand_weighted_{col}"] = (val if face == "si" else -val) * w_enolate

        # Non-face steric (shared but weighted by pathway probability)
        for col in non_face_steric:
            feat_dict[f"cand_w_{col}"] = float(rxn_row[col]) * w_enolate

        # Other features (conditions, auxiliary — same for all candidates)
        for col in other_cols:
            feat_dict[col] = float(rxn_row[col])

        rows.append(feat_dict)

    result_df = pd.DataFrame(rows)
    return result_df


def build_pairwise_data(candidate_features_df, feature_cols):
    """Build pairwise difference-encoded training data.

    For each reaction:
    - True product paired with each false product → 3 pairs
    - Difference encoding: feat_true - feat_false
    - Label = 1 (true > false)
    - Also generate reverse: feat_false - feat_true, label = 0

    Args:
        candidate_features_df: Output of build_candidate_features()
        feature_cols: List of feature columns used for difference encoding

    Returns:
        DataFrame with columns: reaction_id, pair_id, label, diff_feat_1, diff_feat_2, ...
    """
    all_feat_cols = [c for c in candidate_features_df.columns
                     if c not in ("reaction_id", "candidate_id", "is_true_product")]

    rows = []
    pair_id = 0

    for rxn_id, group in candidate_features_df.groupby("reaction_id"):
        true_mask = group["is_true_product"] == 1
        false_mask = group["is_true_product"] == 0

        if true_mask.sum() != 1:
            continue

        true_row = group[true_mask].iloc[0]
        true_feats = true_row[all_feat_cols].values.astype(np.float64)

        for _, false_row in group[false_mask].iterrows():
            false_feats = false_row[all_feat_cols].values.astype(np.float64)

            # Positive pair: true - false, label=1
            diff = true_feats - false_feats
            row_pos = {"reaction_id": rxn_id, "pair_id": pair_id, "label": 1}
            for i, col in enumerate(all_feat_cols):
                row_pos[f"d_{col}"] = diff[i]
            rows.append(row_pos)
            pair_id += 1

            # Negative pair: false - true, label=0
            diff_neg = false_feats - true_feats
            row_neg = {"reaction_id": rxn_id, "pair_id": pair_id, "label": 0}
            for i, col in enumerate(all_feat_cols):
                row_neg[f"d_{col}"] = diff_neg[i]
            rows.append(row_neg)
            pair_id += 1

    pairwise_df = pd.DataFrame(rows)
    logger.info(f"Built {len(pairwise_df)} pairwise samples from "
                f"{pairwise_df['reaction_id'].nunique()} reactions")
    return pairwise_df


def build_ranking_groups(candidate_features_df, feature_cols=None):
    """Build data formatted for XGBRanker (listwise with groups).

    XGBRanker expects:
    - X: feature matrix (n_candidates, n_features)
    - y: relevance labels (1 for true product, 0 for others)
    - group: array of group sizes [4, 4, 4, ...] (each reaction = 1 group of 4)

    Args:
        candidate_features_df: Output of build_candidate_features()
        feature_cols: If None, use all feature columns

    Returns:
        dict with keys: X, y, groups, reaction_ids, candidate_ids
    """
    if feature_cols is None:
        feature_cols = [c for c in candidate_features_df.columns
                        if c not in ("reaction_id", "candidate_id", "is_true_product")]

    # Sort by reaction_id then candidate_id to ensure consistent grouping
    df = candidate_features_df.sort_values(["reaction_id", "candidate_id"]).reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float64)
    y = df["is_true_product"].values.astype(np.float64)
    reaction_ids = df["reaction_id"].values
    candidate_ids = df["candidate_id"].values

    # Group sizes (should all be 4)
    groups = df.groupby("reaction_id").size().values

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "reaction_ids": reaction_ids,
        "candidate_ids": candidate_ids,
        "feature_names": feature_cols,
    }


def split_by_reactions(ranking_data, train_reaction_ids, test_reaction_ids):
    """Split ranking data by reaction IDs (preserving group structure).

    Args:
        ranking_data: Output of build_ranking_groups()
        train_reaction_ids: Set/list of reaction IDs for training
        test_reaction_ids: Set/list of reaction IDs for testing

    Returns:
        (train_data, test_data) dicts with same structure as ranking_data
    """
    train_set = set(train_reaction_ids)
    test_set = set(test_reaction_ids)

    train_mask = np.isin(ranking_data["reaction_ids"], list(train_set))
    test_mask = np.isin(ranking_data["reaction_ids"], list(test_set))

    def _subset(data, mask):
        rxn_ids = data["reaction_ids"][mask]
        unique_rxns = np.unique(rxn_ids)
        # Recompute groups
        groups = []
        for rxn in unique_rxns:
            groups.append(int(np.sum(rxn_ids == rxn)))

        return {
            "X": data["X"][mask],
            "y": data["y"][mask],
            "groups": np.array(groups),
            "reaction_ids": rxn_ids,
            "candidate_ids": data["candidate_ids"][mask],
            "feature_names": data["feature_names"],
        }

    return _subset(ranking_data, train_mask), _subset(ranking_data, test_mask)
