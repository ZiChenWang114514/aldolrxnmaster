"""Feature subset selection registry for AldolRxnMaster."""


# Prefix tuples for each feature group
FEATURE_SUBSETS = {
    "steric": ("Vbur_", "L_", "B1_", "B5_", "sin_tau", "cos_tau",
               "n_conformers", "n_clusters", "ald_"),
    "conditions": ("feat_",),
    "auxiliary": ("aux_", "n_defined_stereocenters"),
    "chirality": ("chiral_",),
    "rgroup": ("aux_rg_", "aux_oppolzer"),
    "chiralenv": ("chiralenv_",),
    "aldpri": ("ald_pri_",),
    "delta_chiral": ("delta_chiral_", "chiral_det_"),
}


def _match_prefixes(col_name, prefixes):
    """Check if column name starts with any of the given prefixes."""
    for p in prefixes:
        if col_name.startswith(p) or col_name == p:
            return True
    return False


def select_features(X, feat_names, include=None, exclude=None):
    """Select feature columns by subset name(s).

    Args:
        X: (n_samples, n_features) array
        feat_names: list of feature column names
        include: subset name(s) to include (str or list). If None, include all.
        exclude: subset name(s) to exclude (str or list). If None, exclude none.

    Returns:
        X_subset: filtered feature array
    """
    if include is not None:
        if isinstance(include, str):
            include = [include]
        prefixes = []
        for name in include:
            if name not in FEATURE_SUBSETS:
                raise ValueError(f"Unknown subset '{name}'. Available: {list(FEATURE_SUBSETS.keys())}")
            prefixes.extend(FEATURE_SUBSETS[name])
        idx = [i for i, c in enumerate(feat_names) if _match_prefixes(c, prefixes)]
    elif exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        prefixes = []
        for name in exclude:
            if name not in FEATURE_SUBSETS:
                raise ValueError(f"Unknown subset '{name}'. Available: {list(FEATURE_SUBSETS.keys())}")
            prefixes.extend(FEATURE_SUBSETS[name])
        idx = [i for i, c in enumerate(feat_names) if not _match_prefixes(c, prefixes)]
    else:
        return X

    return X[:, idx] if idx else X
