"""Compute all features for the Evans clean dataset.

Produces:
  - Morgan fingerprints (2048-bit, r=2) for ketone, aldehyde, product, and reaction diff
  - RDKit 2D descriptors for ketone, aldehyde, product
  - Reaction condition features (metal one-hot, solvent Kamlet-Taft)
  - Combined tabular feature matrix for GBDT/RF/MLP models
  - Reaction SMILES strings for Transformer models
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from scipy import sparse

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# RDKit 2D descriptors to compute
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "FractionCSP3", "NumHeavyAtoms", "RingCount",
    "BertzCT", "Chi0v", "Chi1v", "Chi2v", "HallKierAlpha",
    "LabuteASA", "BalabanJ",
]

DESCRIPTOR_FUNCS = {name: getattr(Descriptors, name) for name in DESCRIPTOR_NAMES if hasattr(Descriptors, name)}


def compute_morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint as bit vector."""
    if pd.isna(smiles) or not str(smiles).strip():
        return np.zeros(nbits, dtype=np.int8)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(nbits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_descriptors(smiles: str) -> dict:
    """Compute RDKit 2D descriptors."""
    if pd.isna(smiles) or not str(smiles).strip():
        return {name: np.nan for name in DESCRIPTOR_FUNCS}
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {name: np.nan for name in DESCRIPTOR_FUNCS}
    result = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            result[name] = func(mol)
        except Exception:
            result[name] = np.nan
    return result


def compute_metal_onehot(df: pd.DataFrame) -> np.ndarray:
    """Compute metal one-hot encoding."""
    metals = sorted(df["metal_clean"].unique())
    metal_to_idx = {m: i for i, m in enumerate(metals)}
    n = len(df)
    d = len(metals)
    onehot = np.zeros((n, d), dtype=np.float32)
    for i, m in enumerate(df["metal_clean"]):
        if m in metal_to_idx:
            onehot[i, metal_to_idx[m]] = 1.0
    return onehot, [f"metal_{m}" for m in metals]


def _parse_reagent_list(s) -> list[str]:
    """Parse a Python list string repr to list of lowercase reagent names."""
    import ast
    if pd.isna(s) or not str(s).strip():
        return []
    s = str(s).strip()
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if x]
        return [str(result).strip().lower()]
    except (ValueError, SyntaxError):
        return [s.lower()]


# Reagent → role classification for Evans aldol reactions
# Bases: control enolate geometry (Z vs E), critical for syn/anti selectivity
BASE_MAP = {
    "n-ethyl-n,n-diisopropylamine": "DIPEA",
    "diisopropylethylamine": "DIPEA",
    "hunig's base": "DIPEA",
    "triethylamine": "Et3N",
    "lithium hexamethyldisilazane": "LiHMDS",
    "lithium bis(trimethylsilyl)amide": "LiHMDS",
    "sodium hexamethyldisilazane": "NaHMDS",
    "sodium bis(trimethylsilyl)amide": "NaHMDS",
    "lithium diisopropylamide": "LDA",
    "lda": "LDA",
    "potassium hexamethyldisilazane": "KHMDS",
    "potassium bis(trimethylsilyl)amide": "KHMDS",
    "n,n-diisopropylethylamine": "DIPEA",
    "2,6-lutidine": "other_base",
    "pyridine": "other_base",
    "imidazole": "other_base",
    "4-dimethylaminopyridine": "other_base",
    "dmap": "other_base",
    "1,8-diazabicyclo[5.4.0]undec-7-ene": "other_base",
    "dbu": "other_base",
    "sparteine": "other_base",
}

# Activators/Lewis acids (some overlap with metal column but not all)
ACTIVATOR_MAP = {
    "di-n-butylboryl trifluoromethanesulfonate": "Bu2BOTf",
    "dibutylboron triflate": "Bu2BOTf",
    "dicyclohexylboron chloride": "Chx2BCl",
    "diisopinocampheylboron chloride": "Ipc2BCl",
    "9-borabicyclo(3.3.1)nonyl trifluoromethanesulfonate": "9BBN_OTf",
    "titanium tetrachloride": "TiCl4",
    "titanium(iv) chloride": "TiCl4",
    "tin(ii) trifluoromethanesulfonate": "Sn_OTf2",
    "tin(ii) triflate": "Sn_OTf2",
    "magnesium chloride": "MgCl2",
    "magnesium bromide": "MgCl2",
    "zinc chloride": "ZnCl2",
    "boron trifluoride diethyl etherate": "BF3_OEt2",
    "boron trifluoride etherate": "BF3_OEt2",
}

# Other reagent roles
OXIDANT_KEYWORDS = ["peroxide", "hydrogen peroxide", "dihydrogen peroxide", "oxone",
                    "mchloroperbenzoic", "mcpba", "sodium periodate", "lithium hydroperoxide"]
SILYLATING_KEYWORDS = ["trimethylsilyl", "chloro-trimethyl-silane", "tmscl", "tms-cl",
                       "tert-butyldimethylsilyl", "tbscl"]
ADDITIVE_MAP = {
    "n,n,n,n,-tetramethylethylenediamine": "TMEDA",
    "tmeda": "TMEDA",
    "n,n'-dimethylpropyleneurea": "DMPU",
    "dmpu": "DMPU",
    "hexamethylphosphoramide": "HMPA",
    "hmpa": "HMPA",
}

# Canonical categories
BASE_CATEGORIES = ["DIPEA", "Et3N", "LiHMDS", "NaHMDS", "LDA", "KHMDS", "other_base", "no_base"]
ACTIVATOR_CATEGORIES = ["Bu2BOTf", "Chx2BCl", "Ipc2BCl", "TiCl4", "Sn_OTf2", "MgCl2", "BF3_OEt2", "other_activator", "no_activator"]


def compute_reagent_features(df: pd.DataFrame):
    """Encode reagent information by chemical role.

    Returns:
        features: (n, d) array of reagent features
        names: list of feature names
    """
    n = len(df)
    reagent_col = "Reagents"

    # Initialize arrays
    base_feats = np.zeros((n, len(BASE_CATEGORIES)), dtype=np.float32)
    activator_feats = np.zeros((n, len(ACTIVATOR_CATEGORIES)), dtype=np.float32)
    has_oxidant = np.zeros((n, 1), dtype=np.float32)
    has_silylating = np.zeros((n, 1), dtype=np.float32)
    has_additive = np.zeros((n, 1), dtype=np.float32)
    reagent_known = np.zeros((n, 1), dtype=np.float32)

    for i, row in df.iterrows():
        reagents = _parse_reagent_list(row.get(reagent_col, ""))

        if not reagents:
            # Unknown reagents: set "no_base" and "no_activator"
            base_feats[i, BASE_CATEGORIES.index("no_base")] = 1.0
            activator_feats[i, ACTIVATOR_CATEGORIES.index("no_activator")] = 1.0
            continue

        reagent_known[i] = 1.0
        found_base = False
        found_activator = False

        for r in reagents:
            # Check base
            if r in BASE_MAP:
                cat = BASE_MAP[r]
                base_feats[i, BASE_CATEGORIES.index(cat)] = 1.0
                found_base = True
            # Check activator
            if r in ACTIVATOR_MAP:
                cat = ACTIVATOR_MAP[r]
                activator_feats[i, ACTIVATOR_CATEGORIES.index(cat)] = 1.0
                found_activator = True
            # Check oxidant
            if any(kw in r for kw in OXIDANT_KEYWORDS):
                has_oxidant[i] = 1.0
            # Check silylating
            if any(kw in r for kw in SILYLATING_KEYWORDS):
                has_silylating[i] = 1.0
            # Check additive
            if r in ADDITIVE_MAP:
                has_additive[i] = 1.0

        if not found_base:
            base_feats[i, BASE_CATEGORIES.index("no_base")] = 1.0
        if not found_activator:
            activator_feats[i, ACTIVATOR_CATEGORIES.index("no_activator")] = 1.0

    features = np.hstack([base_feats, activator_feats, has_oxidant, has_silylating, has_additive, reagent_known])
    names = ([f"base_{c}" for c in BASE_CATEGORIES] +
             [f"activator_{c}" for c in ACTIVATOR_CATEGORIES] +
             ["has_oxidant", "has_silylating", "has_additive", "reagent_known"])

    logger.info(f"  Reagent features: {features.shape} ({len(names)} dims)")
    # Log base distribution
    for j, cat in enumerate(BASE_CATEGORIES):
        logger.info(f"    base_{cat}: {base_feats[:, j].sum():.0f}")

    return features, names


def compute_auxiliary_chirality(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Evans auxiliary chirality features from Ketone SMILES.

    The Evans oxazolidinone auxiliary has a chiral center whose R/S
    configuration directly controls the facial selectivity of the enolate.
    This is the single most important structural variable for predicting
    stereochemical outcome.

    Returns a DataFrame with 6 columns:
      aux_config_R:        1 if primary chiral center is R, 0 if S, -1 if unknown
      aux_n_stereocenters: number of defined stereocenters on auxiliary
      aux_has_benzyl:      1 if auxiliary has benzyl substituent
      aux_has_isopropyl:   1 if auxiliary has isopropyl substituent
      aux_has_phenyl:      1 if auxiliary has phenyl (directly on ring C)
      aux_mw:              molecular weight of auxiliary
    """
    from rdkit.Chem import Descriptors as Desc

    # SMARTS patterns for R-group detection
    pat_benzyl = Chem.MolFromSmarts("[CH2]c1ccccc1")
    pat_isopropyl = Chem.MolFromSmarts("[CH](C)C")
    pat_phenyl_on_ring = Chem.MolFromSmarts("[CH]c1ccccc1")

    n = len(df)
    config_R = np.full(n, -1.0, dtype=np.float32)
    n_stereo = np.zeros(n, dtype=np.float32)
    has_bn = np.zeros(n, dtype=np.float32)
    has_ipr = np.zeros(n, dtype=np.float32)
    has_ph = np.zeros(n, dtype=np.float32)
    mw = np.zeros(n, dtype=np.float32)

    ketone_col = "Ketone"

    for i, smi in enumerate(df[ketone_col]):
        if pd.isna(smi) or not str(smi).strip():
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        # Stereocenter analysis
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            defined = [(idx, cip) for idx, cip in centers if cip in ("R", "S")]

            n_stereo[i] = len(defined)

            if defined:
                # Primary chiral center = first defined one (on the oxazolidinone ring)
                primary_cip = defined[0][1]
                config_R[i] = 1.0 if primary_cip == "R" else 0.0
        except Exception:
            pass

        # R-group detection
        if pat_benzyl is not None and mol.HasSubstructMatch(pat_benzyl):
            has_bn[i] = 1.0
        if pat_isopropyl is not None and mol.HasSubstructMatch(pat_isopropyl):
            has_ipr[i] = 1.0
        if pat_phenyl_on_ring is not None and mol.HasSubstructMatch(pat_phenyl_on_ring):
            has_ph[i] = 1.0

        # Molecular weight
        mw[i] = Desc.MolWt(mol)

    result = pd.DataFrame({
        "aux_config_R": config_R,
        "aux_n_stereocenters": n_stereo,
        "aux_has_benzyl": has_bn,
        "aux_has_isopropyl": has_ipr,
        "aux_has_phenyl": has_ph,
        "aux_mw": mw,
    })

    return result


def build_reaction_smiles(df: pd.DataFrame) -> list[str]:
    """Build canonical reaction SMILES: reactants>>product."""
    rxn_list = []
    for _, row in df.iterrows():
        ketone = str(row.get("Ketone", "")) if pd.notna(row.get("Ketone")) else ""
        aldehyde = str(row.get("Aldehyde", "")) if pd.notna(row.get("Aldehyde")) else ""
        product = str(row.get("Product_", "")) if pd.notna(row.get("Product_")) else ""

        if not product:
            product = str(row.get("Raw_Product_Smiles", ""))

        # Clean atom maps from SMILES for Transformer input
        def clean_smi(s):
            mol = Chem.MolFromSmiles(s) if s else None
            if mol is None:
                return s
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(mol)

        k_clean = clean_smi(ketone)
        a_clean = clean_smi(aldehyde)
        p_clean = clean_smi(product)

        if k_clean and a_clean and p_clean:
            rxn_list.append(f"{k_clean}.{a_clean}>>{p_clean}")
        elif p_clean:
            # Fallback: use raw reaction SMILES
            raw = str(row.get("Raw_Reaction_Smiles", ""))
            rxn_list.append(raw if raw else f">>{p_clean}")
        else:
            rxn_list.append("")

    return rxn_list


def run(project_root: Path = Path(".")) -> dict:
    """Compute all features for Evans clean dataset."""
    processed_dir = project_root / "data" / "processed"
    feat_dir = processed_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Load Evans clean data
    df = pd.read_csv(processed_dir / "evans_clean.csv")
    n = len(df)
    logger.info(f"Computing features for {n} Evans reactions...")

    # ---- 1. Morgan Fingerprints ----
    logger.info("Computing Morgan fingerprints...")
    fp_ketone = np.stack([compute_morgan_fp(s) for s in df.get("Ketone", [""] * n)])
    fp_aldehyde = np.stack([compute_morgan_fp(s) for s in df.get("Aldehyde", [""] * n)])

    product_col = "Product_" if "Product_" in df.columns else "Raw_Product_Smiles"
    fp_product = np.stack([compute_morgan_fp(s) for s in df[product_col]])

    # Reaction difference fingerprint
    fp_rxn_diff = fp_product.astype(np.float32) - fp_ketone.astype(np.float32) - fp_aldehyde.astype(np.float32)

    # Save
    np.savez_compressed(feat_dir / "morgan_fps.npz",
                        ketone=fp_ketone, aldehyde=fp_aldehyde,
                        product=fp_product, rxn_diff=fp_rxn_diff)
    logger.info(f"  Morgan FPs: ketone {fp_ketone.shape}, aldehyde {fp_aldehyde.shape}, product {fp_product.shape}")

    # ---- 2. RDKit Descriptors ----
    logger.info("Computing RDKit descriptors...")
    desc_records = {"ketone": [], "aldehyde": [], "product": []}
    for col_name, smi_col in [("ketone", "Ketone"), ("aldehyde", "Aldehyde"), ("product", product_col)]:
        for smi in df.get(smi_col, [""] * n):
            desc_records[col_name].append(compute_descriptors(smi))

    desc_dfs = {}
    for role, records in desc_records.items():
        role_df = pd.DataFrame(records)
        role_df.columns = [f"{role}_{c}" for c in role_df.columns]
        desc_dfs[role] = role_df

    desc_combined = pd.concat(desc_dfs.values(), axis=1)

    # Fill NaN with column median (from full dataset — ok since this is Evans-only, not train-specific)
    for col in desc_combined.columns:
        median_val = desc_combined[col].median()
        desc_combined[col] = desc_combined[col].fillna(median_val if pd.notna(median_val) else 0.0)

    desc_combined.to_csv(feat_dir / "rdkit_descriptors.csv", index=False)
    logger.info(f"  RDKit descriptors: {desc_combined.shape}")

    # ---- 3. Reaction Conditions ----
    logger.info("Computing reaction condition features...")

    # Metal one-hot
    metal_oh, metal_names = compute_metal_onehot(df)

    # Solvent Kamlet-Taft (already in df)
    solvent_cols = ["solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30"]
    solvent_feats = df[solvent_cols].values.astype(np.float32) if all(c in df.columns for c in solvent_cols) else np.zeros((n, 4), dtype=np.float32)

    # Solvent known flag
    solvent_known = df["solvent_known"].values.astype(np.float32).reshape(-1, 1) if "solvent_known" in df.columns else np.ones((n, 1), dtype=np.float32)

    # Reagent/base encoding (role-based classification)
    reagent_feats, reagent_names = compute_reagent_features(df)

    conditions = np.hstack([metal_oh, solvent_feats, solvent_known, reagent_feats])
    cond_names = metal_names + solvent_cols + ["solvent_known"] + reagent_names

    cond_df = pd.DataFrame(conditions, columns=cond_names)
    cond_df.to_csv(feat_dir / "reaction_conditions.csv", index=False)
    logger.info(f"  Reaction conditions: {conditions.shape} (metal {len(metal_names)}d + solvent 5d + reagent {len(reagent_names)}d)")

    # ---- 4. Combined Tabular Features ----
    logger.info("Building combined tabular feature matrix...")

    # Concat: Morgan FPs (product + rxn_diff) + descriptors + conditions
    # Use product FP + rxn_diff FP (not ketone/aldehyde separately — too high dim)
    fp_combined = np.hstack([fp_product.astype(np.float32), fp_rxn_diff])
    fp_names = [f"fp_prod_{i}" for i in range(2048)] + [f"fp_diff_{i}" for i in range(2048)]

    tabular = np.hstack([fp_combined, desc_combined.values.astype(np.float32), conditions])
    tabular_names = fp_names + list(desc_combined.columns) + cond_names

    # Save as npz (too large for CSV)
    np.savez_compressed(feat_dir / "tabular_features.npz",
                        X=tabular, feature_names=np.array(tabular_names))
    logger.info(f"  Combined tabular: {tabular.shape} ({len(tabular_names)} features)")

    # ---- 5. Reaction SMILES ----
    logger.info("Building reaction SMILES...")
    rxn_smiles = build_reaction_smiles(df)
    df["rxn_smiles_clean"] = rxn_smiles
    df[["Reaction_ID", "rxn_smiles_clean"]].to_csv(feat_dir / "reaction_smiles.csv", index=False)
    n_valid_rxn = sum(1 for s in rxn_smiles if s and ">>" in s)
    logger.info(f"  Valid reaction SMILES: {n_valid_rxn}/{n}")

    # ---- 6. Labels ----
    labels = df[["label_Ca", "label_Cb", "label_SA", "label_joint", "group_id", "Year", "Reaction_ID"]].copy()
    labels.to_csv(feat_dir / "labels.csv", index=False)

    # ---- 7. Auxiliary Chirality Features ----
    logger.info("Computing Evans auxiliary chirality features...")
    aux_feats = compute_auxiliary_chirality(df)
    aux_feats.to_csv(feat_dir / "auxchiral_features.csv", index=False)
    logger.info(f"  Auxiliary chirality: {aux_feats.shape}")
    logger.info(f"    aux_config_R sum: {aux_feats['aux_config_R'].sum():.0f}/{n}")

    logger.info("Feature computation complete!")
    return {
        "n_samples": n,
        "tabular_shape": tabular.shape,
        "n_valid_rxn_smiles": n_valid_rxn,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(Path("/data2/zcwang/aldolrxnmaster"))
