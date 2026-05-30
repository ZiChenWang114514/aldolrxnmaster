"""Step 08: Cross-validate CIP labels with optical yield data, assign final labels."""

import logging

import pandas as pd
from rdkit import Chem

from .audit import AuditTracker
from .utils import safe_mol

logger = logging.getLogger("rebuild_v4.step08")


def run(df: pd.DataFrame, audit: AuditTracker) -> pd.DataFrame:
    """Cross-validate CIP and optical labels, assign final 4-class labels."""
    logger.info("Step 08: Cross-validating labels and assigning final labels...")
    n_start = len(df)

    label_ca_final = []
    label_cb_final = []
    label_confidence = []
    label_source = []

    for _, row in df.iterrows():
        cip_ca = row.get("cip_Ca")
        cip_cb = row.get("cip_Cb")
        optical_syn = row.get("optical_is_major_syn")

        has_cip = pd.notna(cip_ca) and pd.notna(cip_cb)
        has_optical = pd.notna(optical_syn)

        if has_cip and has_optical:
            # Cross-validate: CIP syn/anti vs optical syn/anti
            # NOTE: This heuristic (Ca==Cb → syn) is known to be ~52% accurate.
            # It is kept here only as a rough filter for the rare case when
            # optical yield data IS available. True syn/anti is computed in step08b
            # via 3D dihedral analysis.
            cip_is_syn = (int(cip_ca) == int(cip_cb))
            if cip_is_syn == bool(optical_syn):
                label_ca_final.append(int(cip_ca))
                label_cb_final.append(int(cip_cb))
                label_confidence.append("high")
                label_source.append("cip+optical")
            else:
                # Conflict: discard
                label_ca_final.append(None)
                label_cb_final.append(None)
                label_confidence.append("conflict")
                label_source.append("conflict")

        elif has_cip:
            label_ca_final.append(int(cip_ca))
            label_cb_final.append(int(cip_cb))
            label_confidence.append("medium")
            label_source.append("cip_only")

        elif has_optical:
            # Only optical: can determine syn/anti but not absolute config
            label_ca_final.append(None)
            label_cb_final.append(None)
            label_confidence.append("low_optical_only")
            label_source.append("optical_only")

        else:
            label_ca_final.append(None)
            label_cb_final.append(None)
            label_confidence.append("none")
            label_source.append("none")

    df["label_Ca"] = label_ca_final
    df["label_Cb"] = label_cb_final
    df["label_confidence"] = label_confidence
    df["label_source"] = label_source

    # Compute derived labels where possible
    # NOTE: label_SA = int(Ca==Cb) is NOT chemical syn/anti.
    # It captures the Cb CIP priority-flip effect (aromatic vs aliphatic aldehyde).
    # True syn/anti is computed in step08b via 3D dihedral analysis.
    df["label_SA"] = df.apply(
        lambda r: int(r["label_Ca"] == r["label_Cb"])
        if pd.notna(r["label_Ca"]) and pd.notna(r["label_Cb"]) else None,
        axis=1,
    )
    df["label_joint"] = df.apply(
        lambda r: int(r["label_Ca"]) * 2 + int(r["label_Cb"])
        if pd.notna(r["label_Ca"]) and pd.notna(r["label_Cb"]) else None,
        axis=1,
    )

    # Log confidence distribution
    conf_dist = df["label_confidence"].value_counts()
    logger.info(f"  Label confidence distribution:")
    for level, count in conf_dist.items():
        logger.info(f"    {level}: {count}")

    # --- V5 label recovery: try broader SMARTS for rows without CIP ---
    # Pre-compile once: any beta-hydroxy carbonyl with 2 atom-mapped chiral centers
    _broad_pat = Chem.MolFromSmarts("[CX4:1]([OX2])([#6])[CX4:2]([#6])[CX3](=[OX1])")
    _cb_q = _ca_q = None
    if _broad_pat:
        for _qi in range(_broad_pat.GetNumAtoms()):
            _mn = _broad_pat.GetAtomWithIdx(_qi).GetAtomMapNum()
            if _mn == 1:
                _cb_q = _qi
            elif _mn == 2:
                _ca_q = _qi

    prod_col = "canonical_main_product_smiles" if "canonical_main_product_smiles" in df.columns else "main_product_smiles"
    no_label = df["label_Ca"].isna() | df["label_Cb"].isna()
    n_missing = no_label.sum()
    if n_missing > 0:
        logger.info(f"  Attempting label recovery for {n_missing} rows...")
        recovered = 0
        for idx in df[no_label].index:
            prod_smi = df.at[idx, prod_col]
            mol = safe_mol(prod_smi)
            if mol is None:
                continue
            if _broad_pat is None or _cb_q is None or _ca_q is None:
                continue
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            matches = mol.GetSubstructMatches(_broad_pat)
            if not matches:
                continue
            cb_idx = matches[0][_cb_q]
            ca_idx = matches[0][_ca_q]
            ca_cip = mol.GetAtomWithIdx(ca_idx).GetPropsAsDict().get("_CIPCode")
            cb_cip = mol.GetAtomWithIdx(cb_idx).GetPropsAsDict().get("_CIPCode")
            if ca_cip and cb_cip:
                label_ca = {"R": 0, "S": 1}.get(ca_cip)
                label_cb = {"R": 0, "S": 1}.get(cb_cip)
                if label_ca is not None and label_cb is not None:
                    df.at[idx, "label_Ca"] = label_ca
                    df.at[idx, "label_Cb"] = label_cb
                    df.at[idx, "label_confidence"] = "recovered"
                    df.at[idx, "label_source"] = "broad_smarts"
                    df.at[idx, "label_SA"] = int(label_ca == label_cb)
                    df.at[idx, "label_joint"] = label_ca * 2 + label_cb
                    recovered += 1
        logger.info(f"  Label recovery: {recovered} / {n_missing} rows recovered")

    # --- Drop rows still without usable labels ---
    no_label = df["label_Ca"].isna() | df["label_Cb"].isna()
    audit.record_drop("08_label_validate", df.loc[no_label, "_orig_idx"],
                       "label_" + df.loc[no_label, "label_confidence"].fillna("none"))
    df = df[~no_label].reset_index(drop=True)

    # Log label distribution
    if "label_joint" in df.columns and len(df) > 0:
        joint_dist = df["label_joint"].value_counts().sort_index()
        logger.info(f"  Label distribution (4-class):")
        class_names = {0: "Ca=0,Cb=0", 1: "Ca=0,Cb=1", 2: "Ca=1,Cb=0", 3: "Ca=1,Cb=1"}
        for cls, count in joint_dist.items():
            name = class_names.get(int(cls), "?")
            logger.info(f"    Class {int(cls)} ({name}): {count}")

    audit.record_step("08_label_validate", len(df))
    logger.info(f"  Step 08 complete: {n_start} -> {len(df)} rows")
    return df
