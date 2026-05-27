"""Step 08: Cross-validate CIP labels with optical yield data, assign final labels."""

import logging

import pandas as pd

from .audit import AuditTracker

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

    # --- Drop rows without usable labels ---
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
