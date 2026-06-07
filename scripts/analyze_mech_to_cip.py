#!/usr/bin/env python
"""A0 — Mechanism->CIP inversion analysis (make-or-break gate).

Tests the central hypothesis of the mechanism-frame reframing:
    label_joint (4-class CIP) == f(auxiliary, aux_chirality, dominant Z/E, syn/anti_3d)

If the mapping is (near-)deterministic (high weighted purity), then the CIP
4-class label can be losslessly recovered from mechanism-frame variables, so
predicting in the mechanism frame and post-processing to CIP is valid.

Read-only: loads CSVs, prints summary, writes results/mech_mapping/*.csv.
No model training, no mutation of project data.

Gate (judged on Evans, the current target subset):
    PASS         weighted_purity >= 0.90 AND big-group(n>=10) coverage>=80% with purity>=0.85
    CONDITIONAL  0.80 <= weighted_purity < 0.90  -> Evans high-confidence subset only
    FAIL         < 0.80  -> abandon reframing
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chiralaldol.ze_enolate_generator import get_ze_weights  # noqa: E402
from chiralaldol.config import VALID_AUXILIARIES  # noqa: E402

CLEAN = ROOT / "data" / "clean_v5" / "substrate_aldol_clean.csv"
OUT = ROOT / "results" / "mech_mapping"
LABEL = "label_joint"


def aux_chirality(ketone_smi: str) -> str:
    """Full auxiliary stereo signature from ketone SMILES.

    Returns sorted CIP codes joined by '|', e.g. 'S', 'R', 'R|R', or 'none'/'na'.
    The ketone (acyl auxiliary, pre-aldol) carries only the auxiliary's
    stereocenters, so this faithfully captures auxiliary absolute config.
    """
    if not isinstance(ketone_smi, str) or not ketone_smi.strip():
        return "na"
    mol = Chem.MolFromSmiles(ketone_smi)
    if mol is None:
        return "na"
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    try:
        centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
    except Exception:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    cips = sorted(cip for _, cip in centers if cip in ("R", "S"))
    return "|".join(cips) if cips else "none"


def dominant_ze(base, activator) -> str:
    wz, _ = get_ze_weights(str(base), str(activator))
    return "Z" if wz >= 0.5 else "E"


def purity_table(df: pd.DataFrame, group_cols: list[str], label_col: str = LABEL):
    """Per-group mode/purity/entropy of label_col, plus n-weighted purity."""
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        vc = g[label_col].value_counts()
        p = vc.values / len(g)
        ent = float(-(p * np.log2(p)).sum())
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n": len(g),
                "mode_label": int(vc.index[0]),
                "purity": float(vc.iloc[0]) / len(g),
                "entropy": ent,
            }
        )
    t = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    wp = float(np.average(t["purity"], weights=t["n"]))
    return t, wp


def big_group_coverage(t: pd.DataFrame, n_min: int = 10, pur_min: int = 0.85):
    """Fraction of samples in big groups, and of those, fraction meeting purity."""
    total = t["n"].sum()
    big = t[t["n"] >= n_min]
    big_cov = big["n"].sum() / total if total else 0.0
    good = big[big["purity"] >= pur_min]
    good_cov = good["n"].sum() / total if total else 0.0
    return big_cov, good_cov


def report(df: pd.DataFrame, tag: str):
    print("\n" + "=" * 70)
    print(f"  MECHANISM->CIP INVERSION  [{tag}]   n={len(df)}")
    print("=" * 70)

    valid = df[df["label_syn_anti_3d"].notna()].copy()
    n_drop = len(df) - len(valid)
    print(f"  rows with valid syn/anti_3d: {len(valid)}  (dropped {n_drop} NaN)")

    # --- baseline: knowing nothing, predict global majority ---
    gmaj = df[LABEL].value_counts(normalize=True).iloc[0]
    print(f"\n  [baseline] global-majority purity (know nothing) : {gmaj:.4f}")

    # --- ablation A: conditions only, NO syn/anti ---
    tA, wpA = purity_table(valid, ["auxiliary_type", "aux_chir", "ze"])
    print(f"  [ablation A] (aux, aux_chir, ze)            purity : {wpA:.4f}")

    # --- ablation C: use CIP heuristic label_SA instead of 3D syn/anti ---
    tC, wpC = purity_table(valid, ["auxiliary_type", "aux_chir", "ze", "label_SA"])
    print(f"  [ablation C] +label_SA (CIP heuristic)      purity : {wpC:.4f}")

    # --- MAIN: full mechanistic key with real 3D syn/anti ---
    tM, wpM = purity_table(valid, ["auxiliary_type", "aux_chir", "ze", "label_syn_anti_3d"])
    big_cov, good_cov = big_group_coverage(tM)
    print(f"  [MAIN]      +label_syn_anti_3d (real geom)  purity : {wpM:.4f}  <==")
    print(f"              jump from ablation A (syn/anti adds)    : +{wpM - wpA:.4f}")
    print(f"              big-group(n>=10) coverage={big_cov:.3f}, "
          f"of-which purity>=0.85 coverage={good_cov:.3f}")

    # --- high-confidence syn/anti subset (upper bound, excludes label noise) ---
    if "synanti_confidence" in valid.columns:
        hc = valid[valid["synanti_confidence"] >= 0.7]
        if len(hc) >= 30:
            tH, wpH = purity_table(
                hc, ["auxiliary_type", "aux_chir", "ze", "label_syn_anti_3d"]
            )
            print(f"  [MAIN|high-conf syn/anti>=0.7, n={len(hc)}] purity : {wpH:.4f}")

    # gate verdict
    if wpM >= 0.90 and good_cov >= 0.80:
        verdict = "PASS"
    elif wpM >= 0.80:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"
    print(f"\n  >>> GATE [{tag}] : {verdict}  (weighted_purity={wpM:.4f})")

    return tM, wpM, verdict


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN)
    df = df[df["auxiliary_type"].isin(VALID_AUXILIARIES)].copy()
    print(f"Loaded {len(df)} valid-auxiliary rows from {CLEAN.name}")

    print("Extracting auxiliary chirality from ketone SMILES...")
    df["aux_chir"] = df["canonical_ketone_smiles"].map(aux_chirality)
    df["ze"] = [dominant_ze(b, a) for b, a in zip(df["base_type"], df["activator_type"])]

    print("\naux_chir distribution:")
    print(df["aux_chir"].value_counts(dropna=False).to_string())
    print("\ndominant Z/E distribution:")
    print(df["ze"].value_counts(dropna=False).to_string())

    # all valid auxiliaries
    tM_all, wp_all, v_all = report(df, "ALL-AUX")
    tM_all.to_csv(OUT / "inversion_table_all.csv", index=False)

    # Evans-only (the current target)
    evans = df[df["auxiliary_type"] == "evans"].copy()
    tM_ev, wp_ev, v_ev = report(evans, "EVANS-ONLY")
    tM_ev.to_csv(OUT / "inversion_table_evans.csv", index=False)

    print("\n" + "=" * 70)
    print("  EVANS MECHANISTIC GROUPS (the inversion lookup table)")
    print("=" * 70)
    show = tM_ev[["aux_chir", "ze", "label_syn_anti_3d", "n", "mode_label", "purity"]]
    print(show.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"  DECISION (target=Evans): GATE = {v_ev}, weighted_purity = {wp_ev:.4f}")
    print(f"  (all-aux for reference : GATE = {v_all}, weighted_purity = {wp_all:.4f})")
    print("=" * 70)
    print(f"\n  Wrote: {OUT/'inversion_table_evans.csv'}")
    print(f"         {OUT/'inversion_table_all.csv'}")


if __name__ == "__main__":
    main()
