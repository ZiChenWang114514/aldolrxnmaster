"""Phase C1: Quasi-Transition-State (qTS) Builder for Evans Aldol.

Constructs Zimmerman-Traxler 6-membered ring TS scaffolds and runs
GFN2-xTB constrained geometry optimizations to extract pseudo-activation
barriers for all 4 competing face channels.

Physical motivation:
    Evans aldol proceeds through a chair-like Zimmerman-Traxler TS.
    The facial selectivity is determined by which face of the enolate attacks
    which face of the aldehyde in the lowest-energy TS geometry.

    The current #1 feature (sin_tau1) approximates the TS dihedral from the
    ground-state enolate geometry — an indirect proxy. qTS replaces this with
    direct ΔE_qTS computed by constrained xTB optimization.

4 competing channels:
    1. si_face / chair-like TS  (typically lowest E for Evans syn)
    2. si_face / twist-boat TS
    3. re_face / chair-like TS
    4. re_face / twist-boat TS

Output features (4d per reaction, after normalization):
    qts_dE_si_re_chair  = E(si_chair) - E(re_chair)   [kcal/mol]
    qts_dE_si_re_twist  = E(si_twist) - E(re_twist)   [kcal/mol]
    qts_E_si_chair_kcal = absolute qTS energy (normalized)
    qts_boltzmann_si    = Boltzmann weight of si_chair at 298K

Expected gain: +5-10% temporal bal_acc (current V2: 0.783 → target 0.85+)
Scale: 1822 reactions × 4 channels = 7308 xTB constrained optimizations
Estimated runtime: ~12-24h on 8 CPU cores

References:
    Zimmerman, H. E.; Traxler, M. D. J. Am. Chem. Soc. 1957, 79, 1920.
    Legault, C. Y. CYLview20 (2020): cylview.org
    Grimme, S. et al. J. Chem. Theory Comput. 2019, 15, 1652. (GFN2-xTB)

Usage:
    python scripts/run_qts_pipeline.py
    (or via run_chiralaldol_pipeline.py stage4_qts)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RT_KCAL = 0.5922  # RT at 298 K in kcal/mol
HARTREE_TO_KCAL = 627.509  # Ha → kcal/mol

# Zimmerman-Traxler TS geometry constraints
ZT_CC_BOND_ANG = 2.1      # Å — forming C-C bond (enolate Cα to aldehyde C)
ZT_METAL_O_ANG = 1.9      # Å — metal-O (enolate oxygen)
ZT_METAL_O2_ANG = 2.2     # Å — metal-O (aldehyde oxygen)

QTS_FEATURE_NAMES = [
    "qts_dE_si_re_chair",    # ΔE(si_chair - re_chair) kcal/mol; negative = si favored
    "qts_dE_si_re_twist",    # ΔE(si_twist - re_twist) kcal/mol
    "qts_min_E_kcal",        # lowest of 4 channel energies (normalized to mean)
    "qts_boltzmann_si",      # Boltzmann weight of si_chair at 298K
]


# ── TODO: Implementation (Phase C1) ─────────────────────────────────────────
#
# The following functions need to be implemented:
#
# 1. build_zt_scaffold(enolate_smi, aldehyde_smi, metal, face, conformation)
#    → RDKit Mol with approximate ZT 6-ring geometry
#    Strategy:
#      a. Generate 3D conformers for enolate + aldehyde separately
#      b. Align forming bond: place aldehyde C at ZT_CC_BOND_ANG from enolate Cα
#      c. Close the 6-membered ring: metal bridges both oxygens
#      d. Chair/twist-boat: adjust ring pucker angle (~60°/~30°)
#      e. For si vs re: flip aldehyde orientation (R-group axial vs equatorial)
#
# 2. run_xtb_constrained_opt(mol, constraints)
#    → optimized energy (Ha) and geometry
#    Uses tblite.interface.Calculator with:
#      - method = "GFN2-xTB"
#      - constraints: fix forming CC bond distance, fix metal-O distances
#      - max_iterations = 500
#      - electronic_temperature = 300 K
#
# 3. compute_qts_features(enolate_smi, aldehyde_smi, metal, conf_ensemble)
#    → dict with QTS_FEATURE_NAMES
#    a. Build 4 scaffolds (si/re × chair/twist)
#    b. Run constrained xTB opt on each
#    c. Extract energies, compute ΔE and Boltzmann weights
#    d. Return NaN dict if any channel fails
#
# 4. compute_qts_features_batch(enolates_df, aldehydes_smi, metals, n_workers=8)
#    → DataFrame (1822 × 4)
#    Parallel computation with checkpointing (same pattern as xtb_descriptors.py)
#
# ────────────────────────────────────────────────────────────────────────────


def qts_not_implemented(*args, **kwargs):
    raise NotImplementedError(
        "Phase C1 qTS modeling is not yet implemented. "
        "See chiralaldol/qts_builder.py for the implementation plan."
    )


# Placeholder exports
build_zt_scaffold = qts_not_implemented
run_xtb_constrained_opt = qts_not_implemented
compute_qts_features = qts_not_implemented
compute_qts_features_batch = qts_not_implemented
