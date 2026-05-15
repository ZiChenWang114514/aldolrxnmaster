"""Phase C1: Quasi-Transition-State (qTS) Builder for Evans Aldol.

STATUS (2026-05-14): NEGATIVE RESULT — V4-XGB temporal 0.628 vs V2 0.783 (-15.5%)
Root cause: approximate ZT geometry gives inconsistent si/re face assignment across
different aldehyde structures (the C=O orientation after alignment varies), so
VDW steric clash features are essentially random noise (r ≈ -0.03 with label).

GFN1/GFN2-xTB is too slow (50–120 s/molecule × 7288 = 60+ h) for the full dataset.

For a successful qTS implementation, need:
  1. Explicit 6-membered ZT chair ring coordinates (proper equatorial/axial geometry)
  2. Correct si/re defined by: R-group equatorial = si (Evans syn preferred)
  3. OR: accept slow xTB on a GPU cluster

Strategy (current): Single-point VDW steric clash on approximate ZT TS geometry
  - Build ZT 6-membered scaffold via coordinate algebra (no covalent bond editing)
  - Total charge = 0 (enolate -1 + Li⁺ +1 + aldehyde 0)
  - 4 channels: si/re × chair/twist-boat
  - VDW Gaussian overlap → relative steric energy between channels

Physical motivation:
    The Evans aldol proceeds through a Zimmerman-Traxler chair TS.
    si-face attack (syn product): aldehyde R-group equatorial → low steric strain
    re-face attack (anti product): R-group axial → higher steric strain
    Single-point xTB on correctly oriented fragments captures this steric difference.

6-membered ZT ring:  Li−O₁−C₂=Cα···C_ald−O₂  (back to Li)
    Cα = enolate alpha carbon (nucleophile)
    C₂ = enolate beta carbon (bearing O₁ and N-oxazolidinone)
    O₁ = enolate oxygen (formerly C=O)
    C_ald = aldehyde carbonyl carbon (electrophile)
    O₂ = aldehyde oxygen

Output features (4d per reaction):
    qts_dE_si_re_chair  = E(si_chair) - E(re_chair)   [kcal/mol]; negative → si favored
    qts_dE_si_re_twist  = E(si_twist) - E(re_twist)   [kcal/mol]
    qts_min_E_kcal      = min(4 channel energies) relative to mean  [kcal/mol]
    qts_boltzmann_si    = Boltzmann weight of si_chair at 298K

References:
    Zimmerman, H.E.; Traxler, M.D. JACS 1957, 79, 1920.
"""

import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RT_KCAL = 0.5922           # RT at 298 K in kcal/mol
HARTREE_TO_KCAL = 627.509  # Ha → kcal/mol
ANG_TO_BOHR = 1.0 / 0.529177

ZT_CC_DIST = 2.1    # Å — forming Cα···C_ald bond
ZT_LI_O1   = 1.90  # Å — Li-O_enolate
ZT_LI_O2   = 2.10  # Å — Li-O_aldehyde

QTS_FEATURE_NAMES = [
    "qts_dE_si_re_chair",  # ΔE(si_chair − re_chair) kcal/mol; negative = si favored
    "qts_dE_si_re_twist",  # ΔE(si_twist − re_twist) kcal/mol
    "qts_min_E_kcal",      # lowest of 4 channel energies relative to mean
    "qts_boltzmann_si",    # Boltzmann weight of si_chair at 298K
]

# SMARTS patterns (reused from steric_descriptors + xtb_descriptors)
_ENOLATE_SMARTS = Chem.MolFromSmarts("[CH,CH2,C:1]=[CX3:2](-[OX1,OX2:3])-[NX3:4]")
_ALDEHYDE_SMARTS = Chem.MolFromSmarts("[CX3;H1:1](=[OX1:2])")

# ── Geometry helpers ─────────────────────────────────────────────────────────

def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around unit axis by angle (radians)."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))


def _align_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix that rotates unit vector v_from onto v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    cross = np.cross(v_from, v_to)
    c = np.dot(v_from, v_to)
    s = np.linalg.norm(cross)
    if s < 1e-10:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[0, -cross[2], cross[1]],
                     [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))


def _smiles_to_3d(smiles: str) -> tuple:
    """Generate 3D conformer. Returns (mol_no_h, mol_h, atomic_nums, coords) or (None,)*4."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    ret = AllChem.EmbedMolecule(mol_h, params)
    if ret != 0:
        ret = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv2())
    if ret != 0:
        return None, None, None, None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
    except Exception:
        pass
    conf = mol_h.GetConformer()
    nums = [a.GetAtomicNum() for a in mol_h.GetAtoms()]
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol_h.GetNumAtoms())])
    return mol, mol_h, nums, coords


# ── Core scaffold builder ─────────────────────────────────────────────────────

def build_zt_scaffold(
    enolate_smi: str,
    aldehyde_smi: str,
    face: str = "si",
    conformation: str = "chair",
) -> tuple | None:
    """Build approximate Zimmerman-Traxler TS scaffold.

    Returns (atomic_nums, coords_angstrom) for the combined system
    [enolate_H + aldehyde_H + Li], total charge = 0.
    Returns None on failure.

    Geometry:
        - Enolate placed with Cα at origin, C₂-Cα along +x, O₁ in xy plane
        - C_ald placed at Bürgi-Dunitz approach position (2.1 Å from Cα)
          above (+z) for si face, below (−z) for re face
        - Chair: approach vector (−0.6, 0, ±1.98) → BD angle ~110° from C₂
        - Twist: approach vector (−0.3, 0.8, ±1.88) → twisted approach
        - Aldehyde C=O oriented to point toward Li coordination site
        - Li bridging O₁ and O₂ (midpoint with ZT distance correction)
    """
    # ── 1. Enolate 3D ─────────────────────────────────────────────────────
    mol_enol, mol_enol_h, enol_nums, enol_coords = _smiles_to_3d(enolate_smi)
    if mol_enol is None:
        return None

    # Find reactive center (Cα, C₂, O₁, N)
    matches = mol_enol.GetSubstructMatches(_ENOLATE_SMARTS)
    if not matches:
        return None
    ca_idx, c2_idx, o1_idx, _ = matches[0]  # heavy-atom indices

    # Map heavy-atom indices to mol_h indices (H atoms are appended after heavy atoms)
    # In AddHs, heavy atoms keep their original indices
    Ca = enol_coords[ca_idx].copy()
    C2 = enol_coords[c2_idx].copy()
    O1 = enol_coords[o1_idx].copy()

    # ── 2. Build local frame at Cα ─────────────────────────────────────────
    # x: along C₂→Cα (enolate double bond direction, pointing away from C₂)
    x_ax = Ca - C2
    x_ax /= np.linalg.norm(x_ax)

    # z: normal to enolate plane (C₂, Cα, O₁ are coplanar sp2)
    v_co = O1 - Ca
    z_ax = np.cross(x_ax, v_co)
    if np.linalg.norm(z_ax) < 1e-8:
        return None
    z_ax /= np.linalg.norm(z_ax)

    # y: completes right-hand frame
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)

    # Rotation matrix: local frame columns are [x_ax, y_ax, z_ax]
    R = np.column_stack([x_ax, y_ax, z_ax])  # (3,3): global = R @ local

    # ── 3. Translate enolate so Cα is at origin ───────────────────────────
    enol_coords_loc = enol_coords - Ca  # in global coords, Cα at origin

    # ── 4. Aldehyde 3D ────────────────────────────────────────────────────
    mol_ald, mol_ald_h, ald_nums, ald_coords = _smiles_to_3d(aldehyde_smi)
    if mol_ald is None:
        return None

    matches_ald = mol_ald.GetSubstructMatches(_ALDEHYDE_SMARTS)
    if not matches_ald:
        return None
    cho_idx, o2_idx = matches_ald[0]  # heavy-atom indices

    C_ald = ald_coords[cho_idx].copy()
    O2 = ald_coords[o2_idx].copy()

    # ── 5. Define C_ald approach position in local frame ─────────────────
    # Bürgi-Dunitz angle ~107–110° from Cα-C₂ bond (= from +x axis)
    # sign of z: +1 = si (above enolate π face), −1 = re (below)
    z_sign = 1.0 if face == "si" else -1.0

    if conformation == "chair":
        # approach vector in local coords: tilted 110° from +x, in xz plane
        # cos(110°) ≈ -0.342, sin(110°) ≈ 0.940 → scaled to 2.1 Å
        local_approach = np.array([-0.342, 0.0, z_sign * 0.940]) * ZT_CC_DIST
    else:  # twist-boat
        # slightly twisted: rotated ~20° around x axis from chair
        local_approach = np.array([-0.342, 0.321, z_sign * 0.882]) * ZT_CC_DIST

    # Convert to global coords
    C_ald_target = R @ local_approach  # global position for C_ald (relative to Cα=0)

    # ── 6. Orient aldehyde at target position ─────────────────────────────
    # Translate aldehyde to put C_ald at origin
    ald_centered = ald_coords - C_ald

    # Current C=O direction
    co_dir_cur = (O2 - C_ald)
    co_dir_cur /= np.linalg.norm(co_dir_cur)

    # Target C=O direction: O₂ should point roughly toward Li coordination site.
    # Li bridges O₁ and O₂. O₁ is at enol_coords_loc[o1_idx] in global.
    # Target: O₂ points toward Li, which is ~toward O₁ from C_ald position.
    O1_global = enol_coords_loc[o1_idx]  # enolate oxygen in global (Cα=origin)
    # Approximate O₂ target direction: toward O₁ from C_ald
    co_dir_target = O1_global - C_ald_target
    if np.linalg.norm(co_dir_target) < 0.1:
        co_dir_target = -R[:, 0]  # fallback: along -x
    co_dir_target /= np.linalg.norm(co_dir_target)

    # Rotation to align C=O
    Rco = _align_rotation(co_dir_cur, co_dir_target)
    ald_oriented = (Rco @ ald_centered.T).T  # (n_ald, 3)

    # ── 7. si vs re: rotate aldehyde around Cα→C_ald axis ────────────────
    # re face: flip the R-group orientation (180° rotation around forming bond axis)
    if face == "re":
        bond_axis = C_ald_target / np.linalg.norm(C_ald_target)
        cos_a, sin_a = -1.0, 0.0  # 180° rotation
        cross_mat = np.array([
            [0,             -bond_axis[2],  bond_axis[1]],
            [ bond_axis[2], 0,             -bond_axis[0]],
            [-bond_axis[1],  bond_axis[0],  0           ]
        ])
        R180 = (cos_a * np.eye(3)
                + (1 - cos_a) * np.outer(bond_axis, bond_axis)
                + sin_a * cross_mat)
        ald_oriented = (R180 @ ald_oriented.T).T

    # Translate aldehyde to C_ald target position
    ald_final = ald_oriented + C_ald_target

    # ── 8. Add Li bridging O₁ and O₂ ──────────────────────────────────────
    O1_pos = enol_coords_loc[o1_idx]        # global, Cα=0
    O2_pos = ald_final[o2_idx]             # global
    Li_pos = (O1_pos + O2_pos) / 2.0       # rough midpoint

    # Correct Li position to be at ZT distances
    v_liO1 = O1_pos - Li_pos
    v_liO2 = O2_pos - Li_pos
    d_liO1 = np.linalg.norm(v_liO1)
    d_liO2 = np.linalg.norm(v_liO2)
    if d_liO1 > 0.1:
        Li_pos = O1_pos - v_liO1 / d_liO1 * ZT_LI_O1 * 0.5 + O2_pos - v_liO2 / d_liO2 * ZT_LI_O2 * 0.5
        Li_pos /= 2.0  # rebalance

    # ── 9. Return fragments separately (for VDW clash) ────────────────────
    return (
        np.array(enol_nums, dtype=np.int64), enol_coords_loc,   # enolate
        np.array(ald_nums, dtype=np.int64),  ald_final,          # aldehyde
    )


# ── xTB single-point ─────────────────────────────────────────────────────────

# UFF VDW radii (Å) for common atoms (used for steric clash calculation)
_UFF_R = {
    1: 1.285, 5: 1.819, 6: 1.908, 7: 1.830, 8: 1.750,
    9: 1.682, 14: 2.147, 15: 2.074, 16: 2.017, 17: 1.974,
    35: 2.038, 53: 2.228, 3: 1.226,   # Li
}
_DEFAULT_R = 1.9


def vdw_steric_energy(
    enol_nums: np.ndarray,
    enol_coords: np.ndarray,
    ald_nums: np.ndarray,
    ald_coords: np.ndarray,
) -> float:
    """Compute pairwise steric overlap energy between two fragments.

    Uses a Gaussian overlap function:
        E = Σᵢⱼ exp(−r_ij² / σ_ij²)  where σ_ij = (R_i + R_j) / 2

    Values are in [0, n_pairs] range (numerically stable, monotone in steric clash).
    This captures steric differentiation between si and re face TS geometries.
    Evans aldol selectivity is sterically controlled (Phase 11-B1 confirmed),
    so inter-fragment steric overlap in the TS geometry is the key predictor.
    """
    r_enol = np.array([_UFF_R.get(int(n), _DEFAULT_R) for n in enol_nums])
    r_ald = np.array([_UFF_R.get(int(n), _DEFAULT_R) for n in ald_nums])

    # (n_enol, n_ald)
    diff = enol_coords[:, None, :] - ald_coords[None, :, :]
    dists2 = np.sum(diff ** 2, axis=2)

    sigma2 = ((r_enol[:, None] + r_ald[None, :]) / 2) ** 2  # (n_enol, n_ald)

    # Gaussian overlap: large when atoms overlap (r → 0), small when far apart
    energy = np.sum(np.exp(-dists2 / sigma2))

    return float(energy)


# ── Per-reaction qTS features ─────────────────────────────────────────────────

def compute_qts_features(
    enolate_smi: str,
    aldehyde_smi: str,
) -> dict:
    """Compute 4 qTS features for one Evans aldol reaction.

    Builds 4 ZT TS scaffolds (si/re × chair/twist) and computes VDW steric
    clash energy between enolate and aldehyde fragments in each geometry.

    Physical basis: Evans aldol is sterically controlled (confirmed Phase 11-B1).
    The relative steric repulsion between fragments in the 4 TS geometries
    captures the facial selectivity preference (si_chair preferred for Z-enolate).

    Returns dict with QTS_FEATURE_NAMES, all NaN on failure.
    """
    nan_result = {k: float("nan") for k in QTS_FEATURE_NAMES}

    channels = [("si", "chair"), ("re", "chair"), ("si", "twist"), ("re", "twist")]
    energies = {}

    for face, conf in channels:
        scaffold = build_zt_scaffold(enolate_smi, aldehyde_smi, face=face, conformation=conf)
        if scaffold is None:
            return nan_result
        enol_nums, enol_coords, ald_nums, ald_coords = scaffold
        e = vdw_steric_energy(enol_nums, enol_coords, ald_nums, ald_coords)
        energies[(face, conf)] = e

    e_si_chair = energies[("si", "chair")]
    e_re_chair = energies[("re", "chair")]
    e_si_twist = energies[("si", "twist")]
    e_re_twist = energies[("re", "twist")]

    dE_si_re_chair = e_si_chair - e_re_chair
    dE_si_re_twist = e_si_twist - e_re_twist

    all_E = np.array([e_si_chair, e_re_chair, e_si_twist, e_re_twist])
    E_mean = all_E.mean()
    min_E_rel = all_E.min() - E_mean

    # Boltzmann-like weight of si_chair channel
    # Higher steric overlap = less favorable (more repulsive)
    # Use negative energy so lower-E (less steric) channel gets higher weight
    neg_E = -all_E  # convert "high overlap = bad" to "low overlap = good"
    neg_E_rel = neg_E - neg_E.max()
    # Scale: use range to normalize, so the probability is well-behaved
    E_range = all_E.max() - all_E.min() + 1e-8
    boltz = np.exp(neg_E_rel / (E_range * 0.1 + 1e-8))
    boltz_si = float(boltz[0] / boltz.sum())

    return {
        "qts_dE_si_re_chair": dE_si_re_chair,
        "qts_dE_si_re_twist": dE_si_re_twist,
        "qts_min_E_kcal": min_E_rel,
        "qts_boltzmann_si": boltz_si,
    }


# ── Worker for parallel execution ─────────────────────────────────────────────

def _worker_qts(args):
    """Top-level worker (must be picklable)."""
    idx, enolate_smi, aldehyde_smi = args
    try:
        result = compute_qts_features(enolate_smi, aldehyde_smi)
    except Exception as e:
        result = {k: float("nan") for k in QTS_FEATURE_NAMES}
    return idx, result


# ── Batch computation with checkpointing ─────────────────────────────────────

PER_REACTION_TIMEOUT = 120   # seconds — 4 single-points per reaction
BATCH_SIZE_MULTIPLIER = 1    # batch = n_workers (one batch per n_workers reactions)


def compute_qts_features_batch(
    enolates_df: pd.DataFrame,
    aldehydes_smi: list,
    checkpoint_path: Path,
    n_workers: int = 8,
) -> pd.DataFrame:
    """Compute qTS features for all 1822 reactions in parallel with checkpointing.

    Args:
        enolates_df: DataFrame with 'enolate_smiles' column (1822 rows)
        aldehydes_smi: list of aldehyde SMILES (1822 entries, None if not found)
        checkpoint_path: path to pkl checkpoint file
        n_workers: parallel workers

    Returns:
        DataFrame (1822 × 4) with QTS_FEATURE_NAMES columns
    """
    n = len(enolates_df)
    assert len(aldehydes_smi) == n

    # Load checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        logger.info(f"Resumed checkpoint: {len(results)}/{n} done")
    else:
        results = {}

    # Build work queue
    todo = []
    for i in range(n):
        if i in results:
            continue
        enol_smi = str(enolates_df["enolate_smiles"].iloc[i])
        ald_smi = aldehydes_smi[i]
        if ald_smi is None:
            results[i] = {k: float("nan") for k in QTS_FEATURE_NAMES}
            continue
        todo.append((i, enol_smi, ald_smi))

    logger.info(f"qTS: {len(todo)} reactions to compute ({len(results)} cached)")

    batch_size = n_workers
    batch_timeout = PER_REACTION_TIMEOUT * batch_size

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]
        done_count = len(results)
        logger.info(f"  Batch {batch_start//batch_size + 1}/"
                    f"{(len(todo)-1)//batch_size + 1}  "
                    f"({done_count}/{n} done)")

        pool = ProcessPoolExecutor(max_workers=n_workers)
        futures = {pool.submit(_worker_qts, args): args[0] for args in batch}

        try:
            for fut in as_completed(futures, timeout=batch_timeout):
                idx_done = futures[fut]
                try:
                    idx_res, feat = fut.result()
                    results[idx_res] = feat
                except Exception:
                    results[idx_done] = {k: float("nan") for k in QTS_FEATURE_NAMES}
        except TimeoutError:
            logger.warning(f"Batch timeout — collecting partial results")
            for fut, idx_done in futures.items():
                if idx_done not in results:
                    if fut.done():
                        try:
                            idx_res, feat = fut.result()
                            results[idx_res] = feat
                        except Exception:
                            pass
                    results.setdefault(idx_done, {k: float("nan") for k in QTS_FEATURE_NAMES})
        finally:
            pids = list((pool._processes or {}).keys())
            pool.shutdown(wait=False, cancel_futures=True)
            for pid in pids:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass

        # Save checkpoint after each batch
        with open(checkpoint_path, "wb") as f:
            pickle.dump(results, f)

    # Assemble DataFrame
    rows = []
    for i in range(n):
        if i in results:
            rows.append(results[i])
        else:
            rows.append({k: float("nan") for k in QTS_FEATURE_NAMES})

    df = pd.DataFrame(rows, columns=QTS_FEATURE_NAMES)
    logger.info(f"qTS complete: {df.notna().all(axis=1).sum()}/{n} fully computed")
    return df
