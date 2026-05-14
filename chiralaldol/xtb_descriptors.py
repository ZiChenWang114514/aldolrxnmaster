"""B1: GFN2-xTB Electronic Descriptors for Evans Aldol Reactants.

Computes frontier molecular orbital (FMO) properties for:
  - Enolate (nucleophile): HOMO/LUMO energies, gap, dipole, Mulliken charge at Cα,
    Fukui f⁻(Cα) via N−1 electron finite difference
  - Aldehyde (electrophile): HOMO/LUMO energies, gap, dipole, Mulliken charge at CHO-C,
    Fukui f⁺(CHO-C) via N+1 electron finite difference

Physical motivation:
  The Zimmerman-Traxler TS involves nucleophilic attack of enolate Cα on the
  aldehyde carbonyl carbon. The Fukui function directly quantifies the reactivity
  of these sites: f⁻(Cα) measures nucleophilicity, f⁺(CHO-C) measures electrophilicity.
  HOMO-LUMO gap of the enolate-aldehyde pair drives the soft-acid/soft-base
  interaction that controls facial selectivity.

Features (12d total):
  Enolate (6d):
    enol_HOMO_eV, enol_LUMO_eV, enol_gap_eV   : orbital energies / gap
    enol_dipole_D                               : dipole magnitude (Debye)
    enol_Ca_charge                              : Mulliken charge at Cα
    enol_Ca_fukui_minus                         : Fukui f⁻(Cα) = q(N−1,Cα) − q(N,Cα)
  Aldehyde (6d):
    ald_HOMO_eV, ald_LUMO_eV, ald_gap_eV       : orbital energies / gap
    ald_dipole_D                                : dipole magnitude (Debye)
    ald_CHO_charge                              : Mulliken charge at CHO-C
    ald_CHO_fukui_plus                          : Fukui f⁺(CHO-C) = q(N,CHO-C) − q(N+1,CHO-C)
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# Unit conversion constants
HARTREE_TO_EV = 27.2114
ANG_TO_BOHR = 1.0 / 0.529177
DEBYE_CONV = 2.5418  # ea₀ → Debye

# SMARTS for Cα in enolate: R-Cα=C([O-])-N-[Oxaz]
# Atom :1 is Cα (NOT bonded to O⁻ or N directly)
_ENOLATE_Ca_SMARTS = Chem.MolFromSmarts("[CX3:1]=[CX3]([O-])[NX3]")

# Reuse from aldehyde_steric: [CX3;H1:1]=[OX1]
_ALDEHYDE_SMARTS = Chem.MolFromSmarts("[CX3;H1:1]=[OX1]")

# Feature name lists
ENOLATE_XTB_NAMES = [
    "enol_HOMO_eV", "enol_LUMO_eV", "enol_gap_eV",
    "enol_dipole_D", "enol_Ca_charge", "enol_Ca_fukui_minus",
]

ALDEHYDE_XTB_NAMES = [
    "ald_HOMO_eV", "ald_LUMO_eV", "ald_gap_eV",
    "ald_dipole_D", "ald_CHO_charge", "ald_CHO_fukui_plus",
]

XTB_FEATURE_NAMES = ENOLATE_XTB_NAMES + ALDEHYDE_XTB_NAMES  # 12d


# ─── Atom finding helpers ────────────────────────────────────────────────────

def find_Ca_idx(mol: Chem.Mol) -> int | None:
    """Find Cα index in enolate molecule (R-Cα=C([O-])-N)."""
    m = mol.GetSubstructMatches(_ENOLATE_Ca_SMARTS)
    return m[0][0] if m else None


def find_CHO_idx(mol: Chem.Mol) -> int | None:
    """Find carbonyl-C index in aldehyde molecule (R-CHO)."""
    m = mol.GetSubstructMatches(_ALDEHYDE_SMARTS)
    return m[0][0] if m else None


# ─── 3D generation ──────────────────────────────────────────────────────────

def smiles_to_3d(smiles: str) -> tuple[list[int], np.ndarray] | tuple[None, None]:
    """Generate a single 3D conformer (ETKDG + light MMFF polish).

    Keeps MMFF iterations low (200) so large Evans auxiliary enolates don't
    spend minutes on geometry alone — GFN2-xTB is tolerant of imperfect input.

    Returns:
        (atomic_nums, coords_ang) with explicit Hs, or (None, None) on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    ret = AllChem.EmbedMolecule(mol_h, params)
    if ret != 0:
        ret = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv2())
    if ret != 0:
        return None, None

    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
    except Exception:
        pass

    conf = mol_h.GetConformer()
    atomic_nums = [a.GetAtomicNum() for a in mol_h.GetAtoms()]
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol_h.GetNumAtoms())])
    return atomic_nums, coords


# ─── xTB single-point ───────────────────────────────────────────────────────

def _run_xtb(atomic_nums: list[int], coords_ang: np.ndarray, charge: int = 0):
    """Run GFN2-xTB single-point. Returns tblite Result or None."""
    try:
        from tblite.interface import Calculator
        coords_bohr = np.array(coords_ang, dtype=np.float64) * ANG_TO_BOHR
        calc = Calculator("GFN2-xTB", np.array(atomic_nums, dtype=np.int64), coords_bohr,
                          charge=charge)
        calc.set("verbosity", 0)
        calc.set("max-iter", 200)   # limit SCF to 200 cycles; prevents infinite loops
        return calc.singlepoint()
    except Exception as e:
        logger.debug(f"xTB failed (charge={charge}): {e}")
        return None


def _get_orbital_features(res) -> tuple[float, float, float]:
    """Extract (HOMO_eV, LUMO_eV, gap_eV) or (nan, nan, nan)."""
    nan3 = float("nan"), float("nan"), float("nan")
    e = res.get("orbital-energies")
    occ = res.get("orbital-occupations")
    if e is None or occ is None:
        return nan3
    occ_mask = occ > 0.5
    virt_mask = occ < 0.5
    if not occ_mask.any() or not virt_mask.any():
        return nan3
    homo = float(e[occ_mask][-1]) * HARTREE_TO_EV
    lumo = float(e[virt_mask][0]) * HARTREE_TO_EV
    return homo, lumo, lumo - homo


def _get_dipole(res) -> float:
    """Extract dipole magnitude in Debye."""
    d = res.get("dipole")
    return float(np.linalg.norm(d)) * DEBYE_CONV if d is not None else float("nan")


# ─── Per-molecule xTB descriptors ───────────────────────────────────────────

def compute_enolate_xtb(smiles: str) -> dict:
    """Compute 6 xTB electronic features for an enolate SMILES.

    Enolate has formal charge -1. Fukui f⁻(Cα) uses N-1 electron calc (charge=0).
    Returns dict with keys = ENOLATE_XTB_NAMES, values are floats (nan on failure).
    """
    nan_result = {k: float("nan") for k in ENOLATE_XTB_NAMES}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return nan_result

    ca_idx = find_Ca_idx(mol)
    if ca_idx is None:
        logger.debug(f"Enolate Ca not found: {smiles}")
        return nan_result

    atomic_nums, coords = smiles_to_3d(smiles)
    if atomic_nums is None:
        return nan_result

    # Main calculation: charge=-1 (enolate anion)
    res_n = _run_xtb(atomic_nums, coords, charge=-1)
    if res_n is None:
        return nan_result

    homo, lumo, gap = _get_orbital_features(res_n)
    dipole = _get_dipole(res_n)
    charges_n = res_n.get("charges")
    ca_charge = float(charges_n[ca_idx]) if charges_n is not None else float("nan")

    # N-1 electron calc for Fukui f⁻: charge=0 (remove one electron)
    res_n1 = _run_xtb(atomic_nums, coords, charge=0)
    ca_fukui_minus = float("nan")
    if res_n1 is not None:
        charges_n1 = res_n1.get("charges")
        if charges_n1 is not None:
            # f⁻(Cα) = q(N-1, Cα) - q(N, Cα)
            ca_fukui_minus = float(charges_n1[ca_idx]) - float(charges_n[ca_idx])

    return {
        "enol_HOMO_eV": homo,
        "enol_LUMO_eV": lumo,
        "enol_gap_eV": gap,
        "enol_dipole_D": dipole,
        "enol_Ca_charge": ca_charge,
        "enol_Ca_fukui_minus": ca_fukui_minus,
    }


def compute_aldehyde_xtb(smiles: str) -> dict:
    """Compute 6 xTB electronic features for an aldehyde SMILES.

    Aldehyde has charge=0. Fukui f⁺(CHO-C) uses N+1 electron calc (charge=-1).
    Returns dict with keys = ALDEHYDE_XTB_NAMES, values are floats (nan on failure).
    """
    nan_result = {k: float("nan") for k in ALDEHYDE_XTB_NAMES}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return nan_result

    cho_idx = find_CHO_idx(mol)
    if cho_idx is None:
        logger.debug(f"Aldehyde CHO not found: {smiles}")
        return nan_result

    atomic_nums, coords = smiles_to_3d(smiles)
    if atomic_nums is None:
        return nan_result

    # Main calculation: charge=0
    res_n = _run_xtb(atomic_nums, coords, charge=0)
    if res_n is None:
        return nan_result

    homo, lumo, gap = _get_orbital_features(res_n)
    dipole = _get_dipole(res_n)
    charges_n = res_n.get("charges")
    cho_charge = float(charges_n[cho_idx]) if charges_n is not None else float("nan")

    # N+1 electron calc for Fukui f⁺: charge=-1 (add one electron)
    res_n1 = _run_xtb(atomic_nums, coords, charge=-1)
    cho_fukui_plus = float("nan")
    if res_n1 is not None:
        charges_n1 = res_n1.get("charges")
        if charges_n1 is not None:
            # f⁺(CHO-C) = q(N, CHO-C) - q(N+1, CHO-C)
            cho_fukui_plus = float(charges_n[cho_idx]) - float(charges_n1[cho_idx])

    return {
        "ald_HOMO_eV": homo,
        "ald_LUMO_eV": lumo,
        "ald_gap_eV": gap,
        "ald_dipole_D": dipole,
        "ald_CHO_charge": cho_charge,
        "ald_CHO_fukui_plus": cho_fukui_plus,
    }


# ─── Worker functions (top-level for pickling) ──────────────────────────────

def _worker_enolate(args):
    smiles, idx_list = args
    try:
        result = compute_enolate_xtb(smiles)
    except Exception:
        result = {k: float("nan") for k in ENOLATE_XTB_NAMES}
    return smiles, result, idx_list


def _worker_aldehyde(args):
    smiles, idx_list = args
    try:
        result = compute_aldehyde_xtb(smiles)
    except Exception:
        result = {k: float("nan") for k in ALDEHYDE_XTB_NAMES}
    return smiles, result, idx_list


# ─── Batch computation ───────────────────────────────────────────────────────

def _save_ckpt(path: Path, enol_results: dict, ald_results: dict) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"enol_results": enol_results, "ald_results": ald_results}, f)


def _load_ckpt(path: Path) -> tuple[dict, dict]:
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    # Support both old (DataFrame) and new (dict) formats
    if isinstance(d, dict) and "enol_results" in d:
        return d["enol_results"], d["ald_results"]
    return {}, {}   # old format — restart from scratch


def compute_xtb_features_batch(
    enolate_smiles: pd.Series,
    aldehyde_smiles: pd.Series,
    n_workers: int = 8,
    checkpoint_path: Path | None = None,
    ckpt_interval: int = 50,
) -> pd.DataFrame:
    """Compute xTB electronic features for all 1822 reactions.

    Deduplicates by unique SMILES before calculation, then joins back.
    Checkpoints intermediate results every ckpt_interval molecules so a
    restart doesn't lose all work.

    Args:
        enolate_smiles: Series of enolate SMILES (index = reaction idx)
        aldehyde_smiles: Series of aldehyde SMILES (index = reaction idx)
        n_workers: parallel workers (ProcessPoolExecutor)
        checkpoint_path: path to save/load checkpoint pkl
        ckpt_interval: save checkpoint every N completed molecules

    Returns:
        DataFrame with index = reaction idx, columns = XTB_FEATURE_NAMES (12d)
    """
    import pickle

    n_total = len(enolate_smiles)
    assert len(aldehyde_smiles) == n_total

    # Build unique SMILES → index mapping
    enolate_unique: dict[str, list[int]] = {}
    for idx, smi in enolate_smiles.items():
        enolate_unique.setdefault(str(smi), []).append(idx)

    aldehyde_unique: dict[str, list[int]] = {}
    for idx, smi in aldehyde_smiles.items():
        aldehyde_unique.setdefault(str(smi), []).append(idx)

    logger.info(f"xTB: {len(enolate_unique)} unique enolates, "
                f"{len(aldehyde_unique)} unique aldehydes")

    # Load intermediate checkpoint if available
    enol_results: dict[str, dict] = {}
    ald_results: dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        enol_results, ald_results = _load_ckpt(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {len(enol_results)} enolates, "
                    f"{len(ald_results)} aldehydes already done")

    # Per-future timeout: skip molecules stuck in C-level tblite loops.
    # SIGALRM is unreliable for C extensions; instead we use as_completed(timeout=)
    # to detect a stuck batch and force-kill remaining workers.
    PER_MOL_TIMEOUT = 150   # seconds before declaring a molecule stuck

    def _run_batch(worker_fn, todo, results, feat_names, label):
        """Submit a batch, collect results, skip timed-out molecules."""
        n_done = 0
        # Small batch = n_workers so a single stuck molecule only blocks one batch (≤5 min)
        BATCH = n_workers
        for chunk_start in range(0, len(todo), BATCH):
            chunk = todo[chunk_start: chunk_start + BATCH]
            pool = ProcessPoolExecutor(max_workers=n_workers)
            pending = {pool.submit(worker_fn, t): t for t in chunk}
            batch_timeout = PER_MOL_TIMEOUT * 3  # 450s max per n_workers-molecule batch
            try:
                for fut in as_completed(pending, timeout=batch_timeout):
                    try:
                        smi, result, _ = fut.result()
                    except Exception as e:
                        task = pending[fut]
                        smi = task[0]
                        result = {k: float("nan") for k in feat_names}
                        logger.warning(f"  {label} error on {smi[:30]}: {e}")
                    results[smi] = result
                    n_done += 1
            except TimeoutError:
                logger.warning(f"  {label} batch timed out — collecting partial, skipping stuck")
                for fut, task in pending.items():
                    smi = task[0]
                    if smi in results:
                        continue
                    if fut.done():
                        try:
                            _, result, _ = fut.result()
                        except Exception:
                            result = {k: float("nan") for k in feat_names}
                    else:
                        result = {k: float("nan") for k in feat_names}
                        fut.cancel()
                    results[smi] = result
                    n_done += 1
            finally:
                import os as _os
                pids = list((pool._processes or {}).keys())
                pool.shutdown(wait=False, cancel_futures=True)
                for pid in pids:
                    try:
                        _os.kill(pid, 9)
                    except Exception:
                        pass
            if checkpoint_path:
                _save_ckpt(checkpoint_path, enol_results, ald_results)
            logger.info(f"  {label}: {len(results)}/{len(todo) + chunk_start} done")
        return n_done

    # ── Enolate batch ────────────────────────────────────────────────────────
    enol_todo = [(smi, idxs) for smi, idxs in enolate_unique.items()
                 if smi not in enol_results]
    if enol_todo:
        logger.info(f"Computing {len(enol_todo)} enolate xTB properties "
                    f"({len(enolate_unique)-len(enol_todo)} cached)...")
        _run_batch(_worker_enolate, enol_todo, enol_results, ENOLATE_XTB_NAMES, "Enolate xTB")
    logger.info(f"  Enolate xTB complete: {len(enol_results)}/{len(enolate_unique)}")

    # ── Aldehyde batch ───────────────────────────────────────────────────────
    ald_todo = [(smi, idxs) for smi, idxs in aldehyde_unique.items()
                if smi not in ald_results]
    if ald_todo:
        logger.info(f"Computing {len(ald_todo)} aldehyde xTB properties "
                    f"({len(aldehyde_unique)-len(ald_todo)} cached)...")
        _run_batch(_worker_aldehyde, ald_todo, ald_results, ALDEHYDE_XTB_NAMES, "Aldehyde xTB")
    logger.info(f"  Aldehyde xTB complete: {len(ald_results)}/{len(aldehyde_unique)}")

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    rows = []
    for idx in enolate_smiles.index:
        esmi = str(enolate_smiles[idx])
        asmi = str(aldehyde_smiles[idx])
        er = enol_results.get(esmi, {k: float("nan") for k in ENOLATE_XTB_NAMES})
        ar = ald_results.get(asmi, {k: float("nan") for k in ALDEHYDE_XTB_NAMES})
        row = {"idx": idx}
        row.update(er)
        row.update(ar)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("idx")
    df = df[XTB_FEATURE_NAMES]  # enforce column order

    # Report success rate
    n_ok = df["enol_HOMO_eV"].notna().sum()
    logger.info(f"xTB success: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%)")

    # Clean up checkpoint
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    return df
