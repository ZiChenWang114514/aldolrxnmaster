#!/usr/bin/env python3
"""V3 Data Rebuild Pipeline — SOTA quality cleaning from raw alldata.csv.

Usage:
    conda run -n aldol-rxn python scripts/run_rebuild_v3.py [--skip-conformers] [--skip-rxnmapper]

This script orchestrates 17 sequential steps to rebuild the dataset:
  Step 0:  SA convention literature confirmation
  Step 1:  Raw data loading + initial audit
  Step 2:  SMILES strict isomeric canonicalization
  Step 3:  Stereocenter validation (delete <2)
  Step 4:  CIP label cross-validation (delete mismatches)
  Step 5:  Atom mapping dual verification (template + RXNMapper)
  Step 6:  Auxiliary chirality SMARTS extraction
  Step 7:  Template-based deduplication (role-aware)
  Step 8:  Solvent semantic parsing
  Step 9:  Condition feature engineering (44d)
  Step 10: Conformer generation with fallback chain
  Step 11: 3D steric features (34d)
  Step 12: Feature integration (~85d) + split-aware normalization
  Step 13: Data splitting (TSCV + scaffold + grouped)
  Step 14: Row-level audit report
  Step 15: Non-Evans processing
  Step 16: Verification suite

Output: data/v3/ directory with all results.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from chiralaldol.rebuild.logger import setup_logging
from chiralaldol.rebuild.constants import RAW_DIR, V3_DIR
from chiralaldol.rebuild import (
    step00_literature,
    step01_load,
    step02_canonicalize,
    step03_stereocenters,
    step04_cip_validation,
    step05_atom_mapping,
    step06_auxiliary,
    step07_dedup,
    step08_solvent,
    step09_conditions,
    step10_conformers,
    step11_steric,
    step12_features,
    step13_splits,
    step14_audit,
    step15_non_evans,
    step16_verify,
)

logger = logging.getLogger("rebuild_v3")


def main():
    parser = argparse.ArgumentParser(description="V3 Data Rebuild Pipeline")
    parser.add_argument("--skip-conformers", action="store_true",
                        help="Skip Steps 10-11 (conformer generation + steric features)")
    parser.add_argument("--skip-rxnmapper", action="store_true",
                        help="Skip RXNMapper in Step 5 (use template-only verification)")
    parser.add_argument("--date-suffix", default="20260515",
                        help="Date suffix for output files (default: 20260515)")
    args = parser.parse_args()

    # Setup
    V3_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["interim", "features", "splits", "splits/normalized",
                    "audit", "verification", "literature"]:
        (V3_DIR / subdir).mkdir(parents=True, exist_ok=True)

    log_path = V3_DIR / "rebuild_v3.log"
    setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("V3 Data Rebuild Pipeline — Starting")
    logger.info("=" * 60)
    t0 = time.time()

    # Initialize context
    context = {
        "project_dir": PROJECT_DIR,
        "raw_dir": RAW_DIR,
        "output_dir": V3_DIR,
        "date_suffix": args.date_suffix,
        "skip_rxnmapper": args.skip_rxnmapper,
    }

    # ── Execute pipeline ──
    steps = [
        ("Step 0: Literature", step00_literature.run),
        ("Step 1: Load", step01_load.run),
        ("Step 2: Canonicalize", step02_canonicalize.run),
        ("Step 3: Stereocenters", step03_stereocenters.run),
        ("Step 4: CIP Validation", step04_cip_validation.run),
        ("Step 5: Atom Mapping", step05_atom_mapping.run),
        ("Step 6: Auxiliary", step06_auxiliary.run),
        ("Step 7: Dedup", step07_dedup.run),
        ("Step 8: Solvent", step08_solvent.run),
        ("Step 9: Conditions", step09_conditions.run),
    ]

    if not args.skip_conformers:
        steps += [
            ("Step 10: Conformers", step10_conformers.run),
            ("Step 11: Steric", step11_steric.run),
        ]
    else:
        logger.info("Skipping Steps 10-11 (conformer generation + steric features)")

    steps += [
        ("Step 12: Features", step12_features.run),
        ("Step 13: Splits", step13_splits.run),
        ("Step 14: Audit", step14_audit.run),
        ("Step 15: Non-Evans", step15_non_evans.run),
        ("Step 16: Verify", step16_verify.run),
    ]

    for step_name, step_fn in steps:
        t_step = time.time()
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Running {step_name}...")
        logger.info(f"{'─' * 40}")
        try:
            context = step_fn(context)
        except Exception as e:
            logger.error(f"FAILED at {step_name}: {e}", exc_info=True)
            raise
        elapsed = time.time() - t_step
        n_rows = len(context.get("df", []))
        logger.info(f"  [{step_name}] done in {elapsed:.1f}s, {n_rows} rows remaining")

    # ── Save final Evans dataset ──
    df = context["df"]
    evans_mask = df["Reaction_Class"] == "EvansAux"
    evans_df = df[evans_mask].reset_index(drop=True)
    date_suffix = args.date_suffix

    evans_path = V3_DIR / f"evans_clean_{date_suffix}.csv"
    evans_df.to_csv(evans_path, index=False)
    logger.info(f"\nFinal Evans dataset: {len(evans_df)} rows → {evans_path}")

    # ── Summary ──
    total_time = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"V3 Pipeline Complete in {total_time:.1f}s")
    logger.info(f"  Evans V3: {len(evans_df)} rows")
    logger.info(f"  Non-Evans V3: {len(df[~evans_mask])} rows")
    logger.info(f"  Output: {V3_DIR}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
