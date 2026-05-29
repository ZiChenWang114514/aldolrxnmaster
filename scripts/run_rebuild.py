#!/usr/bin/env python3
"""V4 Rebuild Pipeline: Raw Reaxys (134K) -> substrate-controlled aldol clean dataset.

Usage:
    conda run -n aldol-rxn python scripts/run_rebuild_v4.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiralaldol.config import PROJECT_DIR
from chiralaldol.rebuild import (
    step01_load_filter,
    step02_parse_products,
    step03_auxiliary_detect,
    step04_canonicalize,
    step05_stereocenters,
    step06_atom_mapping,
    step07_label_extract,
    step08_label_validate,
    step08b_3d_synanti,
    step09_dedup,
    step10_conditions_extract,
    step11_conditions_engineer,
    step12_audit_output,
)
from chiralaldol.rebuild.audit import AuditTracker
from chiralaldol.rebuild.utils import setup_logging


def main():
    setup_logging()
    t0 = time.time()
    print("=" * 60)
    print("V4 REBUILD: Raw Reaxys -> Substrate-Controlled Aldol")
    print("=" * 60)

    # Initial row count for audit
    n_total = sum(1 for _ in open(PROJECT_DIR / "data" / "data.csv")) - 1
    audit = AuditTracker(n_total)

    # Step 01: Load + filter aldol reactions
    df = step01_load_filter.run(audit)

    # Step 02: Parse reaction SMILES, identify main product
    df = step02_parse_products.run(df, audit)

    # Step 03: Detect auxiliaries, exclude chiral catalysis
    df = step03_auxiliary_detect.run(df, audit)

    # Step 04: Canonicalize SMILES with stereo preservation
    df = step04_canonicalize.run(df, audit)

    # Step 05: Filter by stereocenter count
    df = step05_stereocenters.run(df, audit)

    # Step 06: Atom mapping (RXNMapper + template)
    df = step06_atom_mapping.run(df, audit)

    # Step 07: Extract stereochemistry labels
    df = step07_label_extract.run(df, audit)

    # Step 08: Cross-validate labels
    df = step08_label_validate.run(df, audit)

    # Step 08b: 3D dihedral-based syn/anti
    df = step08b_3d_synanti.run(df, audit)

    # Step 09: Deduplication + group_id
    df = step09_dedup.run(df, audit)

    # Step 10: Extract conditions
    df = step10_conditions_extract.run(df, audit)

    # Step 11: Condition feature engineering
    df, feat_df = step11_conditions_engineer.run(df, audit)

    # Step 12: Output + audit
    step12_audit_output.run(df, feat_df, audit)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
