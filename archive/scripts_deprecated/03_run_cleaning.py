#!/usr/bin/env python
"""Orchestrate the full data cleaning pipeline (Steps 1-7)."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aldolrxnmaster.data import consolidate, deduplicate, validate_smiles, unify_labels, impute, split, quality_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    root = Path("/data2/zcwang/aldolrxnmaster")

    logger.info("=" * 60)
    logger.info("AldolRxnMaster Data Cleaning Pipeline")
    logger.info("=" * 60)

    # Step 1: Consolidate
    logger.info("\n>>> Step 1/7: Consolidate raw data...")
    consolidate.run(root)

    # Step 2: Deduplicate
    logger.info("\n>>> Step 2/7: Deduplicate...")
    deduplicate.run(root)

    # Step 3: Validate SMILES
    logger.info("\n>>> Step 3/7: Validate SMILES & chirality audit...")
    validate_smiles.run(root)

    # Step 4: Unify labels
    logger.info("\n>>> Step 4/7: Unify labels to 4-class joint...")
    unify_labels.run(root)

    # Step 5: Impute missing values
    logger.info("\n>>> Step 5/7: Impute missing values...")
    impute.run(root)

    # Step 6: Generate splits
    logger.info("\n>>> Step 6/7: Generate splits...")
    split.run(root, subset="all")
    split.run(root, subset="evans")

    # Step 7: Quality report
    logger.info("\n>>> Step 7/7: Generate quality report...")
    quality_report.run(root)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
