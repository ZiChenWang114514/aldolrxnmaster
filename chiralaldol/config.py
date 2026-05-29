"""Centralized paths and constants for AldolRxnMaster."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Data paths
FEAT_DIR = PROJECT_DIR / "data" / "features_v4"
SPLITS_DIR = PROJECT_DIR / "data" / "splits_v4"
CLEAN_DIR = PROJECT_DIR / "data" / "clean_v4"

# Results paths
PRED_DIR = PROJECT_DIR / "results" / "predictions_v4"
RESULTS_DIR = PROJECT_DIR / "results"
OPTUNA_DIR = PROJECT_DIR / "results" / "optuna"

# ML constants
TARGET_LABEL = "label_joint"
N_CLASSES = 4
N_JOBS = 8

# Valid auxiliary types (mechanistically well-defined for ZT aldol)
VALID_AUXILIARIES = ["evans", "crimmins_thione", "crimmins_oxathione", "oppolzer"]
