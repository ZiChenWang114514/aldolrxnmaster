"""Centralized paths and constants for AldolRxnMaster."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Data paths
FEAT_DIR = PROJECT_DIR / "data" / "features_v5"
SPLITS_DIR = PROJECT_DIR / "data" / "splits_v5"
CLEAN_DIR = PROJECT_DIR / "data" / "clean_v5"

# Results paths
PRED_DIR = PROJECT_DIR / "results" / "predictions_v5"
RESULTS_DIR = PROJECT_DIR / "results"
OPTUNA_DIR = PROJECT_DIR / "results" / "optuna"

# ML constants
TARGET_LABEL = "label_joint"
N_CLASSES = 4
N_JOBS = 8

# Valid auxiliary types (mechanistically well-defined for substrate-controlled aldol)
VALID_AUXILIARIES = [
    "evans", "crimmins_thione", "crimmins_oxathione", "oppolzer",
    "myers", "abiko", "super_quat",
    "menthyl_ester", "borneol_ester", "oxazoline",
]

# SPMS feature paths
SPMS_DIR = FEAT_DIR / "spms"
