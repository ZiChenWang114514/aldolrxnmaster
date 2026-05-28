"""Chemical constants, SMARTS patterns, and lookup tables for V3 rebuild.

Sources:
  - Kamlet-Taft: Reichardt, Chem. Rev. 1994; Marcus, Chem. Soc. Rev. 1993
  - Physical params: CRC Handbook of Chemistry and Physics, 97th ed.
  - Base pKa: Bordwell pKa table (DMSO scale → approximate THF scale)
  - Lewis acidity: Pearson, JACS 1963; Parr & Pearson, JACS 1983
  - Sterimol A-values: Winstein & Holness, JACS 1955; Eliel, Stereochemistry of Carbon Compounds
"""

from pathlib import Path

# ─────────────────── Project paths ───────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
V3_DIR = PROJECT_DIR / "data" / "v3"

# ─────────────────── SMARTS patterns ───────────────────

# Evans oxazolidinone: match C4 chiral center on the ring
# C4 is bonded to: ring-O, ring-N, H, and substituent (Bn/iPr/Ph)
OXAZ_SMARTS = "[C:1]1([*])COC(=O)N1"

# Beta-hydroxy-N-acyl oxazolidinone product (Evans aldol product core)
# :1 = Cb (OH-bearing carbon from aldehyde)
# :2 = Ca (alpha-carbon from enolate, connected to C(=O)N)
EVANS_PRODUCT_SMARTS = "[C:1]([OH1])([#6])[C:2]([#6])[C](=O)[N]"

# Enolate reactive center (for steric descriptors)
ENOLATE_SMARTS = "[CH,CH2,C:1]=[CX3:2](-[OX1,OX2:3])-[NX3:4]"
KETONE_SMARTS = "[CH2,CH;X3,X4:1]-[CX3:2](=[OX1:3])-[NX3:4]"

# ─────────────────── Solvent parameters ───────────────────
# Extended: Kamlet-Taft (alpha, beta, pi_star, ET30) + physical (epsilon, viscosity_cP, bp_C)
#         + Amar/Sigman 2019 descriptors: mw, density_g_mL, refractive_index, dipole_D, molar_vol_cm3, logP
# Sources: CRC Handbook 97th ed., Stenutz tables, PubChem, DrugBank
SOLVENT_DB = {
    "dichloromethane": {
        "alpha": 0.13, "beta": 0.10, "pi_star": 0.82, "ET30": 40.7,
        "epsilon": 8.93, "viscosity_cP": 0.44, "bp_C": 40.0,
        "mw": 84.93, "density_g_mL": 1.327, "refractive_index": 1.424, "dipole_D": 1.60, "molar_vol_cm3": 64.0, "logP": 1.25,
    },
    "tetrahydrofuran": {
        "alpha": 0.00, "beta": 0.55, "pi_star": 0.58, "ET30": 37.4,
        "epsilon": 7.52, "viscosity_cP": 0.46, "bp_C": 66.0,
        "mw": 72.11, "density_g_mL": 0.889, "refractive_index": 1.407, "dipole_D": 1.75, "molar_vol_cm3": 81.1, "logP": 0.46,
    },
    "ethyl acetate": {
        "alpha": 0.00, "beta": 0.45, "pi_star": 0.55, "ET30": 38.1,
        "epsilon": 6.02, "viscosity_cP": 0.45, "bp_C": 77.1,
        "mw": 88.11, "density_g_mL": 0.902, "refractive_index": 1.372, "dipole_D": 1.78, "molar_vol_cm3": 97.7, "logP": 0.73,
    },
    "diethyl ether": {
        "alpha": 0.00, "beta": 0.47, "pi_star": 0.27, "ET30": 34.5,
        "epsilon": 4.27, "viscosity_cP": 0.22, "bp_C": 34.6,
        "mw": 74.12, "density_g_mL": 0.713, "refractive_index": 1.353, "dipole_D": 1.15, "molar_vol_cm3": 104.0, "logP": 0.89,
    },
    "toluene": {
        "alpha": 0.00, "beta": 0.11, "pi_star": 0.54, "ET30": 33.9,
        "epsilon": 2.38, "viscosity_cP": 0.59, "bp_C": 110.6,
        "mw": 92.14, "density_g_mL": 0.867, "refractive_index": 1.497, "dipole_D": 0.36, "molar_vol_cm3": 106.3, "logP": 2.73,
    },
    "pentane": {
        "alpha": 0.00, "beta": 0.00, "pi_star": -0.08, "ET30": 31.1,
        "epsilon": 1.84, "viscosity_cP": 0.24, "bp_C": 36.1,
        "mw": 72.15, "density_g_mL": 0.626, "refractive_index": 1.358, "dipole_D": 0.00, "molar_vol_cm3": 115.2, "logP": 3.39,
    },
    "hexane": {
        "alpha": 0.00, "beta": 0.00, "pi_star": -0.04, "ET30": 31.0,
        "epsilon": 1.88, "viscosity_cP": 0.33, "bp_C": 69.0,
        "mw": 86.18, "density_g_mL": 0.659, "refractive_index": 1.375, "dipole_D": 0.00, "molar_vol_cm3": 130.7, "logP": 3.90,
    },
    "heptane": {
        "alpha": 0.00, "beta": 0.00, "pi_star": -0.02, "ET30": 31.1,
        "epsilon": 1.92, "viscosity_cP": 0.42, "bp_C": 98.4,
        "mw": 100.20, "density_g_mL": 0.684, "refractive_index": 1.388, "dipole_D": 0.00, "molar_vol_cm3": 146.5, "logP": 4.66,
    },
    "acetonitrile": {
        "alpha": 0.19, "beta": 0.40, "pi_star": 0.75, "ET30": 45.6,
        "epsilon": 36.6, "viscosity_cP": 0.37, "bp_C": 82.0,
        "mw": 41.05, "density_g_mL": 0.786, "refractive_index": 1.344, "dipole_D": 3.92, "molar_vol_cm3": 52.2, "logP": -0.34,
    },
    "methanol": {
        "alpha": 0.98, "beta": 0.66, "pi_star": 0.60, "ET30": 55.4,
        "epsilon": 32.7, "viscosity_cP": 0.54, "bp_C": 64.7,
        "mw": 32.04, "density_g_mL": 0.791, "refractive_index": 1.329, "dipole_D": 1.70, "molar_vol_cm3": 40.5, "logP": -0.77,
    },
    "ethanol": {
        "alpha": 0.86, "beta": 0.75, "pi_star": 0.54, "ET30": 51.9,
        "epsilon": 24.5, "viscosity_cP": 1.07, "bp_C": 78.4,
        "mw": 46.07, "density_g_mL": 0.789, "refractive_index": 1.361, "dipole_D": 1.69, "molar_vol_cm3": 58.4, "logP": -0.31,
    },
    "isopropanol": {
        "alpha": 0.76, "beta": 0.84, "pi_star": 0.48, "ET30": 48.4,
        "epsilon": 19.9, "viscosity_cP": 2.04, "bp_C": 82.6,
        "mw": 60.10, "density_g_mL": 0.786, "refractive_index": 1.377, "dipole_D": 1.58, "molar_vol_cm3": 76.5, "logP": 0.05,
    },
    "water": {
        "alpha": 1.17, "beta": 0.47, "pi_star": 1.09, "ET30": 63.1,
        "epsilon": 80.1, "viscosity_cP": 1.00, "bp_C": 100.0,
        "mw": 18.02, "density_g_mL": 0.998, "refractive_index": 1.333, "dipole_D": 1.85, "molar_vol_cm3": 18.1, "logP": -1.38,
    },
    "chloroform": {
        "alpha": 0.20, "beta": 0.10, "pi_star": 0.58, "ET30": 39.1,
        "epsilon": 4.81, "viscosity_cP": 0.54, "bp_C": 61.2,
        "mw": 119.38, "density_g_mL": 1.489, "refractive_index": 1.446, "dipole_D": 1.04, "molar_vol_cm3": 80.2, "logP": 1.97,
    },
    "1,2-dichloroethane": {
        "alpha": 0.00, "beta": 0.10, "pi_star": 0.81, "ET30": 41.3,
        "epsilon": 10.4, "viscosity_cP": 0.84, "bp_C": 83.5,
        "mw": 98.96, "density_g_mL": 1.253, "refractive_index": 1.445, "dipole_D": 1.80, "molar_vol_cm3": 79.0, "logP": 1.48,
    },
    "dimethylformamide": {
        "alpha": 0.00, "beta": 0.69, "pi_star": 0.88, "ET30": 43.2,
        "epsilon": 36.7, "viscosity_cP": 0.92, "bp_C": 153.0,
        "mw": 73.09, "density_g_mL": 0.944, "refractive_index": 1.431, "dipole_D": 3.82, "molar_vol_cm3": 77.4, "logP": -1.01,
    },
    "dmso": {
        "alpha": 0.00, "beta": 0.76, "pi_star": 1.00, "ET30": 45.1,
        "epsilon": 46.7, "viscosity_cP": 1.99, "bp_C": 189.0,
        "mw": 78.13, "density_g_mL": 1.100, "refractive_index": 1.479, "dipole_D": 3.96, "molar_vol_cm3": 71.0, "logP": -1.35,
    },
    "acetone": {
        "alpha": 0.08, "beta": 0.43, "pi_star": 0.71, "ET30": 42.2,
        "epsilon": 20.5, "viscosity_cP": 0.32, "bp_C": 56.1,
        "mw": 58.08, "density_g_mL": 0.791, "refractive_index": 1.359, "dipole_D": 2.88, "molar_vol_cm3": 73.4, "logP": -0.24,
    },
    "benzene": {
        "alpha": 0.00, "beta": 0.10, "pi_star": 0.59, "ET30": 34.3,
        "epsilon": 2.28, "viscosity_cP": 0.65, "bp_C": 80.1,
        "mw": 78.11, "density_g_mL": 0.879, "refractive_index": 1.501, "dipole_D": 0.00, "molar_vol_cm3": 88.9, "logP": 2.13,
    },
    "carbon tetrachloride": {
        "alpha": 0.00, "beta": 0.10, "pi_star": 0.28, "ET30": 32.4,
        "epsilon": 2.24, "viscosity_cP": 0.97, "bp_C": 76.7,
        "mw": 153.82, "density_g_mL": 1.594, "refractive_index": 1.460, "dipole_D": 0.00, "molar_vol_cm3": 96.5, "logP": 2.83,
    },
    "dioxane": {
        "alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0,
        "epsilon": 2.21, "viscosity_cP": 1.54, "bp_C": 101.1,
        "mw": 88.11, "density_g_mL": 1.034, "refractive_index": 1.422, "dipole_D": 0.00, "molar_vol_cm3": 85.2, "logP": -0.27,
    },
    "pyridine": {
        "alpha": 0.00, "beta": 0.64, "pi_star": 0.87, "ET30": 40.5,
        "epsilon": 12.3, "viscosity_cP": 0.94, "bp_C": 115.3,
        "mw": 79.10, "density_g_mL": 0.982, "refractive_index": 1.510, "dipole_D": 2.22, "molar_vol_cm3": 80.6, "logP": 0.65,
    },
    "1-methyl-pyrrolidin-2-one": {
        "alpha": 0.00, "beta": 0.77, "pi_star": 0.92, "ET30": 42.2,
        "epsilon": 32.2, "viscosity_cP": 1.67, "bp_C": 202.0,
        "mw": 99.13, "density_g_mL": 1.028, "refractive_index": 1.470, "dipole_D": 4.09, "molar_vol_cm3": 96.4, "logP": -0.54,
    },
    "difluoromethane": {
        "alpha": 0.05, "beta": 0.05, "pi_star": 0.40, "ET30": 35.0,
        "epsilon": 14.2, "viscosity_cP": 0.12, "bp_C": -51.6,
        "mw": 52.02, "density_g_mL": 1.100, "refractive_index": 1.195, "dipole_D": 1.97, "molar_vol_cm3": 47.3, "logP": 0.20,
    },
    "dimethyl sulfoxide": {
        "alpha": 0.00, "beta": 0.76, "pi_star": 1.00, "ET30": 45.1,
        "epsilon": 46.7, "viscosity_cP": 1.99, "bp_C": 189.0,
        "mw": 78.13, "density_g_mL": 1.100, "refractive_index": 1.479, "dipole_D": 3.96, "molar_vol_cm3": 71.0, "logP": -1.35,
    },
    "methyl tert-butyl ether": {
        "alpha": 0.00, "beta": 0.55, "pi_star": 0.27, "ET30": 34.7,
        "epsilon": 4.50, "viscosity_cP": 0.36, "bp_C": 55.2,
        "mw": 88.15, "density_g_mL": 0.741, "refractive_index": 1.369, "dipole_D": 1.32, "molar_vol_cm3": 118.9, "logP": 0.94,
    },
    "2-methyltetrahydrofuran": {
        "alpha": 0.00, "beta": 0.53, "pi_star": 0.53, "ET30": 36.5,
        "epsilon": 6.97, "viscosity_cP": 0.47, "bp_C": 80.0,
        "mw": 86.13, "density_g_mL": 0.855, "refractive_index": 1.406, "dipole_D": 1.38, "molar_vol_cm3": 100.7, "logP": 0.96,
    },
    "nitromethane": {
        "alpha": 0.22, "beta": 0.06, "pi_star": 0.85, "ET30": 46.3,
        "epsilon": 35.9, "viscosity_cP": 0.63, "bp_C": 101.2,
        "mw": 61.04, "density_g_mL": 1.137, "refractive_index": 1.382, "dipole_D": 3.46, "molar_vol_cm3": 53.7, "logP": -0.35,
    },
    "1,4-dioxane": {
        "alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0,
        "epsilon": 2.21, "viscosity_cP": 1.54, "bp_C": 101.1,
        "mw": 88.11, "density_g_mL": 1.034, "refractive_index": 1.422, "dipole_D": 0.00, "molar_vol_cm3": 85.2, "logP": -0.27,
    },
    "cyclopentyl methyl ether": {
        "alpha": 0.00, "beta": 0.52, "pi_star": 0.30, "ET30": 34.8,
        "epsilon": 4.76, "viscosity_cP": 0.55, "bp_C": 106.0,
        "mw": 100.16, "density_g_mL": 0.860, "refractive_index": 1.419, "dipole_D": 1.30, "molar_vol_cm3": 116.5, "logP": 1.29,
    },
}

# Solvent name aliases (normalize dataset names → canonical names)
SOLVENT_ALIASES = {
    "1,2-dichloro-ethane": "1,2-dichloroethane",
    "Difluoromethane": "difluoromethane",
    "N,N-dimethylformamide": "dimethylformamide",
    "n,n-dimethylformamide": "dimethylformamide",
    "dmf": "dimethylformamide",
    "thf": "tetrahydrofuran",
    "dcm": "dichloromethane",
    "ch2cl2": "dichloromethane",
    "ch₂cl₂": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "etoac": "ethyl acetate",
    "et2o": "diethyl ether",
    "ether": "diethyl ether",
    "etoh": "ethanol",
    "meoh": "methanol",
    "mecn": "acetonitrile",
    "chcl3": "chloroform",
    "ccl4": "carbon tetrachloride",
    "phme": "toluene",
    "nmp": "1-methyl-pyrrolidin-2-one",
    "2-methf": "2-methyltetrahydrofuran",
    "2-me-thf": "2-methyltetrahydrofuran",
    "mtbe": "methyl tert-butyl ether",
    "tbme": "methyl tert-butyl ether",
    "cpme": "cyclopentyl methyl ether",
    "i-proh": "isopropanol",
    "2-propanol": "isopropanol",
    "dimethyl sulfoxide": "dmso",
}

# Metal → default solvent inference (from Evans aldol conventions)
METAL_DEFAULT_SOLVENT = {
    "B":  "dichloromethane",
    "Ti": "dichloromethane",
    "Sn": "dichloromethane",
    "Li": "tetrahydrofuran",
    "Mg": "ethyl acetate",
    "Zn": "dichloromethane",
    "Cu": "dichloromethane",
    "Zr": "dichloromethane",
}

# ─────────────────── Base properties ───────────────────
# pKa: conjugate acid pKa in THF (approximate from Bordwell DMSO → THF correction)
# steric_A: Winstein A-value (kcal/mol) — larger = bulkier
# nucleophilicity: Mayr nucleophilicity parameter N (approximate)
BASE_PROPERTIES = {
    "DIPEA":     {"pKa": 35.0, "steric_A": 2.5, "nucleophilicity": 12.0},
    "Et3N":      {"pKa": 33.0, "steric_A": 1.8, "nucleophilicity": 17.1},
    "LiHMDS":    {"pKa": 30.0, "steric_A": 3.0, "nucleophilicity": 5.0},
    "NaHMDS":    {"pKa": 30.0, "steric_A": 3.0, "nucleophilicity": 5.5},
    "LDA":       {"pKa": 36.0, "steric_A": 2.8, "nucleophilicity": 4.0},
    "KHMDS":     {"pKa": 30.0, "steric_A": 3.0, "nucleophilicity": 6.0},
    "other_base": {"pKa": 25.0, "steric_A": 1.5, "nucleophilicity": 10.0},
    "no_base":   {"pKa": 0.0,  "steric_A": 0.0, "nucleophilicity": 0.0},
}

# ─────────────────── Metal / Lewis acid properties ───────────────────
# coordination_num: typical coordination number in aldol TS
# ionic_radius_pm: Shannon ionic radius (pm)
# pearson_hardness: Pearson absolute hardness η (eV)
METAL_PROPERTIES = {
    "B":    {"coordination_num": 4, "ionic_radius_pm": 27,  "pearson_hardness": 8.0},
    "Ti":   {"coordination_num": 6, "ionic_radius_pm": 86,  "pearson_hardness": 3.4},
    "Sn":   {"coordination_num": 6, "ionic_radius_pm": 93,  "pearson_hardness": 5.0},
    "Li":   {"coordination_num": 4, "ionic_radius_pm": 76,  "pearson_hardness": 35.1},
    "Mg":   {"coordination_num": 6, "ionic_radius_pm": 72,  "pearson_hardness": 32.5},
    "Zn":   {"coordination_num": 4, "ionic_radius_pm": 74,  "pearson_hardness": 10.9},
    "Cu":   {"coordination_num": 4, "ionic_radius_pm": 73,  "pearson_hardness": 8.3},
    "Zr":   {"coordination_num": 8, "ionic_radius_pm": 84,  "pearson_hardness": 3.2},
    "none": {"coordination_num": 0, "ionic_radius_pm": 0,   "pearson_hardness": 0.0},
    "unknown": {"coordination_num": 0, "ionic_radius_pm": 0, "pearson_hardness": 0.0},
}

# ─────────────────── Reagent → role classification ───────────────────
BASE_MAP = {
    "n-ethyl-n,n-diisopropylamine": "DIPEA",
    "diisopropylethylamine": "DIPEA",
    "hunig's base": "DIPEA",
    "n,n-diisopropylethylamine": "DIPEA",
    "triethylamine": "Et3N",
    "lithium hexamethyldisilazane": "LiHMDS",
    "lithium bis(trimethylsilyl)amide": "LiHMDS",
    "sodium hexamethyldisilazane": "NaHMDS",
    "sodium bis(trimethylsilyl)amide": "NaHMDS",
    "lithium diisopropylamide": "LDA",
    "lda": "LDA",
    "potassium hexamethyldisilazane": "KHMDS",
    "potassium bis(trimethylsilyl)amide": "KHMDS",
    "2,6-lutidine": "other_base",
    "pyridine": "other_base",
    "imidazole": "other_base",
    "4-dimethylaminopyridine": "other_base",
    "dmap": "other_base",
    "1,8-diazabicyclo[5.4.0]undec-7-ene": "other_base",
    "dbu": "other_base",
    "sparteine": "other_base",
}

ACTIVATOR_MAP = {
    "di-n-butylboryl trifluoromethanesulfonate": "Bu2BOTf",
    "dibutylboron triflate": "Bu2BOTf",
    "dicyclohexylboron chloride": "Chx2BCl",
    "diisopinocampheylboron chloride": "Ipc2BCl",
    "9-borabicyclo(3.3.1)nonyl trifluoromethanesulfonate": "9BBN_OTf",
    "titanium tetrachloride": "TiCl4",
    "titanium(iv) chloride": "TiCl4",
    "tin(ii) trifluoromethanesulfonate": "Sn_OTf2",
    "tin(ii) triflate": "Sn_OTf2",
    "magnesium chloride": "MgCl2",
    "boron trifluoride diethyl etherate": "BF3_OEt2",
    "boron trifluoride etherate": "BF3_OEt2",
}

# Canonical category lists (for one-hot encoding)
BASE_CATEGORIES = ["DIPEA", "Et3N", "LiHMDS", "NaHMDS", "LDA", "KHMDS", "other_base", "no_base"]
METAL_CATEGORIES = ["B", "Cu", "Li", "Mg", "Sn", "Ti", "Zn", "Zr", "none", "unknown"]
ACTIVATOR_CATEGORIES = [
    "Bu2BOTf", "Chx2BCl", "Ipc2BCl", "9BBN_OTf", "TiCl4",
    "Sn_OTf2", "MgCl2", "BF3_OEt2", "other_activator",
]

# Auxiliary R-group types on C4 of Evans oxazolidinone
AUX_RGROUP_TYPES = ["benzyl", "isopropyl", "phenyl", "tert_butyl", "methyl", "indanyl", "other"]

# Solvent feature column names (14d: 8d original + 6d Amar/Sigman 2019 descriptors)
SOLVENT_FEATURE_NAMES = [
    "solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30",
    "solvent_epsilon", "solvent_viscosity", "solvent_bp",
    "solvent_known",
    "solvent_mw", "solvent_density", "solvent_refractive_index",
    "solvent_dipole", "solvent_molar_vol", "solvent_logP",
]
