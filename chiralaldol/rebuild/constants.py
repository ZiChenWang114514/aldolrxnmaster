"""Chemical constants, SMARTS patterns, and lookup tables for V5 rebuild.

Extends V3 constants with multi-auxiliary SMARTS, chiral catalyst exclusion,
and Reaxys column mappings.

Sources:
  - Kamlet-Taft: Reichardt, Chem. Rev. 1994; Marcus, Chem. Soc. Rev. 1993
  - Physical params: CRC Handbook of Chemistry and Physics, 97th ed.
  - Base pKa: Bordwell pKa table (DMSO scale -> approximate THF scale)
  - Lewis acidity: Pearson, JACS 1963; Parr & Pearson, JACS 1983
  - Sterimol A-values: Winstein & Holness, JACS 1955
"""

from pathlib import Path

# ==================== Project paths ====================
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_CSV = DATA_DIR / "data.csv"
CLEAN_DIR = DATA_DIR / "clean_v5"
AUDIT_DIR = CLEAN_DIR / "audit"
INTERIM_DIR = DATA_DIR / "interim_v5"

# ==================== Reaxys column names ====================
REAXYS_COLS = [
    "Reaction ID",
    "Reaction",              # reaction SMILES (reactants>>products)
    "Reactant",
    "Product",               # product name
    "Product.1",             # product name (named product)
    "Bin",
    "Record Type",
    "Reaction Type",
    "Named Reaction",
    "Yield (numerical)",
    "Yield (optical)",       # ee/dr data
    "Reagent",
    "Catalyst",
    "Solvent (Reaction Details)",
    "Temperature (Reaction Details) [C]",
    "Time (Reaction Details) [h]",
    "Pressure (Reaction Details) [Torr]",
    "pH-Value (Reaction Details)",
    "Other Conditions",
    "References",
    "Number of Reaction Steps",
    "Number of Stages",
]

# ==================== Auxiliary SMARTS patterns ====================
# Each pattern matches the chiral auxiliary on the REACTANT (ketone/acyl) side.

AUXILIARY_SMARTS = {
    # Evans: 4-substituted oxazolidin-2-one
    # 5-membered ring: C4(R)-CH2-O-C(=O)-N
    "evans": "[C:1]1([*])COC(=O)N1",

    # Crimmins thiazolidinethione: 4-substituted thiazolidine-2-thione
    # 5-membered ring: C4(R)-CH2-S-C(=S)-N
    "crimmins_thione": "[C:1]1([*])CSC(=S)N1",

    # Crimmins oxazolidinethione: 4-substituted oxazolidine-2-thione
    # 5-membered ring: C4(R)-CH2-O-C(=S)-N
    "crimmins_oxathione": "[C:1]1([*])COC(=S)N1",

    # Oppolzer camphorsultam: bornane-10,2-sultam
    # 5-membered sultam ring: N-C-C-C-S(=O)2 fused to norbornane
    "oppolzer": "[NX3]1[C][C]CS1(=O)=O",

    # Myers pseudoephedrine amide (N-acyl form, broadened V5)
    # Covers N-methyl, N-benzyl, and OH-protected variants
    "myers": "[CX3](=[OX1])N[CH]([CH3])[CH]([OH,O])c",

    # Super Quat: 4,4-disubstituted oxazolidinone (gem-disubstituted)
    "super_quat": "[C:1]1([*])([*])COC(=O)N1",

    # Abiko: N-sulfonyl amino alcohol ester (V5 new)
    # C(=O)-O-CH-CH-N(R)-SO2Ar
    "abiko": "[CX3](=[OX1])O[CH][CH]N([*])S(=O)(=O)",

    # Menthyl ester: L-menthol / 8-phenylmenthol chiral ester (V5 new)
    "menthyl_ester": "C(=O)OC1CC(C(C)C)CCC1C",

    # Borneol ester: isoborneol / borneol chiral ester (V5 new)
    "borneol_ester": "C(=O)OC1CC2CCC1(C)C2(C)C",

    # Oxazoline: Meyers' 2-oxazoline auxiliary (V5 new)
    # 5-membered ring with C=N, different chelation from oxazolidinone
    "oxazoline": "C1=N[CH]([*])CO1",
}

# Broader catch-all: any 5-membered heterocyclic amide/thioamide with chiral C
# Used ONLY if none of the specific patterns match
GENERIC_AUXILIARY_SMARTS = [
    # Cyclic N-acyl with chiral center
    "[C:1]1([*])[CH2][O,S]C(=[O,S])N1",
    # Sultam pattern (broader)
    "[N:1]([*])S(=O)(=O)[C,c]",
]

# ==================== Aldol product SMARTS ====================
# Beta-hydroxy carbonyl (substrate-auxiliary type)
# :1 = Cb (OH-bearing, from aldehyde)
# :2 = Ca (alpha-carbon, from enolate)
ALDOL_PRODUCT_SMARTS_AUX = "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])[NX3]"

# Beta-hydroxy ester (V5: for menthyl/borneol/abiko chiral ester auxiliaries)
ALDOL_PRODUCT_SMARTS_ESTER = "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])[CX3](=[OX1])[OX2]"

# Oxazoline-type product (V5: Meyers' oxazoline auxiliary)
ALDOL_PRODUCT_SMARTS_OXAZ = "[CX4:1]([OX2H1])([#6])[CX4:2]([#6])C1=N[CH]CO1"

# Generic beta-hydroxy carbonyl (any aldol)
ALDOL_PRODUCT_SMARTS_GENERIC = "[CX4]([OX2H1])([#6])[CX4]([#6])[CX3](=[OX1])"

# Dehydration product (alpha,beta-unsaturated carbonyl) - to EXCLUDE
DEHYDRATION_SMARTS = "[CX3](=[OX1])/[CX3]=[CX3]"

# Aldehyde functional group (for reactant classification)
ALDEHYDE_SMARTS = "[CX3H1](=O)"

# Ynamide exclusion (V5: keteniminium mechanism, not substrate-controlled ZT)
YNAMIDE_EXCLUDE_SMARTS = "[N]C#C"

# ==================== Chiral catalyst exclusion ====================
# Keywords in Catalyst/Reagent fields that indicate chiral catalysis (NOT substrate control)
CHIRAL_CATALYST_KEYWORDS = [
    # Organocatalysts
    "proline", "l-proline", "d-proline",
    "prolinamide", "prolinol",
    "cinchona", "cinchonidine", "cinchonine",
    "quinine", "quinidine",
    "thiourea catalyst", "squaramide",
    # Chiral ligand names
    "binap", "binol", "segphos", "difluorphos",
    "salen", "jacobsen",
    "box", "bisoxazoline", "pybox",
    "cbs", "corey-bakshi-shibata",
    "taddol", "duphos",
    # Chiral metal complexes
    "chiral lewis acid", "asymmetric catalyst",
    # Phase transfer
    "chiral phase transfer", "cinchoninium",
]

# Named Reaction patterns indicating chiral catalysis (exclude these)
CHIRAL_CATALYSIS_NAMED_REACTIONS = [
    "proline-catalyzed",
    "organocatalytic aldol",
    "asymmetric catalytic aldol",
    "mukaiyama aldol",  # typically uses chiral Lewis acid catalyst
]

# ==================== Solvent parameters ====================
# Kamlet-Taft + physical (from V3, unchanged)
SOLVENT_DB = {
    "dichloromethane": {
        "alpha": 0.13, "beta": 0.10, "pi_star": 0.82, "ET30": 40.7,
        "epsilon": 8.93, "viscosity_cP": 0.44, "bp_C": 40.0,
    },
    "tetrahydrofuran": {
        "alpha": 0.00, "beta": 0.55, "pi_star": 0.58, "ET30": 37.4,
        "epsilon": 7.52, "viscosity_cP": 0.46, "bp_C": 66.0,
    },
    "ethyl acetate": {
        "alpha": 0.00, "beta": 0.45, "pi_star": 0.55, "ET30": 38.1,
        "epsilon": 6.02, "viscosity_cP": 0.45, "bp_C": 77.1,
    },
    "diethyl ether": {
        "alpha": 0.00, "beta": 0.47, "pi_star": 0.27, "ET30": 34.5,
        "epsilon": 4.27, "viscosity_cP": 0.22, "bp_C": 34.6,
    },
    "toluene": {
        "alpha": 0.00, "beta": 0.11, "pi_star": 0.54, "ET30": 33.9,
        "epsilon": 2.38, "viscosity_cP": 0.59, "bp_C": 110.6,
    },
    "pentane": {
        "alpha": 0.00, "beta": 0.00, "pi_star": -0.08, "ET30": 31.1,
        "epsilon": 1.84, "viscosity_cP": 0.24, "bp_C": 36.1,
    },
    "hexane": {
        "alpha": 0.00, "beta": 0.00, "pi_star": -0.04, "ET30": 31.0,
        "epsilon": 1.88, "viscosity_cP": 0.33, "bp_C": 69.0,
    },
    "heptane": {
        "alpha": 0.00, "beta": 0.00, "pi_star": -0.02, "ET30": 31.1,
        "epsilon": 1.92, "viscosity_cP": 0.42, "bp_C": 98.4,
    },
    "acetonitrile": {
        "alpha": 0.19, "beta": 0.40, "pi_star": 0.75, "ET30": 45.6,
        "epsilon": 36.6, "viscosity_cP": 0.37, "bp_C": 82.0,
    },
    "methanol": {
        "alpha": 0.98, "beta": 0.66, "pi_star": 0.60, "ET30": 55.4,
        "epsilon": 32.7, "viscosity_cP": 0.54, "bp_C": 64.7,
    },
    "ethanol": {
        "alpha": 0.86, "beta": 0.75, "pi_star": 0.54, "ET30": 51.9,
        "epsilon": 24.5, "viscosity_cP": 1.07, "bp_C": 78.4,
    },
    "isopropanol": {
        "alpha": 0.76, "beta": 0.84, "pi_star": 0.48, "ET30": 48.4,
        "epsilon": 19.9, "viscosity_cP": 2.04, "bp_C": 82.6,
    },
    "water": {
        "alpha": 1.17, "beta": 0.47, "pi_star": 1.09, "ET30": 63.1,
        "epsilon": 80.1, "viscosity_cP": 1.00, "bp_C": 100.0,
    },
    "chloroform": {
        "alpha": 0.20, "beta": 0.10, "pi_star": 0.58, "ET30": 39.1,
        "epsilon": 4.81, "viscosity_cP": 0.54, "bp_C": 61.2,
    },
    "1,2-dichloroethane": {
        "alpha": 0.00, "beta": 0.10, "pi_star": 0.81, "ET30": 41.3,
        "epsilon": 10.4, "viscosity_cP": 0.84, "bp_C": 83.5,
    },
    "dimethylformamide": {
        "alpha": 0.00, "beta": 0.69, "pi_star": 0.88, "ET30": 43.2,
        "epsilon": 36.7, "viscosity_cP": 0.92, "bp_C": 153.0,
    },
    "dmso": {
        "alpha": 0.00, "beta": 0.76, "pi_star": 1.00, "ET30": 45.1,
        "epsilon": 46.7, "viscosity_cP": 1.99, "bp_C": 189.0,
    },
    "acetone": {
        "alpha": 0.08, "beta": 0.43, "pi_star": 0.71, "ET30": 42.2,
        "epsilon": 20.5, "viscosity_cP": 0.32, "bp_C": 56.1,
    },
    "benzene": {
        "alpha": 0.00, "beta": 0.10, "pi_star": 0.59, "ET30": 34.3,
        "epsilon": 2.28, "viscosity_cP": 0.65, "bp_C": 80.1,
    },
    "carbon tetrachloride": {
        "alpha": 0.00, "beta": 0.10, "pi_star": 0.28, "ET30": 32.4,
        "epsilon": 2.24, "viscosity_cP": 0.97, "bp_C": 76.7,
    },
    "dioxane": {
        "alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0,
        "epsilon": 2.21, "viscosity_cP": 1.54, "bp_C": 101.1,
    },
    "pyridine": {
        "alpha": 0.00, "beta": 0.64, "pi_star": 0.87, "ET30": 40.5,
        "epsilon": 12.3, "viscosity_cP": 0.94, "bp_C": 115.3,
    },
    "1-methyl-pyrrolidin-2-one": {
        "alpha": 0.00, "beta": 0.77, "pi_star": 0.92, "ET30": 42.2,
        "epsilon": 32.2, "viscosity_cP": 1.67, "bp_C": 202.0,
    },
    "methyl tert-butyl ether": {
        "alpha": 0.00, "beta": 0.55, "pi_star": 0.27, "ET30": 34.7,
        "epsilon": 4.50, "viscosity_cP": 0.36, "bp_C": 55.2,
    },
    "2-methyltetrahydrofuran": {
        "alpha": 0.00, "beta": 0.53, "pi_star": 0.53, "ET30": 36.5,
        "epsilon": 6.97, "viscosity_cP": 0.47, "bp_C": 80.0,
    },
    "nitromethane": {
        "alpha": 0.22, "beta": 0.06, "pi_star": 0.85, "ET30": 46.3,
        "epsilon": 35.9, "viscosity_cP": 0.63, "bp_C": 101.2,
    },
    "1,4-dioxane": {
        "alpha": 0.00, "beta": 0.37, "pi_star": 0.55, "ET30": 36.0,
        "epsilon": 2.21, "viscosity_cP": 1.54, "bp_C": 101.1,
    },
    "cyclopentyl methyl ether": {
        "alpha": 0.00, "beta": 0.52, "pi_star": 0.30, "ET30": 34.8,
        "epsilon": 4.76, "viscosity_cP": 0.55, "bp_C": 106.0,
    },
}

SOLVENT_ALIASES = {
    "1,2-dichloro-ethane": "1,2-dichloroethane",
    "n,n-dimethylformamide": "dimethylformamide",
    "dmf": "dimethylformamide",
    "thf": "tetrahydrofuran",
    "dcm": "dichloromethane",
    "ch2cl2": "dichloromethane",
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
    "dimethylsulfoxide": "dmso",
    "petroleum ether": "hexane",
    "pet ether": "hexane",
    "ligroin": "hexane",
}

METAL_DEFAULT_SOLVENT = {
    "B": "dichloromethane", "Ti": "dichloromethane", "Sn": "dichloromethane",
    "Li": "tetrahydrofuran", "Mg": "ethyl acetate", "Zn": "dichloromethane",
    "Cu": "dichloromethane", "Zr": "dichloromethane",
}

# ==================== Base / Metal / Activator properties ====================
BASE_PROPERTIES = {
    "DIPEA":      {"pKa": 35.0, "steric_A": 2.5, "nucleophilicity": 12.0},
    "Et3N":       {"pKa": 33.0, "steric_A": 1.8, "nucleophilicity": 17.1},
    "LiHMDS":     {"pKa": 30.0, "steric_A": 3.0, "nucleophilicity": 5.0},
    "NaHMDS":     {"pKa": 30.0, "steric_A": 3.0, "nucleophilicity": 5.5},
    "LDA":        {"pKa": 36.0, "steric_A": 2.8, "nucleophilicity": 4.0},
    "KHMDS":      {"pKa": 30.0, "steric_A": 3.0, "nucleophilicity": 6.0},
    "other_base": {"pKa": 25.0, "steric_A": 1.5, "nucleophilicity": 10.0},
    "no_base":    {"pKa": 0.0,  "steric_A": 0.0, "nucleophilicity": 0.0},
}

METAL_PROPERTIES = {
    "B":       {"coordination_num": 4, "ionic_radius_pm": 27,  "pearson_hardness": 8.0},
    "Ti":      {"coordination_num": 6, "ionic_radius_pm": 86,  "pearson_hardness": 3.4},
    "Sn":      {"coordination_num": 6, "ionic_radius_pm": 93,  "pearson_hardness": 5.0},
    "Li":      {"coordination_num": 4, "ionic_radius_pm": 76,  "pearson_hardness": 35.1},
    "Mg":      {"coordination_num": 6, "ionic_radius_pm": 72,  "pearson_hardness": 32.5},
    "Zn":      {"coordination_num": 4, "ionic_radius_pm": 74,  "pearson_hardness": 10.9},
    "Cu":      {"coordination_num": 4, "ionic_radius_pm": 73,  "pearson_hardness": 8.3},
    "Zr":      {"coordination_num": 8, "ionic_radius_pm": 84,  "pearson_hardness": 3.2},
    "none":    {"coordination_num": 0, "ionic_radius_pm": 0,   "pearson_hardness": 0.0},
    "unknown": {"coordination_num": 0, "ionic_radius_pm": 0,   "pearson_hardness": 0.0},
}

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

# Metal identification from reagent/catalyst strings
METAL_KEYWORDS = {
    "B":  ["boron", "9-bbn", "bu2botf", "chx2bcl", "ipc2bcl", "dibutylboron",
           "dicyclohexylboron", "diisopinocampheylboron", "triflylborane"],
    "Ti": ["titanium", "ticl4", "ticl3", "ti(oi-pr)4", "titanium tetrachloride",
           "titanium(iv)"],
    "Sn": ["tin", "sn(otf)2", "stannous", "dibutyltin"],
    "Li": ["lithium", "n-butyllithium", "n-buli", "lda", "lihmds",
           "lithium diisopropylamide", "lithium bis(trimethylsilyl)amide"],
    "Mg": ["magnesium", "mgcl2", "mgbr2", "grignard"],
    "Zn": ["zinc", "zncl2", "znbr2", "diethylzinc"],
    "Cu": ["copper", "cu(otf)2", "cucl2", "cuso4"],
    "Zr": ["zirconium", "zrcl4", "cp2zrcl2"],
}

BASE_CATEGORIES = ["DIPEA", "Et3N", "LiHMDS", "NaHMDS", "LDA", "KHMDS", "other_base", "no_base"]
METAL_CATEGORIES = ["B", "Cu", "Li", "Mg", "Sn", "Ti", "Zn", "Zr", "none", "unknown"]
ACTIVATOR_CATEGORIES = [
    "Bu2BOTf", "Chx2BCl", "Ipc2BCl", "9BBN_OTf", "TiCl4",
    "Sn_OTf2", "MgCl2", "BF3_OEt2", "other_activator",
]

SOLVENT_FEATURE_NAMES = [
    "solvent_alpha", "solvent_beta", "solvent_pi_star", "solvent_ET30",
    "solvent_epsilon", "solvent_viscosity", "solvent_bp", "solvent_known",
]
