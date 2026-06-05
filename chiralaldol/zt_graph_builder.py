"""Zimmerman-Traxler Transition State Graph Builder.

Constructs a pseudo-transition state graph representing the 6-membered
chair-like Zimmerman-Traxler (ZT) TS from reactant SMILES and conditions.

Chemistry:
  Evans aldol proceeds through a chelated, metal-mediated 6-membered TS:

        O(aux)                   O(aux)
       / \\                      / \\
      C   M  ← metal            C   M
      ‖   |                     ‖   |
      C   O                     C   O
       \\ / \\                    \\ / \\
        C   R(ald)  [Evans syn]   C   H   [Evans anti]
        |                         |
        H                         R(ald)

  The graph encodes:
  1. The 6 core ring atoms (M, O_metal, C_enolate, C=C, O_ald, C_ald)
  2. Substituent subgraphs attached to ring atoms
  3. Node features: atom type, axial/equatorial, Sterimol, CIP
  4. Edge features: bond type, ring membership

Reference:
  Zimmerman & Traxler, JACS 1957, 79, 1920
  Bernardi et al., JACS 2010, 132, 9982 (Evans aldol TS DFT)
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
from rdkit import Chem, RDLogger

from .utils import ACYL_ALPHA_SMARTS

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)


# ═══════════════════════════ Constants ═══════════════════════════

class ZTNodeType(IntEnum):
    """Node types in the ZT transition state graph."""
    METAL = 0           # M (Ti, B, Li, Mg, Sn, Zn)
    O_METAL = 1         # Oxygen coordinated to metal (from enolate C=O → C-O⁻)
    C_CARBONYL = 2      # Carbonyl carbon (C=C-O⁻, formerly C(=O))
    C_ALPHA = 3         # Alpha carbon (enolate C=C, bears R1 from aux)
    O_ALDEHYDE = 4      # Aldehyde oxygen (coordinated to metal)
    C_ALDEHYDE = 5      # Aldehyde carbon (bears R_ald)
    # Substituent atoms
    SUBSTITUENT = 6     # Generic substituent atom


class ZTEdgeType(IntEnum):
    """Edge types in the ZT TS graph."""
    RING_SINGLE = 0     # Single bond in 6-membered ring
    RING_DOUBLE = 1     # Double bond in ring (C=C enolate)
    RING_COORD = 2      # Metal coordination bond (M-O)
    SUBSTITUENT = 3     # Bond to substituent
    WITHIN_SUB = 4      # Bond within substituent subgraph


# Metal properties for node features
METAL_PROPERTIES = {
    "Ti":  {"ionic_radius": 0.605, "coord_num": 6, "hardness": 3.37, "charge": 4},
    "B":   {"ionic_radius": 0.27,  "coord_num": 4, "hardness": 4.29, "charge": 3},
    "Li":  {"ionic_radius": 0.76,  "coord_num": 4, "hardness": 2.39, "charge": 1},
    "Mg":  {"ionic_radius": 0.72,  "coord_num": 6, "hardness": 4.37, "charge": 2},
    "Sn":  {"ionic_radius": 0.69,  "coord_num": 4, "hardness": 3.05, "charge": 2},
    "Zn":  {"ionic_radius": 0.74,  "coord_num": 4, "hardness": 4.94, "charge": 2},
    "Zr":  {"ionic_radius": 0.72,  "coord_num": 6, "hardness": 3.21, "charge": 4},
    "Cu":  {"ionic_radius": 0.73,  "coord_num": 4, "hardness": 3.25, "charge": 2},
    "none":    {"ionic_radius": 0.0,  "coord_num": 0, "hardness": 0.0, "charge": 0},
    "unknown": {"ionic_radius": 0.0,  "coord_num": 0, "hardness": 0.0, "charge": 0},
}

# Evans auxiliary SMARTS for matching C4 and the acyl chain
EVANS_KETONE_SMARTS = Chem.MolFromSmarts("[C:1]1([*])COC(=O)N1")
# Primary: aldehyde R-CHO; fallback: any carbonyl C(=O) for ketone/ester electrophiles
ALDEHYDE_SMARTS = Chem.MolFromSmarts("[CX3H1:1](=[OX1:2])")
CARBONYL_SMARTS = Chem.MolFromSmarts("[CX3:1](=[OX1:2])")


# ═══════════════════════════ Data Structures ═══════════════════════════

@dataclass
class ZTGraph:
    """A Zimmerman-Traxler transition state graph."""
    # Node arrays
    node_types: np.ndarray          # (n_nodes,) ZTNodeType enum values
    node_features: np.ndarray       # (n_nodes, n_node_feat) continuous features
    # Edge arrays (COO format)
    edge_index: np.ndarray          # (2, n_edges) source, target indices
    edge_types: np.ndarray          # (n_edges,) ZTEdgeType enum values
    edge_features: np.ndarray       # (n_edges, n_edge_feat) continuous features
    # Metadata
    n_ring_atoms: int = 6           # Always 6 for ZT
    metal: str = ""
    auxiliary_type: str = ""
    status: str = "success"
    # Substituent mapping
    ring_atom_indices: list = field(default_factory=list)  # indices of the 6 ring atoms


# ═══════════════════════════ Substituent Feature Extraction ═══════════════════════════

def _get_substituent_smiles(mol, anchor_idx, exclude_indices):
    """Extract SMILES of the substituent attached to anchor_idx, excluding certain atoms."""
    visited = set(exclude_indices)
    queue = [anchor_idx]
    sub_atoms = []

    while queue:
        atom_idx = queue.pop(0)
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        sub_atoms.append(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited:
                queue.append(n_idx)

    if not sub_atoms:
        return None, []

    return sub_atoms, sub_atoms


def _compute_substituent_features(mol, sub_atom_indices):
    """Compute features for a substituent subgraph.

    Returns dict with: n_heavy, n_aromatic, n_heteroatom, max_atomic_num,
                        has_halogen, molecular_weight (of fragment)
    """
    if not sub_atom_indices:
        return {
            "n_heavy": 0, "n_aromatic": 0, "n_heteroatom": 0,
            "max_atomic_num": 0, "has_halogen": 0, "mw": 0.0,
        }

    n_heavy = len(sub_atom_indices)
    n_aromatic = 0
    n_heteroatom = 0
    max_atomic_num = 0
    has_halogen = 0
    total_mass = 0.0

    for idx in sub_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        anum = atom.GetAtomicNum()
        total_mass += atom.GetMass()
        if anum > max_atomic_num:
            max_atomic_num = anum
        if atom.GetIsAromatic():
            n_aromatic += 1
        if anum not in (1, 6):  # not H or C
            n_heteroatom += 1
        if anum in (9, 17, 35, 53):  # F, Cl, Br, I
            has_halogen = 1

    return {
        "n_heavy": n_heavy,
        "n_aromatic": n_aromatic,
        "n_heteroatom": n_heteroatom,
        "max_atomic_num": max_atomic_num,
        "has_halogen": has_halogen,
        "mw": total_mass,
    }


def _atom_features(atom):
    """Compute per-atom features for substituent atoms."""
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
        atom.GetMass() / 100.0,  # normalized
    ]

ATOM_FEAT_DIM = 7


# ═══════════════════════════ Core Builder ═══════════════════════════

def build_zt_graph_evans(ketone_smi: str, aldehyde_smi: str,
                          metal: str = "none",
                          activator: str = "",
                          include_substituent_graph: bool = True) -> ZTGraph:
    """Build a Zimmerman-Traxler TS graph for an Evans aldol reaction.

    Args:
        ketone_smi: SMILES of the ketone (with Evans oxazolidinone)
        aldehyde_smi: SMILES of the aldehyde
        metal: Metal type (Ti, B, Li, Mg, Sn, Zn, none, unknown)
        activator: Activator type (e.g., Bu2BOTf, TiCl4)
        include_substituent_graph: If True, include full substituent subgraphs

    Returns:
        ZTGraph object
    """
    # --- Parse molecules ---
    ket_mol = Chem.MolFromSmiles(ketone_smi)
    ald_mol = Chem.MolFromSmiles(aldehyde_smi)

    if ket_mol is None or ald_mol is None:
        return ZTGraph(
            node_types=np.array([], dtype=np.int64),
            node_features=np.zeros((0, 0)),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_types=np.array([], dtype=np.int64),
            edge_features=np.zeros((0, 0)),
            status="parse_fail",
        )

    # --- Match Evans auxiliary on ketone ---
    aux_matches = ket_mol.GetSubstructMatches(EVANS_KETONE_SMARTS)
    if not aux_matches:
        return ZTGraph(
            node_types=np.array([], dtype=np.int64),
            node_features=np.zeros((0, 0)),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_types=np.array([], dtype=np.int64),
            edge_features=np.zeros((0, 0)),
            status="no_evans_match",
        )

    # --- Match acyl alpha position ---
    # Primary: CH2/CH adjacent to C(=O)-N (standard Evans propionyl)
    # Fallback: any carbon adjacent to C(=O)-N (covers ynamides, vinyl, etc.)
    acyl_matches = ket_mol.GetSubstructMatches(ACYL_ALPHA_SMARTS)
    if not acyl_matches:
        # Broader pattern: any carbon bonded to acyl C(=O)-N
        broad_acyl = Chem.MolFromSmarts("[#6:1]-[CX3:2](=[OX1:3])-[NX3:4]")
        acyl_matches = ket_mol.GetSubstructMatches(broad_acyl)
        if not acyl_matches:
            # Even broader: look for any N-acyl pattern including C#C-N
            # For ynamides (C#C-N-oxaz), the "alpha" is the triple-bond carbon
            n_acyl = Chem.MolFromSmarts("[#6:1][NX3:4]([#6])[CX4]")  # N in ring
            acyl_matches_alt = ket_mol.GetSubstructMatches(n_acyl)
            if not acyl_matches_alt:
                return ZTGraph(
                    node_types=np.array([], dtype=np.int64),
                    node_features=np.zeros((0, 0)),
                    edge_index=np.zeros((2, 0), dtype=np.int64),
                    edge_types=np.array([], dtype=np.int64),
                    edge_features=np.zeros((0, 0)),
                    status="no_acyl_match",
                )
            # For ynamide fallback: use the first carbon bonded to N as "alpha"
            # and find the nearest C=O in the ring
            alpha_idx = acyl_matches_alt[0][0]
            n_idx = acyl_matches_alt[0][1]
            # Find C=O in the Evans ring
            co_pat = Chem.MolFromSmarts("[CX3:1](=[OX1:2])")
            co_matches = ket_mol.GetSubstructMatches(co_pat)
            if co_matches:
                carbonyl_c_idx, carbonyl_o_idx = co_matches[0]
            else:
                return ZTGraph(
                    node_types=np.array([], dtype=np.int64),
                    node_features=np.zeros((0, 0)),
                    edge_index=np.zeros((2, 0), dtype=np.int64),
                    edge_types=np.array([], dtype=np.int64),
                    edge_features=np.zeros((0, 0)),
                    status="no_acyl_match",
                )
        else:
            alpha_idx, carbonyl_c_idx, carbonyl_o_idx, n_idx = acyl_matches[0]
    else:
        alpha_idx, carbonyl_c_idx, carbonyl_o_idx, n_idx = acyl_matches[0]

    # --- Match aldehyde (or ketone/ester electrophile as fallback) ---
    ald_matches = ald_mol.GetSubstructMatches(ALDEHYDE_SMARTS)
    if not ald_matches:
        ald_matches = ald_mol.GetSubstructMatches(CARBONYL_SMARTS)
    if not ald_matches:
        return ZTGraph(
            node_types=np.array([], dtype=np.int64),
            node_features=np.zeros((0, 0)),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_types=np.array([], dtype=np.int64),
            edge_features=np.zeros((0, 0)),
            status="no_aldehyde_match",
        )

    ald_c_idx, ald_o_idx = ald_matches[0]

    # --- Infer metal from activator if metal is "none" ---
    effective_metal = metal
    if effective_metal in ("none", "unknown", ""):
        if "B" in activator or "BOTf" in activator or "BCl" in activator or "BBN" in activator:
            effective_metal = "B"
        elif "Ti" in activator:
            effective_metal = "Ti"
        elif "Sn" in activator:
            effective_metal = "Sn"
        elif "Mg" in activator:
            effective_metal = "Mg"

    metal_props = METAL_PROPERTIES.get(effective_metal, METAL_PROPERTIES["none"])

    # ═══════════════════════════ Build ZT Ring ═══════════════════════════
    #
    # ZT 6-membered ring connectivity (chair-like):
    #   0: M (metal)
    #   1: O_metal (enolate oxygen, coord to M)
    #   2: C_carbonyl (formerly C=O, now C-O⁻, part of enolate C=C)
    #   3: C_alpha (alpha carbon, enolate C=C terminus, bears R1)
    #   4: C_aldehyde (aldehyde carbon, bears R_ald)
    #   5: O_aldehyde (aldehyde oxygen, coord to M)
    #
    # Ring edges: M-O_metal, O_metal-C_carbonyl, C_carbonyl=C_alpha,
    #             C_alpha-C_aldehyde (new C-C bond), C_aldehyde-O_aldehyde,
    #             O_aldehyde-M

    # --- Substituent analysis ---
    # R1: substituent on C_alpha (from acyl chain, e.g., -CH3 for propionyl)
    # Get neighbors of alpha carbon excluding carbonyl
    alpha_atom = ket_mol.GetAtomWithIdx(alpha_idx)
    r1_atoms = []
    ring_ketone_atoms = {alpha_idx, carbonyl_c_idx, carbonyl_o_idx, n_idx}
    # Add Evans ring atoms
    for m in aux_matches:
        ring_ketone_atoms.update(m)
    for neighbor in alpha_atom.GetNeighbors():
        n_idx_neighbor = neighbor.GetIdx()
        if n_idx_neighbor not in ring_ketone_atoms:
            _, sub_atoms = _get_substituent_smiles(ket_mol, n_idx_neighbor, ring_ketone_atoms)
            r1_atoms.extend(sub_atoms)

    r1_feats = _compute_substituent_features(ket_mol, r1_atoms)

    # R_ald: substituent on aldehyde carbon
    ald_c_atom = ald_mol.GetAtomWithIdx(ald_c_idx)
    r_ald_atoms = []
    ring_ald_atoms = {ald_c_idx, ald_o_idx}
    for neighbor in ald_c_atom.GetNeighbors():
        n_idx_neighbor = neighbor.GetIdx()
        if n_idx_neighbor not in ring_ald_atoms and neighbor.GetAtomicNum() != 1:
            _, sub_atoms = _get_substituent_smiles(ald_mol, n_idx_neighbor, ring_ald_atoms)
            r_ald_atoms.extend(sub_atoms)

    r_ald_feats = _compute_substituent_features(ald_mol, r_ald_atoms)

    # Auxiliary substituent (C4 substituent on Evans ring)
    c4_idx = aux_matches[0][0]  # C4 from SMARTS match
    c4_atom = ket_mol.GetAtomWithIdx(c4_idx)
    aux_sub_atoms = []
    evans_ring_atoms = set(aux_matches[0])
    for neighbor in c4_atom.GetNeighbors():
        n_idx_neighbor = neighbor.GetIdx()
        if n_idx_neighbor not in evans_ring_atoms and n_idx_neighbor != alpha_idx:
            _, sub_atoms = _get_substituent_smiles(ket_mol, n_idx_neighbor,
                                                    evans_ring_atoms | {alpha_idx})
            aux_sub_atoms.extend(sub_atoms)

    aux_sub_feats = _compute_substituent_features(ket_mol, aux_sub_atoms)

    # ═══════════════════════════ Node Features ═══════════════════════════
    #
    # Per node: [node_type_onehot(7), metal_props(4), substituent_feats(6),
    #            is_ring(1), axial_equatorial(2)]
    # Total: 7 + 4 + 6 + 1 + 2 = 20

    NODE_FEAT_DIM = 20

    nodes = []
    node_types_list = []

    def _make_ring_node(ntype, sub_feats, is_axial=0):
        """Create feature vector for a ring node."""
        # Node type one-hot (7 types)
        type_oh = [0] * 7
        type_oh[int(ntype)] = 1
        # Metal properties (same for all nodes in this reaction)
        mp = [metal_props["ionic_radius"], metal_props["coord_num"] / 6.0,
              metal_props["hardness"] / 5.0, metal_props["charge"] / 4.0]
        # Substituent features (normalized)
        sf = [sub_feats["n_heavy"] / 20.0, sub_feats["n_aromatic"] / 10.0,
              sub_feats["n_heteroatom"] / 5.0, sub_feats["max_atomic_num"] / 53.0,
              float(sub_feats["has_halogen"]), sub_feats["mw"] / 200.0]
        # Ring membership + axial/equatorial
        ring = [1.0]
        ax_eq = [float(is_axial), 1.0 - float(is_axial)]
        return type_oh + mp + sf + ring + ax_eq

    empty_sub = {"n_heavy": 0, "n_aromatic": 0, "n_heteroatom": 0,
                 "max_atomic_num": 0, "has_halogen": 0, "mw": 0.0}

    # Node 0: Metal
    nodes.append(_make_ring_node(ZTNodeType.METAL, empty_sub, is_axial=0))
    node_types_list.append(ZTNodeType.METAL)

    # Node 1: O_metal (enolate oxygen)
    nodes.append(_make_ring_node(ZTNodeType.O_METAL, empty_sub, is_axial=0))
    node_types_list.append(ZTNodeType.O_METAL)

    # Node 2: C_carbonyl
    nodes.append(_make_ring_node(ZTNodeType.C_CARBONYL, empty_sub, is_axial=0))
    node_types_list.append(ZTNodeType.C_CARBONYL)

    # Node 3: C_alpha (bears R1 — axial in Evans syn TS)
    nodes.append(_make_ring_node(ZTNodeType.C_ALPHA, r1_feats, is_axial=1))
    node_types_list.append(ZTNodeType.C_ALPHA)

    # Node 4: C_aldehyde (bears R_ald — equatorial in Evans syn TS)
    nodes.append(_make_ring_node(ZTNodeType.C_ALDEHYDE, r_ald_feats, is_axial=0))
    node_types_list.append(ZTNodeType.C_ALDEHYDE)

    # Node 5: O_aldehyde
    nodes.append(_make_ring_node(ZTNodeType.O_ALDEHYDE, empty_sub, is_axial=0))
    node_types_list.append(ZTNodeType.O_ALDEHYDE)

    ring_node_count = 6

    # ═══════════════════════════ Substituent Subgraphs ═══════════════════════════

    sub_node_offset = ring_node_count
    sub_edges_src = []
    sub_edges_dst = []
    sub_edge_types = []

    if include_substituent_graph:
        # Add R1 substituent atoms
        for i, atom_idx in enumerate(r1_atoms):
            atom = ket_mol.GetAtomWithIdx(atom_idx)
            feat = [0] * 7  # node type one-hot
            feat[int(ZTNodeType.SUBSTITUENT)] = 1
            feat += [metal_props["ionic_radius"], metal_props["coord_num"] / 6.0,
                     metal_props["hardness"] / 5.0, metal_props["charge"] / 4.0]
            feat += _atom_features(atom)[:6]  # first 6 atom features
            feat += [0.0, 0.0, 0.0]  # pad to 20
            nodes.append(feat[:NODE_FEAT_DIM])
            node_types_list.append(ZTNodeType.SUBSTITUENT)

        # Connect R1 root to C_alpha (node 3)
        if r1_atoms:
            sub_edges_src.append(3)
            sub_edges_dst.append(sub_node_offset)
            sub_edge_types.append(ZTEdgeType.SUBSTITUENT)
            sub_edges_src.append(sub_node_offset)
            sub_edges_dst.append(3)
            sub_edge_types.append(ZTEdgeType.SUBSTITUENT)

        # Internal R1 edges
        r1_node_map = {atom_idx: sub_node_offset + i for i, atom_idx in enumerate(r1_atoms)}
        for atom_idx in r1_atoms:
            atom = ket_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                n_idx_n = neighbor.GetIdx()
                if n_idx_n in r1_node_map and n_idx_n > atom_idx:
                    src = r1_node_map[atom_idx]
                    dst = r1_node_map[n_idx_n]
                    sub_edges_src.extend([src, dst])
                    sub_edges_dst.extend([dst, src])
                    sub_edge_types.extend([ZTEdgeType.WITHIN_SUB, ZTEdgeType.WITHIN_SUB])

        sub_node_offset += len(r1_atoms)

        # Add R_ald substituent atoms
        for i, atom_idx in enumerate(r_ald_atoms):
            atom = ald_mol.GetAtomWithIdx(atom_idx)
            feat = [0] * 7
            feat[int(ZTNodeType.SUBSTITUENT)] = 1
            feat += [metal_props["ionic_radius"], metal_props["coord_num"] / 6.0,
                     metal_props["hardness"] / 5.0, metal_props["charge"] / 4.0]
            feat += _atom_features(atom)[:6]
            feat += [0.0, 0.0, 0.0]
            nodes.append(feat[:NODE_FEAT_DIM])
            node_types_list.append(ZTNodeType.SUBSTITUENT)

        # Connect R_ald root to C_aldehyde (node 4)
        if r_ald_atoms:
            sub_edges_src.append(4)
            sub_edges_dst.append(sub_node_offset)
            sub_edge_types.append(ZTEdgeType.SUBSTITUENT)
            sub_edges_src.append(sub_node_offset)
            sub_edges_dst.append(4)
            sub_edge_types.append(ZTEdgeType.SUBSTITUENT)

        # Internal R_ald edges
        r_ald_node_map = {atom_idx: sub_node_offset + i for i, atom_idx in enumerate(r_ald_atoms)}
        for atom_idx in r_ald_atoms:
            atom = ald_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                n_idx_n = neighbor.GetIdx()
                if n_idx_n in r_ald_node_map and n_idx_n > atom_idx:
                    src = r_ald_node_map[atom_idx]
                    dst = r_ald_node_map[n_idx_n]
                    sub_edges_src.extend([src, dst])
                    sub_edges_dst.extend([dst, src])
                    sub_edge_types.extend([ZTEdgeType.WITHIN_SUB, ZTEdgeType.WITHIN_SUB])

        sub_node_offset += len(r_ald_atoms)

        # Add auxiliary C4 substituent atoms
        for i, atom_idx in enumerate(aux_sub_atoms):
            atom = ket_mol.GetAtomWithIdx(atom_idx)
            feat = [0] * 7
            feat[int(ZTNodeType.SUBSTITUENT)] = 1
            feat += [metal_props["ionic_radius"], metal_props["coord_num"] / 6.0,
                     metal_props["hardness"] / 5.0, metal_props["charge"] / 4.0]
            feat += _atom_features(atom)[:6]
            feat += [0.0, 0.0, 0.0]
            nodes.append(feat[:NODE_FEAT_DIM])
            node_types_list.append(ZTNodeType.SUBSTITUENT)

        # Connect aux sub root to C_alpha (node 3) — represents 1,3-diaxial interaction
        if aux_sub_atoms:
            sub_edges_src.append(3)
            sub_edges_dst.append(sub_node_offset)
            sub_edge_types.append(ZTEdgeType.SUBSTITUENT)
            sub_edges_src.append(sub_node_offset)
            sub_edges_dst.append(3)
            sub_edge_types.append(ZTEdgeType.SUBSTITUENT)

        # Internal aux sub edges
        aux_node_map = {atom_idx: sub_node_offset + i for i, atom_idx in enumerate(aux_sub_atoms)}
        for atom_idx in aux_sub_atoms:
            atom = ket_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                n_idx_n = neighbor.GetIdx()
                if n_idx_n in aux_node_map and n_idx_n > atom_idx:
                    src = aux_node_map[atom_idx]
                    dst = aux_node_map[n_idx_n]
                    sub_edges_src.extend([src, dst])
                    sub_edges_dst.extend([dst, src])
                    sub_edge_types.extend([ZTEdgeType.WITHIN_SUB, ZTEdgeType.WITHIN_SUB])

    # ═══════════════════════════ Edges ═══════════════════════════

    # Ring edges (bidirectional)
    ring_edges = [
        (0, 1, ZTEdgeType.RING_COORD),   # M — O_metal (coordination)
        (1, 2, ZTEdgeType.RING_SINGLE),   # O_metal — C_carbonyl
        (2, 3, ZTEdgeType.RING_DOUBLE),   # C_carbonyl = C_alpha (enolate)
        (3, 4, ZTEdgeType.RING_SINGLE),   # C_alpha — C_aldehyde (new C-C bond)
        (4, 5, ZTEdgeType.RING_SINGLE),   # C_aldehyde — O_aldehyde
        (5, 0, ZTEdgeType.RING_COORD),    # O_aldehyde — M (coordination)
    ]

    edge_src = []
    edge_dst = []
    edge_type_list = []

    for src, dst, etype in ring_edges:
        edge_src.extend([src, dst])
        edge_dst.extend([dst, src])
        edge_type_list.extend([etype, etype])

    # Add substituent edges
    edge_src.extend(sub_edges_src)
    edge_dst.extend(sub_edges_dst)
    edge_type_list.extend(sub_edge_types)

    # ═══════════════════════════ Edge Features ═══════════════════════════

    EDGE_FEAT_DIM = 5  # [edge_type_onehot(5)]
    edge_features_list = []
    for etype in edge_type_list:
        ef = [0] * 5
        ef[int(etype)] = 1
        edge_features_list.append(ef)

    # ═══════════════════════════ Assemble ═══════════════════════════

    node_types_arr = np.array(node_types_list, dtype=np.int64)
    node_features_arr = np.array(nodes, dtype=np.float32)
    edge_index_arr = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_types_arr = np.array(edge_type_list, dtype=np.int64)
    edge_features_arr = np.array(edge_features_list, dtype=np.float32)

    return ZTGraph(
        node_types=node_types_arr,
        node_features=node_features_arr,
        edge_index=edge_index_arr,
        edge_types=edge_types_arr,
        edge_features=edge_features_arr,
        n_ring_atoms=6,
        metal=effective_metal,
        auxiliary_type="evans",
        status="success",
        ring_atom_indices=list(range(6)),
    )


# ═══════════════════════════ Batch Builder ═══════════════════════════

def build_zt_graphs_batch(df, ketone_col="ketone_smiles", aldehyde_col="aldehyde_smiles",
                           metal_col="metal", activator_col="activator_type",
                           aux_type_col="auxiliary_type"):
    """Build ZT graphs for a DataFrame of reactions.

    Args:
        df: DataFrame with reaction data
        ketone_col, aldehyde_col: column names for SMILES
        metal_col, activator_col: column names for conditions
        aux_type_col: column name for auxiliary type

    Returns:
        list of ZTGraph objects
    """
    graphs = []
    n_success = 0
    n_fail = 0

    for i in range(len(df)):
        row = df.iloc[i]
        aux_type = str(row.get(aux_type_col, ""))

        if aux_type == "evans":
            g = build_zt_graph_evans(
                ketone_smi=str(row[ketone_col]),
                aldehyde_smi=str(row[aldehyde_col]),
                metal=str(row.get(metal_col, "none")),
                activator=str(row.get(activator_col, "")),
            )
        else:
            # Non-Evans: placeholder graph (future extension)
            g = ZTGraph(
                node_types=np.array([], dtype=np.int64),
                node_features=np.zeros((0, 0)),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_types=np.array([], dtype=np.int64),
                edge_features=np.zeros((0, 0)),
                status=f"unsupported_aux_{aux_type}",
            )

        graphs.append(g)
        if g.status == "success":
            n_success += 1
        else:
            n_fail += 1

    logger.info(f"Built {n_success} ZT graphs ({n_fail} failures out of {len(df)})")
    return graphs
