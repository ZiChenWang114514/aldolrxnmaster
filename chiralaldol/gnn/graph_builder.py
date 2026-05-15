"""Build molecular graph representations for GNN training.

4 graph types:
  1. Reaction diff graph — atom-mapped reactant→product with bond change edges
  2. Multi-view graph — separate reactant and product graphs + condition vector
  3. 3D spatial graph — distance-based edges from conformer coordinates
  4. TS approx graph — combined reactant/product with reaction center annotation

All return torch_geometric.data.Data objects.
"""

import logging
import re
from typing import Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = logging.getLogger(__name__)

# Atom features
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_FEAT_DIM = len(ATOM_TYPES) + len(HYBRIDIZATIONS) + 7  # +7: degree, formal_charge, numHs, aromatic, ring, chiral, mass
BOND_FEAT_DIM = len(BOND_TYPES) + 3  # +3: conjugated, ring, stereo


def atom_features(atom: Chem.Atom) -> list[float]:
    """Compute atom-level features (one-hot + numeric)."""
    symbol = atom.GetSymbol()
    one_hot = [1.0 if symbol == t else 0.0 for t in ATOM_TYPES[:-1]]
    one_hot.append(1.0 if symbol not in ATOM_TYPES[:-1] else 0.0)  # "other"

    hyb = atom.GetHybridization()
    hyb_oh = [1.0 if hyb == h else 0.0 for h in HYBRIDIZATIONS]

    chiral_tag = atom.GetChiralTag()
    chiral = 1.0 if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0

    features = one_hot + hyb_oh + [
        atom.GetDegree() / 4.0,
        atom.GetFormalCharge() / 2.0,
        atom.GetTotalNumHs() / 4.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        chiral,
        atom.GetMass() / 100.0,
    ]
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    """Compute bond-level features."""
    bt = bond.GetBondType()
    bt_oh = [1.0 if bt == t else 0.0 for t in BOND_TYPES]
    features = bt_oh + [
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
        1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0.0,
    ]
    return features


def mol_to_graph(mol: Chem.Mol, coords_3d: Optional[np.ndarray] = None) -> Data:
    """Convert RDKit Mol to PyG Data object.

    Args:
        mol: RDKit molecule
        coords_3d: Optional (N, 3) array of 3D coordinates

    Returns:
        Data with x (node features), edge_index, edge_attr, pos (if 3D)
    """
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()

    # Node features
    x = torch.tensor([atom_features(mol.GetAtomWithIdx(i)) for i in range(n_atoms)],
                      dtype=torch.float)

    # Edge features (bidirectional)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=n_atoms)

    # 3D coordinates
    if coords_3d is not None and len(coords_3d) == n_atoms:
        data.pos = torch.tensor(coords_3d, dtype=torch.float)

    return data


# ========================================================================
# Graph Type 1: Reaction Difference Graph
# ========================================================================

def build_diff_graph(mapped_rxn: str, label: int, condition_vec: np.ndarray) -> Optional[Data]:
    """Build a reaction difference graph from atom-mapped SMILES.

    Nodes = product atoms with features.
    Edges = product bonds + extra "reaction center" edges for new/broken bonds.
    Node-level annotation: which atoms changed (reaction center mask).
    """
    parts = str(mapped_rxn).split(">>")
    if len(parts) != 2:
        return None

    r_mol = Chem.MolFromSmiles(parts[0])
    p_mol = Chem.MolFromSmiles(parts[1])
    if r_mol is None or p_mol is None:
        return None

    # Get bond sets using atom map numbers
    def get_bond_set(mol):
        bonds = set()
        for b in mol.GetBonds():
            a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomMapNum()
            a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomMapNum()
            if a1 > 0 and a2 > 0:
                bonds.add((min(a1, a2), max(a1, a2)))
        return bonds

    r_bonds = get_bond_set(r_mol)
    p_bonds = get_bond_set(p_mol)
    new_bonds = p_bonds - r_bonds  # bonds formed
    broken_bonds = r_bonds - p_bonds  # bonds broken

    # Build product graph
    n_atoms = p_mol.GetNumAtoms()
    x_list = []
    map2idx = {}

    for i in range(n_atoms):
        atom = p_mol.GetAtomWithIdx(i)
        feat = atom_features(atom)
        map_num = atom.GetAtomMapNum()
        map2idx[map_num] = i

        # Add reaction center annotation
        is_reaction_center = 0.0
        if map_num > 0:
            for nb in new_bonds | broken_bonds:
                if map_num in nb:
                    is_reaction_center = 1.0
                    break
        feat.append(is_reaction_center)
        x_list.append(feat)

    x = torch.tensor(x_list, dtype=torch.float)

    # Edges: product bonds + annotated
    edge_index = []
    edge_attr = []
    for bond in p_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # Annotate: is this a newly formed bond?
        a1_map = p_mol.GetAtomWithIdx(i).GetAtomMapNum()
        a2_map = p_mol.GetAtomWithIdx(j).GetAtomMapNum()
        is_new = 1.0 if (min(a1_map, a2_map), max(a1_map, a2_map)) in new_bonds else 0.0
        bf_annotated = bf + [is_new]

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf_annotated)
        edge_attr.append(bf_annotated)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM + 1), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        condition=torch.tensor(condition_vec, dtype=torch.float).unsqueeze(0),
        num_nodes=n_atoms,
    )
    return data


# ========================================================================
# Graph Type 2: Multi-View Graph
# ========================================================================

def build_multiview_graph(
    reactant_smi: str,
    product_smi: str,
    label: int,
    condition_vec: np.ndarray,
) -> Optional[Data]:
    """Build separate reactant and product graphs packaged together.

    Returns a Data object with:
      - x_r, edge_index_r, edge_attr_r: reactant graph
      - x_p, edge_index_p, edge_attr_p: product graph
      - condition: reaction condition vector
      - y: label
    """
    r_mol = Chem.MolFromSmiles(reactant_smi)
    p_mol = Chem.MolFromSmiles(product_smi)
    if r_mol is None or p_mol is None:
        return None

    r_graph = mol_to_graph(r_mol)
    p_graph = mol_to_graph(p_mol)
    if r_graph is None or p_graph is None:
        return None

    data = Data(
        x_r=r_graph.x,
        edge_index_r=r_graph.edge_index,
        edge_attr_r=r_graph.edge_attr,
        num_nodes_r=r_graph.num_nodes,
        x_p=p_graph.x,
        edge_index_p=p_graph.edge_index,
        edge_attr_p=p_graph.edge_attr,
        num_nodes_p=p_graph.num_nodes,
        num_nodes=r_graph.num_nodes + p_graph.num_nodes,  # for PyG batching
        y=torch.tensor([label], dtype=torch.long),
        condition=torch.tensor(condition_vec, dtype=torch.float).unsqueeze(0),
    )
    return data


# ========================================================================
# Graph Type 3: 3D Spatial Graph
# ========================================================================

def build_3d_graph(
    mol: Chem.Mol,
    coords: np.ndarray,
    label: int,
    condition_vec: np.ndarray,
    cutoff: float = 5.0,
) -> Optional[Data]:
    """Build 3D spatial graph with distance-based edges.

    Edges connect atoms within `cutoff` Angstroms.
    Edge features include distance and bond type (if covalent).
    """
    if mol is None or coords is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if len(coords) != n_atoms:
        return None

    # Node features
    x = torch.tensor([atom_features(mol.GetAtomWithIdx(i)) for i in range(n_atoms)],
                      dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)

    # Distance-based edges
    dist_matrix = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))

    # Build covalent bond set for annotation
    covalent = set()
    for bond in mol.GetBonds():
        covalent.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        covalent.add((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    edge_index = []
    edge_attr = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = dist_matrix[i, j]
            if d < cutoff:
                is_covalent = 1.0 if (i, j) in covalent else 0.0
                feat = [d / cutoff, is_covalent]  # normalized distance + covalent flag
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(feat)
                edge_attr.append(feat)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor([label], dtype=torch.long),
        condition=torch.tensor(condition_vec, dtype=torch.float).unsqueeze(0),
        num_nodes=n_atoms,
    )
    return data


# ========================================================================
# Graph Type 4: TS Approximation Graph
# ========================================================================

def build_ts_graph(
    mapped_rxn: str,
    label: int,
    condition_vec: np.ndarray,
) -> Optional[Data]:
    """Build a transition-state approximation graph.

    Union of reactant and product bonds, with edge annotations:
      - bond_type: single/double/triple/aromatic
      - bond_change: +1 (formed), -1 (broken), 0 (unchanged)
      - bond_order_change: product_order - reactant_order

    This captures the "halfway" between reactant and product.
    """
    parts = str(mapped_rxn).split(">>")
    if len(parts) != 2:
        return None

    r_mol = Chem.MolFromSmiles(parts[0])
    p_mol = Chem.MolFromSmiles(parts[1])
    if r_mol is None or p_mol is None:
        return None

    # Use product atoms as the node basis
    n_atoms = p_mol.GetNumAtoms()
    map2idx = {}
    x_list = []

    for i in range(n_atoms):
        atom = p_mol.GetAtomWithIdx(i)
        feat = atom_features(atom)
        map2idx[atom.GetAtomMapNum()] = i
        x_list.append(feat)

    x = torch.tensor(x_list, dtype=torch.float)

    # Get bond order functions
    def get_bond_orders(mol):
        orders = {}
        for b in mol.GetBonds():
            a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomMapNum()
            a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomMapNum()
            if a1 > 0 and a2 > 0:
                key = (min(a1, a2), max(a1, a2))
                orders[key] = b.GetBondTypeAsDouble()
        return orders

    r_orders = get_bond_orders(r_mol)
    p_orders = get_bond_orders(p_mol)

    # Union of all bonds
    all_bond_keys = set(r_orders.keys()) | set(p_orders.keys())

    edge_index = []
    edge_attr = []
    for key in all_bond_keys:
        r_order = r_orders.get(key, 0.0)
        p_order = p_orders.get(key, 0.0)

        # Map atom map numbers to product indices
        idx_a = map2idx.get(key[0])
        idx_b = map2idx.get(key[1])
        if idx_a is None or idx_b is None:
            continue

        # Bond change annotation
        if r_order == 0:
            change = 1.0  # formed
        elif p_order == 0:
            change = -1.0  # broken
        else:
            change = 0.0  # unchanged

        order_change = p_order - r_order
        avg_order = (r_order + p_order) / 2.0

        feat = [avg_order / 3.0, change, order_change / 3.0]

        edge_index.append([idx_a, idx_b])
        edge_index.append([idx_b, idx_a])
        edge_attr.append(feat)
        edge_attr.append(feat)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        condition=torch.tensor(condition_vec, dtype=torch.float).unsqueeze(0),
        num_nodes=n_atoms,
    )
    return data
