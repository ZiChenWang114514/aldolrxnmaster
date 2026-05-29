"""Generate 3D coordinates for Zimmerman-Traxler transition state graphs.

The ZT 6-membered chair TS has well-defined geometry:
- Chair conformation with alternating axial/equatorial positions
- Bond lengths: C-C ~1.54Å, C=C ~1.34Å, C-O ~1.43Å, M-O ~2.0Å (varies by metal)
- Bond angles: ~109.5° (sp3) or ~120° (sp2)
- Ring torsions: ±60° (chair)

Two approaches:
1. Rule-based: construct ideal chair geometry from standard bond parameters
2. RDKit: embed a SMILES approximation of the TS ring

Reference:
  Bernardi et al., JACS 2010, 132, 9982 — DFT-optimized Evans aldol TS
  Ti-O: 1.83Å, C-C(new): 2.23Å (partial bond in TS)
"""

import logging

import numpy as np

from .zt_graph_builder import ZTGraph, ZTNodeType

logger = logging.getLogger(__name__)

# Standard bond lengths for ZT chair (Å)
# Based on Bernardi 2010 DFT-optimized Evans aldol TS
BOND_LENGTHS = {
    "M_O": 2.00,       # Metal-Oxygen coordination (avg)
    "O_C": 1.35,       # Enolate O-C (partial double bond character)
    "C_C_double": 1.38, # Enolate C=C
    "C_C_new": 1.80,   # New C-C bond (partially formed in TS)
    "C_O_ald": 1.28,   # Aldehyde C=O (elongated in TS)
    "C_sub": 1.52,      # C-substituent
}

# Metal-specific M-O distances (Å)
METAL_MO_DIST = {
    "Ti": 1.83,   # Bernardi 2010
    "B": 1.55,    # Boron enolate
    "Li": 1.96,   # Lithium
    "Mg": 2.05,   # Magnesium
    "Sn": 2.15,   # Tin
    "Zn": 2.05,   # Zinc
    "none": 2.00,
    "unknown": 2.00,
}


def _build_chair_ring_3d(metal="Ti"):
    """Build idealized 3D coordinates for the 6-membered ZT chair ring.

    Returns (6, 3) array of coordinates for:
        [M, O_metal, C_carbonyl, C_alpha, C_aldehyde, O_aldehyde]

    The chair is oriented with:
    - Ring mean plane approximately in xy
    - Axial substituents along z
    """
    mo_dist = METAL_MO_DIST.get(metal, 2.00)

    # Chair ring coordinates (idealized cyclohexane-like)
    # Using standard chair with puckering parameter ~0.56Å
    puck = 0.30  # puckering amplitude (smaller than cyclohexane due to heteroatoms)

    # Ring atom positions in chair conformation
    # Numbering: 0=M, 1=O_metal, 2=C_carbonyl, 3=C_alpha, 4=C_aldehyde, 5=O_aldehyde
    # Alternating up/down puckering
    coords = np.zeros((6, 3), dtype=np.float64)

    # Place atoms around a hexagonal ring in xy-plane with z-puckering
    for i in range(6):
        angle = i * np.pi / 3  # 60° increments
        r = 1.40  # approximate ring radius

        # Adjust radius for M-O vs C-C bonds
        if i == 0:  # Metal — larger radius
            r = mo_dist * 0.7
        elif i in (1, 5):  # O atoms — slightly smaller
            r = 1.30

        coords[i, 0] = r * np.cos(angle)
        coords[i, 1] = r * np.sin(angle)
        # Chair puckering: alternating +/- z
        coords[i, 2] = puck * (1 if i % 2 == 0 else -1)

    return coords.astype(np.float32)


def _place_substituent_3d(ring_coords, anchor_idx, n_sub_atoms, direction="equatorial"):
    """Place substituent atoms extending from a ring atom.

    Args:
        ring_coords: (6, 3) ring atom coordinates
        anchor_idx: index of the ring atom to attach to
        n_sub_atoms: number of substituent atoms to place
        direction: "axial" or "equatorial"

    Returns:
        (n_sub_atoms, 3) array of substituent coordinates
    """
    if n_sub_atoms == 0:
        return np.zeros((0, 3), dtype=np.float32)

    anchor = ring_coords[anchor_idx]

    # Direction vectors
    # Axial: along z (perpendicular to ring plane)
    # Equatorial: roughly in ring plane, away from center
    ring_center = ring_coords.mean(axis=0)
    radial = anchor - ring_center
    radial_norm = radial / (np.linalg.norm(radial) + 1e-8)

    if direction == "axial":
        # z-direction, same sign as anchor's z-puckering
        base_dir = np.array([0, 0, 1.0 if anchor[2] > 0 else -1.0])
    else:
        # equatorial: mix of radial + slight z
        base_dir = radial_norm * 0.9 + np.array([0, 0, -0.1 * np.sign(anchor[2])])
        base_dir /= np.linalg.norm(base_dir)

    # Place atoms in a chain extending from anchor
    sub_coords = np.zeros((n_sub_atoms, 3), dtype=np.float32)
    bond_len = BOND_LENGTHS["C_sub"]

    for i in range(n_sub_atoms):
        # Add slight random perturbation for each subsequent atom
        # (mimics chain branching)
        perturb = np.random.RandomState(42 + i).randn(3) * 0.15
        sub_coords[i] = anchor + base_dir * bond_len * (i + 1) + perturb * (i + 1)

    return sub_coords


def add_3d_coords_to_zt_graph(graph, seed=42):
    """Add 3D coordinates to a ZT graph.

    Generates idealized chair geometry for the 6-membered ring
    and extends substituent atoms outward.

    Args:
        graph: ZTGraph object (modified in place)
        seed: random seed for substituent placement

    Returns:
        (n_nodes, 3) float32 array of 3D coordinates, also stored as graph.pos
    """
    if graph.status != "success" or len(graph.node_types) < 6:
        n = max(len(graph.node_types), 0)
        pos = np.zeros((n, 3), dtype=np.float32)
        graph.pos = pos
        return pos

    n_nodes = len(graph.node_types)
    metal = graph.metal if graph.metal else "none"

    # Build ring coordinates
    ring_coords = _build_chair_ring_3d(metal)

    # Initialize full position array
    pos = np.zeros((n_nodes, 3), dtype=np.float32)
    pos[:6] = ring_coords

    # Place substituent atoms
    # Find which ring atoms have substituent connections
    if n_nodes > 6:
        edge_src = graph.edge_index[0]
        edge_dst = graph.edge_index[1]

        # For each ring atom, find connected substituent atoms
        sub_start = 6
        for ring_idx in range(6):
            # Find substituent atoms connected to this ring atom
            connected_subs = []
            for e in range(len(edge_src)):
                if edge_src[e] == ring_idx and edge_dst[e] >= 6:
                    connected_subs.append(int(edge_dst[e]))

            if not connected_subs:
                continue

            # Determine direction based on node type
            # C_alpha (type 3) has R1 in axial position (Evans syn TS)
            # C_aldehyde (type 5) has R_ald in equatorial position
            ntype = graph.node_types[ring_idx]
            if ntype == ZTNodeType.C_ALPHA:
                direction = "axial"
            else:
                direction = "equatorial"

            # Find all substituent atoms reachable from this connection
            sub_tree = _get_subtree(graph, connected_subs[0])
            n_sub = len(sub_tree)

            sub_pos = _place_substituent_3d(ring_coords, ring_idx, n_sub, direction)
            for i, sub_idx in enumerate(sub_tree):
                if i < len(sub_pos):
                    pos[sub_idx] = sub_pos[i]

    graph.pos = pos
    return pos


def _get_subtree(graph, start_idx):
    """BFS to get all substituent nodes reachable from start_idx."""
    visited = {start_idx}
    queue = [start_idx]
    result = [start_idx]

    while queue:
        node = queue.pop(0)
        edge_src = graph.edge_index[0]
        edge_dst = graph.edge_index[1]
        for e in range(len(edge_src)):
            if edge_src[e] == node:
                neighbor = int(edge_dst[e])
                if neighbor not in visited and neighbor >= 6:  # only substituent atoms
                    visited.add(neighbor)
                    queue.append(neighbor)
                    result.append(neighbor)

    return sorted(result)


def add_3d_coords_batch(graphs, seed=42):
    """Add 3D coordinates to all graphs in a list.

    Args:
        graphs: list of ZTGraph
        seed: random seed

    Returns:
        list of (n_nodes, 3) arrays
    """
    all_pos = []
    n_success = 0
    for i, g in enumerate(graphs):
        pos = add_3d_coords_to_zt_graph(g, seed=seed + i)
        all_pos.append(pos)
        if g.status == "success":
            n_success += 1

    logger.info(f"Added 3D coordinates to {n_success}/{len(graphs)} ZT graphs")
    return all_pos
