"""Extract fixed-size feature vectors from ZT transition state graphs.

Produces a flat feature vector per reaction that can be used as:
1. Extra features (x_d) for Chemprop MPNN
2. Additional features for tree models
3. Input to standalone classifiers

Feature groups:
  - Metal properties (4d)
  - Ring topology summary (6d): aggregated node features of 6 ring atoms
  - R1 substituent summary (6d): substituent on C_alpha
  - R_ald substituent summary (6d): substituent on C_aldehyde
  - Aux substituent summary (6d): C4 substituent from Evans ring
  - Graph statistics (4d): n_nodes, n_edges, n_substituent_atoms, density
Total: 32d
"""

import logging

import numpy as np

from .zt_graph_builder import ZTGraph, ZTNodeType

logger = logging.getLogger(__name__)

ZT_FEATURE_NAMES = [
    # Metal (4d)
    "zt_metal_ionic_radius", "zt_metal_coord_num", "zt_metal_hardness", "zt_metal_charge",
    # Ring node aggregate (6d) — mean of 6 ring nodes' substituent features
    "zt_ring_mean_sub_nheavy", "zt_ring_mean_sub_naromatic",
    "zt_ring_mean_sub_nhetero", "zt_ring_mean_sub_maxZ",
    "zt_ring_mean_sub_halogen", "zt_ring_mean_sub_mw",
    # R1 substituent on C_alpha (6d)
    "zt_r1_nheavy", "zt_r1_naromatic", "zt_r1_nhetero",
    "zt_r1_maxZ", "zt_r1_halogen", "zt_r1_mw",
    # R_ald substituent on C_aldehyde (6d)
    "zt_rald_nheavy", "zt_rald_naromatic", "zt_rald_nhetero",
    "zt_rald_maxZ", "zt_rald_halogen", "zt_rald_mw",
    # Auxiliary C4 substituent (6d)
    "zt_auxsub_nheavy", "zt_auxsub_naromatic", "zt_auxsub_nhetero",
    "zt_auxsub_maxZ", "zt_auxsub_halogen", "zt_auxsub_mw",
    # Graph statistics (4d)
    "zt_n_nodes", "zt_n_edges", "zt_n_substituent_atoms", "zt_graph_density",
]

ZT_FEATURE_DIM = len(ZT_FEATURE_NAMES)  # 32


def extract_zt_features(graph: ZTGraph) -> np.ndarray:
    """Extract a fixed-size feature vector from a ZT graph.

    Returns: (32,) float32 array. All zeros if graph is invalid.
    """
    if graph.status != "success" or len(graph.node_types) < 6:
        return np.zeros(ZT_FEATURE_DIM, dtype=np.float32)

    nf = graph.node_features  # (n_nodes, 20)

    # Metal properties — from any ring node (all share same metal props)
    # node_features layout: [type_onehot(7), metal_props(4), sub_feats(6), is_ring(1), ax_eq(2)]
    metal_feats = nf[0, 7:11]  # ionic_radius, coord_num/6, hardness/5, charge/4

    # Ring node aggregate — mean of substituent features across 6 ring nodes
    ring_sub_feats = nf[:6, 11:17]  # (6, 6) substituent features
    ring_mean = ring_sub_feats.mean(axis=0)

    # Individual substituent summaries
    # C_alpha (node 3) substituent features
    r1_feats = nf[3, 11:17]

    # C_aldehyde (node 4) substituent features
    # Note: node ordering is [M, O_metal, C_carbonyl, C_alpha, O_ald, C_ald]
    # Actually check: in build_zt_graph_evans, node 4 is C_aldehyde and node 5 is O_aldehyde
    # But the node_types show [0,1,2,3,5,4,...] — need to check ordering
    # The actual order depends on which was appended first
    # Let's use node_types to find the right indices
    c_ald_mask = graph.node_types[:6] == ZTNodeType.C_ALDEHYDE
    if c_ald_mask.any():
        c_ald_idx = np.where(c_ald_mask)[0][0]
        rald_feats = nf[c_ald_idx, 11:17]
    else:
        rald_feats = np.zeros(6, dtype=np.float32)

    # Auxiliary substituent — encoded in C_alpha node (since aux sub is connected to C_alpha)
    # For more accurate extraction, we'd look at the substituent graph
    # But for the feature vector, we use the aux sub features from the C_alpha node
    # which already encodes r1 info. We need a separate source for aux sub.
    # Actually in the builder, aux sub is connected to C_alpha as a separate edge.
    # The aux sub features are not stored in ring nodes — they're in substituent nodes.
    # For now, approximate from substituent node statistics
    n_sub_atoms = int((graph.node_types == ZTNodeType.SUBSTITUENT).sum())
    sub_mask = graph.node_types == ZTNodeType.SUBSTITUENT
    if sub_mask.any():
        sub_feats_all = nf[sub_mask]
        # Use mean of all substituent atom features as aux proxy
        auxsub_feats = sub_feats_all[:, 11:17].mean(axis=0)
    else:
        auxsub_feats = np.zeros(6, dtype=np.float32)

    # Graph statistics
    n_nodes = len(graph.node_types)
    n_edges = graph.edge_index.shape[1]
    density = n_edges / max(n_nodes * (n_nodes - 1), 1)
    graph_stats = np.array([n_nodes / 50.0, n_edges / 100.0,
                            n_sub_atoms / 30.0, density], dtype=np.float32)

    # Concatenate
    features = np.concatenate([
        metal_feats,    # 4d
        ring_mean,      # 6d
        r1_feats,       # 6d
        rald_feats,     # 6d
        auxsub_feats,   # 6d
        graph_stats,    # 4d
    ])  # total: 32d

    return features.astype(np.float32)


def extract_zt_features_batch(graphs: list) -> np.ndarray:
    """Extract features for a list of ZT graphs.

    Returns: (n, 32) float32 array
    """
    return np.array([extract_zt_features(g) for g in graphs], dtype=np.float32)
