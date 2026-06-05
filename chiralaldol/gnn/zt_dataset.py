"""PyG Dataset for Zimmerman-Traxler transition state graphs."""


import torch
from torch_geometric.data import Data


def zt_graph_to_pyg(zt_graph, label=None, extra_features=None):
    """Convert a ZTGraph to a PyG Data object.

    Args:
        zt_graph: ZTGraph from zt_graph_builder
        label: integer class label (0-3)
        extra_features: optional (d,) array of global features

    Returns:
        torch_geometric.data.Data or None if graph is invalid
    """
    if zt_graph.status != "success" or len(zt_graph.node_types) < 6:
        return None

    x = torch.tensor(zt_graph.node_features, dtype=torch.float)
    edge_index = torch.tensor(zt_graph.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(zt_graph.edge_features, dtype=torch.float)
    node_type = torch.tensor(zt_graph.node_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_type = node_type
    data.n_ring = zt_graph.n_ring_atoms

    # 3D coordinates (if available)
    if hasattr(zt_graph, "pos") and zt_graph.pos is not None:
        data.pos = torch.tensor(zt_graph.pos, dtype=torch.float)

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    if extra_features is not None:
        data.x_global = torch.tensor(extra_features, dtype=torch.float).unsqueeze(0)

    return data


def build_pyg_dataset(graphs, labels, extra_features=None):
    """Convert lists of ZT graphs + labels to list of PyG Data.

    Args:
        graphs: list of ZTGraph
        labels: (n,) int array of labels
        extra_features: optional (n, d) array

    Returns:
        list of Data objects (skipping invalid graphs),
        list of original indices
    """
    data_list = []
    valid_indices = []

    for i, g in enumerate(graphs):
        xf = extra_features[i] if extra_features is not None else None
        d = zt_graph_to_pyg(g, label=int(labels[i]), extra_features=xf)
        if d is not None:
            data_list.append(d)
            valid_indices.append(i)

    return data_list, valid_indices


# ═══════════════════════════ Multi-TS Extensions ═══════════════════════════


def multi_ts_to_pyg_list(ts_set, label=None, extra_features=None):
    """Convert a MultiTSGraphSet to a list of 4 PyG Data objects.

    Failed TS graphs become minimal dummy graphs (6 zero-nodes + 12 self-edges)
    so the DataLoader never drops reactions.

    Returns:
        list of 4 Data objects (one per TSType).
    """
    from chiralaldol.zt_graph_builder import NODE_FEAT_DIM_MULTI, EDGE_FEAT_DIM_MULTI

    data_list = []
    for i, g in enumerate(ts_set.graphs):
        if g.status == "success" and len(g.node_types) >= 6:
            x = torch.tensor(g.node_features, dtype=torch.float)
            ei = torch.tensor(g.edge_index, dtype=torch.long)
            ea = torch.tensor(g.edge_features, dtype=torch.float)
            nt = torch.tensor(g.node_types, dtype=torch.long)
            pos = torch.tensor(g.pos, dtype=torch.float) if hasattr(g, "pos") else torch.zeros(x.size(0), 3)
        else:
            # Dummy graph: 6 zero-nodes, ring of self-edges
            x = torch.zeros(6, NODE_FEAT_DIM_MULTI)
            ei = torch.stack([torch.arange(6), torch.arange(6)], dim=0).long()
            ei = torch.cat([ei, ei.flip(0)], dim=1)
            ea = torch.zeros(ei.size(1), EDGE_FEAT_DIM_MULTI)
            nt = torch.arange(6, dtype=torch.long)
            pos = torch.zeros(6, 3)

        d = Data(x=x, edge_index=ei, edge_attr=ea, pos=pos)
        d.node_type = nt
        d.ts_type = torch.tensor([i], dtype=torch.long)
        d.ze_weights = torch.tensor(ts_set.ze_weights, dtype=torch.float)
        d.rxn_idx = torch.tensor([ts_set.reaction_idx], dtype=torch.long)
        d.is_dummy = torch.tensor([int(g.status != "success")], dtype=torch.long)

        if label is not None:
            d.y = torch.tensor([label], dtype=torch.long)
        if extra_features is not None:
            d.x_global = torch.tensor(extra_features, dtype=torch.float).unsqueeze(0)

        data_list.append(d)

    return data_list


def build_multi_ts_pyg_dataset(ts_sets, labels, extra_features=None):
    """Convert list of MultiTSGraphSet → flat list of PyG Data.

    Each reaction produces 4 consecutive Data objects.

    Returns:
        all_data: flat list of Data (len = 4 * n_reactions)
        valid_indices: list of reaction indices that have ≥1 successful TS graph
    """
    all_data = []
    valid_indices = []

    for i, ts_set in enumerate(ts_sets):
        n_ok = sum(1 for g in ts_set.graphs if g.status == "success")
        xf = extra_features[i] if extra_features is not None else None
        data_4 = multi_ts_to_pyg_list(ts_set, label=int(labels[i]), extra_features=xf)
        all_data.extend(data_4)
        if n_ok > 0:
            valid_indices.append(i)

    return all_data, valid_indices


class MultiTSDataLoader:
    """DataLoader that batches reactions (groups of 4 TS graphs).

    Guarantees each batch contains complete reaction 4-tuples.
    Shuffles at the reaction level, not the graph level.
    """

    def __init__(self, reaction_data_list, batch_size=32, shuffle=True):
        """
        Args:
            reaction_data_list: list of list[Data], each inner list has 4 Data objects
            batch_size: number of *reactions* per batch (actual graphs = 4 * batch_size)
        """
        self.reactions = reaction_data_list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.reactions) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        from torch_geometric.data import Batch

        indices = list(range(len(self.reactions)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            # Flatten 4 graphs per reaction
            flat = []
            for ri in batch_idx:
                flat.extend(self.reactions[ri])
            batch = Batch.from_data_list(flat)
            # Add reaction-level batch index: maps each graph to its reaction
            rxn_batch = torch.tensor(
                [j for j, ri in enumerate(batch_idx) for _ in range(4)],
                dtype=torch.long,
            )
            batch.rxn_batch = rxn_batch
            batch.n_reactions = len(batch_idx)
            yield batch
