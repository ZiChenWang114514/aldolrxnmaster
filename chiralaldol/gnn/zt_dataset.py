"""PyG Dataset for Zimmerman-Traxler transition state graphs."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


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
