"""G1: MPNN on Reaction Difference Graph.

Message-passing neural network operating on the product graph with
reaction center annotations (new bond edges marked). This directly
encodes the chemical transformation.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data

from .condition_fusion import FiLMLayer, ReadoutConcat, NodeInject


class MPNNDiff(nn.Module):
    """MPNN for reaction difference graphs.

    Architecture:
      1. Node embedding (input_dim → hidden_dim)
      2. N message-passing layers with edge-conditioned convolution
      3. Readout (global mean pool or Set2Set)
      4. Condition fusion (FiLM / concat / inject)
      5. Classification head
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 4,
        condition_dim: int = 35,
        fusion: str = "concat",  # "film", "concat", "inject"
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_type = fusion
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Adjust input dim for inject fusion
        actual_node_dim = node_input_dim + (condition_dim if fusion == "inject" else 0)

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(actual_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Edge network for NNConv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            edge_net = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            conv = NNConv(hidden_dim, hidden_dim, edge_net, aggr="mean")
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # FiLM layers (one per GNN layer)
        if fusion == "film":
            self.film_layers = nn.ModuleList([
                FiLMLayer(hidden_dim, condition_dim) for _ in range(num_layers)
            ])

        # Readout
        self.readout = global_mean_pool

        # Classification head
        if fusion == "concat":
            self.classifier = ReadoutConcat(hidden_dim, condition_dim, num_classes,
                                           hidden_dim, dropout)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        # Node inject module
        if fusion == "inject":
            self.node_inject = NodeInject(condition_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        condition = data.condition.squeeze(1) if data.condition.dim() == 3 else data.condition

        # Node inject fusion (pre-processing)
        if self.fusion_type == "inject":
            x = self.node_inject(x, condition, batch)

        # Node embedding
        h = self.node_embed(x)

        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = self.act(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

            # FiLM modulation after each layer
            if self.fusion_type == "film":
                cond_per_node = condition[batch]
                h = self.film_layers[i](h, cond_per_node)

        # Graph readout
        graph_emb = self.readout(h, batch)  # (B, hidden_dim)

        # Classification
        if self.fusion_type == "concat":
            logits = self.classifier(graph_emb, condition)
        else:
            logits = self.classifier(graph_emb)

        return logits
