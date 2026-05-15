"""G2: GAT Multi-View Graph Model.

Processes reactant and product graphs separately with GAT,
then combines embeddings with attention-weighted fusion.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

from .condition_fusion import FiLMLayer, ReadoutConcat, NodeInject


class GATEncoder(nn.Module):
    """GAT encoder for a single molecular graph."""

    def __init__(self, input_dim, hidden_dim, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout,
                                       concat=(i < num_layers - 1)))
            out_dim = hidden_dim * heads if i < num_layers - 1 else hidden_dim
            self.norms.append(nn.LayerNorm(out_dim))

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
        return global_mean_pool(h, batch)


class GATMultiView(nn.Module):
    """Multi-view model: separate GAT for reactant and product, then fuse.

    Architecture:
      1. GAT encoder for reactant graph → emb_r
      2. GAT encoder for product graph → emb_p
      3. Attention-weighted combination: emb = α·emb_r + (1-α)·emb_p
      4. Condition fusion
      5. Classification head
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 4,
        condition_dim: int = 35,
        fusion: str = "concat",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_type = fusion

        actual_dim = node_input_dim + (condition_dim if fusion == "inject" else 0)

        self.reactant_encoder = GATEncoder(actual_dim, hidden_dim, num_layers, heads, dropout)
        self.product_encoder = GATEncoder(actual_dim, hidden_dim, num_layers, heads, dropout)

        # Attention for view combination
        self.view_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # FiLM
        if fusion == "film":
            self.film = FiLMLayer(hidden_dim, condition_dim)

        # Classification
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

        if fusion == "inject":
            self.node_inject = NodeInject(condition_dim)

    def forward(self, data: Data) -> torch.Tensor:
        condition = data.condition.squeeze(1) if data.condition.dim() == 3 else data.condition

        # Process reactant graph
        x_r = data.x_r
        if self.fusion_type == "inject":
            x_r = self.node_inject(x_r, condition, data.x_r_batch)
        emb_r = self.reactant_encoder(x_r, data.edge_index_r, data.x_r_batch)

        # Process product graph
        x_p = data.x_p
        if self.fusion_type == "inject":
            x_p = self.node_inject(x_p, condition, data.x_p_batch)
        emb_p = self.product_encoder(x_p, data.edge_index_p, data.x_p_batch)

        # Attention-weighted fusion
        alpha = self.view_attention(torch.cat([emb_r, emb_p], dim=-1))
        graph_emb = alpha * emb_r + (1 - alpha) * emb_p

        # FiLM
        if self.fusion_type == "film":
            graph_emb = self.film(graph_emb, condition)

        # Classification
        if self.fusion_type == "concat":
            return self.classifier(graph_emb, condition)
        else:
            return self.classifier(graph_emb)
