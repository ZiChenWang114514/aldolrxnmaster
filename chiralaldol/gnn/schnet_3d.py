"""G4: SchNet/DimeNet++ for 3D Spatial Graphs.

3D-aware GNN that operates on atom positions with continuous
distance-based convolutions. Suitable for the 3D conformer inputs.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet as SchNetBase
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

from .condition_fusion import FiLMLayer, ReadoutConcat


class SchNet3D(nn.Module):
    """SchNet wrapper for stereochemistry prediction.

    Uses PyG's built-in SchNet for 3D molecular property prediction,
    adapted for classification with condition fusion.
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 4,
        condition_dim: int = 35,
        fusion: str = "concat",
        dropout: float = 0.3,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.fusion_type = fusion
        self.hidden_dim = hidden_dim

        # Node embedding (atom features → hidden)
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
        )

        # Continuous-filter convolution layers
        self.interactions = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.interactions.append(
                ContinuousFilterConv(hidden_dim, cutoff=cutoff)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # FiLM
        if fusion == "film":
            self.film_layers = nn.ModuleList([
                FiLMLayer(hidden_dim, condition_dim) for _ in range(num_layers)
            ])

        # Classification
        if fusion == "concat":
            self.classifier = ReadoutConcat(hidden_dim, condition_dim, num_classes,
                                           hidden_dim, dropout)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

    def forward(self, data: Data) -> torch.Tensor:
        x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr
        batch = data.batch
        condition = data.condition.squeeze(1) if data.condition.dim() == 3 else data.condition

        # Node embedding
        h = self.node_embed(x)

        # Interaction layers
        for i, (interaction, norm) in enumerate(zip(self.interactions, self.norms)):
            h_new = interaction(h, pos, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

            if self.fusion_type == "film":
                cond_per_node = condition[batch]
                h = self.film_layers[i](h, cond_per_node)

        # Readout
        graph_emb = global_mean_pool(h, batch)

        # Classification
        if self.fusion_type == "concat":
            return self.classifier(graph_emb, condition)
        else:
            return self.classifier(graph_emb)


class ContinuousFilterConv(nn.Module):
    """Continuous-filter convolution layer (SchNet-style).

    Uses RBF expansion of interatomic distances as filter basis.
    """

    def __init__(self, hidden_dim: int, cutoff: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        self.cutoff = cutoff

        # Distance expansion (Gaussian RBF)
        self.rbf = GaussianRBF(num_gaussians=num_gaussians, cutoff=cutoff)

        # Filter network
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index

        # Compute distances
        diff = pos[row] - pos[col]
        dist = diff.norm(dim=-1, keepdim=True)

        # RBF expansion
        rbf = self.rbf(dist.squeeze(-1))

        # Filter
        W = self.filter_net(rbf)

        # Message: element-wise product of neighbor features and filter
        msg = h[col] * W

        # Aggregate
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        # Update
        return self.update(agg)


class GaussianRBF(nn.Module):
    """Gaussian Radial Basis Function expansion."""

    def __init__(self, num_gaussians: int = 50, cutoff: float = 5.0):
        super().__init__()
        offset = torch.linspace(0, cutoff, num_gaussians)
        self.register_buffer("offset", offset)
        self.width = (cutoff / num_gaussians) * 0.5

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distance into Gaussian basis. dist: (E,) → (E, num_gaussians)"""
        return torch.exp(-((dist.unsqueeze(-1) - self.offset) ** 2) / (2 * self.width ** 2))
