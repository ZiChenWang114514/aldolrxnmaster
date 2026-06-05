"""G3: SE(3)-Equivariant Transformer for 3D Stereochemistry Prediction.

Simplified Equiformer-style architecture using:
  - Invariant scalar features (from atom types)
  - Equivariant vector features (from 3D positions)
  - Dot-product attention with distance bias

This captures 3D spatial relationships while being
rotation/translation equivariant — ideal for stereo prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

from .condition_fusion import FiLMLayer, ReadoutConcat
from .schnet_3d import GaussianRBF


class EquivariantAttentionLayer(nn.Module):
    """SE(3)-equivariant attention layer.

    Scalar features are updated via attention-weighted messages.
    Vector features are updated via equivariant operations.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention projections (scalar)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Distance bias
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.dist_proj = nn.Linear(num_rbf, num_heads)

        # Output
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Vector update (equivariant)
        self.vec_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h, vec, pos, edge_index):
        """
        Args:
            h: scalar features (N, hidden_dim)
            vec: vector features (N, 3, hidden_dim) — equivariant
            pos: 3D positions (N, 3)
            edge_index: (2, E) edges
        """
        row, col = edge_index

        # Compute distances
        diff = pos[row] - pos[col]  # (E, 3)
        dist = diff.norm(dim=-1)  # (E,)
        unit = diff / (dist.unsqueeze(-1) + 1e-8)  # (E, 3)

        # Distance bias
        rbf = self.rbf(dist)  # (E, num_rbf)
        dist_bias = self.dist_proj(rbf)  # (E, num_heads)

        # Q, K, V projections
        Q = self.q_proj(h[row]).view(-1, self.num_heads, self.head_dim)  # (E, H, D)
        K = self.k_proj(h[col]).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(h[col]).view(-1, self.num_heads, self.head_dim)

        # Attention scores with distance bias
        attn = (Q * K).sum(-1) / (self.head_dim ** 0.5) + dist_bias  # (E, H)

        # Softmax per target node
        attn_max = torch.zeros(h.size(0), self.num_heads, device=h.device)
        attn_max.scatter_reduce_(0, row.unsqueeze(-1).expand_as(attn), attn, reduce="amax")
        attn = torch.exp(attn - attn_max[row])

        attn_sum = torch.zeros(h.size(0), self.num_heads, device=h.device)
        attn_sum.scatter_add_(0, row.unsqueeze(-1).expand_as(attn), attn)
        attn = attn / (attn_sum[row] + 1e-8)

        # Weighted message aggregation (scalar)
        msg = (attn.unsqueeze(-1) * V).reshape(-1, self.hidden_dim)  # (E, hidden_dim)
        h_out = torch.zeros_like(h)
        h_out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        h_out = self.out_proj(h_out)

        # Vector update (equivariant)
        # Use attention weights to aggregate direction-weighted messages
        vec_msg = self.vec_proj(h[col])  # (E, hidden_dim)
        vec_msg = unit.unsqueeze(-1) * vec_msg.unsqueeze(1)  # (E, 3, hidden_dim)
        attn_expanded = attn.mean(dim=-1, keepdim=True).unsqueeze(1)  # (E, 1, 1)
        vec_msg = vec_msg * attn_expanded

        vec_out = torch.zeros_like(vec)
        idx = row.unsqueeze(-1).unsqueeze(-1).expand_as(vec_msg)
        vec_out.scatter_add_(0, idx, vec_msg)

        return h_out, vec_out


class SimpleEquiformer(nn.Module):
    """Simplified Equiformer for stereochemistry prediction.

    Uses SE(3)-equivariant attention layers that update both
    scalar (invariant) and vector (equivariant) representations.
    Final prediction uses only scalar features (invariant output).
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_classes: int = 4,
        condition_dim: int = 35,
        fusion: str = "concat",
        dropout: float = 0.3,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.fusion_type = fusion
        self.hidden_dim = hidden_dim

        # Node embedding
        self.node_embed = nn.Linear(node_input_dim, hidden_dim)

        # Equivariant attention layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EquivariantAttentionLayer(hidden_dim, num_heads, cutoff=cutoff)
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
        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = data.batch
        condition = data.condition.squeeze(1) if data.condition.dim() == 3 else data.condition

        # Embed scalars
        h = self.node_embed(x)

        # Initialize vector features as zero
        vec = torch.zeros(h.size(0), 3, self.hidden_dim, device=h.device)

        # Equivariant attention layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_new, vec_new = layer(h, vec, pos, edge_index)
            h = h + self.dropout(norm(h_new))  # residual + norm
            vec = vec + vec_new

            if self.fusion_type == "film":
                cond_per_node = condition[batch]
                h = self.film_layers[i](h, cond_per_node)

        # Readout (scalar only — invariant)
        graph_emb = global_mean_pool(h, batch)

        # Classification
        if self.fusion_type == "concat":
            return self.classifier(graph_emb, condition)
        else:
            return self.classifier(graph_emb)
