"""Condition fusion modules for GNN models.

Three fusion strategies for incorporating reaction conditions (35d) into GNN:
  F1. FiLM (Feature-wise Linear Modulation) — condition generates γ, β per layer
  F2. Readout Concat — concatenate condition to graph readout then MLP
  F3. Node Inject — broadcast condition to all nodes as extra features
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x_out = γ * x + β.

    γ and β are generated from condition vector via MLP.
    """

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features (N, hidden_dim)
            condition: condition vector (B, condition_dim) — broadcast to per-node
        """
        gamma = self.gamma_net(condition) + 1.0  # center around 1
        beta = self.beta_net(condition)
        return gamma * x + beta


class ReadoutConcat(nn.Module):
    """Concatenate graph readout with condition vector, then MLP classifier."""

    def __init__(self, graph_dim: int, condition_dim: int, num_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, graph_emb: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_emb: graph-level embedding (B, graph_dim)
            condition: condition vector (B, condition_dim)
        Returns:
            logits (B, num_classes)
        """
        combined = torch.cat([graph_emb, condition], dim=-1)
        return self.classifier(combined)


class NodeInject(nn.Module):
    """Broadcast condition vector to all nodes as extra features.

    Adds condition_dim features to each node before GNN processing.
    Used as a preprocessing step — modifies x before passing to GNN.
    """

    def __init__(self, condition_dim: int):
        super().__init__()
        self.condition_dim = condition_dim

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features (N_total, feat_dim)
            condition: condition vectors (B, condition_dim)
            batch: batch assignment (N_total,)
        Returns:
            augmented node features (N_total, feat_dim + condition_dim)
        """
        # Expand condition to per-node
        cond_per_node = condition[batch]  # (N_total, condition_dim)
        return torch.cat([x, cond_per_node], dim=-1)
