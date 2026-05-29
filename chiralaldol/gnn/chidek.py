"""ChiDeK: Chiral Determinant Kernel for ZT Transition State Graphs.

Implements the core ideas from Shi et al. 2026 (ICLR):
"Learning Molecular Chirality via Chiral Determinant Kernels"

Key components:
1. Chirality Matrix: 3×3 matrix from neighbor displacement vectors
2. Chiral Determinant: det(Q) from QR decomposition — sign encodes handedness
3. Chiral Cross-Attention: chiral atoms attend to all atoms with distance bias

Adapted for ZT graphs where C_alpha and C_aldehyde are the stereogenic centers.

Reference: arxiv 2602.07415
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class ChiralDeterminantLayer(nn.Module):
    """Compute chirality-sensitive embeddings using determinant of neighbor matrix.

    For each stereogenic atom:
    1. Build chirality matrix M from 3D displacements of neighbors
    2. QR decompose: M = QR
    3. Compute sign = det(Q) ∈ {-1, +1}
    4. Produce gated embedding: h_chiral = sign * f(R_features)

    This is SE(3)-invariant but reflection-sensitive — enantiomers
    produce different sign values.
    """

    def __init__(self, hidden_dim, n_stereo_types=2):
        super().__init__()
        # Learned transform for R matrix features
        self.R_transform = nn.Sequential(
            nn.Linear(9, hidden_dim),  # 3×3 R matrix flattened
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Combine chiral embedding with original node embedding
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, pos, edge_index, node_type):
        """
        Args:
            x: (N, hidden_dim) node embeddings
            pos: (N, 3) 3D coordinates
            edge_index: (2, E)
            node_type: (N,) node type indices

        Returns:
            (N, hidden_dim) updated embeddings with chirality info
        """
        device = x.device
        N = x.size(0)
        chiral_emb = torch.zeros_like(x)

        # Identify stereogenic atoms: C_alpha (type 3) and C_aldehyde (type 5)
        stereo_mask = (node_type == 3) | (node_type == 5)

        if not stereo_mask.any():
            return self.merge(torch.cat([x, chiral_emb], dim=1))

        stereo_indices = torch.where(stereo_mask)[0]

        for idx in stereo_indices:
            # Find neighbors from edge_index
            neighbors = edge_index[1, edge_index[0] == idx]
            if len(neighbors) < 3:
                # Need at least 3 neighbors for 3×3 matrix
                # Pad with zeros if fewer
                continue

            # Take first 3 neighbors (or up to 4 for tetrahedral)
            n_use = min(len(neighbors), 3)
            neighbor_pos = pos[neighbors[:n_use]]  # (k, 3)
            center_pos = pos[idx]  # (3,)

            # Displacement vectors
            displacements = neighbor_pos - center_pos.unsqueeze(0)  # (k, 3)

            # Build chirality matrix M (3×3)
            if n_use < 3:
                M = torch.zeros(3, 3, device=device)
                M[:n_use] = displacements
            else:
                M = displacements[:3]  # (3, 3)

            # QR decomposition
            try:
                Q, R = torch.linalg.qr(M)
                # Determinant of Q: +1 or -1 (reflection sensitivity)
                det_Q = torch.linalg.det(Q)
                sign = torch.sign(det_Q)  # ±1

                # Features from R matrix (upper triangular, encodes distances/angles)
                R_flat = R.reshape(-1)  # (9,)
                R_features = self.R_transform(R_flat)  # (hidden_dim,)

                # Gated chiral embedding
                chiral_emb[idx] = sign * R_features

            except Exception:
                # QR can fail for degenerate matrices
                continue

        return self.merge(torch.cat([x, chiral_emb], dim=1))


class ChiralCrossAttention(nn.Module):
    """Cross-attention from chiral atoms to all atoms.

    Chiral atoms (C_alpha, C_aldehyde) attend to all atoms,
    with distance-aware Gaussian kernel bias (GKPT).
    """

    def __init__(self, hidden_dim, n_heads=4, n_rbf=16):
        super().__init__()
        self.n_heads = n_heads
        head_dim = hidden_dim // n_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

        # Gaussian RBF distance bias
        self.rbf_centers = nn.Parameter(torch.linspace(0.5, 5.0, n_rbf))
        self.rbf_widths = nn.Parameter(torch.ones(n_rbf) * 0.5)
        self.distance_bias = nn.Linear(n_rbf, n_heads)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pos, node_type):
        """
        Args:
            x: (N, hidden_dim)
            pos: (N, 3)
            node_type: (N,)

        Returns:
            (N, hidden_dim) — only chiral atoms are updated
        """
        device = x.device
        N, D = x.shape

        stereo_mask = (node_type == 3) | (node_type == 5)
        if not stereo_mask.any():
            return x

        stereo_idx = torch.where(stereo_mask)[0]
        n_stereo = len(stereo_idx)

        # Q from chiral atoms, K/V from all atoms
        Q = self.W_q(x[stereo_idx]).view(n_stereo, self.n_heads, -1)
        K = self.W_k(x).view(N, self.n_heads, -1)
        V = self.W_v(x).view(N, self.n_heads, -1)

        # Attention scores
        scale = Q.size(-1) ** 0.5
        scores = torch.einsum("qhd,nhd->qnh", Q, K) / scale  # (n_stereo, N, n_heads)

        # Distance bias (GKPT)
        d_stereo = pos[stereo_idx]  # (n_stereo, 3)
        d_all = pos  # (N, 3)
        dists = torch.cdist(d_stereo, d_all)  # (n_stereo, N)

        # Gaussian RBF
        rbf = torch.exp(
            -((dists.unsqueeze(-1) - self.rbf_centers) ** 2)
            / (2 * self.rbf_widths ** 2)
        )  # (n_stereo, N, n_rbf)
        dist_bias = self.distance_bias(rbf)  # (n_stereo, N, n_heads)

        scores = scores + dist_bias

        attn = F.softmax(scores, dim=1)  # (n_stereo, N, n_heads)
        out = torch.einsum("qnh,nhd->qhd", attn, V)
        out = out.reshape(n_stereo, -1)
        out = self.W_out(out)

        # Update only chiral atoms
        x_new = x.clone()
        x_new[stereo_idx] = self.norm(x[stereo_idx] + out)

        return x_new


class ZTChiDeK(nn.Module):
    """ChiDeK-inspired GNN for ZT transition state graphs.

    Architecture:
    1. GIN backbone for initial embeddings (topology-aware)
    2. ChiralDeterminant layers for chirality encoding (3D-aware)
    3. ChiralCrossAttention for global chirality signal propagation
    4. Ring-aware readout + classifier
    """

    def __init__(self, node_dim=20, edge_dim=5, hidden_dim=128,
                 n_backbone_layers=3, n_chiral_layers=2,
                 n_classes=4, dropout=0.2, global_feat_dim=0):
        super().__init__()

        # Stage 1: GIN backbone
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.backbone = nn.ModuleList()
        self.backbone_bn = nn.ModuleList()
        for _ in range(n_backbone_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.backbone.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.backbone_bn.append(nn.BatchNorm1d(hidden_dim))

        # Stage 2: Chiral layers
        self.chiral_det_layers = nn.ModuleList([
            ChiralDeterminantLayer(hidden_dim) for _ in range(n_chiral_layers)
        ])
        self.chiral_attn_layers = nn.ModuleList([
            ChiralCrossAttention(hidden_dim) for _ in range(n_chiral_layers)
        ])
        self.chiral_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_chiral_layers)
        ])

        # Stage 3: Classifier
        clf_input = hidden_dim + global_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        x = self.node_embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr)

        # GIN backbone
        for conv, bn in zip(self.backbone, self.backbone_bn):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Chiral layers (need 3D coordinates)
        if hasattr(data, "pos") and data.pos is not None:
            for det_layer, attn_layer, norm in zip(
                self.chiral_det_layers, self.chiral_attn_layers, self.chiral_norms
            ):
                x = det_layer(x, data.pos, data.edge_index, data.node_type)
                x = attn_layer(x, data.pos, data.node_type)
                x = norm(x)

        out = global_mean_pool(x, data.batch)

        if hasattr(data, "x_global") and data.x_global is not None:
            out = torch.cat([out, data.x_global], dim=1)

        return self.classifier(out)
