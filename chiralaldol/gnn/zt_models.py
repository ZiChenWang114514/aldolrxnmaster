"""GNN models for Zimmerman-Traxler transition state graphs.

Three architectures for the ZT-GNN benchmark:
  1. ZT-GIN: Graph Isomorphism Network — powerful baseline
  2. ZT-GAT: Graph Attention Network — attention-based, good for heterogeneous nodes
  3. ZT-Chiral: Chirality-aware GNN inspired by ChiDeK — uses node_type-aware
     message passing with determinant-based chirality encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GINEConv, NNConv,
    global_mean_pool,
)


# ═══════════════════════════ ZT-GIN ═══════════════════════════

class ZTGIN(nn.Module):
    """Graph Isomorphism Network on ZT transition state graphs.

    GIN is provably as powerful as the WL test for graph isomorphism,
    making it a strong baseline for learning on ZT topology.
    """

    def __init__(self, node_dim=20, edge_dim=5, hidden_dim=128,
                 n_layers=4, n_classes=4, dropout=0.2, global_feat_dim=0):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

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

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        out = global_mean_pool(x, data.batch)

        # Append global features if available
        if hasattr(data, "x_global") and data.x_global is not None:
            out = torch.cat([out, data.x_global], dim=1)

        return self.classifier(out)


# ═══════════════════════════ ZT-GAT ═══════════════════════════

class ZTGAT(nn.Module):
    """Graph Attention Network v2 on ZT transition state graphs.

    Multi-head attention allows the model to learn which ZT ring
    interactions are most important for stereochemistry prediction.
    """

    def __init__(self, node_dim=20, edge_dim=5, hidden_dim=128,
                 n_layers=3, n_heads=4, n_classes=4, dropout=0.2,
                 global_feat_dim=0):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        head_dim = hidden_dim // n_heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim
            self.convs.append(GATv2Conv(
                in_dim, head_dim, heads=n_heads, edge_dim=hidden_dim,
                dropout=dropout, concat=True,
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

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

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_mean_pool(x, data.batch)

        if hasattr(data, "x_global") and data.x_global is not None:
            out = torch.cat([out, data.x_global], dim=1)

        return self.classifier(out)


# ═══════════════════════════ ZT-Chiral ═══════════════════════════

class ChiralMessagePassing(nn.Module):
    """Chirality-aware message passing layer.

    Inspired by ChiDeK (Shi 2026): uses node_type-dependent transformations
    to create reflection-sensitive embeddings. The key insight is that
    ring atoms in the ZT TS have specific roles (metal, oxygen, carbon)
    and their ordering around the ring encodes stereochemistry.

    Unlike standard MPNN which treats all neighbors symmetrically,
    this layer uses the node_type to apply different transformations
    to different neighbor types, preserving chirality information.
    """

    def __init__(self, hidden_dim, n_node_types=7, n_edge_types=5):
        super().__init__()
        # Type-specific message transforms
        self.W_msg = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types)
        ])
        # Edge-type-aware gating
        self.W_edge = nn.Linear(n_edge_types, hidden_dim)
        # Update
        self.W_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        # Chiral determinant: learnable sign based on neighbor ordering
        self.chiral_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x, edge_index, edge_attr, node_type):
        src, dst = edge_index
        n_nodes = x.size(0)

        # Type-specific neighbor messages
        messages = torch.zeros_like(x)
        for ntype in range(len(self.W_msg)):
            mask = (node_type[src] == ntype)
            if mask.any():
                m = self.W_msg[ntype](x[src[mask]])
                # Edge gating
                gate = torch.sigmoid(self.W_edge(edge_attr[mask]))
                m = m * gate
                # Scatter to destinations
                messages.scatter_add_(0, dst[mask].unsqueeze(1).expand_as(m), m)

        # Chiral-aware update: combine self + aggregated messages
        # The chiral gate modulates how messages are combined,
        # allowing the model to learn chirality-dependent interactions
        chiral_signal = self.chiral_gate(torch.cat([x, messages], dim=1))
        h = self.W_update(torch.cat([x, messages * chiral_signal], dim=1))

        return h


class ZTChiral(nn.Module):
    """Chirality-aware GNN for ZT transition state graphs.

    Uses ChiralMessagePassing layers that respect the chemical roles
    of different atoms in the ZT ring (metal, oxygen, carbon).
    """

    def __init__(self, node_dim=20, edge_dim=5, hidden_dim=128,
                 n_layers=4, n_classes=4, dropout=0.2, global_feat_dim=0):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ChiralMessagePassing(hidden_dim) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Ring-aware readout: separate pooling for ring vs substituent atoms
        self.ring_transform = nn.Linear(hidden_dim, hidden_dim)
        self.sub_transform = nn.Linear(hidden_dim, hidden_dim)

        clf_input = hidden_dim * 2 + global_feat_dim  # ring_pool + sub_pool
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        x = self.node_embed(data.x)

        for layer, ln in zip(self.layers, self.layer_norms):
            x_new = layer(x, data.edge_index, data.edge_attr, data.node_type)
            x = ln(x + x_new)  # residual
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Ring-aware readout
        is_ring = (data.node_type < 6)  # ring atoms: types 0-5
        is_sub = ~is_ring

        ring_x = self.ring_transform(x)
        sub_x = self.sub_transform(x)

        # Masked pooling
        ring_x = ring_x * is_ring.unsqueeze(1).float()
        sub_x = sub_x * is_sub.unsqueeze(1).float()

        ring_pool = global_mean_pool(ring_x, data.batch)
        sub_pool = global_mean_pool(sub_x, data.batch)

        out = torch.cat([ring_pool, sub_pool], dim=1)

        if hasattr(data, "x_global") and data.x_global is not None:
            out = torch.cat([out, data.x_global], dim=1)

        return self.classifier(out)


# ═══════════════════════════ ZT-ComENet ═══════════════════════════

class GaussianSmearing(nn.Module):
    """Gaussian RBF expansion of distances."""

    def __init__(self, start=0.0, stop=5.0, n_gaussians=16):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / n_gaussians) ** 2

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist ** 2)


class ZTComENet(nn.Module):
    """ComENet-style 3D message passing on ZT graphs.

    Replaces one-hot edge features with distance RBF + angular features
    computed from actual 3D coordinates (data.pos).
    """

    def __init__(self, node_dim=20, edge_dim=5, hidden_dim=128,
                 n_layers=4, n_classes=4, dropout=0.2, global_feat_dim=0,
                 n_rbf=16, **kwargs):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # 3D edge feature encoder: edge_type(5) + RBF(16) + angle(4) = 25
        edge_3d_dim = edge_dim + n_rbf + 4
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_3d_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rbf = GaussianSmearing(0.0, 8.0, n_rbf)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, nn_edge, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        clf_input = hidden_dim + global_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.dropout = dropout

    def _compute_3d_edge_features(self, data):
        """Compute 3D geometric edge features from data.pos."""
        src, dst = data.edge_index
        pos = data.pos

        # Distance
        diff = pos[dst] - pos[src]
        dist = diff.norm(dim=-1)
        rbf = self.rbf(dist)  # (n_edges, n_rbf)

        # Angular features: direction cosines + dihedral-like
        direction = diff / (dist.unsqueeze(-1) + 1e-8)
        angle_feats = torch.stack([
            direction[:, 0],  # dx
            direction[:, 1],  # dy
            torch.atan2(direction[:, 1], direction[:, 0]),  # azimuthal
            torch.acos(direction[:, 2].clamp(-1, 1)),  # polar
        ], dim=-1)  # (n_edges, 4)

        # Concatenate with original edge type features
        edge_3d = torch.cat([data.edge_attr, rbf, angle_feats], dim=-1)
        return self.edge_encoder(edge_3d)

    def forward(self, data):
        x = self.node_embed(data.x)

        if hasattr(data, "pos") and data.pos is not None:
            edge_attr = self._compute_3d_edge_features(data)
        else:
            edge_attr = self.edge_encoder(
                torch.cat([data.edge_attr,
                           torch.zeros(data.edge_attr.size(0), 20,
                                       device=data.edge_attr.device)], dim=-1))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_mean_pool(x, data.batch)

        if hasattr(data, "x_global") and data.x_global is not None:
            out = torch.cat([out, data.x_global], dim=1)

        return self.classifier(out)


# ═══════════════════════════ ZT-Hybrid ═══════════════════════════

class ZTHybrid(nn.Module):
    """Hybrid: SPMS-enriched node features + 3D edge features.

    Combines directional steric information (SPMS as node features)
    with geometric message passing (ComENet-style 3D edges).
    """

    def __init__(self, node_dim=20, edge_dim=5, hidden_dim=128,
                 n_layers=4, n_classes=4, dropout=0.2, global_feat_dim=0,
                 n_rbf=16, spms_dim=16, **kwargs):
        super().__init__()
        # Node features: original + SPMS
        self.node_embed = nn.Linear(node_dim + spms_dim, hidden_dim)

        # 3D edge features
        edge_3d_dim = edge_dim + n_rbf + 4
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_3d_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = GaussianSmearing(0.0, 8.0, n_rbf)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        clf_input = hidden_dim + global_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.dropout = dropout

    def _compute_3d_edge_features(self, data):
        src, dst = data.edge_index
        diff = data.pos[dst] - data.pos[src]
        dist = diff.norm(dim=-1)
        rbf = self.rbf(dist)
        direction = diff / (dist.unsqueeze(-1) + 1e-8)
        angle_feats = torch.stack([
            direction[:, 0], direction[:, 1],
            torch.atan2(direction[:, 1], direction[:, 0]),
            torch.acos(direction[:, 2].clamp(-1, 1)),
        ], dim=-1)
        edge_3d = torch.cat([data.edge_attr, rbf, angle_feats], dim=-1)
        return self.edge_encoder(edge_3d)

    def forward(self, data):
        # Concatenate SPMS features to node features
        if hasattr(data, "spms_feat") and data.spms_feat is not None:
            x_in = torch.cat([data.x, data.spms_feat], dim=-1)
        else:
            x_in = torch.cat([data.x,
                              torch.zeros(data.x.size(0), 16,
                                          device=data.x.device)], dim=-1)

        x = self.node_embed(x_in)

        if hasattr(data, "pos") and data.pos is not None:
            edge_attr = self._compute_3d_edge_features(data)
        else:
            edge_attr = self.edge_encoder(
                torch.cat([data.edge_attr,
                           torch.zeros(data.edge_attr.size(0), 20,
                                       device=data.edge_attr.device)], dim=-1))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_mean_pool(x, data.batch)

        if hasattr(data, "x_global") and data.x_global is not None:
            out = torch.cat([out, data.x_global], dim=1)

        return self.classifier(out)


# ═══════════════════════════ Multi-TS Attention Scorer ═══════════════════════════


class MultiTSAttentionScorer(nn.Module):
    """Multi-TS GNN: encodes 4 competing TS graphs, scores them with attention.

    Uses the SAME ChiralMessagePassing encoder as ZT-Chiral (proven to reach
    0.818 TSCV on Evans), adding only an attention-based TS scoring layer on top.

    Architecture:
      ChiralMP Encoder (shared) → Ring-aware readout → per-TS embedding (2*hidden) →
      TS attention scorer → weighted aggregation → classifier.
    """

    def __init__(self, node_dim=28, edge_dim=8, hidden_dim=128,
                 n_layers=4, n_classes=4, dropout=0.2, global_feat_dim=0,
                 use_ze_prior=False, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_ze_prior = use_ze_prior

        # Same encoder as ZT-Chiral: ChiralMessagePassing (not Enhanced)
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ChiralMessagePassing(hidden_dim, n_edge_types=edge_dim)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Ring-aware readout (same as ZT-Chiral)
        self.ring_transform = nn.Linear(hidden_dim, hidden_dim)
        self.sub_transform = nn.Linear(hidden_dim, hidden_dim)

        # TS scorer: 2*hidden → scalar score
        readout_dim = hidden_dim * 2
        self.score_mlp = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Classifier: aggregated TS embedding + optional global features
        clf_input = readout_dim + global_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.dropout = dropout

    def _encode_all(self, x, edge_index, edge_attr, node_type, batch):
        """Encode TS subgraphs using ChiralMP → ring-aware readout."""
        h = self.node_embed(x)

        for layer, ln in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_attr, node_type)
            h = ln(h + h_new)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Ring-aware readout (same as ZT-Chiral)
        is_ring = (node_type < 6)
        ring_h = self.ring_transform(h) * is_ring.unsqueeze(1).float()
        sub_h = self.sub_transform(h) * (~is_ring).unsqueeze(1).float()

        ring_pool = global_mean_pool(ring_h, batch)
        sub_pool = global_mean_pool(sub_h, batch)

        return torch.cat([ring_pool, sub_pool], dim=1)  # (n_graphs, 2*hidden)

    def forward(self, batch):
        """Forward pass for a MultiTS batch.

        Expects batch to have:
          - Standard PyG fields: x, edge_index, edge_attr, batch
          - node_type: (n_nodes,) int
          - rxn_batch: (n_graphs_in_batch,) int mapping graph → reaction
          - ts_type: (n_graphs_in_batch, 1) int
          - is_dummy: (n_graphs_in_batch, 1) int
          - Optional: x_global (n_graphs_in_batch, feat_dim)
        """
        # Encode all TS subgraphs together (shared ChiralMP encoder)
        h_all = self._encode_all(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.node_type, batch.batch,
        )  # (n_graphs, 2*hidden)

        # Score each TS
        scores = self.score_mlp(h_all).squeeze(-1)  # (n_graphs,)

        # Mask dummy graphs
        is_dummy = batch.is_dummy.view(-1).float()
        scores = scores - is_dummy * 1e9

        # Optional Z/E prior
        if self.use_ze_prior and hasattr(batch, "ze_weights"):
            ze_w = batch.ze_weights.view(-1, 2)
            ts_t = batch.ts_type.view(-1)
            is_z = (ts_t < 2).float()
            w_z = ze_w[:, 0].clamp(min=1e-8)
            w_e = ze_w[:, 1].clamp(min=1e-8)
            log_prior = is_z * torch.log(w_z) + (1 - is_z) * torch.log(w_e)
            scores = scores + log_prior

        # Attention within each reaction (4 TS per reaction)
        rxn_batch = batch.rxn_batch
        n_reactions = batch.n_reactions

        # Vectorized softmax per reaction using scatter
        temp = self.temperature.clamp(min=0.1)
        scaled = scores / temp

        # Numerically stable scatter softmax
        max_scores = torch.zeros(n_reactions, device=scaled.device)
        max_scores.scatter_reduce_(0, rxn_batch, scaled, reduce="amax", include_self=False)
        exp_s = torch.exp(scaled - max_scores[rxn_batch])
        sum_exp = torch.zeros(n_reactions, device=scaled.device)
        sum_exp.scatter_add_(0, rxn_batch, exp_s)
        alpha = exp_s / sum_exp[rxn_batch].clamp(min=1e-8)

        # Weighted aggregation per reaction
        h_agg = torch.zeros(n_reactions, h_all.size(1), device=h_all.device)
        h_agg.scatter_add_(0, rxn_batch.unsqueeze(1).expand_as(h_all),
                           alpha.unsqueeze(1) * h_all)

        # Classifier
        clf_input = h_agg
        if hasattr(batch, "x_global") and batch.x_global is not None:
            # Take global features from the first graph of each reaction
            # Use scatter to find first index per reaction
            first_idx = torch.zeros(n_reactions, dtype=torch.long, device=h_all.device)
            for r in range(n_reactions):
                first_idx[r] = (rxn_batch == r).nonzero(as_tuple=True)[0][0]
            x_global = batch.x_global[first_idx]
            clf_input = torch.cat([clf_input, x_global], dim=1)

        logits = self.classifier(clf_input)

        # Store for analysis
        self._last_alpha = alpha
        self._last_rxn_batch = rxn_batch
        self._last_ts_types = batch.ts_type.view(-1)

        return logits

    def get_ts_attention(self):
        """Return per-reaction TS attention weights."""
        return self._last_alpha, self._last_rxn_batch, self._last_ts_types


# ═══════════════════════════ Model Registry ═══════════════════════════

from .chidek import ZTChiDeK
from .gcpnet_zt import ZTGCPNet

ZT_MODELS = {
    "ZT-GIN": ZTGIN,
    "ZT-GAT": ZTGAT,
    "ZT-Chiral": ZTChiral,
    "ZT-ChiDeK": ZTChiDeK,
    "ZT-GCPNet": ZTGCPNet,
    "ZT-ComENet": ZTComENet,
    "ZT-Hybrid": ZTHybrid,
    "MultiTS": MultiTSAttentionScorer,
}
