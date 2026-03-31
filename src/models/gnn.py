"""
GNN models for molecular property prediction.

Models:
  - AttentiveFPModel  : AttentiveFP (PyG) or GCN fallback. Default featurization.
  - SMERGCNModel      : RGCN + smask-weighted pooling. SME featurization (Wu et al. 2023).

Both expose:
  forward(data) -> (pred: Tensor[B], latent: Tensor[B, D])
  get_latent_dim() -> int
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

try:
    from torch_geometric.nn.models import AttentiveFP
except ImportError:  # pragma: no cover
    AttentiveFP = None

try:
    from torch_geometric.nn import RGCNConv
except ImportError:  # pragma: no cover
    RGCNConv = None

from src.features.graph import SME_NUM_RELS


# ---------------------------------------------------------------------------
# AttentiveFP model  (default featurization)
# ---------------------------------------------------------------------------

class _FallbackGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.reg_head = nn.Linear(hidden_dim, 1)
        self._latent_dim = hidden_dim

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        latent = global_mean_pool(x, batch)
        pred = self.reg_head(latent).view(-1)
        return pred, latent

    def get_latent_dim(self):
        return self._latent_dim


class AttentiveFPModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        edge_dim: int = 6,
        hidden_dim: int = 200,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._latent_dim = hidden_dim
        self.use_attentive = AttentiveFP is not None

        if self.use_attentive:
            self.backbone = AttentiveFP(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim,
                num_layers=num_layers,
                num_timesteps=2,
                dropout=dropout,
            )
            self.reg_head = nn.Linear(hidden_dim, 1)
        else:
            self.backbone = _FallbackGCN(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                layers=num_layers,
                dropout=dropout,
            )

    def forward(self, data):
        if self.use_attentive:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            latent = self.backbone(x, edge_index, edge_attr, batch)
            pred = self.reg_head(latent).view(-1)
            return pred, latent
        return self.backbone(data)

    def get_latent_dim(self) -> int:
        return self._latent_dim


# ---------------------------------------------------------------------------
# SME RGCN model  (Wu et al. 2023, ported DGL → PyG)
# ---------------------------------------------------------------------------

class _SMERGCNLayer(nn.Module):
    """Single RGCN layer with residual connection and batch normalization.

    Mirrors SME's RGCNLayer (maskgnn.py) but uses PyG RGCNConv.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_rels: int = SME_NUM_RELS,
        dropout: float = 0.25,
        residual: bool = True,
        batchnorm: bool = True,
    ):
        super().__init__()
        if RGCNConv is None:
            raise ImportError("torch_geometric is required for SMERGCNModel. "
                              "Install with: pip install torch-geometric")
        self.conv = RGCNConv(in_feats, out_feats, num_relations=num_rels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.residual = residual
        if residual:
            self.res_lin = nn.Linear(in_feats, out_feats)

        self.use_bn = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_type)
        out = self.act(out)
        out = self.dropout(out)
        if self.residual:
            out = out + self.act(self.res_lin(x))
        if self.use_bn:
            out = self.bn(out)
        return out


class _SMEWeightedPool(nn.Module):
    """Substructure-mask-gated attention pooling.

    Mirrors SME's WeightAndSum (maskgnn.py).
    Graph-level representation = sum_i( sigmoid(W h_i) * smask_i * h_i )
    """

    def __init__(self, in_feats: int):
        super().__init__()
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,          # (N, F)
        smask: torch.Tensor,       # (N, 1) — per-atom substructure mask
        batch: torch.Tensor,       # (N,)   — graph assignment
    ) -> torch.Tensor:             # (B, F)
        weight = self.atom_weighting(x) * smask  # (N, 1)
        weighted = x * weight                    # (N, F)

        # Scatter sum over graphs
        B = int(batch.max().item()) + 1
        out = torch.zeros(B, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted), weighted)
        return out


class SMERGCNModel(nn.Module):
    """RGCN encoder with substructure-mask-weighted pooling + MLP regression head.

    Architecture (mirrors SME's RGCN + BaseGNN):
      RGCNLayer × N  →  WeightedPool (smask)  →  FC × 3  →  scalar prediction

    Args:
        in_feats:      Atom feature dim. Default = SME_NODE_DIM (40).
        hidden_feats:  List of RGCN hidden dims per layer.
        ffn_hidden:    Hidden dim for the 3-layer MLP head.
        num_rels:      Number of bond-relation types for RGCN.
        rgcn_dropout:  Dropout inside RGCN layers.
        ffn_dropout:   Dropout inside MLP head.
    """

    def __init__(
        self,
        in_feats: int = 40,
        hidden_feats: list = None,
        ffn_hidden: int = 200,
        ffn_dims: list = None,
        num_rels: int = SME_NUM_RELS,
        rgcn_dropout: float = 0.25,
        ffn_dropout: float = 0.25,
    ):
        super().__init__()
        if hidden_feats is None:
            hidden_feats = [200, 200]

        # RGCN layers
        self.rgcn_layers = nn.ModuleList()
        cur_dim = in_feats
        for h in hidden_feats:
            self.rgcn_layers.append(
                _SMERGCNLayer(cur_dim, h, num_rels=num_rels,
                              dropout=rgcn_dropout, residual=True, batchnorm=True)
            )
            cur_dim = h

        # Smask-weighted pooling (readout)
        self.readout = _SMEWeightedPool(cur_dim)

        def _fc(in_d, out_d):
            return nn.Sequential(
                nn.Dropout(ffn_dropout),
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.BatchNorm1d(out_d),
            )

        if ffn_dims is not None:
            # Funnel architecture: ffn_dims = [128, 64, latent_dim]
            self.fc_layers = nn.ModuleList()
            in_d = cur_dim
            for out_d in ffn_dims:
                self.fc_layers.append(_fc(in_d, out_d))
                in_d = out_d
            self.out_layer = nn.Linear(in_d, 1)
            self._latent_dim = in_d
            self._use_ffn_dims = True
        else:
            # Legacy flat architecture: 3 layers of ffn_hidden
            self.fc1 = _fc(cur_dim, ffn_hidden)
            self.fc2 = _fc(ffn_hidden, ffn_hidden)
            self.fc3 = _fc(ffn_hidden, ffn_hidden)
            self.out_layer = nn.Linear(ffn_hidden, 1)
            self._latent_dim = ffn_hidden
            self._use_ffn_dims = False

    def forward(self, data):
        """
        Args:
            data: PyG Data / Batch with fields:
                  x          (N, in_feats)
                  edge_index (2, E)
                  edge_type  (E,)  — integer RGCN bond type
                  smask      (N, 1) — substructure mask
                  batch      (N,)   — optional, defaults to single graph

        Returns:
            pred   : (B,) float
            latent : (B, ffn_hidden) float
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type
        smask = data.smask  # (N, 1)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # RGCN message passing
        for layer in self.rgcn_layers:
            x = layer(x, edge_index, edge_type)

        # Smask-weighted graph readout
        graph_feat = self.readout(x, smask, batch)  # (B, hidden)

        # MLP head
        if self._use_ffn_dims:
            h = graph_feat
            for layer in self.fc_layers:
                h = layer(h)
            latent = h
        else:
            h = self.fc1(graph_feat)
            h = self.fc2(h)
            latent = self.fc3(h)
        pred = self.out_layer(latent).view(-1)

        return pred, latent

    def get_latent_dim(self) -> int:
        return self._latent_dim


# Allow type annotation in the class body above
