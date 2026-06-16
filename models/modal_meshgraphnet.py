from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

DEFAULT_NODE_FEATURE_DIM = 21
DEFAULT_EDGE_FEATURE_DIM = 4


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 n_layers: int = 2,
                 dropout: float = 0.0,
                 layer_norm: bool = True,
                 final_layer_norm: bool = True):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        if final_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeshGraphBlock(nn.Module):
    """Residual MeshGraphNet block with edge and node updates."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = MLP(hidden_dim * 3, hidden_dim, hidden_dim, n_layers=2, dropout=dropout)
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, n_layers=2, dropout=dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return h, e

        src, dst = edge_index
        e_update = self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))
        e = e + e_update

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, e)

        deg = torch.zeros(h.shape[0], 1, dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones(dst.numel(), 1, dtype=h.dtype, device=h.device))
        agg = agg / deg.clamp_min(1.0)

        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h, e


class OmegaHead(nn.Module):
    """Monotonic modal frequency head.

    The first-stage setting uses n_modes=3. The implementation also supports
    larger K by predicting f1 and positive frequency gaps in Hz, then converts
    to rad/s.
    """

    def __init__(self, hidden: int, n_modes: int = 3):
        super().__init__()
        self.n_modes = int(n_modes)
        if self.n_modes < 1:
            raise ValueError("n_modes must be positive.")
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, self.n_modes),
        )
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, graph_latent: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(graph_latent)
        f1 = 200.0 + 1800.0 * torch.sigmoid(raw[:, 0:1])
        if self.n_modes == 1:
            freq_hz = f1
        else:
            gaps = 50.0 + 2600.0 * torch.sigmoid(raw[:, 1:])
            freq_hz = torch.cat([f1, f1 + torch.cumsum(gaps, dim=-1)], dim=-1)
        return freq_hz * (2.0 * torch.pi)


class ModeShapeZDecoder(nn.Module):
    """Decode node latent + graph latent + mode token to z-direction mode shapes."""

    def __init__(self, hidden: int, n_modes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_modes = int(n_modes)
        self.mode_tokens = nn.Parameter(torch.randn(self.n_modes, hidden) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_latent: torch.Tensor, graph_latent: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        n = node_latent.shape[0]
        k = self.n_modes

        node_expand = node_latent.unsqueeze(1).expand(n, k, -1)
        graph_expand = graph_latent[batch].unsqueeze(1).expand(n, k, -1)
        mode_expand = self.mode_tokens.unsqueeze(0).expand(n, k, -1)

        x = torch.cat([node_expand, graph_expand, mode_expand], dim=-1)
        return self.decoder(x.reshape(n * k, -1)).view(n, k)


class MeshModalNet(nn.Module):
    """Lightweight z-only modal MeshGraphNet.

    Outputs:
        omega: [B,K] rad/s
        phi_z: [total_N,K]

    The model intentionally predicts only the z-direction mode-shape component
    for the first-stage Z-Z FRF task. Damping and FRF reconstruction stay outside
    the training target.
    """

    def __init__(self,
                 node_in_dim: int = DEFAULT_NODE_FEATURE_DIM,
                 edge_in_dim: int = DEFAULT_EDGE_FEATURE_DIM,
                 hidden: int = 128,
                 n_layers: int = 6,
                 n_modes: int = 3,
                 dropout: float = 0.05,
                 **unused_kwargs):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden = hidden
        self.n_modes = int(n_modes)

        self.node_encoder = MLP(node_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.edge_encoder = MLP(edge_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.blocks = nn.ModuleList([MeshGraphBlock(hidden, dropout=dropout) for _ in range(n_layers)])
        self.global_proj = MLP(hidden * 2, hidden, hidden, n_layers=2, dropout=dropout)
        self.omega_head = OmegaHead(hidden, n_modes=self.n_modes)
        self.phi_decoder = ModeShapeZDecoder(hidden, n_modes=self.n_modes, dropout=dropout)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        if node_features.shape[-1] != self.node_in_dim:
            raise ValueError(f"node_features dim={node_features.shape[-1]}, expected {self.node_in_dim}")
        if edge_attr.shape[-1] != self.edge_in_dim:
            raise ValueError(f"edge_attr dim={edge_attr.shape[-1]}, expected {self.edge_in_dim}")

        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            h, e = block(h, edge_index, e)

        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        graph_latent = self.global_proj(torch.cat([g_mean, g_max], dim=-1))

        omega = self.omega_head(graph_latent)
        phi_z = self.phi_decoder(h, graph_latent, batch)

        return {
            "omega": omega,
            "phi_z": phi_z,
            # Backward-compatible alias. It is now [total_N,K], not xyz.
            "phi": phi_z,
            "node_latent": h,
            "graph_latent": graph_latent,
        }


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    out = x.new_zeros(n_graphs, x.shape[-1])
    out.index_add_(0, batch, x)
    count = x.new_zeros(n_graphs, 1)
    count.index_add_(0, batch, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
    return out / count.clamp_min(1.0)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    outs = []
    for graph_id in range(n_graphs):
        mask = batch == graph_id
        outs.append(x[mask].max(dim=0).values if torch.any(mask) else x.new_zeros(x.shape[-1]))
    return torch.stack(outs, dim=0) if outs else x.new_zeros(0, x.shape[-1])


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None) -> MeshModalNet:
    kwargs = {
        "node_in_dim": DEFAULT_NODE_FEATURE_DIM,
        "edge_in_dim": DEFAULT_EDGE_FEATURE_DIM,
    }
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return MeshModalNet(**kwargs)
