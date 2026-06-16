from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


DEFAULT_NODE_FEATURE_DIM = 21
DEFAULT_EDGE_FEATURE_DIM = 4


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.0,
                 layer_norm: bool = True, final_zero: bool = False):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

        if final_zero:
            with torch.no_grad():
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class MeshGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = MLP(hidden_dim * 3, hidden_dim, hidden_dim,
                            n_layers=2, dropout=dropout, layer_norm=True)
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim,
                            n_layers=2, dropout=dropout, layer_norm=True)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class PhysicsPriorOmegaHead(nn.Module):
    def __init__(self, hidden: int = 128, n_modes: int = 3, phys_dim: int = 14):
        super().__init__()
        if n_modes != 3:
            raise ValueError("PhysicsPriorOmegaHead currently supports n_modes=3.")
        self.n_modes = n_modes
        self.prior_mlp = nn.Sequential(
            nn.Linear(phys_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, n_modes),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden + phys_dim, 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, n_modes),
        )

        self.f1_min, self.f1_max = 700.0, 1250.0
        self.g21_min, self.g21_max = 700.0, 2600.0
        self.g32_min, self.g32_max = 200.0, 1000.0
        self.f1_span = self.f1_max - self.f1_min
        self.g21_span = self.g21_max - self.g21_min
        self.g32_span = self.g32_max - self.g32_min

        def inv_sigmoid(p):
            p = torch.tensor(p).clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1.0 - p))

        b1 = inv_sigmoid((949.7 - self.f1_min) / self.f1_span)
        b2 = inv_sigmoid((1390.8 - self.g21_min) / self.g21_span)
        b3 = inv_sigmoid((522.1 - self.g32_min) / self.g32_span)
        with torch.no_grad():
            nn.init.zeros_(self.prior_mlp[-1].weight)
            self.prior_mlp[-1].bias.copy_(torch.tensor([b1, b2, b3]))
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(self, graph_latent: torch.Tensor, phys_features: torch.Tensor) -> torch.Tensor:
        prior_raw = self.prior_mlp(phys_features)
        delta_raw = 0.35 * torch.tanh(self.delta_mlp(torch.cat([graph_latent, phys_features], dim=-1)))
        raw = prior_raw + delta_raw
        s = torch.sigmoid(raw)
        f1 = self.f1_min + self.f1_span * s[:, 0:1]
        g21 = self.g21_min + self.g21_span * s[:, 1:2]
        g32 = self.g32_min + self.g32_span * s[:, 2:3]
        f_hz = torch.cat([f1, f1 + g21, f1 + g21 + g32], dim=-1)
        return f_hz * (2.0 * torch.pi)


class ModeTokenPhiZDecoder(nn.Module):
    def __init__(self, hidden: int = 128, n_modes: int = 3, dropout: float = 0.05):
        super().__init__()
        self.n_modes = n_modes
        self.mode_tokens = nn.Parameter(torch.randn(n_modes, hidden) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_latent: torch.Tensor, graph_latent: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        n = node_latent.shape[0]
        k = self.n_modes
        node_expand = node_latent.unsqueeze(1).expand(n, k, -1)
        graph_expand = graph_latent[batch].unsqueeze(1).expand(n, k, -1)
        mode_expand = self.mode_tokens.unsqueeze(0).expand(n, k, -1)
        x = torch.cat([node_expand, graph_expand, mode_expand], dim=-1)
        return self.mlp(x.reshape(n * k, -1)).view(n, k)


class MeshGraphZOnlyModel(nn.Module):
    def __init__(self, node_in_dim: int = DEFAULT_NODE_FEATURE_DIM,
                 edge_in_dim: int = DEFAULT_EDGE_FEATURE_DIM,
                 hidden: int = 128, n_layers: int = 4, n_modes: int = 3,
                 dropout: float = 0.05, **unused):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden = hidden
        self.n_modes = n_modes
        self.node_encoder = MLP(node_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.edge_encoder = MLP(edge_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.blocks = nn.ModuleList([MeshGraphBlock(hidden, dropout=dropout) for _ in range(n_layers)])
        self.global_proj = MLP(hidden * 2, hidden, hidden, n_layers=2, dropout=dropout)
        self.omega_head = PhysicsPriorOmegaHead(hidden, n_modes=n_modes, phys_dim=14)
        self.phi_decoder = ModeTokenPhiZDecoder(hidden, n_modes=n_modes, dropout=dropout)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                compute_phi: bool = True) -> Dict[str, torch.Tensor]:
        if node_features.shape[-1] != self.node_in_dim:
            raise ValueError(f"node_features dim mismatch: got {node_features.shape[-1]}, expected {self.node_in_dim}.")
        if edge_attr.shape[-1] != self.edge_in_dim:
            raise ValueError(f"edge_attr dim mismatch: got {edge_attr.shape[-1]}, expected {self.edge_in_dim}.")

        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            h, e = block(h, edge_index, e)

        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        graph_latent = self.global_proj(torch.cat([g_mean, g_max], dim=-1))
        omega_features = build_graph_physics_features(node_features, batch)
        omega = self.omega_head(graph_latent, omega_features)

        out = {"omega": omega, "node_latent": h, "graph_latent": graph_latent}
        if compute_phi:
            phi_z = self.phi_decoder(h, graph_latent, batch)
            out["phi_z"] = phi_z
            out["phi"] = phi_z
        return out


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = x.new_zeros(n_graphs, x.shape[-1])
    out.index_add_(0, batch, x)
    count = x.new_zeros(n_graphs, 1)
    count.index_add_(0, batch, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
    return out / count.clamp_min(1.0)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    outs = []
    for g in range(n_graphs):
        mask = batch == g
        outs.append(x[mask].max(dim=0).values if torch.any(mask) else x.new_zeros(x.shape[-1]))
    return torch.stack(outs, dim=0) if outs else x.new_zeros(0, x.shape[-1])


def _safe_stats(x: torch.Tensor):
    if x.numel() == 0:
        z = x.new_tensor(0.0)
        return z, z, z, z
    return x.mean(), x.std(unbiased=False), x.min(), x.max()


def build_graph_physics_features(node_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    feats = []
    for g in range(n_graphs):
        m = batch == g
        nf = node_features[m]
        dtype, device = nf.dtype, nf.device
        if nf.numel() == 0:
            feats.append(torch.zeros(14, dtype=dtype, device=device))
            continue
        e_ratio = nf[:, 3].mean()
        rho_ratio = nf[:, 4].mean()
        thickness = nf[:, 6]
        k_node = nf[:, 7:10].mean(dim=-1)
        spring_flag = nf[:, 10]
        spring_mask = spring_flag > 0
        side_ratio = nf[:, 14].mean()
        corner_ratio = nf[:, 15].mean()
        fixed_ratio = spring_flag.mean()
        k_mean, k_std, k_min, k_max = _safe_stats(k_node[spring_mask])
        th_mean = thickness.mean()
        th_std = thickness.std(unbiased=False)
        th_min = thickness.min()
        th_max = thickness.max()
        f_theory = th_mean * torch.sqrt(torch.clamp(e_ratio / (rho_ratio + 1e-6), min=1e-6))
        feats.append(torch.stack([
            e_ratio, rho_ratio,
            th_mean, th_std, th_min, th_max,
            fixed_ratio, corner_ratio, side_ratio,
            k_mean, k_std, k_min, k_max,
            f_theory,
        ]))
    return torch.stack(feats, dim=0)


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    kwargs = {"node_in_dim": DEFAULT_NODE_FEATURE_DIM, "edge_in_dim": DEFAULT_EDGE_FEATURE_DIM}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return MeshGraphZOnlyModel(**kwargs)
