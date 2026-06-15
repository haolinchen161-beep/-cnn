"""
meshgraphnet_frf_model.py — FEM-aware MeshGraphNet modal surrogate.

This branch is intentionally rebuilt from the current CNN physics pipeline:
    mesh graph → omega / zeta / 3D phi → PhysicsDecoder → FRF

The old Z-only GNN implementation is replaced.  The model now predicts full
node-level 3D mode shapes [N, K, 3] and keeps the same modal bottleneck used by
the CNN baseline.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import PhysicsDecoder


DEFAULT_NODE_FEATURE_DIM = 25
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
    """MeshGraphNet-style residual message passing block.

    Edge update:
        e_ij <- e_ij + MLP(e_ij, h_i, h_j)

    Node update:
        h_j <- h_j + MLP(h_j, mean_i e_ij)
    """

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

        h_update = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = h + h_update
        return h, e


class PhysicsPriorOmegaHead(nn.Module):
    """Frequency head copied from the current CNN idea: physics prior + graph residual.

    It predicts f1, gap21, gap32 and then reconstructs monotonically increasing
    natural frequencies.  Output is physical rad/s.
    """

    def __init__(self, hidden: int = 128, n_modes: int = 3, phys_dim: int = 14):
        super().__init__()
        if n_modes != 3:
            raise ValueError("PhysicsPriorOmegaHead currently expects n_modes=3.")
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
        self.g32_min, self.g32_max = 150.0, 1000.0

        self.f1_span = self.f1_max - self.f1_min
        self.g21_span = self.g21_max - self.g21_min
        self.g32_span = self.g32_max - self.g32_min

        def inv_sigmoid(p):
            p = torch.tensor(p).clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1.0 - p))

        # Same initialization used by the current CNN branch.
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
        delta_raw = 0.35 * torch.tanh(
            self.delta_mlp(torch.cat([graph_latent, phys_features], dim=-1))
        )
        raw = prior_raw + delta_raw
        s = torch.sigmoid(raw)

        f1 = self.f1_min + self.f1_span * s[:, 0:1]
        g21 = self.g21_min + self.g21_span * s[:, 1:2]
        g32 = self.g32_min + self.g32_span * s[:, 2:3]
        f2 = f1 + g21
        f3 = f2 + g32
        f_hz = torch.cat([f1, f2, f3], dim=-1)
        return f_hz * (2.0 * torch.pi)


class ZetaHead(nn.Module):
    """Damping head: frequency + K/C/material prior plus graph residual."""

    def __init__(self, hidden: int = 128, n_modes: int = 3):
        super().__init__()
        self.n_modes = n_modes
        self.prior_mlp = nn.Sequential(
            nn.Linear(n_modes + 4, 64),
            nn.GELU(),
            nn.Linear(64, n_modes),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden + n_modes + 4, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, n_modes),
        )

        with torch.no_grad():
            nn.init.zeros_(self.prior_mlp[-1].weight)
            self.prior_mlp[-1].bias.fill_(-2.197)
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(self, graph_latent: torch.Tensor, omega_norm: torch.Tensor,
                zeta_phys_features: torch.Tensor):
        phys_input = torch.cat([omega_norm, zeta_phys_features], dim=-1)
        prior_raw = self.prior_mlp(phys_input)
        delta_raw = torch.tanh(self.delta_mlp(torch.cat([graph_latent, phys_input], dim=-1)))
        raw = prior_raw + delta_raw
        zeta = torch.sigmoid(raw) * 0.030 + 0.001
        return torch.log(zeta), zeta


class DirectionBranchHead(nn.Module):
    """Predict each mode's XYZ energy ratio. Used for KL supervision and phi conditioning."""

    def __init__(self, hidden: int = 128, n_modes: int = 3):
        super().__init__()
        self.n_modes = n_modes
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, n_modes * 3),
        )

    def forward(self, graph_latent: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(graph_latent).view(graph_latent.shape[0], self.n_modes, 3)
        return F.log_softmax(logits, dim=-1)


class PhiScaleHead(nn.Module):
    """Per-mode, per-direction amplitude scale [B,K,3]."""

    def __init__(self, hidden: int = 128, n_modes: int = 3):
        super().__init__()
        self.n_modes = n_modes
        self.mlp = nn.Sequential(
            nn.Linear(hidden + n_modes * 3, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_modes * 3),
        )
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, graph_latent: torch.Tensor, branch_probs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([graph_latent, branch_probs.reshape(graph_latent.shape[0], -1)], dim=-1)
        raw = self.mlp(x).view(-1, self.n_modes, 3)
        raw = torch.clamp(raw, -2.0, 2.0)
        return torch.exp(raw)


class ModeTokenPhiDecoder(nn.Module):
    """Decode node latent + graph latent + mode token to full 3D mode shape."""

    def __init__(self, hidden: int = 128, n_modes: int = 3, dropout: float = 0.05):
        super().__init__()
        self.n_modes = n_modes
        self.mode_tokens = nn.Parameter(torch.randn(n_modes, hidden) * 0.02)
        # node h, graph g, mode token, branch prob for this mode
        in_dim = hidden * 3 + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, node_latent: torch.Tensor, graph_latent: torch.Tensor,
                batch: torch.Tensor, branch_probs: torch.Tensor) -> torch.Tensor:
        n = node_latent.shape[0]
        k = self.n_modes
        g_node = graph_latent[batch]

        node_expand = node_latent.unsqueeze(1).expand(n, k, -1)
        graph_expand = g_node.unsqueeze(1).expand(n, k, -1)
        mode_expand = self.mode_tokens.unsqueeze(0).expand(n, k, -1)
        branch_expand = branch_probs[batch]  # [N,K,3]

        x = torch.cat([node_expand, graph_expand, mode_expand, branch_expand], dim=-1)
        return self.mlp(x.reshape(n * k, -1)).view(n, k, 3)


class MeshGraphFRFModel(nn.Module):
    """FEM-aware MeshGraphNet + modal bottleneck + physics decoder."""

    def __init__(self,
                 node_in_dim: int = DEFAULT_NODE_FEATURE_DIM,
                 edge_in_dim: int = DEFAULT_EDGE_FEATURE_DIM,
                 hidden: int = 128,
                 n_layers: int = 4,
                 n_modes: int = 3,
                 amp_scale: float = 500000.0,
                 freq_min: float = 1.0,
                 freq_max: float = 5000.0,
                 dropout: float = 0.05):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden = hidden
        self.n_modes = n_modes

        self.node_encoder = MLP(node_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.edge_encoder = MLP(edge_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.blocks = nn.ModuleList([MeshGraphBlock(hidden, dropout=dropout) for _ in range(n_layers)])
        self.global_proj = MLP(hidden * 2, hidden, hidden, n_layers=2, dropout=dropout)

        self.omega_head = PhysicsPriorOmegaHead(hidden, n_modes, phys_dim=14)
        self.zeta_head = ZetaHead(hidden, n_modes)
        self.branch_head = DirectionBranchHead(hidden, n_modes)
        self.phi_decoder = ModeTokenPhiDecoder(hidden, n_modes, dropout=dropout)
        self.phi_scale_head = PhiScaleHead(hidden, n_modes)

        self.physics = PhysicsDecoder(amp_scale=amp_scale, freq_min=freq_min, freq_max=freq_max)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                frequencies: Optional[torch.Tensor] = None,
                phi_exc: Optional[torch.Tensor] = None,
                excitation_index_global: Optional[torch.Tensor] = None,
                force_vector: Optional[torch.Tensor] = None,
                alpha: float = 1.0,
                omega_true: Optional[torch.Tensor] = None,
                detach_modal_for_frf: bool = True):
        if node_features.shape[-1] != self.node_in_dim:
            raise ValueError(
                f"node_features dim mismatch: got {node_features.shape[-1]}, expected {self.node_in_dim}."
            )
        if edge_attr.shape[-1] != self.edge_in_dim:
            raise ValueError(
                f"edge_attr dim mismatch: got {edge_attr.shape[-1]}, expected {self.edge_in_dim}."
            )

        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            h, e = block(h, edge_index, e)

        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        graph_latent = self.global_proj(torch.cat([g_mean, g_max], dim=-1))

        omega_phys_features, zeta_phys_features = build_graph_physics_features(node_features, batch)
        omega_phys = self.omega_head(graph_latent, omega_phys_features)

        omega_norm = omega_phys.detach() / 5000.0
        log_zeta, zeta = self.zeta_head(graph_latent, omega_norm, zeta_phys_features)

        self.branch_log_probs = self.branch_head(graph_latent)  # [B,K,3]
        branch_probs = torch.exp(self.branch_log_probs)
        phi_raw = self.phi_decoder(h, graph_latent, batch, branch_probs)  # [N,K,3]

        phi = normalize_phi_per_graph(phi_raw, batch)
        phi_scale = self.phi_scale_head(graph_latent, branch_probs)       # [B,K,3]
        phi = phi * phi_scale[batch]

        frf = None
        if frequencies is not None:
            phi_z = phi[..., 2]
            if phi_exc is not None and phi_exc.dim() == 3:
                phi_exc_used = phi_exc[..., 2]
            elif phi_exc is not None:
                phi_exc_used = phi_exc
            elif excitation_index_global is not None:
                phi_exc_used = phi_z[excitation_index_global]
            else:
                phi_exc_used = None

            omega_used = omega_true if omega_true is not None else omega_phys
            if detach_modal_for_frf:
                omega_frf = omega_used.detach()
                zeta_frf = zeta.detach()
            else:
                omega_frf = omega_used
                zeta_frf = zeta

            frf = self.physics(phi_z, omega_frf, zeta_frf, frequencies,
                               phi_exc_used, batch_idx=batch, alpha=alpha)

        return frf, omega_phys, log_zeta, zeta, phi


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
        if torch.any(mask):
            outs.append(x[mask].max(dim=0).values)
        else:
            outs.append(x.new_zeros(x.shape[-1]))
    return torch.stack(outs, dim=0) if outs else x.new_zeros(0, x.shape[-1])


def _safe_stats(x: torch.Tensor):
    if x.numel() == 0:
        z = x.new_tensor(0.0)
        return z, z, z, z
    return x.mean(), x.std(unbiased=False), x.min(), x.max()


def build_graph_physics_features(node_features: torch.Tensor, batch: torch.Tensor):
    """Build graph-level physics descriptors from the 25D node features.

    Layout from data.dataset:
        0:3 normalized xyz
        3:10 point_features = [E_ratio, PRXY, rho_ratio, is_fixed, logK, logC, Z/H]
        10:13 normalized spring_k_xyz
        13:16 normalized spring_c_xyz
        16:21 node_type one-hot: ordinary/bottom/cut/side/corner
    """
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    omega_feats, zeta_feats = [], []

    for g in range(n_graphs):
        m = batch == g
        nf = node_features[m]
        dtype = nf.dtype
        device = nf.device

        if nf.numel() == 0:
            omega_feats.append(torch.zeros(14, dtype=dtype, device=device))
            zeta_feats.append(torch.zeros(4, dtype=dtype, device=device))
            continue

        E_ratio = nf[:, 3].mean()
        rho_ratio = nf[:, 5].mean()
        z_h = nf[:, 9]
        is_fixed = nf[:, 6]
        logK = nf[:, 7]
        logC = nf[:, 8]

        spring_mask = (logK > 0) | (nf[:, 10:13].sum(dim=-1) > 0)
        fixed_ratio = (is_fixed > 0).float().mean()

        if nf.shape[1] >= 21:
            side_ratio = nf[:, 19].mean()
            corner_ratio = nf[:, 20].mean()
        else:
            side_ratio = fixed_ratio.new_tensor(0.0)
            corner_ratio = fixed_ratio.new_tensor(0.0)

        k_mean, k_std, k_min, k_max = _safe_stats(logK[spring_mask])
        c_mean, _, _, _ = _safe_stats(logC[spring_mask])

        z_mean = z_h.mean()
        z_std = z_h.std(unbiased=False)
        z_min = z_h.min()
        z_max = z_h.max()
        f_theory = z_mean * torch.sqrt(torch.clamp(E_ratio / (rho_ratio + 1e-6), min=1e-6))

        omega_feats.append(torch.stack([
            E_ratio, rho_ratio,
            z_mean, z_std, z_min, z_max,
            fixed_ratio, corner_ratio, side_ratio,
            k_mean, k_std, k_min, k_max,
            f_theory,
        ]))

        zeta_feats.append(torch.stack([k_mean, c_mean, E_ratio, rho_ratio]))

    return torch.stack(omega_feats, dim=0), torch.stack(zeta_feats, dim=0)


def normalize_phi_per_graph(phi: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Joint std normalization per graph and per mode over nodes and XYZ."""
    out = torch.empty_like(phi)
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    for g in range(n_graphs):
        m = batch == g
        p = phi[m]
        if p.numel() == 0:
            continue
        std = torch.std(p.transpose(0, 1).reshape(p.shape[1], -1), dim=1) + 1e-8
        out[m] = p / std.view(1, -1, 1)
    return out
