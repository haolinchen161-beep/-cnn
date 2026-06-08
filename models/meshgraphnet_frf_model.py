"""
meshgraphnet_frf_model.py — GNN / MeshGraphNet 模态参数预测模型。

核心思想：
    3D mesh graph + 边界/材料/装夹节点特征
        → MeshGraphNet message passing
        → global modal head: omega, zeta
        → node modal head: phi_z(node, mode)
        → PhysicsDecoder 重建 FRF

该文件不依赖 torch_geometric，便于在现有 PyTorch 环境中直接运行。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import PhysicsDecoder


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.0, layer_norm: bool = True):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class MeshGraphBlock(nn.Module):
    """MeshGraphNet 风格 edge update + node update residual block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = MLP(hidden_dim * 3, hidden_dim, hidden_dim, n_layers=2, dropout=dropout)
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, n_layers=2, dropout=dropout)

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


class MeshGraphFRFModel(nn.Module):
    """GNN/MeshGraphNet + modal bottleneck + physics decoder."""

    def __init__(self,
                 node_in_dim: int = 10,
                 edge_in_dim: int = 4,
                 hidden: int = 256,
                 n_layers: int = 8,
                 n_modes: int = 3,
                 omega_max: float = 25000.0,
                 zeta_min: float = 1e-4,
                 zeta_max: float = 0.08,
                 amp_scale: float = 500000.0,
                 freq_min: float = 1.0,
                 freq_max: float = 5000.0,
                 dropout: float = 0.05,
                 predict_delta_omega: bool = True):
        super().__init__()
        self.n_modes = n_modes
        self.omega_max = omega_max
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.predict_delta_omega = predict_delta_omega

        self.node_encoder = MLP(node_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.edge_encoder = MLP(edge_in_dim, hidden, hidden, n_layers=3, dropout=dropout)
        self.blocks = nn.ModuleList([MeshGraphBlock(hidden, dropout=dropout) for _ in range(n_layers)])

        self.global_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_modes * 2),
        )
        self.phi_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_modes),
        )
        self.physics = PhysicsDecoder(amp_scale=amp_scale, freq_min=freq_min, freq_max=freq_max)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                frequencies: Optional[torch.Tensor] = None,
                phi_exc: Optional[torch.Tensor] = None,
                alpha: float = 1.0):
        """
        Args:
            node_features: (total_N, node_in_dim)
            edge_index:    (2, total_E)
            edge_attr:     (total_E, edge_in_dim)
            batch:         (total_N,) graph index for each node
            frequencies:   (B, F) normalized frequency in [-1, 1], optional
            phi_exc:       (B, K) excitation modal value, optional
        Returns:
            frf, omega_norm, zeta, phi
        """
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            h, e = block(h, edge_index, e)

        g = global_mean_pool(h, batch)
        modal_raw = self.global_head(g)
        omega_raw = modal_raw[:, :self.n_modes]
        zeta_raw = modal_raw[:, self.n_modes:]

        if self.predict_delta_omega:
            delta = F.softplus(omega_raw) + 1e-6
            omega_norm = torch.cumsum(delta, dim=-1)
            omega_norm = omega_norm / (omega_norm[:, -1:].detach() + 1e-6)
            # 只归一化相对顺序会过于强；再给一个 sigmoid 幅值门控，保持 [0,1]
            scale = torch.sigmoid(omega_raw.mean(dim=-1, keepdim=True))
            omega_norm = omega_norm * scale
        else:
            omega_norm = torch.sigmoid(omega_raw)
            omega_norm, sort_idx = torch.sort(omega_norm, dim=-1)
            zeta_raw = torch.gather(zeta_raw, dim=-1, index=sort_idx)

        zeta = self.zeta_min + torch.sigmoid(zeta_raw) * (self.zeta_max - self.zeta_min)
        g_node = g[batch]
        phi = self.phi_head(torch.cat([h, g_node], dim=-1))

        frf = None
        if frequencies is not None:
            omega_phys = omega_norm * self.omega_max
            frf = self.physics(phi, omega_phys, zeta, frequencies, phi_exc,
                               batch_idx=batch, alpha=alpha)
        return frf, omega_norm, zeta, phi


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0:
        return x.new_zeros(0, x.shape[-1])
    n_graphs = int(batch.max().item()) + 1
    out = x.new_zeros(n_graphs, x.shape[-1])
    out.index_add_(0, batch, x)
    count = x.new_zeros(n_graphs, 1)
    count.index_add_(0, batch, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
    return out / count.clamp_min(1.0)
