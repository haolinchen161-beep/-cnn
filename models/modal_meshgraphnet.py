from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

DEFAULT_NODE_FEATURE_DIM = 21
DEFAULT_EDGE_FEATURE_DIM = 4


class MLP(nn.Module):
    """与旧 gnn-meshgraphnet-refactor 分支一致的轻量 MLP。

    隐藏层只做 Linear + GELU + Dropout，不在每个隐藏层后做 LayerNorm。
    对 10 万级边数的图来说，隐藏层 LayerNorm 会明显拖慢训练。
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 n_layers: int = 2,
                 dropout: float = 0.0,
                 layer_norm: bool = True,
                 final_zero: bool = False):
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
    """MeshGraphNet 残差消息传递层：先更新边特征，再聚合到节点。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = MLP(hidden_dim * 3, hidden_dim, hidden_dim, n_layers=2, dropout=dropout, layer_norm=True)
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, n_layers=2, dropout=dropout, layer_norm=True)

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
    """单调频率输出头。

    先预测第一阶频率，再预测正的频率间隔，保证输出频率按阶次递增。
    默认训练前三阶，也支持后续扩展到 K 阶。
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
    """节点隐变量 + 全局隐变量 + 模态 token → Z 向振型。"""

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
    """轻量 Z-only MeshGraphNet 模态预测模型。

    输出：
        omega: [B,K]，单位 rad/s
        phi_z: [total_N,K]

    本阶段只预测 Z 向振型，用于先验证 Z-Z FRF 需要的核心模态分量。
    阻尼和 FRF 重建不放在网络训练目标中。
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

        self.node_encoder = MLP(node_in_dim, hidden, hidden, n_layers=3, dropout=dropout, layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden, hidden, n_layers=3, dropout=dropout, layer_norm=True)
        self.blocks = nn.ModuleList([MeshGraphBlock(hidden, dropout=dropout) for _ in range(n_layers)])
        self.global_proj = MLP(hidden * 2, hidden, hidden, n_layers=2, dropout=dropout, layer_norm=True)
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
