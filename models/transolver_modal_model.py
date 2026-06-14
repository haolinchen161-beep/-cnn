"""Transolver-Modal：面向模态参数与 FRF 重建的轻量节点/网格模型。

本文件替换旧版 Transolver 实验代码，保留“节点输入 + slice token 注意力”的核心思想，
但训练接口和物理监督口径对齐当前已经验证正确的 CNN 版本：

1. 图级输出：前三阶固有圆频率 omega、阻尼比 zeta；
2. 节点级输出：三维振型 phi_xyz [total_N, K, 3]；
3. FRF 由 PhysicsDecoder/ModalFRFDecoder 通过模态叠加公式重建；
4. omega 使用 f1 + gap21 + gap32 的单调频率头，初始化在数据均值附近；
5. phi 使用显式 mode tokens，避免把三阶模态当作普通 9 个通道回归；
6. 不在模型内部做 batch 混合 loss，所有 MAC/phi/std/符号对齐都交给 training/losses.py 逐图计算。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import ModalFRFDecoder


# ---------------------------------------------------------------------------
# 变长 batch padding 工具
# ---------------------------------------------------------------------------

def pad_batch(x: torch.Tensor, node_counts: list[int]):
    """把 [total_N, C] padding 成 [B, Nmax, C]。"""
    bsz = len(node_counts)
    max_n = max(int(c) for c in node_counts)
    out = x.new_zeros(bsz, max_n, *x.shape[1:])
    mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=x.device)
    ptr = 0
    for b, c in enumerate(node_counts):
        c = int(c)
        out[b, :c] = x[ptr:ptr + c]
        mask[b, :c] = True
        ptr += c
    return out, mask


def unpad_batch(x_dense: torch.Tensor, node_counts: list[int]) -> torch.Tensor:
    """把 [B, Nmax, C] 还原为 [total_N, C]。"""
    return torch.cat([x_dense[b, :int(c)] for b, c in enumerate(node_counts)], dim=0)


def sanitize(x: torch.Tensor, clamp_value: float = 20.0) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    return x.clamp(-clamp_value, clamp_value)


# ---------------------------------------------------------------------------
# 轻量图边 stem：只做局部邻域预混合，不把它当完整 GNN 使用
# ---------------------------------------------------------------------------

class EdgeStem(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x
        src, dst = edge_index[0].long(), edge_index[1].long()
        msg = self.msg(torch.cat([x[src], x[dst] - x[src]], dim=-1))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        deg = x.new_zeros(x.shape[0])
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.norm(x + agg)


class SliceTransolverBlock(nn.Module):
    """Transolver 风格 physics-slice token block。

    节点先软分配到 S 个 slice token，token 之间做 self-attention，再广播回节点。
    复杂度近似 O(N*S + S^2)，避免普通全节点 attention 的 O(N^2)。
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, dropout: float = 0.1):
        super().__init__()
        self.assign = nn.Linear(hidden_dim, num_slices)
        self.token_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.node_norm1 = nn.LayerNorm(hidden_dim)
        self.node_norm2 = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, Nmax, H], mask: [B, Nmax]
        logits = self.assign(x).masked_fill(~mask.unsqueeze(-1), -1e4)
        assign = torch.softmax(logits, dim=1).masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = assign.sum(dim=1).clamp_min(1e-6)  # [B, S]
        tokens = torch.bmm(assign.transpose(1, 2), x) / denom.unsqueeze(-1)

        t = self.token_norm(tokens)
        t, _ = self.token_attn(t, t, t, need_weights=False)
        back = torch.bmm(assign, t)

        y = self.node_norm1(x + back)
        y = self.node_norm2(y + self.ffn(y))
        return y.masked_fill(~mask.unsqueeze(-1), 0.0)


# ---------------------------------------------------------------------------
# 物理头：沿用当前 CNN 中已经稳定的频率/阻尼参数化思想
# ---------------------------------------------------------------------------

class OmegaHead(nn.Module):
    """单调频率头：预测 f1 + gap21 + gap32，再转 rad/s。"""

    def __init__(self, hidden_dim: int, aux_dim: int = 4, n_modes: int = 3):
        super().__init__()
        if n_modes != 3:
            raise ValueError("当前 OmegaHead 只支持前三阶 n_modes=3")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + aux_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_modes),
        )

        # 当前凹槽板数据范围，单位 Hz，与 CNN 版本保持一致。
        self.f1_min, self.f1_max = 700.0, 1250.0
        self.g21_min, self.g21_max = 700.0, 2600.0
        self.g32_min, self.g32_max = 150.0, 1000.0
        self.f1_span = self.f1_max - self.f1_min
        self.g21_span = self.g21_max - self.g21_min
        self.g32_span = self.g32_max - self.g32_min

        def inv_sigmoid(p: float):
            p = torch.tensor(p).clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1.0 - p))

        b1 = inv_sigmoid((957.0 - self.f1_min) / self.f1_span)
        b2 = inv_sigmoid((1632.0 - self.g21_min) / self.g21_span)
        b3 = inv_sigmoid((388.0 - self.g32_min) / self.g32_span)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([b1, b2, b3]))
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, graph_latent: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        out = self.mlp(torch.cat([graph_latent, aux], dim=-1))
        s = torch.sigmoid(out)
        f1 = self.f1_min + self.f1_span * s[:, 0:1]
        g21 = self.g21_min + self.g21_span * s[:, 1:2]
        g32 = self.g32_min + self.g32_span * s[:, 2:3]
        f2 = f1 + g21
        f3 = f2 + g32
        f_hz = torch.cat([f1, f2, f3], dim=-1)
        return f_hz * (2.0 * torch.pi)


class ZetaHead(nn.Module):
    """阻尼头：graph token + omega + 边界 C 统计 → ζ。"""

    def __init__(self, hidden_dim: int, n_modes: int = 3, aux_dim: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_modes + aux_dim, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, n_modes),
        )

    def forward(self, graph_latent: torch.Tensor, omega: torch.Tensor, aux: torch.Tensor):
        # omega 用 Hz/5000 尺度喂给阻尼头，防止 rad/s 数值过大。
        f_hz_scaled = omega / (2.0 * torch.pi * 5000.0)
        out = self.mlp(torch.cat([graph_latent, f_hz_scaled.detach(), aux], dim=-1))
        zeta = torch.sigmoid(out) * 0.030 + 0.001
        return torch.log(zeta), zeta


class ModeTokenPhiHead(nn.Module):
    """节点振型头：node latent + graph latent + mode token → phi_xyz。"""

    def __init__(self, hidden_dim: int, n_modes: int = 3):
        super().__init__()
        self.n_modes = n_modes
        self.mode_tokens = nn.Parameter(torch.randn(n_modes, hidden_dim) * 0.02)
        self.local = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )
        self.scale = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.GELU(),
            nn.Linear(128, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.scale[-1].weight)
            nn.init.zeros_(self.scale[-1].bias)

    def forward(self,
                node_latent: torch.Tensor,
                graph_latent: torch.Tensor,
                batch: torch.Tensor,
                node_counts: list[int]) -> torch.Tensor:
        total_n, hidden = node_latent.shape
        bsz = graph_latent.shape[0]
        g_node = graph_latent[batch.long()]  # [total_N, H]

        node_expand = node_latent.unsqueeze(1).expand(total_n, self.n_modes, hidden)
        graph_expand = g_node.unsqueeze(1).expand(total_n, self.n_modes, hidden)
        mode_expand = self.mode_tokens.unsqueeze(0).expand(total_n, self.n_modes, hidden)
        raw = self.local(torch.cat([node_expand, graph_expand, mode_expand], dim=-1))

        # 每图每阶联合 std 归一化，再乘 graph-mode scale，等价于 CNN 中的形状/尺度解耦思想。
        scale_in = torch.cat([
            graph_latent.unsqueeze(1).expand(bsz, self.n_modes, hidden),
            self.mode_tokens.unsqueeze(0).expand(bsz, self.n_modes, hidden),
        ], dim=-1)
        scale = torch.exp(self.scale(scale_in).squeeze(-1)).clamp(0.05, 20.0)  # [B,K]

        out_parts = []
        ptr = 0
        for b, c in enumerate(node_counts):
            c = int(c)
            p = raw[ptr:ptr + c]
            std = torch.std(p.transpose(0, 1).reshape(self.n_modes, -1), dim=1).clamp_min(1e-6)
            p = p / std.view(1, self.n_modes, 1)
            p = p * scale[b].view(1, self.n_modes, 1)
            out_parts.append(p)
            ptr += c
        return torch.cat(out_parts, dim=0)


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class TransolverModalFRF(nn.Module):
    """轻量 Transolver-Modal 模型。"""

    def __init__(self,
                 in_dim: int = 28,
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 n_slices: int = 32,
                 n_modes: int = 3,
                 dropout: float = 0.1,
                 use_edge_stem: bool = True,
                 amp_scale: float = 500000.0,
                 response_direction: str = "Z",
                 force_direction: str = "Z",
                 **unused):
        super().__init__()
        self.n_modes = n_modes
        self.use_edge_stem = use_edge_stem
        self.response_direction = response_direction.upper()
        self.force_direction = force_direction.upper()
        self.response_dir_index = {"X": 0, "Y": 1, "Z": 2}.get(self.response_direction, 2)
        self.force_dir_index = {"X": 0, "Y": 1, "Z": 2}.get(self.force_direction, 2)

        self.input_proj = nn.Sequential(
            nn.Linear(3 + in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.edge_stem = EdgeStem(hidden_dim)
        self.blocks = nn.ModuleList([
            SliceTransolverBlock(hidden_dim, n_heads, n_slices, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.omega_head = OmegaHead(hidden_dim, aux_dim=4, n_modes=n_modes)
        self.zeta_head = ZetaHead(hidden_dim, n_modes=n_modes, aux_dim=2)
        self.phi_head = ModeTokenPhiHead(hidden_dim, n_modes=n_modes)
        self.physics = ModalFRFDecoder(amp_scale=amp_scale)

    def _graph_pool(self, x_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.pool_gate(x_dense).squeeze(-1).masked_fill(~mask, -1e4)
        gate = torch.softmax(logits, dim=1).masked_fill(~mask, 0.0)
        return (gate.unsqueeze(-1) * x_dense).sum(dim=1)

    def _graph_aux(self, node_features: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        """从节点特征中提取图级材料/边界统计。

        默认字段兼容 data/dataset.py：
        3:E/E0, 4:rho/rho0, 15..17:logK, 18..20:logC。
        """
        feat = sanitize(node_features, 20.0)
        device = feat.device
        dtype = feat.dtype
        fdim = feat.shape[-1]

        def col(idx: int, default: float):
            if fdim > idx:
                return feat[:, idx]
            return feat.new_full((feat.shape[0],), float(default))

        e = col(3, 1.0)
        rho = col(4, 1.0)
        logk = torch.stack([col(15, 0.0), col(16, 0.0), col(17, 0.0)], dim=-1)
        logc = torch.stack([col(18, 0.0), col(19, 0.0), col(20, 0.0)], dim=-1)
        logk_mean_node = logk.mean(dim=-1)
        logc_mean_node = logc.mean(dim=-1)
        logk_max_node = logk.max(dim=-1).values
        logc_max_node = logc.max(dim=-1).values

        sums = feat.new_zeros(num_graphs, 6)
        cnt = feat.new_zeros(num_graphs, 1)
        vals = torch.stack([e, rho, logk_mean_node, logk_max_node, logc_mean_node, logc_max_node], dim=-1)
        sums.index_add_(0, batch.long(), vals)
        cnt.index_add_(0, batch.long(), torch.ones(feat.shape[0], 1, device=device, dtype=dtype))
        mean = sums / cnt.clamp_min(1.0)
        omega_aux = mean[:, [0, 1, 2, 3]]
        zeta_aux = mean[:, [4, 5]]
        return omega_aux, zeta_aux

    def encode(self,
               points: torch.Tensor,
               node_features: torch.Tensor,
               batch: torch.Tensor,
               edge_index: torch.Tensor | None,
               node_counts: list[int]):
        feat = sanitize(node_features, 20.0).to(points.dtype)
        points = sanitize(points, 10.0).to(points.dtype)
        x = self.input_proj(torch.cat([points, feat], dim=-1))
        if self.use_edge_stem:
            x = self.edge_stem(x, edge_index)
        x_dense, mask = pad_batch(x, node_counts)
        for block in self.blocks:
            x_dense = block(x_dense, mask)
        node_latent = unpad_batch(x_dense, node_counts)
        graph_latent = self._graph_pool(x_dense, mask)
        return node_latent, graph_latent

    def forward(self,
                points: torch.Tensor,
                node_features: torch.Tensor,
                batch: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                boundary_c_xyz: torch.Tensor | None = None,
                excitation_index: torch.Tensor | None = None,
                frequencies: torch.Tensor | None = None,
                num_graphs: int | None = None,
                node_counts: list[int] | None = None,
                omega_true: torch.Tensor | None = None,
                physics_alpha: float = 1.0,
                **unused):
        if node_counts is None:
            node_counts = batch.bincount().detach().cpu().tolist()
        if num_graphs is None:
            num_graphs = len(node_counts)

        node_latent, graph_latent = self.encode(points, node_features, batch, edge_index, node_counts)
        omega_aux, zeta_aux = self._graph_aux(node_features, batch, num_graphs)

        omega = self.omega_head(graph_latent, omega_aux)
        log_zeta, zeta = self.zeta_head(graph_latent, omega, zeta_aux)
        phi_xyz = self.phi_head(node_latent, graph_latent, batch, node_counts)

        phi_response = phi_xyz[..., self.response_dir_index]
        phi_force = phi_xyz[..., self.force_dir_index]

        frf = None
        phi_exc_force = None
        if excitation_index is not None:
            phi_exc_force = phi_force[excitation_index.long()]  # [B,K]
            if frequencies is not None:
                omega_used = omega_true if omega_true is not None else omega
                # 与当前 CNN 版本一致：FRF 弱约束默认只修 phi，避免把 omega/zeta 拉崩。
                frf = self.physics(
                    phi_response,
                    phi_exc_force,
                    omega_used.detach(),
                    zeta.detach(),
                    frequencies,
                    batch,
                )

        return {
            'frf': frf,
            'modal_omega': omega,
            'modal_zeta': zeta,
            'log_zeta': log_zeta,
            'modal_phi_xyz': phi_xyz,
            'modal_phi_response': phi_response,
            'modal_phi_force': phi_force,
            'modal_phi_exc_force': phi_exc_force,
            'modal_phi_z': phi_xyz[..., 2],
            'response_dir_index': self.response_dir_index,
            'force_dir_index': self.force_dir_index,
        }
