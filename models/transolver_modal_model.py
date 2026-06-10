"""物理引导 Transolver 风格模型，用于模态 FRF 预测（Dense Padding 极速版）。

输入:
    非结构化 ANSYS 节点 + Transolver 节点特征 + 可选网格边。
输出:
    模态固有频率、模态阻尼比、逐节点 XYZ 三向振型，以及方向性 FRF。

全链路使用 Dense Padding + bmm 实现，0 隐式同步，100% 向量化。

阻尼比 ζ = 物理耗散基底 + 学习残差。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import ModalFRFDecoder


# ======================= Dense Padding 工具 =======================
def pad_batch(x: torch.Tensor, node_counts: list):
    """将 (total_N, H) 填充为 (B, N_max, H)，解锁 Tensor Core bmm。"""
    B = len(node_counts)
    max_len = max(node_counts)
    out = x.new_zeros(B, max_len, *x.shape[1:])
    mask = x.new_zeros(B, max_len, dtype=torch.bool)
    ptr = 0
    for i, c in enumerate(node_counts):
        out[i, :c] = x[ptr:ptr + c]
        mask[i, :c] = True
        ptr += c
    return out, mask


def unpad_batch(x_dense: torch.Tensor, node_counts: list) -> torch.Tensor:
    """将 (B, N_max, H) 压缩回 (total_N, H)。"""
    out_list = [x_dense[i, :c] for i, c in enumerate(node_counts)]
    return torch.cat(out_list, dim=0)


class GraphEdgeConv(nn.Module):
    """可选的局部网格感知 stem，利用保存的单元连接边。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x
        src, dst = edge_index[0].long(), edge_index[1].long()
        msg = self.msg(torch.cat([x[src], x[dst] - x[src]], dim=-1)).to(x.dtype)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.norm(x + agg)


class SliceTransolverBlock(nn.Module):
    """Dense 张量 Transolver 注意力层（0 切片循环，100% bmm 向量化）。"""

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, dropout: float = 0.0):
        super().__init__()
        self.assign = nn.Linear(hidden_dim, num_slices)
        self.token_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.node_norm1 = nn.LayerNorm(hidden_dim)
        self.node_norm2 = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N_max, H), mask: (B, N_max)
        assign_logits = self.assign(x).masked_fill(~mask.unsqueeze(-1), -1e4)
        assign = torch.softmax(assign_logits, dim=1).masked_fill(~mask.unsqueeze(-1), 0.0)  # (B, N, S)

        # bmm 并行投影到 token 空间
        denom = assign.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B, 1, S)
        tokens = torch.bmm(assign.transpose(1, 2), x) / denom.transpose(1, 2)  # (B, S, H)

        # Token 注意力
        tokens_norm = self.token_norm(tokens)
        tokens_attn, _ = self.token_attn(tokens_norm, tokens_norm, tokens_norm,
                                          need_weights=False)  # (B, S, H)

        # bmm 回写节点
        back = torch.bmm(assign, tokens_attn)  # (B, N, H)

        y = self.node_norm1(x + back)
        y = self.node_norm2(y + self.ffn(y))
        return y.masked_fill(~mask.unsqueeze(-1), 0.0)


class TransolverModalFRF(nn.Module):
    """Transolver 编码器 + 模态预测头 + 可微 FRF 解码器（Dense Padding 版）。

    参数
    ----------
    response_direction : str
        响应测量的笛卡尔轴。可选 ``"X"``、``"Y"``、``"Z"``。
        默认 ``"Z"``（薄板面外方向）。
    force_direction : str
        力激励的笛卡尔轴。默认 ``"Z"``。
    """

    def __init__(self,
                 in_dim: int = 28,
                 hidden_dim: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 n_slices: int = 64,
                 n_modes: int = 3,
                 dropout: float = 0.0,
                 use_edge_stem: bool = True,
                 amp_scale: float = 500000.0,
                 response_direction: str = "Z",
                 force_direction: str = "Z",
                 omega_scale: float = 8000.0):
        super().__init__()
        self.n_modes = n_modes
        self.use_edge_stem = use_edge_stem

        _DIR_MAP = {"X": 0, "Y": 1, "Z": 2}
        self.response_dir_index = _DIR_MAP[response_direction.upper()]
        self.force_dir_index = _DIR_MAP[force_direction.upper()]

        self.input_proj = nn.Sequential(
            nn.Linear(3 + in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.edge_stem = GraphEdgeConv(hidden_dim)
        self.blocks = nn.ModuleList([
            SliceTransolverBlock(hidden_dim, n_heads, n_slices, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.omega_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )
        self.zeta_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes * 3),
        )

        self.zeta_context_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.zeta_mode_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 数据驱动阻尼头：纯数据路径，用于训练早期物理路径不稳定时的热启动
        self.zeta_direct_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )

        self.physics = ModalFRFDecoder(amp_scale=amp_scale)
        self.omega_scale = nn.Parameter(torch.tensor(omega_scale, dtype=torch.float32))

    def encode(self, points, node_features, edge_index=None, node_counts=None):
        """编码：拼接输入 → edge stem → dense padding → slice blocks → unpadding。"""
        x = self.input_proj(torch.cat([points, node_features], dim=-1))
        if self.use_edge_stem and edge_index is not None and edge_index.numel() > 0:
            x = self.edge_stem(x, edge_index)

        x_dense, mask = pad_batch(x, node_counts)
        for block in self.blocks:
            x_dense = block(x_dense, mask)

        latent_flat = unpad_batch(x_dense, node_counts)
        return latent_flat, x_dense, mask

    def global_pool(self, x_dense, mask):
        """对 dense 张量做门控全局池化。"""
        gate_logits = self.pool_gate(x_dense).squeeze(-1).masked_fill(~mask, -1e4)
        gate = torch.softmax(gate_logits, dim=1).masked_fill(~mask, 0.0)
        return (gate.unsqueeze(-1) * x_dense).sum(dim=1)  # (B, H)

    def compute_physics_zeta(self, phi_xyz_dense, boundary_c_xyz_dense, omega, mask):
        """根据材料阻尼与三向边界耗散计算基线阻尼比（dense 版）。"""
        diss = (boundary_c_xyz_dense.unsqueeze(2) * phi_xyz_dense.pow(2)).sum(dim=-1)  # (B, N, K)
        diss_per_graph = diss.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)  # (B, K)
        return 0.002 + diss_per_graph / (2.0 * omega.clamp_min(1.0))

    def mode_weighted_pool(self, latent_dense, phi_xyz_dense, boundary_c_xyz_dense, mask):
        """振型加权池化（dense 版，逐模态 masked softmax）。"""
        modal_energy = phi_xyz_dense.pow(2).sum(dim=-1)  # (B, N, K)
        boundary_strength = torch.log1p(boundary_c_xyz_dense.abs().sum(dim=-1))  # (B, N)

        context_modes = []
        for k in range(self.n_modes):
            score_input = torch.cat([
                latent_dense,
                modal_energy[..., k:k + 1],
                boundary_strength.unsqueeze(-1),
            ], dim=-1)
            learned_score = self.zeta_context_gate(score_input).squeeze(-1)  # (B, N)
            score = (learned_score
                     + torch.log(modal_energy[..., k] + 1e-8)
                     + 0.1 * boundary_strength)  # (B, N)
            score = score.masked_fill(~mask, -1e4)
            w = torch.softmax(score, dim=1).masked_fill(~mask, 0.0)  # (B, N)
            context_modes.append((w.unsqueeze(-1) * latent_dense).sum(dim=1))  # (B, H)
        return torch.stack(context_modes, dim=1)  # (B, K, H)

    def forward(self,
                points: torch.Tensor,
                node_features: torch.Tensor,
                batch: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                boundary_c_xyz: torch.Tensor | None = None,
                excitation_index: torch.Tensor | None = None,
                frequencies: torch.Tensor | None = None,
                num_graphs: int | None = None,
                node_counts: list | None = None,
                physics_alpha: float = 1.0):
        if node_counts is None:
            node_counts = batch.bincount().tolist()

        # --- 编码 ---
        latent, latent_dense, mask = self.encode(
            points, node_features, edge_index=edge_index, node_counts=node_counts)
        global_latent = self.global_pool(latent_dense, mask)

        # --- 固有频率 ---
        # 直接预测，不排序，不累加
        omega = F.softplus(self.omega_head(global_latent)) * F.softplus(self.omega_scale)  # (B, K)

        # --- 模态振型（三向） ---
        phi_xyz = self.phi_head(latent).view(-1, self.n_modes, 3)

        # --- 方向感知投影 ---
        phi_response = phi_xyz[..., self.response_dir_index]  # (total_N, K)
        phi_force = phi_xyz[..., self.force_dir_index]        # (total_N, K)

        # --- 阻尼比（渐进式物理融合） ---
        # 1. 数据驱动路径：纯数据预测，用于训练早期热启动
        zeta_direct = F.softplus(self.zeta_direct_head(global_latent)) + 1e-4  # (B, K)

        # 2. 物理路径：物理耗散基底 + 非线性残差修正
        if boundary_c_xyz is not None:
            phi_xyz_dense, _ = pad_batch(phi_xyz, node_counts)
            boundary_c_xyz_dense, _ = pad_batch(boundary_c_xyz, node_counts)

            zeta_phys = self.compute_physics_zeta(phi_xyz_dense, boundary_c_xyz_dense, omega, mask)
            mode_context = self.mode_weighted_pool(latent_dense, phi_xyz_dense, boundary_c_xyz_dense, mask)
            zeta_residual = self.zeta_mode_residual_head(mode_context).squeeze(-1)  # (B, K)
            # 0.5 允许网络进行 ±60% 的修正（exp(0.5)≈1.65, exp(-0.5)≈0.61）
            zeta_phys_corrected = zeta_phys * torch.exp(0.5 * torch.tanh(zeta_residual))
        else:
            # 无边界阻尼信息时，物理路径退化为数据路径
            zeta_phys_corrected = zeta_direct

        # 3. 渐进式融合：alpha=0 纯数据驱动，alpha=1 纯物理路径
        zeta = (1.0 - physics_alpha) * zeta_direct + physics_alpha * zeta_phys_corrected

        # --- FRF 物理重建 ---
        frf = None
        phi_force_exc = None
        if excitation_index is not None:
            phi_force_exc = phi_force[excitation_index.long()]  # (B, K)
            if frequencies is not None:
                frf = self.physics(phi_response, phi_force_exc,
                                   omega, zeta, frequencies, batch)

        return {
            'frf': frf,
            'modal_omega': omega,
            'modal_zeta': zeta,
            'modal_phi_xyz': phi_xyz,
            'modal_phi_response': phi_response,
            'modal_phi_force': phi_force,
            'modal_phi_exc_force': phi_force_exc,
            'response_dir_index': self.response_dir_index,
            'force_dir_index': self.force_dir_index,
            'modal_phi_z': phi_xyz[..., 2],
            'latent': latent,
        }
