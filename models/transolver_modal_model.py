"""物理引导 Transolver 风格模型，用于模态 FRF 预测。

输入:
    非结构化 ANSYS 节点 + Transolver 节点特征 + 可选网格边。
输出:
    模态固有频率、模态阻尼比、逐节点 XYZ 三向振型，以及方向性 FRF。

这是 Transolver 的轻量级、零外部依赖实现。使用物理感知的 learned slices：
节点被软分配到固定数量的状态 token，在 token 空间做注意力，再将信息
散射回节点。这保持了与 Transolver 思路的接近，同时无需外部 Transolver 包即可运行。

方向约定
--------
模型不再硬编码 Z 方向。改为接受 ``response_direction`` 和 ``force_direction``
（各为 ``"X"``、``"Y"`` 或 ``"Z"``），内部自动选择正确的笛卡尔轴索引。
解码出的 FRF 为 H_ab，其中 ``a`` = 响应方向，``b`` = 激励方向。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import ModalFRFDecoder


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
        msg = self.msg(torch.cat([x[src], x[dst] - x[src]], dim=-1))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.norm(x + agg)


class SliceTransolverBlock(nn.Module):
    """Transolver 风格 slice attention 块，支持变长批次网格。"""

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, dropout: float = 0.0):
        super().__init__()
        self.num_slices = num_slices
        self.assign = nn.Linear(hidden_dim, num_slices)
        self.token_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.node_norm1 = nn.LayerNorm(hidden_dim)
        self.node_norm2 = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        out = torch.empty_like(x)
        for g in range(num_graphs):
            mask = batch == g
            xg = x[mask]
            if xg.numel() == 0:
                continue
            assign = torch.softmax(self.assign(xg), dim=-1)  # (N_g, S)
            denom = assign.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
            tokens = assign.transpose(0, 1) @ xg / denom  # (S, H)
            tokens = self.token_norm(tokens).unsqueeze(0)
            tokens_attn, _ = self.token_attn(tokens, tokens, tokens, need_weights=False)
            tokens_attn = tokens_attn.squeeze(0)
            back = assign @ tokens_attn
            yg = self.node_norm1(xg + back)
            yg = self.node_norm2(yg + self.ffn(yg))
            out[mask] = yg
        return out


class TransolverModalFRF(nn.Module):
    """Transolver 编码器 + 模态预测头 + 可微 FRF 解码器。

    参数
    ----------
    response_direction : str
        响应测量的笛卡尔轴。可选 ``"X"``、``"Y"``、``"Z"``。
        默认 ``"Y"``（铣削切削平面内方向）。
    force_direction : str
        力激励的笛卡尔轴。默认 ``"Y"``。
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
                 response_direction: str = "Y",
                 force_direction: str = "Y"):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_dim = hidden_dim
        self.use_edge_stem = use_edge_stem

        # ======================= 方向配置 =======================
        _DIR_MAP = {"X": 0, "Y": 1, "Z": 2}
        self.response_direction = response_direction.upper()
        self.force_direction = force_direction.upper()
        self.response_dir_index = _DIR_MAP[self.response_direction]
        self.force_dir_index = _DIR_MAP[self.force_direction]

        # ======================= 编码器主干 =======================
        self.input_proj = nn.Sequential(
            nn.Linear(3 + in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.edge_stem = GraphEdgeConv(hidden_dim)
        self.blocks = nn.ModuleList([
            SliceTransolverBlock(hidden_dim, n_heads, n_slices, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ======================= 全局池化与预测头 =======================
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.omega_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )
        # 旧全局 zeta 残差预测头 —— 当 boundary_c_xyz 为空时作为回退
        self.zeta_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes * 3),
        )

        # ======================= 振型加权 zeta 残差（局部边界感知） =======================
        # 门控网络：根据 latent、模态能量和边界强度为每个 (节点, 模态) 对打分
        self.zeta_context_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 残差预测器：对每模态池化后的 context 做预测
        self.zeta_mode_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ======================= 物理解码器 =======================
        self.physics = ModalFRFDecoder(amp_scale=amp_scale)

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode(self, points, node_features, batch, edge_index=None, num_graphs=None):
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1
        x = self.input_proj(torch.cat([points, node_features], dim=-1))
        if self.use_edge_stem and edge_index is not None:
            x = self.edge_stem(x, edge_index)
        for block in self.blocks:
            x = block(x, batch, num_graphs)
        return x

    # ------------------------------------------------------------------
    # 全局池化
    # ------------------------------------------------------------------
    def global_pool(self, x, batch, num_graphs):
        pooled = []
        for g in range(num_graphs):
            mask = batch == g
            xg = x[mask]
            gate = torch.softmax(self.pool_gate(xg).squeeze(-1), dim=0)
            pooled.append((gate.unsqueeze(-1) * xg).sum(dim=0))
        return torch.stack(pooled, dim=0)

    # ------------------------------------------------------------------
    # 物理阻尼基底：材料阻尼 + 边界耗散
    # ------------------------------------------------------------------
    def compute_physics_zeta(self, modal_phi_xyz, boundary_c_xyz, omega, batch, num_graphs):
        """根据材料阻尼与三向边界耗散计算基线阻尼比。

        ζ_k = ζ_material + Σ_i Σ_d C_{i,d} · φ_{i,d,k}² / (2 · ω_k)
        """
        zeta = modal_phi_xyz.new_zeros(num_graphs, self.n_modes)
        for g in range(num_graphs):
            mask = batch == g
            phi = modal_phi_xyz[mask]       # (N_g, K, 3)
            c = boundary_c_xyz[mask]        # (N_g, 3)
            diss = (c.unsqueeze(1) * phi.pow(2)).sum(dim=(0, 2))
            zeta[g] = 0.002 + diss / (2.0 * omega[g].clamp_min(1.0))
        return zeta

    # ------------------------------------------------------------------
    # 振型加权池化：局部边界感知的 zeta 残差
    # ------------------------------------------------------------------
    def mode_weighted_pool(self, latent, phi_xyz, boundary_c_xyz, batch, num_graphs):
        """按图和模态对节点隐层特征做加权池化，权重由模态能量和局部边界强度决定。

        返回:
            context: (B, K, H)，B=图数, K=模态阶数, H=隐层维度。
        """
        n_modes = self.n_modes
        H = latent.shape[-1]
        device = latent.device

        context = latent.new_zeros(num_graphs, n_modes, H)

        for g in range(num_graphs):
            mask = batch == g
            if not torch.any(mask):
                continue
            latent_g = latent[mask]              # (N_g, H)
            phi_g = phi_xyz[mask]                # (N_g, K, 3)
            c_g = boundary_c_xyz[mask]           # (N_g, 3)

            # 模态能量：XYZ 三分量平方和，形状 (N_g, K)
            modal_energy = phi_g.pow(2).sum(dim=-1)  # (N_g, K)

            # 边界强度：每节点阻尼总量的 log1p，形状 (N_g,)
            boundary_strength = torch.log1p(c_g.abs().sum(dim=-1))  # (N_g,)

            # 打分输入：[latent | modal_energy | boundary_strength]，按 (节点, 模态)
            latent_exp = latent_g.unsqueeze(1).expand(-1, n_modes, -1)   # (N_g, K, H)
            energy_exp = modal_energy.unsqueeze(-1)                       # (N_g, K, 1)
            boundary_exp = (boundary_strength.unsqueeze(-1)
                            .unsqueeze(-1).expand(-1, n_modes, 1))       # (N_g, K, 1)

            score_input = torch.cat([latent_exp, energy_exp, boundary_exp], dim=-1)
            # (N_g, K, H+2) → (N_g, K, 1) → (N_g, K)
            learned_score = self.zeta_context_gate(score_input).squeeze(-1)  # (N_g, K)

            # 综合得分 = 学习得分 + 物理先验（模态能量 + 边界强度）
            score = (learned_score
                     + torch.log(modal_energy + 1e-8)
                     + 0.1 * boundary_strength.unsqueeze(-1))  # (N_g, K)

            weight = torch.softmax(score, dim=0)  # 同图内所有节点做 softmax

            # 加权求和：Σ_i w_{i,k} · latent_i  → (K, H)
            context[g] = (weight.unsqueeze(-1) * latent_g.unsqueeze(1)).sum(dim=0)

        return context

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self,
                points: torch.Tensor,
                node_features: torch.Tensor,
                batch: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                boundary_c_xyz: torch.Tensor | None = None,
                excitation_index: torch.Tensor | None = None,
                frequencies: torch.Tensor | None = None,
                num_graphs: int | None = None):
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1

        # --- 编码 ---
        latent = self.encode(points, node_features, batch,
                             edge_index=edge_index, num_graphs=num_graphs)
        global_latent = self.global_pool(latent, batch, num_graphs)

        # --- 固有频率预测 ---
        omega_inc = F.softplus(self.omega_head(global_latent)) + 1e-3
        omega = torch.cumsum(omega_inc, dim=-1)

        # --- 模态振型预测 ---
        phi_xyz = self.phi_head(latent).view(-1, self.n_modes, 3)

        # --- 方向感知投影（替代硬编码 Z 向） ---
        phi_response = phi_xyz[..., self.response_dir_index]  # (total_N, K)
        phi_force = phi_xyz[..., self.force_dir_index]        # (total_N, K)

        # --- 阻尼比预测 ---
        if boundary_c_xyz is not None:
            # 物理基底阻尼
            zeta_phys = self.compute_physics_zeta(phi_xyz, boundary_c_xyz,
                                                  omega, batch, num_graphs)
            # 振型加权残差（局部边界感知）
            mode_context = self.mode_weighted_pool(latent, phi_xyz,
                                                   boundary_c_xyz, batch, num_graphs)
            # (B, K, H) → (B, K, 1) → (B, K)
            zeta_residual = self.zeta_mode_residual_head(mode_context).squeeze(-1)
            zeta = zeta_phys * torch.exp(0.1 * torch.tanh(zeta_residual))
        else:
            # 回退：无边界信息时使用全局 zeta 预测头
            zeta = F.softplus(self.zeta_residual_head(global_latent)) + 1e-4

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
            # 方向感知主输出
            'modal_phi_response': phi_response,
            'modal_phi_force': phi_force,
            'modal_phi_exc_force': phi_force_exc,
            'response_dir_index': self.response_dir_index,
            'force_dir_index': self.force_dir_index,
            # 向后兼容别名（仅当方向为 Z 时与旧代码一致）
            'modal_phi_z': phi_response if self.response_dir_index == 2 else phi_xyz[..., 2],
            'modal_phi_exc_z': (phi_force_exc if self.force_dir_index == 2
                                else (phi_xyz[..., 2][excitation_index.long()]
                                      if excitation_index is not None else None)),
            'latent': latent,
        }
