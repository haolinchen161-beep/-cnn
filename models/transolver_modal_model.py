"""物理引导 Transolver 风格模型，用于模态 FRF 预测。

输入:
    非结构化 ANSYS 节点 + Transolver 节点特征 + 可选网格边。
输出:
    模态固有频率、模态阻尼比、逐节点 XYZ 三向振型，以及方向性 FRF。

这是 Transolver 的轻量级、零外部依赖实现。使用物理感知的 learned slices：
节点被软分配到固定数量的状态 token，在 token 空间做注意力，再将信息
散射回节点。全链路使用纯 PyTorch 操作（无 torch_scatter 依赖），
通过指针切片 + cat/stack 实现极速计算。

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
        msg = self.msg(torch.cat([x[src], x[dst] - x[src]], dim=-1)).to(x.dtype)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.norm(x + agg)


class SliceTransolverBlock(nn.Module):
    """Transolver 风格 slice attention 块，纯 PyTorch 指针切片实现。"""

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

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int, node_counts: list) -> torch.Tensor:
        assign_logits = self.assign(x)  # (total_N, S)

        # 逐图 softmax（替代 scatter_softmax）
        assign_parts = []
        ptr = 0
        for c in node_counts:
            assign_parts.append(torch.softmax(assign_logits[ptr:ptr + c], dim=0))
            ptr += c
        assign = torch.cat(assign_parts, dim=0)  # (total_N, S)

        # 投影到 token 空间：逐图矩阵乘法
        tokens_parts, denom_parts = [], []
        ptr = 0
        for c in node_counts:
            xg = x[ptr:ptr + c]              # (c, H)
            ag = assign[ptr:ptr + c]         # (c, S)
            tokens_parts.append(ag.t() @ xg) # (S, c) @ (c, H) → (S, H)
            denom_parts.append(ag.sum(dim=0).clamp_min(1e-6))
            ptr += c
        tokens = torch.stack(tokens_parts, dim=0)                # (B, S, H)
        denom = torch.stack(denom_parts, dim=0).unsqueeze(-1)    # (B, S, 1)
        tokens = tokens / denom

        # Token 注意力
        tokens_norm = self.token_norm(tokens)
        tokens_attn, _ = self.token_attn(tokens_norm, tokens_norm, tokens_norm,
                                          need_weights=False)   # (B, S, H)

        # 回写节点：用 cat 替代原地赋值，避免 CopySlicesBackward
        back_parts = []
        ptr = 0
        for g, c in enumerate(node_counts):
            back_parts.append(assign[ptr:ptr + c] @ tokens_attn[g])  # (c, S) @ (S, H) → (c, H)
            ptr += c
        back = torch.cat(back_parts, dim=0)  # (total_N, H)

        y = self.node_norm1(x + back)
        y = self.node_norm2(y + self.ffn(y))
        return y


class TransolverModalFRF(nn.Module):
    """Transolver 编码器 + 模态预测头 + 可微 FRF 解码器。

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
        self.zeta_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes * 3),
        )

        # ======================= 振型加权 zeta 残差（局部边界感知） =======================
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

        # ======================= 物理解码器 =======================
        self.physics = ModalFRFDecoder(amp_scale=amp_scale)
        self.omega_scale = nn.Parameter(torch.tensor(omega_scale, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode(self, points, node_features, batch, edge_index=None, num_graphs=None, node_counts=None):
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1
        if node_counts is None:
            node_counts = batch.bincount().tolist()
        x = self.input_proj(torch.cat([points, node_features], dim=-1))
        if self.use_edge_stem and edge_index is not None:
            x = self.edge_stem(x, edge_index)
        for block in self.blocks:
            x = block(x, batch, num_graphs, node_counts)
        return x

    # ------------------------------------------------------------------
    # 全局池化（纯 PyTorch 指针切片）
    # ------------------------------------------------------------------
    def global_pool(self, x, batch, num_graphs, node_counts):
        gate_logits = self.pool_gate(x).squeeze(-1)  # (total_N,)
        out_parts = []
        ptr = 0
        for c in node_counts:
            g_c = torch.softmax(gate_logits[ptr:ptr + c], dim=0)
            out_parts.append((g_c.unsqueeze(-1) * x[ptr:ptr + c]).sum(dim=0))
            ptr += c
        return torch.stack(out_parts, dim=0)  # (B, H)

    # ------------------------------------------------------------------
    # 物理阻尼基底：材料阻尼 + 边界耗散（纯 PyTorch 指针切片）
    # ------------------------------------------------------------------
    def compute_physics_zeta(self, modal_phi_xyz, boundary_c_xyz, omega, batch, num_graphs, node_counts):
        diss_per_node = (boundary_c_xyz.unsqueeze(1) * modal_phi_xyz.pow(2)).sum(dim=-1)  # (total_N, K)
        diss_parts = []
        ptr = 0
        for c in node_counts:
            diss_parts.append(diss_per_node[ptr:ptr + c].sum(dim=0))
            ptr += c
        diss_per_graph = torch.stack(diss_parts, dim=0)  # (B, K)
        zeta = 0.002 + diss_per_graph / (2.0 * omega.clamp_min(1.0))
        return zeta

    # ------------------------------------------------------------------
    # 振型加权池化：局部边界感知的 zeta 残差（纯 PyTorch 指针切片）
    # ------------------------------------------------------------------
    def mode_weighted_pool(self, latent, phi_xyz, boundary_c_xyz, batch, num_graphs, node_counts):
        n_modes = self.n_modes
        modal_energy = phi_xyz.pow(2).sum(dim=-1)                     # (total_N, K)
        boundary_strength = torch.log1p(boundary_c_xyz.abs().sum(dim=-1))  # (total_N,)

        context_modes = []
        for k in range(n_modes):
            score_input = torch.cat([
                latent,
                modal_energy[:, k:k + 1],
                boundary_strength.unsqueeze(-1),
            ], dim=-1)
            learned_score = self.zeta_context_gate(score_input).squeeze(-1)  # (total_N,)
            score = (learned_score
                     + torch.log(modal_energy[:, k] + 1e-8)
                     + 0.1 * boundary_strength)  # (total_N,)

            ctx_parts = []
            ptr = 0
            for c in node_counts:
                w_c = torch.softmax(score[ptr:ptr + c], dim=0)
                ctx_parts.append((w_c.unsqueeze(-1) * latent[ptr:ptr + c]).sum(dim=0))
                ptr += c
            context_modes.append(torch.stack(ctx_parts, dim=0))  # each (B, H)
        return torch.stack(context_modes, dim=1)  # (B, K, H)

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
        node_counts = batch.bincount().tolist()  # 整个前向只做 1 次统计

        # --- 编码 ---
        latent = self.encode(points, node_features, batch,
                             edge_index=edge_index, num_graphs=num_graphs,
                             node_counts=node_counts)
        global_latent = self.global_pool(latent, batch, num_graphs, node_counts)

        # --- 固有频率预测 ---
        omega_raw = F.softplus(self.omega_head(global_latent)) * F.softplus(self.omega_scale)
        omega, _ = torch.sort(omega_raw, dim=-1)  # 保证 ω1 < ω2 < ω3

        # --- 模态振型预测 ---
        phi_xyz = self.phi_head(latent).view(-1, self.n_modes, 3)

        # --- 方向感知投影（替代硬编码 Z 向） ---
        phi_response = phi_xyz[..., self.response_dir_index]  # (total_N, K)
        phi_force = phi_xyz[..., self.force_dir_index]        # (total_N, K)

        # --- 阻尼比预测 ---
        if boundary_c_xyz is not None:
            zeta_phys = self.compute_physics_zeta(phi_xyz, boundary_c_xyz,
                                                  omega, batch, num_graphs, node_counts)
            mode_context = self.mode_weighted_pool(latent, phi_xyz,
                                                   boundary_c_xyz, batch, num_graphs, node_counts)
            zeta_residual = self.zeta_mode_residual_head(mode_context).squeeze(-1)  # (B, K)
            zeta = zeta_phys * torch.exp(0.1 * torch.tanh(zeta_residual))
        else:
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
            'modal_phi_response': phi_response,
            'modal_phi_force': phi_force,
            'modal_phi_exc_force': phi_force_exc,
            'response_dir_index': self.response_dir_index,
            'force_dir_index': self.force_dir_index,
            'modal_phi_z': phi_response if self.response_dir_index == 2 else phi_xyz[..., 2],
            'modal_phi_exc_z': (phi_force_exc if self.force_dir_index == 2
                                else (phi_xyz[..., 2][excitation_index.long()]
                                      if excitation_index is not None else None)),
            'latent': latent,
        }
