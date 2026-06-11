"""物理引导 Transolver 风格模型，用于模态 FRF 预测（Dense Padding 极速版）。

输入:
    非结构化 ANSYS 节点 + Transolver 节点特征 + 可选网格边。
输出:
    模态固有频率、模态阻尼比、逐节点 XYZ 三向振型，以及方向性 FRF。

全链路使用 Dense Padding + bmm 实现，0 隐式同步，100% 向量化。

阻尼比 ζ = 物理耗散基底 + 学习残差。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import ModalFRFDecoder


def sanitize_feature_tensor(x: torch.Tensor, clamp_value: float = 20.0) -> torch.Tensor:
    """Remove NaN/Inf and clamp feature scale before any network layer.

    Node features contain log stiffness/damping and derived geometry flags. A single
    NaN/Inf value can poison the Transolver encoder, branch coeff distillation, and
    omega loss at epoch 0. Keep this operation inside the model so every caller
    (stage1 trunk, stage2 encoder, evaluation) uses the same safe path.
    """
    y = x.float()
    y = torch.nan_to_num(y, nan=0.0, posinf=float(clamp_value), neginf=-float(clamp_value))
    return y.clamp(-float(clamp_value), float(clamp_value))


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


class DeepONetPhiHead(nn.Module):
    """低秩 branch-trunk 振型头。

    三阶段训练版本采用更标准的 DeepONet 划分：
    - trunk 只看节点坐标和节点物理/几何特征，学习可冻结的空间基；
    - branch 从整件工件的 global latent + 显式全局物理描述符预测模态系数；
    - 二者点乘得到逐节点 XYZ 三向振型。

    注意：这里 trunk 不再依赖 node_latent。这样阶段1才能在没有 Transolver
    encoder 的情况下，先用 sample_id 的可学习 coeff_table 单独学习 trunk basis。
    """

    def __init__(self,
                 hidden_dim: int,
                 node_feat_dim: int,
                 n_modes: int = 3,
                 rank: int = 64,
                 dropout: float = 0.0,
                 branch_extra_dim: int = 0):
        super().__init__()
        self.n_modes = n_modes
        self.rank = rank
        self.branch_extra_dim = branch_extra_dim
        out_dim = n_modes * 3 * rank
        branch_in_dim = hidden_dim + branch_extra_dim

        self.branch_extra_norm = nn.LayerNorm(branch_extra_dim) if branch_extra_dim > 0 else None
        self.branch = nn.Sequential(
            nn.Linear(branch_in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(3 + node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = 1.0 / math.sqrt(float(rank))

    def branch_coeff(self, global_latent: torch.Tensor,
                     global_features: torch.Tensor | None = None) -> torch.Tensor:
        B = global_latent.shape[0]
        global_latent = torch.nan_to_num(global_latent.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(global_latent.dtype)
        if self.branch_extra_dim > 0:
            if global_features is None:
                global_features = global_latent.new_zeros(B, self.branch_extra_dim)
            gf = sanitize_feature_tensor(global_features, 20.0)
            gf = self.branch_extra_norm(gf).to(global_latent.dtype)
            branch_input = torch.cat([global_latent, gf], dim=-1)
        else:
            branch_input = global_latent
        out = self.branch(branch_input)
        out = torch.nan_to_num(out.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(branch_input.dtype)
        return out.view(B, self.n_modes, 3, self.rank)

    def trunk_basis(self, points: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        N = points.shape[0]
        feat = sanitize_feature_tensor(node_features, 20.0).to(points.dtype)
        trunk_in = torch.cat([points, feat], dim=-1)
        out = self.trunk(trunk_in)
        out = torch.nan_to_num(out.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(trunk_in.dtype)
        return out.view(N, self.n_modes, 3, self.rank)

    def combine(self, coeff: torch.Tensor, basis: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        coeff_n = coeff[batch.long()]  # (N, K, 3, R)
        phi = (coeff_n * basis).sum(dim=-1) * self.norm  # (N, K, 3)
        return torch.nan_to_num(phi.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(basis.dtype)

    def forward(self,
                global_latent: torch.Tensor,
                node_latent: torch.Tensor,
                points: torch.Tensor,
                node_features: torch.Tensor,
                batch: torch.Tensor,
                global_features: torch.Tensor | None = None) -> torch.Tensor:
        # node_latent 保留在接口中是为了兼容 TransolverModalFRF.forward，当前 trunk 不使用它。
        coeff = self.branch_coeff(global_latent, global_features)
        basis = self.trunk_basis(points, node_features)
        return self.combine(coeff, basis, batch)


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
                 phi_rank: int = 64):
        super().__init__()
        if n_modes != 3:
            raise ValueError("当前 CNN-style omega gap head 只支持 n_modes=3")
        self.n_modes = n_modes
        self.use_edge_stem = use_edge_stem
        self.node_feat_dim = in_dim
        # 维度保持为 4*in_dim，不破坏当前训练脚本和旧 stage1 加载；
        # 内容由盲目 mean/std/min/max 改成结构化物理描述符，剩余维度补零。
        self.branch_stats_dim = in_dim * 4

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
        # 频率 gap head 标定常量，单位 rad/s
        # 基于当前数据集，并保留 f3 - f2 >= 200Hz 的过滤条件
        self.register_buffer("omega_w1_base", torch.tensor(4712.389, dtype=torch.float32))      # 2π × 750 Hz
        self.register_buffer("omega_w1_scale", torch.tensor(1817.471, dtype=torch.float32))

        self.register_buffer("omega_gap21_min", torch.tensor(4398.230, dtype=torch.float32))    # 2π × 700 Hz
        self.register_buffer("omega_gap21_scale", torch.tensor(6290.185, dtype=torch.float32))

        self.register_buffer("omega_gap32_min", torch.tensor(1256.637, dtype=torch.float32))    # 2π × 200 Hz
        self.register_buffer("omega_gap32_scale", torch.tensor(2923.069, dtype=torch.float32))

        # 让初始 out≈0，使初始频率接近数据均值，同时保留梯度通路
        nn.init.normal_(self.omega_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.omega_head[-1].bias)

        self.zeta_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )
        self.phi_head = DeepONetPhiHead(
            hidden_dim=hidden_dim,
            node_feat_dim=in_dim,
            n_modes=n_modes,
            rank=phi_rank,
            dropout=dropout,
            branch_extra_dim=self.branch_stats_dim,
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

    def encode(self, points, node_features, edge_index=None, node_counts=None):
        """编码：拼接输入 → edge stem → dense padding → slice blocks → unpadding。"""
        feat = sanitize_feature_tensor(node_features, 20.0).to(points.dtype)
        x = self.input_proj(torch.cat([points, feat], dim=-1))
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(points.dtype)
        if self.use_edge_stem and edge_index is not None and edge_index.numel() > 0:
            x = self.edge_stem(x, edge_index)
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(points.dtype)

        x_dense, mask = pad_batch(x, node_counts)
        for block in self.blocks:
            x_dense = block(x_dense, mask)
            x_dense = torch.nan_to_num(x_dense.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(x.dtype)

        latent_flat = unpad_batch(x_dense, node_counts)
        return latent_flat, x_dense, mask

    def global_pool(self, x_dense, mask):
        """对 dense 张量做门控全局池化。"""
        gate_logits = self.pool_gate(x_dense).squeeze(-1).masked_fill(~mask, -1e4)
        gate_logits = torch.nan_to_num(gate_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(x_dense.dtype)
        gate = torch.softmax(gate_logits, dim=1).masked_fill(~mask, 0.0)
        pooled = (gate.unsqueeze(-1) * x_dense).sum(dim=1)  # (B, H)
        return torch.nan_to_num(pooled.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(x_dense.dtype)

    def global_feature_summary(self, node_features: torch.Tensor, node_counts: list) -> torch.Tensor:
        """构造结构化全局物理描述符。

        数据生成阶段的 transolver_point_features 已经包含材料、凹槽、装夹、K/C、
        激励距离、表面标记等物理字段。这里不再盲目 concat mean/std/min/max，
        而是按字段语义提取 branch 更需要的全局物理量：
        - 材料: E/rho/PRXY 与 sqrt(E/rho)
        - 几何: 剩余厚度、凹槽深度、凹槽底面/切削带/表面比例
        - 边界: 角点/侧顶杆比例、三向 logK/logC 的弹簧节点统计
        - 激励: 激励点坐标、距离激励点统计

        输出维度仍固定为 self.branch_stats_dim = 4*in_dim，前若干维为物理描述符，
        剩余维补零，避免破坏已有 stage1 trunk/coeff_table 的复用流程。
        """
        feat = sanitize_feature_tensor(node_features, 20.0)
        feat_dense, mask = pad_batch(feat, node_counts)
        B, N, Fdim = feat_dense.shape
        dtype = feat_dense.dtype
        device = feat_dense.device
        valid = mask
        valid_f = valid.to(dtype)
        denom = valid_f.sum(dim=1).clamp_min(1.0)

        def col(idx: int, default: float = 0.0) -> torch.Tensor:
            if Fdim > idx:
                return feat_dense[:, :, idx]
            return feat_dense.new_full((B, N), float(default))

        def masked_mean(v: torch.Tensor, sel: torch.Tensor | None = None) -> torch.Tensor:
            m = valid if sel is None else (valid & sel)
            w = m.to(dtype)
            return (v * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

        def masked_std(v: torch.Tensor, sel: torch.Tensor | None = None) -> torch.Tensor:
            m = valid if sel is None else (valid & sel)
            w = m.to(dtype)
            d = w.sum(dim=1).clamp_min(1.0)
            mu = (v * w).sum(dim=1) / d
            var = ((v - mu.unsqueeze(1)).pow(2) * w).sum(dim=1) / d
            return torch.sqrt(var.clamp_min(0.0) + 1e-8)

        def masked_min(v: torch.Tensor, sel: torch.Tensor | None = None) -> torch.Tensor:
            m = valid if sel is None else (valid & sel)
            has = m.any(dim=1)
            out = v.masked_fill(~m, 20.0).min(dim=1).values
            return torch.where(has, out, torch.zeros_like(out))

        def masked_max(v: torch.Tensor, sel: torch.Tensor | None = None) -> torch.Tensor:
            m = valid if sel is None else (valid & sel)
            has = m.any(dim=1)
            out = v.masked_fill(~m, -20.0).max(dim=1).values
            return torch.where(has, out, torch.zeros_like(out))

        def ratio(sel: torch.Tensor) -> torch.Tensor:
            return (sel & valid).to(dtype).sum(dim=1) / denom

        x_norm = col(0)
        y_norm = col(1)
        z_norm = col(2)
        e_ratio = col(3, 1.0)
        rho_ratio = col(4, 1.0)
        prxy = col(5, 0.33)
        pocket_active = col(6) > 0.5
        pocket_bottom = col(7) > 0.5
        cutting_band = col(8) > 0.5
        pocket_depth = col(10)
        remaining = col(11, 1.0)
        dist_edge = col(12, 1.0)
        fixture_corner = col(13) > 0.5
        fixture_side = col(14) > 0.5
        logkx = col(15, -1.0)
        logky = col(16, -1.0)
        logkz = col(17, -1.0)
        logcx = col(18, -1.0)
        logcy = col(19, -1.0)
        logcz = col(20, -1.0)
        dist_exc = col(21, 1.0)
        excitation_flag = col(22) > 0.5
        free_surface = col(23) > 0.5
        top_surface = col(24) > 0.5
        bottom_surface = col(25) > 0.5
        external_side = col(26) > 0.5
        pocket_sidewall = col(27) > 0.5

        k_any = (logkx > 0.0) | (logky > 0.0) | (logkz > 0.0) | fixture_corner | fixture_side
        c_any = (logcx > 0.0) | (logcy > 0.0) | (logcz > 0.0) | fixture_corner | fixture_side
        logk_max_node = torch.stack([logkx, logky, logkz], dim=-1).max(dim=-1).values
        logc_max_node = torch.stack([logcx, logcy, logcz], dim=-1).max(dim=-1).values
        logk_mean_node = torch.stack([logkx, logky, logkz], dim=-1).mean(dim=-1)
        logc_mean_node = torch.stack([logcx, logcy, logcz], dim=-1).mean(dim=-1)

        e_mean = masked_mean(e_ratio)
        rho_mean = masked_mean(rho_ratio).clamp_min(1e-4)
        sqrt_e_rho = torch.sqrt((e_mean / rho_mean).abs().clamp_min(1e-8))
        rem_mean = masked_mean(remaining)
        freq_prior = rem_mean * sqrt_e_rho  # 类似 H_eff * sqrt(E/rho)，交给 head 学尺度

        # 激励点坐标：优先用 excitation_flag；没有 flag 时用 dist_exc 最小点近似。
        exc_has = (excitation_flag & valid).any(dim=1)
        exc_w = (excitation_flag & valid).to(dtype)
        min_dist = dist_exc.masked_fill(~valid, 20.0).min(dim=1, keepdim=True).values
        nearest_exc = (dist_exc <= min_dist + 1e-6) & valid
        near_w = nearest_exc.to(dtype)
        exc_w = torch.where(exc_has.view(B, 1), exc_w, near_w)
        exc_denom = exc_w.sum(dim=1).clamp_min(1.0)
        exc_x = (x_norm * exc_w).sum(dim=1) / exc_denom
        exc_y = (y_norm * exc_w).sum(dim=1) / exc_denom
        exc_z = (z_norm * exc_w).sum(dim=1) / exc_denom

        extra28 = col(28, 0.0)
        extra29 = col(29, 0.0)

        physics_values = [
            # 材料与频率主尺度
            e_mean,
            rho_mean,
            masked_mean(prxy),
            sqrt_e_rho,
            freq_prior,
            # 几何/凹槽
            masked_mean(z_norm),
            rem_mean,
            masked_std(remaining),
            masked_min(remaining),
            masked_mean(pocket_depth, pocket_active),
            masked_max(pocket_depth, pocket_active),
            ratio(pocket_active),
            ratio(pocket_bottom),
            ratio(cutting_band),
            masked_mean(dist_edge, pocket_active),
            masked_min(dist_edge, pocket_active),
            # 装夹/弹簧
            ratio(fixture_corner),
            ratio(fixture_side),
            ratio(k_any),
            masked_mean(logkx, k_any),
            masked_mean(logky, k_any),
            masked_mean(logkz, k_any),
            masked_max(logk_max_node, k_any),
            masked_std(logk_mean_node, k_any),
            masked_mean(logcx, c_any),
            masked_mean(logcy, c_any),
            masked_mean(logcz, c_any),
            masked_max(logc_max_node, c_any),
            masked_std(logc_mean_node, c_any),
            # 激励位置/距离
            masked_mean(dist_exc),
            masked_min(dist_exc),
            masked_std(dist_exc),
            exc_x,
            exc_y,
            exc_z,
            # 表面结构
            ratio(free_surface),
            ratio(top_surface),
            ratio(bottom_surface),
            ratio(external_side),
            ratio(pocket_sidewall),
            # 网格与可选过程字段
            denom / 10000.0,
            masked_mean(extra28),
            masked_min(extra28),
            masked_mean(extra29),
            masked_min(extra29),
        ]

        vals = torch.stack(physics_values, dim=-1)
        vals = torch.nan_to_num(vals, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        summary = feat_dense.new_zeros(B, self.branch_stats_dim)
        n = min(vals.shape[-1], self.branch_stats_dim)
        summary[:, :n] = vals[:, :n]
        return summary

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

        node_features_safe = sanitize_feature_tensor(node_features, 20.0).to(points.dtype)

        # --- 编码 ---
        latent, latent_dense, mask = self.encode(
            points, node_features_safe, edge_index=edge_index, node_counts=node_counts)
        global_latent = self.global_pool(latent_dense, mask)
        branch_features = self.global_feature_summary(node_features_safe, node_counts)

        # --- 固有频率 ---
        # CNN 同款单调 gap head：w1 + gap21 + gap32
        out = self.omega_head(global_latent)
        out = torch.nan_to_num(out.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(global_latent.dtype)

        w1 = F.softplus(out[:, 0:1]) * self.omega_w1_scale + self.omega_w1_base
        gap21 = F.softplus(out[:, 1:2]) * self.omega_gap21_scale + self.omega_gap21_min
        gap32 = F.softplus(out[:, 2:3]) * self.omega_gap32_scale + self.omega_gap32_min

        w2 = w1 + gap21
        w3 = w2 + gap32

        omega = torch.cat([w1, w2, w3], dim=-1)  # (B, 3), rad/s
        omega = torch.nan_to_num(omega.float(), nan=1.0, posinf=60000.0, neginf=1.0).clamp(1.0, 60000.0).to(global_latent.dtype)

        # --- 模态振型（三向）---
        # DeepONet branch-trunk head: global latent + structured physical descriptor 给模态系数
        modal_phi_coeff = self.phi_head.branch_coeff(global_latent, branch_features)
        basis = self.phi_head.trunk_basis(points, node_features_safe)
        phi_xyz = self.phi_head.combine(modal_phi_coeff, basis, batch)

        # --- 方向感知投影 ---
        phi_response = phi_xyz[..., self.response_dir_index]  # (total_N, K)
        phi_force = phi_xyz[..., self.force_dir_index]        # (total_N, K)

        # --- 阻尼比（渐进式物理融合） ---
        # 1. 数据驱动路径：纯数据预测，用于训练早期热启动
        zeta_direct = F.softplus(self.zeta_direct_head(global_latent)) + 1e-4  # (B, K)
        zeta_direct = torch.nan_to_num(zeta_direct.float(), nan=0.003, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0).to(global_latent.dtype)

        # 2. 物理路径：物理耗散基底 + 非线性残差修正
        if boundary_c_xyz is not None:
            boundary_c_xyz = torch.nan_to_num(boundary_c_xyz.float(), nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6).to(phi_xyz.dtype)
            phi_xyz_dense, _ = pad_batch(phi_xyz, node_counts)
            boundary_c_xyz_dense, _ = pad_batch(boundary_c_xyz, node_counts)

            zeta_phys = self.compute_physics_zeta(phi_xyz_dense, boundary_c_xyz_dense, omega, mask)
            mode_context = self.mode_weighted_pool(latent_dense, phi_xyz_dense, boundary_c_xyz_dense, mask)
            zeta_residual = self.zeta_mode_residual_head(mode_context).squeeze(-1)  # (B, K)
            # 0.5 允许网络进行 ±60% 的修正（exp(0.5)≈1.65, exp(-0.5)≈0.61）
            zeta_phys_corrected = zeta_phys * torch.exp(0.5 * torch.tanh(zeta_residual))
            zeta_phys_corrected = torch.nan_to_num(zeta_phys_corrected.float(), nan=0.003, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0).to(global_latent.dtype)
        else:
            # 无边界阻尼信息时，物理路径退化为数据路径
            zeta_phys_corrected = zeta_direct

        # 3. 渐进式融合：alpha=0 纯数据驱动，alpha=1 纯物理路径
        zeta = (1.0 - physics_alpha) * zeta_direct + physics_alpha * zeta_phys_corrected
        zeta = torch.nan_to_num(zeta.float(), nan=0.003, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0).to(global_latent.dtype)

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
            'modal_phi_coeff': modal_phi_coeff,
            'branch_global_features': branch_features,
            'response_dir_index': self.response_dir_index,
            'force_dir_index': self.force_dir_index,
            'modal_phi_z': phi_xyz[..., 2],
            'latent': latent,
        }
