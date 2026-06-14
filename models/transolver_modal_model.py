"""物理引导 Transolver 风格模型，用于模态 FRF 预测（Dense Padding 极速版）。

输入:
    非结构化 ANSYS 节点 + Transolver 节点特征 + 可选网格边。
输出:
    模态固有频率、模态阻尼比、逐节点 XYZ 三向振型，以及方向性 FRF。

本版本对输入做任务隔离：
    - 固有频率 omega 和本征振型 phi 只看结构本征输入；
    - 阻尼 C、激励点距离、激励点标记、刀具距离不进入 omega/phi 编码器；
    - omega 采用物理先验直连 + Transolver 残差的 monotonic gap head；
    - 阻尼 zeta 仍通过 boundary_c_xyz + predicted phi + omega 的物理路径计算。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import ModalFRFDecoder


# ---------------------------------------------------------------------------
# SIREN: 正弦表示网络，用于拟合高频空间畸变
# ---------------------------------------------------------------------------
class Sine(nn.Module):
    """SIREN 的正弦激活函数。"""

    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """带有严格频率初始化的 SIREN 线性层。"""

    def __init__(self, dim_in: int, dim_out: int, w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.w0 = w0
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 修复高维隐空间输入的线性崩溃，使用 Kaiming Uniform 标准
                b = math.sqrt(3.0 / self.linear.in_features)
            else:
                b = math.sqrt(6.0 / self.linear.in_features) / self.w0
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


def sanitize_feature_tensor(x: torch.Tensor, clamp_value: float = 20.0) -> torch.Tensor:
    """Remove NaN/Inf and clamp feature scale before any network layer."""
    y = x.float()
    y = torch.nan_to_num(y, nan=0.0, posinf=float(clamp_value), neginf=-float(clamp_value))
    return y.clamp(-float(clamp_value), float(clamp_value))


def make_modal_structural_features(node_features: torch.Tensor) -> torch.Tensor:
    """给 omega/phi 使用的结构本征特征。

    原始 transolver_point_features 默认字段：
        0..17  : 坐标、材料、凹槽、剩余厚度、装夹 K 等结构信息
        18..20 : log10_Cx/Cy/Cz，阻尼信息，只给 zeta 用
        21..22 : distance_to_excitation / excitation_flag，FRF 查询信息，不应进入本征模态
        23..27 : 表面标记，属于结构几何，可保留
        28/29  : dataset 追加字段。常见情况：in_dim=29 时 28 是 dist_to_tool；
                 in_dim>=30 时 28 可能是 node_active_flag，29 是 dist_to_tool。

    因此：omega/phi 去掉 C、激励点、刀具距离；保留几何、材料、K、表面/拓扑。
    """
    feat = sanitize_feature_tensor(node_features, 20.0).clone()
    fdim = feat.shape[-1]

    # 阻尼 C 不影响无阻尼固有频率/本征振型，不进入 omega/phi。
    for idx in (18, 19, 20):
        if fdim > idx:
            feat[..., idx] = 0.0

    # 激励点是 FRF 查询条件，不是结构本征属性，不进入 omega/phi。
    for idx in (21, 22):
        if fdim > idx:
            feat[..., idx] = 0.0

    # dataset 追加字段：in_dim=29 时通常 28 是 dist_to_tool；in_dim>=30 时通常 29 是 dist_to_tool。
    # 如果后续存在 node_active_flag，它仍可作为拓扑/过程几何信息保留在 28 位。
    if fdim == 29:
        feat[..., 28] = 0.0
    elif fdim >= 30:
        feat[..., 29] = 0.0

    return feat


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
        assign_logits = self.assign(x).masked_fill(~mask.unsqueeze(-1), -1e4)
        assign = torch.softmax(assign_logits, dim=1).masked_fill(~mask.unsqueeze(-1), 0.0)

        denom = assign.sum(dim=1, keepdim=True).clamp_min(1e-6)
        tokens = torch.bmm(assign.transpose(1, 2), x) / denom.transpose(1, 2)

        tokens_norm = self.token_norm(tokens)
        tokens_attn, _ = self.token_attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)

        back = torch.bmm(assign, tokens_attn)
        y = self.node_norm1(x + back)
        y = self.node_norm2(y + self.ffn(y))
        return y.masked_fill(~mask.unsqueeze(-1), 0.0)


class TransolverModalFRF(nn.Module):
    """Transolver 编码器 + 模态预测头 + 可微 FRF 解码器（Dense Padding 版）。"""

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
            raise ValueError("当前 omega gap head 只支持 n_modes=3")
        self.n_modes = n_modes
        self.use_edge_stem = use_edge_stem
        self.node_feat_dim = in_dim
        self.branch_stats_dim = in_dim * 4
        self.omega_prior_dim = 24

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

        # Transolver 残差频率头：只学物理先验解释不了的部分。
        self.omega_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )

        # 物理先验直连频率头：类似 geometric_frf2 的 skip_omega，但针对凹槽板扩展为多维描述符。
        self.omega_prior_norm = nn.LayerNorm(self.omega_prior_dim)
        self.omega_prior_head = nn.Sequential(
            nn.Linear(self.omega_prior_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )

        # 频率 gap head 标定常量，单位 rad/s
        self.register_buffer("omega_w1_base", torch.tensor(4712.389, dtype=torch.float32))
        self.register_buffer("omega_w1_scale", torch.tensor(1817.471, dtype=torch.float32))
        self.register_buffer("omega_gap21_min", torch.tensor(4398.230, dtype=torch.float32))
        self.register_buffer("omega_gap21_scale", torch.tensor(6290.185, dtype=torch.float32))
        self.register_buffer("omega_gap32_min", torch.tensor(1256.637, dtype=torch.float32))
        self.register_buffer("omega_gap32_scale", torch.tensor(2923.069, dtype=torch.float32))

        # 初始时和旧模型接近：out≈0，频率在数据中位附近；训练中 prior/residual 再共同修正。
        nn.init.normal_(self.omega_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.omega_head[-1].bias)
        nn.init.zeros_(self.omega_prior_head[-1].weight)
        nn.init.zeros_(self.omega_prior_head[-1].bias)

        # 结构先验振型 + SIREN 畸变修正
        self.prior_phi_head = nn.Linear(hidden_dim, n_modes * 3)
        siren_w0 = 10.0
        self.fusion_mlp = nn.Sequential(
            SirenLayer(hidden_dim * 2, hidden_dim * 2, w0=siren_w0, is_first=True),
            SirenLayer(hidden_dim * 2, hidden_dim, w0=siren_w0, is_first=False),
            SirenLayer(hidden_dim, hidden_dim, w0=siren_w0, is_first=False),
            nn.Linear(hidden_dim, n_modes * 3),
        )
        with torch.no_grad():
            b = math.sqrt(6.0 / hidden_dim) / siren_w0
            self.fusion_mlp[-1].weight.uniform_(-b, b)
            self.fusion_mlp[-1].bias.uniform_(-b, b)

        self.zeta_context_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.zeta_mode_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
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
        gate_logits = self.pool_gate(x_dense).squeeze(-1).masked_fill(~mask, -1e4)
        gate_logits = torch.nan_to_num(gate_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(x_dense.dtype)
        gate = torch.softmax(gate_logits, dim=1).masked_fill(~mask, 0.0)
        pooled = (gate.unsqueeze(-1) * x_dense).sum(dim=1)
        return torch.nan_to_num(pooled.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(x_dense.dtype)

    def omega_physics_descriptor(self, node_features_modal: torch.Tensor, node_counts: list) -> torch.Tensor:
        """构造 omega 专用物理先验描述符。

        只使用结构本征字段，不使用阻尼 C、激励点或刀具距离。
        输出作为 omega_prior_head 的直接输入，作用相当于 geometric_frf2 中的
        (H/L^2)*sqrt(E/rho) skip prior，但对凹槽板扩展为多维结构描述符。
        """
        feat = sanitize_feature_tensor(node_features_modal, 20.0)
        feat_dense, mask = pad_batch(feat, node_counts)
        B, N, Fdim = feat_dense.shape
        dtype = feat_dense.dtype
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
        free_surface = col(23) > 0.5
        top_surface = col(24) > 0.5
        bottom_surface = col(25) > 0.5
        external_side = col(26) > 0.5
        pocket_sidewall = col(27) > 0.5

        k_any = (logkx > 0.0) | (logky > 0.0) | (logkz > 0.0) | fixture_corner | fixture_side
        logk_mean_node = torch.stack([logkx, logky, logkz], dim=-1).mean(dim=-1)
        logk_max_node = torch.stack([logkx, logky, logkz], dim=-1).max(dim=-1).values

        e_mean = masked_mean(e_ratio)
        rho_mean = masked_mean(rho_ratio).clamp_min(1e-4)
        sqrt_e_rho = torch.sqrt((e_mean / rho_mean).abs().clamp_min(1e-8))
        rem_mean = masked_mean(remaining)
        rem_min = masked_min(remaining)
        rem_std = masked_std(remaining)
        pocket_ratio = ratio(pocket_active)
        pocket_depth_mean = masked_mean(pocket_depth, pocket_active)
        pocket_depth_max = masked_max(pocket_depth, pocket_active)

        # 两个主频率尺度：薄板趋势用有效厚度，凹槽削弱用最小剩余厚度补充。
        freq_prior_mean = rem_mean * sqrt_e_rho
        freq_prior_min = rem_min * sqrt_e_rho

        vals = torch.stack([
            e_mean,
            rho_mean,
            masked_mean(prxy),
            sqrt_e_rho,
            rem_mean,
            rem_min,
            rem_std,
            freq_prior_mean,
            freq_prior_min,
            pocket_ratio,
            ratio(pocket_bottom),
            ratio(cutting_band),
            pocket_depth_mean,
            pocket_depth_max,
            masked_mean(dist_edge, pocket_active),
            masked_min(dist_edge, pocket_active),
            ratio(fixture_corner),
            ratio(fixture_side),
            ratio(k_any),
            masked_mean(logk_mean_node, k_any),
            masked_max(logk_max_node, k_any),
            masked_std(logk_mean_node, k_any),
            ratio(pocket_sidewall) + ratio(external_side),
            masked_mean(z_norm) + ratio(top_surface) - ratio(bottom_surface) + ratio(free_surface),
        ], dim=-1)

        vals = torch.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
        return vals

    def global_feature_summary(self, node_features: torch.Tensor, node_counts: list) -> torch.Tensor:
        """结构化全局物理描述符，仅作诊断/后续扩展，不直接进入 omega/phi。"""
        feat = sanitize_feature_tensor(node_features, 20.0)
        feat_dense, mask = pad_batch(feat, node_counts)
        B, N, Fdim = feat_dense.shape
        dtype = feat_dense.dtype
        valid = mask
        denom = valid.to(dtype).sum(dim=1).clamp_min(1.0)

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
        freq_prior = rem_mean * sqrt_e_rho

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
            e_mean, rho_mean, masked_mean(prxy), sqrt_e_rho, freq_prior,
            masked_mean(z_norm), rem_mean, masked_std(remaining), masked_min(remaining),
            masked_mean(pocket_depth, pocket_active), masked_max(pocket_depth, pocket_active),
            ratio(pocket_active), ratio(pocket_bottom), ratio(cutting_band),
            masked_mean(dist_edge, pocket_active), masked_min(dist_edge, pocket_active),
            ratio(fixture_corner), ratio(fixture_side), ratio(k_any),
            masked_mean(logkx, k_any), masked_mean(logky, k_any), masked_mean(logkz, k_any),
            masked_max(logk_max_node, k_any), masked_std(logk_mean_node, k_any),
            masked_mean(logcx, c_any), masked_mean(logcy, c_any), masked_mean(logcz, c_any),
            masked_max(logc_max_node, c_any), masked_std(logc_mean_node, c_any),
            masked_mean(dist_exc), masked_min(dist_exc), masked_std(dist_exc), exc_x, exc_y, exc_z,
            ratio(free_surface), ratio(top_surface), ratio(bottom_surface), ratio(external_side), ratio(pocket_sidewall),
            denom / 10000.0, masked_mean(extra28), masked_min(extra28), masked_mean(extra29), masked_min(extra29),
        ]

        vals = torch.stack(physics_values, dim=-1)
        vals = torch.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
        summary = feat_dense.new_zeros(B, self.branch_stats_dim)
        n = min(vals.shape[-1], self.branch_stats_dim)
        summary[:, :n] = vals[:, :n]
        return summary

    def compute_physics_zeta(self, phi_xyz_dense, boundary_c_xyz_dense, omega, mask):
        diss = (boundary_c_xyz_dense.unsqueeze(2) * phi_xyz_dense.pow(2)).sum(dim=-1)
        diss_per_graph = diss.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
        return 0.002 + diss_per_graph / (2.0 * omega.clamp_min(1.0))

    def mode_weighted_pool(self, latent_dense, phi_xyz_dense, boundary_c_xyz_dense, mask):
        modal_energy = phi_xyz_dense.pow(2).sum(dim=-1)
        boundary_strength = torch.log1p(boundary_c_xyz_dense.abs().sum(dim=-1))

        context_modes = []
        for k in range(self.n_modes):
            score_input = torch.cat([
                latent_dense,
                modal_energy[..., k:k + 1],
                boundary_strength.unsqueeze(-1),
            ], dim=-1)
            learned_score = self.zeta_context_gate(score_input).squeeze(-1)
            score = learned_score + torch.log(modal_energy[..., k] + 1e-8) + 0.1 * boundary_strength
            score = score.masked_fill(~mask, -1e4)
            w = torch.softmax(score, dim=1).masked_fill(~mask, 0.0)
            context_modes.append((w.unsqueeze(-1) * latent_dense).sum(dim=1))
        return torch.stack(context_modes, dim=1)

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

        node_features_full = sanitize_feature_tensor(node_features, 20.0).to(points.dtype)
        node_features_modal = make_modal_structural_features(node_features_full).to(points.dtype)

        # --- 编码：omega/phi 只使用结构本征特征，不看 C / excitation / tool distance ---
        latent, latent_dense, mask = self.encode(
            points, node_features_modal, edge_index=edge_index, node_counts=node_counts)
        global_latent = self.global_pool(latent_dense, mask)
        branch_features = self.global_feature_summary(node_features_full, node_counts)

        # --- 固有频率：物理先验 descriptor + Transolver residual → monotonic gap head ---
        omega_desc = self.omega_physics_descriptor(node_features_modal, node_counts).to(global_latent.dtype)
        omega_prior_logits = self.omega_prior_head(self.omega_prior_norm(omega_desc))
        omega_residual_logits = self.omega_head(global_latent)
        out = omega_prior_logits + omega_residual_logits
        out = torch.nan_to_num(out.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(global_latent.dtype)

        w1 = F.softplus(out[:, 0:1]) * self.omega_w1_scale + self.omega_w1_base
        gap21 = F.softplus(out[:, 1:2]) * self.omega_gap21_scale + self.omega_gap21_min
        gap32 = F.softplus(out[:, 2:3]) * self.omega_gap32_scale + self.omega_gap32_min
        w2 = w1 + gap21
        w3 = w2 + gap32
        omega = torch.cat([w1, w2, w3], dim=-1)
        omega = torch.nan_to_num(omega.float(), nan=1.0, posinf=60000.0, neginf=1.0).clamp(1.0, 60000.0).to(global_latent.dtype)

        # --- 模态振型：clean structural latent → prior + SIREN delta ---
        prior_phi_flat = self.prior_phi_head(latent_dense)
        B_N = latent_dense.shape[:2]
        global_expanded = global_latent.unsqueeze(1).expand(-1, B_N[1], -1)
        fused_latent = torch.cat([global_expanded, latent_dense], dim=-1)
        delta_phi_flat = self.fusion_mlp(fused_latent)
        final_phi_flat = prior_phi_flat + delta_phi_flat
        final_phi_unpadded = unpad_batch(final_phi_flat, node_counts)
        phi_xyz = final_phi_unpadded.view(-1, self.n_modes, 3)
        phi_xyz = torch.nan_to_num(phi_xyz.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).to(global_latent.dtype)

        phi_response = phi_xyz[..., self.response_dir_index]
        phi_force = phi_xyz[..., self.force_dir_index]

        # --- 阻尼比：可先在 loss 中关闭。物理路径仍允许使用 boundary_c_xyz。 ---
        zeta_direct = F.softplus(self.zeta_direct_head(global_latent)) + 1e-4
        zeta_direct = torch.nan_to_num(zeta_direct.float(), nan=0.003, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0).to(global_latent.dtype)

        if boundary_c_xyz is not None:
            boundary_c_xyz = torch.nan_to_num(boundary_c_xyz.float(), nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6).to(phi_xyz.dtype)
            phi_xyz_dense, _ = pad_batch(phi_xyz, node_counts)
            boundary_c_xyz_dense, _ = pad_batch(boundary_c_xyz, node_counts)
            zeta_phys = self.compute_physics_zeta(phi_xyz_dense, boundary_c_xyz_dense, omega, mask)
            mode_context = self.mode_weighted_pool(latent_dense, phi_xyz_dense, boundary_c_xyz_dense, mask)
            zeta_residual = self.zeta_mode_residual_head(mode_context).squeeze(-1)
            zeta_phys_corrected = zeta_phys * torch.exp(0.5 * torch.tanh(zeta_residual))
            zeta_phys_corrected = torch.nan_to_num(zeta_phys_corrected.float(), nan=0.003, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0).to(global_latent.dtype)
        else:
            zeta_phys_corrected = zeta_direct

        zeta = (1.0 - physics_alpha) * zeta_direct + physics_alpha * zeta_phys_corrected
        zeta = torch.nan_to_num(zeta.float(), nan=0.003, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0).to(global_latent.dtype)

        frf = None
        phi_force_exc = None
        if excitation_index is not None:
            phi_force_exc = phi_force[excitation_index.long()]
            if frequencies is not None:
                frf = self.physics(phi_response, phi_force_exc, omega, zeta, frequencies, batch)

        return {
            'frf': frf,
            'modal_omega': omega,
            'modal_zeta': zeta,
            'modal_phi_xyz': phi_xyz,
            'modal_phi_response': phi_response,
            'modal_phi_force': phi_force,
            'modal_phi_exc_force': phi_force_exc,
            'modal_phi_coeff': None,
            'branch_global_features': branch_features,
            'omega_physics_descriptor': omega_desc,
            'response_dir_index': self.response_dir_index,
            'force_dir_index': self.force_dir_index,
            'modal_phi_z': phi_xyz[..., 2],
            'latent': latent,
        }
