"""Transolver-Modal 损失函数。

核心原则与当前 CNN 正确版本一致：
1. 振型符号对齐必须逐图、逐模态计算；
2. MAC/std/φn/φa 必须逐图计算后再 batch 平均；
3. 频率用 Hz-space + 相对误差 + gap loss；
4. 阻尼在 log 空间监督；
5. FRF 只作为 Phase2 弱约束，默认由 trainer 控制权重。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础逐图指标
# ---------------------------------------------------------------------------

def _node_counts(batch_data: Dict[str, torch.Tensor]) -> list[int]:
    if 'node_counts' in batch_data:
        return [int(x) for x in batch_data['node_counts']]
    batch = batch_data['batch']
    return batch.bincount().detach().cpu().tolist()


def _mac_per_graph(phi_pred: torch.Tensor, phi_target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """三维单图 MAC: [N,K,3] -> [K]。"""
    num = torch.sum(phi_pred * phi_target, dim=(0, 2)) ** 2
    den = torch.sum(phi_pred ** 2, dim=(0, 2)) * torch.sum(phi_target ** 2, dim=(0, 2)) + eps
    return num / den


def _align_target_per_graph(phi_pred: torch.Tensor,
                            phi_target: torch.Tensor,
                            node_counts: list[int]) -> torch.Tensor:
    """每个样本内部独立做符号对齐，不能把 batch 拼成一个图。"""
    aligned = torch.empty_like(phi_target)
    ptr = 0
    for c in node_counts:
        c = int(c)
        p = phi_pred[ptr:ptr + c]
        t = phi_target[ptr:ptr + c]
        dot = torch.sum(p * t, dim=(0, 2), keepdim=True)
        aligned[ptr:ptr + c] = t * torch.sign(dot + 1e-8)
        ptr += c
    return aligned


def _phi_metrics(phi_pred: torch.Tensor,
                 phi_target_aligned: torch.Tensor,
                 node_counts: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回逐模态 MAC、φn、φa，均为 [K]。"""
    mac_list, phi_n_list, phi_a_list = [], [], []
    ptr = 0
    for c in node_counts:
        c = int(c)
        p = phi_pred[ptr:ptr + c]
        t = phi_target_aligned[ptr:ptr + c]

        mac_list.append(_mac_per_graph(p, t))

        rmse = torch.sqrt(torch.mean((p - t) ** 2, dim=(0, 2)))
        t_std = torch.std(t.transpose(0, 1).reshape(t.shape[1], -1), dim=1) + 1e-8
        phi_n_list.append(rmse / t_std * 100.0)

        norm_p = torch.sqrt(torch.sum(p ** 2, dim=(0, 2)))
        norm_t = torch.sqrt(torch.sum(t ** 2, dim=(0, 2))) + 1e-8
        phi_a_list.append(torch.abs(norm_p - norm_t) / norm_t * 100.0)
        ptr += c

    return (
        torch.stack(mac_list, dim=0).mean(dim=0),
        torch.stack(phi_n_list, dim=0).mean(dim=0),
        torch.stack(phi_a_list, dim=0).mean(dim=0),
    )


def per_graph_direction_norm_loss(phi_pred: torch.Tensor,
                                  phi_target: torch.Tensor,
                                  node_counts: list[int],
                                  mode_weights: torch.Tensor | None = None) -> torch.Tensor:
    """逐方向范数对数误差，约束 XYZ 分量幅值比例。"""
    losses = []
    ptr = 0
    for c in node_counts:
        c = int(c)
        p = phi_pred[ptr:ptr + c]
        t = phi_target[ptr:ptr + c]
        p_norm = torch.sqrt(torch.sum(p ** 2, dim=0) + 1e-8)  # [K,3]
        t_norm = torch.sqrt(torch.sum(t ** 2, dim=0) + 1e-8)
        losses.append(torch.abs(torch.log((p_norm + 1e-8) / (t_norm + 1e-8))))
        ptr += c
    loss = torch.stack(losses, dim=0)
    if mode_weights is None:
        mode_weights = loss.new_tensor([0.5, 2.0, 3.0])
    return torch.mean(loss * mode_weights.view(1, -1, 1))


# ---------------------------------------------------------------------------
# 模态损失
# ---------------------------------------------------------------------------

def modal_loss(outputs: Dict[str, torch.Tensor],
               batch_data: Dict[str, torch.Tensor],
               weights: Dict[str, float] | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weights = weights or {}
    node_counts = _node_counts(batch_data)

    omega_pred = outputs['modal_omega']
    zeta_pred = outputs['modal_zeta']
    log_zeta_pred = outputs.get('log_zeta', torch.log(zeta_pred.clamp_min(1e-8)))
    phi_pred = outputs['modal_phi_xyz']

    omega_target = batch_data['modal_omega'].to(omega_pred.device)
    zeta_target = batch_data['modal_zeta'].to(zeta_pred.device)
    phi_target = batch_data['modal_phi_xyz'].to(phi_pred.device)

    omega_weight = weights.get('omega', weights.get('omega_loss_weight', 1.0))
    zeta_weight = weights.get('zeta', weights.get('zeta_loss_weight', 10.0))
    phi_weight = weights.get('phi', weights.get('phi_loss_weight', 3.0))

    # ------------------------------------------------------------
    # 1. 频率损失：Hz absolute + relative + gap + peak-sensitive
    # ------------------------------------------------------------
    f_pred_hz = omega_pred / (2.0 * torch.pi)
    f_true_hz = omega_target / (2.0 * torch.pi)
    mode_w = f_pred_hz.new_tensor([1.0, 1.3, 1.6]).view(1, 3)

    loss_freq_abs = torch.mean(F.smooth_l1_loss(f_pred_hz, f_true_hz, reduction='none') * mode_w)
    rel_err = (f_pred_hz - f_true_hz) / (f_true_hz + 1e-8)
    loss_freq_rel = torch.mean(
        F.smooth_l1_loss(rel_err * 100.0, torch.zeros_like(rel_err), reduction='none') * mode_w
    )
    gap_pred = f_pred_hz[:, 1:] - f_pred_hz[:, :-1]
    gap_true = f_true_hz[:, 1:] - f_true_hz[:, :-1]
    gap_w = f_pred_hz.new_tensor([1.2, 1.8]).view(1, 2)
    loss_gap = torch.mean(F.smooth_l1_loss(gap_pred, gap_true, reduction='none') * gap_w)
    peak_sensitive = torch.clamp(torch.abs(rel_err) / (zeta_target + 1e-8), max=100.0)
    loss_omega_raw = 0.5 * loss_freq_abs + 10.0 * loss_freq_rel + 0.5 * loss_gap + 0.05 * peak_sensitive.mean()
    loss_omega = loss_omega_raw * omega_weight

    # ------------------------------------------------------------
    # 2. 阻尼损失：log 域
    # ------------------------------------------------------------
    log_zeta_target = torch.log(zeta_target + 1e-8)
    loss_zeta_raw = F.smooth_l1_loss(log_zeta_pred, log_zeta_target)
    loss_zeta = loss_zeta_raw * zeta_weight

    # ------------------------------------------------------------
    # 3. 三维振型损失：逐图符号对齐 + std norm + MAC + scale/direction
    # ------------------------------------------------------------
    phi_target_aligned = _align_target_per_graph(phi_pred, phi_target, node_counts)

    p_std_list, t_std_list, dir_weight_list = [], [], []
    ptr = 0
    for c in node_counts:
        c = int(c)
        p = phi_pred[ptr:ptr + c]
        t = phi_target_aligned[ptr:ptr + c]
        p_std = torch.std(p.transpose(0, 1).reshape(p.shape[1], -1), dim=1) + 1e-8
        t_std = torch.std(t.transpose(0, 1).reshape(t.shape[1], -1), dim=1) + 1e-8
        p_std_list.append(p_std)
        t_std_list.append(t_std)

        energy = torch.sum(t ** 2, dim=0)  # [K,3]
        dir_weight = (energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)) * 3.0
        dir_weight_list.append(dir_weight.unsqueeze(0).expand(c, -1, -1))
        ptr += c

    p_std = torch.stack(p_std_list, dim=0)  # [B,K]
    t_std = torch.stack(t_std_list, dim=0)
    batch_idx = batch_data['batch'].to(phi_pred.device).long()
    p_norm = phi_pred / p_std[batch_idx].unsqueeze(-1)
    t_norm = phi_target_aligned / t_std[batch_idx].unsqueeze(-1)
    dir_weight = torch.cat(dir_weight_list, dim=0)

    raw_phi_mse = torch.mean(F.mse_loss(p_norm, t_norm, reduction='none') * dir_weight)

    mac_loss_total = phi_pred.new_tensor(0.0)
    ptr = 0
    for c in node_counts:
        c = int(c)
        mac = _mac_per_graph(phi_pred[ptr:ptr + c], phi_target_aligned[ptr:ptr + c])
        mac_loss_total = mac_loss_total + (1.0 - mac).mean()
        ptr += c
    loss_mac = mac_loss_total / max(len(node_counts), 1)

    loss_std = F.smooth_l1_loss(p_std, t_std)
    loss_dir_norm = per_graph_direction_norm_loss(phi_pred, phi_target_aligned, node_counts)
    loss_phi_raw = 10.0 * raw_phi_mse + 40.0 * loss_mac + 20.0 * loss_std + 10.0 * loss_dir_norm
    loss_phi = loss_phi_raw * phi_weight

    total = loss_omega + loss_zeta + loss_phi

    # ------------------------------------------------------------
    # 4. 日志指标
    # ------------------------------------------------------------
    omega_rel_per_mode = torch.mean(torch.abs(omega_pred - omega_target) / (omega_target + 1e-8), dim=0)
    zeta_rel_per_mode = torch.mean(torch.abs(zeta_pred - zeta_target) / (zeta_target + 1e-8), dim=0)
    mac_per_mode, phi_n_per_mode, phi_a_per_mode = _phi_metrics(phi_pred, phi_target_aligned, node_counts)

    logs = {
        'loss_total': total.detach(),
        'loss_modal': total.detach(),
        'loss_omega': loss_omega.detach(),
        'loss_zeta': loss_zeta.detach(),
        'loss_phi': loss_phi.detach(),
        'loss_phi_mse': raw_phi_mse.detach(),
        'loss_mac': loss_mac.detach(),
        'loss_phi_std': loss_std.detach(),
        'loss_phi_dir': loss_dir_norm.detach(),
        **{f'omega_k{k}': omega_rel_per_mode[k].detach() for k in range(omega_rel_per_mode.shape[0])},
        **{f'zeta_k{k}': zeta_rel_per_mode[k].detach() for k in range(zeta_rel_per_mode.shape[0])},
        **{f'mac_k{k}': mac_per_mode[k].detach() for k in range(mac_per_mode.shape[0])},
        **{f'phi_n_k{k}': phi_n_per_mode[k].detach() for k in range(phi_n_per_mode.shape[0])},
        **{f'phi_a_k{k}': phi_a_per_mode[k].detach() for k in range(phi_a_per_mode.shape[0])},
    }
    return total, logs


# ---------------------------------------------------------------------------
# FRF 损失
# ---------------------------------------------------------------------------

def frf_loss(frf_pred: torch.Tensor, frf_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    amp_pred = torch.linalg.norm(frf_pred, dim=-1) + 1e-12
    amp_target = torch.linalg.norm(frf_target, dim=-1) + 1e-12
    loss_db = F.mse_loss(20.0 * torch.log10(amp_pred), 20.0 * torch.log10(amp_target))

    amp_pred_norm = amp_pred / amp_pred.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    amp_target_norm = amp_target / amp_target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cdf_pred = torch.cumsum(amp_pred_norm, dim=-1)
    cdf_target = torch.cumsum(amp_target_norm, dim=-1)
    loss_cdf = F.l1_loss(cdf_pred, cdf_target)
    total = loss_db + 10.0 * loss_cdf
    return total, {
        'loss_frf': total.detach(),
        'loss_frf_db': loss_db.detach(),
        'loss_frf_cdf': loss_cdf.detach(),
    }


def total_loss(outputs: Dict[str, torch.Tensor],
               batch_data: Dict[str, torch.Tensor],
               config: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """兼容旧 trainer 的总损失入口。"""
    weights = config.get('modal_loss_weights', {
        'omega': config.get('omega_loss_weight', 1.0),
        'zeta': config.get('zeta_loss_weight', 10.0),
        'phi': config.get('phi_loss_weight', 3.0),
    })
    total, logs = modal_loss(outputs, batch_data, weights)
    if outputs.get('frf') is not None and config.get('use_frf_loss', config.get('enable_phase2', False)):
        lf, flogs = frf_loss(outputs['frf'], batch_data['point_frf'].to(outputs['frf'].device))
        total = total + config.get('frf_loss_weight', 0.02) * lf
        logs.update(flogs)
        logs['loss_total'] = total.detach()
    return total, logs
