"""
losses.py — MeshGraphNet/GNN 模态参数损失 + FRF 损失。

设计目标：
1. 支持可变节点数 disjoint graph batch。
2. 固有频率/阻尼使用排序后的相对误差。
3. 振型使用符号不敏感的 MAC loss + 小权重幅值 MSE。
4. FRF 使用复数 Re/Im、log amplitude、CDF envelope 的组合损失。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mode_weights(n_modes: int, device, dtype):
    if n_modes == 3:
        return torch.tensor([1.0, 1.5, 2.0], device=device, dtype=dtype).unsqueeze(0)
    return torch.linspace(1.0, 2.0, n_modes, device=device, dtype=dtype).unsqueeze(0)


def _sort_modal_outputs(omega_pred, zeta_pred, phi_pred, batch_idx=None):
    """按预测频率升序重排 omega/zeta/phi。"""
    omega_sorted, sort_idx = torch.sort(omega_pred, dim=-1)
    zeta_sorted = torch.gather(zeta_pred, dim=-1, index=sort_idx)

    if phi_pred is None:
        return omega_sorted, zeta_sorted, None, sort_idx

    if phi_pred.dim() == 2:
        if batch_idx is None:
            raise ValueError("phi_pred is flattened [total_N,K], batch_idx is required")
        sort_idx_phi = sort_idx[batch_idx]
        phi_sorted = torch.gather(phi_pred, dim=-1, index=sort_idx_phi)
    elif phi_pred.dim() == 3:
        sort_idx_phi = sort_idx.unsqueeze(1).expand(-1, phi_pred.shape[1], -1)
        phi_sorted = torch.gather(phi_pred, dim=-1, index=sort_idx_phi)
    else:
        raise ValueError(f"Unsupported phi_pred shape: {tuple(phi_pred.shape)}")
    return omega_sorted, zeta_sorted, phi_sorted, sort_idx


def _flatten_phi(phi_pred, phi_target):
    if phi_pred.dim() == 3:
        phi_pred = phi_pred.reshape(-1, phi_pred.shape[-1])
    if phi_target.dim() == 3:
        phi_target = phi_target.reshape(-1, phi_target.shape[-1])
    return phi_pred, phi_target


def sign_aligned_mse(phi_pred, phi_target, batch_idx=None, eps=1e-8):
    """符号不敏感 MSE，用于约束振型幅值尺度。"""
    phi_pred, phi_target = _flatten_phi(phi_pred, phi_target)
    if batch_idx is None:
        dot = torch.sum(phi_pred * phi_target, dim=0, keepdim=True)
        sign = torch.sign(dot + eps)
        return F.mse_loss(phi_pred * sign, phi_target)

    losses = []
    n_graphs = int(batch_idx.max().item()) + 1
    for g in range(n_graphs):
        mask = batch_idx == g
        if not torch.any(mask):
            continue
        p = phi_pred[mask]
        t = phi_target[mask]
        dot = torch.sum(p * t, dim=0, keepdim=True)
        sign = torch.sign(dot + eps)
        losses.append(F.mse_loss(p * sign, t))
    return torch.stack(losses).mean() if losses else phi_pred.new_tensor(0.0)


def mac_loss(phi_pred, phi_target, batch_idx=None, eps=1e-8):
    """符号不敏感 MAC loss，适合模态振型监督。"""
    phi_pred, phi_target = _flatten_phi(phi_pred, phi_target)
    mac_terms = []

    if batch_idx is None:
        num = torch.sum(phi_pred * phi_target, dim=0) ** 2
        den = torch.sum(phi_pred ** 2, dim=0) * torch.sum(phi_target ** 2, dim=0) + eps
        return torch.mean(1.0 - num / den.clamp_min(eps))

    n_graphs = int(batch_idx.max().item()) + 1
    for g in range(n_graphs):
        mask = batch_idx == g
        if not torch.any(mask):
            continue
        p = phi_pred[mask]
        t = phi_target[mask]
        num = torch.sum(p * t, dim=0) ** 2
        den = torch.sum(p ** 2, dim=0) * torch.sum(t ** 2, dim=0) + eps
        mac_terms.append(torch.mean(1.0 - num / den.clamp_min(eps)))
    return torch.stack(mac_terms).mean() if mac_terms else phi_pred.new_tensor(0.0)


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               omega_weight=200.0, zeta_weight=10.0, phi_weight=100.0,
               phi_mse_weight=0.05):
    """GNN 模态监督损失。

    Args:
        omega_pred:   [B,K] normalized predicted omega
        omega_target: [B,K] normalized target omega
        zeta_pred:    [B,K]
        zeta_target:  [B,K]
        phi_pred:     [total_N,K] or [B,N,K]
        phi_target:   same flattened layout as phi_pred target
        batch_idx:    [total_N] for variable-size graph batches
    Returns:
        total_loss, weighted_omega_loss, weighted_zeta_loss, weighted_phi_loss
    """
    omega_sorted, zeta_sorted, phi_sorted, _ = _sort_modal_outputs(
        omega_pred, zeta_pred, phi_pred, batch_idx=batch_idx
    )

    n_modes = omega_pred.shape[-1]
    mode_w = _mode_weights(n_modes, omega_pred.device, omega_pred.dtype)
    omega_rel = (omega_sorted - omega_target) / (omega_target.abs() + 1e-8)
    zeta_rel = (zeta_sorted - zeta_target) / (zeta_target.abs() + 1e-8)
    loss_omega = torch.mean(omega_rel ** 2 * mode_w) * omega_weight
    loss_zeta = torch.mean(zeta_rel ** 2 * mode_w) * zeta_weight

    raw_mac = mac_loss(phi_sorted, phi_target, batch_idx=batch_idx)
    raw_mse = sign_aligned_mse(phi_sorted, phi_target, batch_idx=batch_idx)
    loss_phi = (raw_mac + phi_mse_weight * raw_mse) * phi_weight

    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, loss_phi


def frf_loss(frf_pred, frf_target,
             complex_weight=0.1,
             log_amp_weight=1.0,
             cdf_weight=10.0,
             eps=1e-12):
    """FRF 复数响应损失。

    frf_pred/frf_target: [total_N,F,2] or [B,N,F,2]
    """
    if frf_pred.shape != frf_target.shape:
        frf_target = frf_target.reshape(frf_pred.shape)

    amp_pred = torch.linalg.norm(frf_pred, dim=-1).clamp_min(eps)
    amp_target = torch.linalg.norm(frf_target, dim=-1).clamp_min(eps)

    loss_complex = F.l1_loss(frf_pred, frf_target)
    loss_log_amp = F.mse_loss(torch.log10(amp_pred), torch.log10(amp_target))

    amp_pred_norm = amp_pred / amp_pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    amp_target_norm = amp_target / amp_target.sum(dim=-1, keepdim=True).clamp_min(eps)
    cdf_pred = torch.cumsum(amp_pred_norm, dim=-1)
    cdf_target = torch.cumsum(amp_target_norm, dim=-1)
    loss_cdf = F.l1_loss(cdf_pred, cdf_target)

    return complex_weight * loss_complex + log_amp_weight * loss_log_amp + cdf_weight * loss_cdf


def zeta_physics_loss(zeta_pred, omega_pred_phys, phi_xyz_pred, spring_c_xyz,
                      batch_idx, zeta_material=0.002, eps=1e-8):
    """可选：三方向振型耗散得到的阻尼一致性损失。

    当前主模型只预测 Z 向振型，因此训练循环默认不调用该项。
    若后续模型输出 phi_xyz_pred=[total_N,K,3]，可启用：
        zeta_phys = zeta_material + Σ(Cxyz * phixyz²)/(2ω)
    """
    if phi_xyz_pred is None:
        return zeta_pred.new_tensor(0.0)
    n_graphs, n_modes = zeta_pred.shape
    zeta_phys = torch.zeros_like(zeta_pred)
    for g in range(n_graphs):
        mask = batch_idx == g
        if not torch.any(mask):
            continue
        phi_g = phi_xyz_pred[mask]      # [Ng,K,3]
        c_g = spring_c_xyz[mask]        # [Ng,3]
        diss = torch.sum(c_g.unsqueeze(1) * phi_g ** 2, dim=(0, 2))
        zeta_phys[g] = zeta_material + diss / (2.0 * omega_pred_phys[g].clamp_min(eps))
    return F.mse_loss(zeta_pred, zeta_phys)
