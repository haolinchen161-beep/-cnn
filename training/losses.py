"""
losses.py — 模态参数损失 + 符号对齐MSE + db/CDF FRF损失。
"""
import torch
import torch.nn.functional as F


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               omega_weight=200.0, zeta_weight=10.0, phi_weight=1.0):
    """模态参数损失。符号对齐MSE强制匹配振型绝对幅值。"""

    loss_omega = torch.mean(((omega_pred - omega_target) / (omega_target + 1e-8))**2) * omega_weight
    loss_zeta  = torch.mean(((zeta_pred - zeta_target) / (zeta_target + 1e-8))**2) * zeta_weight

    if batch_idx is not None:
        if phi_pred.dim() == 3:
            phi_pred = phi_pred.view(-1, phi_pred.shape[-1])
            phi_target = phi_target.view(-1, phi_target.shape[-1])

        raw_phi_mse = 0.0
        num_graphs = int(batch_idx.max().item()) + 1
        for i in range(num_graphs):
            mask = (batch_idx == i)
            p_p = phi_pred[mask]; p_t = phi_target[mask]
            dot = torch.sum(p_p * p_t, dim=0, keepdim=True)
            sign = torch.sign(dot + 1e-8)
            aligned_t = p_t * sign
            raw_phi_mse += F.mse_loss(p_p, aligned_t)
        raw_phi_mse = raw_phi_mse / num_graphs
    else:
        dot = torch.sum(phi_pred * phi_target, dim=1, keepdim=True)
        sign = torch.sign(dot + 1e-8)
        aligned_t = phi_target * sign
        raw_phi_mse = F.mse_loss(phi_pred, aligned_t)

    loss_phi = raw_phi_mse * phi_weight
    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, raw_phi_mse


def frf_loss(frf_pred, frf_target):
    # dB域 MSE + CDF 横向引力
    amp_pred = torch.norm(frf_pred, dim=-1) + 1e-12
    amp_target = torch.norm(frf_target, dim=-1) + 1e-12
    loss_db = F.mse_loss(20 * torch.log10(amp_pred), 20 * torch.log10(amp_target))

    amp_pred_norm = amp_pred / amp_pred.sum(dim=-1, keepdim=True)
    amp_target_norm = amp_target / amp_target.sum(dim=-1, keepdim=True)
    cdf_pred = torch.cumsum(amp_pred_norm, dim=-1)
    cdf_target = torch.cumsum(amp_target_norm, dim=-1)
    loss_cdf = F.l1_loss(cdf_pred, cdf_target)

    return loss_db + 50.0 * loss_cdf
