"""
losses.py — 模态参数损失 + 排序对齐 + db/CDF FRF损失。
"""
import torch
import torch.nn.functional as F


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               omega_weight=200.0, zeta_weight=10.0, phi_weight=100.0):

    # 排序: 强制 ω₁<ω₂<ω₃
    omega_pred_sorted, sort_idx = torch.sort(omega_pred, dim=-1)          # [B,K]
    zeta_pred_sorted = torch.gather(zeta_pred, dim=-1, index=sort_idx)   # [B,K]

    # phi 排序: sort_idx[batch_idx] 将 (B,K)→(total_N,K)
    if phi_pred.dim() == 2:
        sort_idx_phi = sort_idx[batch_idx]                                # [total_N,K]
        phi_pred_sorted = torch.gather(phi_pred, dim=-1, index=sort_idx_phi)
    else:  # [B,N,K]
        sort_idx_phi = sort_idx.unsqueeze(1).expand(-1, phi_pred.shape[1], -1)
        phi_pred_sorted = torch.gather(phi_pred, dim=-1, index=sort_idx_phi)
        phi_pred_sorted = phi_pred_sorted.view(-1, phi_pred_sorted.shape[-1])
        phi_target = phi_target.view(-1, phi_target.shape[-1])

    # 标量损失
    mode_w = torch.tensor([1.0, 1.5, 2.0], device=omega_pred.device).unsqueeze(0)
    loss_omega = torch.mean(((omega_pred_sorted - omega_target) / (omega_target + 1e-8))**2 * mode_w) * omega_weight
    loss_zeta  = torch.mean(((zeta_pred_sorted - zeta_target) / (zeta_target + 1e-8))**2 * mode_w) * zeta_weight

    # 振型损失
    if batch_idx is not None:
        raw_phi_mse = 0.0
        num_graphs = int(batch_idx.max().item()) + 1
        for i in range(num_graphs):
            mask = (batch_idx == i)
            p_p = phi_pred_sorted[mask]; p_t = phi_target[mask]
            dot = torch.sum(p_p * p_t, dim=0, keepdim=True)
            sign = torch.sign(dot + 1e-8)
            raw_phi_mse += F.mse_loss(p_p, p_t * sign)
        raw_phi_mse = raw_phi_mse / num_graphs
    else:
        dot = torch.sum(phi_pred_sorted * phi_target, dim=1, keepdim=True)
        sign = torch.sign(dot + 1e-8)
        raw_phi_mse = F.mse_loss(phi_pred_sorted, phi_target * sign)

    loss_phi = raw_phi_mse * phi_weight
    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, loss_phi


def frf_loss(frf_pred, frf_target):
    amp_pred = torch.norm(frf_pred, dim=-1) + 1e-12
    amp_target = torch.norm(frf_target, dim=-1) + 1e-12
    loss_db = F.mse_loss(20 * torch.log10(amp_pred), 20 * torch.log10(amp_target))

    amp_pred_norm = amp_pred / amp_pred.sum(dim=-1, keepdim=True)
    amp_target_norm = amp_target / amp_target.sum(dim=-1, keepdim=True)
    cdf_pred = torch.cumsum(amp_pred_norm, dim=-1)
    cdf_target = torch.cumsum(amp_target_norm, dim=-1)
    loss_cdf = F.l1_loss(cdf_pred, cdf_target)

    return loss_db + 10.0 * loss_cdf
