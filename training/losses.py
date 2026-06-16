"""
losses.py — MeshGraphNet modal parameter losses + dB/CDF FRF loss.

Rebuilt to match the current CNN physics branch:
- omega is physical rad/s, not normalized.
- phi is full 3D [total_N,K,3].
- mode-shape loss uses per-graph sign alignment, joint std normalization,
  3D MAC, and XYZ direction norm consistency.
- branch_loss supervises each mode's XYZ energy ratio and mildly reweights
  mode-3 X/Y minority types.
- modal_loss_z_only is a simplified channel for Z-FRF studies: it supervises
  only omega and the Z projection of phi, and may disable zeta supervision.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _mac_per_graph(phi_pred, phi_target):
    """3D single-graph MAC: phi [N,K,3] -> [K]."""
    num = torch.sum(phi_pred * phi_target, dim=(0, 2)) ** 2
    den = torch.sum(phi_pred ** 2, dim=(0, 2)) * torch.sum(phi_target ** 2, dim=(0, 2)) + 1e-8
    return num / den


def _mac_z_per_graph(phi_z_pred, phi_z_target):
    """Z-projection single-graph MAC: phi_z [N,K] -> [K]."""
    num = torch.sum(phi_z_pred * phi_z_target, dim=0) ** 2
    den = torch.sum(phi_z_pred ** 2, dim=0) * torch.sum(phi_z_target ** 2, dim=0) + 1e-8
    return num / den


def _ensure_phi3d(phi):
    if phi.dim() == 4:  # [B,N,K,3]
        return phi.reshape(-1, phi.shape[-2], phi.shape[-1])
    if phi.dim() == 3:
        return phi
    if phi.dim() == 2:  # legacy Z-only [N,K]
        out = phi.new_zeros(phi.shape[0], phi.shape[1], 3)
        out[..., 2] = phi
        return out
    raise ValueError(f"Unsupported phi shape: {tuple(phi.shape)}")


def _frequency_loss(omega_phys_pred, omega_phys_target, zeta_target, omega_weight=1.0):
    """Frequency loss in Hz space, shared by full-XYZ and Z-only channels."""
    f_pred_hz = omega_phys_pred / (2.0 * torch.pi)
    f_true_hz = omega_phys_target / (2.0 * torch.pi)

    mode_w = f_pred_hz.new_tensor([1.0, 1.5, 2.2]).view(1, 3)

    abs_err = F.smooth_l1_loss(f_pred_hz, f_true_hz, reduction='none')
    loss_freq_abs = torch.mean(abs_err * mode_w)

    rel_err = (f_pred_hz - f_true_hz) / (f_true_hz + 1e-8)
    rel_loss = F.smooth_l1_loss(rel_err * 100.0, torch.zeros_like(rel_err), reduction='none')
    loss_freq_rel = torch.mean(rel_loss * mode_w)

    if f_pred_hz.shape[-1] >= 3:
        gap_pred = f_pred_hz[:, 1:] - f_pred_hz[:, :-1]
        gap_true = f_true_hz[:, 1:] - f_true_hz[:, :-1]
        gap_err = F.smooth_l1_loss(gap_pred, gap_true, reduction='none')
        gap_w = f_pred_hz.new_tensor([1.2, 1.8]).view(1, 2)
        loss_gap = torch.mean(gap_err * gap_w)
    else:
        loss_gap = f_pred_hz.new_tensor(0.0)

    rel_abs = torch.abs(rel_err)
    peak_sensitive = torch.clamp(rel_abs / (zeta_target + 1e-8), max=100.0)

    return (
        0.5 * loss_freq_abs + 10.0 * loss_freq_rel + 0.5 * loss_gap + 0.05 * peak_sensitive.mean()
    ) * omega_weight


def modal_loss(omega_phys_pred, omega_phys_target,
               log_zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               omega_weight=1.0, zeta_weight=10.0, phi_weight=3.0):
    """CNN-compatible physical modal loss for the MeshGraphNet branch.

    Args:
        omega_phys_pred:   [B,K] rad/s
        omega_phys_target: [B,K] rad/s
        log_zeta_pred:     [B,K]
        zeta_target:       [B,K]
        phi_pred:          [total_N,K,3]
        phi_target:        [total_N,K,3]
        batch_idx:         [total_N]
    """
    loss_omega = _frequency_loss(omega_phys_pred, omega_phys_target, zeta_target, omega_weight)

    # Damping loss in log domain. It becomes exactly zero when zeta_weight=0.
    if zeta_weight > 0:
        log_zeta_target = torch.log(zeta_target + 1e-8)
        loss_zeta = F.smooth_l1_loss(log_zeta_pred, log_zeta_target) * zeta_weight
    else:
        loss_zeta = loss_omega.new_tensor(0.0)

    # Full 3D mode-shape loss.
    phi_pred = _ensure_phi3d(phi_pred)
    phi_target = _ensure_phi3d(phi_target)

    if batch_idx is not None:
        aligned_target = torch.empty_like(phi_target)
        n_graphs_sign = int(batch_idx.max().item()) + 1
        for b in range(n_graphs_sign):
            m = batch_idx == b
            dot_b = torch.sum(phi_pred[m] * phi_target[m], dim=(0, 2), keepdim=True)
            aligned_target[m] = phi_target[m] * torch.sign(dot_b + 1e-8)
    else:
        dot = torch.sum(phi_pred * phi_target, dim=(0, 2), keepdim=True)
        aligned_target = phi_target * torch.sign(dot + 1e-8)

    if batch_idx is not None:
        n_graphs = int(batch_idx.max().item()) + 1
        p_std_list, t_std_list, direc_weight_list = [], [], []
        for i in range(n_graphs):
            mask = (batch_idx == i)
            p_i = phi_pred[mask]
            t_i = aligned_target[mask]

            p_std_i = torch.std(p_i.transpose(0, 1).reshape(p_i.shape[1], -1), dim=1) + 1e-8
            t_std_i = torch.std(t_i.transpose(0, 1).reshape(t_i.shape[1], -1), dim=1) + 1e-8
            p_std_list.append(p_std_i)
            t_std_list.append(t_std_i)

            energy_i = torch.sum(t_i ** 2, dim=0)  # [K,3]
            w_i = (energy_i / (energy_i.sum(dim=-1, keepdim=True) + 1e-8)) * 3.0
            direc_weight_list.append(w_i.unsqueeze(0).expand(mask.sum(), -1, -1))

        p_std = torch.stack(p_std_list, dim=0)
        t_std = torch.stack(t_std_list, dim=0)
        p_std_view = p_std[batch_idx].unsqueeze(-1)
        t_std_view = t_std[batch_idx].unsqueeze(-1)
        direc_weight = torch.cat(direc_weight_list, dim=0)
    else:
        p_std = torch.std(phi_pred.transpose(0, 1).reshape(phi_pred.shape[1], -1), dim=1) + 1e-8
        t_std = torch.std(aligned_target.transpose(0, 1).reshape(phi_pred.shape[1], -1), dim=1) + 1e-8
        p_std_view = p_std.view(1, -1, 1)
        t_std_view = t_std.view(1, -1, 1)
        energy = torch.sum(aligned_target ** 2, dim=0)
        direc_weight = (energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)) * 3.0

    phi_pred_norm = phi_pred / p_std_view
    phi_target_norm = aligned_target / t_std_view

    mse_elements = F.mse_loss(phi_pred_norm, phi_target_norm, reduction='none')
    if batch_idx is not None:
        raw_phi_mse = torch.mean(mse_elements * direc_weight)
    else:
        raw_phi_mse = torch.mean(mse_elements * direc_weight.unsqueeze(0))

    if batch_idx is not None:
        n_graphs = int(batch_idx.max().item()) + 1
        mac_loss_total = 0.0
        mac_list = []
        for i in range(n_graphs):
            mask = (batch_idx == i)
            mac = _mac_per_graph(phi_pred[mask], aligned_target[mask])
            mac_loss_total += (1.0 - mac).mean()
            mac_list.append(mac)
        loss_mac = mac_loss_total / n_graphs
        mac_per_mode = torch.stack(mac_list, dim=0).mean(dim=0)
    else:
        mac = _mac_per_graph(phi_pred, aligned_target)
        loss_mac = (1.0 - mac).mean()
        mac_per_mode = mac

    loss_std = F.smooth_l1_loss(p_std, t_std)
    loss_dir_norm = per_graph_direction_norm_loss(phi_pred, aligned_target, batch_idx)

    loss_phi = (10.0 * raw_phi_mse + 40.0 * loss_mac + 20.0 * loss_std + 10.0 * loss_dir_norm) * phi_weight

    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, loss_phi, mac_per_mode.detach()


def modal_loss_z_only(omega_phys_pred, omega_phys_target,
                      log_zeta_pred, zeta_target,
                      phi_pred, phi_target, batch_idx=None,
                      omega_weight=1.0, zeta_weight=0.0, phi_weight=3.0):
    """Z-projection modal loss for Z-FRF-focused experiments.

    The network may still output full phi[N,K,3], but this loss supervises only
    phi[..., 2]. This removes the X/Y direction-classification burden while
    preserving the physically relevant Z projection for Z-direction FRF.
    """
    loss_omega = _frequency_loss(omega_phys_pred, omega_phys_target, zeta_target, omega_weight)

    if zeta_weight > 0:
        log_zeta_target = torch.log(zeta_target + 1e-8)
        loss_zeta = F.smooth_l1_loss(log_zeta_pred, log_zeta_target) * zeta_weight
    else:
        loss_zeta = loss_omega.new_tensor(0.0)

    phi_pred = _ensure_phi3d(phi_pred)[..., 2]
    phi_target = _ensure_phi3d(phi_target)[..., 2]

    if batch_idx is not None:
        aligned_target = torch.empty_like(phi_target)
        n_graphs_sign = int(batch_idx.max().item()) + 1
        for b in range(n_graphs_sign):
            m = batch_idx == b
            dot_b = torch.sum(phi_pred[m] * phi_target[m], dim=0, keepdim=True)
            aligned_target[m] = phi_target[m] * torch.sign(dot_b + 1e-8)
    else:
        dot = torch.sum(phi_pred * phi_target, dim=0, keepdim=True)
        aligned_target = phi_target * torch.sign(dot + 1e-8)

    if batch_idx is not None:
        n_graphs = int(batch_idx.max().item()) + 1
        p_std_list, t_std_list = [], []
        for i in range(n_graphs):
            mask = batch_idx == i
            p_i = phi_pred[mask]
            t_i = aligned_target[mask]
            p_std_list.append(torch.std(p_i, dim=0) + 1e-8)
            t_std_list.append(torch.std(t_i, dim=0) + 1e-8)
        p_std = torch.stack(p_std_list, dim=0)
        t_std = torch.stack(t_std_list, dim=0)
        p_std_view = p_std[batch_idx]
        t_std_view = t_std[batch_idx]
    else:
        p_std = torch.std(phi_pred, dim=0).unsqueeze(0) + 1e-8
        t_std = torch.std(aligned_target, dim=0).unsqueeze(0) + 1e-8
        p_std_view = p_std
        t_std_view = t_std

    phi_pred_norm = phi_pred / p_std_view
    phi_target_norm = aligned_target / t_std_view
    raw_phi_mse = F.mse_loss(phi_pred_norm, phi_target_norm)

    if batch_idx is not None:
        n_graphs = int(batch_idx.max().item()) + 1
        mac_loss_total = 0.0
        mac_list = []
        for i in range(n_graphs):
            mask = batch_idx == i
            mac = _mac_z_per_graph(phi_pred[mask], aligned_target[mask])
            mac_loss_total += (1.0 - mac).mean()
            mac_list.append(mac)
        loss_mac = mac_loss_total / n_graphs
        mac_per_mode = torch.stack(mac_list, dim=0).mean(dim=0)
    else:
        mac = _mac_z_per_graph(phi_pred, aligned_target)
        loss_mac = (1.0 - mac).mean()
        mac_per_mode = mac

    loss_std = F.smooth_l1_loss(p_std, t_std)
    norm_p = torch.sqrt(torch.sum(phi_pred ** 2, dim=0) + 1e-8)
    norm_t = torch.sqrt(torch.sum(aligned_target ** 2, dim=0) + 1e-8)
    loss_amp = torch.mean(torch.abs(torch.log((norm_p + 1e-8) / (norm_t + 1e-8))))

    loss_phi = (10.0 * raw_phi_mse + 40.0 * loss_mac + 20.0 * loss_std + 10.0 * loss_amp) * phi_weight
    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, loss_phi, mac_per_mode.detach()


def frf_loss(frf_pred, frf_target):
    """dB + normalized-amplitude CDF loss used by the current CNN branch."""
    if frf_pred.shape != frf_target.shape:
        frf_target = frf_target.reshape(frf_pred.shape)

    amp_pred = torch.norm(frf_pred, dim=-1) + 1e-12
    amp_target = torch.norm(frf_target, dim=-1) + 1e-12

    loss_db = F.mse_loss(20 * torch.log10(amp_pred), 20 * torch.log10(amp_target))

    amp_pred_norm = amp_pred / amp_pred.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    amp_target_norm = amp_target / amp_target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cdf_pred = torch.cumsum(amp_pred_norm, dim=-1)
    cdf_target = torch.cumsum(amp_target_norm, dim=-1)
    loss_cdf = F.l1_loss(cdf_pred, cdf_target)

    return loss_db + 10.0 * loss_cdf


def per_graph_direction_norm_loss(phi_pred, phi_target, batch_idx, mode_weights=None):
    """Per-graph XYZ norm consistency loss."""
    phi_pred = _ensure_phi3d(phi_pred)
    phi_target = _ensure_phi3d(phi_target)

    if batch_idx is not None:
        n_graphs = int(batch_idx.max().item()) + 1
        losses = []
        for b in range(n_graphs):
            m = batch_idx == b
            p = phi_pred[m]
            t = phi_target[m]
            p_norm = torch.sqrt(torch.sum(p ** 2, dim=0) + 1e-8)
            t_norm = torch.sqrt(torch.sum(t ** 2, dim=0) + 1e-8)
            losses.append(torch.abs(torch.log((p_norm + 1e-8) / (t_norm + 1e-8))))
        loss = torch.stack(losses, dim=0)
    else:
        p_norm = torch.sqrt(torch.sum(phi_pred ** 2, dim=0) + 1e-8)
        t_norm = torch.sqrt(torch.sum(phi_target ** 2, dim=0) + 1e-8)
        loss = torch.abs(torch.log((p_norm + 1e-8) / (t_norm + 1e-8))).unsqueeze(0)

    if mode_weights is None:
        mode_weights = loss.new_tensor([0.5, 2.0, 4.0])

    return torch.mean(loss * mode_weights.view(1, -1, 1))


def branch_loss(branch_log_probs, phi_target, batch_idx, mode_weights=None):
    """KL supervision for mode-wise XYZ energy ratio.

    It mildly reweights minority mode-3 X/Y types according to the diagnostics
    observed on the CNN baseline.
    """
    phi_target = _ensure_phi3d(phi_target)
    n_graphs = int(batch_idx.max().item()) + 1
    kl_per_list = []
    class_weight_list = []

    for i in range(n_graphs):
        mask = batch_idx == i
        phi_i = phi_target[mask]
        energy = torch.sum(phi_i ** 2, dim=0)
        target_probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)

        kl_per = F.kl_div(
            branch_log_probs[i:i + 1],
            target_probs.unsqueeze(0),
            reduction='none'
        ).sum(dim=-1)  # [1,K]

        sample_w = torch.ones_like(kl_per)

        if target_probs.shape[0] >= 2:
            mode2_dir = torch.argmax(target_probs[1]).item()
            if mode2_dir != 2:
                sample_w[:, 1] = 2.0

        if target_probs.shape[0] >= 3:
            mode3_dir = torch.argmax(target_probs[2]).item()
            if mode3_dir == 0:
                sample_w[:, 2] = 3.0
            elif mode3_dir == 1:
                sample_w[:, 2] = 3.5
            else:
                sample_w[:, 2] = 1.0

        kl_per_list.append(kl_per)
        class_weight_list.append(sample_w)

    kl_all = torch.cat(kl_per_list, dim=0)
    class_w = torch.cat(class_weight_list, dim=0)

    if mode_weights is None:
        mode_weights = kl_all.new_tensor([0.5, 2.0, 4.0])

    return (kl_all * class_w * mode_weights.view(1, -1)).mean()


def zeta_physics_loss(*args, **kwargs):
    """Reserved compatibility hook."""
    if len(args) > 0 and torch.is_tensor(args[0]):
        return args[0].new_tensor(0.0)
    return torch.tensor(0.0)
