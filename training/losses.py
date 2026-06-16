from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12


def frequency_loss(omega_pred: torch.Tensor, omega_true: torch.Tensor, weight: float = 1.0):
    f_pred = omega_pred / (2.0 * torch.pi)
    f_true = omega_true / (2.0 * torch.pi)
    loss = F.smooth_l1_loss(torch.log(f_pred.clamp_min(1e-6)), torch.log(f_true.clamp_min(1e-6))) * weight
    rel = torch.abs(f_pred - f_true) / f_true.abs().clamp_min(1e-6) * 100.0
    return loss, rel.mean(dim=0), rel.mean()


def _align_phi_z(phi_pred: torch.Tensor, phi_true: torch.Tensor, batch_idx: torch.Tensor):
    out = torch.empty_like(phi_true)
    n_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    for g in range(n_graphs):
        m = batch_idx == g
        p = phi_pred[m]
        t = phi_true[m]
        sign = torch.sign(torch.sum(p * t, dim=0) + EPS)
        out[m] = t * sign.view(1, -1)
    return out


def phi_z_loss(phi_pred: torch.Tensor, phi_true: torch.Tensor, batch_idx: torch.Tensor,
               node_weight: torch.Tensor | None = None,
               weight: float = 1.0, mac_weight: float = 5.0, scale_weight: float = 1.0):
    if node_weight is None:
        node_weight = torch.ones(phi_true.shape[0], dtype=phi_true.dtype, device=phi_true.device)
    else:
        node_weight = node_weight.to(device=phi_true.device, dtype=phi_true.dtype)

    phi_true = _align_phi_z(phi_pred, phi_true, batch_idx)
    n_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    losses, mac_list, rmse_list, amp_list = [], [], [], []

    for g in range(n_graphs):
        m = batch_idx == g
        p = phi_pred[m]
        t = phi_true[m]
        w = node_weight[m].view(-1, 1)
        w = w / w.mean().clamp_min(1e-8)
        denom = w.sum().clamp_min(EPS)

        t_rms = torch.sqrt(torch.sum(w * t ** 2, dim=0) / denom + EPS)
        p_rms = torch.sqrt(torch.sum(w * p ** 2, dim=0) / denom + EPS)
        scale_floor = 0.1 * torch.median(t_rms.detach()).clamp_min(EPS) + EPS
        scale = torch.clamp(t_rms.detach(), min=scale_floor)

        mse = torch.sum(w * ((p - t) / scale.view(1, -1)) ** 2, dim=0) / denom
        amp = torch.abs(torch.log((p_rms + scale_floor) / (t_rms + scale_floor)))
        dot = torch.sum(w * p * t, dim=0)
        pp = torch.sum(w * p ** 2, dim=0)
        tt = torch.sum(w * t ** 2, dim=0)
        mac = dot ** 2 / (pp * tt + EPS)
        loss_k = mse + scale_weight * amp + mac_weight * (1.0 - mac)

        losses.append(loss_k)
        mac_list.append(mac)
        rmse_list.append(torch.sqrt(torch.mean((p - t) ** 2, dim=0)))
        amp_list.append(torch.abs(p_rms - t_rms) / (t_rms.abs() + EPS) * 100.0)

    loss_modes = torch.stack(losses, dim=0)
    mac = torch.stack(mac_list, dim=0).mean(dim=0)
    rmse = torch.stack(rmse_list, dim=0).mean(dim=0)
    amp_err = torch.stack(amp_list, dim=0).mean(dim=0)
    return loss_modes.mean() * weight, mac, rmse, amp_err


def modal_loss(out: dict, batch: dict, omega_weight: float = 1.0, phi_weight: float = 1.0,
               mac_weight: float = 5.0, scale_weight: float = 1.0):
    loss_w, w_per_mode, w_mean = frequency_loss(out["omega"], batch["modal_omega_phys"], weight=omega_weight)

    if phi_weight <= 0.0 or "phi_z" not in out:
        zero = loss_w.detach().new_tensor(0.0)
        return loss_w, {
            "loss": loss_w.detach(),
            "loss_omega": loss_w.detach(),
            "loss_phi": zero,
            "freq_percent": w_per_mode.detach(),
            "freq_mean_percent": w_mean.detach(),
            "mac": torch.zeros_like(w_per_mode.detach()),
            "phi_rmse": torch.zeros_like(w_per_mode.detach()),
            "phi_amp_percent": torch.zeros_like(w_per_mode.detach()),
        }

    loss_p, mac, phi_rmse, phi_amp = phi_z_loss(
        out["phi_z"], batch["modal_phi_z"], batch["batch"],
        node_weight=batch.get("node_weight"),
        weight=phi_weight, mac_weight=mac_weight, scale_weight=scale_weight,
    )
    loss = loss_w + loss_p
    return loss, {
        "loss": loss.detach(),
        "loss_omega": loss_w.detach(),
        "loss_phi": loss_p.detach(),
        "freq_percent": w_per_mode.detach(),
        "freq_mean_percent": w_mean.detach(),
        "mac": mac.detach(),
        "phi_rmse": phi_rmse.detach(),
        "phi_amp_percent": phi_amp.detach(),
    }
