from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def modal_loss(outputs: Dict[str, torch.Tensor],
               batch: Dict[str, torch.Tensor],
               freq_weight: float = 1.0,
               phi_weight: float = 1.0,
               mac_weight: float = 20.0,
               std_weight: float = 2.0,
               direction_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    omega_pred = outputs["omega"]
    phi_pred = outputs["phi"]
    omega_true = batch["modal_omega_phys"]
    phi_true = batch["modal_phi"]
    batch_idx = batch["batch"]

    loss_f, fm = frequency_loss(omega_pred, omega_true)
    phi_ref, orient = orient_target_per_graph(phi_pred, phi_true, batch_idx)
    loss_p, pm = mode_loss(phi_pred, phi_ref, batch_idx, batch.get("node_weight"),
                           mac_weight, std_weight, direction_weight)

    total = freq_weight * loss_f + phi_weight * loss_p
    metrics = {**fm, **pm}
    metrics.update({
        "loss": total.detach(),
        "loss_freq": loss_f.detach(),
        "loss_phi": loss_p.detach(),
        "orient_mean": orient.float().mean().detach(),
    })
    return total, metrics


def frequency_loss(omega_pred: torch.Tensor, omega_true: torch.Tensor):
    f_pred = omega_pred / (2.0 * torch.pi)
    f_true = omega_true / (2.0 * torch.pi)
    rel = (f_pred - f_true) / f_true.clamp_min(1e-6)

    l_log = F.smooth_l1_loss(torch.log(f_pred.clamp_min(1e-6)), torch.log(f_true.clamp_min(1e-6)))
    l_rel = F.smooth_l1_loss(rel * 100.0, torch.zeros_like(rel))
    l_gap = F.smooth_l1_loss(f_pred[:, 1:] - f_pred[:, :-1], f_true[:, 1:] - f_true[:, :-1])
    loss = 5.0 * l_log + 0.1 * l_rel + 0.001 * l_gap

    return loss, {
        "freq_mape_percent": (torch.abs(rel).mean() * 100.0).detach(),
        "freq_mae_hz": torch.abs(f_pred - f_true).mean().detach(),
        "loss_freq_log": l_log.detach(),
        "loss_freq_rel": l_rel.detach(),
        "loss_gap": l_gap.detach(),
    }


def orient_target_per_graph(phi_pred: torch.Tensor, phi_true: torch.Tensor, batch_idx: torch.Tensor):
    out = torch.empty_like(phi_true)
    orient_list = []
    n_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    for gid in range(n_graphs):
        m = batch_idx == gid
        dot = torch.sum(phi_pred[m] * phi_true[m], dim=(0, 2))
        orient = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
        out[m] = phi_true[m] * orient.view(1, -1, 1)
        orient_list.append(orient)
    return out, torch.stack(orient_list, dim=0) if orient_list else phi_true.new_zeros(0)


def mode_loss(phi_pred: torch.Tensor,
              phi_target: torch.Tensor,
              batch_idx: torch.Tensor,
              node_weight: torch.Tensor | None,
              mac_weight: float,
              std_weight: float,
              direction_weight: float):
    if node_weight is None:
        node_weight = torch.ones(phi_pred.shape[0], dtype=phi_pred.dtype, device=phi_pred.device)
    else:
        node_weight = node_weight.to(device=phi_pred.device, dtype=phi_pred.dtype)

    mse_terms, mac_terms, std_terms, dir_terms = [], [], [], []
    n_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    for gid in range(n_graphs):
        m = batch_idx == gid
        p = phi_pred[m]
        t = phi_target[m]
        w = node_weight[m].view(-1, 1, 1)
        w = w / w.mean().clamp_min(1e-8)

        p_std = torch.std(p.transpose(0, 1).reshape(p.shape[1], -1), dim=1).clamp_min(1e-8)
        t_std = torch.std(t.transpose(0, 1).reshape(t.shape[1], -1), dim=1).clamp_min(1e-8)

        mse_terms.append(torch.mean(w * ((p / p_std.view(1, -1, 1)) - (t / t_std.view(1, -1, 1))) ** 2))
        mac_terms.append(mac_per_mode(p, t))
        std_terms.append(torch.mean(torch.abs(torch.log(p_std / t_std))))

        p_dir = torch.sqrt(torch.sum(p ** 2, dim=0) + 1e-8)
        t_dir = torch.sqrt(torch.sum(t ** 2, dim=0) + 1e-8)
        dir_terms.append(torch.mean(torch.abs(torch.log((p_dir + 1e-8) / (t_dir + 1e-8)))))

    mse = torch.stack(mse_terms).mean()
    mac_values = torch.stack(mac_terms, dim=0)
    mac_loss = (1.0 - mac_values).mean()
    std_loss = torch.stack(std_terms).mean()
    dir_loss = torch.stack(dir_terms).mean()
    total = mse + mac_weight * mac_loss + std_weight * std_loss + direction_weight * dir_loss

    return total, {
        "phi_nrmse": torch.sqrt(mse.detach().clamp_min(0.0)),
        "phi_mac": mac_values.mean().detach(),
        "phi_mac_mode1": mac_values[:, 0].mean().detach(),
        "phi_mac_mode2": mac_values[:, 1].mean().detach(),
        "phi_mac_mode3": mac_values[:, 2].mean().detach(),
        "loss_phi_nrmse": mse.detach(),
        "loss_phi_mac": mac_loss.detach(),
        "loss_phi_std": std_loss.detach(),
        "loss_phi_dir": dir_loss.detach(),
    }


def mac_per_mode(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = torch.sum(a * b, dim=(0, 2)) ** 2
    den = torch.sum(a ** 2, dim=(0, 2)) * torch.sum(b ** 2, dim=(0, 2))
    return num / den.clamp_min(1e-12)


@torch.no_grad()
def evaluate_modal_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    _, fm = frequency_loss(outputs["omega"], batch["modal_omega_phys"])
    phi_ref, _ = orient_target_per_graph(outputs["phi"], batch["modal_phi"], batch["batch"])
    _, pm = mode_loss(outputs["phi"], phi_ref, batch["batch"], batch.get("node_weight"), 20.0, 2.0, 1.0)
    return {**fm, **pm}
