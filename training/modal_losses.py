from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12


def frequency_loss(pred: torch.Tensor, target: torch.Tensor):
    """固有频率损失：在 Hz 空间使用 log loss，频率本身不按方向加权。"""
    fp = pred / (2.0 * torch.pi)
    ft = target / (2.0 * torch.pi)
    rel = (fp - ft) / ft.clamp_min(1e-6)
    loss = F.smooth_l1_loss(torch.log(fp.clamp_min(1e-6)), torch.log(ft.clamp_min(1e-6)))
    return loss, {
        "freq_mae_hz": torch.abs(fp - ft).mean().detach(),
        "freq_mape_percent": (torch.abs(rel).mean() * 100.0).detach(),
    }


def _as_phi_z(phi: torch.Tensor) -> torch.Tensor:
    if phi.ndim == 3:
        return phi[..., 2]
    if phi.ndim == 2:
        return phi
    raise ValueError(f"Expected phi as [N,K] or [N,K,3], got {tuple(phi.shape)}")


def align_phi_z(pred_z: torch.Tensor, target_z: torch.Tensor, batch: torch.Tensor, node_weight: torch.Tensor | None = None) -> torch.Tensor:
    """逐样本、逐模态符号对齐。模态振型整体正负号任意，因此训练前需要对齐。"""
    out = torch.empty_like(target_z)
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if node_weight is None:
        node_weight = torch.ones(pred_z.shape[0], dtype=pred_z.dtype, device=pred_z.device)
    else:
        node_weight = node_weight.to(pred_z.device, dtype=pred_z.dtype)

    for g in range(n_graphs):
        mask = batch == g
        p = pred_z[mask]
        t = target_z[mask]
        w = node_weight[mask].view(-1, 1)
        dot = torch.sum(w * p * t, dim=0)
        sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
        out[mask] = t * sign.view(1, -1)
    return out


def weighted_phi_z_terms(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    target_xyz: torch.Tensor,
    batch: torch.Tensor,
    node_weight: torch.Tensor | None = None,
    min_mode_weight: float = 0.2,
    scale_floor_ratio: float = 0.1,
):
    """Z 向振型损失项，并按每阶模态的 Z 向主导程度加权。"""
    pred_z = _as_phi_z(pred_z)
    target_z = _as_phi_z(target_z)
    if target_xyz.ndim != 3 or target_xyz.shape[-1] != 3:
        raise ValueError(f"target_xyz must be [N,K,3], got {tuple(target_xyz.shape)}")

    if node_weight is None:
        node_weight = torch.ones(pred_z.shape[0], dtype=pred_z.dtype, device=pred_z.device)
    else:
        node_weight = node_weight.to(pred_z.device, dtype=pred_z.dtype)

    target_z = align_phi_z(pred_z, target_z, batch, node_weight)

    mse_all, scale_all, mac_all = [], [], []
    z_ratio_all, mode_weight_all, mac_gate_all = [], [], []
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

    for g in range(n_graphs):
        mask = batch == g
        p = pred_z[mask]
        t = target_z[mask]
        xyz = target_xyz[mask]
        w_node = node_weight[mask].view(-1, 1)
        w_node = w_node / w_node.mean().clamp_min(1e-8)

        ez = torch.sum(w_node * t ** 2, dim=0)
        eall = torch.sum(w_node.unsqueeze(-1) * xyz ** 2, dim=(0, 2)).clamp_min(EPS)
        z_ratio = (ez / eall).clamp(0.0, 1.0)
        mode_weight = min_mode_weight + (1.0 - min_mode_weight) * z_ratio.detach()

        denom_node = w_node.sum().clamp_min(EPS)
        true_rms = torch.sqrt(torch.sum(w_node * t ** 2, dim=0) / denom_node + EPS)
        pred_rms = torch.sqrt(torch.sum(w_node * p ** 2, dim=0) / denom_node + EPS)

        # 给尺度归一化设置下限，避免非 Z 主导模态的极小 phi_z 使 loss 被过度放大。
        median_rms = torch.median(true_rms.detach()).clamp_min(EPS)
        scale_floor = scale_floor_ratio * median_rms + EPS
        scale = torch.clamp(true_rms.detach(), min=scale_floor)

        mse_k = torch.sum(w_node * ((p - t) / scale.view(1, -1)) ** 2, dim=0) / denom_node
        scale_k = torch.abs(torch.log((pred_rms + scale_floor) / (true_rms + scale_floor)))

        dot = torch.sum(w_node * p * t, dim=0)
        pp = torch.sum(w_node * p ** 2, dim=0)
        tt = torch.sum(w_node * t ** 2, dim=0)
        mac_k = dot ** 2 / (pp * tt + EPS)

        # 真实 phi_z 极小时，MAC 不稳定，因此降低 MAC 项影响。
        mac_gate = (true_rms.detach() / (true_rms.detach() + scale_floor)).clamp(0.0, 1.0)

        mse_all.append(mse_k)
        scale_all.append(scale_k)
        mac_all.append(mac_k)
        z_ratio_all.append(z_ratio.detach())
        mode_weight_all.append(mode_weight.detach())
        mac_gate_all.append(mac_gate.detach())

    mse = torch.stack(mse_all, dim=0)
    scale = torch.stack(scale_all, dim=0)
    mac = torch.stack(mac_all, dim=0)
    z_ratio = torch.stack(z_ratio_all, dim=0)
    mode_weight = torch.stack(mode_weight_all, dim=0)
    mac_gate = torch.stack(mac_gate_all, dim=0)
    return mse, scale, mac, z_ratio, mode_weight, mac_gate


def modal_loss(
    out,
    batch,
    freq_weight: float = 1.0,
    phi_weight: float = 1.0,
    mac_weight: float = 5.0,
    scale_weight: float = 1.0,
    min_mode_weight: float = 0.2,
):
    lf, fm = frequency_loss(out["omega"], batch["modal_omega_phys"])

    pred_z = out.get("phi_z", out.get("phi"))
    target_z = batch.get("modal_phi_z", batch["modal_phi"])
    target_xyz = batch.get("modal_phi_xyz")
    if target_xyz is None:
        z = _as_phi_z(target_z)
        target_xyz = torch.zeros(z.shape[0], z.shape[1], 3, dtype=z.dtype, device=z.device)
        target_xyz[..., 2] = z

    mse_k, scale_k, mac_k, z_ratio, mode_weight, mac_gate = weighted_phi_z_terms(
        pred_z,
        target_z,
        target_xyz,
        batch["batch"],
        batch.get("node_weight"),
        min_mode_weight=min_mode_weight,
    )

    loss_k = mse_k + scale_weight * scale_k + mac_weight * mac_gate * (1.0 - mac_k)
    lphi = torch.sum(mode_weight * loss_k) / mode_weight.sum().clamp_min(EPS)
    loss = freq_weight * lf + phi_weight * lphi

    metrics = {
        "loss": loss.detach(),
        "loss_freq": lf.detach(),
        "loss_phi": lphi.detach(),
        "phi_z_mse": torch.mean(mse_k).detach(),
        "phi_z_scale": torch.mean(scale_k).detach(),
        "phi_z_mac": torch.mean(mac_k).detach(),
        "phi_z_mac_gated": torch.sum(mode_weight * mac_gate * mac_k).detach() / torch.sum(mode_weight * mac_gate).clamp_min(EPS).detach(),
        "mode_weight_mean": torch.mean(mode_weight).detach(),
        "dir_z_ratio_mean": torch.mean(z_ratio).detach(),
        **fm,
    }

    k_modes = int(mac_k.shape[1])
    for k in range(k_modes):
        metrics[f"phi_z_mac_mode{k + 1}"] = mac_k[:, k].mean().detach()
        metrics[f"dir_z_ratio_mode{k + 1}"] = z_ratio[:, k].mean().detach()
        metrics[f"mode_weight_mode{k + 1}"] = mode_weight[:, k].mean().detach()

    metrics["phi_mse"] = metrics["phi_z_mse"]
    metrics["phi_scale"] = metrics["phi_z_scale"]
    metrics["phi_mac"] = metrics["phi_z_mac"]
    if k_modes >= 1:
        metrics["phi_mac_mode1"] = metrics["phi_z_mac_mode1"]
    if k_modes >= 2:
        metrics["phi_mac_mode2"] = metrics["phi_z_mac_mode2"]
    if k_modes >= 3:
        metrics["phi_mac_mode3"] = metrics["phi_z_mac_mode3"]

    return loss, metrics
