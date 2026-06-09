"""Transolver 模态-FRF 训练的损失函数（纯 PyTorch 指针切片版）。"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def relative_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """相对 L1 损失：|pred - target| / (|target| + eps)。"""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))


def log_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """对数空间 L1 损失：|log(pred) - log(target)|。"""
    return torch.mean(torch.abs(
        torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))
    ))


def sign_invariant_mse(pred: torch.Tensor,
                       target: torch.Tensor,
                       node_counts: list,
                       normalize: bool = True,
                       norm_eps: float = 1e-8) -> torch.Tensor:
    """逐样本、逐模态符号对齐的振型 MSE（纯 PyTorch 指针切片版）。

    对于 3D 张量 (N, K, 3)，符号在节点和坐标轴维度上联合计算，
    确保每个模态只有一个全局符号，避免独立翻转坐标轴。
    """
    mse_parts = []
    ptr = 0
    is_3d = pred.dim() == 3  # (N, K, 3)

    for c in node_counts:
        p = pred[ptr:ptr + c]
        t = target[ptr:ptr + c]

        if normalize:
            p_rms = torch.sqrt(p.pow(2).mean(dim=0) + norm_eps)
            t_rms = torch.sqrt(t.pow(2).mean(dim=0) + norm_eps)
            p = p / p_rms.clamp_min(norm_eps)
            t = t / t_rms.clamp_min(norm_eps)

        # 核心修改：3D 张量在节点和坐标轴维度联合求和，得到唯一全局符号
        if is_3d:
            dot = (p * t).sum(dim=(0, 2))  # (K,)
            sign = torch.where(dot >= 0,
                               torch.tensor(1.0, device=dot.device),
                               torch.tensor(-1.0, device=dot.device))
            sign = sign.unsqueeze(0).unsqueeze(-1)  # (1, K, 1) 广播
        else:
            dot = (p * t).sum(dim=0)
            sign = torch.where(dot >= 0,
                               torch.tensor(1.0, device=dot.device),
                               torch.tensor(-1.0, device=dot.device))

        mse_parts.append((p - t * sign).pow(2).mean())
        ptr += c

    if not mse_parts:
        return pred.new_tensor(0.0)
    return torch.stack(mse_parts).mean()


def mac_loss(pred: torch.Tensor,
             target: torch.Tensor,
             node_counts: list,
             eps: float = 1e-12) -> torch.Tensor:
    """1 - MAC 损失（纯 PyTorch 指针切片版）。"""
    mac_parts = []
    ptr = 0
    for c in node_counts:
        p = pred[ptr:ptr + c]
        t = target[ptr:ptr + c]
        sum_pt = (p * t).sum(dim=0)
        sum_p2 = p.pow(2).sum(dim=0)
        sum_t2 = t.pow(2).sum(dim=0)
        mac_parts.append(1.0 - sum_pt.pow(2) / (sum_p2 * sum_t2 + eps))
        ptr += c

    if not mac_parts:
        return pred.new_tensor(0.0)
    return torch.stack(mac_parts).mean()


def modal_loss(outputs: Dict[str, torch.Tensor],
               batch_data: Dict[str, torch.Tensor],
               weights: Dict[str, float] | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算模态参数损失（纯 PyTorch 指针切片版）。

    目标:
        modal_omega      (B, K), rad/s
        modal_zeta       (B, K)
        modal_phi_xyz    (total_N, K, 3)
    """
    weights = weights or {}
    batch = batch_data['batch']
    # 优先读取 Dataset 传来的 CPU 列表，避免 GPU bincount() 同步
    node_counts = batch_data.get('node_counts', batch.bincount().tolist())
    num_graphs = len(node_counts)

    omega_pred = outputs['modal_omega']
    zeta_pred = outputs['modal_zeta']
    phi_pred = outputs['modal_phi_xyz']
    omega_target = batch_data['modal_omega']
    zeta_target = batch_data['modal_zeta']
    phi_target = batch_data['modal_phi_xyz']

    # 固有频率损失
    loss_omega = relative_l1(omega_pred, omega_target, eps=1e-5)

    # 阻尼比损失
    loss_zeta = (
        0.5 * log_l1(zeta_pred, zeta_target, eps=1e-5) +
        0.5 * relative_l1(zeta_pred, zeta_target, eps=1e-5)
    )

    # 逐模态误差（用于日志）
    omega_per_mode = torch.mean(
        torch.abs(omega_pred - omega_target) / (torch.abs(omega_target) + 1e-5), dim=0)  # (K,)
    zeta_per_mode = torch.mean(
        torch.abs(zeta_pred - zeta_target) / (torch.abs(zeta_target) + 1e-5), dim=0)  # (K,)

    # 方向感知振型损失
    resp_idx = int(batch_data.get('response_dir_index',
                                   torch.tensor(2)).flatten()[0].item())
    phi_resp_pred = phi_pred[..., resp_idx]   # (total_N, K)
    phi_resp_target = phi_target[..., resp_idx]

    loss_phi_resp = sign_invariant_mse(phi_resp_pred, phi_resp_target, node_counts, normalize=True)
    # 直接传入 3D 张量，sign_invariant_mse 内部自动处理 3D 符号对齐
    loss_phi_xyz = sign_invariant_mse(phi_pred, phi_target, node_counts, normalize=True)
    loss_mac = mac_loss(phi_resp_pred, phi_resp_target, node_counts)

    # 逐模态 φ 误差（指针切片版）
    phi_per_mode_parts = []
    ptr = 0
    for c in node_counts:
        p = phi_resp_pred[ptr:ptr + c]
        t = phi_resp_target[ptr:ptr + c]
        p_rms = torch.sqrt(p.pow(2).mean(dim=0) + 1e-8)
        t_rms = torch.sqrt(t.pow(2).mean(dim=0) + 1e-8)
        p_n = p / p_rms.clamp_min(1e-8)
        t_n = t / t_rms.clamp_min(1e-8)
        dot = (p_n * t_n).sum(dim=0)
        sign = torch.where(dot >= 0,
                           torch.tensor(1.0, device=dot.device),
                           torch.tensor(-1.0, device=dot.device))
        phi_per_mode_parts.append((p_n - t_n * sign).pow(2).mean(dim=0))
        ptr += c
    phi_per_mode = torch.stack(phi_per_mode_parts).mean(dim=0)  # (K,)

    total = (
        weights.get('omega', 1.0) * loss_omega +
        weights.get('zeta', 0.5) * loss_zeta +
        weights.get('phi_resp', 1.0) * loss_phi_resp +
        weights.get('phi_xyz', 0.25) * loss_phi_xyz +
        weights.get('mac', 0.2) * loss_mac
    )
    logs = {
        'loss_omega': loss_omega.detach(),
        'loss_zeta': loss_zeta.detach(),
        'loss_phi_resp': loss_phi_resp.detach(),
        'loss_phi_xyz': loss_phi_xyz.detach(),
        'loss_mac': loss_mac.detach(),
        **{f'omega_k{k}': omega_per_mode[k].detach() for k in range(omega_per_mode.shape[0])},
        **{f'zeta_k{k}': zeta_per_mode[k].detach() for k in range(zeta_per_mode.shape[0])},
        **{f'phi_k{k}': phi_per_mode[k].detach() for k in range(phi_per_mode.shape[0])},
    }
    return total, logs


def frf_loss(frf_pred: torch.Tensor, frf_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """复数 FRF 损失：复数 L1 + 对数幅值 + dB 项。"""
    loss_complex = F.l1_loss(frf_pred, frf_target)

    amp_pred = torch.linalg.norm(frf_pred, dim=-1).clamp_min(1e-8)
    amp_target = torch.linalg.norm(frf_target, dim=-1).clamp_min(1e-8)

    loss_log_amp = F.l1_loss(torch.log(amp_pred), torch.log(amp_target))
    loss_db = F.l1_loss(20.0 * torch.log10(amp_pred), 20.0 * torch.log10(amp_target))

    total = loss_complex + 0.1 * loss_log_amp + 0.01 * loss_db
    return total, {
        'loss_frf_complex': loss_complex.detach(),
        'loss_frf_log_amp': loss_log_amp.detach(),
        'loss_frf_db': loss_db.detach(),
    }


def total_loss(outputs: Dict[str, torch.Tensor],
               batch_data: Dict[str, torch.Tensor],
               config: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """联合模态 + 可选 FRF 损失。"""
    modal, modal_logs = modal_loss(outputs, batch_data, config.get('modal_loss_weights', {}))
    logs = {'loss_modal': modal.detach(), **modal_logs}
    total = modal
    if outputs.get('frf') is not None and config.get('use_frf_loss', True):
        frf, frf_logs = frf_loss(outputs['frf'], batch_data['point_frf'])
        total = total + config.get('frf_loss_weight', 1.0) * frf
        logs.update({'loss_frf': frf.detach(), **frf_logs})
    logs['loss_total'] = total.detach()
    return total, logs
