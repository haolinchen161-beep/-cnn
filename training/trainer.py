"""Transolver 模态-FRF 训练的损失函数（纯 PyTorch 指针切片版）。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def align_targets_by_mac(phi_pred: torch.Tensor,
                         phi_target: torch.Tensor,
                         omega_target: torch.Tensor,
                         zeta_target: torch.Tensor,
                         node_counts: list) -> tuple:
    """基于 MAC 矩阵的匈牙利动态匹配，解决模态跳转引发的物理错位问题。

    对每个样本计算预测振型与目标振型之间的 K×K MAC 矩阵，
    然后用匈牙利算法找到最优排列，使目标按物理对应关系重新对齐。
    """
    aligned_phi_target = torch.empty_like(phi_target)
    aligned_omega_target = torch.empty_like(omega_target)
    aligned_zeta_target = torch.empty_like(zeta_target)

    ptr = 0
    for b, c in enumerate(node_counts):
        p = phi_pred[ptr:ptr + c]    # (N_c, K, 3)
        t = phi_target[ptr:ptr + c]  # (N_c, K, 3)

        # 计算 K×K MAC 矩阵
        # 跨节点(dim=0)与跨XYZ(dim=2)求内积
        sum_pt = torch.einsum('nki,nmi->km', p, t)  # (K, K)
        sum_p2 = p.pow(2).sum(dim=(0, 2)).clamp_min(1e-8)  # (K,)
        sum_t2 = t.pow(2).sum(dim=(0, 2)).clamp_min(1e-8)  # (K,)

        mac = sum_pt.pow(2) / (sum_p2.unsqueeze(1) * sum_t2.unsqueeze(0))  # (K, K)

        # 匈牙利算法：寻找使 MAC 对角线之和最大的最优排列
        mac_np = mac.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(mac_np, maximize=True)

        # 将目标按物理对应关系重新排列
        aligned_phi_target[ptr:ptr + c] = t[:, col_ind, :]
        aligned_omega_target[b] = omega_target[b, col_ind]
        aligned_zeta_target[b] = zeta_target[b, col_ind]

        ptr += c

    return aligned_phi_target, aligned_omega_target, aligned_zeta_target


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
            if is_3d:
                # 3D 向量总能量归一化：联合 XYZ 三向计算统一的 RMS
                # 避免 X/Y 微小值被独立放大到和 Z 同等权重
                p_rms = torch.sqrt(p.pow(2).sum(dim=-1).mean(dim=0).unsqueeze(-1) + norm_eps)
                t_rms = torch.sqrt(t.pow(2).sum(dim=-1).mean(dim=0).unsqueeze(-1) + norm_eps)
            else:
                # 1D (如 Z 向振型)，维持原有逻辑
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

    # 核心修改：通过匈牙利匹配将 Target 在物理上对齐网络预测
    # 解决模态跳转导致的通道错位问题
    phi_target, omega_target, zeta_target = align_targets_by_mac(
        phi_pred.detach(),  # 使用 detach 避免影响计算图
        batch_data['modal_phi_xyz'],
        batch_data['modal_omega'],
        batch_data['modal_zeta'],
        node_counts
    )

    # 固有频率损失（现在是在匹配后的通道上算）
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
