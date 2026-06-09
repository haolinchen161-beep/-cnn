"""Transolver 模态-FRF 训练的损失函数。"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch_scatter


def relative_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """相对 L1 损失：|pred - target| / (|target| + eps)。"""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))


def log_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """对数空间 L1 损失：|log(pred) - log(target)|。

    eps 默认为 1e-5，避免 1e-12 带来的数值风险。
    """
    return torch.mean(torch.abs(
        torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))
    ))


def rms_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMS 归一化：x / sqrt(mean(x²) + eps)，防止振型尺度主导训练。"""
    rms = torch.sqrt(torch.mean(x.pow(2), dim=0, keepdim=True) + eps)
    return x / rms.clamp_min(eps)


def sign_invariant_mse(pred: torch.Tensor,
                       target: torch.Tensor,
                       batch: torch.Tensor,
                       num_graphs: int,
                       normalize: bool = True,
                       norm_eps: float = 1e-8) -> torch.Tensor:
    """逐样本、逐模态符号对齐的振型 MSE（向量化版本）。

    先做 RMS 归一化消除振型尺度的影响，再逐模态确定符号使 MSE 最小。
    """
    # RMS 归一化（逐图统计，向量化）
    if normalize:
        p_rms = torch.sqrt(torch_scatter.scatter_mean(pred.pow(2), batch, dim=0) + norm_eps)  # (B, D)
        t_rms = torch.sqrt(torch_scatter.scatter_mean(target.pow(2), batch, dim=0) + norm_eps)
        pred_n = pred / p_rms[batch].clamp_min(norm_eps)
        target_n = target / t_rms[batch].clamp_min(norm_eps)
    else:
        pred_n, target_n = pred, target

    # 逐图逐模态符号对齐（向量化）
    dot = torch_scatter.scatter_sum(pred_n * target_n, batch, dim=0)  # (B, D)
    sign = torch.where(dot >= 0, torch.tensor(1.0, device=dot.device, dtype=dot.dtype),
                       torch.tensor(-1.0, device=dot.device, dtype=dot.dtype))

    # MSE（逐图平均后全局平均）
    diff = pred_n - target_n * sign[batch]
    mse_per_graph = torch_scatter.scatter_mean(diff.pow(2).flatten(1), batch, dim=0)  # (B,)
    mse_per_graph = mse_per_graph.flatten()
    return mse_per_graph.mean()


def mac_loss(pred: torch.Tensor,
             target: torch.Tensor,
             batch: torch.Tensor,
             num_graphs: int,
             eps: float = 1e-12) -> torch.Tensor:
    """1 - MAC 损失，衡量振型之间的模态置信度准则（向量化版本）。"""
    # 逐图统计（向量化 scatter）
    sum_pt = torch_scatter.scatter_sum(pred * target, batch, dim=0)      # (B, D)
    sum_p2 = torch_scatter.scatter_sum(pred.pow(2), batch, dim=0)        # (B, D)
    sum_t2 = torch_scatter.scatter_sum(target.pow(2), batch, dim=0)      # (B, D)
    numerator = sum_pt.pow(2)
    denominator = sum_p2 * sum_t2 + eps
    mac_per_mode = 1.0 - numerator / denominator
    return mac_per_mode.mean()


def modal_loss(outputs: Dict[str, torch.Tensor],
               batch_data: Dict[str, torch.Tensor],
               weights: Dict[str, float] | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算模态参数损失。

    目标:
        modal_omega      (B, K), rad/s
        modal_zeta       (B, K)
        modal_phi_xyz    (total_N, K, 3)
    """
    weights = weights or {}
    batch = batch_data['batch']
    num_graphs = int(batch_data.get('num_graphs', int(batch.max().item()) + 1))

    omega_pred = outputs['modal_omega']
    zeta_pred = outputs['modal_zeta']
    phi_pred = outputs['modal_phi_xyz']
    omega_target = batch_data['modal_omega']
    zeta_target = batch_data['modal_zeta']
    phi_target = batch_data['modal_phi_xyz']

    # 固有频率损失：使用 relative_l1（非 log_l1），更数值稳定
    loss_omega = relative_l1(omega_pred, omega_target, eps=1e-5)

    # 阻尼比损失：混合 log_l1 + relative_l1，兼顾量级与数值稳定性
    loss_zeta = (
        0.5 * log_l1(zeta_pred, zeta_target, eps=1e-5) +
        0.5 * relative_l1(zeta_pred, zeta_target, eps=1e-5)
    )

    # 逐模态频率和阻尼误差（用于训练日志）
    omega_per_mode = torch.mean(
        torch.abs(omega_pred - omega_target) / (torch.abs(omega_target) + 1e-5), dim=0)  # (K,)
    zeta_per_mode = torch.mean(
        torch.abs(zeta_pred - zeta_target) / (torch.abs(zeta_target) + 1e-5), dim=0)  # (K,)

    # 方向感知振型损失：取 response_dir_index 对应的分量
    resp_idx = int(batch_data.get('response_dir_index',
                                   torch.tensor(2)).flatten()[0].item())
    phi_resp_pred = phi_pred[..., resp_idx]   # (total_N, K)
    phi_resp_target = phi_target[..., resp_idx]
    loss_phi_resp = sign_invariant_mse(
        phi_resp_pred, phi_resp_target,
        batch, num_graphs, normalize=True
    )

    # 逐模态 φ 误差（向量化）
    p_rms = torch.sqrt(torch_scatter.scatter_mean(phi_resp_pred.pow(2), batch, dim=0) + 1e-8)  # (B, K)
    t_rms = torch.sqrt(torch_scatter.scatter_mean(phi_resp_target.pow(2), batch, dim=0) + 1e-8)
    p_n = phi_resp_pred / p_rms[batch].clamp_min(1e-8)
    t_n = phi_resp_target / t_rms[batch].clamp_min(1e-8)
    dot = torch_scatter.scatter_sum(p_n * t_n, batch, dim=0)  # (B, K)
    sign = torch.where(dot >= 0, torch.ones_like(dot, device=dot.device),
                       -torch.ones_like(dot, device=dot.device))
    phi_per_mode = torch_scatter.scatter_mean(
        (p_n - t_n * sign[batch]).pow(2), batch, dim=0)  # (B, K)
    phi_per_mode = phi_per_mode.mean(dim=0)  # (K,) 跨图平均

    # 完整 XYZ 三向振型损失（展平为每节点 3*K 维向量后做符号对齐）
    loss_phi_xyz = sign_invariant_mse(
        phi_pred.reshape(phi_pred.shape[0], -1),
        phi_target.reshape(phi_target.shape[0], -1),
        batch, num_graphs, normalize=True
    )

    # MAC 损失：使用响应方向振型
    loss_mac = mac_loss(
        phi_pred[..., resp_idx], phi_target[..., resp_idx],
        batch, num_graphs
    )

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

    # 幅值 clamp 使用 1e-8（非 1e-12），防止梯度爆炸
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
