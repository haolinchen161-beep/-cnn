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

def phi_1d_shape_scale_loss(pred: torch.Tensor,
                            target: torch.Tensor,
                            node_counts: list,
                            eps: float = 1e-8):
    """目标方向 1D 振型的 shape + scale 解耦损失。

    pred/target:
        (total_N, K)，例如 Z/Z 训练时的 phi_z。

    返回:
        shape_loss:
            RMS 归一化后的形状误差，解决弱 Z 分量被忽略的问题。

        scale_loss:
            log-RMS 尺度误差，解决形状对了但幅值错的问题。

        per_mode_shape:
            每阶 shape 误差，用于日志 phi_k0/1/2。

        per_mode_scale:
            每阶 scale 误差，用于额外日志。
    """
    shape_parts = []
    scale_parts = []
    per_mode_shape_parts = []
    per_mode_scale_parts = []

    ptr = 0
    for c in node_counts:
        p = pred[ptr:ptr + c]      # (N_b, K)
        t = target[ptr:ptr + c]    # (N_b, K)

        p_rms = torch.sqrt(p.pow(2).mean(dim=0) + eps)  # (K,)
        t_rms = torch.sqrt(t.pow(2).mean(dim=0) + eps)  # (K,)

        # ---------- shape：归一化后比较空间形状 ----------
        p_n = p / p_rms.clamp_min(eps)
        t_n = t / t_rms.clamp_min(eps)

        dot = (p_n * t_n).sum(dim=0)
        sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))

        per_mode_shape = (p_n - t_n * sign).pow(2).mean(dim=0)  # (K,)

        # ---------- scale：log-RMS，避免大幅值再次掩蔽小幅值 ----------
        per_mode_scale = torch.abs(
            torch.log(p_rms.clamp_min(eps)) -
            torch.log(t_rms.clamp_min(eps))
        )  # (K,)

        shape_parts.append(per_mode_shape.mean())
        scale_parts.append(per_mode_scale.mean())
        per_mode_shape_parts.append(per_mode_shape)
        per_mode_scale_parts.append(per_mode_scale)

        ptr += c

    if not shape_parts:
        zero = pred.new_tensor(0.0)
        return zero, zero, zero, zero

    shape_loss = torch.stack(shape_parts).mean()
    scale_loss = torch.stack(scale_parts).mean()
    per_mode_shape = torch.stack(per_mode_shape_parts).mean(dim=0)
    per_mode_scale = torch.stack(per_mode_scale_parts).mean(dim=0)

    return shape_loss, scale_loss, per_mode_shape, per_mode_scale


def phi_participation_loss(phi_resp_pred: torch.Tensor,
                           phi_resp_target: torch.Tensor,
                           phi_force_pred: torch.Tensor,
                           phi_force_target: torch.Tensor,
                           excitation_index: torch.Tensor,
                           node_counts: list,
                           eps: float = 1e-8):
    """FRF 分子参与因子损失。

    对 H_ab:
        participation(node, k) = phi_response_a(node, k) * phi_force_b(exc, k)

    这个量直接对应模态叠加 FRF 的分子项，比单独监督 phi_resp 更接近最终目标。
    """
    loss_parts = []
    per_mode_parts = []

    ptr = 0
    for b, c in enumerate(node_counts):
        p_resp = phi_resp_pred[ptr:ptr + c]      # (N_b, K)
        t_resp = phi_resp_target[ptr:ptr + c]    # (N_b, K)

        # excitation_index 在 collate 后是全局节点索引，不是局部索引
        exc_global = excitation_index[b].long()

        p_exc = phi_force_pred[exc_global]       # (K,)
        t_exc = phi_force_target[exc_global]     # (K,)

        part_pred = p_resp * p_exc.unsqueeze(0)      # (N_b, K)
        part_target = t_resp * t_exc.unsqueeze(0)    # (N_b, K)

        # 每阶按目标 participation RMS 归一化，避免强模态再次掩蔽弱模态
        part_scale = torch.sqrt(part_target.pow(2).mean(dim=0) + eps)

        # 防爆：如果某阶 participation 极小，用当前样本各模态平均尺度的 5% 托底
        # 避免除以接近 0 的数导致 loss / 梯度突然爆炸
        scale_floor = (0.05 * part_scale.detach().mean()).clamp_min(eps)
        part_scale = part_scale.clamp_min(scale_floor)

        per_mode = ((part_pred - part_target) / part_scale).pow(2).mean(dim=0)

        loss_parts.append(per_mode.mean())
        per_mode_parts.append(per_mode)

        ptr += c

    if not loss_parts:
        zero = phi_resp_pred.new_tensor(0.0)
        return zero, zero

    loss = torch.stack(loss_parts).mean()
    per_mode = torch.stack(per_mode_parts).mean(dim=0)

    return loss, per_mode

def modal_loss(outputs: Dict[str, torch.Tensor],
               batch_data: Dict[str, torch.Tensor],
               weights: Dict[str, float] | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算模态参数损失。

    新版逻辑:
    1. omega / zeta 保持原监督。
    2. 响应方向 phi_resp 拆成 shape + scale。
    3. 增加 participation loss，直接监督 FRF 分子项。
    4. phi_xyz 拆成 normalized shape + raw energy。
    5. 保留 loss_phi_resp / loss_phi_xyz 旧 key，兼容 trainer 日志。
    """
    weights = weights or {}
    batch = batch_data['batch']
    node_counts = batch_data.get('node_counts', batch.bincount().tolist())

    omega_pred = outputs['modal_omega']
    zeta_pred = outputs['modal_zeta']
    phi_pred = outputs['modal_phi_xyz']

    phi_target = batch_data['modal_phi_xyz']
    omega_target = batch_data['modal_omega']
    zeta_target = batch_data['modal_zeta']

    # ------------------------------------------------------------
    # 1. 频率损失
    # ------------------------------------------------------------
    loss_omega = relative_l1(omega_pred, omega_target, eps=1e-5)

    # ------------------------------------------------------------
    # 2. 阻尼损失
    # ------------------------------------------------------------
    loss_zeta = (
        0.5 * log_l1(zeta_pred, zeta_target, eps=1e-5) +
        0.5 * relative_l1(zeta_pred, zeta_target, eps=1e-5)
    )

    omega_per_mode = torch.mean(
        torch.abs(omega_pred - omega_target) / (torch.abs(omega_target) + 1e-5),
        dim=0,
    )
    zeta_per_mode = torch.mean(
        torch.abs(zeta_pred - zeta_target) / (torch.abs(zeta_target) + 1e-5),
        dim=0,
    )

    # ------------------------------------------------------------
    # 3. 提取响应方向和激励方向
    # ------------------------------------------------------------
    resp_idx = int(batch_data.get(
        'response_dir_index',
        torch.tensor(2, device=phi_pred.device)
    ).flatten()[0].item())

    force_idx = int(batch_data.get(
        'force_dir_index',
        torch.tensor(resp_idx, device=phi_pred.device)
    ).flatten()[0].item())

    phi_resp_pred = phi_pred[..., resp_idx]       # (total_N, K)
    phi_resp_target = phi_target[..., resp_idx]   # (total_N, K)

    phi_force_pred = phi_pred[..., force_idx]     # (total_N, K)
    phi_force_target = phi_target[..., force_idx] # (total_N, K)

    # ------------------------------------------------------------
    # 4. 响应方向 shape + scale
    # ------------------------------------------------------------
    loss_phi_resp_shape, loss_phi_resp_scale, phi_per_mode, z_scale_per_mode = \
        phi_1d_shape_scale_loss(
            phi_resp_pred,
            phi_resp_target,
            node_counts,
        )

    # ------------------------------------------------------------
    # 5. FRF 分子 participation loss
    # ------------------------------------------------------------
    if 'excitation_index' in batch_data:
        loss_participation, part_per_mode = phi_participation_loss(
            phi_resp_pred=phi_resp_pred,
            phi_resp_target=phi_resp_target,
            phi_force_pred=phi_force_pred,
            phi_force_target=phi_force_target,
            excitation_index=batch_data['excitation_index'],
            node_counts=node_counts,
        )
    else:
        loss_participation = phi_pred.new_tensor(0.0)
        part_per_mode = phi_pred.new_zeros(phi_pred.shape[1])

    # ------------------------------------------------------------
    # 6. 完整 XYZ 振型：shape + energy
    # ------------------------------------------------------------
    loss_phi_xyz_shape = sign_invariant_mse(
        phi_pred,
        phi_target,
        node_counts,
        normalize=True,
    )

    loss_phi_xyz_energy = sign_invariant_mse(
        phi_pred,
        phi_target,
        node_counts,
        normalize=False,
    )

    # ------------------------------------------------------------
    # 7. MAC，保留辅助，但权重建议很小
    # ------------------------------------------------------------
    loss_mac = mac_loss(phi_resp_pred, phi_resp_target, node_counts)

    # ------------------------------------------------------------
    # 8. 组合 legacy loss，兼容 trainer 原有日志
    # ------------------------------------------------------------
    resp_scale_ratio = weights.get('phi_resp_scale_ratio', 0.3)
    participation_ratio = weights.get('participation_ratio', 0.3)
    xyz_energy_ratio = weights.get('phi_xyz_energy_ratio', 0.1)

    loss_phi_resp = (
        loss_phi_resp_shape +
        resp_scale_ratio * loss_phi_resp_scale +
        participation_ratio * loss_participation
    )

    loss_phi_xyz = (
        loss_phi_xyz_shape +
        xyz_energy_ratio * loss_phi_xyz_energy
    )

    # ------------------------------------------------------------
    # 9. 总损失
    # ------------------------------------------------------------
    total = (
        weights.get('omega', 10.0) * loss_omega +
        weights.get('zeta', 1.0) * loss_zeta +
        weights.get('phi_resp', 1.0) * loss_phi_resp +
        weights.get('phi_xyz', 0.5) * loss_phi_xyz +
        weights.get('mac', 0.05) * loss_mac
    )

    # ------------------------------------------------------------
    # 10. 日志
    # ------------------------------------------------------------
    logs = {
        'loss_omega': loss_omega.detach(),
        'loss_zeta': loss_zeta.detach(),

        # 保留旧 key，trainer.py 依赖这些字段
        'loss_phi_resp': loss_phi_resp.detach(),
        'loss_phi_xyz': loss_phi_xyz.detach(),
        'loss_mac': loss_mac.detach(),

        # 新增细分日志
        'loss_phi_resp_shape': loss_phi_resp_shape.detach(),
        'loss_phi_resp_scale': loss_phi_resp_scale.detach(),
        'loss_phi_participation': loss_participation.detach(),
        'loss_phi_xyz_shape': loss_phi_xyz_shape.detach(),
        'loss_phi_xyz_energy': loss_phi_xyz_energy.detach(),

        **{f'omega_k{k}': omega_per_mode[k].detach() for k in range(omega_per_mode.shape[0])},
        **{f'zeta_k{k}': zeta_per_mode[k].detach() for k in range(zeta_per_mode.shape[0])},
        **{f'phi_k{k}': phi_per_mode[k].detach() for k in range(phi_per_mode.shape[0])},
        **{f'z_scale_k{k}': z_scale_per_mode[k].detach() for k in range(z_scale_per_mode.shape[0])},
        **{f'part_k{k}': part_per_mode[k].detach() for k in range(part_per_mode.shape[0])},
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
