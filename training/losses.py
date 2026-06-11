"""
losses.py — 模态参数损失 + 排序对齐 + db/CDF FRF损失。
CNN-ModalV2: 物理单位直接计算，OmegaHead 单调保证无需 sort。
"""
import torch
import torch.nn.functional as F


def _mac_per_graph(phi_pred, phi_target):
    """单图 MAC: phi [N,K] → mac [K]。目标: 最小化 1-MAC。"""
    num = torch.sum(phi_pred * phi_target, dim=0) ** 2
    den = torch.sum(phi_pred ** 2, dim=0) * torch.sum(phi_target ** 2, dim=0) + 1e-8
    return num / den


def modal_loss(omega_phys_pred, omega_phys_target,
               log_zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               omega_weight=200.0, zeta_weight=100.0, phi_weight=1.0):
    """
    CNN-ModalV2 模态损失。

    Args:
        omega_phys_pred:  [B,K] 物理 rad/s (OmegaHead 已保证单调)
        omega_phys_target:[B,K] 物理 rad/s
        log_zeta_pred:    [B,K] 对数阻尼
        zeta_target:      [B,K] 物理阻尼比
        phi_pred:         [N,K] 或 [B,N,K] 预测振型
        phi_target:       [N,K] 或 [B,N,K] 真实振型
        batch_idx:        [N,]   批次索引 (phi 为 [N,K] 时)
    """

    # ====================================================
    # 1. 频率损失: Hz-space smooth_l1 + peak-sensitive
    #    smooth_l1 在 Hz 空间有恒定梯度 (|error|>1Hz → grad=1)
    # ====================================================
    f_pred_hz = omega_phys_pred / (2.0 * torch.pi)
    f_true_hz = omega_phys_target / (2.0 * torch.pi)

    # Hz 空间 smooth_l1 — 永不衰减的恒定梯度引擎
    loss_freq_hz = F.smooth_l1_loss(f_pred_hz, f_true_hz)

    # 峰值敏感: 阻尼越小惩罚越重
    rel = torch.abs(omega_phys_pred - omega_phys_target) / (omega_phys_target + 1e-8)
    peak_sensitive = rel / (zeta_target + 1e-8)
    peak_sensitive = torch.clamp(peak_sensitive, max=100.0)

    loss_omega = (loss_freq_hz + 0.1 * peak_sensitive.mean()) * omega_weight

    # ====================================================
    # 2. 对数域阻尼损失
    # ====================================================
    log_zeta_target = torch.log(zeta_target + 1e-8)
    loss_zeta = F.smooth_l1_loss(log_zeta_pred, log_zeta_target) * zeta_weight

    # ====================================================
    # 3. 振型损失: MSE(归一化) + MAC + Std Ratio
    #    phi 真实值极小 → 先 std 归一化再算 MSE，否则梯度近乎零
    # ====================================================
    if phi_pred.dim() == 3:
        phi_pred = phi_pred.view(-1, phi_pred.shape[-1])
        phi_target = phi_target.view(-1, phi_target.shape[-1])

    # 符号对齐
    dot = torch.sum(phi_pred * phi_target, dim=0, keepdim=True)
    sign = torch.sign(dot + 1e-8)
    aligned_target = phi_target * sign

    # 逐模态归一化到 unit std，再算 MSE (解决 phi 真实值过小问题)
    p_std = torch.std(phi_pred, dim=0) + 1e-8
    t_std = torch.std(phi_target, dim=0) + 1e-8
    phi_pred_norm = phi_pred / p_std
    phi_target_norm = aligned_target / t_std
    raw_phi_mse = F.mse_loss(phi_pred_norm, phi_target_norm)

    # MAC: 尺度无关，直接算 (按图)
    if batch_idx is not None:
        n_graphs = int(batch_idx.max().item()) + 1
        mac_loss_total = 0.0
        for i in range(n_graphs):
            mask = (batch_idx == i)
            mac = _mac_per_graph(phi_pred[mask], aligned_target[mask])
            mac_loss_total += (1.0 - mac).mean()
        loss_mac = mac_loss_total / n_graphs
    else:
        mac = _mac_per_graph(phi_pred, aligned_target)
        loss_mac = (1.0 - mac).mean()

    # Std Ratio: 预测/真实的幅值比例，clip 防极端样本 (某些 Mode2/3 std≈0.01)
    loss_std = torch.mean(torch.clamp(torch.abs(p_std / (t_std + 1e-8) - 1.0), max=5.0))

    # 归一化后 MSE 在 0~2 之间 (unit std 向量差), MAC 在 0~1
    # MAC (用于日志): 逐模态 [K]
    if batch_idx is not None:
        mac_list = []
        for i in range(n_graphs):
            mask = (batch_idx == i)
            mac_list.append(_mac_per_graph(phi_pred[mask], aligned_target[mask]))
        mac_per_mode = torch.stack(mac_list, dim=0).mean(dim=0)  # [K]
    else:
        mac_per_mode = _mac_per_graph(phi_pred, aligned_target)   # [K]

    loss_phi = (10.0 * raw_phi_mse + 40.0 * loss_mac + 20.0 * loss_std) * phi_weight

    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, loss_phi, mac_per_mode.detach()


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
