"""
losses.py — 模态参数损失 + 排序对齐 + db/CDF FRF损失。
CNN-ModalV2: 物理单位直接计算，OmegaHead 单调保证无需 sort。
"""
import torch
import torch.nn.functional as F


def _mac_per_graph(phi_pred, phi_target):
    """三维单图 MAC: phi [N,K,3] → mac [K]。沿节点(dim=0)和坐标轴(dim=2)同时求内积。"""
    num = torch.sum(phi_pred * phi_target, dim=(0, 2)) ** 2
    den = torch.sum(phi_pred ** 2, dim=(0, 2)) * torch.sum(phi_target ** 2, dim=(0, 2)) + 1e-8
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
        phi_pred:         [N,K,3] 或 [B,N,K,3] 预测三维振型
        phi_target:       [N,K,3] 或 [B,N,K,3] 真实三维振型
        batch_idx:        [N,]   批次索引 (phi 为 [N,K,3] 时)
    """

    # ====================================================
    # 1. 频率损失: Hz-space smooth_l1 + peak-sensitive
    # ====================================================
    f_pred_hz = omega_phys_pred / (2.0 * torch.pi)
    f_true_hz = omega_phys_target / (2.0 * torch.pi)

    loss_freq_hz = F.smooth_l1_loss(f_pred_hz, f_true_hz)

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
    # 3. 三维振型损失: MSE(归一化) + MAC + Std Ratio
    #    phi 为 [N,K,3]，沿 N 和 XYZ 两维共同计算
    # ====================================================
    if phi_pred.dim() == 4:  # [B, N, K, 3]
        phi_pred = phi_pred.view(-1, phi_pred.shape[-2], phi_pred.shape[-1])
        phi_target = phi_target.view(-1, phi_target.shape[-2], phi_target.shape[-1])

    # 符号对齐: 三维内积 [N,K,3] → 沿 (0,2) 求和
    dot = torch.sum(phi_pred * phi_target, dim=(0, 2), keepdim=True)   # [1, K, 1]
    sign = torch.sign(dot + 1e-8)
    aligned_target = phi_target * sign

    # 三维总 Std: [K, N*3] → std(dim=1) → [K]
    p_std = torch.std(phi_pred.transpose(0, 1).reshape(phi_pred.shape[1], -1), dim=1) + 1e-8
    t_std = torch.std(phi_target.transpose(0, 1).reshape(phi_target.shape[1], -1), dim=1) + 1e-8
    p_std_view = p_std.view(1, -1, 1)
    t_std_view = t_std.view(1, -1, 1)

    phi_pred_norm = phi_pred / p_std_view
    phi_target_norm = aligned_target / t_std_view
    raw_phi_mse = F.mse_loss(phi_pred_norm, phi_target_norm)

    # MAC: 三维，尺度无关
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

    # Std Ratio (对数域)
    log_p_std = torch.log(p_std)
    log_t_std = torch.log(t_std + 1e-8)
    loss_std = F.smooth_l1_loss(log_p_std, log_t_std)

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
