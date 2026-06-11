"""
evaluate.py — 训练后模态参数评估 + FRF重建保存。

核心目标：
1. 评估前三阶固有频率 f/ω
2. 评估前三阶阻尼 ζ
3. 评估全节点前三阶振型 φ，包括 MAC / NRMSE / std ratio
4. 使用预测的 ω, ζ, φ 通过 PhysicsDecoder 重建 FRF
5. 保存 final_results.npz，供 对比图.py 使用

用法:
    F:/pytorch_cuda12/python.exe sample/evaluate.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from models import build_geometric_model
from data.dataset import GeometricHDF5Dataset


CONFIG = {
    'freq_min': 1.0,
    'freq_max': 5000.0,
}

MODEL_CFG = {
    'encoder_kwargs': {
        'in_ch': 6,
        'hidden': 768,
        'n_modes': 3,
        'amp_scale': 500000.0,
        'freq_min': 1.0,
        'freq_max': 5000.0,
    },
    'decoder_kwargs': {},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data')
out_dir = os.path.join(os.path.dirname(__file__), 'output')
ckpt_path = os.path.join(out_dir, 'checkpoint_best')


def to_obj(arr_list):
    """可变长度数组保存为 object array。"""
    out = np.empty(len(arr_list), dtype=object)
    for i, a in enumerate(arr_list):
        out[i] = a
    return out


def sign_align_phi(pred_phi, true_phi, eps=1e-8):
    """
    对每阶振型做符号对齐。
    pred_phi/true_phi: [N, K]
    return:
        pred_phi_aligned [N,K]
        signs [K]
    """
    signs = []
    aligned = pred_phi.clone()

    for k in range(pred_phi.shape[1]):
        dot = torch.sum(pred_phi[:, k] * true_phi[:, k])
        sign = torch.sign(dot + eps)
        aligned[:, k] = pred_phi[:, k] * sign
        signs.append(sign)

    signs = torch.stack(signs)
    return aligned, signs


def phi_metrics(pred_phi, true_phi, eps=1e-8):
    """
    pred_phi/true_phi: [N,K]，已做符号对齐或未对齐均可，内部会再次对齐。
    return:
        mac [K]
        nrmse [K]
        std_ratio [K]
        pred_phi_aligned [N,K]
    """
    pred_phi_aligned, signs = sign_align_phi(pred_phi, true_phi, eps=eps)

    macs = []
    nrmse = []
    std_ratio = []

    for k in range(true_phi.shape[1]):
        p = pred_phi_aligned[:, k]
        t = true_phi[:, k]

        mac = (torch.sum(p * t) ** 2) / (
            torch.sum(p ** 2) * torch.sum(t ** 2) + eps
        )

        rmse = torch.sqrt(torch.mean((p - t) ** 2))
        t_std = torch.std(t) + eps
        p_std = torch.std(p) + eps
        # 防极端样本: t_std 太小时用 p_std 替代
        safe_t_std = torch.max(t_std, p_std * 0.1)

        macs.append(mac)
        nrmse.append(rmse / safe_t_std)
        std_ratio.append(torch.clamp(p_std / safe_t_std, 0.1, 10.0))

    return (
        torch.stack(macs),
        torch.stack(nrmse),
        torch.stack(std_ratio),
        pred_phi_aligned,
        signs,
    )


def compute_peak_metrics(freqs_hz, pred_amp, true_amp, true_freq_hz, true_zeta):
    """
    基于 FRF 幅值 envelope 估计每阶峰值偏移。
    freqs_hz: [F]
    pred_amp/true_amp: [N,F]
    true_freq_hz: [K]
    true_zeta: [K]

    返回:
      peak_shift_hz [K]
      peak_amp_rel [K]
      pred_peak_freq [K]
      true_peak_freq [K]
    """
    freqs = np.asarray(freqs_hz)
    pred_env = np.mean(pred_amp, axis=0)
    true_env = np.mean(true_amp, axis=0)

    peak_shift = []
    peak_amp_rel = []
    pred_peak_freq = []
    true_peak_freq = []

    for fk, zk in zip(true_freq_hz, true_zeta):
        # 半功率带宽约 2ζf，取 ±3倍带宽，至少 ±10Hz
        bw = max(float(6.0 * zk * fk), 10.0)
        mask = (freqs >= fk - bw) & (freqs <= fk + bw)

        if not np.any(mask):
            idx = int(np.argmin(np.abs(freqs - fk)))
            mask = np.zeros_like(freqs, dtype=bool)
            mask[idx] = True

        local_freqs = freqs[mask]
        local_pred = pred_env[mask]
        local_true = true_env[mask]

        idx_t = int(np.argmax(local_true))
        idx_p = int(np.argmax(local_pred))

        tf = float(local_freqs[idx_t])
        pf = float(local_freqs[idx_p])

        ta = float(local_true[idx_t])
        pa = float(local_pred[idx_p])

        peak_shift.append(abs(pf - tf))
        peak_amp_rel.append(abs(pa - ta) / (abs(ta) + 1e-12))
        pred_peak_freq.append(pf)
        true_peak_freq.append(tf)

    return (
        np.asarray(peak_shift, dtype=np.float32),
        np.asarray(peak_amp_rel, dtype=np.float32),
        np.asarray(pred_peak_freq, dtype=np.float32),
        np.asarray(true_peak_freq, dtype=np.float32),
    )


def main():
    print("=" * 80)
    print("模型评估：模态参数 + FRF 重建")
    print("=" * 80)

    testset = GeometricHDF5Dataset(
        ['test.h5'],
        CONFIG,
        data_dir=data_dir,
        normalization=True,
        test=True,
    )

    testset_raw = GeometricHDF5Dataset(
        ['test.h5'],
        CONFIG,
        data_dir=data_dir,
        normalization=False,
        test=True,
    )

    print(f"测试集: {len(testset)} 样本")
    print(f"Checkpoint: {ckpt_path}")

    model = build_geometric_model(
        MODEL_CFG['encoder_kwargs'],
        MODEL_CFG['decoder_kwargs'],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Checkpoint epoch={ckpt.get('epoch', 'NA')}, loss={ckpt.get('loss', -1):.6f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    all_points = []
    all_freqs = []

    all_pred_amp = []
    all_true_amp = []
    all_pred_re = []
    all_true_re = []
    all_pred_im = []
    all_true_im = []

    all_pred_omega = []
    all_true_omega = []
    all_pred_freq_hz = []
    all_true_freq_hz = []
    all_freq_rel = []

    all_pred_zeta = []
    all_true_zeta = []
    all_zeta_rel = []

    all_pred_phi = []
    all_true_phi = []
    all_phi_mac = []
    all_phi_nrmse = []
    all_phi_std_ratio = []

    all_peak_shift_hz = []
    all_peak_amp_rel = []
    all_pred_peak_freq = []
    all_true_peak_freq = []

    for idx in range(len(testset)):
        sn = testset[idx]
        sr = testset_raw[idx]

        img = sn['image_tensor'].unsqueeze(0).to(device)
        coords = sn['query_coords'].unsqueeze(0).to(device)
        freqs_norm = sn['frequencies'].unsqueeze(0).to(device)
        bt = torch.zeros(coords.shape[1], dtype=torch.long, device=device)
        # 传入 node 信息 (NodePhiRefiner 需要)
        _nx = sn.get('node_xyz'); _nf = sn.get('node_features')
        _nx = _nx.to(device).unsqueeze(0) if _nx is not None else None
        _nf = _nf.to(device).unsqueeze(0) if _nf is not None else None

        true_phi = sn['modal_phi'].to(device)          # [N,K]
        true_zeta = sn['modal_zeta'].to(device)        # [K]
        true_omega = sn['modal_omega_phys'].to(device) # [K]
        true_freq_hz = true_omega / (2.0 * torch.pi)

        phi_exc_true = sn.get('modal_phi_exc')
        phi_exc_true = phi_exc_true.to(device) if phi_exc_true is not None else None

        with torch.no_grad():
            # OmegaHead 保证单调 → 不需要 sort; omega_phys 已是 rad/s
            _, omega_phys_pred, log_zeta_pred, zeta_pred, phi_pred = model(
                img, coords, None, None, bt,
                node_xyz=_nx.squeeze(0) if _nx is not None else None,
                node_features=_nf.squeeze(0) if _nf is not None else None)

            omega_phys_pred = omega_phys_pred.squeeze(0)  # [K]
            zeta_pred = zeta_pred.squeeze(0)              # [K]
            phi_pred = phi_pred                           # [N,K]

            # 与真实振型符号对齐
            mac, nrmse, std_ratio, phi_aligned, signs = phi_metrics(phi_pred, true_phi)

            # 频率 (已经是 rad/s)
            freq_hz_pred = omega_phys_pred / (2.0 * torch.pi)

            # phi_aligned = phi_pred * sign → phi_exc 也要同符号翻转
            if phi_exc_true is not None:
                phi_exc_for_frf = (phi_exc_true * signs).unsqueeze(0)  # [1,K]
            else:
                phi_exc_for_frf = None

            frf_pred = model.physics(
                phi_aligned,
                omega_phys_pred.unsqueeze(0),
                zeta_pred.unsqueeze(0),
                freqs_norm,
                phi_exc_for_frf,
                batch_idx=bt,
                alpha=1.0,
            )

        p = frf_pred.detach().cpu()          # [N,F,2]
        t = sr['point_frf'].detach().cpu()   # [N,F,2]

        pred_amp = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-12).numpy()
        true_amp = torch.sqrt(t[..., 0] ** 2 + t[..., 1] ** 2 + 1e-12).numpy()

        pred_re = p[..., 0].numpy()
        pred_im = p[..., 1].numpy()
        true_re = t[..., 0].numpy()
        true_im = t[..., 1].numpy()

        true_omega_cpu = true_omega.detach().cpu()
        pred_omega_cpu = omega_phys_pred.detach().cpu()

        true_freq_cpu = true_freq_hz.detach().cpu()
        pred_freq_cpu = freq_hz_pred.detach().cpu()

        true_zeta_cpu = true_zeta.detach().cpu()
        pred_zeta_cpu = zeta_pred.detach().cpu()

        freq_rel = torch.abs(pred_freq_cpu - true_freq_cpu) / (true_freq_cpu + 1e-8)
        zeta_rel = torch.abs(pred_zeta_cpu - true_zeta_cpu) / (true_zeta_cpu + 1e-8)

        freqs_phys = sr['frequencies'].numpy()
        peak_shift, peak_amp_rel, pred_peak_f, true_peak_f = compute_peak_metrics(
            freqs_phys,
            pred_amp,
            true_amp,
            true_freq_cpu.numpy(),
            true_zeta_cpu.numpy(),
        )

        all_points.append(sr['points'].numpy())
        all_freqs.append(freqs_phys)

        all_pred_amp.append(pred_amp)
        all_true_amp.append(true_amp)
        all_pred_re.append(pred_re)
        all_true_re.append(true_re)
        all_pred_im.append(pred_im)
        all_true_im.append(true_im)

        all_pred_omega.append(pred_omega_cpu.numpy())
        all_true_omega.append(true_omega_cpu.numpy())
        all_pred_freq_hz.append(pred_freq_cpu.numpy())
        all_true_freq_hz.append(true_freq_cpu.numpy())
        all_freq_rel.append(freq_rel.numpy())

        all_pred_zeta.append(pred_zeta_cpu.numpy())
        all_true_zeta.append(true_zeta_cpu.numpy())
        all_zeta_rel.append(zeta_rel.numpy())

        all_pred_phi.append(phi_aligned.detach().cpu().numpy())
        all_true_phi.append(true_phi.detach().cpu().numpy())
        all_phi_mac.append(mac.detach().cpu().numpy())
        all_phi_nrmse.append(nrmse.detach().cpu().numpy())
        all_phi_std_ratio.append(std_ratio.detach().cpu().numpy())

        all_peak_shift_hz.append(peak_shift)
        all_peak_amp_rel.append(peak_amp_rel)
        all_pred_peak_freq.append(pred_peak_f)
        all_true_peak_freq.append(true_peak_f)

        if idx % 10 == 0:
            print(f"[{idx+1:03d}/{len(testset)}] "
                  f"freq_rel%={freq_rel.mean().item()*100:.3f}, "
                  f"zeta_rel%={zeta_rel.mean().item()*100:.2f}, "
                  f"MAC={mac.mean().item():.4f}")

    # 汇总指标
    freq_rel_all = np.stack(all_freq_rel, axis=0)
    zeta_rel_all = np.stack(all_zeta_rel, axis=0)
    phi_mac_all = np.stack(all_phi_mac, axis=0)
    phi_nrmse_all = np.stack(all_phi_nrmse, axis=0)
    phi_std_ratio_all = np.stack(all_phi_std_ratio, axis=0)
    peak_shift_all = np.stack(all_peak_shift_hz, axis=0)
    peak_amp_rel_all = np.stack(all_peak_amp_rel, axis=0)

    amp_mse_vals = [
        np.mean((all_pred_amp[i] - all_true_amp[i]) ** 2)
        for i in range(len(all_pred_amp))
    ]
    amp_l1_vals = [
        np.mean(np.abs(all_pred_amp[i] - all_true_amp[i]))
        for i in range(len(all_pred_amp))
    ]

    print("\n" + "=" * 80)
    print("测试集汇总")
    print("=" * 80)
    print(f"FRF amplitude MSE mean = {np.mean(amp_mse_vals):.6e}")
    print(f"FRF amplitude L1  mean = {np.mean(amp_l1_vals):.6e}")

    print("\n频率相对误差 per mode (%)")
    print(np.mean(freq_rel_all, axis=0) * 100.0)
    print(f"频率相对误差 mean (%) = {np.mean(freq_rel_all) * 100.0:.4f}")
    print(f"频率相对误差 max  (%) = {np.max(freq_rel_all) * 100.0:.4f}")

    print("\n阻尼相对误差 per mode (%)")
    print(np.mean(zeta_rel_all, axis=0) * 100.0)
    print(f"阻尼相对误差 mean (%) = {np.mean(zeta_rel_all) * 100.0:.4f}")

    print("\n振型 MAC per mode")
    print(np.mean(phi_mac_all, axis=0))
    print(f"振型 MAC mean = {np.mean(phi_mac_all):.6f}")

    print("\n振型 NRMSE per mode")
    print(np.mean(phi_nrmse_all, axis=0))

    print("\n振型 std_ratio pred/true per mode")
    print(np.mean(phi_std_ratio_all, axis=0))

    print("\nFRF 峰值频率偏移 per mode (Hz)")
    print(np.mean(peak_shift_all, axis=0))
    print(f"FRF 峰值频率偏移 mean (Hz) = {np.mean(peak_shift_all):.4f}")

    print("\nFRF 峰值幅值相对误差 per mode (%)")
    print(np.mean(peak_amp_rel_all, axis=0) * 100.0)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'final_results.npz')

    np.savez(
        save_path,

        points=to_obj(all_points),
        frequencies=to_obj(all_freqs),

        predicted_frf=to_obj(all_pred_amp),
        target_frf=to_obj(all_true_amp),
        predicted_re=to_obj(all_pred_re),
        target_re=to_obj(all_true_re),
        predicted_im=to_obj(all_pred_im),
        target_im=to_obj(all_true_im),

        pred_omega=to_obj(all_pred_omega),
        true_omega=to_obj(all_true_omega),
        pred_freq_hz=to_obj(all_pred_freq_hz),
        true_freq_hz=to_obj(all_true_freq_hz),
        freq_rel=to_obj(all_freq_rel),

        pred_zeta=to_obj(all_pred_zeta),
        true_zeta=to_obj(all_true_zeta),
        zeta_rel=to_obj(all_zeta_rel),

        pred_phi=to_obj(all_pred_phi),
        true_phi=to_obj(all_true_phi),
        phi_mac=to_obj(all_phi_mac),
        phi_nrmse=to_obj(all_phi_nrmse),
        phi_std_ratio=to_obj(all_phi_std_ratio),

        peak_shift_hz=to_obj(all_peak_shift_hz),
        peak_amp_rel=to_obj(all_peak_amp_rel),
        pred_peak_freq=to_obj(all_pred_peak_freq),
        true_peak_freq=to_obj(all_true_peak_freq),
    )

    print(f"\n数据保存: {save_path}")
    print("评估完成!")


if __name__ == '__main__':
    main()
