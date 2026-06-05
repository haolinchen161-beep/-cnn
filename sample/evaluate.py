"""
evaluate.py — 训练后评估+可视化。
加载检查点 → 预测模态参数 → 物理重建FRF → 对比+保存。
支持可变节点数和可变频率点数 (ANSYS per-sample-group 格式)。

用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/evaluate.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from models import build_geometric_model
from data.dataset import GeometricHDF5Dataset

CONFIG = {'freq_min': 1.0, 'freq_max': 5000.0}
MODEL_CFG = {
    'encoder_kwargs': {
        'in_ch': 6, 'hidden': 512, 'n_modes': 3,
        'amp_scale': 500000.0, 'freq_min': 1.0, 'freq_max': 5000.0,
    },
    'decoder_kwargs': {},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
ckpt_path = os.path.join(out_dir, "checkpoint_best")


def main():
    print("=" * 60)
    print("模型评估 + 可视化 (模态参数预测)")
    print("=" * 60)

    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir,
                                   normalization=True, test=True)
    testset_raw = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir,
                                       normalization=False, test=True)
    print(f"测试集: {len(testset)} 样本")

    model = build_geometric_model(MODEL_CFG['encoder_kwargs'],
                                  MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    all_preds, all_targets, all_freqs = [], [], []
    all_preds_re, all_preds_im = [], []
    all_targets_re, all_targets_im = [], []
    all_points_list = []
    all_pred_omega, all_true_omega = [], []
    omega_errs, zeta_errs = [], []

    for idx in range(len(testset)):
        sn = testset[idx]; sr = testset_raw[idx]
        img = sn['image_tensor'].unsqueeze(0).to(device)
        coords = sn['query_coords'].unsqueeze(0).to(device)
        bt = torch.zeros(coords.shape[1], dtype=torch.long, device=device)
        phi_exc = sn.get('modal_phi_exc')
        phi_exc_t = phi_exc.unsqueeze(0).to(device) if phi_exc is not None else None
        with torch.no_grad():
            if phi_exc_t is not None:
                _, _, _, phi_scan = model(img, coords, sn['frequencies'].unsqueeze(0).to(device), None, bt)
                mp = sn['modal_phi'].unsqueeze(0).to(device)
                dot = torch.sum(phi_scan * mp, dim=1)
                phi_exc_t = phi_exc_t * torch.sign(dot + 1e-8)
            frf_p, op, zp, pp = model(img, coords, sn['frequencies'].unsqueeze(0).to(device), phi_exc_t, bt)
        frf_p = frf_p.squeeze(0).cpu()
        p = frf_p
        t = sr['point_frf']

        omega_errs.append((op.cpu() * 25000.0 - sn['modal_omega_phys']).abs())
        zeta_errs.append((zp.cpu() - sn['modal_zeta']).abs())

        all_preds.append(torch.sqrt(p[...,0]**2+p[...,1]**2+1e-8).numpy())
        all_targets.append(torch.sqrt(t[...,0]**2+t[...,1]**2+1e-8).numpy())
        all_preds_re.append(p[...,0].numpy()); all_preds_im.append(p[...,1].numpy())
        all_targets_re.append(t[...,0].numpy()); all_targets_im.append(t[...,1].numpy())
        all_freqs.append(sr['frequencies'].numpy())
        all_points_list.append(sr['points'].numpy())
        all_pred_omega.append(op.cpu().numpy() * 25000.0)   # 归一化→物理 rad/s
        all_true_omega.append(sn['modal_omega_phys'].numpy())  # 物理 rad/s

    # 可变大小 → object 数组
    def to_obj(arr_list):
        out = np.empty(len(arr_list), dtype=object)
        for i, a in enumerate(arr_list):
            out[i] = a
        return out

    omega_mae = torch.cat(omega_errs).mean().item()
    zeta_mae = torch.cat(zeta_errs).mean().item()
    # 逐样本MSE (标量均值)
    mse_vals = [np.mean((all_preds[i] - all_targets[i])**2) for i in range(len(all_preds))]
    l1_vals = [np.mean(np.abs(all_preds[i] - all_targets[i])) for i in range(len(all_preds))]
    print(f"幅值MSE={np.mean(mse_vals):.1f} L1={np.mean(l1_vals):.1f} | ω_MAE={omega_mae:.1f}rad/s ζ_MAE={zeta_mae:.5f}")

    np.savez(os.path.join(out_dir, "final_results.npz"),
             points=to_obj(all_points_list), frequencies=to_obj(all_freqs),
             predicted_frf=to_obj(all_preds), target_frf=to_obj(all_targets),
             predicted_re=to_obj(all_preds_re), target_re=to_obj(all_targets_re),
             predicted_im=to_obj(all_preds_im), target_im=to_obj(all_targets_im),
             pred_omega=to_obj(all_pred_omega), true_omega=to_obj(all_true_omega))
    print(f"数据保存: {out_dir}/final_results.npz")
    print(f"评估完成!")


if __name__ == '__main__':
    main()
