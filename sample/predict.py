"""
predict.py — 加载 MeshGraphNet checkpoint 对测试样本预测 FRF。

用法:
    F:/pytorch_cuda12/python.exe sample/predict.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from models import build_geometric_model
from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM


CONFIG = {
    'freq_min': 1.0,
    'freq_max': 5000.0,
    'omega_max': 25000.0,
    'graph': {'knn_k': 12},
}

MODEL_CFG = {
    'encoder_kwargs': {
        'node_in_dim': NODE_FEATURE_DIM,
        'edge_in_dim': 4,
        'hidden': 256,
        'n_layers': 8,
        'n_modes': 3,
        'omega_max': 25000.0,
        'amp_scale': 500000.0,
        'freq_min': 1.0,
        'freq_max': 5000.0,
        'dropout': 0.05,
    },
    'decoder_kwargs': {},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data')
out_dir = os.path.join(os.path.dirname(__file__), 'output')
ckpt_path = os.path.join(out_dir, 'checkpoint_best')


def to_obj(arr_list):
    out = np.empty(len(arr_list), dtype=object)
    for i, a in enumerate(arr_list):
        out[i] = a
    return out


def align_phi(pred_phi, true_phi):
    aligned = pred_phi.clone()
    signs = []
    for k in range(pred_phi.shape[1]):
        dot = torch.sum(pred_phi[:, k] * true_phi[:, k])
        sign = torch.sign(dot + 1e-8)
        aligned[:, k] = pred_phi[:, k] * sign
        signs.append(sign)
    return aligned, torch.stack(signs)


def main():
    testset = GraphHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset_raw = GraphHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)

    net = build_geometric_model(MODEL_CFG['encoder_kwargs'], MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"Checkpoint epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}, params={sum(p.numel() for p in net.parameters()):,}")
    print(f"Using NODE_FEATURE_DIM={NODE_FEATURE_DIM}")

    omega_max = float(CONFIG.get('omega_max', 25000.0))
    all_preds, all_targets = [], []
    all_pred_re, all_pred_im = [], []
    all_true_re, all_true_im = [], []

    for idx in range(len(testset)):
        sn, sr = testset[idx], testset_raw[idx]
        node_features = sn['node_features'].to(device)
        edge_index = sn['edge_index'].to(device)
        edge_attr = sn['edge_attr'].to(device)
        batch_idx = torch.zeros(sn['points'].shape[0], dtype=torch.long, device=device)
        freqs = sn['frequencies'].unsqueeze(0).to(device)
        true_phi = sn['modal_phi'].to(device)
        phi_exc = sn.get('modal_phi_exc')
        phi_exc_t = phi_exc.unsqueeze(0).to(device) if phi_exc is not None else None

        with torch.no_grad():
            _, omega_norm, zeta_pred, phi_pred = net(
                node_features, edge_index, edge_attr, batch_idx,
                frequencies=None, phi_exc=None,
            )
            omega_norm = omega_norm.squeeze(0)
            zeta_pred = zeta_pred.squeeze(0)
            omega_norm_sorted, sort_idx = torch.sort(omega_norm)
            zeta_sorted = zeta_pred[sort_idx]
            phi_sorted = phi_pred[:, sort_idx]
            phi_aligned, _ = align_phi(phi_sorted, true_phi)
            omega_phys = omega_norm_sorted * omega_max

            frf_p = net.physics(
                phi_aligned,
                omega_phys.unsqueeze(0),
                zeta_sorted.unsqueeze(0),
                freqs,
                phi_exc_t,
                batch_idx=batch_idx,
                alpha=1.0,
            )

        p = frf_p.detach().cpu()
        t = sr['point_frf'].detach().cpu()
        all_preds.append(torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-8).numpy())
        all_targets.append(torch.sqrt(t[..., 0] ** 2 + t[..., 1] ** 2 + 1e-8).numpy())
        all_pred_re.append(p[..., 0].numpy())
        all_pred_im.append(p[..., 1].numpy())
        all_true_re.append(t[..., 0].numpy())
        all_true_im.append(t[..., 1].numpy())

        if idx % 10 == 0:
            mse_i = np.mean((all_preds[-1] - all_targets[-1]) ** 2)
            print(f'[{idx+1:03d}/{len(testset)}] amp MSE={mse_i:.6e}')

    mse_vals = [np.mean((all_preds[i] - all_targets[i]) ** 2) for i in range(len(all_preds))]
    print(f'Test amplitude MSE: {np.mean(mse_vals):.6e}')

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, 'predictions.npz'),
        predicted=to_obj(all_preds),
        target=to_obj(all_targets),
        predicted_re=to_obj(all_pred_re),
        predicted_im=to_obj(all_pred_im),
        target_re=to_obj(all_true_re),
        target_im=to_obj(all_true_im),
    )
    print(f'Saved: {out_dir}/predictions.npz')


if __name__ == '__main__':
    main()
