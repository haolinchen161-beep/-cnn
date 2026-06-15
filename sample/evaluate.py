"""MeshGraphNet modal parameter evaluation + FRF result export."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM
from models import build_geometric_model


CONFIG = {
    "freq_min": 1.0,
    "freq_max": 5000.0,
    "omega_max": 32000.0,
    "filter_g32": False,
    "graph": {"knn_k": 12},
}

MODEL_CFG = {
    "encoder_kwargs": {
        "node_in_dim": NODE_FEATURE_DIM,
        "edge_in_dim": 4,
        "hidden": 128,
        "n_layers": 4,
        "n_modes": 3,
        "amp_scale": 500000.0,
        "freq_min": 1.0,
        "freq_max": 5000.0,
        "dropout": 0.05,
    },
    "decoder_kwargs": {},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data")
out_dir = os.path.join(os.path.dirname(__file__), "output_meshgraphnet")
ckpt_path = os.path.join(out_dir, "checkpoint_best_modal")
EPS = 1e-12


def to_obj(arr_list):
    out = np.empty(len(arr_list), dtype=object)
    for i, a in enumerate(arr_list):
        out[i] = a
    return out


def sign_align_phi(pred_phi: torch.Tensor, true_phi: torch.Tensor, eps=1e-8):
    aligned = pred_phi.clone()
    signs = []
    for k in range(pred_phi.shape[1]):
        dot = torch.sum(pred_phi[:, k] * true_phi[:, k])
        sign = torch.sign(dot + eps)
        aligned[:, k] = pred_phi[:, k] * sign
        signs.append(sign)
    return aligned, torch.stack(signs)


def phi_metrics(pred_phi: torch.Tensor, true_phi: torch.Tensor, eps=1e-8):
    pred_phi_aligned, signs = sign_align_phi(pred_phi, true_phi, eps=eps)
    macs, nrmse, phi_a = [], [], []
    for k in range(true_phi.shape[1]):
        p, t = pred_phi_aligned[:, k], true_phi[:, k]
        mac = (torch.sum(p * t) ** 2) / (torch.sum(p ** 2) * torch.sum(t ** 2) + eps)
        rmse = torch.sqrt(torch.mean((p - t) ** 2))
        true_std = torch.std(t.reshape(-1)) + eps
        norm_p = torch.sqrt(torch.sum(p ** 2) + eps)
        norm_t = torch.sqrt(torch.sum(t ** 2) + eps)
        macs.append(mac)
        nrmse.append(rmse / true_std)
        phi_a.append(torch.abs(norm_p - norm_t) / (norm_t + eps))
    return torch.stack(macs), torch.stack(nrmse), torch.stack(phi_a), pred_phi_aligned, signs


def compute_peak_metrics(freqs_hz, pred_amp, true_amp, true_freq_hz, true_zeta):
    freqs = np.asarray(freqs_hz)
    pred_env = np.mean(pred_amp, axis=0)
    true_env = np.mean(true_amp, axis=0)
    peak_shift, peak_amp_rel, pred_peak_freq, true_peak_freq = [], [], [], []
    for fk, zk in zip(true_freq_hz, true_zeta):
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
        tf, pf = float(local_freqs[idx_t]), float(local_freqs[idx_p])
        ta, pa = float(local_true[idx_t]), float(local_pred[idx_p])
        peak_shift.append(abs(pf - tf))
        peak_amp_rel.append(abs(pa - ta) / (abs(ta) + EPS))
        pred_peak_freq.append(pf)
        true_peak_freq.append(tf)
    return [np.asarray(x, dtype=np.float32) for x in (peak_shift, peak_amp_rel, pred_peak_freq, true_peak_freq)]


def main():
    print("=" * 80)
    print("MeshGraphNet evaluation")
    print("=" * 80)

    testset = GraphHDF5Dataset(["test.h5"], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset_raw = GraphHDF5Dataset(["test.h5"], CONFIG, data_dir=data_dir, normalization=False, test=True)
    print(f"Test samples: {len(testset)} | node_dim={NODE_FEATURE_DIM} | device={device}")

    model = build_geometric_model(MODEL_CFG["encoder_kwargs"], MODEL_CFG["decoder_kwargs"]).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint epoch={ckpt.get('epoch', 'NA')}, loss={ckpt.get('loss', -1):.6f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    all_points, all_freqs = [], []
    all_pred_amp, all_true_amp = [], []
    all_pred_re, all_true_re = [], []
    all_pred_im, all_true_im = [], []
    all_pred_omega, all_true_omega = [], []
    all_pred_freq_hz, all_true_freq_hz, all_freq_rel = [], [], []
    all_pred_zeta, all_true_zeta, all_zeta_rel = [], [], []
    all_pred_phi, all_true_phi = [], []
    all_phi_mac, all_phi_nrmse, all_phi_a = [], [], []
    all_peak_shift_hz, all_peak_amp_rel = [], []
    all_pred_peak_freq, all_true_peak_freq = [], []

    for idx in range(len(testset)):
        sn = testset[idx]
        sr = testset_raw[idx]

        batch_vec = torch.zeros(sn["points"].shape[0], dtype=torch.long, device=device)
        node_features = sn["node_features"].to(device)
        edge_index = sn["edge_index"].to(device)
        edge_attr = sn["edge_attr"].to(device)
        frequencies = sn["frequencies"].unsqueeze(0).to(device)

        true_phi = sn["modal_phi"].to(device)
        true_zeta = sn["modal_zeta"].to(device)
        true_omega = sn["modal_omega_phys"].to(device)
        true_freq_hz = true_omega / (2.0 * torch.pi)

        phi_exc = sn.get("modal_phi_exc")
        phi_exc = phi_exc.unsqueeze(0).to(device) if phi_exc is not None else None
        exc_idx = sn.get("excitation_index")
        exc_idx_global = exc_idx.unsqueeze(0).to(device) if exc_idx is not None else None

        with torch.no_grad():
            frf_pred, omega_pred, log_zeta, zeta_pred, phi_pred = model(
                node_features, edge_index, edge_attr, batch_vec,
                frequencies=frequencies,
                phi_exc=phi_exc,
                excitation_index_global=exc_idx_global,
            )
            omega_pred = omega_pred.squeeze(0)
            zeta_pred = zeta_pred.squeeze(0)
            mac, nrmse, phi_a, phi_aligned, _ = phi_metrics(phi_pred, true_phi)
            freq_hz_pred = omega_pred / (2.0 * torch.pi)

        p = frf_pred.detach().cpu()
        t = sr["point_frf"].detach().cpu()
        pred_amp = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + EPS).numpy()
        true_amp = torch.sqrt(t[..., 0] ** 2 + t[..., 1] ** 2 + EPS).numpy()
        pred_re, pred_im = p[..., 0].numpy(), p[..., 1].numpy()
        true_re, true_im = t[..., 0].numpy(), t[..., 1].numpy()

        true_omega_cpu = true_omega.detach().cpu()
        pred_omega_cpu = omega_pred.detach().cpu()
        true_freq_cpu = true_freq_hz.detach().cpu()
        pred_freq_cpu = freq_hz_pred.detach().cpu()
        true_zeta_cpu = true_zeta.detach().cpu()
        pred_zeta_cpu = zeta_pred.detach().cpu()

        freq_rel = torch.abs(pred_freq_cpu - true_freq_cpu) / (true_freq_cpu + 1e-8)
        zeta_rel = torch.abs(pred_zeta_cpu - true_zeta_cpu) / (true_zeta_cpu + 1e-8)

        freqs_phys = sr["frequencies"].numpy()
        peak_shift, peak_amp_rel, pred_peak_f, true_peak_f = compute_peak_metrics(
            freqs_phys, pred_amp, true_amp, true_freq_cpu.numpy(), true_zeta_cpu.numpy()
        )

        all_points.append(sr["points"].numpy())
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
        all_phi_a.append(phi_a.detach().cpu().numpy())
        all_peak_shift_hz.append(peak_shift)
        all_peak_amp_rel.append(peak_amp_rel)
        all_pred_peak_freq.append(pred_peak_f)
        all_true_peak_freq.append(true_peak_f)

        if idx % 10 == 0:
            print(f"[{idx+1:03d}/{len(testset)}] freq_rel%={freq_rel.mean().item()*100:.3f}, "
                  f"zeta_rel%={zeta_rel.mean().item()*100:.2f}, MAC={mac.mean().item():.4f}")

    freq_rel_all = np.stack(all_freq_rel, axis=0)
    zeta_rel_all = np.stack(all_zeta_rel, axis=0)
    phi_mac_all = np.stack(all_phi_mac, axis=0)
    phi_nrmse_all = np.stack(all_phi_nrmse, axis=0)
    phi_a_all = np.stack(all_phi_a, axis=0)
    peak_shift_all = np.stack(all_peak_shift_hz, axis=0)
    peak_amp_rel_all = np.stack(all_peak_amp_rel, axis=0)
    amp_mse_vals = [np.mean((all_pred_amp[i] - all_true_amp[i]) ** 2) for i in range(len(all_pred_amp))]
    amp_l1_vals = [np.mean(np.abs(all_pred_amp[i] - all_true_amp[i])) for i in range(len(all_pred_amp))]

    print("\n" + "=" * 80)
    print("Test summary")
    print("=" * 80)
    print(f"FRF amplitude MSE mean = {np.mean(amp_mse_vals):.6e}")
    print(f"FRF amplitude L1  mean = {np.mean(amp_l1_vals):.6e}")
    print(f"Frequency relative error mean (%) = {np.mean(freq_rel_all) * 100.0:.4f}")
    print(f"Zeta relative error mean (%) = {np.mean(zeta_rel_all) * 100.0:.4f}")
    print(f"Mode MAC mean = {np.mean(phi_mac_all):.6f}")
    print(f"Mode phi_n mean (%) = {np.mean(phi_nrmse_all) * 100.0:.4f}")
    print(f"Mode phi_a mean (%) = {np.mean(phi_a_all) * 100.0:.4f}")
    print(f"FRF peak frequency shift mean (Hz) = {np.mean(peak_shift_all):.4f}")
    print(f"FRF peak amplitude relative error mean (%) = {np.mean(peak_amp_rel_all) * 100.0:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "final_results.npz")
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
        phi_a=to_obj(all_phi_a),
        peak_shift_hz=to_obj(all_peak_shift_hz),
        peak_amp_rel=to_obj(all_peak_amp_rel),
        pred_peak_freq=to_obj(all_pred_peak_freq),
        true_peak_freq=to_obj(all_true_peak_freq),
    )
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
