"""对比 Transolver 网格批次的预测 FRF vs 目标 FRF。

显示幅值、实部和虚部。
"""
from __future__ import annotations

import os
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt

from data.dataset import TransolverModalDataset, collate_mesh_batch


def parse_args():
    parser = argparse.ArgumentParser(description='预测 vs 真实 FRF 对比图。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output_mid_192_4_48_noedge'))
    parser.add_argument('--response-dir', default='Z', choices=['X', 'Y', 'Z'])
    parser.add_argument('--force-dir', default='Z', choices=['X', 'Y', 'Z'])
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--slices', type=int, default=48)
    parser.add_argument('--no-edges', action='store_true', default=True)
    parser.add_argument('--sample-idx', type=int, default=0, help='要可视化的样本索引。')
    return parser.parse_args()


def main():
    from models import build_geometric_model

    args = parse_args()

    dataset = TransolverModalDataset(['test.h5'], data_dir=args.data_dir, use_edges=not args.no_edges)

    # 使用 collate 生成 batch 索引
    sample = dataset[args.sample_idx]
    batch = collate_mesh_batch([sample])

    model = build_geometric_model({
        'in_dim': batch['node_features'].shape[1],
        'hidden_dim': args.hidden_dim,
        'n_layers': args.layers,
        'n_heads': 8,
        'n_slices': args.slices,
        'n_modes': batch['modal_omega'].shape[1],
        'use_edge_stem': not args.no_edges,
        'amp_scale': 500000.0,
        'response_direction': args.response_dir,
        'force_direction': args.force_dir,
    }, {})

    ckpt_path = os.path.join(args.output_dir, 'checkpoint_best')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'已加载: {ckpt_path}')
    else:
        print(f'警告: 未找到 {ckpt_path}，使用随机权重')
    model.eval()

    with torch.no_grad():
        out = model(
            points=batch['points'],
            node_features=batch['node_features'],
            batch=batch['batch'],
            edge_index=batch.get('edge_index'),
            boundary_c_xyz=batch.get('boundary_c_xyz'),
            excitation_index=batch.get('excitation_index'),
            frequencies=batch.get('frequencies'),
            node_counts=batch.get('node_counts'),
        )

    frf_pred = out['frf'].numpy()
    frf_true = batch['point_frf'].numpy()
    amp_pred = np.linalg.norm(frf_pred, axis=-1)
    amp_true = np.linalg.norm(frf_true, axis=-1)

    # 激励点的 FRF
    exc_idx = int(batch['excitation_index'][0])
    frf_label = f"H_{args.response_dir}{args.force_dir}"

    plt.figure(figsize=(14, 8))
    plt.suptitle(f'FRF 预测 vs 真实 [{frf_label}] — 样本 {args.sample_idx} (激励点 {exc_idx})')

    plt.subplot(2, 2, 1)
    plt.semilogy(amp_true[exc_idx], label='真实')
    plt.semilogy(amp_pred[exc_idx], label='预测')
    plt.title(f'幅值 [{frf_label}]')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(frf_true[exc_idx, :, 0], label='真实 Re')
    plt.plot(frf_pred[exc_idx, :, 0], label='预测 Re')
    plt.title('实部')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(frf_true[exc_idx, :, 1], label='真实 Im')
    plt.plot(frf_pred[exc_idx, :, 1], label='预测 Im')
    plt.title('虚部')
    plt.legend()

    # 模态参数对比
    omega_pred = out['modal_omega'][0].numpy()
    omega_true = batch['modal_omega'][0].numpy()
    zeta_pred = out['modal_zeta'][0].numpy()
    zeta_true = batch['modal_zeta'][0].numpy()

    plt.subplot(2, 2, 4)
    freqs = batch['frequencies'][0].numpy()
    for k in range(len(omega_pred)):
        f_k = omega_pred[k] / (2 * np.pi)
        plt.axvline(x=f_k, color=f'C{k}', linestyle='--', alpha=0.7,
                     label=f'预测 ω{k}: {f_k:.0f}Hz, ζ{k}: {zeta_pred[k]*100:.2f}%')
    for k in range(len(omega_true)):
        f_k = omega_true[k] / (2 * np.pi)
        plt.axvline(x=f_k, color=f'C{k}', linestyle=':', alpha=0.5,
                     label=f'真实 ω{k}: {f_k:.0f}Hz, ζ{k}: {zeta_true[k]*100:.2f}%')
    plt.semilogy(freqs, amp_true[exc_idx], 'k-', alpha=0.3)
    plt.title('模态频率标注')
    plt.legend(fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, f'compare_{args.sample_idx}.png')
    plt.savefig(save_path, dpi=150)
    print(f'已保存: {save_path}')
    plt.show()


if __name__ == '__main__':
    main()
