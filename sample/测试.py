"""
测试.py — 查看新版 GraphHDF5Dataset 真实 FRF 与图数据字段。

用法:
    F:/pytorch_cuda12/python.exe sample/测试.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM


CONFIG = {'freq_min': 1.0, 'freq_max': 5000.0, 'omega_max': 32000.0, 'graph': {'knn_k': 12}}
data_dir = os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data')
out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)


def pick_nodes(points, amp):
    x_sorted = np.argsort(points[:, 0])
    base = [x_sorted[0], x_sorted[len(x_sorted)//4], x_sorted[len(x_sorted)//2], x_sorted[3*len(x_sorted)//4], x_sorted[-1]]
    max_amp = int(np.argmax(np.mean(amp, axis=1)))
    selected = []
    for idx in base + [max_amp]:
        if idx not in selected:
            selected.append(idx)
    return selected[:6]


def main():
    ds = GraphHDF5Dataset(['train.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)
    sample = ds[0]
    points = sample['points'].numpy()
    freqs = sample['frequencies'].numpy()
    frf = sample['point_frf']
    amp = torch.sqrt(frf[..., 0] ** 2 + frf[..., 1] ** 2 + 1e-12).numpy()

    print(f"nodes={points.shape[0]}, edges={sample['edge_index'].shape[1]}, node_feature_dim={sample['node_features'].shape[-1]}/{NODE_FEATURE_DIM}")
    print(f"freq range=[{freqs[0]:.2f}, {freqs[-1]:.2f}] Hz, F={len(freqs)}")
    if 'modal_omega_phys' in sample:
        print('true modal Hz:', [f'{x:.2f}' for x in (sample['modal_omega_phys'] / (2 * torch.pi)).numpy()])
    if 'modal_zeta' in sample:
        print('true zeta:', [f'{x:.6f}' for x in sample['modal_zeta'].numpy()])

    nodes = pick_nodes(points, amp)
    fig, axes = plt.subplots(len(nodes), 1, figsize=(13, 2.5 * len(nodes)), sharex=True)
    if len(nodes) == 1:
        axes = [axes]
    for ax, node_idx in zip(axes, nodes):
        ax.plot(freqs, amp[node_idx], linewidth=1.2)
        ax.set_ylabel(f'N{node_idx}\nx={points[node_idx,0]:.3f}')
        ax.grid(alpha=0.3)
        for fk in (sample['modal_omega_phys'] / (2 * torch.pi)).numpy():
            ax.axvline(fk, linestyle='--', linewidth=0.8, alpha=0.6)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('True FRF amplitude from GraphHDF5Dataset')
    fig.tight_layout()
    save_path = os.path.join(out_dir, 'true_frf_graph_dataset.png')
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'saved: {save_path}')


if __name__ == '__main__':
    main()
