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
from models import build_geometric_model
from utils.direction import (
    DEFAULT_FORCE_DIRECTION,
    DEFAULT_RESPONSE_DIRECTION,
    direction_to_frf_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description='预测 vs 真实 FRF 对比图。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--response-dir', default=DEFAULT_RESPONSE_DIRECTION,
                        choices=['X', 'Y', 'Z'], help='响应测量方向（默认 Y）。')
    parser.add_argument('--force-dir', default=DEFAULT_FORCE_DIRECTION,
                        choices=['X', 'Y', 'Z'], help='力激励方向（默认 Y）。')
    parser.add_argument('--no-edges', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    frf_label = direction_to_frf_label(args.response_dir, args.force_dir)

    dataset = TransolverModalDataset(['test.h5'], data_dir=args.data_dir, use_edges=not args.no_edges)
    batch = dataset[0]

    model = build_geometric_model({
        'in_dim': batch['node_features'].shape[1],
        'hidden_dim': 256,
        'n_layers': 6,
        'n_heads': 8,
        'n_slices': 64,
        'n_modes': batch['modal_omega'].shape[0],
        'use_edge_stem': not args.no_edges,
        'amp_scale': 500000.0,
        'response_direction': args.response_dir,
        'force_direction': args.force_dir,
    }, {})

    ckpt_path = os.path.join('output', 'checkpoint_best')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    points = batch['points']
    node_features = batch['node_features']
    batch_idx = batch['batch']

    with torch.no_grad():
        out = model(
            points, node_features, batch_idx,
            edge_index=batch.get('edge_index'),
            boundary_c_xyz=batch.get('boundary_c_xyz'),
            excitation_index=batch.get('excitation_index'),
            frequencies=batch.get('frequencies'),
            num_graphs=batch.get('num_graphs'),
        )

    frf_pred = out['frf'].numpy()
    frf_true = batch['point_frf'].numpy()
    amp_pred = np.linalg.norm(frf_pred, axis=-1)
    amp_true = np.linalg.norm(frf_true, axis=-1)

    idx_node = 0  # 可视化首个节点

    plt.figure(figsize=(12, 4))
    plt.suptitle(f'FRF 预测 vs 真实 [{frf_label}] — 节点 {idx_node}')

    plt.subplot(1, 3, 1)
    plt.plot(amp_true[idx_node], label='真实')
    plt.plot(amp_pred[idx_node], label='预测')
    plt.title(f'幅值 [{frf_label}]')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(frf_true[idx_node, :, 0], label='真实 Re')
    plt.plot(frf_pred[idx_node, :, 0], label='预测 Re')
    plt.title('实部')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(frf_true[idx_node, :, 1], label='真实 Im')
    plt.plot(frf_pred[idx_node, :, 1], label='预测 Im')
    plt.title('虚部')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
