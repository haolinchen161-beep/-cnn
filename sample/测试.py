"""可视化 Transolver 网格批次的 FRF 幅值和模态振型。"""
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
    parser = argparse.ArgumentParser(description='Transolver FRF 快速可视化。')
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

    frf = out['frf'].numpy()
    amp = np.linalg.norm(frf, axis=-1)

    plt.figure(figsize=(8, 4))
    plt.plot(amp[0])
    plt.title(f'预测 FRF 幅值（首个节点） [{frf_label}]')
    plt.xlabel('频率索引')
    plt.ylabel('幅值')
    plt.show()


if __name__ == '__main__':
    main()
