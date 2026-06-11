"""用训练好的 Transolver 模态-FRF 检查点预测 FRF。

用法:
    python sample/predict.py
    python sample/predict.py --sample-index 0 --query-node 10
    python sample/predict.py --response-dir Y --force-dir Y --tool-position 0.08 0.03 0.005
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from utils.direction import (
    DEFAULT_FORCE_DIRECTION,
    DEFAULT_RESPONSE_DIRECTION,
    direction_to_frf_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Transolver 网格 FRF 推理。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--split', default='test.h5')
    parser.add_argument('--checkpoint', default=os.path.join(os.path.dirname(__file__), 'output', 'checkpoint_best'))
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--query-node', type=int, default=None, help='可选：仅导出指定局部节点的 FRF。')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--slices', type=int, default=64)
    parser.add_argument('--no-edges', action='store_true')
    # 方向配置
    parser.add_argument('--response-dir', default=DEFAULT_RESPONSE_DIRECTION,
                        choices=['X', 'Y', 'Z'], help='响应测量方向（默认 Y）。')
    parser.add_argument('--force-dir', default=DEFAULT_FORCE_DIRECTION,
                        choices=['X', 'Y', 'Z'], help='力激励方向（默认 Y）。')
    # 刀触点（可选）
    parser.add_argument('--tool-position', nargs=3, type=float, default=None,
                        metavar=('X', 'Y', 'Z'), help='刀具位置坐标 (m)，用于查找最近节点作为激励点。')
    return parser.parse_args()


def build_model(dataset, args):
    first = dataset[0]
    return build_geometric_model({
        'in_dim': first['node_features'].shape[1],
        'hidden_dim': args.hidden_dim,
        'n_layers': args.layers,
        'n_heads': args.heads,
        'n_slices': args.slices,
        'n_modes': first['modal_omega'].shape[0],
        'use_edge_stem': not args.no_edges,
        'amp_scale': 500000.0,
        'response_direction': args.response_dir,
        'force_direction': args.force_dir,
    }, {})


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    frf_label = direction_to_frf_label(args.response_dir, args.force_dir)

    dataset = TransolverModalDataset([args.split], data_dir=args.data_dir, use_edges=not args.no_edges)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f'sample-index {args.sample_index} 超出范围 0..{len(dataset)-1}')

    sample = dataset[args.sample_index]
    batch = collate_mesh_batch([sample])

    # 如果传入了 --tool-position，查找最近节点作为激励点
    if args.tool_position is not None:
        tool_pos = torch.tensor(args.tool_position, dtype=torch.float32)
        dists = torch.norm(batch['points'] - tool_pos.unsqueeze(0), dim=1)
        nearest_idx = int(torch.argmin(dists))
        batch['excitation_index'] = torch.tensor([nearest_idx], dtype=torch.long)
        print(f'刀具位置: {args.tool_position}')
        print(f'最近节点: {nearest_idx}, 坐标: {batch["points"][nearest_idx].numpy()}')

    model = build_model(dataset, args).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    batch_dev = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(
            points=batch_dev['points'],
            node_features=batch_dev['node_features'],
            batch=batch_dev['batch'],
            edge_index=batch_dev.get('edge_index'),
            boundary_c_xyz=batch_dev.get('boundary_c_xyz'),
            excitation_index=batch_dev.get('excitation_index'),
            frequencies=batch_dev.get('frequencies'),
            num_graphs=batch_dev.get('num_graphs'),
        )

    frf = out['frf'].detach().cpu().numpy()
    amp = np.linalg.norm(frf, axis=-1)
    freqs = batch['frequencies'][0].numpy()
    pred_omega = out['modal_omega'].detach().cpu().numpy()[0]
    pred_zeta = out['modal_zeta'].detach().cpu().numpy()[0]

    if args.query_node is not None:
        if args.query_node < 0 or args.query_node >= frf.shape[0]:
            raise IndexError(f'query-node {args.query_node} 超出范围 0..{frf.shape[0]-1}')
        save_path = os.path.join(args.output_dir,
                                 f'prediction_{frf_label}_sample{args.sample_index}_node{args.query_node}.npz')
        np.savez(save_path,
                 frequency_hz=freqs,
                 frf_re=frf[args.query_node, :, 0],
                 frf_im=frf[args.query_node, :, 1],
                 frf_amp=amp[args.query_node],
                 pred_omega=pred_omega,
                 pred_zeta=pred_zeta,
                 query_point=batch['points'][args.query_node].numpy(),
                 response_direction=args.response_dir,
                 force_direction=args.force_dir,
                 frf_label=frf_label)
    else:
        save_path = os.path.join(args.output_dir,
                                 f'prediction_{frf_label}_sample{args.sample_index}.npz')
        np.savez(save_path,
                 frequency_hz=freqs,
                 frf=frf,
                 frf_amp=amp,
                 pred_omega=pred_omega,
                 pred_zeta=pred_zeta,
                 points=batch['points'].numpy(),
                 response_direction=args.response_dir,
                 force_direction=args.force_dir,
                 frf_label=frf_label)

    print(f'预测样本 {args.sample_index} ({frf_label}): 节点={frf.shape[0]}, 频率={frf.shape[1]}')
    print(f'预测固有频率 (rad/s): {pred_omega}')
    print(f'预测阻尼比: {pred_zeta}')
    print(f'已保存: {save_path}')


if __name__ == '__main__':
    main()
