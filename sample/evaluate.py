"""评估训练好的 Transolver 模态-FRF 检查点。

用法:
    python sample/evaluate.py
    python sample/evaluate.py --data-dir ansys/data --checkpoint sample/output/checkpoint_best
    python sample/evaluate.py --response-dir Y --force-dir Y

输出:
    sample/output/final_results.npz
    sample/output/eval_summary.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from training.trainer import TransolverTrainer
from utils.direction import (
    DEFAULT_FORCE_DIRECTION,
    DEFAULT_RESPONSE_DIRECTION,
    direction_to_frf_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description='评估 Transolver 模态-FRF 检查点。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--split', default='test.h5')
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--checkpoint', default=None)
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
    return parser.parse_args()


def object_array(items: List[np.ndarray]) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def build_model_from_dataset(dataset, args):
    first = dataset[0]
    model = build_geometric_model({
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
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = args.checkpoint or os.path.join(args.output_dir, 'checkpoint_best')

    frf_label = direction_to_frf_label(args.response_dir, args.force_dir)
    print('=' * 60)
    print(f'Transolver 模态-FRF 评估 ({frf_label})')
    print('=' * 60)
    print(f'方向: 响应={args.response_dir}, 激励={args.force_dir}')
    print(f'数据: {args.data_dir}/{args.split}')
    print(f'检查点: {checkpoint}')

    dataset = TransolverModalDataset([args.split], data_dir=args.data_dir, use_edges=not args.no_edges)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_mesh_batch)

    model = build_model_from_dataset(dataset, args).to(args.device)
    ckpt = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    trainer = TransolverTrainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-6), device=args.device)
    metrics = trainer.evaluate(loader, {
        'use_frf_loss': True,
        'frf_loss_weight': 1.0,
        'modal_loss_weights': {'omega': 1.0, 'zeta': 0.5, 'phi_resp': 1.0, 'mac': 0.2},
    })

    all_points, all_freqs = [], []
    all_pred_amp, all_true_amp = [], []
    all_pred_re, all_true_re = [], []
    all_pred_im, all_true_im = [], []
    all_pred_omega, all_true_omega = [], []
    all_pred_zeta, all_true_zeta = [], []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
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
            pred = out['frf'].detach().cpu()
            true = batch['point_frf'].detach().cpu()
            pred_amp = torch.linalg.norm(pred, dim=-1).numpy()
            true_amp = torch.linalg.norm(true, dim=-1).numpy()

            all_points.append(batch['points'].numpy())
            all_freqs.append(batch['frequencies'][0].numpy())
            all_pred_amp.append(pred_amp)
            all_true_amp.append(true_amp)
            all_pred_re.append(pred[..., 0].numpy())
            all_true_re.append(true[..., 0].numpy())
            all_pred_im.append(pred[..., 1].numpy())
            all_true_im.append(true[..., 1].numpy())
            all_pred_omega.append(out['modal_omega'].detach().cpu().numpy())
            all_true_omega.append(batch['modal_omega'].numpy())
            all_pred_zeta.append(out['modal_zeta'].detach().cpu().numpy())
            all_true_zeta.append(batch['modal_zeta'].numpy())

    np.savez(
        os.path.join(args.output_dir, 'final_results.npz'),
        points=object_array(all_points),
        frequencies=object_array(all_freqs),
        predicted_frf=object_array(all_pred_amp),
        target_frf=object_array(all_true_amp),
        predicted_re=object_array(all_pred_re),
        target_re=object_array(all_true_re),
        predicted_im=object_array(all_pred_im),
        target_im=object_array(all_true_im),
        pred_omega=object_array(all_pred_omega),
        true_omega=object_array(all_true_omega),
        pred_zeta=object_array(all_pred_zeta),
        true_zeta=object_array(all_true_zeta),
    )

    summary_path = os.path.join(args.output_dir, 'eval_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Transolver 模态-FRF 评估 ({frf_label})\n')
        f.write(f'检查点: {checkpoint}\n')
        f.write(f'样本数: {len(dataset)}\n')
        f.write(f'响应方向: {args.response_dir}\n')
        f.write(f'激励方向: {args.force_dir}\n')
        f.write(f'FRF 标签: {frf_label}\n')
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')

    print(f'评估指标 ({frf_label}): {metrics}')
    print(f'已保存: {os.path.join(args.output_dir, "final_results.npz")}')
    print(f'摘要: {summary_path}')


if __name__ == '__main__':
    main()
