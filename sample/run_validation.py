"""训练 Transolver 模态-FRF 模型（ANSYS 网格 HDF5 数据）。

用法:
    python sample/run_validation.py
    python sample/run_validation.py --data-dir ansys/data --epochs 300 --batch-size 2
    python sample/run_validation.py --response-dir Y --force-dir Y
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from training.trainer import train, evaluate
from utils.direction import (
    DEFAULT_FORCE_DIRECTION,
    DEFAULT_RESPONSE_DIRECTION,
    direction_to_frf_label,
)


DEFAULT_CONFIG = {
    'epochs': 300,
    'validation_frequency': 5,
    'use_frf_loss': False,
    'frf_loss_weight': 1.0,
    'gradient_clip': 1.0,

    'physics_alpha_warmup': 50,

    'modal_loss_weights': {
        # 频率和阻尼
        'omega': 10.0,
        'zeta': 1.0,

        # 响应方向振型：内部已经包含 shape + scale + participation
        'phi_resp': 1.0,

        # 完整三向振型：内部包含 xyz shape + 少量 energy
        'phi_xyz': 0.5,

        # MAC 只作为辅助，避免弱 Z 模态被 MAC 过度惩罚
        'mac': 0.05,

        # 内部比例项
        'phi_resp_scale_ratio': 0.3,
        'participation_ratio': 0.3,
        'phi_xyz_energy_ratio': 0.1,
    },

    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-4,
    },
}


class Args:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='训练 Transolver 模态-FRF 模型。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch-size', type=int, default=2, help='GTX 1650/4GB 推荐 2，显存更多可 3-4。')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--slices', type=int, default=64)
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['optimizer']['lr'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_CONFIG['optimizer']['weight_decay'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-fp16', action='store_true', help='禁用混合精度（默认开启 fp16）。')
    parser.add_argument('--no-edges', action='store_true', help='禁用单元连接边 stem。')
    # 方向配置
    parser.add_argument('--response-dir', default=DEFAULT_RESPONSE_DIRECTION,
                        choices=['X', 'Y', 'Z'], help='响应测量方向（默认 Y）。')
    parser.add_argument('--force-dir', default=DEFAULT_FORCE_DIRECTION,
                        choices=['X', 'Y', 'Z'], help='力激励方向（默认 Y）。')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data_dir, filename, batch_size, shuffle, use_edges):
    dataset = TransolverModalDataset([filename], data_dir=data_dir, use_edges=use_edges)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mesh_batch,
    )
    return dataset, loader


def main():
    cli = parse_args()
    set_seed(cli.seed)
    os.makedirs(cli.output_dir, exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    config['epochs'] = cli.epochs
    config['optimizer'] = {'lr': cli.lr, 'weight_decay': cli.weight_decay}

    # FRF 标签（H_ab）
    frf_label = direction_to_frf_label(cli.response_dir, cli.force_dir)

    print('=' * 72)
    print(f'Transolver 模态-FRF 训练 ({frf_label})')
    print('=' * 72)
    print(f'设备:     {cli.device}')
    print(f'数据:     {cli.data_dir}')
    print(f'输出:     {cli.output_dir}')
    print(f'方向:     响应={cli.response_dir}, 激励={cli.force_dir} → {frf_label}')

    trainset, trainloader = make_loader(cli.data_dir, 'train.h5', cli.batch_size, True, not cli.no_edges)
    valset, valloader = make_loader(cli.data_dir, 'val.h5', 1, False, not cli.no_edges)
    testset, testloader = make_loader(cli.data_dir, 'test.h5', 1, False, not cli.no_edges)

    first = trainset[0]
    in_dim = first['node_features'].shape[1]
    n_modes = first['modal_omega'].shape[0]
    print(f'训练/验证/测试: {len(trainset)}/{len(valset)}/{len(testset)} 样本')
    print(f'节点特征维度: {in_dim}, 模态阶数: {n_modes}, 首个网格节点数: {first["points"].shape[0]}')

    model_cfg = {
        'encoder_kwargs': {
            'in_dim': in_dim,
            'hidden_dim': cli.hidden_dim,
            'n_layers': cli.layers,
            'n_heads': cli.heads,
            'n_slices': cli.slices,
            'n_modes': n_modes,
            'use_edge_stem': not cli.no_edges,
            'amp_scale': 500000.0,
            'response_direction': cli.response_dir,
            'force_direction': cli.force_dir,
        },
        'decoder_kwargs': {},
    }
    net = build_geometric_model(model_cfg['encoder_kwargs'], model_cfg['decoder_kwargs']).to(cli.device)
    print(f'模型参数: {sum(p.numel() for p in net.parameters()):,}')

    optimizer = torch.optim.AdamW(net.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.epochs, eta_min=1e-6)

    args = Args()
    args.device = cli.device
    args.fp16 = not cli.no_fp16
    args.dir = cli.output_dir

    ckpt_last = os.path.join(cli.output_dir, 'checkpoint_last')
    start_epoch = 0
    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location=cli.device)
        net.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        print(f'从 epoch {start_epoch} 恢复训练')

    t0 = time.time()
    net = train(args, config, model_cfg, net, trainloader, optimizer, valloader, scheduler, start_epoch=start_epoch)
    elapsed = time.time() - t0

    best_path = os.path.join(cli.output_dir, 'checkpoint_best')
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=cli.device)['model_state_dict'])
    metrics = evaluate(args, config, net, testloader, verbose=True)
    print(f'训练完成, 耗时 {elapsed:.1f}s')
    print(f'测试指标 ({frf_label}): {metrics}')


if __name__ == '__main__':
    main()
