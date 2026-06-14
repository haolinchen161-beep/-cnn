"""训练 Transolver-Modal 模态参数预测模型。

用法：
    F:/pytorch_cuda12/python.exe -B sample/run_validation.py

该脚本使用新的轻量 Transolver-Modal 架构：
    节点/网格输入 → slice tokens → graph heads(ω,ζ) + mode-token node head(φ) → PhysicsDecoder(FRF)

训练流程对齐当前 CNN 正确版本：
    Phase0: 频率专属预训练
    Phase1: 全模态训练
    Phase2: FRF 弱约束微调
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings

warnings.filterwarnings('ignore', message='Detected call of')
warnings.filterwarnings('ignore', message='To get the last learning rate')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from training.trainer import train, evaluate
from training.losses import modal_loss


CONFIG = {
    'epochs': 500,
    'validation_frequency': 5,

    # 阶段控制
    'omega_pretrain_epochs': 50,
    'enable_phase2': True,
    'phase2_min_epoch': 300,
    'frf_teacher_epochs': 50,

    # 模态损失权重
    'omega_loss_weight': 1.0,
    'zeta_loss_weight': 10.0,
    'phi_loss_weight': 3.0,

    # FRF 弱约束
    'frf_loss_weight': 0.02,
    'frf_warmup_epochs': 20,

    'optimizer': {
        'name': 'AdamW',
        'kwargs': {'lr': 5e-4, 'weight_decay': 0.005, 'betas': (0.9, 0.999)},
        'gradient_clip': 2.0,
    },
}


class Args:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Transolver-Modal 训练脚本')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output_transolver_modal'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--slices', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-edges', action='store_true')
    parser.add_argument('--response-dir', default='Z', choices=['X', 'Y', 'Z'])
    parser.add_argument('--force-dir', default='Z', choices=['X', 'Y', 'Z'])
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data_dir: str, filename: str, batch_size: int, shuffle: bool, use_edges: bool):
    dataset = TransolverModalDataset([filename], data_dir=data_dir, use_edges=use_edges, require_frf=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mesh_batch,
    )
    return dataset, loader


def main():
    cli = parse_args()
    set_seed(cli.seed)
    config = dict(CONFIG)
    config['epochs'] = cli.epochs

    print('=' * 72)
    print('Transolver-Modal — 模态参数预测 + 物理 FRF 重建')
    print('=' * 72)
    print(f"Device: {cli.device}, Batch: {cli.batch_size}, data_dir={cli.data_dir}")

    print('\n--- Step 1: DataLoader ---')
    trainset, trainloader = make_loader(cli.data_dir, 'train.h5', cli.batch_size, True, not cli.no_edges)
    valset, valloader = make_loader(cli.data_dir, 'val.h5', 1, False, not cli.no_edges)
    testset, testloader = make_loader(cli.data_dir, 'test.h5', 1, False, not cli.no_edges)
    print(f"  Train: {len(trainset)} samples, {len(trainloader)} batches")
    first = trainset[0]
    print(f"  nodes[0]: {first['points'].shape}, node_features={first['node_features'].shape}")

    in_dim = first['node_features'].shape[1]
    n_modes = first['modal_omega'].shape[0]
    model_cfg = {
        'encoder_kwargs': {
            'in_dim': in_dim,
            'hidden_dim': cli.hidden_dim,
            'n_layers': cli.layers,
            'n_heads': cli.heads,
            'n_slices': cli.slices,
            'dropout': cli.dropout,
            'n_modes': n_modes,
            'use_edge_stem': not cli.no_edges,
            'amp_scale': 500000.0,
            'response_direction': cli.response_dir,
            'force_direction': cli.force_dir,
        },
        'decoder_kwargs': {},
    }

    print('\n--- Step 2: Model ---')
    net = build_geometric_model(model_cfg['encoder_kwargs'], model_cfg['decoder_kwargs']).to(cli.device)
    print(f"  Params: {sum(p.numel() for p in net.parameters()):,}")

    print('\n--- Step 3: Forward test ---')
    batch = next(iter(trainloader))
    batch_dev = {k: (v.to(cli.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    net.eval()
    with torch.no_grad():
        out = net(
            points=batch_dev['points'],
            node_features=batch_dev['node_features'],
            batch=batch_dev['batch'],
            edge_index=batch_dev.get('edge_index'),
            boundary_c_xyz=batch_dev.get('boundary_c_xyz'),
            excitation_index=batch_dev.get('excitation_index'),
            frequencies=batch_dev.get('frequencies'),
            node_counts=batch_dev.get('node_counts'),
            num_graphs=batch_dev.get('num_graphs'),
        )
        init_loss, init_logs = modal_loss(out, batch_dev, {'omega': 1.0, 'zeta': 10.0, 'phi': 3.0})
    print(f"  FRF={None if out['frf'] is None else list(out['frf'].shape)}, omega={list(out['modal_omega'].shape)}, phi={list(out['modal_phi_xyz'].shape)}")
    print(f"  init modal loss={init_loss.item():.2f}")
    print(f"  ω pred[0] Hz: {[f'{x/(2*torch.pi):.0f}' for x in out['modal_omega'][0].tolist()]}")
    print(f"  ω true[0] Hz: {[f'{x/(2*torch.pi):.0f}' for x in batch_dev['modal_omega'][0].tolist()]}")

    print('\n--- Step 4: Train ---')
    args = Args()
    args.device = cli.device
    args.fp16 = cli.fp16
    args.dir = cli.output_dir
    os.makedirs(args.dir, exist_ok=True)

    optimizer = torch.optim.AdamW(net.parameters(), **config['optimizer']['kwargs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    start_epoch = 0
    ckpt_last = os.path.join(args.dir, 'checkpoint_last')
    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location=cli.device)
        net.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        print(f"  Resume from epoch {start_epoch}")

    t0 = time.time()
    net = train(args, config, model_cfg, net, trainloader, optimizer, valloader, scheduler, logger=None, start_epoch=start_epoch)
    elapsed = time.time() - t0
    print(f"  Done, {elapsed:.0f}s")

    print('\n--- Step 5: Test checkpoints ---')
    for ckpt_name in ['checkpoint_best_modal', 'checkpoint_best', 'checkpoint_last']:
        path = os.path.join(args.dir, ckpt_name)
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=cli.device)
        net.load_state_dict(ckpt['model_state_dict'])
        print(f"\n[{ckpt_name}] epoch={ckpt.get('epoch', 'NA')}, loss={ckpt.get('loss', -1)}")
        evaluate(args, config, net, testloader, verbose=True, phase1=False)


if __name__ == '__main__':
    main()
