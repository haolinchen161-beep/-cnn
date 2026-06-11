"""训练 Transolver 模态-FRF 模型（自动化两阶段课程学习版）。

用法:
    python sample/run_validation.py
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

# =================================================================
# 基础配置 (Phase 1 预训练状态)
# =================================================================
DEFAULT_CONFIG = {
    'validation_frequency': 5,
    'gradient_clip': 1.0,
    'physics_alpha_warmup': 50,

    # 默认开启 Phase 1 状态：关闭 FRF
    'use_frf_loss': False,
    'frf_loss_weight': 1.0,

    'modal_loss_weights': {
        'omega': 20.0,              # 极高权重，强迫网络咬死刚度退化
        'zeta': 1.0,
        'phi_resp': 1.0,            # 响应方向振型复合损失
        'phi_xyz': 0.5,             # 三向振型复合损失
        'mac': 0.05,
        # 内部比例项
        'phi_resp_scale_ratio': 0.5,
        'participation_ratio': 0.5,
        'phi_xyz_energy_ratio': 0.1,
    },

    'optimizer': {
        'lr': 1e-4,                 # Phase 1 初始学习率
        'weight_decay': 1e-4,
    },
}

class Args:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='自动化两阶段训练 Transolver。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    # 课程学习阶段控制
    parser.add_argument('--phase1-epochs', type=int, default=150, help='阶段1(纯模态预训练)的轮数')
    parser.add_argument('--total-epochs', type=int, default=200, help='总训练轮数(包含阶段2 FRF微调)')
    
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--slices', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-fp16', action='store_true')
    parser.add_argument('--no-edges', action='store_true')
    parser.add_argument('--response-dir', default=DEFAULT_RESPONSE_DIRECTION, choices=['X', 'Y', 'Z'])
    parser.add_argument('--force-dir', default=DEFAULT_FORCE_DIRECTION, choices=['X', 'Y', 'Z'])
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
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=4, pin_memory=torch.cuda.is_available(), collate_fn=collate_mesh_batch,
    )
    return dataset, loader

def main():
    cli = parse_args()
    set_seed(cli.seed)
    os.makedirs(cli.output_dir, exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    frf_label = direction_to_frf_label(cli.response_dir, cli.force_dir)

    print('=' * 72)
    print(f'Transolver 自动化两阶段训练引擎 ({frf_label})')
    print('=' * 72)

    trainset, trainloader = make_loader(cli.data_dir, 'train.h5', cli.batch_size, True, not cli.no_edges)
    valset, valloader = make_loader(cli.data_dir, 'val.h5', 1, False, not cli.no_edges)
    testset, testloader = make_loader(cli.data_dir, 'test.h5', 1, False, not cli.no_edges)

    in_dim = trainset[0]['node_features'].shape[1]
    n_modes = trainset[0]['modal_omega'].shape[0]

    model_cfg = {
        'encoder_kwargs': {
            'in_dim': in_dim, 'hidden_dim': cli.hidden_dim, 'n_layers': cli.layers,
            'n_heads': cli.heads, 'n_slices': cli.slices, 'n_modes': n_modes,
            'use_edge_stem': not cli.no_edges, 'amp_scale': 500000.0,
            'response_direction': cli.response_dir, 'force_direction': cli.force_dir,
        },
        'decoder_kwargs': {},
    }
    net = build_geometric_model(model_cfg['encoder_kwargs'], model_cfg['decoder_kwargs']).to(cli.device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])

    args = Args()
    args.device = cli.device
    args.fp16 = not cli.no_fp16
    args.dir = cli.output_dir

    # ---------------------------------------------------------
    # 断点续训逻辑判断
    # ---------------------------------------------------------
    start_epoch = 0
    ckpt_last = os.path.join(cli.output_dir, 'checkpoint_last')
    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location=cli.device)
        net.load_state_dict(ckpt['model_state_dict'], strict=False)
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        print(f'[状态恢复] 从 epoch {start_epoch} 继续训练...')

    t0 = time.time()

    # =========================================================
    # Phase 1: 纯模态预训练 (频率霸权)
    # =========================================================
    if start_epoch < cli.phase1_epochs:
        print(f"\n>>> 启动 Phase 1: 纯模态参数预训练 (目标: 压制频率误差) <<<")
        print(f"当前策略: 关闭 FRF 损失, Omega 权重 = {config['modal_loss_weights']['omega']}")
        
        config['epochs'] = cli.phase1_epochs
        config['use_frf_loss'] = False
        
        # Scheduler 负责 Phase 1 的学习率退火
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.phase1_epochs, eta_min=1e-6)
        # 恢复 Scheduler 状态（如果是在 Phase 1 中断）
        if os.path.exists(ckpt_last) and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        net = train(args, config, model_cfg, net, trainloader, optimizer, valloader, scheduler, start_epoch=start_epoch)
        
        start_epoch = cli.phase1_epochs  # Phase 1 结束，更新进度

    # =========================================================
    # Phase 2: 解开封印，FRF 联合微调
    # =========================================================
    if start_epoch >= cli.phase1_epochs and start_epoch < cli.total_epochs:
        print(f"\n>>> 启动 Phase 2: 自动转段！开启 FRF 联合微调 <<<")
        print(f"当前策略: 开启 FRF, 学习率骤降至 1e-5, 退隐 Omega 权重")
        
        config['epochs'] = cli.total_epochs
        config['use_frf_loss'] = True
        
        # 调整权重：让 FRF 接管主导权，削弱 omega 的霸权
        config['modal_loss_weights']['omega'] = 5.0
        
        # 强制将学习率重置为 1e-5 (微调量级)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5

        # 为剩余的轮数重新生成一个平滑的 Scheduler
        phase2_epochs = cli.total_epochs - cli.phase1_epochs
        scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)
        
        # 恢复 Scheduler 状态（如果是在 Phase 2 中断）
        if os.path.exists(ckpt_last) and start_epoch > cli.phase1_epochs and 'scheduler_state_dict' in ckpt:
            scheduler_phase2.load_state_dict(ckpt['scheduler_state_dict'])

        net = train(args, config, model_cfg, net, trainloader, optimizer, valloader, scheduler_phase2, start_epoch=start_epoch)

    # ---------------------------------------------------------
    # 最终测试与评估
    # ---------------------------------------------------------
    elapsed = time.time() - t0
    best_path = os.path.join(cli.output_dir, 'checkpoint_best')
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=cli.device)['model_state_dict'], strict=False)
    metrics = evaluate(args, config, net, testloader, verbose=True)
    
    print(f'\n=========================================================')
    print(f'训练总耗时: {elapsed:.1f}s')
    print(f'最终测试指标 ({frf_label}):')
    for k, v in metrics.items():
        print(f"  - {k}: {v:.6f}")
    print(f'=========================================================')

if __name__ == '__main__':
    main()
