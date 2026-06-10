"""三阶段 DeepONet 振型训练入口。

阶段1：只训练 trunk basis + 每个训练样本的可学习 coeff_table。
阶段2：冻结 trunk，训练 Transolver encoder + branch/omega/zeta。
阶段3：解冻 trunk，小学习率整体微调。

该脚本不修改原有 loss，不依赖 dataset 返回 sample_index；直接用
batch['sample_path'] + batch['sample_group'] 映射到训练集索引。
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from training.losses import phi_1d_shape_scale_loss, sign_invariant_mse, total_loss
from training.trainer import TransolverTrainer, move_batch_to_device, save_checkpoint
from utils.direction import (
    DEFAULT_FORCE_DIRECTION,
    DEFAULT_RESPONSE_DIRECTION,
    direction_to_frf_label,
)


DEFAULT_CONFIG = {
    'validation_frequency': 5,
    'use_frf_loss': False,
    'frf_loss_weight': 1.0,
    'gradient_clip': 1.0,
    'physics_alpha_warmup': 999999,
    'modal_loss_weights': {
        'omega': 10.0,
        'zeta': 0.2,
        'phi_resp': 3.0,
        'phi_xyz': 0.5,
        'mac': 0.0,
        'phi_resp_scale_ratio': 0.0,
        'participation_ratio': 0.0,
        'phi_xyz_energy_ratio': 0.0,
    },
}


class Args:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='三阶段训练 DeepONet 振型头。')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output_phi_staged'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--slices', type=int, default=48)
    parser.add_argument('--phi-rank', type=int, default=64)
    parser.add_argument('--stage1-epochs', type=int, default=80)
    parser.add_argument('--stage2-epochs', type=int, default=120)
    parser.add_argument('--stage3-epochs', type=int, default=40)
    parser.add_argument('--lr-stage1', type=float, default=1e-4)
    parser.add_argument('--lr-stage2', type=float, default=1e-4)
    parser.add_argument('--lr-stage3', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mesh_batch,
    )
    return dataset, loader


def build_sample_index_map(dataset: TransolverModalDataset) -> Dict[Tuple[str, str], int]:
    return {(path, group): i for i, (path, group) in enumerate(dataset.samples)}


def batch_sample_ids(batch: Dict, sample_index_map: Dict[Tuple[str, str], int], device: str) -> torch.Tensor:
    ids = [sample_index_map[(path, group)] for path, group in zip(batch['sample_path'], batch['sample_group'])]
    return torch.tensor(ids, dtype=torch.long, device=device)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def stage1_phi_loss(net: nn.Module,
                    coeff_table: nn.Embedding,
                    batch: Dict,
                    sample_ids: torch.Tensor,
                    response_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    phi_head = net.phi_head
    points = batch['points']
    node_features = batch['node_features']
    batch_index = batch['batch']
    node_counts = batch['node_counts']

    coeff = coeff_table(sample_ids).view(sample_ids.shape[0], phi_head.n_modes, 3, phi_head.rank)
    basis = phi_head.trunk_basis(points, node_features)
    phi_xyz = phi_head.combine(coeff, basis, batch_index)

    phi_target = batch['modal_phi_xyz']
    phi_resp_target = batch['modal_phi_response']
    phi_resp = phi_xyz[..., response_idx]

    loss_z_shape, _, phi_per_mode, _ = phi_1d_shape_scale_loss(
        phi_resp, phi_resp_target, node_counts
    )
    loss_xyz_shape = sign_invariant_mse(phi_xyz, phi_target, node_counts, normalize=True)

    loss = 3.0 * loss_z_shape + 0.5 * loss_xyz_shape
    logs = {
        'loss': loss.detach(),
        'z_shape': loss_z_shape.detach(),
        'xyz_shape': loss_xyz_shape.detach(),
        'phi_k0': phi_per_mode[0].detach(),
        'phi_k1': phi_per_mode[1].detach(),
        'phi_k2': phi_per_mode[2].detach(),
    }
    return loss, logs


def train_stage1(args, net, trainloader, trainset, response_idx: int):
    """阶段1：只训练 trunk basis 和训练集 coeff_table。"""
    out_dir = os.path.join(args.output_dir, 'stage1_trunk')
    os.makedirs(out_dir, exist_ok=True)

    sample_index_map = build_sample_index_map(trainset)
    rank = net.phi_head.rank
    n_modes = net.phi_head.n_modes
    coeff_table = nn.Embedding(len(trainset), n_modes * 3 * rank).to(args.device)
    nn.init.normal_(coeff_table.weight, mean=0.0, std=0.02)

    # stage1 只让 trunk 和 coeff_table 训练，其他模块不动。
    set_requires_grad(net, False)
    set_requires_grad(net.phi_head.trunk, True)
    optimizer = torch.optim.AdamW(
        list(net.phi_head.trunk.parameters()) + list(coeff_table.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.stage1_epochs, 1), eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_fp16)

    csv_path = os.path.join(out_dir, 'stage1_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'loss', 'z_shape', 'xyz_shape', 'phi_k0', 'phi_k1', 'phi_k2'])

        for epoch in range(args.stage1_epochs):
            net.train()
            sums, count = {}, 0
            for batch in trainloader:
                batch = move_batch_to_device(batch, args.device)
                sample_ids = batch_sample_ids(batch, sample_index_map, args.device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=not args.no_fp16):
                    loss, logs = stage1_phi_loss(net, coeff_table, batch, sample_ids, response_idx)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(net.phi_head.trunk.parameters()) + list(coeff_table.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()

                for k, v in logs.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach().cpu())
                count += 1

            scheduler.step()
            row = {k: v / max(1, count) for k, v in sums.items()}
            lr = optimizer.param_groups[0]['lr']
            print(
                f"[stage1] epoch {epoch:04d} | lr={lr:.1e} | loss={row.get('loss', 0):.4f} "
                f"z_shape={row.get('z_shape', 0):.4f} xyz_shape={row.get('xyz_shape', 0):.4f} "
                f"phi_k=[{row.get('phi_k0', 0)*100:.2f}% {row.get('phi_k1', 0)*100:.2f}% {row.get('phi_k2', 0)*100:.2f}%]"
            )
            writer.writerow([
                epoch, f'{lr:.1e}',
                round(row.get('loss', 0), 6),
                round(row.get('z_shape', 0), 6),
                round(row.get('xyz_shape', 0), 6),
                round(row.get('phi_k0', 0) * 100, 4),
                round(row.get('phi_k1', 0) * 100, 4),
                round(row.get('phi_k2', 0) * 100, 4),
            ])
            f.flush()

            if epoch % 10 == 0 or epoch == args.stage1_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'coeff_table_state_dict': coeff_table.state_dict(),
                    'metrics': row,
                }, os.path.join(out_dir, 'checkpoint_last'))

    # 阶段1结束后，保留 trunk 权重即可；coeff_table 只用于学习 trunk，不用于新样本推理。
    torch.save({
        'model_state_dict': net.state_dict(),
        'coeff_table_state_dict': coeff_table.state_dict(),
    }, os.path.join(out_dir, 'checkpoint_trunk'))
    return net


def train_supervised_stage(args, config, net, trainloader, valloader, stage_name: str,
                           epochs: int, lr: float, freeze_trunk: bool):
    out_dir = os.path.join(args.output_dir, stage_name)
    os.makedirs(out_dir, exist_ok=True)

    set_requires_grad(net, True)
    if freeze_trunk:
        set_requires_grad(net.phi_head.trunk, False)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    stage_args = Args()
    stage_args.device = args.device
    stage_args.fp16 = not args.no_fp16
    stage_args.dir = out_dir

    trainer = TransolverTrainer(net, optimizer, device=args.device, scheduler=scheduler, fp16=not args.no_fp16)
    best = float('inf')

    csv_path = os.path.join(out_dir, f'{stage_name}_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_total', 'train_omega', 'train_phi_resp', 'train_phi_xyz',
                         'phi_k0', 'phi_k1', 'phi_k2', 'val_total', 'omega_rel', 'zeta_rel', 'val_phi_resp', 'val_phi_xyz'])

        for epoch in range(epochs):
            train_logs = trainer.train_epoch(trainloader, config, epoch)
            lr_now = optimizer.param_groups[0]['lr']

            val_logs = None
            if epoch % config.get('validation_frequency', 5) == 0 or epoch == epochs - 1:
                val_logs = trainer.evaluate(valloader, config)
                metric = val_logs.get('loss_total', val_logs.get('loss_modal', float('inf')))
                save_checkpoint(os.path.join(out_dir, 'checkpoint_last'), net, optimizer, epoch, val_logs)
                if metric < best:
                    best = metric
                    save_checkpoint(os.path.join(out_dir, 'checkpoint_best'), net, optimizer, epoch, val_logs)

            msg = (
                f"[{stage_name}] epoch {epoch:04d} | lr={lr_now:.1e} | "
                f"total={train_logs.get('loss_total', 0):.4f} "
                f"omega={train_logs.get('loss_omega', 0):.4f} "
                f"phi_resp={train_logs.get('loss_phi_resp', 0):.4f} "
                f"phi_k=[{train_logs.get('phi_k0', 0)*100:.2f}% "
                f"{train_logs.get('phi_k1', 0)*100:.2f}% {train_logs.get('phi_k2', 0)*100:.2f}%]"
            )
            if val_logs is not None:
                msg += (
                    f" -> val={val_logs.get('loss_total', 0):.4f} "
                    f"omega_rel={val_logs.get('omega_rel', 0)*100:.2f}% "
                    f"phi_resp={val_logs.get('loss_phi_resp', 0):.4f}"
                )
            print(msg)

            writer.writerow([
                epoch, f'{lr_now:.1e}',
                round(float(train_logs.get('loss_total', 0)), 6),
                round(float(train_logs.get('loss_omega', 0)), 6),
                round(float(train_logs.get('loss_phi_resp', 0)), 6),
                round(float(train_logs.get('loss_phi_xyz', 0)), 6),
                round(float(train_logs.get('phi_k0', 0)) * 100, 4),
                round(float(train_logs.get('phi_k1', 0)) * 100, 4),
                round(float(train_logs.get('phi_k2', 0)) * 100, 4),
                round(float(val_logs.get('loss_total', 0)), 6) if val_logs else '',
                round(float(val_logs.get('omega_rel', 0)), 6) if val_logs else '',
                round(float(val_logs.get('zeta_rel', 0)), 6) if val_logs else '',
                round(float(val_logs.get('loss_phi_resp', 0)), 6) if val_logs else '',
                round(float(val_logs.get('loss_phi_xyz', 0)), 6) if val_logs else '',
            ])
            f.flush()

    best_path = os.path.join(out_dir, 'checkpoint_best')
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=args.device)['model_state_dict'], strict=False)
    return net


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    frf_label = direction_to_frf_label(args.response_dir, args.force_dir)

    print('=' * 72)
    print(f'三阶段 DeepONet 振型训练 ({frf_label})')
    print('=' * 72)
    print(f'设备: {args.device}')
    print(f'数据: {args.data_dir}')
    print(f'输出: {args.output_dir}')
    print(f'phi_rank: {args.phi_rank}')
    print(f'epochs: stage1={args.stage1_epochs}, stage2={args.stage2_epochs}, stage3={args.stage3_epochs}')

    trainset, trainloader = make_loader(args.data_dir, 'train.h5', args.batch_size, True, not args.no_edges)
    valset, valloader = make_loader(args.data_dir, 'val.h5', 1, False, not args.no_edges)
    testset, testloader = make_loader(args.data_dir, 'test.h5', 1, False, not args.no_edges)

    first = trainset[0]
    in_dim = first['node_features'].shape[1]
    n_modes = first['modal_omega'].shape[0]
    response_idx = {'X': 0, 'Y': 1, 'Z': 2}[args.response_dir.upper()]
    print(f'训练/验证/测试: {len(trainset)}/{len(valset)}/{len(testset)} 样本')
    print(f'节点特征维度: {in_dim}, 模态阶数: {n_modes}, 首个网格节点数: {first["points"].shape[0]}')

    model_cfg = {
        'encoder_kwargs': {
            'in_dim': in_dim,
            'hidden_dim': args.hidden_dim,
            'n_layers': args.layers,
            'n_heads': args.heads,
            'n_slices': args.slices,
            'n_modes': n_modes,
            'use_edge_stem': not args.no_edges,
            'amp_scale': 500000.0,
            'response_direction': args.response_dir,
            'force_direction': args.force_dir,
            'phi_rank': args.phi_rank,
        },
        'decoder_kwargs': {},
    }
    net = build_geometric_model(model_cfg['encoder_kwargs'], model_cfg['decoder_kwargs']).to(args.device)
    print(f'模型参数: {sum(p.numel() for p in net.parameters()):,}')

    t0 = time.time()

    if args.stage1_epochs > 0:
        net = train_stage1(args, net, trainloader, trainset, response_idx)

    if args.stage2_epochs > 0:
        net = train_supervised_stage(
            args, config, net, trainloader, valloader,
            stage_name='stage2_frozen_trunk',
            epochs=args.stage2_epochs,
            lr=args.lr_stage2,
            freeze_trunk=True,
        )

    if args.stage3_epochs > 0:
        net = train_supervised_stage(
            args, config, net, trainloader, valloader,
            stage_name='stage3_finetune',
            epochs=args.stage3_epochs,
            lr=args.lr_stage3,
            freeze_trunk=False,
        )

    # 最终测试用 stage3 best；如果没有 stage3，就用 stage2 best；如果只有 stage1，则只保存 trunk，不做测试。
    if args.stage2_epochs > 0 or args.stage3_epochs > 0:
        eval_args = Args()
        eval_args.device = args.device
        eval_args.fp16 = not args.no_fp16
        eval_args.dir = args.output_dir
        trainer = TransolverTrainer(net, torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-6),
                                    device=args.device, fp16=not args.no_fp16)
        metrics = trainer.evaluate(testloader, config)
        print(f'测试指标 ({frf_label}): {metrics}')

    print(f'训练完成, 耗时 {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
