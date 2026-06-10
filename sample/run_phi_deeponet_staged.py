"""Three-stage DeepONet phi training with optional coeff distillation.

Stage1: train phi trunk + per-sample coeff table.
Stage2: freeze trunk, train encoder/branch/omega/zeta; warm up branch by distilling coeff table.
Stage3: unfreeze all and finetune.
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
import torch.nn.functional as F

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from training.losses import phi_1d_shape_scale_loss, sign_invariant_mse, total_loss
from training.trainer import TransolverTrainer, move_batch_to_device
from utils.direction import DEFAULT_FORCE_DIRECTION, DEFAULT_RESPONSE_DIRECTION, direction_to_frf_label


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


def parse_args():
    p = argparse.ArgumentParser(description='Staged DeepONet phi training')
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    p.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output_phi_staged'))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--hidden-dim', type=int, default=192)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--slices', type=int, default=48)
    p.add_argument('--phi-rank', type=int, default=64)
    p.add_argument('--stage1-epochs', type=int, default=80)
    p.add_argument('--stage2-epochs', type=int, default=120)
    p.add_argument('--stage3-epochs', type=int, default=40)
    p.add_argument('--lr-stage1', type=float, default=1e-4)
    p.add_argument('--lr-stage2', type=float, default=1e-4)
    p.add_argument('--lr-stage3', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--branch-init-clamp', type=float, default=20.0)
    p.add_argument('--coeff-distill-epochs', type=int, default=25)
    p.add_argument('--coeff-distill-weight', type=float, default=0.2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--no-edges', action='store_true')
    p.add_argument('--response-dir', default=DEFAULT_RESPONSE_DIRECTION, choices=['X', 'Y', 'Z'])
    p.add_argument('--force-dir', default=DEFAULT_FORCE_DIRECTION, choices=['X', 'Y', 'Z'])
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data_dir, filename, batch_size, shuffle, use_edges):
    ds = TransolverModalDataset([filename], data_dir=data_dir, use_edges=use_edges)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=4, pin_memory=torch.cuda.is_available(), collate_fn=collate_mesh_batch)
    return ds, dl


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def sample_map(ds: TransolverModalDataset):
    return {(path, group): i for i, (path, group) in enumerate(ds.samples)}


def sample_ids(batch: Dict, smap: Dict[Tuple[str, str], int], device: str):
    ids = [smap[(p, g)] for p, g in zip(batch['sample_path'], batch['sample_group'])]
    return torch.tensor(ids, dtype=torch.long, device=device)


def save_ckpt(path, net, opt, sched, epoch, metrics, best=None, coeff_table=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict() if opt is not None else None,
        'scheduler_state_dict': sched.state_dict() if sched is not None else None,
        'metrics': metrics,
        'best': best,
    }
    if coeff_table is not None:
        obj['coeff_table_state_dict'] = coeff_table.state_dict()
    torch.save(obj, path)


def sanitize_coeff(w: torch.Tensor, clamp_value: float):
    w = w.detach().float().cpu()
    w = torch.nan_to_num(w, nan=0.0, posinf=float(clamp_value), neginf=-float(clamp_value))
    return w.clamp(-float(clamp_value), float(clamp_value))


def init_branch_from_coeff(net: nn.Module, coeff_weight: torch.Tensor, clamp_value: float):
    w = sanitize_coeff(coeff_weight, clamp_value)
    mean = w.mean(dim=0)
    final = net.phi_head.branch[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError('phi_head.branch[-1] must be nn.Linear')
    with torch.no_grad():
        nn.init.normal_(final.weight, mean=0.0, std=1e-5)
        final.bias.copy_(mean.to(final.bias.device, dtype=final.bias.dtype))
    print(f'[stage2 init] branch bias <- coeff mean | mean_abs={mean.abs().mean():.4f}, std={w.std():.4f}, max_abs={w.abs().max():.4f}')


def load_stage1(args, net):
    path = os.path.join(args.output_dir, 'stage1_trunk', 'checkpoint_trunk')
    if not os.path.exists(path):
        print('[stage1 load] checkpoint_trunk not found')
        return net, None
    ckpt = torch.load(path, map_location=args.device)
    net.load_state_dict(ckpt['model_state_dict'], strict=False)
    coeff = (ckpt.get('coeff_table_state_dict') or {}).get('weight')
    print(f'[stage1 load] loaded {path}')
    return net, coeff


def stage1_loss(net, coeff_table, batch, ids, response_idx):
    ph = net.phi_head
    coeff = coeff_table(ids).view(ids.shape[0], ph.n_modes, 3, ph.rank)
    basis = ph.trunk_basis(batch['points'], batch['node_features'])
    phi = ph.combine(coeff, basis, batch['batch'])
    z = phi[..., response_idx]
    z_loss, _, per_mode, _ = phi_1d_shape_scale_loss(z, batch['modal_phi_response'], batch['node_counts'])
    xyz_loss = sign_invariant_mse(phi, batch['modal_phi_xyz'], batch['node_counts'], normalize=True)
    loss = 3.0 * z_loss + 0.5 * xyz_loss
    logs = {'loss': loss.detach(), 'z_shape': z_loss.detach(), 'xyz_shape': xyz_loss.detach(),
            'phi_k0': per_mode[0].detach(), 'phi_k1': per_mode[1].detach(), 'phi_k2': per_mode[2].detach()}
    return loss, logs


def train_stage1(args, net, loader, trainset, response_idx):
    out = os.path.join(args.output_dir, 'stage1_trunk')
    os.makedirs(out, exist_ok=True)
    smap = sample_map(trainset)
    ph = net.phi_head
    coeff_table = nn.Embedding(len(trainset), ph.n_modes * 3 * ph.rank).to(args.device)
    nn.init.normal_(coeff_table.weight, mean=0.0, std=0.02)
    set_requires_grad(net, False)
    set_requires_grad(net.phi_head.trunk, True)
    opt = torch.optim.AdamW(list(net.phi_head.trunk.parameters()) + list(coeff_table.parameters()), lr=args.lr_stage1, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.stage1_epochs, 1), eta_min=1e-6)
    start = 0
    last = os.path.join(out, 'checkpoint_last')
    if args.resume and os.path.exists(last):
        ckpt = torch.load(last, map_location=args.device)
        net.load_state_dict(ckpt['model_state_dict'], strict=False)
        coeff_table.load_state_dict(ckpt['coeff_table_state_dict'])
        if ckpt.get('optimizer_state_dict'):
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            sched.load_state_dict(ckpt['scheduler_state_dict'])
        start = int(ckpt.get('epoch', -1)) + 1
        print(f'[stage1 resume] from epoch {start}')
    if start >= args.stage1_epochs:
        torch.save({'model_state_dict': net.state_dict(), 'coeff_table_state_dict': coeff_table.state_dict()}, os.path.join(out, 'checkpoint_trunk'))
        return net, coeff_table.weight.detach().cpu()

    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_fp16)
    csv_path = os.path.join(out, 'stage1_log.csv')
    append = args.resume and os.path.exists(csv_path) and start > 0
    with open(csv_path, 'a' if append else 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        if not append:
            wr.writerow(['epoch','lr','loss','z_shape','xyz_shape','phi_k0','phi_k1','phi_k2'])
        for ep in range(start, args.stage1_epochs):
            net.train()
            sums = {}
            n = 0
            for batch in loader:
                batch = move_batch_to_device(batch, args.device)
                ids = sample_ids(batch, smap, args.device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=not args.no_fp16):
                    loss, logs = stage1_loss(net, coeff_table, batch, ids, response_idx)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(net.phi_head.trunk.parameters()) + list(coeff_table.parameters()), 1.0)
                scaler.step(opt)
                scaler.update()
                for k, v in logs.items():
                    sums[k] = sums.get(k, 0.0) + float(v.cpu())
                n += 1
            sched.step()
            row = {k: v / max(1, n) for k, v in sums.items()}
            lr = opt.param_groups[0]['lr']
            print(f"[stage1] epoch {ep:04d} | lr={lr:.1e} | loss={row.get('loss',0):.4f} z_shape={row.get('z_shape',0):.4f} xyz_shape={row.get('xyz_shape',0):.4f} phi_k=[{row.get('phi_k0',0)*100:.2f}% {row.get('phi_k1',0)*100:.2f}% {row.get('phi_k2',0)*100:.2f}%]")
            wr.writerow([ep, f'{lr:.1e}', round(row.get('loss',0),6), round(row.get('z_shape',0),6), round(row.get('xyz_shape',0),6), round(row.get('phi_k0',0)*100,4), round(row.get('phi_k1',0)*100,4), round(row.get('phi_k2',0)*100,4)])
            f.flush()
            if ep % 10 == 0 or ep == args.stage1_epochs - 1:
                save_ckpt(last, net, opt, sched, ep, row, coeff_table=coeff_table)
    torch.save({'model_state_dict': net.state_dict(), 'coeff_table_state_dict': coeff_table.state_dict()}, os.path.join(out, 'checkpoint_trunk'))
    return net, coeff_table.weight.detach().cpu()


def coeff_loss_norm(pred, target, eps=1e-6):
    p = pred.float()
    t = target.float().view_as(p)
    p = p / torch.sqrt(p.pow(2).mean(dim=(1,2,3), keepdim=True) + eps).clamp_min(eps)
    t = t / torch.sqrt(t.pow(2).mean(dim=(1,2,3), keepdim=True) + eps).clamp_min(eps)
    return F.smooth_l1_loss(p, t)


def branch_coeff_from_batch(net, batch):
    _, dense, mask = net.encode(batch['points'], batch['node_features'], edge_index=batch.get('edge_index'), node_counts=batch.get('node_counts'))
    glob = net.global_pool(dense, mask)
    if hasattr(net, 'global_feature_summary'):
        branch_features = net.global_feature_summary(batch['node_features'], batch.get('node_counts'))
        return net.phi_head.branch_coeff(glob, branch_features)
    return net.phi_head.branch_coeff(glob)


def train_epoch_supervised(trainer, loader, config, epoch, stage_name, args, coeff_weight, smap):
    trainer.model.train()
    sums = {}
    n = 0
    config['physics_alpha'] = min(1.0, epoch / max(config.get('physics_alpha_warmup', 50), 1))
    use_cd = stage_name == 'stage2_frozen_trunk' and coeff_weight is not None and smap is not None and epoch < args.coeff_distill_epochs and args.coeff_distill_weight > 0
    for batch in loader:
        batch = move_batch_to_device(batch, trainer.device)
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=trainer.fp16):
            outputs = trainer.forward_batch(batch, config)
            loss, logs = total_loss(outputs, batch, config)
            cd = loss.new_tensor(0.0)
            if use_cd:
                ids = sample_ids(batch, smap, trainer.device)
                target = coeff_weight[ids].view(ids.shape[0], trainer.model.phi_head.n_modes, 3, trainer.model.phi_head.rank)
                pred = branch_coeff_from_batch(trainer.model, batch)
                cd = coeff_loss_norm(pred, target)
                loss = loss + args.coeff_distill_weight * cd
        trainer.scaler.scale(loss).backward()
        if config.get('gradient_clip', 1.0) is not None:
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.get('gradient_clip', 1.0))
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        if torch.cuda.is_available() and str(trainer.device).startswith('cuda'):
            torch.cuda.synchronize()
        logs = dict(logs)
        logs['loss_coeff'] = cd.detach()
        logs['loss_total_with_coeff'] = loss.detach()
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().cpu())
        n += 1
    if trainer.scheduler is not None:
        trainer.scheduler.step()
    return {k: v / max(1, n) for k, v in sums.items()}


def train_supervised_stage(args, config, net, trainloader, trainset, valloader, stage_name, epochs, lr, freeze_trunk, coeff_weight=None):
    out = os.path.join(args.output_dir, stage_name)
    os.makedirs(out, exist_ok=True)
    set_requires_grad(net, True)
    if freeze_trunk:
        set_requires_grad(net.phi_head.trunk, False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1), eta_min=1e-6)
    trainer = TransolverTrainer(net, opt, device=args.device, scheduler=sched, fp16=not args.no_fp16)
    best = float('inf')
    start = 0
    last = os.path.join(out, 'checkpoint_last')
    if args.resume and os.path.exists(last):
        ckpt = torch.load(last, map_location=args.device)
        net.load_state_dict(ckpt['model_state_dict'], strict=False)
        if ckpt.get('optimizer_state_dict'):
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            sched.load_state_dict(ckpt['scheduler_state_dict'])
        best = float(ckpt.get('best', float('inf')) or float('inf'))
        start = int(ckpt.get('epoch', -1)) + 1
        print(f'[{stage_name} resume] from epoch {start}, best={best:.6f}')
    if start >= epochs:
        best_path = os.path.join(out, 'checkpoint_best')
        if os.path.exists(best_path):
            net.load_state_dict(torch.load(best_path, map_location=args.device)['model_state_dict'], strict=False)
        return net

    smap = sample_map(trainset) if coeff_weight is not None else None
    cd_weight = sanitize_coeff(coeff_weight, args.branch_init_clamp).to(args.device) if coeff_weight is not None else None
    if cd_weight is not None:
        print(f'[{stage_name}] coeff distill epochs={args.coeff_distill_epochs}, weight={args.coeff_distill_weight}, target={tuple(cd_weight.shape)}')
    csv_path = os.path.join(out, f'{stage_name}_log.csv')
    append = args.resume and os.path.exists(csv_path) and start > 0
    with open(csv_path, 'a' if append else 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        if not append:
            wr.writerow(['epoch','lr','train_total','train_total_with_coeff','train_omega','train_phi_resp','train_phi_xyz','coeff','phi_k0','phi_k1','phi_k2','val_total','omega_rel','zeta_rel','val_phi_resp','val_phi_xyz'])
        for ep in range(start, epochs):
            tr = train_epoch_supervised(trainer, trainloader, config, ep, stage_name, args, cd_weight, smap)
            lr_now = opt.param_groups[0]['lr']
            val = None
            if ep % config.get('validation_frequency', 5) == 0 or ep == epochs - 1:
                val = trainer.evaluate(valloader, config)
                metric = val.get('loss_total', val.get('loss_modal', float('inf')))
                save_ckpt(last, net, opt, sched, ep, val, best=best)
                if metric < best:
                    best = metric
                    save_ckpt(os.path.join(out, 'checkpoint_best'), net, opt, sched, ep, val, best=best)
            msg = f"[{stage_name}] epoch {ep:04d} | lr={lr_now:.1e} | total={tr.get('loss_total',0):.4f} coeff={tr.get('loss_coeff',0):.4f} omega={tr.get('loss_omega',0):.4f} phi_resp={tr.get('loss_phi_resp',0):.4f} phi_k=[{tr.get('phi_k0',0)*100:.2f}% {tr.get('phi_k1',0)*100:.2f}% {tr.get('phi_k2',0)*100:.2f}%]"
            if val is not None:
                msg += f" -> val={val.get('loss_total',0):.4f} omega_rel={val.get('omega_rel',0)*100:.2f}% phi_resp={val.get('loss_phi_resp',0):.4f}"
            print(msg)
            wr.writerow([ep, f'{lr_now:.1e}', round(float(tr.get('loss_total',0)),6), round(float(tr.get('loss_total_with_coeff',tr.get('loss_total',0))),6), round(float(tr.get('loss_omega',0)),6), round(float(tr.get('loss_phi_resp',0)),6), round(float(tr.get('loss_phi_xyz',0)),6), round(float(tr.get('loss_coeff',0)),6), round(float(tr.get('phi_k0',0))*100,4), round(float(tr.get('phi_k1',0))*100,4), round(float(tr.get('phi_k2',0))*100,4), round(float(val.get('loss_total',0)),6) if val else '', round(float(val.get('omega_rel',0)),6) if val else '', round(float(val.get('zeta_rel',0)),6) if val else '', round(float(val.get('loss_phi_resp',0)),6) if val else '', round(float(val.get('loss_phi_xyz',0)),6) if val else ''])
            f.flush()
    best_path = os.path.join(out, 'checkpoint_best')
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=args.device)['model_state_dict'], strict=False)
    return net


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    label = direction_to_frf_label(args.response_dir, args.force_dir)
    print('='*72)
    print(f'Staged DeepONet phi training ({label})')
    print('='*72)
    print(f'device={args.device} data={args.data_dir} output={args.output_dir} phi_rank={args.phi_rank}')
    print(f'epochs stage1={args.stage1_epochs} stage2={args.stage2_epochs} stage3={args.stage3_epochs} coeff_distill={args.coeff_distill_epochs}@{args.coeff_distill_weight} resume={args.resume}')
    trainset, trainloader = make_loader(args.data_dir, 'train.h5', args.batch_size, True, not args.no_edges)
    valset, valloader = make_loader(args.data_dir, 'val.h5', 1, False, not args.no_edges)
    testset, testloader = make_loader(args.data_dir, 'test.h5', 1, False, not args.no_edges)
    first = trainset[0]
    in_dim = first['node_features'].shape[1]
    n_modes = first['modal_omega'].shape[0]
    resp_idx = {'X':0,'Y':1,'Z':2}[args.response_dir.upper()]
    net = build_geometric_model({'in_dim': in_dim, 'hidden_dim': args.hidden_dim, 'n_layers': args.layers, 'n_heads': args.heads, 'n_slices': args.slices, 'n_modes': n_modes, 'use_edge_stem': not args.no_edges, 'amp_scale': 500000.0, 'response_direction': args.response_dir, 'force_direction': args.force_dir, 'phi_rank': args.phi_rank}, {}).to(args.device)
    print(f'train/val/test={len(trainset)}/{len(valset)}/{len(testset)} in_dim={in_dim} params={sum(p.numel() for p in net.parameters()):,}')
    t0 = time.time()
    coeff_weight = None
    if args.stage1_epochs > 0:
        net, coeff_weight = train_stage1(args, net, trainloader, trainset, resp_idx)
    else:
        net, coeff_weight = load_stage1(args, net)
    if args.stage2_epochs > 0:
        has_stage2 = os.path.exists(os.path.join(args.output_dir, 'stage2_frozen_trunk', 'checkpoint_last'))
        if args.resume and has_stage2:
            print('[stage2 init] resume stage2; skip branch mean init')
        elif coeff_weight is not None:
            init_branch_from_coeff(net, coeff_weight, args.branch_init_clamp)
        else:
            print('[stage2 init] no coeff_table; branch remains random')
        net = train_supervised_stage(args, cfg, net, trainloader, trainset, valloader, 'stage2_frozen_trunk', args.stage2_epochs, args.lr_stage2, True, coeff_weight)
    if args.stage3_epochs > 0:
        net = train_supervised_stage(args, cfg, net, trainloader, trainset, valloader, 'stage3_finetune', args.stage3_epochs, args.lr_stage3, False, None)
    if args.stage2_epochs > 0 or args.stage3_epochs > 0:
        trainer = TransolverTrainer(net, torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-6), device=args.device, fp16=not args.no_fp16)
        print(f'test metrics ({label}): {trainer.evaluate(testloader, cfg)}')
    print(f'done, time={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
