"""Transolver-Modal 训练与评估循环。

该文件不再使用旧版错误 trainer，而是对齐当前 CNN 中已经验证正确的训练策略：
Phase0 频率专属预训练 → Phase1 全模态训练 → Phase2 FRF 弱约束；
同时每次验证都打印验证集三阶 w/z/MAC/φn/φa，并用验证集模态指标保存 checkpoint_best_modal。
"""
from __future__ import annotations

import csv
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .losses import modal_loss, frf_loss


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _log(msg: str, logger=None) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _forward_model(net, batch: Dict, use_frf: bool, omega_true: torch.Tensor | None = None):
    return net(
        points=batch['points'],
        node_features=batch['node_features'],
        batch=batch['batch'],
        edge_index=batch.get('edge_index'),
        boundary_c_xyz=batch.get('boundary_c_xyz'),
        excitation_index=batch.get('excitation_index'),
        frequencies=batch.get('frequencies') if use_frf else None,
        num_graphs=batch.get('num_graphs'),
        node_counts=batch.get('node_counts'),
        omega_true=omega_true,
    )


def _mean_logs(log_list: list[Dict[str, torch.Tensor | float]]) -> Dict[str, float]:
    sums, counts = {}, {}
    for logs in log_list:
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu())
            elif isinstance(v, (int, float, np.floating)):
                v = float(v)
            else:
                continue
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / max(1, counts[k]) for k in sums}


def _apply_gradient_clip(net, optimizer, scaler, config, fp16: bool):
    grad_clip = config.get('optimizer', {}).get('gradient_clip', config.get('gradient_clip', 2.0))
    if grad_clip is None:
        return
    if fp16:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))


def save_model(savepath: str, epoch: int, model, optimizer, loss, name: str = "checkpoint_best"):
    os.makedirs(savepath, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': float(loss) if isinstance(loss, (int, float, np.floating)) else loss,
    }, os.path.join(savepath, name))


def _format_val_modal_row(val_results: Dict) -> list[str]:
    if not val_results or 'val_w_rel_per_mode' not in val_results:
        return [''] * 15
    vw = val_results['val_w_rel_per_mode']
    vz = val_results['val_z_rel_per_mode']
    vm = val_results['val_mac_per_mode']
    vpn = val_results['val_phi_n_per_mode']
    vpa = val_results['val_phi_a_per_mode']
    return [
        f'{vw[0]:.3f}', f'{vw[1]:.3f}', f'{vw[2]:.3f}',
        f'{vz[0]:.1f}', f'{vz[1]:.1f}', f'{vz[2]:.1f}',
        f'{vm[0]:.3f}', f'{vm[1]:.3f}', f'{vm[2]:.3f}',
        f'{vpn[0]:.1f}', f'{vpn[1]:.1f}', f'{vpn[2]:.1f}',
        f'{vpa[0]:.1f}', f'{vpa[1]:.1f}', f'{vpa[2]:.1f}',
    ]


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------

def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler=None, logger=None, start_epoch: int = 0):
    total_epochs = int(config.get('epochs', 900))
    frf_weight = float(config.get('frf_loss_weight', 0.02))
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(args, 'fp16', False))
    os.makedirs(args.dir, exist_ok=True)

    log_path = os.path.join(args.dir, 'loss_log.csv')
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if not log_exists:
        val_header = [
            'val_w1%', 'val_w2%', 'val_w3%',
            'val_z1%', 'val_z2%', 'val_z3%',
            'val_MAC1', 'val_MAC2', 'val_MAC3',
            'val_phiN1%', 'val_phiN2%', 'val_phiN3%',
            'val_phiA1%', 'val_phiA2%', 'val_phiA3%',
        ]
        log_writer.writerow([
            '轮次', '训练损失',
            'w1%', 'w2%', 'w3%',
            'z1%', 'z2%', 'z3%',
            'φloss', 'φn1', 'φn2', 'φn3', 'φa1', 'φa2', 'φa3',
            'MAC1', 'MAC2', 'MAC3',
            'w占比%', 'z占比%', 'phi占比%', 'FRF占比%',
            '验证MSE', '幅值MAE', '幅值MAPE%',
            'FRFraw', '学习率',
        ] + val_header)

    omega_pretrain_epochs = int(config.get('omega_pretrain_epochs', 50))
    phase2_min_epoch = int(config.get('phase2_min_epoch', 300))
    enable_phase2 = bool(config.get('enable_phase2', True))
    phase2_unlocked = start_epoch > phase2_min_epoch
    unlock_epoch = phase2_min_epoch if phase2_unlocked else start_epoch
    lowest = np.inf
    lowest_modal = np.inf

    try:
        for epoch in range(start_epoch, total_epochs):
            net.train()
            in_phase0 = epoch < omega_pretrain_epochs
            in_phase2 = phase2_unlocked
            in_phase1 = not phase2_unlocked

            if epoch == 0:
                _log("=== 阶段0: 频率专属预训练 (仅训频率) ===", logger)
            if epoch == omega_pretrain_epochs:
                _log("=== 阶段1: 全模态联合训练 (ω/ζ/φ) ===", logger)

            if in_phase0:
                current_weights = {
                    'omega': config.get('omega_loss_weight', 1.0) * 5.0,
                    'zeta': 0.0,
                    'phi': 0.0,
                }
            else:
                current_weights = {
                    'omega': config.get('omega_loss_weight', 1.0),
                    'zeta': config.get('zeta_loss_weight', 10.0),
                    'phi': config.get('phi_loss_weight', 3.0),
                }

            batch_logs = []
            losses = []
            frf_raw_values = []

            for batch in dataloader:
                batch = move_batch_to_device(batch, args.device)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=getattr(args, 'fp16', False)):
                    omega_true = None
                    use_frf = False
                    current_frf_w = 0.0
                    if in_phase2:
                        phase2_epoch = epoch - unlock_epoch
                        teacher_epochs = int(config.get('frf_teacher_epochs', 50))
                        omega_true = batch['modal_omega'] if phase2_epoch < teacher_epochs else None
                        use_frf = True
                        warm = int(config.get('frf_warmup_epochs', 20))
                        current_frf_w = frf_weight * min(1.0, phase2_epoch / max(warm, 1))

                    outputs = _forward_model(net, batch, use_frf=use_frf, omega_true=omega_true)
                    loss_m, logs = modal_loss(outputs, batch, current_weights)
                    loss = loss_m
                    raw_frf = outputs['modal_omega'].new_tensor(0.0)
                    if use_frf and outputs.get('frf') is not None:
                        raw_frf, frf_logs = frf_loss(outputs['frf'], batch['point_frf'])
                        logs.update(frf_logs)
                        loss = loss + current_frf_w * raw_frf

                losses.append(float(loss.detach().cpu()))
                logs['loss_total'] = loss.detach()
                logs['loss_frf_raw'] = raw_frf.detach()
                batch_logs.append(logs)
                frf_raw_values.append(float(raw_frf.detach().cpu()))

                scaler.scale(loss).backward()
                _apply_gradient_clip(net, optimizer, scaler, config, getattr(args, 'fp16', False))
                scaler.step(optimizer)
                scaler.update()

            if scheduler is not None:
                scheduler.step()

            train_logs = _mean_logs(batch_logs)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            lr = optimizer.param_groups[0]['lr']

            w = np.array([train_logs.get(f'omega_k{k}', 0.0) * 100.0 for k in range(3)])
            z = np.array([train_logs.get(f'zeta_k{k}', 0.0) * 100.0 for k in range(3)])
            mac = np.array([train_logs.get(f'mac_k{k}', 0.0) for k in range(3)])
            phi_n = np.array([train_logs.get(f'phi_n_k{k}', 0.0) for k in range(3)])
            phi_a = np.array([train_logs.get(f'phi_a_k{k}', 0.0) for k in range(3)])

            w_loss = train_logs.get('loss_omega', 0.0)
            z_loss = train_logs.get('loss_zeta', 0.0)
            p_loss = train_logs.get('loss_phi', 0.0)
            f_loss = frf_weight * train_logs.get('loss_frf_raw', 0.0) if in_phase2 else 0.0
            denom = max(w_loss + z_loss + p_loss + f_loss, 1e-12)
            w_share, z_share, p_share, f_share = [x / denom * 100.0 for x in (w_loss, z_loss, p_loss, f_loss)]

            _log(
                f"Epoch {epoch:4d} | "
                f"w=[{w[0]:.1f}/{w[1]:.1f}/{w[2]:.1f}]% "
                f"z=[{z[0]:.0f}/{z[1]:.0f}/{z[2]:.0f}]% "
                f"φn=[{phi_n[0]:.1f}/{phi_n[1]:.1f}/{phi_n[2]:.1f}]% "
                f"φa=[{phi_a[0]:.1f}/{phi_a[1]:.1f}/{phi_a[2]:.1f}]% "
                f"MAC=[{mac[0]:.3f}/{mac[1]:.3f}/{mac[2]:.3f}] | "
                f"w{w_share:.0f}z{z_share:.0f}ph{p_share:.0f}frf{f_share:.0f} | "
                f"loss={mean_loss:.1f}",
                logger,
            )

            # Phase2 动态解锁在 epoch 末尾触发，与当前 CNN 训练逻辑一致。
            if not phase2_unlocked and enable_phase2 and epoch >= phase2_min_epoch:
                phase2_unlocked = True
                unlock_epoch = epoch
                _log(f">>> Phase2 unlocked at epoch {epoch} (FRF weak constraint) <<<", logger)

            val_results = None
            val_freq = int(config.get('validation_frequency', 5))
            if epoch % val_freq == 0 or epoch == total_epochs - 1:
                save_model(args.dir, epoch, net, optimizer, mean_loss, 'checkpoint_last')
                val_results = evaluate(args, config, net, valloader, logger, epoch, verbose=True, phase1=not phase2_unlocked)

                if in_phase1:
                    best_metric = val_results.get('ω_MAE (rad/s)', val_results.get('val_w_rel_mean', np.inf))
                    metric_name = 'val_ω_MAE'
                else:
                    best_metric = val_results.get('loss (MSE)', val_results.get('val_w_rel_mean', np.inf))
                    metric_name = 'val_loss'
                if best_metric < lowest:
                    _log(f"best model ({metric_name}={best_metric:.6f})", logger)
                    save_model(args.dir, epoch, net, optimizer, best_metric, 'checkpoint_best')
                    lowest = best_metric

                if epoch >= omega_pretrain_epochs and 'val_w_rel_mean' in val_results:
                    val_modal_score = (
                        val_results['val_w_rel_mean']
                        + 0.3 * val_results['val_z_rel_mean']
                        + (1.0 - val_results['val_mac_mean']) * 100.0
                        + 0.05 * val_results['val_phi_a_mean']
                    )
                    if val_modal_score < lowest_modal:
                        _log(f"best val modal model (val_modal_score={val_modal_score:.4f})", logger)
                        save_model(args.dir, epoch, net, optimizer, val_modal_score, 'checkpoint_best_modal')
                        lowest_modal = val_modal_score

            row = [
                epoch, f'{mean_loss:.2e}',
                f'{w[0]:.3f}', f'{w[1]:.3f}', f'{w[2]:.3f}',
                f'{z[0]:.1f}', f'{z[1]:.1f}', f'{z[2]:.1f}',
                f'{p_loss:.2f}',
                f'{phi_n[0]:.1f}', f'{phi_n[1]:.1f}', f'{phi_n[2]:.1f}',
                f'{phi_a[0]:.1f}', f'{phi_a[1]:.1f}', f'{phi_a[2]:.1f}',
                f'{mac[0]:.3f}', f'{mac[1]:.3f}', f'{mac[2]:.3f}',
                f'{w_share:.1f}', f'{z_share:.1f}', f'{p_share:.1f}', f'{f_share:.1f}',
                f'{val_results.get("loss (MSE)", 0):.4f}' if val_results else '',
                f'{val_results.get("Amplitude MAE", 0):.4f}' if val_results else '',
                f'{val_results.get("Amplitude MAPE (%)", 0):.2f}' if val_results else '',
                f'{np.mean(frf_raw_values):.4f}', f'{lr:.2e}',
            ] + (_format_val_modal_row(val_results) if val_results else [''] * 15)
            log_writer.writerow(row)
            log_file.flush()

    finally:
        log_file.close()

    return net


# ---------------------------------------------------------------------------
# 验证/测试
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True, phase1=False):
    was_training = net.training
    net.eval()
    device = args.device

    modal_logs = []
    frf_preds, frf_targets = [], []
    omega_abs_errs = []

    try:
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = _forward_model(net, batch, use_frf=not phase1, omega_true=None)
            _, logs = modal_loss(outputs, batch, {'omega': 1.0, 'zeta': 1.0, 'phi': 1.0})
            modal_logs.append(logs)
            omega_abs_errs.append(torch.abs(outputs['modal_omega'] - batch['modal_omega']).detach().cpu())
            if not phase1 and outputs.get('frf') is not None:
                frf_preds.append(outputs['frf'].detach().cpu())
                frf_targets.append(batch['point_frf'].detach().cpu())

        logs = _mean_logs(modal_logs)
        results: Dict[str, object] = {}
        omega_abs = torch.cat([e.flatten() for e in omega_abs_errs]) if omega_abs_errs else torch.tensor([])
        if omega_abs.numel() > 0:
            results['ω_MAE (rad/s)'] = float(omega_abs.mean())

        val_w = np.array([logs.get(f'omega_k{k}', 0.0) * 100.0 for k in range(3)])
        val_z = np.array([logs.get(f'zeta_k{k}', 0.0) * 100.0 for k in range(3)])
        val_mac = np.array([logs.get(f'mac_k{k}', 0.0) for k in range(3)])
        val_phi_n = np.array([logs.get(f'phi_n_k{k}', 0.0) for k in range(3)])
        val_phi_a = np.array([logs.get(f'phi_a_k{k}', 0.0) for k in range(3)])

        results['val_w_rel_per_mode'] = val_w
        results['val_z_rel_per_mode'] = val_z
        results['val_mac_per_mode'] = val_mac
        results['val_phi_n_per_mode'] = val_phi_n
        results['val_phi_a_per_mode'] = val_phi_a
        results['val_w_rel_mean'] = float(val_w.mean())
        results['val_z_rel_mean'] = float(val_z.mean())
        results['val_mac_mean'] = float(val_mac.mean())
        results['val_phi_n_mean'] = float(val_phi_n.mean())
        results['val_phi_a_mean'] = float(val_phi_a.mean())

        if frf_preds:
            pred = torch.cat(frf_preds, dim=0)
            target = torch.cat(frf_targets, dim=0)
            if pred.shape != target.shape:
                target = target.reshape(pred.shape)
            results['loss (MSE)'] = float(F.mse_loss(pred, target))
            p_amp = torch.linalg.norm(pred, dim=-1)
            t_amp = torch.linalg.norm(target, dim=-1)
            results['Amplitude MAE'] = float(F.l1_loss(p_amp, t_amp))
            results['Amplitude MAPE (%)'] = float((torch.abs(p_amp - t_amp) / (t_amp + 1e-6)).mean() * 100.0)

        if verbose:
            if 'loss (MSE)' in results:
                _log(
                    f"Val FRF | MSE={results['loss (MSE)']:.4f} "
                    f"MAE={results['Amplitude MAE']:.4f} "
                    f"MAPE={results['Amplitude MAPE (%)']:.2f}%",
                    logger,
                )
            _log(
                f"Val modal | "
                f"w=[{val_w[0]:.3f}/{val_w[1]:.3f}/{val_w[2]:.3f}]% "
                f"z=[{val_z[0]:.1f}/{val_z[1]:.1f}/{val_z[2]:.1f}]% "
                f"MAC=[{val_mac[0]:.3f}/{val_mac[1]:.3f}/{val_mac[2]:.3f}] "
                f"φn=[{val_phi_n[0]:.1f}/{val_phi_n[1]:.1f}/{val_phi_n[2]:.1f}]% "
                f"φa=[{val_phi_a[0]:.1f}/{val_phi_a[1]:.1f}/{val_phi_a[2]:.1f}]%",
                logger,
            )
        return results
    finally:
        if was_training:
            net.train()
