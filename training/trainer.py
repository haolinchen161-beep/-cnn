"""Training and evaluation loop for Transolver modal-FRF model."""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .losses import total_loss


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


class TransolverTrainer:
    """Small trainer for variable-size mesh batches."""

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu',
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 fp16: bool = False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.fp16 = fp16

    def forward_batch(self, batch: Dict, config: Dict | None = None) -> Dict[str, torch.Tensor]:
        # 根据 use_frf_loss 决定是否传入 frequencies（节省计算）
        use_frf = True if config is None else config.get('use_frf_loss', False)
        # 渐进式物理融合权重：训练时由 epoch 计算，评估时默认 1.0（纯物理）
        physics_alpha = 1.0 if config is None else config.get('physics_alpha', 1.0)

        return self.model(
            points=batch['points'],
            node_features=batch['node_features'],
            batch=batch['batch'],
            edge_index=batch.get('edge_index'),
            boundary_c_xyz=batch.get('boundary_c_xyz'),
            excitation_index=batch.get('excitation_index'),
            frequencies=batch.get('frequencies') if use_frf else None,
            num_graphs=batch.get('num_graphs'),
            node_counts=batch.get('node_counts'),
            physics_alpha=physics_alpha,
        )

    def train_epoch(self, loader, config: Dict, epoch: int = 0) -> Dict[str, float]:
        import time
        self.model.train()
        sums, count = {}, 0

        # 渐进式物理融合：前 warmup_epochs 从 0 线性升到 1
        # epoch=0 时 alpha≈0（纯数据驱动），epoch=warmup 时 alpha=1（纯物理路径）
        warmup_epochs = config.get('physics_alpha_warmup', 50)
        physics_alpha = min(1.0, epoch / max(warmup_epochs, 1))
        config['physics_alpha'] = physics_alpha

        t_fwd = t_bwd = 0.0
        t_wait = 0.0  # DataLoader 批次间等待（含 HDF5 读盘）
        t_loop_start = time.time()
        for batch in loader:
            t_wait += time.time() - t_loop_start  # 等 DataLoader 出下一批的时间

            batch = move_batch_to_device(batch, self.device)
            t0 = time.time()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.forward_batch(batch, config)
                loss, logs = total_loss(outputs, batch, config)
            torch.cuda.synchronize()
            t_fwd += time.time() - t0

            t0 = time.time()
            self.scaler.scale(loss).backward()
            grad_clip = config.get('gradient_clip', 1.0)
            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            torch.cuda.synchronize()
            t_bwd += time.time() - t0

            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            t_loop_start = time.time()

        if count > 0 and epoch % 10 == 0:
            print(f"  [计时] epoch {epoch}: wait={t_wait:.1f}s fwd={t_fwd:.1f}s bwd={t_bwd:.1f}s ({count}批)")

        if self.scheduler is not None:
            self.scheduler.step()
        return {key: value / max(1, count) for key, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, loader, config: Dict) -> Dict[str, float]:
        self.model.eval()
        sums, count = {}, 0
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self.forward_batch(batch, config)
            # total_loss 返回当前 loss 结构下的评估日志
            _, logs = total_loss(outputs, batch, config)
            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            count += 1
        metrics = {key: value / max(1, count) for key, value in sums.items()}
        # 从对齐后的 loss 中提取真实的 ω_rel 和 zeta_rel
        metrics['omega_rel'] = metrics.get('loss_omega', 0.0)
        zeta_k_sum = sum([metrics.get(f'zeta_k{k}', 0.0) for k in range(3)])
        metrics['zeta_rel'] = zeta_k_sum / 3.0
        return metrics


def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'metrics': metrics,
        'loss': metrics.get('loss_total', metrics.get('loss_modal', 0.0)),
    }, path)


def train(args, config, model_cfg, net, dataloader, optimizer, valloader, scheduler=None, logger=None, start_epoch=0):
    """训练入口，供 sample/run_validation.py 调用。"""
    import csv

    trainer = TransolverTrainer(net, optimizer, device=args.device, scheduler=scheduler, fp16=getattr(args, 'fp16', False))
    best = float('inf')
    epochs = config.get('epochs', 200)
    out_dir = getattr(args, 'dir', 'output')

    # 打开 CSV 日志
    csv_path = os.path.join(out_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # CSV 表头
    csv_header = ['epoch', 'lr',
                  'total',  # 加权总损失
                  'omega', 'zeta', 'phi_resp', 'phi_xyz', 'mac',  # 原始损失值
                  'omega%', 'zeta%', 'phi_resp%', 'phi_xyz%',  # 损失占比(%)
                  'omega_k0', 'omega_k1', 'omega_k2',
                  'zeta_k0', 'zeta_k1', 'zeta_k2',
                  'phi_k0', 'phi_k1', 'phi_k2',

                  # 新增细分振型损失
                  'z_shape', 'z_scale', 'part', 'xyz_shape', 'xyz_energy',
                  'z_scale_k0', 'z_scale_k1', 'z_scale_k2',
                  'part_k0', 'part_k1', 'part_k2',

                  'frf', 'frf_complex', 'frf_log_amp', 'frf_db',
                  'val_total', 'omega_rel', 'zeta_rel']
    csv_writer.writerow(csv_header)

    # 损失权重
    mw = config.get('modal_loss_weights', {})
    w_omega = mw.get('omega', 1.0)
    w_zeta = mw.get('zeta', 0.5)
    w_phi_resp = mw.get('phi_resp', 1.0)
    w_phi_xyz = mw.get('phi_xyz', 0.25)
    w_mac = mw.get('mac', 0.2)
    w_frf = config.get('frf_loss_weight', 1.0)

    def _r4(v):
        """四舍五入到 4 位小数"""
        return round(float(v), 4) if isinstance(v, (int, float, torch.Tensor)) else v

    for epoch in range(start_epoch, epochs):
        train_logs = trainer.train_epoch(dataloader, config, epoch)

        # 当前学习率
        lr = optimizer.param_groups[0]['lr']

        # 计算各损失占比
        raw_o = train_logs.get('loss_omega', 0)
        raw_z = train_logs.get('loss_zeta', 0)
        raw_p = train_logs.get('loss_phi_resp', 0)
        raw_x = train_logs.get('loss_phi_xyz', 0)
        raw_m = train_logs.get('loss_mac', 0)
        raw_f = train_logs.get('loss_frf', 0)

        wv_o = w_omega * raw_o
        wv_z = w_zeta * raw_z
        wv_p = w_phi_resp * raw_p
        wv_x = w_phi_xyz * raw_x
        wv_m = w_mac * raw_m
        wv_f = w_frf * raw_f
        total_w = wv_o + wv_z + wv_p + wv_x + wv_m + wv_f + 1e-12

        pct_o = wv_o / total_w * 100
        pct_z = wv_z / total_w * 100
        pct_p = wv_p / total_w * 100
        pct_x = wv_x / total_w * 100

        # 每轮打印（全部百分比形式 + 三阶分开展示 + 学习率）
        omega_pct = ' '.join([f"{train_logs.get(f'omega_k{k}', 0)*100:.1f}%" for k in range(3)])
        zeta_pct = ' '.join([f"{train_logs.get(f'zeta_k{k}', 0)*100:.1f}%" for k in range(3)])
        phi_pct = ' '.join([f"{train_logs.get(f'phi_k{k}', 0)*100:.1f}%" for k in range(3)])
        alpha = config.get('physics_alpha', 1.0)
        train_msg = (f"Epoch {epoch:04d} | lr={lr:.1e} | α={alpha:.2f} | total={total_w:.3e} "
                     f"[ω={pct_o:.0f}% ζ={pct_z:.0f}% φ_resp={pct_p:.0f}% φ_xyz={pct_x:.0f}%] "
                     f"ω_k=[{omega_pct}] ζ_k=[{zeta_pct}] φ_k=[{phi_pct}]")
        print(train_msg)

        # 验证和保存 checkpoint 按频率触发
        val_logs = None
        if epoch % config.get('validation_frequency', 5) == 0 or epoch == epochs - 1:
            val_logs = trainer.evaluate(valloader, config)
            metric = val_logs.get('loss_total', val_logs.get('loss_modal', float('inf')))
            val_msg = (f"  -> val={metric:.4e} "
                       f"ω_rel={val_logs.get('omega_rel', 0)*100:.1f}% "
                       f"ζ_rel={val_logs.get('zeta_rel', 0)*100:.1f}% "
                       f"φ_resp={val_logs.get('loss_phi_resp', 0):.4f} "
                       f"φ_xyz={val_logs.get('loss_phi_xyz', 0):.4f}")
            print(val_msg)
            save_checkpoint(os.path.join(out_dir, 'checkpoint_last'), net, optimizer, epoch, val_logs)
            if metric < best:
                best = metric
                save_checkpoint(os.path.join(out_dir, 'checkpoint_best'), net, optimizer, epoch, val_logs)

        # 写入 CSV
        row = [epoch, f"{lr:.1e}", _r4(total_w),
               _r4(raw_o), _r4(raw_z), _r4(raw_p), _r4(raw_x), _r4(raw_m),
               _r4(pct_o), _r4(pct_z), _r4(pct_p), _r4(pct_x),
               *[_r4(train_logs.get(f'omega_k{k}', 0) * 100) for k in range(3)],
               *[_r4(train_logs.get(f'zeta_k{k}', 0) * 100) for k in range(3)],
               *[_r4(train_logs.get(f'phi_k{k}', 0) * 100) for k in range(3)],

               # 新增细分振型损失
               _r4(train_logs.get('loss_phi_resp_shape', 0)),
               _r4(train_logs.get('loss_phi_resp_scale', 0)),
               _r4(train_logs.get('loss_phi_participation', 0)),
               _r4(train_logs.get('loss_phi_xyz_shape', 0)),
               _r4(train_logs.get('loss_phi_xyz_energy', 0)),
               *[_r4(train_logs.get(f'z_scale_k{k}', 0)) for k in range(3)],
               *[_r4(train_logs.get(f'part_k{k}', 0)) for k in range(3)],

               _r4(raw_f),
               _r4(train_logs.get('loss_frf_complex', 0)),
               _r4(train_logs.get('loss_frf_log_amp', 0)),
               _r4(train_logs.get('loss_frf_db', 0)),
               _r4(val_logs.get('loss_total', '')) if val_logs else '',
               _r4(val_logs.get('omega_rel', '')) if val_logs else '',
               _r4(val_logs.get('zeta_rel', '')) if val_logs else '']
        csv_writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    print(f"训练日志已保存: {csv_path}")
    return net


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=False):
    trainer = TransolverTrainer(net, optimizer=torch.optim.AdamW(net.parameters(), lr=1e-6), device=args.device)
    metrics = trainer.evaluate(dataloader, config)
    if verbose:
        print(metrics)
    return metrics
