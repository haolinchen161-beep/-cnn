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

    def forward_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return self.model(
            points=batch['points'],
            node_features=batch['node_features'],
            batch=batch['batch'],
            edge_index=batch.get('edge_index'),
            boundary_c_xyz=batch.get('boundary_c_xyz'),
            excitation_index=batch.get('excitation_index'),
            frequencies=batch.get('frequencies'),
            num_graphs=batch.get('num_graphs'),
        )

    def train_epoch(self, loader, config: Dict, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        sums, count = {}, 0
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.forward_batch(batch)
                loss, logs = total_loss(outputs, batch, config)
            self.scaler.scale(loss).backward()
            grad_clip = config.get('gradient_clip', 1.0)
            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            count += 1
        if self.scheduler is not None:
            self.scheduler.step()
        return {key: value / max(1, count) for key, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, loader, config: Dict) -> Dict[str, float]:
        self.model.eval()
        sums, count = {}, 0
        omega_rel, zeta_rel = 0.0, 0.0
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self.forward_batch(batch)
            _, logs = total_loss(outputs, batch, config)
            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            omega_rel += float((torch.abs(outputs['modal_omega'] - batch['modal_omega']) / batch['modal_omega'].abs().clamp_min(1e-8)).mean().cpu())
            zeta_rel += float((torch.abs(outputs['modal_zeta'] - batch['modal_zeta']) / batch['modal_zeta'].abs().clamp_min(1e-8)).mean().cpu())
            count += 1
        metrics = {key: value / max(1, count) for key, value in sums.items()}
        metrics['omega_rel'] = omega_rel / max(1, count)
        metrics['zeta_rel'] = zeta_rel / max(1, count)
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
    """Compatibility function used by sample/run_validation.py."""
    trainer = TransolverTrainer(net, optimizer, device=args.device, scheduler=scheduler, fp16=getattr(args, 'fp16', False))
    best = float('inf')
    epochs = config.get('epochs', 200)
    out_dir = getattr(args, 'dir', 'output')
    for epoch in range(start_epoch, epochs):
        train_logs = trainer.train_epoch(dataloader, config, epoch)
        if epoch % config.get('validation_frequency', 5) == 0 or epoch == epochs - 1:
            val_logs = trainer.evaluate(valloader, config)
            metric = val_logs.get('loss_total', val_logs.get('loss_modal', float('inf')))
            msg = (f"Epoch {epoch:04d} | train={train_logs.get('loss_total', 0):.4e} "
                   f"val={metric:.4e} omega_rel={val_logs.get('omega_rel', 0):.3e} "
                   f"zeta_rel={val_logs.get('zeta_rel', 0):.3e}")
            print(msg)
            save_checkpoint(os.path.join(out_dir, 'checkpoint_last'), net, optimizer, epoch, val_logs)
            if metric < best:
                best = metric
                save_checkpoint(os.path.join(out_dir, 'checkpoint_best'), net, optimizer, epoch, val_logs)
    return net


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=False):
    trainer = TransolverTrainer(net, optimizer=torch.optim.AdamW(net.parameters(), lr=1e-6), device=args.device)
    metrics = trainer.evaluate(dataloader, config)
    if verbose:
        print(metrics)
    return metrics
