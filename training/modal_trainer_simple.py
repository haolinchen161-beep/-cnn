from __future__ import annotations

import os
import time
from collections import defaultdict

import torch

from .modal_losses_scaled import modal_loss


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _loss_kwargs(cfg):
    return {
        "freq_weight": float(cfg.get("freq_weight", 1.0)),
        "phi_weight": float(cfg.get("phi_weight", 1.0)),
        "mac_weight": float(cfg.get("mac_weight", 5.0)),
        "scale_weight": float(cfg.get("scale_weight", 1.0)),
        "min_mode_weight": float(cfg.get("min_mode_weight", 0.2)),
    }


def train_modal(args, cfg, model, train_loader, val_loader=None):
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    os.makedirs(args.dir, exist_ok=True)
    best_path = os.path.join(args.dir, "checkpoint_best.pt")
    last_path = os.path.join(args.dir, "checkpoint_last.pt")
    best = float("inf")
    validation_frequency = int(cfg.get("validation_frequency", 5))

    for epoch in range(int(cfg.get("epochs", 200))):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, cfg, opt, epoch=epoch)
        train_time = time.time() - t0

        print(
            f"Epoch {epoch:04d} 训练完成 | "
            f"loss={tr['loss']:.4g} freq={tr['freq_mape_percent']:.3f}% "
            f"zMAC={tr['phi_z_mac']:.4f} scale={tr['phi_z_scale']:.4f} "
            f"zRatio={tr['dir_z_ratio_mean']:.3f} wMode={tr['mode_weight_mean']:.3f} "
            f"time={train_time:.1f}s",
            flush=True,
        )

        should_validate = (
            val_loader is not None
            and (epoch % validation_frequency == 0 or epoch == int(cfg.get("epochs", 200)) - 1)
        )
        if should_validate:
            va = evaluate_modal(args, cfg, model, val_loader)
            print(
                f"Epoch {epoch:04d} 验证完成 | "
                f"val={va['loss']:.4g} freq={va['freq_mape_percent']:.3f}% "
                f"zMAC={va['phi_z_mac']:.4f} scale={va['phi_z_scale']:.4f} "
                f"zRatio={va['dir_z_ratio_mean']:.3f} wMode={va['mode_weight_mean']:.3f}",
                flush=True,
            )
            score = float(va["loss"])
        else:
            va = tr
            score = float(tr["loss"])

        if score < best:
            best = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "best_val": best,
                "config": cfg,
            }, best_path)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "best_val": best,
            "config": cfg,
        }, last_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    return model


def run_epoch(model, loader, device, cfg, opt=None, epoch: int | None = None):
    model.train(opt is not None)
    sums = defaultdict(float)
    n = 0
    total = len(loader)
    progress_interval = int(cfg.get("progress_interval", 10))
    t0 = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        out = model(batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"])
        loss, metrics = modal_loss(out, batch, **_loss_kwargs(cfg))
        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 2.0)))
            opt.step()

        for k, v in metrics.items():
            sums[k] += float(v.detach().cpu())
        n += 1

        if opt is not None and progress_interval > 0:
            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total:
                elapsed = time.time() - t0
                avg = elapsed / max(batch_idx, 1)
                prefix = f"Epoch {epoch:04d}" if epoch is not None else "训练"
                print(
                    f"{prefix} | batch {batch_idx}/{total} | "
                    f"loss={float(loss.detach().cpu()):.4g} | "
                    f"avg={avg:.2f}s/batch | elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_modal(args, cfg, model, loader, verbose=False):
    device = torch.device(args.device)
    model.eval()
    sums = defaultdict(float)
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"])
        _, metrics = modal_loss(out, batch, **_loss_kwargs(cfg))
        for k, v in metrics.items():
            sums[k] += float(v.detach().cpu())
        n += 1
    res = {k: v / max(n, 1) for k, v in sums.items()}
    if verbose:
        print(
            f"Eval | loss={res['loss']:.4g} freq={res['freq_mape_percent']:.3f}% "
            f"zMAC={res['phi_z_mac']:.4f} scale={res['phi_z_scale']:.4f} "
            f"zRatio={res['dir_z_ratio_mean']:.3f} wMode={res['mode_weight_mean']:.3f}",
            flush=True,
        )
    return res
