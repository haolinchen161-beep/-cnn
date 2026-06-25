# -*- coding: utf-8 -*-
"""Training logic for Baseline MLP natural frequency model."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_frequency import FrequencyH5Dataset
from model_frequency import FrequencyTokenMLP

def to_namespace(cfg: Any) -> SimpleNamespace:
    if isinstance(cfg, SimpleNamespace):
        return cfg
    if is_dataclass(cfg):
        return SimpleNamespace(**asdict(cfg))
    if isinstance(cfg, dict):
        return SimpleNamespace(**cfg)
    return cfg

def cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def make_loader(ds, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=g)

def compute_stats(loader: DataLoader, target_modes: int, eps: float) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    sqs: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    specs = {
        "pocket_features": (0, 1),
        "clamp_features": (0, 1),
        "global_features": (0,)
    }
    
    omega_sum = torch.zeros(target_modes, dtype=torch.float64)
    omega_sq = torch.zeros(target_modes, dtype=torch.float64)
    omega_count = 0

    for batch in loader:
        for key, dims in specs.items():
            x = batch[key].float()
            n = 1
            for d in dims:
                n *= x.shape[d]
            sx = x.sum(dim=dims)
            ss = (x * x).sum(dim=dims)
            if key not in sums:
                sums[key], sqs[key], counts[key] = sx, ss, n
            else:
                sums[key] += sx
                sqs[key] += ss
                counts[key] += n
                
        omega_raw = batch["omega"].double()
        lo_ref = torch.log(omega_raw.clamp_min(eps))
        
        omega_sum += lo_ref.sum(dim=0)
        omega_sq += (lo_ref * lo_ref).sum(dim=0)
        omega_count += lo_ref.shape[0]

    stats: dict[str, torch.Tensor] = {}
    for key in specs:
        mean = sums[key].double() / max(counts[key], 1)
        var = sqs[key].double() / max(counts[key], 1) - mean * mean
        stats[key + "_mean"] = mean.float()
        stats[key + "_std"] = torch.sqrt(var.clamp_min(eps * eps)).float().clamp_min(eps)
        
    om = omega_sum / max(omega_count, 1)
    ov = omega_sq / max(omega_count, 1) - om * om
    stats["omega_log_mean"] = om.float()
    stats["omega_log_std"] = torch.sqrt(ov.clamp_min(eps * eps)).float().clamp_min(eps)
    return stats

def stats_to(stats: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in stats.items()}

def normalize(batch: dict[str, torch.Tensor], stats: dict[str, torch.Tensor], device: torch.device, eps: float) -> dict[str, torch.Tensor]:
    p = batch["pocket_features"].float().to(device)
    c = batch["clamp_features"].float().to(device)
    g = batch["global_features"].float().to(device)
    w = batch["omega"].float().to(device)
    p = (p - stats["pocket_features_mean"].view(1, 1, -1)) / stats["pocket_features_std"].view(1, 1, -1).clamp_min(eps)
    c = (c - stats["clamp_features_mean"].view(1, 1, -1)) / stats["clamp_features_std"].view(1, 1, -1).clamp_min(eps)
    g = (g - stats["global_features_mean"].view(1, -1)) / stats["global_features_std"].view(1, -1).clamp_min(eps)
    logw = torch.log(w.clamp_min(eps))
    y = (logw - stats["omega_log_mean"].view(1, -1)) / stats["omega_log_std"].view(1, -1).clamp_min(eps)
    return {"pocket_features": p, "clamp_features": c, "global_features": g, "omega": w, "target": y}

def pred_to_omega(pred: torch.Tensor, stats: dict[str, torch.Tensor], eps: float) -> torch.Tensor:
    logw = pred * stats["omega_log_std"].view(1, -1) + stats["omega_log_mean"].view(1, -1)
    return torch.exp(logw).clamp_min(eps)

def batch_metrics(pred: torch.Tensor, omega: torch.Tensor, stats: dict[str, torch.Tensor], eps: float, target_modes: int) -> dict[str, float]:
    wp = pred_to_omega(pred, stats, eps)
    rel = torch.abs(wp - omega) / omega.clamp_min(eps)
    out = {"rel_mean": float(rel.mean().detach().cpu()), "rel_max": float(rel.max().detach().cpu())}
    for i in range(target_modes):
        out[f"rel_mode_{i+1}"] = float(rel[:, i].mean().detach().cpu())
        out[f"rel_mode_{i+1}_max"] = float(rel[:, i].max().detach().cpu())
    return out

def run_epoch(model, loader, optimizer, stats, device, cfg, train: bool, epoch: int = 1) -> dict[str, float]:
    model.train(train)
    total_n = 0
    loss_sum = data_sum = order_sum = rel_sum = 0.0
    rel_max = 0.0
    mode_sum = np.zeros(cfg.target_modes, dtype=np.float64)
    mode_max = np.zeros(cfg.target_modes, dtype=np.float64)

    for batch in loader:
        b = normalize(batch, stats, device, cfg.eps)
        with torch.set_grad_enabled(train):
            pred = model(b["pocket_features"], b["clamp_features"], b["global_features"])
            data_loss = F.smooth_l1_loss(pred, b["target"], beta=cfg.smooth_l1_beta)
            wp_log = pred * stats["omega_log_std"].view(1, -1) + stats["omega_log_mean"].view(1, -1)
            if cfg.order_loss_weight > 0 and cfg.target_modes > 1:
                order_v = torch.relu(wp_log[:, :-1] - wp_log[:, 1:] + cfg.order_log_margin)
                order_loss = (order_v * order_v).mean()
            else:
                order_loss = torch.zeros((), device=device)
            loss = data_loss + cfg.order_loss_weight * order_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        bs = b["omega"].shape[0]
        total_n += bs
        loss_sum += float(loss.detach().cpu()) * bs
        data_sum += float(data_loss.detach().cpu()) * bs
        order_sum += float(order_loss.detach().cpu()) * bs
        m = batch_metrics(pred.detach(), b["omega"], stats, cfg.eps, cfg.target_modes)
        rel_sum += m["rel_mean"] * bs
        rel_max = max(rel_max, m["rel_max"])
        for i in range(cfg.target_modes):
            mode_sum[i] += m[f"rel_mode_{i+1}"] * bs
            mode_max[i] = max(mode_max[i], m[f"rel_mode_{i+1}_max"])

    out = {"loss": loss_sum / max(total_n, 1), "data_loss": data_sum / max(total_n, 1),
           "order_loss": order_sum / max(total_n, 1), "rel_mean": rel_sum / max(total_n, 1),
           "rel_max": rel_max}
    for i in range(cfg.target_modes):
        out[f"rel_mode_{i+1}"] = float(mode_sum[i] / max(total_n, 1))
        out[f"rel_mode_{i+1}_max"] = float(mode_max[i])
    return out

def pct(x: float) -> float:
    return float(x) * 100.0

def mode_text(metrics: dict[str, float], name: str, target_modes: int) -> str:
    return ", ".join([f"{name}{i+1}={pct(metrics[f'rel_mode_{i+1}']):.3f}%" for i in range(target_modes)])

def make_header(target_modes: int) -> list[str]:
    h = ["epoch", "lr", "did_validate", "train_loss", "train_data_loss", "train_order_loss", "train_rel_mean_pct", "train_rel_max_pct"]
    h += [f"train_rel_mode_{i+1}_pct" for i in range(target_modes)]
    h += [f"train_rel_mode_{i+1}_max_pct" for i in range(target_modes)]
    h += ["val_loss", "val_data_loss", "val_order_loss", "val_rel_mean_pct", "val_rel_max_pct"]
    h += [f"val_rel_mode_{i+1}_pct" for i in range(target_modes)]
    h += [f"val_rel_mode_{i+1}_max_pct" for i in range(target_modes)]
    return h

def make_row(epoch: int, lr: float, target_modes: int, train_m: dict[str, float], val_m: dict[str, float] | None) -> list:
    row = [epoch, lr, 1 if val_m is not None else 0, train_m["loss"], train_m["data_loss"], train_m["order_loss"], pct(train_m["rel_mean"]), pct(train_m["rel_max"])]
    row += [pct(train_m[f"rel_mode_{i+1}"]) for i in range(target_modes)]
    row += [pct(train_m[f"rel_mode_{i+1}_max"]) for i in range(target_modes)]
    if val_m is None:
        row += [""] * (5 + 2 * target_modes)
    else:
        row += [val_m["loss"], val_m["data_loss"], val_m["order_loss"], pct(val_m["rel_mean"]), pct(val_m["rel_max"])]
        row += [pct(val_m[f"rel_mode_{i+1}"]) for i in range(target_modes)]
        row += [pct(val_m[f"rel_mode_{i+1}_max"]) for i in range(target_modes)]
    return row

def save_ckpt(path: Path, epoch: int, model, optimizer, stats, cfg, val_m) -> None:
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "stats": {k: v.detach().cpu() for k, v in stats.items()}, "config": vars(cfg), "val_metrics": val_m}, path)

def train_frequency_model(config: Any) -> None:
    cfg = to_namespace(config)
    cfg.target_modes = int(cfg_get(cfg, "target_modes", 3))
    cfg.val_interval = 1
    cfg.log_interval = 1
    cfg.save_last_interval = max(1, int(cfg_get(cfg, "save_last_interval", 1)))

    set_seed(int(cfg.seed))
    device = get_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_set = FrequencyH5Dataset(cfg.train_h5, target_modes=cfg.target_modes)
    val_set = FrequencyH5Dataset(cfg.val_h5, target_modes=cfg.target_modes)
    test_set = FrequencyH5Dataset(cfg.test_h5, target_modes=cfg.target_modes) if cfg.test_h5 else None
    
    train_loader = make_loader(train_set, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = make_loader(val_set, cfg.eval_batch_size, False, cfg.num_workers, cfg.seed)
    test_loader = make_loader(test_set, cfg.eval_batch_size, False, cfg.num_workers, cfg.seed) if test_set is not None else None
    stat_loader = make_loader(train_set, cfg.stat_batch_size, False, 0, cfg.seed)
    
    print("Computing train set statistics...")
    stats = stats_to(compute_stats(stat_loader, cfg.target_modes, cfg.eps), device)
    print("Train set statistics computed.")

    model = FrequencyTokenMLP(
        pocket_dim=cfg_get(cfg, "pocket_dim", 8),
        clamp_dim=cfg_get(cfg, "clamp_dim", 11),
        global_dim=cfg.global_dim,
        token_dim=cfg.token_dim,
        hidden_dim=cfg.hidden_dim,
        fusion_dim=cfg.fusion_dim,
        out_modes=cfg.target_modes,
        token_layers=cfg.token_layers,
        fusion_layers=cfg.fusion_layers,
        dropout=cfg.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Baseline MLP Model Initialized. Total Trainable Parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs - 10), eta_min=cfg.min_learning_rate)

    csv_path = out_dir / "logs" / "frequency_train_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(make_header(cfg.target_modes))

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting baseline MLP training of {cfg.target_modes} modes. Warmup = 10 epochs. Total epochs = {cfg.epochs}")
    for epoch in range(1, cfg.epochs + 1):
        if epoch <= 10:
            lr_scale = epoch / 10.0
            current_lr = cfg.learning_rate * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(model, train_loader, optimizer, stats, device, cfg, True, epoch)

        did_val = False
        val_m = None
        if epoch % cfg.val_interval == 0:
            did_val = True
            val_m = run_epoch(model, val_loader, None, stats, device, cfg, False, epoch)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(make_row(epoch, current_lr, cfg.target_modes, train_m, val_m))

        if epoch % cfg.log_interval == 0:
            t_txt = mode_text(train_m, "Mode", cfg.target_modes)
            print(f"Epoch {epoch:03d} | LR {current_lr:.6f} | Train Loss {train_m['loss']:.4f} | Mean {pct(train_m['rel_mean']):.3f}% ({t_txt})")
            if val_m is not None:
                v_txt = mode_text(val_m, "Mode", cfg.target_modes)
                print(f"          | Val Loss   {val_m['loss']:.4f} | Mean {pct(val_m['rel_mean']):.3f}% ({v_txt})")

        if val_m is not None:
            val_loss = val_m["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_ckpt(out_dir / "checkpoints" / "best_frequency_model.pt", epoch, model, optimizer, stats, cfg, val_m)
            else:
                patience_counter += 1

            if patience_counter >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

        if epoch % cfg.save_last_interval == 0:
            save_ckpt(out_dir / "checkpoints" / "last_frequency_model.pt", epoch, model, optimizer, stats, cfg, val_m)

    # Load best model for testing
    best_path = out_dir / "checkpoints" / "best_frequency_model.pt"
    if best_path.exists() and test_loader is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_m = run_epoch(model, test_loader, None, stats, device, cfg, False, cfg.epochs)
        test_txt = mode_text(test_m, "Mode", cfg.target_modes)
        print(f"\n================ Baseline MLP Testing ================")
        print(f"Test Loss: {test_m['loss']:.4f} | Mean Rel Error: {pct(test_m['rel_mean']):.3f}% ({test_txt})")
        print(f"======================================================")
        with open(out_dir / "logs" / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_m, f, indent=4)
