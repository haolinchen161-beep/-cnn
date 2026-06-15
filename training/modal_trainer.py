from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict

import torch

from .losses import evaluate_modal_metrics, modal_loss


def to_device(batch: Dict, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_modal(args, config: Dict, model, train_loader, val_loader=None):
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-2)))
    os.makedirs(args.dir, exist_ok=True)
    best_path = os.path.join(args.dir, "checkpoint_best.pt")
    last_path = os.path.join(args.dir, "checkpoint_last.pt")
    best = float("inf")

    for epoch in range(int(config.get("epochs", 200))):
        tr = run_epoch(model, train_loader, device, config, opt)
        va = evaluate_modal(args, config, model, val_loader) if val_loader is not None else tr
        if float(va["loss"]) < best:
            best = float(va["loss"])
            save_ckpt(best_path, model, opt, epoch, best, config)
        save_ckpt(last_path, model, opt, epoch, best, config)
        print(f"Epoch {epoch:04d} | train={tr['loss']:.4g} val={va['loss']:.4g} freq={va['freq_mape_percent']:.3f}% MAC={va['phi_mac']:.4f}")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    return model


def run_epoch(model, loader, device, config, opt=None):
    model.train(opt is not None)
    sums = defaultdict(float)
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        out = model(batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"])
        loss, metrics = modal_loss(out, batch,
                                   freq_weight=float(config.get("freq_weight", 1.0)),
                                   phi_weight=float(config.get("phi_weight", 1.0)),
                                   mac_weight=float(config.get("mac_weight", 20.0)),
                                   std_weight=float(config.get("std_weight", 2.0)),
                                   direction_weight=float(config.get("direction_weight", 1.0)))
        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 2.0)))
            opt.step()
        for k, v in metrics.items():
            sums[k] += float(v.detach().cpu())
        count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_modal(args, config: Dict, model, loader, verbose: bool = False):
    device = torch.device(args.device)
    model.eval()
    sums = defaultdict(float)
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"])
        loss, parts = modal_loss(out, batch)
        metrics = evaluate_modal_metrics(out, batch)
        metrics["loss"] = loss.detach()
        metrics["loss_freq"] = parts["loss_freq"]
        metrics["loss_phi"] = parts["loss_phi"]
        for k, v in metrics.items():
            sums[k] += float(v.detach().cpu())
        count += 1
    res = {k: v / max(count, 1) for k, v in sums.items()}
    if verbose:
        print(f"Eval | loss={res['loss']:.4g} freq={res['freq_mape_percent']:.3f}% MAC={res['phi_mac']:.4f}")
    return res


def save_ckpt(path, model, opt, epoch, best, config):
    torch.save({"epoch": epoch, "best_val": best, "config": config, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict()}, path)
