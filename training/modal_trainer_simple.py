import os
from collections import defaultdict
import torch
from .modal_losses import modal_loss


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_modal(args, cfg, model, train_loader, val_loader=None):
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-2)))
    os.makedirs(args.dir, exist_ok=True)
    best_path = os.path.join(args.dir, "checkpoint_best.pt")
    best = float("inf")
    for epoch in range(int(cfg.get("epochs", 200))):
        tr = run_epoch(model, train_loader, device, cfg, opt)
        va = evaluate_modal(args, cfg, model, val_loader) if val_loader is not None else tr
        if va["loss"] < best:
            best = va["loss"]
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "best_val": best, "config": cfg}, best_path)
        print(f"Epoch {epoch:04d} | train={tr['loss']:.4g} val={va['loss']:.4g} freq={va['freq_mape_percent']:.3f}% MAC={va['phi_mac']:.4f}")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    return model


def run_epoch(model, loader, device, cfg, opt=None):
    model.train(opt is not None)
    sums = defaultdict(float)
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        out = model(batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"])
        loss, metrics = modal_loss(out, batch, freq_weight=cfg.get("freq_weight", 1.0), phi_weight=cfg.get("phi_weight", 1.0), mac_weight=cfg.get("mac_weight", 5.0))
        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 2.0)))
            opt.step()
        for k, v in metrics.items():
            sums[k] += float(v.detach().cpu())
        n += 1
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
        _, metrics = modal_loss(out, batch, freq_weight=cfg.get("freq_weight", 1.0), phi_weight=cfg.get("phi_weight", 1.0), mac_weight=cfg.get("mac_weight", 5.0))
        for k, v in metrics.items():
            sums[k] += float(v.detach().cpu())
        n += 1
    res = {k: v / max(n, 1) for k, v in sums.items()}
    if verbose:
        print(f"Eval | loss={res['loss']:.4g} freq={res['freq_mape_percent']:.3f}% MAC={res['phi_mac']:.4f}")
    return res
