from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from modal_residue import train_modal_residue_model as base
from modal_residue import train_modal_residue_bottom_model as bottom


N_MODES_USED = 3
EPS = 1e-12


def seed_all(seed: int) -> None:
    bottom.seed_all(seed)


class LimitedCachedBottomSplit:
    """底面训练 split：保留全图，但只截取前 n_modes 阶标签，可选预加载。"""

    def __init__(self, data_dir: Path, split: str, n_modes: int = N_MODES_USED,
                 preload: bool = True, limit: int = 0, source_split: str | None = None):
        self.base = bottom.H5Split(data_dir, source_split or split)
        if limit and limit > 0:
            self.base.keys = self.base.keys[: min(limit, len(self.base.keys))]
        self.n_modes = int(n_modes)
        self._cache = None
        if preload:
            total = len(self.base)
            print(f">>> preload {split}: {total} samples ...", flush=True)
            self._cache = [self._trim(self.base[i]) for i in range(total)]
            print(f">>> preload {split}: done", flush=True)

    def __len__(self) -> int:
        return len(self.base) if self._cache is None else len(self._cache)

    def _trim(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        s = dict(s)
        n = self.n_modes
        if "modal_omega" in s:
            s["modal_omega"] = s["modal_omega"][:n]
        if "modal_residue_z" in s:
            s["modal_residue_z"] = s["modal_residue_z"][:, :n]
        if "modal_phi_xyz" in s:
            s["modal_phi_xyz"] = s["modal_phi_xyz"][:, :n, :]
        if "modal_zeta" in s:
            s["modal_zeta"] = s["modal_zeta"][:n]
        return s

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        if self._cache is not None:
            return self._cache[i]
        return self._trim(self.base[i])


class PerModeResidueNet(nn.Module):
    """R3 下一步模型：共享 MeshGraph encoder，频率 head 共享，每阶 A 使用独立 head。"""

    def __init__(self, node_in_dim: int, edge_in_dim: int, n_modes: int = N_MODES_USED,
                 hidden: int = 96, gnn_layers: int = 3):
        super().__init__()
        self.n_modes = int(n_modes)
        self.node_encoder = base.mlp(node_in_dim, hidden, hidden, layers=3)
        self.edge_encoder = base.mlp(edge_in_dim, hidden, hidden, layers=3)
        self.blocks = nn.ModuleList([base.MeshGraphBlock(hidden) for _ in range(gnn_layers)])
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.omega_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.n_modes)
        )
        head_in = 3 * hidden + 6
        self.residue_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(self.n_modes)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                coords_norm: torch.Tensor, q: torch.Tensor, exc_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr).to(dtype=h.dtype)
        for block in self.blocks:
            h, e = block(h, e, edge_index)

        g_raw = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0)
        g = self.global_mlp(g_raw)
        omega_norm = self.omega_head(g)

        exc_i = torch.clamp(exc_idx.long(), 0, h.shape[0] - 1)
        hq = h[q]
        he = h[exc_i].view(1, -1).expand(hq.shape[0], -1)
        gg = g.view(1, -1).expand(hq.shape[0], -1)
        q_xyz = coords_norm[q].to(dtype=h.dtype)
        rel_xyz = q_xyz - coords_norm[exc_i].view(1, 3).to(dtype=h.dtype)
        head_in = torch.cat([hq, gg, he, q_xyz, rel_xyz], dim=-1)
        residue_y = torch.cat([head(head_in) for head in self.residue_heads], dim=-1)
        return omega_norm, residue_y


def mode_weights(n_modes: int, device: torch.device) -> torch.Tensor:
    if n_modes <= 1:
        return torch.ones(max(n_modes, 1), device=device)
    return torch.linspace(1.0, 1.5, steps=n_modes, device=device)


def weighted_mean(loss_per_mode: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=loss_per_mode.device, dtype=loss_per_mode.dtype)[: loss_per_mode.numel()]
    return torch.sum(loss_per_mode * weights) / torch.clamp(torch.sum(weights), min=EPS)


def per_mode_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float()) ** 2
    loss_per_mode = err if err.dim() == 1 else err.reshape(-1, err.shape[-1]).mean(dim=0)
    return weighted_mean(loss_per_mode, weights)


def per_mode_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    err = nn.functional.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    loss_per_mode = err if err.dim() == 1 else err.reshape(-1, err.shape[-1]).mean(dim=0)
    return weighted_mean(loss_per_mode, weights)


def per_mode_masked_physical_smooth_l1(A_pred: torch.Tensor, A_true: torch.Tensor,
                                       scale: torch.Tensor, mask: torch.Tensor,
                                       weights: torch.Tensor) -> torch.Tensor:
    if mask is None or not bool(mask.any()):
        return A_pred.sum() * 0.0
    err = (A_pred.float() - A_true.float()) / torch.clamp(scale, min=EPS)
    losses = []
    used_w = []
    for r in range(err.shape[-1]):
        mr = mask[:, r]
        if bool(mr.any()):
            losses.append(nn.functional.smooth_l1_loss(err[mr, r], torch.zeros_like(err[mr, r])))
            used_w.append(weights[r])
    if not losses:
        return A_pred.sum() * 0.0
    return weighted_mean(torch.stack(losses), torch.stack(used_w))


def forward_model(model: nn.Module, t: Dict[str, torch.Tensor], q: torch.Tensor):
    return model(t["x"], t["edge_index"], t["edge_attr"], t["coords"], q, t["exc_idx"])


def train_epoch(model, ds, opt, scaler, stats, args, device):
    model.train()
    order = list(range(len(ds)))
    random.shuffle(order)
    sums = np.zeros(4, dtype=np.float64)
    omega_errs: List[np.ndarray] = []
    A_vis_errs: List[np.ndarray] = []
    A_top_errs: List[np.ndarray] = []
    A_sign_errs: List[np.ndarray] = []
    amp_enabled = bool(args.fp16 and device.type == "cuda")

    for i in order:
        t = bottom.to_tensors(ds[i], stats, device)
        q = bottom.select_query(t, args.query_nodes, args.target_region, random_sample=True)
        weights = mode_weights(int(t["omega"].numel()), device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            omega_p, Y_p = forward_model(model, t, q)
            omega_t, Y_t, A_t, scale = bottom.norm_targets(t, stats, q)
            loss_omega = per_mode_mse(omega_p, omega_t, weights)
            loss_full = per_mode_smooth_l1(Y_p, Y_t, weights)
            A_p = base.asinh_to_physical(Y_p, scale, clamp=args.asinh_clamp)
            top_mask = base.top_mode_mask(A_t, args.top_node_frac)
            dom_mask = base.node_dominant_mask(A_t, args.node_dominant_k)
            loss_top = per_mode_masked_physical_smooth_l1(A_p, A_t, scale, top_mask, weights)
            loss_dom = per_mode_masked_physical_smooth_l1(A_p, A_t, scale, dom_mask, weights)
            loss = (
                args.omega_loss_weight * loss_omega
                + args.residue_full_loss_weight * loss_full
                + args.top_aux_loss_weight * loss_top
                + args.node_dominant_loss_weight * loss_dom
            )

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            omega_phys, A_phys = base.denorm_outputs(omega_p.detach(), Y_p.detach(), stats, clamp=args.asinh_clamp)
            w_rel = torch.abs(omega_phys - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=EPS) * 100.0
            A_true = t["A"][q]
            A_vis = base.A_visible_error_pct(A_phys, A_true, args.residue_visible_rel)
            A_top = base.A_top_error_pct(A_phys, A_true, args.top_node_frac)
            A_sign = bottom.A_sign_accuracy_pct(A_phys, A_true, args.sign_visible_rel)
            omega_errs.append(w_rel.detach().cpu().numpy())
            A_vis_errs.append(A_vis.detach().cpu().numpy())
            A_top_errs.append(A_top.detach().cpu().numpy())
            A_sign_errs.append(A_sign.detach().cpu().numpy())

        sums += np.array([
            float(loss.detach().cpu()),
            float(loss_omega.detach().cpu()),
            float(loss_full.detach().cpu()),
            float((args.top_aux_loss_weight * loss_top + args.node_dominant_loss_weight * loss_dom).detach().cpu()),
        ])

    n_modes = len(omega_errs[0]) if omega_errs else 0
    return sums / max(len(order), 1), {
        "w_triplet": base.triplet_percent(omega_errs),
        "A_vis_triplet": base.triplet_percent(A_vis_errs),
        "A_top_triplet": base.triplet_percent(A_top_errs),
        "A_sign_triplet": base.triplet_percent(A_sign_errs),
        "w_modes": base.mode_mean(omega_errs, n_modes),
        "A_vis_modes": base.mode_mean(A_vis_errs, n_modes),
        "A_top_modes": base.mode_mean(A_top_errs, n_modes),
        "A_sign_modes": base.mode_mean(A_sign_errs, n_modes),
    }


@torch.no_grad()
def evaluate(model, ds, stats, args, device):
    model.eval()
    rows: List[Dict[str, float]] = []
    omega_errs: List[np.ndarray] = []
    A_vis_errs: List[np.ndarray] = []
    A_top_errs: List[np.ndarray] = []
    A_sign_errs: List[np.ndarray] = []
    y_losses: List[float] = []
    n_query_values: List[float] = []
    amp_enabled = bool(args.fp16 and device.type == "cuda")

    for i in range(len(ds)):
        t = bottom.to_tensors(ds[i], stats, device)
        q = bottom.select_query(t, args.eval_query_nodes, args.target_region, random_sample=False)
        weights = mode_weights(int(t["omega"].numel()), device)
        omega_t, Y_t, A_true, _ = bottom.norm_targets(t, stats, q)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            omega_n, Y_n = forward_model(model, t, q)
        omega, A = base.denorm_outputs(omega_n, Y_n, stats, clamp=args.asinh_clamp)
        w_rel = torch.abs(omega - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=EPS) * 100.0
        A_vis = base.A_visible_error_pct(A, A_true, args.residue_visible_rel)
        A_top = base.A_top_error_pct(A, A_true, args.top_node_frac)
        A_sign = bottom.A_sign_accuracy_pct(A, A_true, args.sign_visible_rel)
        y_loss = per_mode_smooth_l1(Y_n.float(), Y_t.float(), weights).detach()

        w_np = w_rel.detach().cpu().numpy()
        A_np = A_vis.detach().cpu().numpy()
        T_np = A_top.detach().cpu().numpy()
        S_np = A_sign.detach().cpu().numpy()
        omega_errs.append(w_np)
        A_vis_errs.append(A_np)
        A_top_errs.append(T_np)
        A_sign_errs.append(S_np)
        y_losses.append(float(y_loss.cpu()))
        n_query_values.append(float(q.numel()))

        row = {"sample": i, "n_target_nodes": int(q.numel()), "Y_smooth_l1": float(y_loss.cpu())}
        base.row_add_triplet(row, "w", base.triplet_percent([w_np]))
        base.row_add_triplet(row, "A_vis", base.triplet_percent([A_np]))
        base.row_add_triplet(row, "A_top", base.triplet_percent([T_np]))
        base.row_add_triplet(row, "A_sign", base.triplet_percent([S_np]))
        base.row_add_modes(row, "w", w_np)
        base.row_add_modes(row, "A_vis", A_np)
        base.row_add_modes(row, "A_top", T_np)
        base.row_add_modes(row, "A_sign", S_np)
        rows.append(row)

    n_modes = len(omega_errs[0]) if omega_errs else 0
    mean = {
        "w_triplet": base.triplet_percent(omega_errs),
        "A_vis_triplet": base.triplet_percent(A_vis_errs),
        "A_top_triplet": base.triplet_percent(A_top_errs),
        "A_sign_triplet": base.triplet_percent(A_sign_errs),
        "Y_smooth_l1_triplet": base.scalar_triplet(y_losses),
        "n_target_nodes_triplet": base.scalar_triplet(n_query_values),
        "w_modes": base.mode_mean(omega_errs, n_modes),
        "A_vis_modes": base.mode_mean(A_vis_errs, n_modes),
        "A_top_modes": base.mode_mean(A_top_errs, n_modes),
        "A_sign_modes": base.mode_mean(A_sign_errs, n_modes),
    }
    y_rms = float(mean["Y_smooth_l1_triplet"][2])
    w_rms = float(mean["w_triplet"][2])
    a_vis_mean = float(mean["A_vis_triplet"][0])
    sign_mean = float(mean["A_sign_triplet"][0])
    mean["modal_score"] = float(y_rms + 0.05 * w_rms + 0.001 * a_vis_mean + 0.0002 * max(0.0, 100.0 - sign_mean))
    return mean, rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def append_csv_row(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, epoch, best_value,
                       opt=None, sched=None, scaler=None):
    payload = {
        "model": model.state_dict(),
        "stats": stats,
        "node_in_dim": in_dim,
        "edge_in_dim": edge_dim,
        "n_modes": n_modes,
        "model_type": "PerModeResidueNet",
        "loss_type": "per_mode_weighted_asinh_A",
        "epoch": epoch,
        "best_modal_score": best_value,
        "target_region": args.target_region,
        "target_transform": "signed_asinh_fixed_per_mode_scale_bottom_region_R3",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    if sched is not None:
        payload["scheduler"] = sched.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def load_resume_state(path: Path, model, opt, sched, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    else:
        print(">>> resume warning: checkpoint has no optimizer state; optimizer restarts.")
    if "scheduler" in ckpt:
        sched.load_state_dict(ckpt["scheduler"])
    else:
        print(">>> resume warning: checkpoint has no scheduler state; scheduler restarts.")
    if "scaler" in ckpt and scaler is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f">>> resume warning: failed to load GradScaler state: {exc}")
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best = float(ckpt.get("best_modal_score", float("inf")))
    stats = ckpt.get("stats", None)
    print(f">>> resumed from {path}: previous_epoch={start_epoch - 1}, next_epoch={start_epoch}, best={best:.6g}")
    return start_epoch, best, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("modal_residue/data_modal_residue_fixedclamp300"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/下一步_R3_每阶A头_bottom"))
    p.add_argument("--epochs", type=int, default=150, help="Total target epoch. When --resume is used, training continues until this epoch.")
    p.add_argument("--n-modes-used", type=int, default=N_MODES_USED)
    p.add_argument("--query-nodes", type=int, default=256)
    p.add_argument("--eval-query-nodes", type=int, default=0)
    p.add_argument("--target-region", choices=["bottom", "all"], default="bottom")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--gnn-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--omega-loss-weight", type=float, default=1.0)
    p.add_argument("--residue-full-loss-weight", type=float, default=1.0)
    p.add_argument("--top-aux-loss-weight", type=float, default=0.25)
    p.add_argument("--node-dominant-loss-weight", type=float, default=0.10)
    p.add_argument("--top-node-frac", type=float, default=0.10)
    p.add_argument("--node-dominant-k", type=int, default=1)
    p.add_argument("--asinh-clamp", type=float, default=20.0)
    p.add_argument("--residue-visible-rel", type=float, default=1e-3)
    p.add_argument("--sign-visible-rel", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--no-preload", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from last_model.pt or --resume-path.")
    p.add_argument("--resume-path", type=Path, default=None, help="Checkpoint path for resume. Default: out_dir/last_model.pt.")
    p.add_argument("--debug-train-samples", type=int, default=0)
    p.add_argument("--debug-val-samples", type=int, default=0)
    p.add_argument("--debug-test-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    preload = True
    if args.no_preload:
        preload = False
    if args.preload:
        preload = True

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = args.out_dir / "training_log.csv"
    if log_csv.exists() and not args.resume:
        log_csv.unlink()

    device = torch.device(args.device)
    train = LimitedCachedBottomSplit(args.data_dir, "train", args.n_modes_used, preload=preload, limit=args.debug_train_samples)
    val = LimitedCachedBottomSplit(args.data_dir, "val", args.n_modes_used, preload=preload, limit=args.debug_val_samples)
    test = LimitedCachedBottomSplit(args.data_dir, "test", args.n_modes_used, preload=preload, limit=args.debug_test_samples)
    stats = bottom.compute_stats(train, args.data_dir, target_region=args.target_region)

    first = train[0]
    in_dim = bottom.node_input(first).shape[1]
    edge_dim = first["edge_attr"].shape[1]
    n_modes = first["modal_omega"].shape[0]
    model = PerModeResidueNet(in_dim, edge_dim, n_modes, args.hidden, args.gnn_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))

    start_epoch = 1
    best = float("inf")
    if args.resume:
        resume_path = args.resume_path or (args.out_dir / "last_model.pt")
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_epoch, best, ckpt_stats = load_resume_state(resume_path, model, opt, sched, scaler, device)
        if ckpt_stats is not None:
            stats = ckpt_stats

    np.savez(args.out_dir / "normalization_stats.npz", **stats)

    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> R3 per-mode bottom model: data={args.data_dir}, train/val/test={len(train)}/{len(val)}/{len(test)}")
    print(f">>> device={device}, fp16={args.fp16}, preload={preload}, resume={args.resume}")
    print(f">>> node_dim={in_dim}, edge_dim={edge_dim}, modes={n_modes}, hidden={args.hidden}, layers={args.gnn_layers}, params={total_params:,}")
    print(f">>> loss: per-mode weighted omega + signed-asinh A + top A + dominant A")
    print(f">>> target_region={args.target_region}, query={args.query_nodes}, eval_query={args.eval_query_nodes}")
    print(f">>> A_scale={np.array2string(stats['A_asinh_scale'], precision=6, separator=', ')}")

    hist: List[Dict[str, float]] = []
    if start_epoch > args.epochs:
        print(f">>> resume checkpoint epoch is already {start_epoch - 1}, target epochs={args.epochs}; skip training and run final evaluation.")

    for ep in range(start_epoch, args.epochs + 1):
        tr_loss, tr_m = train_epoch(model, train, opt, scaler, stats, args, device)
        sched.step()
        va, _ = evaluate(model, val, stats, args, device)
        score = float(va["modal_score"])

        payload = checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, ep, best, opt=opt, sched=sched, scaler=scaler)
        if score < best:
            best = score
            payload = checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, ep, best, opt=opt, sched=sched, scaler=scaler)
            torch.save(payload, args.out_dir / "best_model.pt")
        torch.save(payload, args.out_dir / "last_model.pt")

        row: Dict[str, float] = {
            "epoch": ep,
            "lr": float(sched.get_last_lr()[0]),
            "loss": float(tr_loss[0]),
            "loss_w": float(tr_loss[1]),
            "loss_Y_full": float(tr_loss[2]),
            "loss_A_aux_weighted": float(tr_loss[3]),
            "val_modal_score": score,
        }
        for prefix, metrics in [("train", tr_m), ("val", va)]:
            base.row_add_triplet(row, f"{prefix}_w", metrics["w_triplet"])
            base.row_add_triplet(row, f"{prefix}_A_vis", metrics["A_vis_triplet"])
            base.row_add_triplet(row, f"{prefix}_A_top", metrics["A_top_triplet"])
            base.row_add_triplet(row, f"{prefix}_A_sign", metrics["A_sign_triplet"])
            base.row_add_modes(row, f"{prefix}_w", metrics["w_modes"])
            base.row_add_modes(row, f"{prefix}_A_vis", metrics["A_vis_modes"])
            base.row_add_modes(row, f"{prefix}_A_top", metrics["A_top_modes"])
            base.row_add_modes(row, f"{prefix}_A_sign", metrics["A_sign_modes"])
            if "Y_smooth_l1_triplet" in metrics:
                base.row_add_triplet(row, f"{prefix}_Y", metrics["Y_smooth_l1_triplet"])
                base.row_add_triplet(row, f"{prefix}_n_target_nodes", metrics["n_target_nodes_triplet"])
        hist.append(row)
        append_csv_row(log_csv, row)

        if ep == 1 or ep % max(args.log_every, 1) == 0 or ep == args.epochs:
            print(
                f"Epoch {ep:4d} | "
                f"w=[{base.fmt_triplet(tr_m['w_triplet'], 1)}]%  "
                f"A_vis=[{base.fmt_triplet(tr_m['A_vis_triplet'], 1)}]%  "
                f"A_top=[{base.fmt_triplet(tr_m['A_top_triplet'], 1)}]%  "
                f"sign=[{base.fmt_triplet(tr_m['A_sign_triplet'], 1)}]%  "
                f"loss={base.fmt_loss(float(tr_loss[0]))}(w={base.fmt_loss(float(tr_loss[1]))},Y={base.fmt_loss(float(tr_loss[2]))},Aaux={base.fmt_loss(float(tr_loss[3]))})"
            )
            print(
                f"Val | "
                f"n=[{base.fmt_triplet(va['n_target_nodes_triplet'], 0)}]  "
                f"w=[{base.fmt_triplet(va['w_triplet'], 3)}]%  "
                f"A_vis=[{base.fmt_triplet(va['A_vis_triplet'], 1)}]%  "
                f"A_top=[{base.fmt_triplet(va['A_top_triplet'], 1)}]%  "
                f"sign=[{base.fmt_triplet(va['A_sign_triplet'], 1)}]%  "
                f"Y=[{base.fmt_triplet(va['Y_smooth_l1_triplet'], 4)}]"
            )

    if hist:
        existing_rows = []
        if args.resume and (args.out_dir / "history.csv").exists():
            # Keep previous history.csv as-is if it exists; training_log.csv is the continuous log.
            pass
        else:
            write_csv(args.out_dir / "history.csv", hist)

    ckpt_path = args.out_dir / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    va, vr = evaluate(model, val, stats, args, device)
    te, tr = evaluate(model, test, stats, args, device)
    write_csv(args.out_dir / "val_metrics.csv", vr)
    write_csv(args.out_dir / "test_metrics.csv", tr)

    summary = {
        "model_type": "PerModeResidueNet",
        "loss_type": "per_mode_weighted_asinh_A",
        "n_modes_used": n_modes,
        "target_region": args.target_region,
        "query_nodes": args.query_nodes,
        "eval_query_nodes": args.eval_query_nodes,
        "A_asinh_scale": stats["A_asinh_scale"].astype(float).tolist(),
        "best_modal_score": best,
        "val": {
            "w_mean_max_rms_pct": list(va["w_triplet"]),
            "A_vis_mean_max_rms_pct": list(va["A_vis_triplet"]),
            "A_top_mean_max_rms_pct": list(va["A_top_triplet"]),
            "A_sign_mean_max_rms_pct": list(va["A_sign_triplet"]),
            "Y_smooth_l1_mean_max_rms": list(va["Y_smooth_l1_triplet"]),
            "modal_score": va["modal_score"],
        },
        "test": {
            "w_mean_max_rms_pct": list(te["w_triplet"]),
            "A_vis_mean_max_rms_pct": list(te["A_vis_triplet"]),
            "A_top_mean_max_rms_pct": list(te["A_top_triplet"]),
            "A_sign_mean_max_rms_pct": list(te["A_sign_triplet"]),
            "Y_smooth_l1_mean_max_rms": list(te["Y_smooth_l1_triplet"]),
            "modal_score": te["modal_score"],
        },
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
