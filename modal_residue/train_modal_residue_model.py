from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn

GEOM_SCALE = np.array([0.160, 0.060, 0.010], dtype=np.float32)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class H5Split:
    def __init__(self, data_dir: Path, split: str):
        self.path = data_dir / f"{split}.h5"
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as f:
            self.keys = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as f:
            g = f[self.keys[i]]
            return {
                "points": g["points"][:].astype(np.float32),
                "point_features": g["point_features"][:].astype(np.float32),
                "spring_k_xyz": g["spring_k_xyz"][:].astype(np.float32),
                "spring_c_xyz": g["spring_c_xyz"][:].astype(np.float32),
                "node_type": g["node_type"][:].astype(np.int64),
                "modal_omega": g["modal_omega"][:].astype(np.float32),
                "modal_zeta": g["modal_zeta"][:].astype(np.float32),
                "modal_residue_z": g["modal_residue_z"][:].astype(np.float32),
                "frequencies": g["frequencies"][:].astype(np.float32),
                "point_frf": g["point_frf"][:].astype(np.float32),
                "excitation_coord": g["excitation_coord"][:].astype(np.float32),
            }


def node_input(s: Dict[str, np.ndarray]) -> np.ndarray:
    xyz = s["points"] / GEOM_SCALE.reshape(1, 3)
    exc = s["excitation_coord"] / GEOM_SCALE
    rel = xyz - exc.reshape(1, 3)
    k = np.log10(1.0 + np.maximum(s["spring_k_xyz"], 0.0)) / 8.0
    c = np.log10(1.0 + np.maximum(s["spring_c_xyz"], 0.0)) / 4.0
    nt = s["node_type"].astype(np.float32).reshape(-1, 1) / 4.0
    return np.concatenate([xyz, rel, s["point_features"], k, c, nt], axis=1).astype(np.float32)


def compute_stats(ds: H5Split) -> Dict[str, np.ndarray]:
    sx = None
    sx2 = None
    n = 0
    logs = []
    residues = []
    for i in range(len(ds)):
        s = ds[i]
        x = node_input(s).astype(np.float64)
        if sx is None:
            sx = np.zeros(x.shape[1], dtype=np.float64)
            sx2 = np.zeros(x.shape[1], dtype=np.float64)
        sx += x.sum(axis=0)
        sx2 += (x * x).sum(axis=0)
        n += x.shape[0]
        logs.append(np.log(np.maximum(s["modal_omega"], 1e-12)))
        residues.append(s["modal_residue_z"].reshape(-1, s["modal_residue_z"].shape[-1]))
    xm = sx / max(n, 1)
    xs = np.sqrt(np.maximum(sx2 / max(n, 1) - xm * xm, 1e-12))
    lo = np.stack(logs, axis=0).astype(np.float32)
    rr = np.concatenate(residues, axis=0).astype(np.float32)
    return {
        "x_mean": xm.astype(np.float32),
        "x_std": xs.astype(np.float32),
        "omega_log_mean": lo.mean(axis=0).astype(np.float32),
        "omega_log_std": (lo.std(axis=0) + 1e-6).astype(np.float32),
        "residue_mean": rr.mean(axis=0).astype(np.float32),
        "residue_std": (rr.std(axis=0) + 1e-12).astype(np.float32),
    }


class ModalResidueNet(nn.Module):
    def __init__(self, in_dim: int, n_modes: int, hidden: int = 192):
        super().__init__()
        self.node = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.glob = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.omega = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_modes))
        self.residue = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_modes))

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node(x)
        g = self.glob(torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0))
        om = self.omega(g)
        qh = h[q]
        gg = g.unsqueeze(0).expand(qh.shape[0], -1)
        rz = self.residue(torch.cat([qh, gg], dim=1))
        return om, rz


def to_tensors(s: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    x = node_input(s)
    x = (x - stats["x_mean"][None, :]) / stats["x_std"][None, :]
    return {
        "x": torch.from_numpy(x).to(device),
        "omega": torch.from_numpy(s["modal_omega"]).to(device),
        "zeta": torch.from_numpy(s["modal_zeta"]).to(device),
        "residue": torch.from_numpy(s["modal_residue_z"]).to(device),
        "freq": torch.from_numpy(s["frequencies"]).to(device),
        "frf": torch.from_numpy(s["point_frf"]).to(device),
    }


def norm_targets(t: Dict[str, torch.Tensor], stats: Dict[str, np.ndarray], q: torch.Tensor):
    dev = t["x"].device
    om_m = torch.as_tensor(stats["omega_log_mean"], device=dev)
    om_s = torch.as_tensor(stats["omega_log_std"], device=dev)
    rz_m = torch.as_tensor(stats["residue_mean"], device=dev)
    rz_s = torch.as_tensor(stats["residue_std"], device=dev)
    om = (torch.log(torch.clamp(t["omega"], min=1e-12)) - om_m) / om_s
    rz = (t["residue"][q] - rz_m) / rz_s
    return om, rz


def denorm(om_n: torch.Tensor, rz_n: torch.Tensor, stats: Dict[str, np.ndarray]):
    dev = om_n.device
    om = torch.exp(om_n * torch.as_tensor(stats["omega_log_std"], device=dev) + torch.as_tensor(stats["omega_log_mean"], device=dev))
    rz = rz_n * torch.as_tensor(stats["residue_std"], device=dev) + torch.as_tensor(stats["residue_mean"], device=dev)
    return om, rz


def frf_from_modal(omega: torch.Tensor, zeta: torch.Tensor, residue: torch.Tensor, freq_hz: torch.Tensor) -> torch.Tensor:
    w = (2.0 * math.pi * freq_hz).view(1, -1, 1)
    wk = omega.view(1, 1, -1)
    zk = zeta.view(1, 1, -1)
    den = (wk ** 2 - w ** 2) + 1j * (2.0 * zk * wk * w)
    return (residue.unsqueeze(1) / den).sum(dim=-1)


def rand_query(n: int, k: int, device: torch.device) -> torch.Tensor:
    k = min(k, n)
    return torch.randperm(n, device=device)[:k]


def train_epoch(model, ds, opt, stats, args, device):
    model.train()
    order = list(range(len(ds)))
    random.shuffle(order)
    sums = np.zeros(4, dtype=np.float64)
    for i in order:
        t = to_tensors(ds[i], stats, device)
        q = rand_query(t["x"].shape[0], args.query_nodes, device)
        om_p, rz_p = model(t["x"], q)
        om_t, rz_t = norm_targets(t, stats, q)
        loss_om = nn.functional.mse_loss(om_p, om_t)
        loss_rz = nn.functional.mse_loss(rz_p, rz_t)
        loss_f = torch.zeros((), device=device)
        loss = loss_om + loss_rz
        if args.frf_loss_weight > 0:
            om, rz = denorm(om_p, rz_p, stats)
            pred = frf_from_modal(om, t["zeta"], rz, t["freq"])
            true = torch.complex(t["frf"][q, :, 0], t["frf"][q, :, 1])
            loss_f = nn.functional.mse_loss(torch.log10(torch.abs(pred) + 1e-20), torch.log10(torch.abs(true) + 1e-20))
            loss = loss + args.frf_loss_weight * loss_f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sums += np.array([float(loss.detach().cpu()), float(loss_om.detach().cpu()), float(loss_rz.detach().cpu()), float(loss_f.detach().cpu())])
    return sums / max(len(order), 1)


@torch.no_grad()
def evaluate(model, ds, stats, args, device):
    model.eval()
    rows: List[Dict[str, float]] = []
    for i in range(len(ds)):
        t = to_tensors(ds[i], stats, device)
        q = rand_query(t["x"].shape[0], args.eval_query_nodes, device)
        om_n, rz_n = model(t["x"], q)
        om, rz = denorm(om_n, rz_n, stats)
        om_err = torch.mean(torch.abs(om - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=1e-12))
        rz_err = torch.linalg.norm(rz - t["residue"][q]) / torch.clamp(torch.linalg.norm(t["residue"][q]), min=1e-20)
        pred = frf_from_modal(om, t["zeta"], rz, t["freq"])
        true = torch.complex(t["frf"][q, :, 0], t["frf"][q, :, 1])
        frf_err = torch.linalg.norm(pred - true) / torch.clamp(torch.linalg.norm(true), min=1e-20)
        rows.append({"sample": i, "omega_rel": float(om_err.cpu()), "residue_rel_l2": float(rz_err.cpu()), "frf_rel_l2": float(frf_err.cpu())})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ["omega_rel", "residue_rel_l2", "frf_rel_l2"]} if rows else {}
    return mean, rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data_modal_residue_filtered"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/modal_residue_baseline"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--query-nodes", type=int, default=512)
    p.add_argument("--eval-query-nodes", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--frf-loss-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train, val, test = H5Split(args.data_dir, "train"), H5Split(args.data_dir, "val"), H5Split(args.data_dir, "test")
    stats = compute_stats(train)
    np.savez(args.out_dir / "normalization_stats.npz", **stats)
    first = train[0]
    in_dim = node_input(first).shape[1]
    n_modes = first["modal_omega"].shape[0]
    model = ModalResidueNet(in_dim, n_modes, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best = float("inf")
    hist = []
    print(f">>> data={args.data_dir}, train/val/test={len(train)}/{len(val)}/{len(test)}, device={device}")
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train, opt, stats, args, device)
        sched.step()
        va, _ = evaluate(model, val, stats, args, device)
        hist.append({"epoch": ep, "train_loss": tr[0], "train_omega": tr[1], "train_residue": tr[2], "train_frf": tr[3], **{f"val_{k}": v for k, v in va.items()}})
        if va["frf_rel_l2"] < best:
            best = va["frf_rel_l2"]
            torch.save({"model": model.state_dict(), "stats": stats, "in_dim": in_dim, "n_modes": n_modes, "args": vars(args)}, args.out_dir / "best_model.pt")
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"ep {ep:04d} loss={tr[0]:.3e} om={tr[1]:.3e} rz={tr[2]:.3e} frf={tr[3]:.3e} | val om={va['omega_rel']:.3e} rz={va['residue_rel_l2']:.3e} frf={va['frf_rel_l2']:.3e}")
    write_csv(args.out_dir / "history.csv", hist)
    ckpt = torch.load(args.out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    va, vr = evaluate(model, val, stats, args, device)
    te, tr = evaluate(model, test, stats, args, device)
    write_csv(args.out_dir / "val_metrics.csv", vr)
    write_csv(args.out_dir / "test_metrics.csv", tr)
    summary = {"best_val_frf_rel_l2": best, "val": va, "test": te}
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
