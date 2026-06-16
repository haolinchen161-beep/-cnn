from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn

GEOM_SCALE = np.array([0.160, 0.060, 0.010], dtype=np.float32)
EPS_A = 1e-30


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
            out = {
                "points": g["points"][:].astype(np.float32),
                "edge_index": g["edge_index"][:].astype(np.int64),
                "edge_attr": g["edge_attr"][:].astype(np.float32),
                "point_features": g["point_features"][:].astype(np.float32),
                "spring_k_xyz": g["spring_k_xyz"][:].astype(np.float32),
                "spring_c_xyz": g["spring_c_xyz"][:].astype(np.float32),
                "node_type": g["node_type"][:].astype(np.int64),
                "modal_omega": g["modal_omega"][:].astype(np.float32),
                "modal_residue_z": g["modal_residue_z"][:].astype(np.float32),
                "excitation_coord": g["excitation_coord"][:].astype(np.float32),
            }
            if "excitation_index" in g:
                out["excitation_index"] = np.array(int(g["excitation_index"][()]), dtype=np.int64)
            else:
                dist = np.linalg.norm(out["points"] - out["excitation_coord"].reshape(1, 3), axis=1)
                out["excitation_index"] = np.array(int(np.argmin(dist)), dtype=np.int64)
            if "local_thickness_ratio" in g:
                out["local_thickness_ratio"] = g["local_thickness_ratio"][:].astype(np.float32)
            if "pocket_depth_ratio" in g:
                out["pocket_depth_ratio"] = g["pocket_depth_ratio"][:].astype(np.float32)
            return out


def node_input(s: Dict[str, np.ndarray]) -> np.ndarray:
    xyz = s["points"] / GEOM_SCALE.reshape(1, 3)
    k = np.log10(1.0 + np.maximum(s["spring_k_xyz"], 0.0)) / 8.0
    c = np.log10(1.0 + np.maximum(s["spring_c_xyz"], 0.0)) / 4.0
    nt = s["node_type"].astype(np.float32).reshape(-1, 1) / 4.0
    parts = [xyz, s["point_features"], k, c, nt]
    if "local_thickness_ratio" in s:
        parts.append(s["local_thickness_ratio"].reshape(-1, 1))
    if "pocket_depth_ratio" in s:
        parts.append(s["pocket_depth_ratio"].reshape(-1, 1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def np_A_scale(A: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(A.astype(np.float64) ** 2, axis=0) + EPS_A).astype(np.float32)


def compute_stats(ds: H5Split) -> Dict[str, np.ndarray]:
    sx = sx2 = None
    se = se2 = None
    n_node = 0
    n_edge = 0
    omega_logs = []
    A_log_scales = []

    for i in range(len(ds)):
        s = ds[i]
        x = node_input(s).astype(np.float64)
        e = s["edge_attr"].astype(np.float64)
        if sx is None:
            sx = np.zeros(x.shape[1], dtype=np.float64)
            sx2 = np.zeros(x.shape[1], dtype=np.float64)
            se = np.zeros(e.shape[1], dtype=np.float64)
            se2 = np.zeros(e.shape[1], dtype=np.float64)
        sx += x.sum(axis=0)
        sx2 += (x * x).sum(axis=0)
        se += e.sum(axis=0)
        se2 += (e * e).sum(axis=0)
        n_node += x.shape[0]
        n_edge += e.shape[0]
        omega_logs.append(np.log(np.maximum(s["modal_omega"], 1e-12)))
        A_log_scales.append(np.log(np.maximum(np_A_scale(s["modal_residue_z"]), 1e-30)))

    x_mean = sx / max(n_node, 1)
    x_std = np.sqrt(np.maximum(sx2 / max(n_node, 1) - x_mean * x_mean, 1e-12))
    e_mean = se / max(n_edge, 1)
    e_std = np.sqrt(np.maximum(se2 / max(n_edge, 1) - e_mean * e_mean, 1e-12))
    omega_log = np.stack(omega_logs, axis=0).astype(np.float32)
    A_log_scale = np.stack(A_log_scales, axis=0).astype(np.float32)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "edge_mean": e_mean.astype(np.float32),
        "edge_std": e_std.astype(np.float32),
        "omega_log_mean": omega_log.mean(axis=0).astype(np.float32),
        "omega_log_std": (omega_log.std(axis=0) + 1e-6).astype(np.float32),
        "A_log_scale_mean": A_log_scale.mean(axis=0).astype(np.float32),
        "A_log_scale_std": (A_log_scale.std(axis=0) + 1e-6).astype(np.float32),
    }


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(max(layers - 1, 1)):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class MeshGraphBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = mlp(hidden * 3, hidden, hidden, layers=2)
        self.msg_mlp = mlp(hidden * 3, hidden, hidden, layers=2)
        self.node_mlp = mlp(hidden * 2, hidden, hidden, layers=2)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        e = e.to(dtype=h.dtype)

        edge_in = torch.cat([h[src], h[dst], e], dim=-1)
        e_delta = self.edge_mlp(edge_in).to(dtype=e.dtype)
        e = self.edge_norm(e + e_delta).to(dtype=h.dtype)

        msg_in = torch.cat([h[src], h[dst], e], dim=-1)
        msg = self.msg_mlp(msg_in).to(dtype=h.dtype)
        agg = torch.zeros((h.shape[0], msg.shape[1]), dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
        agg = agg / torch.clamp(deg, min=1.0)

        node_in = torch.cat([h, agg], dim=-1)
        h_delta = self.node_mlp(node_in).to(dtype=h.dtype)
        h = self.node_norm(h + h_delta).to(dtype=h.dtype)
        return h, e


class MeshGraphModalResidueNet(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, n_modes: int, hidden: int = 64, gnn_layers: int = 2):
        super().__init__()
        self.node_encoder = mlp(node_in_dim, hidden, hidden, layers=3)
        self.edge_encoder = mlp(edge_in_dim, hidden, hidden, layers=3)
        self.blocks = nn.ModuleList([MeshGraphBlock(hidden) for _ in range(gnn_layers)])
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.omega_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_modes))
        self.A_scale_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_modes))
        self.A_shape_head = nn.Sequential(
            nn.Linear(3 * hidden + 6, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, n_modes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                coords_norm: torch.Tensor, q: torch.Tensor, exc_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr).to(dtype=h.dtype)
        for block in self.blocks:
            h, e = block(h, e, edge_index)

        g_raw = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0)
        g = self.global_mlp(g_raw)
        omega_norm = self.omega_head(g)

        exc_i = torch.clamp(exc_idx.long(), 0, h.shape[0] - 1)
        he0 = h[exc_i]
        A_scale_norm = self.A_scale_head(torch.cat([g, he0], dim=0))

        hq = h[q]
        he = he0.view(1, -1).expand(hq.shape[0], -1)
        gg = g.view(1, -1).expand(hq.shape[0], -1)
        q_xyz = coords_norm[q].to(dtype=h.dtype)
        rel_xyz = q_xyz - coords_norm[exc_i].view(1, 3).to(dtype=h.dtype)
        A_shape = self.A_shape_head(torch.cat([hq, gg, he, q_xyz, rel_xyz], dim=-1))
        return omega_norm, A_shape, A_scale_norm


def torch_A_scale(A: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(A.float() * A.float(), dim=0) + EPS_A)


def to_tensors(s: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    x = node_input(s)
    x = (x - stats["x_mean"][None, :]) / stats["x_std"][None, :]
    edge_attr = (s["edge_attr"] - stats["edge_mean"][None, :]) / stats["edge_std"][None, :]
    coords = s["points"] / GEOM_SCALE.reshape(1, 3)
    return {
        "x": torch.from_numpy(x.astype(np.float32)).to(device, non_blocking=True),
        "edge_index": torch.from_numpy(s["edge_index"]).long().to(device, non_blocking=True),
        "edge_attr": torch.from_numpy(edge_attr.astype(np.float32)).to(device, non_blocking=True),
        "coords": torch.from_numpy(coords.astype(np.float32)).to(device, non_blocking=True),
        "exc_idx": torch.as_tensor(int(s["excitation_index"]), dtype=torch.long, device=device),
        "omega": torch.from_numpy(s["modal_omega"]).to(device, non_blocking=True),
        "A": torch.from_numpy(s["modal_residue_z"]).to(device, non_blocking=True),
    }


def norm_targets(t: Dict[str, torch.Tensor], stats: Dict[str, np.ndarray], q: torch.Tensor):
    dev = t["x"].device
    om_m = torch.as_tensor(stats["omega_log_mean"], device=dev)
    om_s = torch.as_tensor(stats["omega_log_std"], device=dev)
    As_m = torch.as_tensor(stats["A_log_scale_mean"], device=dev)
    As_s = torch.as_tensor(stats["A_log_scale_std"], device=dev)

    omega_norm = (torch.log(torch.clamp(t["omega"], min=1e-12)) - om_m) / om_s
    A_scale = torch_A_scale(t["A"])
    A_shape = t["A"][q].float() / A_scale.view(1, -1)
    A_scale_norm = (torch.log(torch.clamp(A_scale, min=1e-30)) - As_m) / As_s
    return omega_norm, A_shape, A_scale_norm, A_scale


def denorm_outputs(omega_norm: torch.Tensor, A_shape: torch.Tensor, A_scale_norm: torch.Tensor, stats: Dict[str, np.ndarray]):
    dev = omega_norm.device
    omega = torch.exp(omega_norm.float() * torch.as_tensor(stats["omega_log_std"], device=dev)
                      + torch.as_tensor(stats["omega_log_mean"], device=dev))
    A_scale = torch.exp(A_scale_norm.float() * torch.as_tensor(stats["A_log_scale_std"], device=dev)
                        + torch.as_tensor(stats["A_log_scale_mean"], device=dev))
    A = A_shape.float() * A_scale.view(1, -1)
    return omega, A, A_scale


def rand_query(n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.randperm(n, device=device)[:min(k, n)]


def eval_query(n: int, k: int, device: torch.device) -> torch.Tensor:
    if k <= 0 or k >= n:
        return torch.arange(n, device=device)
    return torch.linspace(0, n - 1, steps=k, device=device).round().long().unique()


def triplet_percent(values: List[np.ndarray]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    v = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in values])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0, 0.0
    return float(v.mean()), float(v.max()), float(np.sqrt(np.mean(v * v)))


def mode_mean(values: List[np.ndarray], n_modes: int) -> np.ndarray:
    if not values:
        return np.full(n_modes, np.nan, dtype=np.float64)
    arr = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in values], axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def A_visible_error_pct(pred: torch.Tensor, true: torch.Tensor, visible_rel: float) -> torch.Tensor:
    denom = torch.linalg.norm(true.float(), dim=0)
    scale = torch.clamp(torch.max(denom), min=1e-20)
    visible = denom > (visible_rel * scale)
    err = torch.linalg.norm(pred.float() - true.float(), dim=0) / torch.clamp(denom, min=1e-20) * 100.0
    return torch.where(visible, err, torch.full_like(err, float("nan")))


def fmt_triplet(t: Tuple[float, float, float], ndigits: int = 1) -> str:
    return "/".join(f"{x:.{ndigits}f}" for x in t)


def fmt_loss(v: float) -> str:
    return f"{v:.1f}" if abs(v) >= 0.1 else f"{v:.3e}"


def row_add_triplet(row: Dict[str, float], prefix: str, t: Tuple[float, float, float]) -> None:
    row[f"{prefix}_mean_pct"] = t[0]
    row[f"{prefix}_max_pct"] = t[1]
    row[f"{prefix}_rms_pct"] = t[2]


def row_add_modes(row: Dict[str, float], prefix: str, values: np.ndarray) -> None:
    for i, v in enumerate(values, start=1):
        row[f"{prefix}{i:02d}_pct"] = float(v) if np.isfinite(v) else float("nan")


def modal_score(metrics: Dict[str, Tuple[float, float, float]], best_a_weight: float) -> float:
    return float(metrics["w10_triplet"][2] + best_a_weight * metrics["A_vis_triplet"][2])


def forward_model(model, t, q):
    return model(t["x"], t["edge_index"], t["edge_attr"], t["coords"], q, t["exc_idx"])


def train_epoch(model, ds, opt, scaler, stats, args, device):
    model.train()
    order = list(range(len(ds)))
    random.shuffle(order)
    sums = np.zeros(4, dtype=np.float64)
    omega_errs: List[np.ndarray] = []
    A_vis_errs: List[np.ndarray] = []
    A_scale_errs: List[np.ndarray] = []
    amp_enabled = bool(args.fp16 and device.type == "cuda")

    for i in order:
        t = to_tensors(ds[i], stats, device)
        q = rand_query(t["x"].shape[0], args.query_nodes, device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            omega_p, A_shape_p, A_scale_p = forward_model(model, t, q)
            omega_t, A_shape_t, A_scale_t, A_scale_true = norm_targets(t, stats, q)
            loss_omega = nn.functional.mse_loss(omega_p.float(), omega_t.float())
            loss_A_shape = nn.functional.mse_loss(A_shape_p.float(), A_shape_t.float())
            loss_A_scale = nn.functional.mse_loss(A_scale_p.float(), A_scale_t.float())
            loss = args.omega_loss_weight * loss_omega + args.a_shape_loss_weight * loss_A_shape + args.a_scale_loss_weight * loss_A_scale

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            omega_phys, A_phys, A_scale_phys = denorm_outputs(omega_p.detach(), A_shape_p.detach(), A_scale_p.detach(), stats)
            w_rel = torch.abs(omega_phys - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=1e-12) * 100.0
            A_true = t["A"][q]
            A_vis = A_visible_error_pct(A_phys, A_true, args.residue_visible_rel)
            A_scale_rel = torch.abs(A_scale_phys - A_scale_true) / torch.clamp(A_scale_true, min=1e-20) * 100.0
            omega_errs.append(w_rel.detach().cpu().numpy())
            A_vis_errs.append(A_vis.detach().cpu().numpy())
            A_scale_errs.append(A_scale_rel.detach().cpu().numpy())

        sums += np.array([float(loss.detach().cpu()), float(loss_omega.detach().cpu()), float(loss_A_shape.detach().cpu()), float(loss_A_scale.detach().cpu())])

    n_modes = len(omega_errs[0]) if omega_errs else 0
    return sums / max(len(order), 1), {
        "w10_triplet": triplet_percent(omega_errs),
        "A_vis_triplet": triplet_percent(A_vis_errs),
        "A_scale_triplet": triplet_percent(A_scale_errs),
        "w_modes": mode_mean(omega_errs, n_modes),
        "A_vis_modes": mode_mean(A_vis_errs, n_modes),
        "A_scale_modes": mode_mean(A_scale_errs, n_modes),
    }


@torch.no_grad()
def evaluate(model, ds, stats, args, device):
    model.eval()
    rows: List[Dict[str, float]] = []
    omega_errs: List[np.ndarray] = []
    A_vis_errs: List[np.ndarray] = []
    A_scale_errs: List[np.ndarray] = []
    amp_enabled = bool(args.fp16 and device.type == "cuda")

    for i in range(len(ds)):
        t = to_tensors(ds[i], stats, device)
        q = eval_query(t["x"].shape[0], args.eval_query_nodes, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            omega_n, A_shape_n, A_scale_n = forward_model(model, t, q)
        _, _, _, A_scale_true = norm_targets(t, stats, q)
        omega, A, A_scale = denorm_outputs(omega_n, A_shape_n, A_scale_n, stats)

        w_rel = torch.abs(omega - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=1e-12) * 100.0
        A_true = t["A"][q]
        A_vis = A_visible_error_pct(A, A_true, args.residue_visible_rel)
        A_scale_rel = torch.abs(A_scale - A_scale_true) / torch.clamp(A_scale_true, min=1e-20) * 100.0

        w_np = w_rel.detach().cpu().numpy()
        A_np = A_vis.detach().cpu().numpy()
        S_np = A_scale_rel.detach().cpu().numpy()
        omega_errs.append(w_np)
        A_vis_errs.append(A_np)
        A_scale_errs.append(S_np)

        row = {"sample": i}
        row_add_triplet(row, "w10", triplet_percent([w_np]))
        row_add_triplet(row, "A_vis", triplet_percent([A_np]))
        row_add_triplet(row, "A_scale", triplet_percent([S_np]))
        row_add_modes(row, "w", w_np)
        row_add_modes(row, "A_vis", A_np)
        row_add_modes(row, "A_scale", S_np)
        rows.append(row)

    n_modes = len(omega_errs[0]) if omega_errs else 0
    mean = {
        "w10_triplet": triplet_percent(omega_errs),
        "A_vis_triplet": triplet_percent(A_vis_errs),
        "A_scale_triplet": triplet_percent(A_scale_errs),
        "w_modes": mode_mean(omega_errs, n_modes),
        "A_vis_modes": mode_mean(A_vis_errs, n_modes),
        "A_scale_modes": mode_mean(A_scale_errs, n_modes),
    }
    mean["modal_score"] = modal_score(mean, args.best_a_weight)
    return mean, rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def append_csv_row(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, epoch, best_value):
    return {
        "model": model.state_dict(),
        "stats": stats,
        "node_in_dim": in_dim,
        "edge_in_dim": edge_dim,
        "n_modes": n_modes,
        "epoch": epoch,
        "best_modal_score": best_value,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data_modal_residue_filtered"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/modal_residue_baseline"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--query-nodes", type=int, default=512)
    p.add_argument("--eval-query-nodes", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--omega-loss-weight", type=float, default=1.0)
    p.add_argument("--a-shape-loss-weight", type=float, default=1.0)
    p.add_argument("--a-scale-loss-weight", type=float, default=0.2)
    p.add_argument("--best-a-weight", type=float, default=0.01)
    p.add_argument("--residue-visible-rel", type=float, default=1e-3)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 兼容旧入口参数；如果外部还传 --phi-loss-weight，则映射到 A_shape。
    p.add_argument("--phi-loss-weight", type=float, default=None)
    args = p.parse_args()
    if args.phi_loss_weight is not None:
        args.a_shape_loss_weight = args.phi_loss_weight

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = args.out_dir / "training_log.csv"
    if log_csv.exists():
        log_csv.unlink()

    device = torch.device(args.device)
    train = H5Split(args.data_dir, "train")
    val = H5Split(args.data_dir, "val")
    test = H5Split(args.data_dir, "test")
    stats = compute_stats(train)
    np.savez(args.out_dir / "normalization_stats.npz", **stats)

    first = train[0]
    in_dim = node_input(first).shape[1]
    edge_dim = first["edge_attr"].shape[1]
    n_modes = first["modal_omega"].shape[0]
    model = MeshGraphModalResidueNet(in_dim, edge_dim, n_modes, args.hidden, args.gnn_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> data={args.data_dir}, train/val/test={len(train)}/{len(val)}/{len(test)}, device={device}, fp16={args.fp16}")
    print(f">>> MeshGraph modal-residue model: node_dim={in_dim}, edge_dim={edge_dim}, hidden={args.hidden}, layers={args.gnn_layers}, params={total_params:,}")
    print(">>> A decomposition: A_r(x)=A_scale_r * A_shape_r(x); A_scale is sample/mode RMS of modal_residue_z.")

    best = float("inf")
    hist: List[Dict[str, float]] = []
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_m = train_epoch(model, train, opt, scaler, stats, args, device)
        sched.step()
        va, _ = evaluate(model, val, stats, args, device)
        score = float(va["modal_score"])

        if score < best:
            best = score
            torch.save(checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, ep, best), args.out_dir / "best_model.pt")
        torch.save(checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, ep, best), args.out_dir / "last_model.pt")

        row: Dict[str, float] = {
            "epoch": ep,
            "lr": float(opt.param_groups[0]["lr"]),
            "loss": float(tr_loss[0]),
            "loss_w": float(tr_loss[1]),
            "loss_A_shape": float(tr_loss[2]),
            "loss_A_scale": float(tr_loss[3]),
            "val_modal_score": score,
        }
        for prefix, metrics in [("train", tr_m), ("val", va)]:
            row_add_triplet(row, f"{prefix}_w10", metrics["w10_triplet"])
            row_add_triplet(row, f"{prefix}_A_vis", metrics["A_vis_triplet"])
            row_add_triplet(row, f"{prefix}_A_scale", metrics["A_scale_triplet"])
            row_add_modes(row, f"{prefix}_w", metrics["w_modes"])
            row_add_modes(row, f"{prefix}_A_vis", metrics["A_vis_modes"])
            row_add_modes(row, f"{prefix}_A_scale", metrics["A_scale_modes"])
        hist.append(row)
        append_csv_row(log_csv, row)

        if ep == 1 or ep % max(args.log_every, 1) == 0 or ep == args.epochs:
            print(
                f"Epoch {ep:4d} | "
                f"w10[mean/max/rms]=[{fmt_triplet(tr_m['w10_triplet'], 1)}]%  "
                f"A_vis[mean/max/rms]=[{fmt_triplet(tr_m['A_vis_triplet'], 1)}]%  "
                f"A_scale[mean/max/rms]=[{fmt_triplet(tr_m['A_scale_triplet'], 1)}]%  "
                f"loss={fmt_loss(float(tr_loss[0]))}(w={fmt_loss(float(tr_loss[1]))},Ash={fmt_loss(float(tr_loss[2]))},Asc={fmt_loss(float(tr_loss[3]))})"
            )
            print(
                f"Val modal | "
                f"w10[mean/max/rms]=[{fmt_triplet(va['w10_triplet'], 3)}]%  "
                f"A_vis[mean/max/rms]=[{fmt_triplet(va['A_vis_triplet'], 1)}]%  "
                f"A_scale[mean/max/rms]=[{fmt_triplet(va['A_scale_triplet'], 1)}]%"
            )

    write_csv(args.out_dir / "history.csv", hist)
    ckpt = torch.load(args.out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    va, vr = evaluate(model, val, stats, args, device)
    te, tr = evaluate(model, test, stats, args, device)
    write_csv(args.out_dir / "val_metrics.csv", vr)
    write_csv(args.out_dir / "test_metrics.csv", tr)

    summary = {
        "best_modal_score": best,
        "val": {
            "w10_mean_max_rms_pct": list(va["w10_triplet"]),
            "A_vis_mean_max_rms_pct": list(va["A_vis_triplet"]),
            "A_scale_mean_max_rms_pct": list(va["A_scale_triplet"]),
            "modal_score": va["modal_score"],
        },
        "test": {
            "w10_mean_max_rms_pct": list(te["w10_triplet"]),
            "A_vis_mean_max_rms_pct": list(te["A_vis_triplet"]),
            "A_scale_mean_max_rms_pct": list(te["A_scale_triplet"]),
            "modal_score": te["modal_score"],
        },
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
