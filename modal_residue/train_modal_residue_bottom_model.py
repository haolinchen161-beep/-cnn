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

from modal_residue import train_modal_residue_model as base


EPS_A = base.EPS_A
GEOM_SCALE = base.GEOM_SCALE


def seed_all(seed: int) -> None:
    base.seed_all(seed)


class H5Split:
    """HDF5 split loader that keeps the full graph but exposes bottom masks for target-node selection."""

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
            n = g["points"].shape[0]
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
                "pocket_bottom_mask": g["pocket_bottom_mask"][:].astype(np.uint8) if "pocket_bottom_mask" in g else np.zeros(n, dtype=np.uint8),
                "cut_region_mask": g["cut_region_mask"][:].astype(np.uint8) if "cut_region_mask" in g else np.zeros(n, dtype=np.uint8),
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


def _mask_feature(s: Dict[str, np.ndarray], key: str) -> np.ndarray:
    n = s["points"].shape[0]
    if key not in s:
        return np.zeros((n, 1), dtype=np.float32)
    m = np.asarray(s[key]).reshape(-1)
    if m.shape[0] != n:
        mm = np.zeros(n, dtype=np.float32)
        mm[: min(n, m.shape[0])] = m[: min(n, m.shape[0])]
        m = mm
    return m.astype(np.float32).reshape(-1, 1)


def node_input(s: Dict[str, np.ndarray]) -> np.ndarray:
    """Use the same inputs as the base trainer, plus explicit bottom/cut masks."""
    xyz = s["points"] / GEOM_SCALE.reshape(1, 3)
    k = np.log10(1.0 + np.maximum(s["spring_k_xyz"], 0.0)) / 8.0
    c = np.log10(1.0 + np.maximum(s["spring_c_xyz"], 0.0)) / 4.0
    nt = s["node_type"].astype(np.float32).reshape(-1, 1) / 4.0
    parts = [xyz, s["point_features"], k, c, nt]
    if "local_thickness_ratio" in s:
        parts.append(s["local_thickness_ratio"].reshape(-1, 1))
    if "pocket_depth_ratio" in s:
        parts.append(s["pocket_depth_ratio"].reshape(-1, 1))
    parts.append(_mask_feature(s, "pocket_bottom_mask"))
    parts.append(_mask_feature(s, "cut_region_mask"))
    return np.concatenate(parts, axis=1).astype(np.float32)


def target_mask_np(s: Dict[str, np.ndarray], target_region: str) -> np.ndarray:
    n = s["points"].shape[0]
    if target_region == "bottom":
        m = np.asarray(s.get("pocket_bottom_mask", np.zeros(n, dtype=np.uint8))).reshape(-1).astype(bool)
        if m.shape[0] != n:
            mm = np.zeros(n, dtype=bool)
            mm[: min(n, m.shape[0])] = m[: min(n, m.shape[0])]
            m = mm
        if np.any(m):
            return m
    return np.ones(n, dtype=bool)


def np_A_scale(A: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(A.astype(np.float64) ** 2, axis=0) + EPS_A).astype(np.float32)


def compute_stats(ds: H5Split, data_dir: Path | None = None, target_region: str = "bottom") -> Dict[str, np.ndarray]:
    sx = sx2 = None
    se = se2 = None
    n_node = 0
    n_edge = 0
    omega_logs = []
    A_sample_scales = []
    target_counts = []

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

        m = target_mask_np(s, target_region)
        target_counts.append(int(np.sum(m)))
        A_target = s["modal_residue_z"][m]
        if A_target.shape[0] == 0:
            A_target = s["modal_residue_z"]
        A_sample_scales.append(np_A_scale(A_target))

    x_mean = sx / max(n_node, 1)
    x_std = np.sqrt(np.maximum(sx2 / max(n_node, 1) - x_mean * x_mean, 1e-12))
    e_mean = se / max(n_edge, 1)
    e_std = np.sqrt(np.maximum(se2 / max(n_edge, 1) - e_mean * e_mean, 1e-12))
    omega_log = np.stack(omega_logs, axis=0).astype(np.float32)
    A_sample_scale = np.stack(A_sample_scales, axis=0).astype(np.float32)

    # For bottom-target training, do not reuse full-node external scales; they are often too broad.
    A_asinh_scale = np.median(A_sample_scale.astype(np.float64), axis=0).astype(np.float32)
    A_asinh_scale = np.maximum(A_asinh_scale, np.float32(1e-12))

    target_counts_arr = np.asarray(target_counts, dtype=np.float32)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "edge_mean": e_mean.astype(np.float32),
        "edge_std": e_std.astype(np.float32),
        "omega_log_mean": omega_log.mean(axis=0).astype(np.float32),
        "omega_log_std": (omega_log.std(axis=0) + 1e-6).astype(np.float32),
        "A_asinh_scale": A_asinh_scale.astype(np.float32),
        "A_sample_rms_median": np.median(A_sample_scale.astype(np.float64), axis=0).astype(np.float32),
        "A_sample_rms_p10": np.percentile(A_sample_scale.astype(np.float64), 10, axis=0).astype(np.float32),
        "A_sample_rms_p90": np.percentile(A_sample_scale.astype(np.float64), 90, axis=0).astype(np.float32),
        "target_node_count_min": np.array(float(np.min(target_counts_arr)), dtype=np.float32),
        "target_node_count_p10": np.array(float(np.percentile(target_counts_arr, 10)), dtype=np.float32),
        "target_node_count_median": np.array(float(np.median(target_counts_arr)), dtype=np.float32),
        "target_node_count_max": np.array(float(np.max(target_counts_arr)), dtype=np.float32),
    }


def to_tensors(s: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    x = node_input(s)
    x = (x - stats["x_mean"][None, :]) / stats["x_std"][None, :]
    edge_attr = (s["edge_attr"] - stats["edge_mean"][None, :]) / stats["edge_std"][None, :]
    coords = s["points"] / GEOM_SCALE.reshape(1, 3)
    bottom_mask = np.asarray(s.get("pocket_bottom_mask", np.zeros(s["points"].shape[0], dtype=np.uint8))).reshape(-1).astype(bool)
    if bottom_mask.shape[0] != s["points"].shape[0]:
        fixed = np.zeros(s["points"].shape[0], dtype=bool)
        fixed[: min(len(fixed), len(bottom_mask))] = bottom_mask[: min(len(fixed), len(bottom_mask))]
        bottom_mask = fixed
    return {
        "x": torch.from_numpy(x.astype(np.float32)).to(device, non_blocking=True),
        "edge_index": torch.from_numpy(s["edge_index"]).long().to(device, non_blocking=True),
        "edge_attr": torch.from_numpy(edge_attr.astype(np.float32)).to(device, non_blocking=True),
        "coords": torch.from_numpy(coords.astype(np.float32)).to(device, non_blocking=True),
        "exc_idx": torch.as_tensor(int(s["excitation_index"]), dtype=torch.long, device=device),
        "omega": torch.from_numpy(s["modal_omega"]).to(device, non_blocking=True),
        "A": torch.from_numpy(s["modal_residue_z"]).to(device, non_blocking=True),
        "bottom_mask": torch.from_numpy(bottom_mask).bool().to(device, non_blocking=True),
    }


def residue_scale_tensor(stats: Dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(stats["A_asinh_scale"], dtype=torch.float32, device=device).view(1, -1)


def norm_targets(t: Dict[str, torch.Tensor], stats: Dict[str, np.ndarray], q: torch.Tensor):
    dev = t["x"].device
    om_m = torch.as_tensor(stats["omega_log_mean"], device=dev)
    om_s = torch.as_tensor(stats["omega_log_std"], device=dev)
    scale = residue_scale_tensor(stats, dev)
    omega_norm = (torch.log(torch.clamp(t["omega"], min=1e-12)) - om_m) / om_s
    A_true = t["A"][q].float()
    Y_true = torch.asinh(A_true / torch.clamp(scale, min=1e-12))
    return omega_norm, Y_true, A_true, scale


def select_query(t: Dict[str, torch.Tensor], k: int, target_region: str, random_sample: bool) -> torch.Tensor:
    n = int(t["x"].shape[0])
    if target_region == "bottom" and "bottom_mask" in t and bool(t["bottom_mask"].any()):
        candidates = torch.where(t["bottom_mask"])[0]
    else:
        candidates = torch.arange(n, device=t["x"].device)
    if candidates.numel() == 0:
        candidates = torch.arange(n, device=t["x"].device)

    if k <= 0 or k >= candidates.numel():
        return candidates.long()
    if random_sample:
        perm = torch.randperm(candidates.numel(), device=t["x"].device)[: int(k)]
        return candidates[perm].long()
    pos = torch.linspace(0, candidates.numel() - 1, steps=int(k), device=t["x"].device).round().long().unique()
    return candidates[pos].long()


def A_sign_accuracy_pct(pred: torch.Tensor, true: torch.Tensor, visible_rel: float) -> torch.Tensor:
    pred_f = pred.float()
    true_f = true.float()
    n_nodes, n_modes = true_f.shape
    out = []
    for r in range(n_modes):
        a = true_f[:, r]
        th = torch.clamp(torch.max(torch.abs(a)) * float(visible_rel), min=1e-20)
        mask = torch.abs(a) > th
        if not bool(mask.any()):
            out.append(torch.full((), float("nan"), device=true.device))
            continue
        acc = (torch.sign(pred_f[mask, r]) == torch.sign(a[mask])).float().mean() * 100.0
        out.append(acc)
    return torch.stack(out, dim=0)


def forward_model(model, t, q):
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
        t = to_tensors(ds[i], stats, device)
        q = select_query(t, args.query_nodes, args.target_region, random_sample=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            omega_p, Y_p = forward_model(model, t, q)
            omega_t, Y_t, A_t, scale = norm_targets(t, stats, q)
            loss_omega = nn.functional.mse_loss(omega_p.float(), omega_t.float())
            loss_full = nn.functional.smooth_l1_loss(Y_p.float(), Y_t.float())
            A_p = base.asinh_to_physical(Y_p, scale, clamp=args.asinh_clamp)
            top_mask = base.top_mode_mask(A_t, args.top_node_frac)
            dom_mask = base.node_dominant_mask(A_t, args.node_dominant_k)
            loss_top = base.masked_physical_smooth_l1(A_p, A_t, scale, top_mask)
            loss_dom = base.masked_physical_smooth_l1(A_p, A_t, scale, dom_mask)
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
            w_rel = torch.abs(omega_phys - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=1e-12) * 100.0
            A_true = t["A"][q]
            A_vis = base.A_visible_error_pct(A_phys, A_true, args.residue_visible_rel)
            A_top = base.A_top_error_pct(A_phys, A_true, args.top_node_frac)
            A_sign = A_sign_accuracy_pct(A_phys, A_true, args.sign_visible_rel)
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
        "w10_triplet": base.triplet_percent(omega_errs),
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
        t = to_tensors(ds[i], stats, device)
        q = select_query(t, args.eval_query_nodes, args.target_region, random_sample=False)
        omega_t, Y_t, A_true, _ = norm_targets(t, stats, q)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            omega_n, Y_n = forward_model(model, t, q)
        omega, A = base.denorm_outputs(omega_n, Y_n, stats, clamp=args.asinh_clamp)

        w_rel = torch.abs(omega - t["omega"]) / torch.clamp(torch.abs(t["omega"]), min=1e-12) * 100.0
        A_vis = base.A_visible_error_pct(A, A_true, args.residue_visible_rel)
        A_top = base.A_top_error_pct(A, A_true, args.top_node_frac)
        A_sign = A_sign_accuracy_pct(A, A_true, args.sign_visible_rel)
        y_loss = nn.functional.smooth_l1_loss(Y_n.float(), Y_t.float()).detach()

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
        base.row_add_triplet(row, "w10", base.triplet_percent([w_np]))
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
        "w10_triplet": base.triplet_percent(omega_errs),
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
    mean["modal_score"] = modal_score(mean, args.best_a_weight)
    return mean, rows


def modal_score(metrics, best_a_weight):
    y_triplet = metrics.get("Y_smooth_l1_triplet")
    if y_triplet is None:
        return float(metrics["w10_triplet"][2] + best_a_weight * metrics["A_vis_triplet"][2])
    y_rms = float(y_triplet[2])
    w_rms = float(metrics.get("w10_triplet", (0.0, 0.0, 0.0))[2])
    a_vis_mean = float(metrics.get("A_vis_triplet", (0.0, 0.0, 0.0))[0])
    sign_mean = float(metrics.get("A_sign_triplet", (0.0, 0.0, 0.0))[0])
    return float(y_rms + 0.05 * w_rms + 0.001 * a_vis_mean + 0.0002 * max(0.0, 100.0 - sign_mean))


def checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, epoch, best_value):
    payload = base.checkpoint_payload(model, stats, in_dim, edge_dim, n_modes, args, epoch, best_value)
    payload["target_region"] = args.target_region
    payload["target_transform"] = "signed_asinh_fixed_per_mode_scale_bottom_region"
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("modal_residue/data_modal_residue_fixedclamp300"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/modal_residue_bottom_asinh_fixedclamp300"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--query-nodes", type=int, default=256, help="Training target nodes sampled from target region; 0 means all target-region nodes.")
    p.add_argument("--eval-query-nodes", type=int, default=0, help="Evaluation target nodes; 0 means all target-region nodes.")
    p.add_argument("--target-region", choices=["bottom", "all"], default="bottom")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--omega-loss-weight", type=float, default=1.0)
    p.add_argument("--residue-full-loss-weight", type=float, default=1.0)
    p.add_argument("--top-aux-loss-weight", type=float, default=0.2)
    p.add_argument("--node-dominant-loss-weight", type=float, default=0.1)
    p.add_argument("--top-node-frac", type=float, default=0.10)
    p.add_argument("--node-dominant-k", type=int, default=1)
    p.add_argument("--asinh-clamp", type=float, default=20.0)
    p.add_argument("--best-a-weight", type=float, default=0.01)
    p.add_argument("--residue-visible-rel", type=float, default=1e-3)
    p.add_argument("--sign-visible-rel", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = args.out_dir / "training_log.csv"
    if log_csv.exists():
        log_csv.unlink()

    device = torch.device(args.device)
    train = H5Split(args.data_dir, "train")
    val = H5Split(args.data_dir, "val")
    test = H5Split(args.data_dir, "test")
    stats = compute_stats(train, args.data_dir, target_region=args.target_region)
    np.savez(args.out_dir / "normalization_stats.npz", **stats)

    first = train[0]
    in_dim = node_input(first).shape[1]
    edge_dim = first["edge_attr"].shape[1]
    n_modes = first["modal_omega"].shape[0]
    model = base.MeshGraphModalResidueNet(in_dim, edge_dim, n_modes, args.hidden, args.gnn_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> data={args.data_dir}, train/val/test={len(train)}/{len(val)}/{len(test)}, device={device}, fp16={args.fp16}")
    print(f">>> Bottom-region modal-residue model: node_dim={in_dim}, edge_dim={edge_dim}, hidden={args.hidden}, layers={args.gnn_layers}, params={total_params:,}")
    print(f">>> Target region={args.target_region}; train_query={args.query_nodes if args.query_nodes > 0 else 'all region nodes'}, eval_query={args.eval_query_nodes if args.eval_query_nodes > 0 else 'all region nodes'}")
    print(f">>> target node count in train: min={float(stats['target_node_count_min']):.0f}, p10={float(stats['target_node_count_p10']):.0f}, median={float(stats['target_node_count_median']):.0f}, max={float(stats['target_node_count_max']):.0f}")
    print(">>> Small bottom pockets are handled by using all available bottom nodes; no duplicate padding is used.")
    print(f">>> s_mode={np.array2string(stats['A_asinh_scale'], precision=6, separator=', ')}")

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
            "lr": float(sched.get_last_lr()[0]),
            "loss": float(tr_loss[0]),
            "loss_w": float(tr_loss[1]),
            "loss_Y_full": float(tr_loss[2]),
            "loss_A_aux_weighted": float(tr_loss[3]),
            "val_modal_score": score,
        }
        for prefix, metrics in [("train", tr_m), ("val", va)]:
            base.row_add_triplet(row, f"{prefix}_w10", metrics["w10_triplet"])
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
        base.append_csv_row(log_csv, row)

        if ep == 1 or ep % max(args.log_every, 1) == 0 or ep == args.epochs:
            print(
                f"Epoch {ep:4d} | "
                f"w10=[{base.fmt_triplet(tr_m['w10_triplet'], 1)}]%  "
                f"A_region_vis=[{base.fmt_triplet(tr_m['A_vis_triplet'], 1)}]%  "
                f"A_region_top=[{base.fmt_triplet(tr_m['A_top_triplet'], 1)}]%  "
                f"sign=[{base.fmt_triplet(tr_m['A_sign_triplet'], 1)}]%  "
                f"loss={base.fmt_loss(float(tr_loss[0]))}(w={base.fmt_loss(float(tr_loss[1]))},Y={base.fmt_loss(float(tr_loss[2]))},Aaux={base.fmt_loss(float(tr_loss[3]))})"
            )
            print(
                f"Val region | "
                f"n=[{base.fmt_triplet(va['n_target_nodes_triplet'], 0)}]  "
                f"w10=[{base.fmt_triplet(va['w10_triplet'], 3)}]%  "
                f"A_vis=[{base.fmt_triplet(va['A_vis_triplet'], 1)}]%  "
                f"A_top=[{base.fmt_triplet(va['A_top_triplet'], 1)}]%  "
                f"sign=[{base.fmt_triplet(va['A_sign_triplet'], 1)}]%  "
                f"Y=[{base.fmt_triplet(va['Y_smooth_l1_triplet'], 4)}]"
            )

    base.write_csv(args.out_dir / "history.csv", hist)
    ckpt = torch.load(args.out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    va, vr = evaluate(model, val, stats, args, device)
    te, tr = evaluate(model, test, stats, args, device)
    base.write_csv(args.out_dir / "val_metrics.csv", vr)
    base.write_csv(args.out_dir / "test_metrics.csv", tr)

    summary = {
        "target_transform": "signed_asinh_fixed_per_mode_scale_bottom_region",
        "target_region": args.target_region,
        "query_nodes": args.query_nodes,
        "eval_query_nodes": args.eval_query_nodes,
        "target_node_count_train_min_p10_median_max": [
            float(stats["target_node_count_min"]),
            float(stats["target_node_count_p10"]),
            float(stats["target_node_count_median"]),
            float(stats["target_node_count_max"]),
        ],
        "A_asinh_scale": stats["A_asinh_scale"].astype(float).tolist(),
        "best_modal_score": best,
        "val": {
            "w10_mean_max_rms_pct": list(va["w10_triplet"]),
            "A_vis_mean_max_rms_pct": list(va["A_vis_triplet"]),
            "A_top_mean_max_rms_pct": list(va["A_top_triplet"]),
            "A_sign_mean_max_rms_pct": list(va["A_sign_triplet"]),
            "Y_smooth_l1_mean_max_rms": list(va["Y_smooth_l1_triplet"]),
            "n_target_nodes_mean_max_rms": list(va["n_target_nodes_triplet"]),
            "modal_score": va["modal_score"],
        },
        "test": {
            "w10_mean_max_rms_pct": list(te["w10_triplet"]),
            "A_vis_mean_max_rms_pct": list(te["A_vis_triplet"]),
            "A_top_mean_max_rms_pct": list(te["A_top_triplet"]),
            "A_sign_mean_max_rms_pct": list(te["A_sign_triplet"]),
            "Y_smooth_l1_mean_max_rms": list(te["Y_smooth_l1_triplet"]),
            "n_target_nodes_mean_max_rms": list(te["n_target_nodes_triplet"]),
            "modal_score": te["modal_score"],
        },
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
