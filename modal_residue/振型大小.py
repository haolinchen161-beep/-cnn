# -*- coding: utf-8 -*-
"""
查看 modal_residue_z / 单模态 FRF 贡献 / 总 FRF 峰值的节点云图。

默认读取：
    F:\毕业论文\-cnn-modal-residue-frf\data_modal_residue_filtered100

这个版本重点做“残差大小前百分之多少”的区域筛选：
1. 每一阶模态分别取 |A_r(x)| 最大的前 p% 节点；
2. 对所选模态做 union，得到候选 residue ROI；
3. 输出一张只显示这些 top residue 节点的云图；
4. 合并输出 summary.csv / node_metrics_wide.csv / top_nodes_combined.csv。

物理关系：
    A_r(x) = modal_residue_z(x,r) = phi_r,z(x) * phi_r,z(x_f)

同一个样本、同一阶模态、在该阶共振附近，分母对所有节点相同，因此：
    |A_r(x)| 越大，该阶单模态 FRF 贡献越大。

但总 FRF 是多阶复数叠加，所以：
    |A_r(x)| 大 ≠ 总 FRF 峰值一定最大。
如果 full_FRF_peak 被第 1 阶主导，用 full_FRF_peak top 节点选 ROI 会偏向第 1 阶；
因此更推荐用“每阶 |A| top p% 的 union”来做 residue 训练区域。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ========================= 用户主要改这里 =========================
DATA_DIR = Path(r"F:\毕业论文\-cnn-modal-residue-frf\data_modal_residue_filtered100")
OUT_DIR = None  # None 表示输出到 DATA_DIR / "residue_cloud_maps"

SPLIT = "train"
SAMPLE_INDEX = 0
MODES = list(range(1, 11))

RESIDUE_TOP_PERCENT = 10.0   # 每阶 |A_r(x)| 最大的前百分之多少节点，用于 residue ROI
FRF_TOP_PERCENT = 10.0       # 总 FRF 峰值最大的前百分之多少节点，用于对比
DPI = 220
POINT_SIZE = 7
USE_LOG_COLOR = True
MAX_PLOT_NODES = 12000
SAVE_PER_MODE_IMAGES = False
# ================================================================


def sorted_keys(f: h5py.File) -> List[str]:
    return sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))


def load_sample(data_dir: Path, split: str, sample_index: int) -> Tuple[str, Dict[str, np.ndarray]]:
    h5_path = data_dir / f"{split}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"找不到 H5 文件: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        keys = sorted_keys(f)
        if not keys:
            raise RuntimeError(f"H5 中没有样本: {h5_path}")
        if sample_index < 0 or sample_index >= len(keys):
            raise IndexError(f"sample_index={sample_index} 超出范围；{split}.h5 共有 {len(keys)} 个样本")
        key = keys[sample_index]
        g = f[key]
        required = ["points", "modal_residue_z", "modal_omega"]
        for name in required:
            if name not in g:
                raise KeyError(f"{split}/{key} 缺少字段: {name}")
        data: Dict[str, np.ndarray] = {
            "points": g["points"][:].astype(np.float64),
            "modal_residue_z": g["modal_residue_z"][:].astype(np.float64),
            "modal_omega": g["modal_omega"][:].astype(np.float64),
        }
        optional_names = [
            "modal_zeta", "frequencies", "point_frf",
            "excitation_index", "excitation_coord",
            "pocket_bottom_mask", "cut_region_mask", "node_type", "spring_k_xyz",
            "local_thickness_ratio", "pocket_depth_ratio",
        ]
        for name in optional_names:
            if name in g:
                data[name] = np.asarray(g[name][()])
        return key, data


def as_node_mask(data: Dict[str, np.ndarray], name: str, n: int) -> np.ndarray:
    if name not in data:
        return np.zeros(n, dtype=bool)
    arr = np.asarray(data[name]).reshape(-1)
    if arr.size != n:
        return np.zeros(n, dtype=bool)
    return arr.astype(bool)


def get_masks(data: Dict[str, np.ndarray], n_nodes: int) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {
        "pocket_bottom": as_node_mask(data, "pocket_bottom_mask", n_nodes),
        "cut_region": as_node_mask(data, "cut_region_mask", n_nodes),
    }
    if "spring_k_xyz" in data:
        sk = np.asarray(data["spring_k_xyz"])
        if sk.shape[0] == n_nodes:
            masks["spring"] = np.linalg.norm(sk.reshape(n_nodes, -1), axis=1) > 0
    if "node_type" in data:
        nt = np.asarray(data["node_type"]).reshape(-1)
        if nt.size == n_nodes:
            masks["node_type_3_side_spring"] = nt == 3
            masks["node_type_4_corner_spring"] = nt == 4
    if "excitation_index" in data:
        exc = int(np.asarray(data["excitation_index"]).reshape(-1)[0])
        m = np.zeros(n_nodes, dtype=bool)
        if 0 <= exc < n_nodes:
            m[exc] = True
        masks["excitation"] = m
    return masks


def compute_full_frf_from_modal(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if "point_frf" in data and "frequencies" in data:
        freq_hz = np.asarray(data["frequencies"], dtype=np.float64).reshape(-1)
        pf = np.asarray(data["point_frf"])
        if np.iscomplexobj(pf):
            H = pf.astype(np.complex128)
        elif pf.ndim == 3 and pf.shape[-1] == 2:
            H = pf[..., 0].astype(np.float64) + 1j * pf[..., 1].astype(np.float64)
        else:
            H = pf.astype(np.complex128)
        if H.shape[0] == data["modal_residue_z"].shape[0]:
            return freq_hz, H

    if "frequencies" in data:
        freq_hz = np.asarray(data["frequencies"], dtype=np.float64).reshape(-1)
    else:
        modal_freq = data["modal_omega"] / (2.0 * np.pi)
        f_min = max(1.0, float(modal_freq[0] * 0.6))
        f_max = float(modal_freq[-1] * 1.15)
        freq_hz = np.linspace(f_min, f_max, 240, dtype=np.float64)

    omega = 2.0 * np.pi * freq_hz
    modal_omega = data["modal_omega"].astype(np.float64)
    A = data["modal_residue_z"].astype(np.float64)
    n_nodes, n_modes = A.shape
    if "modal_zeta" in data:
        zeta = np.asarray(data["modal_zeta"], dtype=np.float64).reshape(-1)[:n_modes]
    else:
        zeta = np.full(n_modes, 0.005, dtype=np.float64)

    H = np.zeros((n_nodes, omega.size), dtype=np.complex128)
    for r in range(n_modes):
        den = modal_omega[r] ** 2 - omega ** 2 + 2j * zeta[r] * modal_omega[r] * omega
        H += A[:, r:r + 1] / den.reshape(1, -1)
    return freq_hz, H


def single_mode_peak_amplitude(data: Dict[str, np.ndarray], mode_zero: int) -> np.ndarray:
    A_abs = np.abs(data["modal_residue_z"][:, mode_zero])
    omega_r = float(data["modal_omega"][mode_zero])
    if "modal_zeta" in data:
        zeta = float(np.asarray(data["modal_zeta"]).reshape(-1)[mode_zero])
    else:
        zeta = 0.005
    return A_abs / max(2.0 * zeta * omega_r * omega_r, 1e-300)


def color_values(values: np.ndarray, use_log: bool) -> Tuple[np.ndarray, str]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if use_log:
        return np.log10(np.maximum(np.abs(v), 1e-300)), "log10(|value|)"
    return v, "value"


def equal_axes_3d(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    r = 0.5 * span
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)


def plot_cloud(points_m: np.ndarray, values: np.ndarray, title: str, out_png: Path,
               masks: Dict[str, np.ndarray] | None = None, use_log: bool = True,
               point_size: int = 7, max_nodes: int = 12000, dpi: int = 220,
               selected_mask: np.ndarray | None = None, selected_label: str = "selected") -> None:
    points_mm = points_m * 1000.0
    n = points_mm.shape[0]
    if n > max_nodes:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=max_nodes, replace=False))
        if selected_mask is not None:
            hot_idx = np.where(selected_mask)[0]
            idx = np.unique(np.concatenate([idx, hot_idx]))
    else:
        idx = np.arange(n)

    fig = plt.figure(figsize=(10.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    if selected_mask is not None:
        selected_mask = np.asarray(selected_mask, dtype=bool)
        cold = idx[~selected_mask[idx]]
        hot = idx[selected_mask[idx]]
        if cold.size:
            p = points_mm[cold]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=max(1, point_size // 2), alpha=0.08, label="not selected")
        if hot.size:
            p = points_mm[hot]
            val, label = color_values(values[hot], use_log)
            sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=val, s=point_size + 7, alpha=0.95, label=selected_label)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.03)
            cbar.set_label(label)
    else:
        pts = points_mm[idx]
        val, label = color_values(values[idx], use_log)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=val, s=point_size, alpha=0.90)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.03)
        cbar.set_label(label)

    if masks:
        for name, mask in masks.items():
            mask = np.asarray(mask).astype(bool)
            if mask.size != n or not np.any(mask):
                continue
            show = np.where(mask)[0]
            if show.size > 700:
                show = show[np.linspace(0, show.size - 1, 700).round().astype(int)]
            p = points_mm[show]
            marker = "x" if name == "excitation" else "o"
            size = 30 if name == "excitation" else 4
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=size, marker=marker, label=name)
        ax.legend(loc="upper right", fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    equal_axes_3d(ax, points_mm)
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def top_indices(values: np.ndarray, percent: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(v.size * percent / 100.0)))
    return np.argsort(v)[-k:][::-1]


def make_top_mask(values: np.ndarray, percent: float) -> np.ndarray:
    idx = top_indices(values, percent)
    mask = np.zeros(np.asarray(values).reshape(-1).size, dtype=bool)
    mask[idx] = True
    return mask


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    return {
        "min": float(np.min(v)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "rms": float(np.sqrt(np.mean(v * v))),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if np.sum(a) == 0:
        return float("nan")
    return float(np.sum(a & b) / np.sum(a))


def build_node_metrics_rows(points: np.ndarray, A_abs: np.ndarray, rel_A: np.ndarray,
                            H_single: np.ndarray, H_peak: np.ndarray, H_peak_freq: np.ndarray,
                            masks: Dict[str, np.ndarray], residue_top_union: np.ndarray,
                            full_frf_top_mask: np.ndarray, residue_top_percent: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    n_nodes, n_modes = A_abs.shape
    for i in range(n_nodes):
        row: Dict[str, object] = {
            "node_index": i,
            "x_mm": float(points[i, 0] * 1000.0),
            "y_mm": float(points[i, 1] * 1000.0),
            "z_mm": float(points[i, 2] * 1000.0),
            "full_FRF_peak": float(H_peak[i]),
            "full_FRF_peak_freq_Hz": float(H_peak_freq[i]),
            "full_FRF_top": int(full_frf_top_mask[i]),
            "residue_top_union": int(residue_top_union[i]),
            "residue_top_percent_each_mode": float(residue_top_percent),
            "A_rel_max_percent": float(np.max(rel_A[i, :])),
            "A_rel_max_mode": int(np.argmax(rel_A[i, :]) + 1),
        }
        for name, mask in masks.items():
            if mask.size == n_nodes:
                row[f"mask_{name}"] = int(bool(mask[i]))
        for r in range(n_modes):
            row[f"A_abs_m{r + 1:02d}"] = float(A_abs[i, r])
            row[f"A_rel_percent_m{r + 1:02d}"] = float(rel_A[i, r])
            row[f"single_mode_H_peak_m{r + 1:02d}"] = float(H_single[i, r])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", default=SPLIT, choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument("--modes", nargs="+", type=int, default=MODES, help="模态阶次，1-based，例如 --modes 1 2 5")
    parser.add_argument("--residue-top-percent", type=float, default=RESIDUE_TOP_PERCENT, help="每阶 |A| 最大的前百分之多少节点")
    parser.add_argument("--frf-top-percent", type=float, default=FRF_TOP_PERCENT, help="总 FRF 峰值最大的前百分之多少节点")
    parser.add_argument("--save-per-mode-images", action="store_true", default=SAVE_PER_MODE_IMAGES)
    parser.add_argument("--no-log-color", action="store_true")
    parser.add_argument("--point-size", type=int, default=POINT_SIZE)
    parser.add_argument("--max-plot-nodes", type=int, default=MAX_PLOT_NODES)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.data_dir / "residue_cloud_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    key, data = load_sample(args.data_dir, args.split, args.sample_index)
    points = data["points"]
    A = data["modal_residue_z"]
    n_nodes, n_modes = A.shape
    modes = [m for m in args.modes if 1 <= m <= n_modes]
    if not modes:
        raise RuntimeError(f"没有有效模态；数据只有 {n_modes} 阶")

    A_abs = np.abs(A)
    A_mode_max = np.maximum(np.max(A_abs, axis=0), 1e-300)
    rel_A = A_abs / A_mode_max.reshape(1, -1) * 100.0

    freq_hz, H = compute_full_frf_from_modal(data)
    H_abs = np.abs(H)
    H_peak = np.max(H_abs, axis=1)
    H_peak_idx = np.argmax(H_abs, axis=1)
    H_peak_freq = freq_hz[H_peak_idx]

    H_single = np.zeros_like(A_abs)
    for r in range(n_modes):
        H_single[:, r] = single_mode_peak_amplitude(data, r)

    masks = get_masks(data, n_nodes)
    use_log = not args.no_log_color
    prefix = f"{args.split}_{key}"

    selected_mode_indices = [m - 1 for m in modes]
    residue_top_masks: Dict[int, np.ndarray] = {}
    residue_top_union = np.zeros(n_nodes, dtype=bool)
    for mode in modes:
        r = mode - 1
        mask = make_top_mask(A_abs[:, r], args.residue_top_percent)
        residue_top_masks[mode] = mask
        residue_top_union |= mask

    full_frf_top_mask = make_top_mask(H_peak, args.frf_top_percent)

    # 图 1：总 FRF 峰值 top 区域。
    plot_cloud(
        points, H_peak,
        title=f"{prefix} | full FRF peak top {args.frf_top_percent:g}%",
        out_png=out_dir / f"{prefix}_full_FRF_peak_top_{args.frf_top_percent:g}pct.png",
        masks=masks,
        use_log=use_log,
        point_size=args.point_size,
        max_nodes=args.max_plot_nodes,
        dpi=args.dpi,
        selected_mask=full_frf_top_mask,
        selected_label="full FRF top",
    )

    # 图 2：每阶 |A| top p% 的 union，只显示残差 top 节点。
    residue_top_score = np.max(A_abs[:, selected_mode_indices], axis=1)
    plot_cloud(
        points, residue_top_score,
        title=f"{prefix} | union of per-mode |A| top {args.residue_top_percent:g}% nodes; modes={modes}",
        out_png=out_dir / f"{prefix}_A_top_union_top_{args.residue_top_percent:g}pct_each_mode.png",
        masks=masks,
        use_log=use_log,
        point_size=args.point_size,
        max_nodes=args.max_plot_nodes,
        dpi=args.dpi,
        selected_mask=residue_top_union,
        selected_label="per-mode |A| top union",
    )

    # 图 3：所有节点的 max relative |A| 百分比，作为参考热力图。
    plot_cloud(
        points, np.max(rel_A[:, selected_mode_indices], axis=1),
        title=f"{prefix} | max selected-mode relative |A| percent",
        out_png=out_dir / f"{prefix}_A_relative_max_over_modes.png",
        masks=masks,
        use_log=False,
        point_size=args.point_size,
        max_nodes=args.max_plot_nodes,
        dpi=args.dpi,
    )

    if args.save_per_mode_images:
        for mode in modes:
            r = mode - 1
            modal_freq_hz = float(data["modal_omega"][r] / (2.0 * np.pi))
            plot_cloud(
                points, A_abs[:, r],
                title=f"{prefix} | mode {mode:02d} |A_r(x)|, f_r={modal_freq_hz:.1f} Hz",
                out_png=out_dir / f"{prefix}_mode{mode:02d}_A_abs.png",
                masks=masks,
                use_log=use_log,
                point_size=args.point_size,
                max_nodes=args.max_plot_nodes,
                dpi=args.dpi,
                selected_mask=residue_top_masks[mode],
                selected_label=f"mode {mode:02d} |A| top",
            )

    node_rows = build_node_metrics_rows(
        points, A_abs, rel_A, H_single, H_peak, H_peak_freq, masks,
        residue_top_union, full_frf_top_mask, args.residue_top_percent,
    )
    node_metrics_path = out_dir / f"{prefix}_node_metrics_wide.csv"
    write_csv(node_metrics_path, node_rows)

    # 合并 top 节点到一个长表。
    top_rows: List[Dict[str, object]] = []
    for kind, mode, mask, metric in [
        ("full_FRF_peak_top", "all", full_frf_top_mask, H_peak),
        ("per_mode_A_top_union", "union", residue_top_union, residue_top_score),
    ]:
        idxs = np.where(mask)[0]
        idxs = idxs[np.argsort(metric[idxs])[::-1]] if idxs.size else idxs
        for rank, idx in enumerate(idxs, start=1):
            top_rows.append({
                "kind": kind,
                "mode": mode,
                "rank": rank,
                "node_index": int(idx),
                "x_mm": float(points[idx, 0] * 1000.0),
                "y_mm": float(points[idx, 1] * 1000.0),
                "z_mm": float(points[idx, 2] * 1000.0),
                "metric": float(metric[idx]),
                "full_FRF_peak": float(H_peak[idx]),
                "full_FRF_peak_freq_Hz": float(H_peak_freq[idx]),
                "A_rel_max_percent": float(np.max(rel_A[idx, :])),
                "A_rel_max_mode": int(np.argmax(rel_A[idx, :]) + 1),
            })

    for mode in modes:
        r = mode - 1
        mask = residue_top_masks[mode]
        idxs = np.where(mask)[0]
        idxs = idxs[np.argsort(A_abs[idxs, r])[::-1]] if idxs.size else idxs
        for rank, idx in enumerate(idxs, start=1):
            top_rows.append({
                "kind": "mode_A_abs_top",
                "mode": mode,
                "rank": rank,
                "node_index": int(idx),
                "x_mm": float(points[idx, 0] * 1000.0),
                "y_mm": float(points[idx, 1] * 1000.0),
                "z_mm": float(points[idx, 2] * 1000.0),
                "metric": float(A_abs[idx, r]),
                "A_rel_percent_this_mode": float(rel_A[idx, r]),
                "full_FRF_peak": float(H_peak[idx]),
                "full_FRF_peak_freq_Hz": float(H_peak_freq[idx]),
            })

    top_combined_path = out_dir / f"{prefix}_top_nodes_combined.csv"
    write_csv(top_combined_path, top_rows)

    # summary：每阶统计、相关性、top 区域重合率。
    summary_rows: List[Dict[str, object]] = []
    for mode in modes:
        r = mode - 1
        A_stat = summarize_values(A_abs[:, r])
        Hs_stat = summarize_values(H_single[:, r])
        top_A_mask = residue_top_masks[mode]
        row: Dict[str, object] = {
            "split": args.split,
            "sample": key,
            "mode": mode,
            "modal_freq_Hz": float(data["modal_omega"][r] / (2.0 * np.pi)),
            "residue_top_percent": args.residue_top_percent,
            "A_abs_p50": A_stat["p50"],
            "A_abs_p90": A_stat["p90"],
            "A_abs_p95": A_stat["p95"],
            "A_abs_p99": A_stat["p99"],
            "A_abs_max": A_stat["max"],
            "A_abs_rms": A_stat["rms"],
            "single_mode_H_peak_p95": Hs_stat["p95"],
            "single_mode_H_peak_max": Hs_stat["max"],
            "corr_absA_vs_single_mode_H_peak": pearson(A_abs[:, r], H_single[:, r]),
            "corr_log_absA_vs_full_FRF_peak": pearson(np.log10(A_abs[:, r] + 1e-300), np.log10(H_peak + 1e-300)),
            "topA_overlap_fullFRF_top_fraction": overlap_fraction(top_A_mask, full_frf_top_mask),
            "topA_node_count": int(np.sum(top_A_mask)),
            "topA_node_fraction": float(np.mean(top_A_mask)),
        }
        for name, mask in masks.items():
            if mask.size == n_nodes:
                row[f"topA_in_{name}_fraction"] = overlap_fraction(top_A_mask, mask)
        summary_rows.append(row)

    union_row: Dict[str, object] = {
        "split": args.split,
        "sample": key,
        "mode": "per_mode_A_top_union",
        "residue_top_percent": args.residue_top_percent,
        "residue_top_union_count": int(np.sum(residue_top_union)),
        "residue_top_union_fraction": float(np.mean(residue_top_union)),
        "union_overlap_fullFRF_top_fraction": overlap_fraction(residue_top_union, full_frf_top_mask),
        "fullFRF_top_overlap_union_fraction": overlap_fraction(full_frf_top_mask, residue_top_union),
        "corr_log_max_absA_vs_full_FRF_peak": pearson(
            np.log10(residue_top_score + 1e-300),
            np.log10(H_peak + 1e-300),
        ),
    }
    for name, mask in masks.items():
        if mask.size == n_nodes:
            union_row[f"union_in_{name}_fraction"] = overlap_fraction(residue_top_union, mask)
    summary_rows.append(union_row)

    # 统计 full FRF top 节点里由哪一阶残差最大主导。
    top_idx = np.where(full_frf_top_mask)[0]
    max_mode = np.argmax(A_abs, axis=1) + 1
    for mode in modes:
        count = int(np.sum(max_mode[top_idx] == mode)) if top_idx.size else 0
        summary_rows.append({
            "split": args.split,
            "sample": key,
            "mode": f"fullFRF_top_dominant_A_mode_{mode:02d}",
            "fullFRF_top_count": int(top_idx.size),
            "dominant_mode_count": count,
            "dominant_mode_fraction": float(count / max(top_idx.size, 1)),
        })

    summary_path = out_dir / f"{prefix}_summary.csv"
    write_csv(summary_path, summary_rows)

    info = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "sample": key,
        "n_nodes": int(n_nodes),
        "n_modes": int(n_modes),
        "modes_analyzed": modes,
        "residue_top_percent": args.residue_top_percent,
        "frf_top_percent": args.frf_top_percent,
        "output_dir": str(out_dir),
        "files": {
            "summary": str(summary_path),
            "node_metrics_wide": str(node_metrics_path),
            "top_nodes_combined": str(top_combined_path),
            "full_FRF_peak_top_png": str(out_dir / f"{prefix}_full_FRF_peak_top_{args.frf_top_percent:g}pct.png"),
            "A_top_union_png": str(out_dir / f"{prefix}_A_top_union_top_{args.residue_top_percent:g}pct_each_mode.png"),
            "A_relative_max_png": str(out_dir / f"{prefix}_A_relative_max_over_modes.png"),
        },
        "note": "ROI 推荐看每阶 |A| top p% 的 union，而不是只看 full FRF peak top；否则当总 FRF 被某一阶主导时，训练会偏向该阶。",
    }
    info_path = out_dir / f"{prefix}_run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("完成。输出目录：", out_dir)
    print("合并表：")
    print("  ", summary_path)
    print("  ", node_metrics_path)
    print("  ", top_combined_path)
    print("图：")
    print("  ", out_dir / f"{prefix}_full_FRF_peak_top_{args.frf_top_percent:g}pct.png")
    print("  ", out_dir / f"{prefix}_A_top_union_top_{args.residue_top_percent:g}pct_each_mode.png")
    print("  ", out_dir / f"{prefix}_A_relative_max_over_modes.png")
    print("说明：A_top_union 图只保留每阶 |A| 最大的前 residue_top_percent% 节点的并集。")


if __name__ == "__main__":
    main()
