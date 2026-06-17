# -*- coding: utf-8 -*-
"""
查看 modal_residue_z / 单模态 FRF 贡献 / 总 FRF 峰值的节点云图。

默认读取：
    F:\毕业论文\-cnn-modal-residue-frf\data_modal_residue_filtered100

这个版本把能合并的结果合并到一起：
1. 一个 summary.csv：每阶统计、相关性、top 区域重合率；
2. 一个 node_metrics_wide.csv：每个节点的坐标、FRF 峰值、每阶 |A|、每阶单模态峰值、阈值标记；
3. 一个 top_nodes_combined.csv：总 FRF top 节点、各阶 |A| top 节点、阈值节点全部合并；
4. 一张 A_threshold_union.png：显示“模态残差大于某百分比”的节点区域；
5. 一张 full_FRF_peak.png：显示总 FRF 峰值节点分布。

物理关系：
    A_r(x) = modal_residue_z(x,r) = phi_r,z(x) * phi_r,z(x_f)

同一个样本、同一阶模态、在该阶共振附近，分母对所有节点相同，因此：
    |A_r(x)| 越大，该阶单模态 FRF 贡献越大。

但总 FRF 是多阶复数叠加，所以：
    |A_r(x)| 大 ≠ 总 FRF 峰值一定最大。
本程序同时输出 |A|、单模态峰值、总 FRF 峰值和相关性/重合率，辅助决定后续 ROI 预测区域。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

TOP_FRACTION = 0.10          # top 10% 节点
THRESHOLD_PERCENT = 50.0     # |A_r(x)| >= 50% * max_x |A_r(x)| 的节点画到一张图
DPI = 220
POINT_SIZE = 7
USE_LOG_COLOR = True
MAX_PLOT_NODES = 12000
SAVE_PER_MODE_IMAGES = False  # True 时额外保存每阶 |A| 云图；默认不保存，避免文件太多
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
        # 如果 H5 中已经保存 point_frf，则优先使用；兼容复数或 [real,imag] 格式。
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
               point_size: int = 7, max_nodes: int = 12000, dpi: int = 220) -> None:
    points_mm = points_m * 1000.0
    n = points_mm.shape[0]
    if n > max_nodes:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=max_nodes, replace=False))
    else:
        idx = np.arange(n)
    pts = points_mm[idx]
    val, label = color_values(values[idx], use_log)

    fig = plt.figure(figsize=(9.8, 4.9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=val, s=point_size, alpha=0.90)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label(label)

    if masks:
        for name, mask in masks.items():
            mask = np.asarray(mask).astype(bool)
            if mask.size != n or not np.any(mask):
                continue
            show = np.where(mask)[0]
            if show.size > 800:
                show = show[np.linspace(0, show.size - 1, 800).round().astype(int)]
            p = points_mm[show]
            marker = "x" if name == "excitation" else "o"
            size = 28 if name == "excitation" else 5
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


def plot_threshold_union(points_m: np.ndarray, rel_A: np.ndarray, threshold_percent: float,
                         modes: List[int], masks: Dict[str, np.ndarray], out_png: Path,
                         point_size: int, max_nodes: int, dpi: int) -> np.ndarray:
    """画一张图：任意选定模态的 |A| 超过阈值百分比的节点。返回 union mask。"""
    # rel_A: [N, M]，每阶按该阶 max |A| 归一化为百分比。
    selected = [m - 1 for m in modes]
    selected = [m for m in selected if 0 <= m < rel_A.shape[1]]
    if not selected:
        raise RuntimeError("没有有效模态可画阈值图")
    rel_sel = rel_A[:, selected]
    max_rel = np.max(rel_sel, axis=1)
    mode_arg = np.argmax(rel_sel, axis=1)
    best_mode = np.array([selected[i] + 1 for i in mode_arg], dtype=int)
    union = max_rel >= threshold_percent

    points_mm = points_m * 1000.0
    n = points_mm.shape[0]
    if n > max_nodes:
        rng = np.random.default_rng(42)
        base_idx = np.sort(rng.choice(n, size=max_nodes, replace=False))
        # 保证阈值节点全部画出来；如果太多，则也降采样。
        hot_idx = np.where(union)[0]
        if hot_idx.size > max_nodes:
            hot_idx = np.sort(rng.choice(hot_idx, size=max_nodes, replace=False))
        idx = np.unique(np.concatenate([base_idx, hot_idx]))
    else:
        idx = np.arange(n)

    fig = plt.figure(figsize=(10.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    cold = idx[~union[idx]]
    hot = idx[union[idx]]
    if cold.size:
        p = points_mm[cold]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=max(1, point_size // 2), alpha=0.10, label="below threshold")
    if hot.size:
        p = points_mm[hot]
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=max_rel[hot], s=point_size + 5, alpha=0.95, label="|A| high")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.03)
        cbar.set_label("max_r |A_r(x)| / max_x |A_r(x)| (%)")

    if masks:
        for name, mask in masks.items():
            mask = np.asarray(mask).astype(bool)
            if mask.size != n or not np.any(mask):
                continue
            show = np.where(mask)[0]
            if show.size > 600:
                show = show[np.linspace(0, show.size - 1, 600).round().astype(int)]
            p = points_mm[show]
            marker = "x" if name == "excitation" else "o"
            size = 30 if name == "excitation" else 4
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=size, marker=marker, label=name)

    ax.set_title(f"nodes with modal residue >= {threshold_percent:.1f}% of mode max |A|; modes={modes}")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    equal_axes_3d(ax, points_mm)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return union


def top_indices(values: np.ndarray, frac: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(v.size * frac)))
    return np.argsort(v)[-k:][::-1]


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
    # 合并所有 key，避免不同 row 字段不完全一样。
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
                            masks: Dict[str, np.ndarray], threshold_percent: float) -> List[Dict[str, object]]:
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
        }
        for name, mask in masks.items():
            if mask.size == n_nodes:
                row[f"mask_{name}"] = int(bool(mask[i]))
        row["A_rel_max_percent"] = float(np.max(rel_A[i, :]))
        row["A_rel_max_mode"] = int(np.argmax(rel_A[i, :]) + 1)
        row["A_threshold_any"] = int(np.max(rel_A[i, :]) >= threshold_percent)
        for r in range(n_modes):
            row[f"A_abs_m{r + 1:02d}"] = float(A_abs[i, r])
            row[f"A_rel_percent_m{r + 1:02d}"] = float(rel_A[i, r])
            row[f"A_gt_threshold_m{r + 1:02d}"] = int(rel_A[i, r] >= threshold_percent)
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
    parser.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    parser.add_argument("--threshold-percent", type=float, default=THRESHOLD_PERCENT, help="|A_r| 大于该阶最大 |A_r| 的百分之多少，画到阈值图")
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

    # 图 1：总 FRF 峰值云图。
    plot_cloud(
        points, H_peak,
        title=f"{prefix} | full FRF peak over frequency |H(x,f)|",
        out_png=out_dir / f"{prefix}_full_FRF_peak.png",
        masks=masks,
        use_log=use_log,
        point_size=args.point_size,
        max_nodes=args.max_plot_nodes,
        dpi=args.dpi,
    )

    # 图 2：模态残差大于阈值百分比的一张总图。
    threshold_union = plot_threshold_union(
        points, rel_A, args.threshold_percent, modes, masks,
        out_png=out_dir / f"{prefix}_A_threshold_union_ge_{args.threshold_percent:g}pct.png",
        point_size=args.point_size,
        max_nodes=args.max_plot_nodes,
        dpi=args.dpi,
    )

    # 图 3：每个节点在 10 阶中最大的相对残差百分比，用于看整体热点。
    plot_cloud(
        points, np.max(rel_A[:, [m - 1 for m in modes]], axis=1),
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
            )

    # 合并节点指标到一个宽表。
    node_rows = build_node_metrics_rows(points, A_abs, rel_A, H_single, H_peak, H_peak_freq, masks, args.threshold_percent)
    node_metrics_path = out_dir / f"{prefix}_node_metrics_wide.csv"
    write_csv(node_metrics_path, node_rows)

    # 合并所有 top 节点到一个长表。
    top_rows: List[Dict[str, object]] = []
    top_full = top_indices(H_peak, args.top_fraction)
    top_full_mask = np.zeros(n_nodes, dtype=bool)
    top_full_mask[top_full] = True
    for rank, idx in enumerate(top_full, start=1):
        top_rows.append({
            "kind": "full_FRF_peak_top",
            "mode": "all",
            "rank": rank,
            "node_index": int(idx),
            "x_mm": float(points[idx, 0] * 1000.0),
            "y_mm": float(points[idx, 1] * 1000.0),
            "z_mm": float(points[idx, 2] * 1000.0),
            "metric": float(H_peak[idx]),
            "full_FRF_peak_freq_Hz": float(H_peak_freq[idx]),
            "A_rel_max_percent": float(np.max(rel_A[idx, :])),
            "A_rel_max_mode": int(np.argmax(rel_A[idx, :]) + 1),
        })

    for mode in modes:
        r = mode - 1
        top_A = top_indices(A_abs[:, r], args.top_fraction)
        for rank, idx in enumerate(top_A, start=1):
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
        thr = rel_A[:, r] >= args.threshold_percent
        idxs = np.where(thr)[0]
        # 阈值节点也合并进去；按 |A| 从大到小排序。
        idxs = idxs[np.argsort(A_abs[idxs, r])[::-1]] if idxs.size else idxs
        for rank, idx in enumerate(idxs, start=1):
            top_rows.append({
                "kind": f"mode_A_ge_{args.threshold_percent:g}pct_of_mode_max",
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

    # summary：每阶统计、相关性、和物理 mask / top-FRF 的重合率。
    summary_rows: List[Dict[str, object]] = []
    for mode in modes:
        r = mode - 1
        A_stat = summarize_values(A_abs[:, r])
        Hs_stat = summarize_values(H_single[:, r])
        top_A = top_indices(A_abs[:, r], args.top_fraction)
        top_A_mask = np.zeros(n_nodes, dtype=bool)
        top_A_mask[top_A] = True
        threshold_mask = rel_A[:, r] >= args.threshold_percent
        row: Dict[str, object] = {
            "split": args.split,
            "sample": key,
            "mode": mode,
            "modal_freq_Hz": float(data["modal_omega"][r] / (2.0 * np.pi)),
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
            "topA_overlap_fullFRF_top_fraction": overlap_fraction(top_A_mask, top_full_mask),
            "threshold_percent": args.threshold_percent,
            "threshold_node_count": int(np.sum(threshold_mask)),
            "threshold_node_fraction": float(np.mean(threshold_mask)),
            "threshold_overlap_fullFRF_top_fraction": overlap_fraction(threshold_mask, top_full_mask),
        }
        for name, mask in masks.items():
            if mask.size == n_nodes:
                row[f"topA_in_{name}_fraction"] = overlap_fraction(top_A_mask, mask)
                row[f"threshold_in_{name}_fraction"] = overlap_fraction(threshold_mask, mask)
        summary_rows.append(row)

    # 总阈值 union 的 summary。
    union_row: Dict[str, object] = {
        "split": args.split,
        "sample": key,
        "mode": "union",
        "threshold_percent": args.threshold_percent,
        "threshold_node_count": int(np.sum(threshold_union)),
        "threshold_node_fraction": float(np.mean(threshold_union)),
        "threshold_overlap_fullFRF_top_fraction": overlap_fraction(threshold_union, top_full_mask),
        "corr_log_max_relA_vs_full_FRF_peak": pearson(
            np.log10(np.max(rel_A[:, [m - 1 for m in modes]], axis=1) + 1e-300),
            np.log10(H_peak + 1e-300),
        ),
    }
    for name, mask in masks.items():
        if mask.size == n_nodes:
            union_row[f"threshold_union_in_{name}_fraction"] = overlap_fraction(threshold_union, mask)
    summary_rows.append(union_row)

    summary_path = out_dir / f"{prefix}_summary.csv"
    write_csv(summary_path, summary_rows)

    info = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "sample": key,
        "n_nodes": int(n_nodes),
        "n_modes": int(n_modes),
        "modes_analyzed": modes,
        "top_fraction": args.top_fraction,
        "threshold_percent": args.threshold_percent,
        "output_dir": str(out_dir),
        "files": {
            "summary": str(summary_path),
            "node_metrics_wide": str(node_metrics_path),
            "top_nodes_combined": str(top_combined_path),
            "full_FRF_peak_png": str(out_dir / f"{prefix}_full_FRF_peak.png"),
            "A_threshold_union_png": str(out_dir / f"{prefix}_A_threshold_union_ge_{args.threshold_percent:g}pct.png"),
            "A_relative_max_png": str(out_dir / f"{prefix}_A_relative_max_over_modes.png"),
        },
        "note": "同一阶共振附近，|A_r(x)| 与该阶单模态 FRF 贡献严格同序；总 FRF 峰值还受其他模态和复数叠加影响。",
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
    print("  ", out_dir / f"{prefix}_full_FRF_peak.png")
    print("  ", out_dir / f"{prefix}_A_threshold_union_ge_{args.threshold_percent:g}pct.png")
    print("  ", out_dir / f"{prefix}_A_relative_max_over_modes.png")
    print("说明：阈值图里的节点满足：任一所选模态 |A_r(x)| >= threshold_percent% * max_x |A_r(x)|。")


if __name__ == "__main__":
    main()
