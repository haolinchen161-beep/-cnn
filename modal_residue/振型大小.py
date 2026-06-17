# -*- coding: utf-8 -*-
"""
查看 modal_residue_z / 单模态 FRF 贡献 / FRF 峰值的节点云图。

默认读取：
    F:\毕业论文\-cnn-modal-residue-frf\data_modal_residue_filtered100

用途：
1. 看每个样本每阶 modal_residue_z 的空间大小分布；
2. 判断 |A_r(x)| 大的区域是否对应 FRF 响应大的区域；
3. 导出 top-|A| 或 top-FRF 节点，辅助后续只选 ROI 区域预测，而不是全节点预测。

说明：
    A_r(x) = modal_residue_z(x, r) = phi_r,z(x) * phi_r,z(x_f)

对同一个样本、同一个模态 r、同一个频率 omega，单模态贡献为：
    H_r(x, omega) = A_r(x) / (omega_r^2 - omega^2 + 2j*zeta_r*omega_r*omega)

因此在同一阶模态附近，所有节点的分母相同，|A_r(x)| 越大，单模态 FRF 贡献越大。
但总 FRF 是多阶复数叠加，不同模态之间会叠加/抵消，所以“残差大 = 总 FRF 大”不是绝对成立，需要用本程序同时看 A 云图和 FRF 云图。
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

SPLIT = "train"          # train / val / test
SAMPLE_INDEX = 0          # 第几个样本，从 0 开始
MODES = list(range(1, 11))  # 1~10 阶；也可以改成 [1, 3, 5]

TOP_FRACTION = 0.10       # 导出 top 10% 节点作为候选 ROI
DPI = 220
POINT_SIZE = 7
USE_LOG_COLOR = True      # True: 颜色为 log10(|值|)，更容易看小值和大值分布
MAX_PLOT_NODES = 12000    # 节点很多时随机降采样画图；当前 5000~6000 一般不会触发
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
            "modal_zeta",
            "frequencies",
            "point_frf",
            "excitation_index",
            "excitation_coord",
            "pocket_bottom_mask",
            "cut_region_mask",
            "node_type",
            "spring_k_xyz",
            "local_thickness_ratio",
            "pocket_depth_ratio",
        ]
        for name in optional_names:
            if name in g:
                arr = g[name][()]
                data[name] = np.asarray(arr)
        return key, data


def as_node_mask(data: Dict[str, np.ndarray], name: str, n: int) -> np.ndarray:
    if name not in data:
        return np.zeros(n, dtype=bool)
    arr = np.asarray(data[name]).reshape(-1)
    if arr.size != n:
        return np.zeros(n, dtype=bool)
    return arr.astype(bool)


def compute_full_frf_from_modal(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """返回 frequencies_Hz, H: [n_nodes, n_freq] complex。"""
    if "frequencies" in data:
        freq_hz = np.asarray(data["frequencies"], dtype=np.float64).reshape(-1)
    else:
        # 如果没有频率网格，就围绕 1~10 阶自动生成一个用于观察的频率轴。
        modal_freq = data["modal_omega"] / (2.0 * np.pi)
        f_min = max(1.0, float(modal_freq[0] * 0.6))
        f_max = float(modal_freq[-1] * 1.15)
        freq_hz = np.linspace(f_min, f_max, 240, dtype=np.float64)

    omega = 2.0 * np.pi * freq_hz
    modal_omega = data["modal_omega"].astype(np.float64)
    A = data["modal_residue_z"].astype(np.float64)  # [N, M]
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
    """单模态在自身共振频率附近的 FRF 贡献幅值。"""
    A_abs = np.abs(data["modal_residue_z"][:, mode_zero])
    omega_r = float(data["modal_omega"][mode_zero])
    if "modal_zeta" in data:
        zeta = float(np.asarray(data["modal_zeta"]).reshape(-1)[mode_zero])
    else:
        zeta = 0.005
    denom_mag = max(2.0 * zeta * omega_r * omega_r, 1e-300)
    return A_abs / denom_mag


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

    fig = plt.figure(figsize=(9.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=val, s=point_size, alpha=0.92)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label(label)

    if masks:
        # 用少量黑色/红色点标记已有物理区域，便于判断残差大区域是否落在加工区/装夹区。
        for name, mask in masks.items():
            mask = np.asarray(mask).astype(bool)
            if mask.size != n or not np.any(mask):
                continue
            show = np.where(mask)[0]
            if show.size > 800:
                show = show[np.linspace(0, show.size - 1, 800).round().astype(int)]
            p = points_mm[show]
            marker = "x" if name == "excitation" else "o"
            size = 18 if name == "excitation" else 5
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


def top_indices(values: np.ndarray, frac: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = v.size
    k = max(1, int(math.ceil(n * frac)))
    return np.argsort(v)[-k:][::-1]


def write_node_csv(path: Path, points: np.ndarray, metric: np.ndarray, indices: np.ndarray,
                   extra: Dict[str, np.ndarray] | None = None) -> None:
    extra = extra or {}
    fields = ["rank", "node_index", "x_mm", "y_mm", "z_mm", "metric"] + list(extra.keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, idx in enumerate(indices, start=1):
            row = {
                "rank": rank,
                "node_index": int(idx),
                "x_mm": float(points[idx, 0] * 1000.0),
                "y_mm": float(points[idx, 1] * 1000.0),
                "z_mm": float(points[idx, 2] * 1000.0),
                "metric": float(metric[idx]),
            }
            for name, arr in extra.items():
                arr = np.asarray(arr).reshape(-1)
                if arr.size == points.shape[0]:
                    row[name] = float(arr[idx])
            w.writerow(row)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0 or sy <= 0:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", default=SPLIT, choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument("--modes", nargs="+", type=int, default=MODES, help="模态阶次，1-based，例如 --modes 1 2 5")
    parser.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
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

    freq_hz, H = compute_full_frf_from_modal(data)
    H_abs = np.abs(H)
    H_peak = np.max(H_abs, axis=1)
    H_peak_idx = np.argmax(H_abs, axis=1)
    H_peak_freq = freq_hz[H_peak_idx]

    masks: Dict[str, np.ndarray] = {}
    masks["pocket_bottom"] = as_node_mask(data, "pocket_bottom_mask", n_nodes)
    masks["cut_region"] = as_node_mask(data, "cut_region_mask", n_nodes)
    if "spring_k_xyz" in data:
        sk = np.asarray(data["spring_k_xyz"])
        if sk.shape[0] == n_nodes:
            masks["spring"] = np.linalg.norm(sk.reshape(n_nodes, -1), axis=1) > 0
    if "excitation_index" in data:
        exc = int(np.asarray(data["excitation_index"]).reshape(-1)[0])
        m = np.zeros(n_nodes, dtype=bool)
        if 0 <= exc < n_nodes:
            m[exc] = True
        masks["excitation"] = m

    use_log = not args.no_log_color
    prefix = f"{args.split}_{key}"

    # 1) 全频段总 FRF 峰值云图。
    plot_cloud(
        points,
        H_peak,
        title=f"{prefix} | full FRF peak over frequency |H(x,f)|",
        out_png=out_dir / f"{prefix}_full_FRF_peak.png",
        masks=masks,
        use_log=use_log,
        point_size=args.point_size,
        max_nodes=args.max_plot_nodes,
        dpi=args.dpi,
    )

    top_full = top_indices(H_peak, args.top_fraction)
    extra = {"H_peak_freq_Hz": H_peak_freq}
    write_node_csv(out_dir / f"{prefix}_top_{int(args.top_fraction * 100)}pct_full_FRF_peak_nodes.csv", points, H_peak, top_full, extra=extra)

    # 2) 每阶残差云图、单模态峰值云图、相关性统计。
    rows: List[Dict[str, float]] = []
    for mode in args.modes:
        if mode < 1 or mode > n_modes:
            print(f"跳过 mode={mode}: 数据只有 {n_modes} 阶")
            continue
        r = mode - 1
        A_abs = np.abs(A[:, r])
        H_single_peak = single_mode_peak_amplitude(data, r)

        modal_freq_hz = float(data["modal_omega"][r] / (2.0 * np.pi))
        plot_cloud(
            points,
            A_abs,
            title=f"{prefix} | mode {mode:02d} |A_r(x)|, f_r={modal_freq_hz:.1f} Hz",
            out_png=out_dir / f"{prefix}_mode{mode:02d}_A_abs.png",
            masks=masks,
            use_log=use_log,
            point_size=args.point_size,
            max_nodes=args.max_plot_nodes,
            dpi=args.dpi,
        )
        plot_cloud(
            points,
            H_single_peak,
            title=f"{prefix} | mode {mode:02d} single-mode resonance contribution",
            out_png=out_dir / f"{prefix}_mode{mode:02d}_single_mode_peak_H.png",
            masks=masks,
            use_log=use_log,
            point_size=args.point_size,
            max_nodes=args.max_plot_nodes,
            dpi=args.dpi,
        )

        top_A = top_indices(A_abs, args.top_fraction)
        write_node_csv(out_dir / f"{prefix}_mode{mode:02d}_top_{int(args.top_fraction * 100)}pct_A_nodes.csv", points, A_abs, top_A)
        top_Hs = top_indices(H_single_peak, args.top_fraction)
        write_node_csv(out_dir / f"{prefix}_mode{mode:02d}_top_{int(args.top_fraction * 100)}pct_single_mode_H_nodes.csv", points, H_single_peak, top_Hs)

        sA = summarize_values(A_abs)
        sHs = summarize_values(H_single_peak)
        rows.append({
            "split": args.split,
            "sample": key,
            "mode": mode,
            "modal_freq_Hz": modal_freq_hz,
            "A_abs_p50": sA["p50"],
            "A_abs_p90": sA["p90"],
            "A_abs_p95": sA["p95"],
            "A_abs_p99": sA["p99"],
            "A_abs_max": sA["max"],
            "A_abs_rms": sA["rms"],
            "single_mode_H_peak_p95": sHs["p95"],
            "single_mode_H_peak_max": sHs["max"],
            "corr_absA_vs_single_mode_H_peak": pearson(A_abs, H_single_peak),
            "corr_absA_vs_full_FRF_peak": pearson(np.log10(A_abs + 1e-300), np.log10(H_peak + 1e-300)),
        })

    with open(out_dir / f"{prefix}_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    info = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "sample": key,
        "n_nodes": int(n_nodes),
        "n_modes": int(n_modes),
        "modes_plotted": args.modes,
        "top_fraction": args.top_fraction,
        "output_dir": str(out_dir),
        "note": "同一阶单模态共振附近，|A_r(x)| 与该阶单模态 FRF 贡献节点分布严格同序；总 FRF 峰值还受其他模态、频率和复数叠加影响。",
    }
    with open(out_dir / f"{prefix}_run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("完成。输出目录：", out_dir)
    print("主要文件：")
    print("  ", out_dir / f"{prefix}_full_FRF_peak.png")
    print("  ", out_dir / f"{prefix}_summary.csv")
    print("  ", out_dir / f"{prefix}_top_{int(args.top_fraction * 100)}pct_full_FRF_peak_nodes.csv")
    print("说明：单阶模态附近 |A_r(x)| 大 => 该阶单模态 FRF 贡献大；总 FRF 峰值不一定完全等于 |A| 排序。")


if __name__ == "__main__":
    main()
