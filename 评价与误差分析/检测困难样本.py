# -*- coding: utf-8 -*-
"""
检测验证集中预测误差较大的样本，例如 val sample 2 / 4 / 9。

默认读取：
    F:\毕业论文\-cnn-modal-residue-frf\data_modal_residue_filtered100

主要输出：
1. hard_sample_report.txt
   人能直接看的报告；
2. hard_sample_features.csv
   指定困难样本的几何、边界、频率、残差等指标；
3. train_feature_reference.csv
   train 集各指标的均值、标准差、分位数；
4. hard_sample_mode_frequencies.csv
   困难样本每一阶频率、相邻模态间距、相对间距、相对 train 分布的 z-score；
5. hard_sample_error_metrics.csv
   如果提供 val_metrics.csv，会提取这些样本的预测误差。

用法：
    F:/pytorch_cuda12/python.exe -B modal_residue/检测困难样本.py

带训练结果误差：
    F:/pytorch_cuda12/python.exe -B modal_residue/检测困难样本.py --val-metrics runs/modal_residue_meshgraph/val_metrics.csv

指定样本：
    F:/pytorch_cuda12/python.exe -B modal_residue/检测困难样本.py --sample-indices 2 4 9
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


DATA_DIR = Path(r"F:\毕业论文\-cnn-modal-residue-frf\data_modal_residue_filtered100")
OUT_DIR = None
SPLIT = "val"
SAMPLE_INDICES = [2, 4, 9]


def sorted_keys(f: h5py.File) -> List[str]:
    return sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))


def read_sample(data_dir: Path, split: str, index: int) -> Tuple[str, Dict[str, np.ndarray]]:
    path = data_dir / f"{split}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        keys = sorted_keys(f)
        if index < 0 or index >= len(keys):
            raise IndexError(f"{split} sample index {index} 超出范围；共有 {len(keys)} 个样本")
        key = keys[index]
        g = f[key]
        names = list(g.keys())
        out: Dict[str, np.ndarray] = {}
        for name in names:
            arr = g[name][()]
            out[name] = np.asarray(arr)
        return key, out


def iter_split(data_dir: Path, split: str):
    path = data_dir / f"{split}.h5"
    if not path.exists():
        return
    with h5py.File(path, "r") as f:
        for i, key in enumerate(sorted_keys(f)):
            g = f[key]
            out = {name: np.asarray(g[name][()]) for name in g.keys()}
            yield i, key, out


def get_arr(s: Dict[str, np.ndarray], name: str, default=None):
    return s[name] if name in s else default


def finite_stat(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": math.nan, "mean": math.nan, "std": math.nan, "p05": math.nan, "p50": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def safe_frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask).reshape(-1)
    if mask.size == 0:
        return math.nan
    return float(np.mean(mask.astype(bool)))


def nearest_distance(points: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if not np.any(mask):
        return math.nan
    pts = points.astype(np.float64)
    ref = pts[mask]
    # 样本只有几千节点，直接算可以接受。
    d2 = np.sum((pts[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    return float(np.sqrt(np.min(d2)))


def excitation_distance_to_mask(points: np.ndarray, exc_idx: int, mask: np.ndarray) -> float:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if not np.any(mask) or not (0 <= exc_idx < len(points)):
        return math.nan
    p = points[exc_idx:exc_idx + 1].astype(np.float64)
    ref = points[mask].astype(np.float64)
    d = np.sqrt(np.sum((ref - p) ** 2, axis=1))
    return float(np.min(d))


def sample_features(split: str, index: int, key: str, s: Dict[str, np.ndarray]) -> Dict[str, float | int | str]:
    points = np.asarray(s["points"], dtype=np.float64)
    n = points.shape[0]
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox = bbox_max - bbox_min
    modal_omega = np.asarray(s["modal_omega"], dtype=np.float64).reshape(-1)
    modal_freq = modal_omega / (2.0 * np.pi)
    n_modes = modal_freq.size
    gaps = np.diff(modal_freq)
    rel_gaps = gaps / np.maximum(modal_freq[:-1], 1e-30) if n_modes >= 2 else np.array([], dtype=float)

    edge_index = np.asarray(s.get("edge_index", np.empty((2, 0))), dtype=np.int64)
    edge_attr = np.asarray(s.get("edge_attr", np.empty((0, 0))), dtype=np.float64)
    spring_k = np.asarray(s.get("spring_k_xyz", np.zeros((n, 3))), dtype=np.float64).reshape(n, -1)
    spring_norm = np.linalg.norm(spring_k, axis=1)
    spring_mask = spring_norm > 0
    node_type = np.asarray(s.get("node_type", np.zeros(n)), dtype=np.int64).reshape(-1)
    pocket_mask = np.asarray(s.get("pocket_bottom_mask", np.zeros(n)), dtype=bool).reshape(-1)
    cut_mask = np.asarray(s.get("cut_region_mask", np.zeros(n)), dtype=bool).reshape(-1)
    local_thick = np.asarray(s.get("local_thickness_ratio", np.full(n, np.nan)), dtype=np.float64).reshape(-1)
    pocket_depth = np.asarray(s.get("pocket_depth_ratio", np.full(n, np.nan)), dtype=np.float64).reshape(-1)
    A = np.asarray(s.get("modal_residue_z", np.empty((n, n_modes))), dtype=np.float64)
    A_abs = np.abs(A)
    A_rms_mode = np.sqrt(np.mean(A_abs ** 2, axis=0)) if A_abs.ndim == 2 else np.full(n_modes, np.nan)

    exc_idx = int(np.asarray(s.get("excitation_index", np.array([-1]))).reshape(-1)[0])
    if "excitation_coord" in s:
        exc_coord = np.asarray(s["excitation_coord"], dtype=np.float64).reshape(3)
    elif 0 <= exc_idx < n:
        exc_coord = points[exc_idx]
    else:
        exc_coord = np.full(3, np.nan)

    edge_len_stat = {}
    if edge_index.ndim == 2 and edge_index.shape[0] == 2 and edge_index.shape[1] > 0:
        src = np.clip(edge_index[0], 0, n - 1)
        dst = np.clip(edge_index[1], 0, n - 1)
        edge_len = np.linalg.norm(points[src] - points[dst], axis=1)
        st = finite_stat(edge_len)
        edge_len_stat = {f"edge_len_{k}_m": v for k, v in st.items()}
    if edge_attr.size > 0:
        ea = np.asarray(edge_attr, dtype=np.float64)
        edge_len_stat["edge_attr_abs_mean"] = float(np.mean(np.abs(ea)))
        edge_len_stat["edge_attr_abs_max"] = float(np.max(np.abs(ea)))

    row: Dict[str, float | int | str] = {
        "split": split,
        "sample_index": index,
        "sample_key": key,
        "n_nodes": int(n),
        "n_edges": int(edge_index.shape[1] if edge_index.ndim == 2 else 0),
        "bbox_x_mm": float(bbox[0] * 1000.0),
        "bbox_y_mm": float(bbox[1] * 1000.0),
        "bbox_z_mm": float(bbox[2] * 1000.0),
        "centroid_x_mm": float(np.mean(points[:, 0]) * 1000.0),
        "centroid_y_mm": float(np.mean(points[:, 1]) * 1000.0),
        "centroid_z_mm": float(np.mean(points[:, 2]) * 1000.0),
        "exc_idx": int(exc_idx),
        "exc_x_mm": float(exc_coord[0] * 1000.0),
        "exc_y_mm": float(exc_coord[1] * 1000.0),
        "exc_z_mm": float(exc_coord[2] * 1000.0),
        "exc_x_norm": float((exc_coord[0] - bbox_min[0]) / max(bbox[0], 1e-30)),
        "exc_y_norm": float((exc_coord[1] - bbox_min[1]) / max(bbox[1], 1e-30)),
        "exc_z_norm": float((exc_coord[2] - bbox_min[2]) / max(bbox[2], 1e-30)),
        "spring_node_count": int(np.sum(spring_mask)),
        "spring_node_fraction": safe_frac(spring_mask),
        "spring_k_sum": float(np.sum(spring_norm)),
        "spring_k_mean_nonzero": float(np.mean(spring_norm[spring_mask])) if np.any(spring_mask) else 0.0,
        "node_type_0_count": int(np.sum(node_type == 0)),
        "node_type_1_count": int(np.sum(node_type == 1)),
        "node_type_2_count": int(np.sum(node_type == 2)),
        "node_type_3_count": int(np.sum(node_type == 3)),
        "node_type_4_count": int(np.sum(node_type == 4)),
        "pocket_bottom_fraction": safe_frac(pocket_mask),
        "cut_region_fraction": safe_frac(cut_mask),
        "exc_to_pocket_min_dist_mm": excitation_distance_to_mask(points, exc_idx, pocket_mask) * 1000.0,
        "exc_to_cut_min_dist_mm": excitation_distance_to_mask(points, exc_idx, cut_mask) * 1000.0,
        "exc_to_spring_min_dist_mm": excitation_distance_to_mask(points, exc_idx, spring_mask) * 1000.0,
        "local_thickness_mean": float(np.nanmean(local_thick)),
        "local_thickness_min": float(np.nanmin(local_thick)),
        "local_thickness_p05": float(np.nanpercentile(local_thick, 5)),
        "pocket_depth_mean": float(np.nanmean(pocket_depth)),
        "pocket_depth_max": float(np.nanmax(pocket_depth)),
        "f01_Hz": float(modal_freq[0]) if n_modes >= 1 else math.nan,
        "f10_Hz": float(modal_freq[9]) if n_modes >= 10 else (float(modal_freq[-1]) if n_modes else math.nan),
        "f_mean_Hz": float(np.mean(modal_freq)) if n_modes else math.nan,
        "min_mode_gap_Hz": float(np.min(gaps)) if gaps.size else math.nan,
        "min_relative_gap": float(np.min(rel_gaps)) if rel_gaps.size else math.nan,
        "min_relative_gap_pair": int(np.argmin(rel_gaps) + 1) if rel_gaps.size else -1,
        "A_abs_global_p50": float(np.percentile(A_abs, 50)) if A_abs.size else math.nan,
        "A_abs_global_p95": float(np.percentile(A_abs, 95)) if A_abs.size else math.nan,
        "A_abs_global_p99": float(np.percentile(A_abs, 99)) if A_abs.size else math.nan,
        "A_abs_global_max": float(np.max(A_abs)) if A_abs.size else math.nan,
        "A_rms_mean_over_modes": float(np.nanmean(A_rms_mode)) if A_rms_mode.size else math.nan,
        "A_rms_max_over_modes": float(np.nanmax(A_rms_mode)) if A_rms_mode.size else math.nan,
    }
    row.update(edge_len_stat)
    for i, f in enumerate(modal_freq[:10], start=1):
        row[f"f{i:02d}_Hz"] = float(f)
    for i, g in enumerate(gaps[:9], start=1):
        row[f"gap{i:02d}_{i+1:02d}_Hz"] = float(g)
    for i, g in enumerate(rel_gaps[:9], start=1):
        row[f"rel_gap{i:02d}_{i+1:02d}"] = float(g)
    for i, v in enumerate(A_rms_mode[:10], start=1):
        row[f"A_rms_m{i:02d}"] = float(v)
    return row


def read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    if not path or not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_reference(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for k, v in row.items():
            if k in {"split", "sample_key"}:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                numeric.setdefault(k, []).append(fv)
    ref = {}
    for k, vals in numeric.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size < 2:
            continue
        ref[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return ref


def zscore(value: object, ref: Dict[str, float]) -> float:
    try:
        v = float(value)
    except Exception:
        return math.nan
    std = ref.get("std", math.nan)
    mean = ref.get("mean", math.nan)
    if not math.isfinite(std) or std < 1e-30 or not math.isfinite(mean):
        return math.nan
    return float((v - mean) / std)


def percentile_rank(value: object, vals: List[float]) -> float:
    try:
        v = float(value)
    except Exception:
        return math.nan
    arr = np.asarray([x for x in vals if math.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr <= v) * 100.0)


def build_outlier_rows(hard_rows: List[Dict[str, object]], train_rows: List[Dict[str, object]], ref: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    train_values: Dict[str, List[float]] = {}
    for row in train_rows:
        for k, v in row.items():
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                train_values.setdefault(k, []).append(fv)

    out: List[Dict[str, object]] = []
    important_prefixes = [
        "n_nodes", "spring", "node_type_", "pocket", "cut", "exc_", "local_thickness",
        "f", "gap", "rel_gap", "min_", "A_", "edge_len",
    ]
    for row in hard_rows:
        for k, r in ref.items():
            if not any(k.startswith(p) for p in important_prefixes):
                continue
            z = zscore(row.get(k, math.nan), r)
            if not math.isfinite(z):
                continue
            pr = percentile_rank(row.get(k, math.nan), train_values.get(k, []))
            if abs(z) >= 1.8 or pr <= 5.0 or pr >= 95.0:
                out.append({
                    "sample_index": row["sample_index"],
                    "sample_key": row["sample_key"],
                    "feature": k,
                    "value": row.get(k, math.nan),
                    "train_mean": r["mean"],
                    "train_std": r["std"],
                    "zscore_vs_train": z,
                    "percentile_vs_train": pr,
                    "train_p05": r["p05"],
                    "train_p50": r["p50"],
                    "train_p95": r["p95"],
                })
    out.sort(key=lambda x: abs(float(x["zscore_vs_train"])), reverse=True)
    return out


def mode_frequency_rows(hard_rows: List[Dict[str, object]], train_rows: List[Dict[str, object]], ref: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    rows = []
    for row in hard_rows:
        for i in range(1, 11):
            fk = f"f{i:02d}_Hz"
            rows.append({
                "sample_index": row["sample_index"],
                "sample_key": row["sample_key"],
                "mode": i,
                "freq_Hz": row.get(fk, math.nan),
                "freq_zscore_vs_train": zscore(row.get(fk, math.nan), ref.get(fk, {})),
                "freq_train_p05": ref.get(fk, {}).get("p05", math.nan),
                "freq_train_p50": ref.get(fk, {}).get("p50", math.nan),
                "freq_train_p95": ref.get(fk, {}).get("p95", math.nan),
                "A_rms": row.get(f"A_rms_m{i:02d}", math.nan),
                "A_rms_zscore_vs_train": zscore(row.get(f"A_rms_m{i:02d}", math.nan), ref.get(f"A_rms_m{i:02d}", {})),
                "gap_to_next_Hz": row.get(f"gap{i:02d}_{i+1:02d}_Hz", math.nan) if i < 10 else math.nan,
                "rel_gap_to_next": row.get(f"rel_gap{i:02d}_{i+1:02d}", math.nan) if i < 10 else math.nan,
                "rel_gap_zscore_vs_train": zscore(row.get(f"rel_gap{i:02d}_{i+1:02d}", math.nan), ref.get(f"rel_gap{i:02d}_{i+1:02d}", {})) if i < 10 else math.nan,
            })
    return rows


def metrics_for_hard_samples(val_metrics: List[Dict[str, str]], sample_indices: List[int]) -> List[Dict[str, object]]:
    if not val_metrics:
        return []
    out = []
    for row in val_metrics:
        try:
            sample = int(float(row.get("sample", row.get("sample_index", -1))))
        except Exception:
            continue
        if sample not in sample_indices:
            continue
        out_row: Dict[str, object] = {"sample_index": sample}
        for k, v in row.items():
            try:
                out_row[k] = float(v)
            except Exception:
                out_row[k] = v
        # 自动找最差频率阶次。
        w_vals = []
        for i in range(1, 11):
            key = f"w{i:02d}_pct"
            if key in out_row and isinstance(out_row[key], float) and math.isfinite(out_row[key]):
                w_vals.append((i, out_row[key]))
        if w_vals:
            worst = max(w_vals, key=lambda x: x[1])
            out_row["worst_w_mode"] = worst[0]
            out_row["worst_w_pct"] = worst[1]
        out.append(out_row)
    out.sort(key=lambda r: int(r["sample_index"]))
    return out


def write_report(path: Path, hard_rows: List[Dict[str, object]], outlier_rows: List[Dict[str, object]], mode_rows: List[Dict[str, object]], error_rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("困难验证样本检测报告")
    lines.append("=" * 88)
    lines.append("")
    lines.append("样本概览：")
    for row in hard_rows:
        lines.append(
            f"  val sample {row['sample_index']} / {row['sample_key']}: "
            f"nodes={row['n_nodes']}, f1={float(row['f01_Hz']):.1f}Hz, "
            f"f10={float(row['f10_Hz']):.1f}Hz, min_rel_gap={float(row['min_relative_gap']):.4f}, "
            f"spring_nodes={row['spring_node_count']}, cut_frac={float(row['cut_region_fraction']):.3f}, "
            f"pocket_frac={float(row['pocket_bottom_fraction']):.3f}"
        )
    lines.append("")

    if error_rows:
        lines.append("预测误差摘要：")
        for row in error_rows:
            sample = int(row["sample_index"])
            rms = row.get("w10_rms_pct", row.get("w10_rms", math.nan))
            mean = row.get("w10_mean_pct", math.nan)
            mx = row.get("w10_max_pct", math.nan)
            worst_mode = row.get("worst_w_mode", "")
            worst_pct = row.get("worst_w_pct", math.nan)
            lines.append(f"  val sample {sample}: w10_mean={mean:.3f}%, w10_rms={rms:.3f}%, w10_max={mx:.3f}%, worst_mode={worst_mode}, worst={worst_pct:.3f}%")
        lines.append("")

    lines.append("最显著离群特征（按 |z-score| 排序，前 40 条）：")
    for r in outlier_rows[:40]:
        lines.append(
            f"  sample {r['sample_index']} | {r['feature']}: value={float(r['value']):.6g}, "
            f"train_p50={float(r['train_p50']):.6g}, z={float(r['zscore_vs_train']):+.2f}, "
            f"percentile={float(r['percentile_vs_train']):.1f}%"
        )
    lines.append("")

    lines.append("模态频率 z-score 摘要：")
    for sample in sorted({int(r["sample_index"]) for r in mode_rows}):
        rows = [r for r in mode_rows if int(r["sample_index"]) == sample]
        txt = []
        for r in rows:
            z = float(r["freq_zscore_vs_train"])
            if math.isfinite(z) and abs(z) >= 1.5:
                txt.append(f"m{int(r['mode']):02d}: f={float(r['freq_Hz']):.1f}Hz z={z:+.2f}")
        if not txt:
            txt = ["无明显频率离群阶次(|z|<1.5)"]
        lines.append(f"  sample {sample}: " + "; ".join(txt))
    lines.append("")

    lines.append("建议判断：")
    lines.append("  1. 如果困难样本在 f04~f08、min_relative_gap、cut/pocket比例、激励点位置等特征上离群，说明误差大可能来自几何/模态分布极端。")
    lines.append("  2. 如果困难样本不离群但误差大，说明当前模型结构或特征表达不足，需要改模型或补充特征。")
    lines.append("  3. 如果某些阶频率接近或 rel_gap 很低，中阶模态顺序/局部模态变化会增加学习难度。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=SAMPLE_INDICES)
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--val-metrics", type=Path, default=None, help="可选：val_metrics.csv，用于合并预测误差")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.data_dir / "hard_sample_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("读取 train/reference 样本特征...")
    train_rows: List[Dict[str, object]] = []
    for idx, key, s in iter_split(args.data_dir, args.reference_split):
        train_rows.append(sample_features(args.reference_split, idx, key, s))
    if not train_rows:
        raise RuntimeError(f"没有读取到 reference split: {args.reference_split}")

    ref = make_reference(train_rows)
    ref_rows = []
    for k, v in ref.items():
        row = {"feature": k}
        row.update(v)
        ref_rows.append(row)
    write_csv(out_dir / "train_feature_reference.csv", ref_rows)

    print("读取困难样本特征...")
    hard_rows = []
    for idx in args.sample_indices:
        key, s = read_sample(args.data_dir, args.split, idx)
        hard_rows.append(sample_features(args.split, idx, key, s))
    write_csv(out_dir / "hard_sample_features.csv", hard_rows)

    outlier_rows = build_outlier_rows(hard_rows, train_rows, ref)
    write_csv(out_dir / "hard_sample_outliers.csv", outlier_rows)

    mode_rows = mode_frequency_rows(hard_rows, train_rows, ref)
    write_csv(out_dir / "hard_sample_mode_frequencies.csv", mode_rows)

    error_rows: List[Dict[str, object]] = []
    if args.val_metrics is not None:
        metrics = read_metrics_csv(args.val_metrics)
        error_rows = metrics_for_hard_samples(metrics, args.sample_indices)
        write_csv(out_dir / "hard_sample_error_metrics.csv", error_rows)

    write_report(out_dir / "hard_sample_report.txt", hard_rows, outlier_rows, mode_rows, error_rows)

    print("完成。输出目录：", out_dir)
    print("  hard_sample_report.txt")
    print("  hard_sample_features.csv")
    print("  hard_sample_outliers.csv")
    print("  hard_sample_mode_frequencies.csv")
    print("  train_feature_reference.csv")
    if args.val_metrics is not None:
        print("  hard_sample_error_metrics.csv")


if __name__ == "__main__":
    main()
