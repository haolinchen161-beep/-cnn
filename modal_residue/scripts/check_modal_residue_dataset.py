from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

REQUIRED = [
    "points", "point_features", "spring_k_xyz", "spring_c_xyz", "node_type",
    "modal_omega", "modal_zeta", "modal_phi_xyz", "modal_residue_z",
    "frequencies", "point_frf", "excitation_index", "excitation_coord",
]


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-20) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def check_sample(g, split: str, key: str, min_rel_gap: float, frf_nodes: int) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    for name in REQUIRED:
        if name not in g:
            errors.append(f"missing:{name}")
    if errors:
        return {"split": split, "sample": key, "status": "ERROR", "errors": ";".join(errors), "warnings": ""}

    points = g["points"][:]
    omega = g["modal_omega"][:]
    zeta = g["modal_zeta"][:]
    phi = g["modal_phi_xyz"][:]
    residue = g["modal_residue_z"][:]
    freqs = g["frequencies"][:]
    frf = g["point_frf"][:]
    exc_idx = int(g["excitation_index"][()])
    exc_coord = g["excitation_coord"][:]

    n_nodes = points.shape[0]
    n_modes = omega.shape[0]
    n_freqs = freqs.shape[0]

    arrays = {"points": points, "omega": omega, "zeta": zeta, "phi": phi, "residue": residue, "freqs": freqs, "frf": frf}
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            errors.append(f"nonfinite:{name}")

    if n_modes != 10:
        errors.append(f"n_modes={n_modes}, expected 10")
    if n_freqs != 120:
        warnings.append(f"n_freqs={n_freqs}, expected 120")
    if phi.shape != (n_nodes, n_modes, 3):
        errors.append(f"bad_phi_shape:{phi.shape}")
    if residue.shape != (n_nodes, n_modes):
        errors.append(f"bad_residue_shape:{residue.shape}")
    if frf.shape != (n_nodes, n_freqs, 2):
        errors.append(f"bad_frf_shape:{frf.shape}")
    if not (0 <= exc_idx < n_nodes):
        errors.append(f"bad_excitation_index:{exc_idx}")
    elif np.linalg.norm(points[exc_idx] - exc_coord) > 1e-6:
        errors.append("excitation_coord_mismatch")

    freq_hz = omega / (2.0 * np.pi)
    if not np.all(np.diff(freq_hz) > 0):
        errors.append("modal_frequency_not_increasing")
    if not np.all(np.diff(freqs) > 0):
        errors.append("frequency_grid_not_increasing")
    if freqs[-1] < freq_hz[-1]:
        errors.append("frequency_grid_does_not_cover_10th_mode")

    gaps = np.diff(freq_hz)
    rel_gaps = gaps / np.maximum(freq_hz[:-1], 1e-12)
    min_gap = float(np.min(gaps)) if len(gaps) else float("nan")
    min_rel = float(np.min(rel_gaps)) if len(rel_gaps) else float("nan")
    pair = int(np.argmin(rel_gaps) + 1) if len(rel_gaps) else -1
    if min_rel_gap > 0 and min_rel < min_rel_gap - 1e-6:
        errors.append(f"near_mode_not_filtered:min_rel={min_rel:.5f}")

    # Modal residue formula: A_r(x)=phi_z(x)*phi_z(xf)
    if 0 <= exc_idx < n_nodes:
        phi_z = phi[:, :, 2]
        residue_calc = phi_z * phi_z[exc_idx:exc_idx + 1, :]
        residue_err = rel_l2(residue, residue_calc)
        if residue_err > 1e-6:
            errors.append(f"modal_residue_formula_error:{residue_err:.3e}")
    else:
        residue_err = float("nan")

    # FRF formula check on a deterministic subset of nodes.
    if n_nodes > 0 and n_freqs > 0:
        idx = np.linspace(0, n_nodes - 1, num=min(frf_nodes, n_nodes), dtype=np.int64)
        omega_q = 2.0 * np.pi * freqs.astype(np.float64)
        pred = np.zeros((len(idx), n_freqs), dtype=np.complex128)
        for k in range(n_modes):
            den = omega[k] ** 2 - omega_q ** 2 + 1j * (2.0 * zeta[k] * omega[k] * omega_q)
            pred += residue[idx, k:k + 1] / den.reshape(1, -1)
        true = frf[idx, :, 0].astype(np.float64) + 1j * frf[idx, :, 1].astype(np.float64)
        frf_err = rel_l2(pred, true)
        if frf_err > 1e-3:
            errors.append(f"frf_formula_error:{frf_err:.3e}")
    else:
        frf_err = float("nan")

    status = "ERROR" if errors else ("WARN" if warnings else "OK")
    return {
        "split": split,
        "sample": key,
        "status": status,
        "n_nodes": n_nodes,
        "n_modes": n_modes,
        "n_freqs": n_freqs,
        "f1_Hz": float(freq_hz[0]) if len(freq_hz) else float("nan"),
        "f10_Hz": float(freq_hz[-1]) if len(freq_hz) else float("nan"),
        "freq_max_Hz": float(freqs[-1]) if len(freqs) else float("nan"),
        "min_gap_Hz": min_gap,
        "min_relative_gap": min_rel,
        "min_relative_gap_pair": pair,
        "modal_residue_rel_l2": residue_err,
        "point_frf_rel_l2": frf_err,
        "errors": ";".join(errors),
        "warnings": ";".join(warnings),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data_modal_residue_filtered"))
    p.add_argument("--min-relative-mode-gap", type=float, default=0.03)
    p.add_argument("--frf-check-nodes", type=int, default=64)
    p.add_argument("--out-report", type=Path, default=Path("dataset_quality_report.txt"))
    p.add_argument("--out-csv", type=Path, default=Path("dataset_quality_samples.csv"))
    args = p.parse_args()

    rows: List[Dict[str, object]] = []
    for split in ["train", "val", "test"]:
        h5_path = args.data_dir / f"{split}.h5"
        if not h5_path.exists():
            rows.append({"split": split, "sample": "-", "status": "ERROR", "errors": f"missing_file:{h5_path}", "warnings": ""})
            continue
        with h5py.File(h5_path, "r") as f:
            for key in sorted(f.keys(), key=lambda s: int(s.split("_")[-1])):
                rows.append(check_sample(f[key], split, key, args.min_relative_mode_gap, args.frf_check_nodes))

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    ok = sum(r.get("status") == "OK" for r in rows)
    warn = sum(r.get("status") == "WARN" for r in rows)
    err = sum(r.get("status") == "ERROR" for r in rows)
    numeric = [r for r in rows if isinstance(r.get("n_nodes"), (int, float))]
    lines = [
        "模态留数数据集质量检查报告",
        "=" * 60,
        f"数据目录: {args.data_dir}",
        f"检查样本总数: {len(rows)}",
        f"状态统计: OK={ok}, WARN={warn}, ERROR={err}",
    ]
    if numeric:
        for name in ["n_nodes", "f10_Hz", "freq_max_Hz", "min_relative_gap", "modal_residue_rel_l2", "point_frf_rel_l2"]:
            vals = np.array([float(r[name]) for r in numeric if name in r and np.isfinite(float(r[name]))])
            if vals.size:
                lines.append(f"{name} min/mean/max: {vals.min():.6g} / {vals.mean():.6g} / {vals.max():.6g}")
    bad = [r for r in rows if r.get("status") != "OK"]
    if bad:
        lines.append("\n非 OK 样本:")
        for r in bad[:50]:
            lines.append(f"  {r.get('split')}/{r.get('sample')}: {r.get('status')} {r.get('errors')} {r.get('warnings')}")
    else:
        lines.append("\n结论: 未发现硬错误。该数据集可以进入训练。")
    text = "\n".join(lines)
    args.out_report.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nCSV: {args.out_csv}")


if __name__ == "__main__":
    main()
