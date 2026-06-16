from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


def safe_percentile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def safe_stat(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": safe_percentile(x, 1),
        "p05": safe_percentile(x, 5),
        "p50": safe_percentile(x, 50),
        "p95": safe_percentile(x, 95),
        "p99": safe_percentile(x, 99),
    }


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def iter_samples(data_dir: Path, splits: Iterable[str]):
    for split in splits:
        path = data_dir / f"{split}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            for key in keys:
                yield split, key, f[key]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_hist(values: np.ndarray, bins: np.ndarray) -> List[Dict[str, object]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    hist, edges = np.histogram(values, bins=bins)
    rows = []
    for i, c in enumerate(hist):
        rows.append({
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(c),
            "fraction": float(c / max(values.size, 1)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_modal_residue_filtered"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--abs-thresholds", nargs="+", type=float, default=[1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6])
    parser.add_argument("--rel-thresholds", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    args = parser.parse_args()

    out_dir = args.out_dir or (args.data_dir / "residue_stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, object]] = []
    formula_rows: List[Dict[str, object]] = []
    mode_values: Dict[int, List[np.ndarray]] = {}
    mode_abs_values: Dict[int, List[np.ndarray]] = {}
    all_abs_chunks: List[np.ndarray] = []
    all_log_abs_chunks: List[np.ndarray] = []

    n_samples = 0
    n_nodes_total = 0
    n_modes_global = None

    for split, key, g in iter_samples(args.data_dir, args.splits):
        if "modal_residue_z" not in g:
            print(f"WARN {split}/{key}: missing modal_residue_z")
            continue
        residue = g["modal_residue_z"][:].astype(np.float64)
        if residue.ndim != 2:
            print(f"WARN {split}/{key}: bad modal_residue_z shape {residue.shape}")
            continue

        n_nodes, n_modes = residue.shape
        n_modes_global = n_modes if n_modes_global is None else n_modes_global
        n_samples += 1
        n_nodes_total += n_nodes

        abs_res = np.abs(residue)
        all_abs_chunks.append(abs_res.reshape(-1))
        all_log_abs_chunks.append(np.log10(abs_res.reshape(-1) + 1e-300))

        sample_scale = float(np.max(abs_res)) if abs_res.size else 0.0
        for r in range(n_modes):
            a = residue[:, r]
            aa = np.abs(a)
            mode_values.setdefault(r, []).append(a)
            mode_abs_values.setdefault(r, []).append(aa)
            row: Dict[str, object] = {
                "split": split,
                "sample": key,
                "mode": r + 1,
                "n_nodes": n_nodes,
                "signed_min": float(np.min(a)),
                "signed_max": float(np.max(a)),
                "signed_mean": float(np.mean(a)),
                "signed_std": float(np.std(a)),
                "abs_min": float(np.min(aa)),
                "abs_max": float(np.max(aa)),
                "abs_mean": float(np.mean(aa)),
                "abs_std": float(np.std(aa)),
                "abs_p01": safe_percentile(aa, 1),
                "abs_p05": safe_percentile(aa, 5),
                "abs_p50": safe_percentile(aa, 50),
                "abs_p95": safe_percentile(aa, 95),
                "abs_p99": safe_percentile(aa, 99),
                "l2_norm": float(np.linalg.norm(a)),
                "rms": float(np.sqrt(np.mean(a * a))),
                "positive_frac": float(np.mean(a > 0.0)),
                "negative_frac": float(np.mean(a < 0.0)),
                "exact_zero_frac": float(np.mean(a == 0.0)),
                "sample_abs_max": sample_scale,
            }
            for th in args.abs_thresholds:
                row[f"frac_abs_lt_{th:.0e}"] = float(np.mean(aa < th))
            for th in args.rel_thresholds:
                limit = th * max(sample_scale, 1e-300)
                row[f"frac_abs_lt_{th:.0e}_sample_max"] = float(np.mean(aa < limit))
            sample_rows.append(row)

        if "modal_phi_xyz" in g and "excitation_index" in g:
            phi = g["modal_phi_xyz"][:].astype(np.float64)
            exc = int(g["excitation_index"][()])
            if phi.shape == (n_nodes, n_modes, 3) and 0 <= exc < n_nodes:
                calc = phi[:, :, 2] * phi[exc:exc + 1, :, 2]
                err_all = rel_l2(residue, calc)
                frow: Dict[str, object] = {
                    "split": split,
                    "sample": key,
                    "all_modes_rel_l2": err_all,
                }
                phi_exc_z = phi[exc, :, 2]
                for r in range(n_modes):
                    frow[f"mode{r + 1:02d}_rel_l2"] = rel_l2(residue[:, r], calc[:, r])
                    frow[f"mode{r + 1:02d}_phi_exc_z"] = float(phi_exc_z[r])
                    frow[f"mode{r + 1:02d}_phi_exc_z_abs"] = float(abs(phi_exc_z[r]))
                formula_rows.append(frow)

    if n_samples == 0:
        raise RuntimeError(f"No samples found in {args.data_dir}")

    mode_rows: List[Dict[str, object]] = []
    n_modes = int(n_modes_global or 0)
    for r in range(n_modes):
        vals = np.concatenate(mode_values.get(r, [np.array([], dtype=float)]))
        absv = np.concatenate(mode_abs_values.get(r, [np.array([], dtype=float)]))
        signed = safe_stat(vals)
        absstat = safe_stat(absv)
        row: Dict[str, object] = {
            "mode": r + 1,
            "count": signed["count"],
            "signed_min": signed["min"],
            "signed_max": signed["max"],
            "signed_mean": signed["mean"],
            "signed_std": signed["std"],
            "signed_p01": signed["p01"],
            "signed_p05": signed["p05"],
            "signed_p50": signed["p50"],
            "signed_p95": signed["p95"],
            "signed_p99": signed["p99"],
            "abs_min": absstat["min"],
            "abs_max": absstat["max"],
            "abs_mean": absstat["mean"],
            "abs_std": absstat["std"],
            "abs_p01": absstat["p01"],
            "abs_p05": absstat["p05"],
            "abs_p50": absstat["p50"],
            "abs_p95": absstat["p95"],
            "abs_p99": absstat["p99"],
            "l2_norm_all_nodes_samples": float(np.linalg.norm(vals)),
            "rms_all_nodes_samples": float(np.sqrt(np.mean(vals * vals))) if vals.size else float("nan"),
            "positive_frac": float(np.mean(vals > 0.0)) if vals.size else float("nan"),
            "negative_frac": float(np.mean(vals < 0.0)) if vals.size else float("nan"),
            "exact_zero_frac": float(np.mean(vals == 0.0)) if vals.size else float("nan"),
        }
        max_abs_mode = float(np.max(absv)) if absv.size else 0.0
        for th in args.abs_thresholds:
            row[f"frac_abs_lt_{th:.0e}"] = float(np.mean(absv < th)) if absv.size else float("nan")
        for th in args.rel_thresholds:
            row[f"frac_abs_lt_{th:.0e}_mode_max"] = float(np.mean(absv < th * max(max_abs_mode, 1e-300))) if absv.size else float("nan")
        mode_rows.append(row)

    all_abs = np.concatenate(all_abs_chunks)
    all_log_abs = np.concatenate(all_log_abs_chunks)
    write_csv(out_dir / "residue_mode_stats.csv", mode_rows)
    write_csv(out_dir / "residue_sample_mode_stats.csv", sample_rows)
    if formula_rows:
        write_csv(out_dir / "residue_formula_check.csv", formula_rows)

    abs_bins = np.logspace(-24, 0, 121)
    log_bins = np.linspace(-300, math.ceil(float(np.max(all_log_abs[np.isfinite(all_log_abs)]))), 301)
    write_csv(out_dir / "residue_abs_hist.csv", make_hist(all_abs, abs_bins))
    write_csv(out_dir / "residue_log10_abs_hist.csv", make_hist(all_log_abs, log_bins))

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("modal_residue_z dataset statistics")
    report_lines.append("=" * 80)
    report_lines.append(f"data_dir: {args.data_dir}")
    report_lines.append(f"splits: {args.splits}")
    report_lines.append(f"samples: {n_samples}")
    report_lines.append(f"total_nodes: {n_nodes_total}")
    report_lines.append(f"n_modes: {n_modes}")
    report_lines.append("")
    report_lines.append("Global |modal_residue_z| statistics:")
    gstat = safe_stat(all_abs)
    for k, v in gstat.items():
        report_lines.append(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    report_lines.append("")
    report_lines.append("Global log10(|modal_residue_z| + 1e-300) statistics:")
    lg = safe_stat(all_log_abs)
    for k, v in lg.items():
        report_lines.append(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    report_lines.append("")
    report_lines.append("Near-zero fractions, global absolute thresholds:")
    for th in args.abs_thresholds:
        report_lines.append(f"  |A| < {th:.0e}: {float(np.mean(all_abs < th)):.6f}")
    report_lines.append("")
    report_lines.append("Per-mode summary: mode, abs_p50, abs_p95, abs_p99, abs_max, rms, frac_abs_lt_1e-12")
    for row in mode_rows:
        report_lines.append(
            f"  m{int(row['mode']):02d}: "
            f"p50={row['abs_p50']:.3e}, "
            f"p95={row['abs_p95']:.3e}, "
            f"p99={row['abs_p99']:.3e}, "
            f"max={row['abs_max']:.3e}, "
            f"rms={row['rms_all_nodes_samples']:.3e}, "
            f"frac<1e-12={row.get('frac_abs_lt_1e-12', float('nan')):.6f}"
        )
    if formula_rows:
        errs = np.array([float(r["all_modes_rel_l2"]) for r in formula_rows], dtype=float)
        report_lines.append("")
        report_lines.append("Formula check with modal_phi_xyz:")
        report_lines.append(f"  rel_l2 min/mean/max: {errs.min():.6e} / {errs.mean():.6e} / {errs.max():.6e}")
    report_lines.append("")
    report_lines.append("Output files:")
    report_lines.append("  residue_mode_stats.csv")
    report_lines.append("  residue_sample_mode_stats.csv")
    report_lines.append("  residue_abs_hist.csv")
    report_lines.append("  residue_log10_abs_hist.csv")
    if formula_rows:
        report_lines.append("  residue_formula_check.csv")

    report = "\n".join(report_lines)
    with open(out_dir / "residue_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
