from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

import h5py
import numpy as np

EPS = 1e-12
DIR_LABELS = ["X", "Y", "Z"]


def get_array(grp: h5py.Group, key: str, default=None):
    if key in grp:
        return grp[key][()]
    return default


def mode_direction_ratio(phi_xyz: np.ndarray) -> np.ndarray:
    energy = np.sum(phi_xyz ** 2, axis=0)  # [M, 3]
    return energy / (np.sum(energy, axis=1, keepdims=True) + EPS)


def classify_dir(ratio: np.ndarray, threshold: float) -> list[str]:
    labels = []
    for row in ratio:
        idx = int(np.argmax(row))
        labels.append(DIR_LABELS[idx] if float(row[idx]) >= threshold else "MIX")
    return labels


def region_ratio(phi_xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    total = np.sum(phi_xyz ** 2, axis=(0, 2)) + EPS
    if mask.sum() == 0:
        return np.zeros(phi_xyz.shape[1], dtype=np.float64)
    part = np.sum(phi_xyz[mask] ** 2, axis=(0, 2))
    return part / total


def region_z_ratio(phi_xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    total = np.sum(phi_xyz[:, :, 2] ** 2, axis=0) + EPS
    if mask.sum() == 0:
        return np.zeros(phi_xyz.shape[1], dtype=np.float64)
    part = np.sum(phi_xyz[mask, :, 2] ** 2, axis=0)
    return part / total


def inspect_sample(file_name: str, sample_name: str, grp: h5py.Group, z_threshold: float, dir_threshold: float):
    points = np.asarray(get_array(grp, "points"), dtype=np.float64)
    phi_xyz = np.asarray(get_array(grp, "modal_phi_xyz"), dtype=np.float64)
    omega = np.asarray(get_array(grp, "modal_omega"), dtype=np.float64).reshape(-1)
    node_type = np.asarray(get_array(grp, "node_type", np.zeros(phi_xyz.shape[0])), dtype=np.int64).reshape(-1)
    spring_c_xyz = np.asarray(get_array(grp, "spring_c_xyz", np.zeros((phi_xyz.shape[0], 3))), dtype=np.float64)
    spring_k_xyz = np.asarray(get_array(grp, "spring_k_xyz", np.zeros((phi_xyz.shape[0], 3))), dtype=np.float64)
    bottom_mask = np.asarray(get_array(grp, "pocket_bottom_mask", np.zeros(phi_xyz.shape[0])), dtype=bool).reshape(-1)
    cut_mask = np.asarray(get_array(grp, "cut_region_mask", np.zeros(phi_xyz.shape[0])), dtype=bool).reshape(-1)
    excitation_index = int(np.asarray(get_array(grp, "excitation_index", -1)).reshape(-1)[0])
    modal_effm = np.asarray(get_array(grp, "modal_effm", np.full((phi_xyz.shape[1], 3), np.nan)), dtype=np.float64)
    modal_pfact = np.asarray(get_array(grp, "modal_pfact", np.full((phi_xyz.shape[1], 3), np.nan)), dtype=np.float64)
    modal_zeta = np.asarray(get_array(grp, "modal_zeta", np.full(phi_xyz.shape[1], np.nan)), dtype=np.float64).reshape(-1)

    n_nodes, n_modes, n_dir = phi_xyz.shape
    if n_dir != 3:
        raise ValueError(f"modal_phi_xyz must be [N, M, 3], got {phi_xyz.shape}")

    freq_hz = omega / (2.0 * np.pi)
    dir_ratio = mode_direction_ratio(phi_xyz)
    dir_label = classify_dir(dir_ratio, dir_threshold)

    clamp_mask = ((node_type == 3) | (node_type == 4)) | (np.sum(spring_k_xyz, axis=1) > 0.0)
    exc_phi = np.zeros((n_modes, 3), dtype=np.float64)
    if 0 <= excitation_index < n_nodes:
        exc_phi = phi_xyz[excitation_index]

    clamp_diss_num = np.sum(spring_c_xyz[:, None, :] * phi_xyz ** 2, axis=(0, 2))
    zeta_clamp = clamp_diss_num / (2.0 * omega + EPS)
    phi_z_rms = np.sqrt(np.mean(phi_xyz[:, :, 2] ** 2, axis=0))
    phi_z_max = np.max(np.abs(phi_xyz[:, :, 2]), axis=0)

    total_z = np.sum(phi_xyz[:, :, 2] ** 2, axis=0)
    z_modes = np.where(dir_ratio[:, 2] >= z_threshold)[0]
    z_modes_by_freq = list(z_modes + 1)

    # Simple FRF relevance score for Z-Z FRF. It is not the real FRF, only a ranking signal.
    # A mode matters if it has Z motion globally and non-small Z motion at the excitation node.
    score_z_frf = dir_ratio[:, 2] * np.abs(exc_phi[:, 2]) * phi_z_rms
    top_score_idx = np.argsort(-score_z_frf)[: min(5, n_modes)]

    rows = []
    for k in range(n_modes):
        row = {
            "file": file_name,
            "sample": sample_name,
            "mode_index": k + 1,
            "n_modes": n_modes,
            "n_nodes": n_nodes,
            "freq_hz": float(freq_hz[k]),
            "zeta_total": float(modal_zeta[k]) if k < len(modal_zeta) else np.nan,
            "dom_dir": dir_label[k],
            "dir_x_ratio": float(dir_ratio[k, 0]),
            "dir_y_ratio": float(dir_ratio[k, 1]),
            "dir_z_ratio": float(dir_ratio[k, 2]),
            "is_z_dominant": int(dir_ratio[k, 2] >= z_threshold),
            "phi_z_rms": float(phi_z_rms[k]),
            "phi_z_max_abs": float(phi_z_max[k]),
            "exc_phi_z": float(exc_phi[k, 2]),
            "abs_exc_phi_z": float(abs(exc_phi[k, 2])),
            "score_z_frf": float(score_z_frf[k]),
            "clamp_energy_ratio": float(region_ratio(phi_xyz, clamp_mask)[k]),
            "bottom_energy_ratio": float(region_ratio(phi_xyz, bottom_mask)[k]),
            "cut_energy_ratio": float(region_ratio(phi_xyz, cut_mask)[k]),
            "clamp_z_ratio": float(region_z_ratio(phi_xyz, clamp_mask)[k]),
            "bottom_z_ratio": float(region_z_ratio(phi_xyz, bottom_mask)[k]),
            "cut_z_ratio": float(region_z_ratio(phi_xyz, cut_mask)[k]),
            "clamp_diss_num": float(clamp_diss_num[k]),
            "zeta_clamp": float(zeta_clamp[k]),
            "effm_x": float(modal_effm[k, 0]) if modal_effm.shape[0] > k and modal_effm.shape[1] > 0 else np.nan,
            "effm_y": float(modal_effm[k, 1]) if modal_effm.shape[0] > k and modal_effm.shape[1] > 1 else np.nan,
            "effm_z": float(modal_effm[k, 2]) if modal_effm.shape[0] > k and modal_effm.shape[1] > 2 else np.nan,
            "pfact_x": float(modal_pfact[k, 0]) if modal_pfact.shape[0] > k and modal_pfact.shape[1] > 0 else np.nan,
            "pfact_y": float(modal_pfact[k, 1]) if modal_pfact.shape[0] > k and modal_pfact.shape[1] > 1 else np.nan,
            "pfact_z": float(modal_pfact[k, 2]) if modal_pfact.shape[0] > k and modal_pfact.shape[1] > 2 else np.nan,
        }
        rows.append(row)

    sample_summary = {
        "file": file_name,
        "sample": sample_name,
        "n_modes": n_modes,
        "z_mode_indices": " ".join(map(str, z_modes_by_freq)),
        "n_z_modes": int(len(z_modes_by_freq)),
        "top_score_modes": " ".join(str(i + 1) for i in top_score_idx),
        "top_score_values": " ".join(f"{score_z_frf[i]:.6e}" for i in top_score_idx),
        "first_z_mode": int(z_modes_by_freq[0]) if z_modes_by_freq else -1,
        "second_z_mode": int(z_modes_by_freq[1]) if len(z_modes_by_freq) > 1 else -1,
        "third_z_mode": int(z_modes_by_freq[2]) if len(z_modes_by_freq) > 2 else -1,
    }
    return rows, sample_summary


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], sample_rows: list[dict]) -> list[dict]:
    out = []
    modes = sorted(set(r["mode_index"] for r in rows))
    for mode in modes:
        rs = [r for r in rows if r["mode_index"] == mode]
        dirs = Counter(r["dom_dir"] for r in rs)
        z_count = sum(int(r["is_z_dominant"]) for r in rs)
        def mean(key):
            vals = np.array([r[key] for r in rs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if len(vals) else np.nan
        out.append({
            "mode_index": mode,
            "count": len(rs),
            "z_dominant_count": z_count,
            "z_dominant_ratio": z_count / max(len(rs), 1),
            "dominant_dir_counts": str(dict(dirs)),
            "freq_mean_hz": mean("freq_hz"),
            "dir_z_ratio_mean": mean("dir_z_ratio"),
            "abs_exc_phi_z_mean": mean("abs_exc_phi_z"),
            "score_z_frf_mean": mean("score_z_frf"),
            "clamp_diss_num_mean": mean("clamp_diss_num"),
            "zeta_clamp_mean": mean("zeta_clamp"),
            "effm_z_mean": mean("effm_z"),
        })

    # Add a compact distribution of target Z-mode order positions.
    for key in ["first_z_mode", "second_z_mode", "third_z_mode"]:
        cnt = Counter(r[key] for r in sample_rows)
        out.append({
            "mode_index": key,
            "count": len(sample_rows),
            "z_dominant_count": "",
            "z_dominant_ratio": "",
            "dominant_dir_counts": str(dict(sorted(cnt.items()))),
            "freq_mean_hz": "",
            "dir_z_ratio_mean": "",
            "abs_exc_phi_z_mean": "",
            "score_z_frf_mean": "",
            "clamp_diss_num_mean": "",
            "zeta_clamp_mean": "",
            "effm_z_mean": "",
        })
    return out


def inspect_h5(path: str, z_threshold: float, dir_threshold: float, max_samples: int):
    all_rows = []
    sample_rows = []
    file_name = os.path.basename(path)
    with h5py.File(path, "r") as f:
        keys = sorted([k for k in f.keys() if isinstance(f[k], h5py.Group)], key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else s)
        if max_samples > 0:
            keys = keys[:max_samples]
        for i, key in enumerate(keys):
            rows, ss = inspect_sample(file_name, key, f[key], z_threshold, dir_threshold)
            all_rows.extend(rows)
            sample_rows.append(ss)
            if (i + 1) % 50 == 0:
                print(f"  {file_name}: processed {i + 1}/{len(keys)}")
    return all_rows, sample_rows


def main():
    parser = argparse.ArgumentParser(description="Inspect 20-mode HDF5 modal family data.")
    parser.add_argument("--data_dir", required=True, help="Folder containing train.h5/val.h5/test.h5")
    parser.add_argument("--files", nargs="+", default=["train.h5", "val.h5", "test.h5"])
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--z_threshold", type=float, default=0.60, help="dir_z_ratio threshold for Z-dominant mode")
    parser.add_argument("--dir_threshold", type=float, default=0.55, help="dominant direction threshold; below it is MIX")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "modal_20mode_check")
    os.makedirs(out_dir, exist_ok=True)

    rows_all = []
    sample_all = []
    for fname in args.files:
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f"Skip missing file: {path}")
            continue
        print(f"Reading {path}")
        rows, samples = inspect_h5(path, args.z_threshold, args.dir_threshold, args.max_samples)
        rows_all.extend(rows)
        sample_all.extend(samples)

    if not rows_all:
        print("No modal rows found.")
        return

    rows_csv = os.path.join(out_dir, "modal_20mode_rows.csv")
    sample_csv = os.path.join(out_dir, "modal_20mode_sample_summary.csv")
    summary_csv = os.path.join(out_dir, "modal_20mode_summary.csv")

    write_csv(rows_csv, rows_all)
    write_csv(sample_csv, sample_all)
    write_csv(summary_csv, summarize(rows_all, sample_all))

    print("\nDone. Outputs:")
    print(f"  {rows_csv}")
    print(f"  {sample_csv}")
    print(f"  {summary_csv}")
    print("\nRead modal_20mode_summary.csv first:")
    print("  - z_dominant_ratio tells which fixed frequency indices are often Z-dominant")
    print("  - first_z_mode / second_z_mode / third_z_mode rows tell where Z target modes appear")
    print("  - score_z_frf_mean ranks modes by Z-FRF relevance signal")


if __name__ == "__main__":
    main()
