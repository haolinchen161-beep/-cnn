from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def rel_l2(a, b, eps=1e-20):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def validate_group(g, min_relative_gap):
    required = [
        "points", "modal_omega", "modal_zeta", "modal_phi_xyz", "modal_residue_z",
        "frequencies", "point_frf", "excitation_index", "excitation_coord",
    ]
    errors = []
    for k in required:
        if k not in g:
            errors.append(f"missing {k}")
    if errors:
        return errors, {}

    points = g["points"][:]
    omega = g["modal_omega"][:]
    zeta = g["modal_zeta"][:]
    phi = g["modal_phi_xyz"][:]
    residue = g["modal_residue_z"][:]
    freqs = g["frequencies"][:]
    frf = g["point_frf"][:]
    exc = int(g["excitation_index"][()])

    n_nodes = points.shape[0]
    n_modes = omega.shape[0]
    n_freqs = freqs.shape[0]

    if n_modes != 10:
        errors.append(f"n_modes={n_modes}, expected 10")
    if residue.shape != (n_nodes, n_modes):
        errors.append(f"bad residue shape {residue.shape}")
    if phi.shape != (n_nodes, n_modes, 3):
        errors.append(f"bad phi shape {phi.shape}")
    if frf.shape != (n_nodes, n_freqs, 2):
        errors.append(f"bad frf shape {frf.shape}")
    if not (0 <= exc < n_nodes):
        errors.append(f"bad excitation index {exc}")

    for name, arr in [("points", points), ("omega", omega), ("zeta", zeta), ("phi", phi), ("residue", residue), ("freqs", freqs), ("frf", frf)]:
        if not np.all(np.isfinite(arr)):
            errors.append(f"non-finite {name}")

    f_hz = omega / (2.0 * np.pi)
    if not np.all(np.diff(f_hz) > 0):
        errors.append("modal frequencies are not increasing")
    if not np.all(np.diff(freqs) > 0):
        errors.append("frequency grid is not increasing")
    if freqs[-1] < f_hz[-1]:
        errors.append("frequency grid does not cover 10th mode")

    rel_gaps = np.diff(f_hz) / np.maximum(f_hz[:-1], 1e-12)
    min_rel = float(np.min(rel_gaps))
    if min_relative_gap > 0 and min_rel < min_relative_gap - 1e-6:
        errors.append(f"near mode: min relative gap {min_rel:.5f}")

    if 0 <= exc < n_nodes:
        residue_calc = phi[:, :, 2] * phi[exc:exc + 1, :, 2]
        residue_err = rel_l2(residue, residue_calc)
        if residue_err > 1e-6:
            errors.append(f"residue formula error {residue_err:.3e}")
    else:
        residue_err = np.nan

    idx = np.linspace(0, n_nodes - 1, num=min(64, n_nodes), dtype=np.int64)
    w = 2.0 * np.pi * freqs.astype(np.float64)
    pred = np.zeros((len(idx), len(freqs)), dtype=np.complex128)
    for r in range(n_modes):
        den = omega[r] ** 2 - w ** 2 + 1j * (2.0 * zeta[r] * omega[r] * w)
        pred += residue[idx, r:r + 1] / den.reshape(1, -1)
    true = frf[idx, :, 0].astype(np.float64) + 1j * frf[idx, :, 1].astype(np.float64)
    frf_err = rel_l2(pred, true)
    if frf_err > 1e-3:
        errors.append(f"FRF formula error {frf_err:.3e}")

    return errors, {
        "n_nodes": n_nodes,
        "f10_hz": float(f_hz[-1]),
        "fmax_hz": float(freqs[-1]),
        "min_relative_gap": min_rel,
        "residue_err": residue_err,
        "frf_err": frf_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_modal_residue_filtered"))
    parser.add_argument("--min-relative-gap", type=float, default=0.03)
    args = parser.parse_args()

    total = 0
    bad = 0
    stats = []
    for split in ["train", "val", "test"]:
        path = args.data_dir / f"{split}.h5"
        if not path.exists():
            print(f"ERROR: missing {path}")
            bad += 1
            continue
        with h5py.File(path, "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            for key in keys:
                total += 1
                errors, info = validate_group(f[key], args.min_relative_gap)
                if errors:
                    bad += 1
                    print(f"ERROR {split}/{key}: " + "; ".join(errors))
                else:
                    stats.append(info)

    print("=" * 60)
    print(f"checked samples: {total}")
    print(f"errors: {bad}")
    if stats:
        for name in ["n_nodes", "f10_hz", "fmax_hz", "min_relative_gap", "residue_err", "frf_err"]:
            values = np.array([s[name] for s in stats], dtype=float)
            print(f"{name}: min/mean/max = {values.min():.6g} / {values.mean():.6g} / {values.max():.6g}")
    if bad == 0:
        print("OK: dataset passed quality checks.")


if __name__ == "__main__":
    main()
