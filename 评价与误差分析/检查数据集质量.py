from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def rel_l2(a, b, eps=1e-20):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def validate_group(g, min_relative_gap):
    required = [
        "points", "edge_index", "edge_attr", "point_features",
        "spring_k_xyz", "spring_c_xyz", "node_type",
        "modal_omega", "modal_residue_z",
        "excitation_index", "excitation_coord",
    ]
    errors = []
    warnings = []
    for k in required:
        if k not in g:
            errors.append(f"missing {k}")
    if errors:
        return errors, warnings, {}

    points = g["points"][:]
    edge_index = g["edge_index"][:]
    edge_attr = g["edge_attr"][:]
    omega = g["modal_omega"][:]
    residue = g["modal_residue_z"][:]
    exc = int(g["excitation_index"][()])

    n_nodes = points.shape[0]
    n_modes = omega.shape[0]

    if n_modes != 10:
        warnings.append(f"n_modes={n_modes}, expected 10")
    if residue.shape != (n_nodes, n_modes):
        errors.append(f"bad residue shape {residue.shape}")
    if edge_index.shape[0] != 2:
        errors.append(f"bad edge_index shape {edge_index.shape}")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        errors.append(f"bad edge_attr shape {edge_attr.shape}, edge_index={edge_index.shape}")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n_nodes):
        errors.append("edge_index out of node range")
    if not (0 <= exc < n_nodes):
        errors.append(f"bad excitation index {exc}")

    arrays = [("points", points), ("edge_attr", edge_attr), ("omega", omega), ("residue", residue)]
    for name, arr in arrays:
        if not np.all(np.isfinite(arr)):
            errors.append(f"non-finite {name}")

    f_hz = omega / (2.0 * np.pi)
    if not np.all(np.diff(f_hz) > 0):
        errors.append("modal frequencies are not increasing")

    if n_modes > 1:
        rel_gaps = np.diff(f_hz) / np.maximum(f_hz[:-1], 1e-12)
        min_rel = float(np.min(rel_gaps))
        if min_relative_gap > 0 and min_rel < min_relative_gap - 1e-6:
            warnings.append(f"near mode: min relative gap {min_rel:.5f}")
    else:
        min_rel = np.nan

    residue_err = np.nan
    if "modal_phi_xyz" in g and 0 <= exc < n_nodes:
        phi = g["modal_phi_xyz"][:]
        if phi.shape == (n_nodes, n_modes, 3):
            residue_calc = phi[:, :, 2] * phi[exc:exc + 1, :, 2]
            residue_err = rel_l2(residue, residue_calc)
            if residue_err > 1e-6:
                warnings.append(f"residue formula error {residue_err:.3e}")
        else:
            warnings.append(f"bad modal_phi_xyz shape {phi.shape}")

    return errors, warnings, {
        "n_nodes": n_nodes,
        "n_edges": int(edge_index.shape[1]),
        "f10_hz": float(f_hz[-1]),
        "min_relative_gap": min_rel,
        "residue_err": residue_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("modal_residue/data_modal_residue_fixedclamp300"))
    parser.add_argument("--min-relative-gap", type=float, default=0.03)
    args = parser.parse_args()

    total = 0
    bad = 0
    warn = 0
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
                errors, warnings, info = validate_group(f[key], args.min_relative_gap)
                if errors:
                    bad += 1
                    print(f"ERROR {split}/{key}: " + "; ".join(errors))
                if warnings:
                    warn += 1
                    print(f"WARN  {split}/{key}: " + "; ".join(warnings))
                if not errors:
                    stats.append(info)

    print("=" * 60)
    print(f"checked samples: {total}")
    print(f"errors: {bad}")
    print(f"warnings: {warn}")
    if stats:
        for name in ["n_nodes", "n_edges", "f10_hz", "min_relative_gap", "residue_err"]:
            values = np.array([s[name] for s in stats], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                print(f"{name}: min/mean/max = {values.min():.6g} / {values.mean():.6g} / {values.max():.6g}")
    if bad == 0:
        print("OK: dataset passed required modal/graph checks.")


if __name__ == "__main__":
    main()
