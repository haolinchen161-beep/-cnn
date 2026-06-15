"""Stratified train/val/test split and diagnostics for generated MeshGraphNet HDF5 files.

The generator intentionally does not store derived labels such as mode_type.
This utility derives the third-mode dominant direction from modal_phi_xyz only
for splitting, then rewrites train.h5 / val.h5 / test.h5 with a more balanced
mode-3 X/Y/Z distribution.

It also writes two diagnostics files after splitting:
    stratified_split_metrics.csv
    stratified_split_report.txt

Run after ansys/generate_3d_test.py:
    python ansys/stratified_resplit_h5.py --in-place
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import h5py
import numpy as np


Ref = Tuple[str, str, int]
DIR_NAMES = {0: "X", 1: "Y", 2: "Z"}
REQUIRED_FIELDS = [
    "points", "edge_index", "edge_attr", "point_features", "spring_k_xyz", "spring_c_xyz",
    "node_type", "pocket_bottom_mask", "cut_region_mask", "local_thickness_ratio", "pocket_depth_ratio",
    "point_frf", "frequencies", "modal_omega", "modal_zeta", "modal_phi_xyz", "modal_phi_exc",
    "modal_mass", "modal_stiffness", "modal_effm", "modal_pfact", "excitation_index", "excitation_coord",
]


def _read_phi_xyz(h5_group) -> np.ndarray:
    if "modal_phi_xyz" in h5_group:
        phi = np.asarray(h5_group["modal_phi_xyz"][:])  # [N,K,3]
    elif "modal_phi" in h5_group:
        phi = np.asarray(h5_group["modal_phi"][:])
        if phi.ndim == 2:
            tmp = np.zeros((phi.shape[0], phi.shape[1], 3), dtype=phi.dtype)
            tmp[..., 2] = phi
            phi = tmp
    else:
        raise KeyError("sample does not contain modal_phi_xyz or modal_phi")
    if phi.ndim != 3 or phi.shape[1] < 3 or phi.shape[2] != 3:
        raise ValueError(f"unexpected phi shape: {phi.shape}")
    return phi


def mode_energy_ratio(phi: np.ndarray) -> np.ndarray:
    """Return [K,3] XYZ squared-amplitude ratio for each mode."""
    energy = np.sum(phi ** 2, axis=0).astype(np.float64)  # [K,3]
    denom = np.sum(energy, axis=1, keepdims=True) + 1e-30
    return energy / denom


def mode3_direction(h5_group) -> int:
    """Return 0/1/2 for X/Y/Z dominant direction of mode 3."""
    ratio = mode_energy_ratio(_read_phi_xyz(h5_group))
    return int(np.argmax(ratio[2]))


def sample_keys(f) -> List[str]:
    keys = [k for k in f.keys() if k.startswith("sample_")]
    return sorted(keys, key=lambda k: int(k.split("_")[-1]))


def collect_refs(data_dir: str, filenames: List[str]) -> List[Ref]:
    refs: List[Ref] = []
    for name in filenames:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            for key in sample_keys(f):
                refs.append((path, key, mode3_direction(f[key])))
    return refs


def build_stratified_split(refs: List[Ref], counts: Dict[str, int], seed: int) -> Dict[str, List[Ref]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Ref]] = defaultdict(list)
    for ref in refs:
        by_class[ref[2]].append(ref)
    for cls_refs in by_class.values():
        rng.shuffle(cls_refs)

    total_target = sum(counts.values())
    if total_target != len(refs):
        raise ValueError(f"target split counts {counts} sum to {total_target}, but found {len(refs)} samples")

    result = {"train": [], "val": [], "test": []}
    ratios = {k: counts[k] / total_target for k in result}

    for cls in sorted(by_class):
        cls_refs = by_class[cls]
        n = len(cls_refs)
        n_val = int(round(n * ratios["val"]))
        n_test = int(round(n * ratios["test"]))
        n_train = n - n_val - n_test
        result["train"].extend(cls_refs[:n_train])
        result["val"].extend(cls_refs[n_train:n_train + n_val])
        result["test"].extend(cls_refs[n_train + n_val:])

    # Enforce exact total counts. This may slightly perturb class balance, but
    # avoids off-by-one problems caused by per-class rounding.
    pool: List[Ref] = []
    for split in ["train", "val", "test"]:
        rng.shuffle(result[split])
        while len(result[split]) > counts[split]:
            pool.append(result[split].pop())
    rng.shuffle(pool)
    for split in ["train", "val", "test"]:
        while len(result[split]) < counts[split] and pool:
            result[split].append(pool.pop())

    for split in result:
        rng.shuffle(result[split])
        if len(result[split]) != counts[split]:
            raise RuntimeError(f"failed to build exact split for {split}: {len(result[split])} != {counts[split]}")
    return result


def copy_file_attrs(src_path: str, out_file) -> None:
    with h5py.File(src_path, "r") as src:
        for k, v in src.attrs.items():
            out_file.attrs[k] = v
        out_file.attrs["split_method"] = "mode3_direction_stratified"


def write_split(out_path: str, refs: List[Ref]) -> None:
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with h5py.File(tmp_path, "w") as out:
        copy_file_attrs(refs[0][0], out)
        for i, (src_path, sample_key, _) in enumerate(refs):
            with h5py.File(src_path, "r") as src:
                src.copy(sample_key, out, name=f"sample_{i}")
                out[f"sample_{i}"].attrs["source_file"] = os.path.basename(src_path)
                out[f"sample_{i}"].attrs["source_sample"] = sample_key

    os.replace(tmp_path, out_path)


def class_counts(refs: List[Ref]) -> Dict[str, int]:
    counts = {"X": 0, "Y": 0, "Z": 0}
    for _, _, cls in refs:
        counts[DIR_NAMES[cls]] += 1
    return counts


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _shape_text(arr) -> str:
    return "x".join(str(int(v)) for v in arr.shape)


def inspect_sample(split: str, sample_name: str, g) -> Dict[str, object]:
    row: Dict[str, object] = {"split": split, "sample": sample_name}
    missing = [k for k in REQUIRED_FIELDS if k not in g]
    row["missing_fields"] = ";".join(missing)

    points = np.asarray(g["points"][:]) if "points" in g else np.zeros((0, 3), dtype=np.float32)
    row["n_nodes"] = int(points.shape[0])
    row["points_shape"] = _shape_text(points)

    if "edge_index" in g:
        edge_index = np.asarray(g["edge_index"][:])
        if edge_index.ndim == 2 and edge_index.shape[0] == 2:
            n_edges = edge_index.shape[1]
            edge_oob = bool(edge_index.size > 0 and (edge_index.min() < 0 or edge_index.max() >= max(1, points.shape[0])))
        elif edge_index.ndim == 2 and edge_index.shape[1] == 2:
            n_edges = edge_index.shape[0]
            edge_oob = bool(edge_index.size > 0 and (edge_index.min() < 0 or edge_index.max() >= max(1, points.shape[0])))
        else:
            n_edges = -1
            edge_oob = True
        row["n_edges"] = int(n_edges)
        row["edge_index_shape"] = _shape_text(edge_index)
        row["edge_oob"] = int(edge_oob)
    else:
        row["n_edges"] = -1
        row["edge_index_shape"] = "MISSING"
        row["edge_oob"] = 1

    if "edge_attr" in g:
        edge_attr = np.asarray(g["edge_attr"][:])
        row["edge_attr_shape"] = _shape_text(edge_attr)
        row["edge_attr_dim_ok"] = int(edge_attr.ndim == 2 and edge_attr.shape[-1] == 4)
    else:
        row["edge_attr_shape"] = "MISSING"
        row["edge_attr_dim_ok"] = 0

    if "frequencies" in g:
        freqs = np.asarray(g["frequencies"][:], dtype=np.float64)
        diffs = np.diff(freqs)
        row["freq_grid_len"] = int(len(freqs))
        row["freq_grid_min"] = _safe_float(np.min(freqs)) if len(freqs) else np.nan
        row["freq_grid_max"] = _safe_float(np.max(freqs)) if len(freqs) else np.nan
        row["freq_df_min"] = _safe_float(np.min(diffs)) if len(diffs) else np.nan
        row["freq_df_median"] = _safe_float(np.median(diffs)) if len(diffs) else np.nan
        row["freq_grid_strict"] = int(len(freqs) == 60 and np.all(diffs > 0.0))
    else:
        row.update({"freq_grid_len": 0, "freq_grid_min": np.nan, "freq_grid_max": np.nan,
                    "freq_df_min": np.nan, "freq_df_median": np.nan, "freq_grid_strict": 0})

    if "modal_omega" in g:
        omega = np.asarray(g["modal_omega"][:], dtype=np.float64)
        fhz = omega / (2.0 * np.pi)
        row["f1_Hz"], row["f2_Hz"], row["f3_Hz"] = [_safe_float(v) for v in fhz[:3]]
        row["gap21_Hz"] = _safe_float(fhz[1] - fhz[0]) if len(fhz) >= 2 else np.nan
        row["gap32_Hz"] = _safe_float(fhz[2] - fhz[1]) if len(fhz) >= 3 else np.nan
        row["modal_freq_increasing"] = int(len(fhz) >= 3 and np.all(np.diff(fhz[:3]) > 0.0))
        row["gap32_gt_200"] = int(len(fhz) >= 3 and (fhz[2] - fhz[1]) > 200.0)
    else:
        for k in ["f1_Hz", "f2_Hz", "f3_Hz", "gap21_Hz", "gap32_Hz"]:
            row[k] = np.nan
        row["modal_freq_increasing"] = 0
        row["gap32_gt_200"] = 0

    if "modal_zeta" in g:
        zeta = np.asarray(g["modal_zeta"][:], dtype=np.float64)
        row["zeta1"], row["zeta2"], row["zeta3"] = [_safe_float(v) for v in zeta[:3]]
        row["zeta_positive"] = int(len(zeta) >= 3 and np.all(zeta[:3] > 0.0))
        row["zeta_in_model_range"] = int(len(zeta) >= 3 and np.all((zeta[:3] >= 0.001) & (zeta[:3] <= 0.031)))
        row["log_zeta1"], row["log_zeta2"], row["log_zeta3"] = [_safe_float(v) for v in np.log(np.maximum(zeta[:3], 1e-12))]
    else:
        for k in ["zeta1", "zeta2", "zeta3", "log_zeta1", "log_zeta2", "log_zeta3"]:
            row[k] = np.nan
        row["zeta_positive"] = 0
        row["zeta_in_model_range"] = 0

    if "modal_phi_xyz" in g or "modal_phi" in g:
        phi = _read_phi_xyz(g)
        row["phi_shape"] = _shape_text(phi)
        ratio = mode_energy_ratio(phi)
        for mode in range(min(3, ratio.shape[0])):
            dom = int(np.argmax(ratio[mode]))
            row[f"mode{mode+1}_dom"] = DIR_NAMES[dom]
            row[f"mode{mode+1}_X_ratio"] = _safe_float(ratio[mode, 0])
            row[f"mode{mode+1}_Y_ratio"] = _safe_float(ratio[mode, 1])
            row[f"mode{mode+1}_Z_ratio"] = _safe_float(ratio[mode, 2])
            row[f"mode{mode+1}_phi_l2"] = _safe_float(np.sqrt(np.sum(phi[:, mode, :] ** 2)))
            row[f"mode{mode+1}_phi_std"] = _safe_float(np.std(phi[:, mode, :]))
        row["phi_shape_ok"] = int(phi.ndim == 3 and phi.shape[1] == 3 and phi.shape[2] == 3 and phi.shape[0] == points.shape[0])
    else:
        row["phi_shape"] = "MISSING"
        row["phi_shape_ok"] = 0

    if "modal_phi_exc" in g:
        phi_exc = np.asarray(g["modal_phi_exc"][:])
        row["phi_exc_shape"] = _shape_text(phi_exc)
        row["phi_exc_shape_ok"] = int(phi_exc.shape == (3, 3) or phi_exc.shape == (3,))
        if phi_exc.ndim == 2 and phi_exc.shape[-1] == 3:
            row["phi_exc_z_abs_mean"] = _safe_float(np.mean(np.abs(phi_exc[:, 2])))
            row["phi_exc_used_by_training"] = "Z_column"
        elif phi_exc.ndim == 1:
            row["phi_exc_z_abs_mean"] = _safe_float(np.mean(np.abs(phi_exc)))
            row["phi_exc_used_by_training"] = "already_Z"
        else:
            row["phi_exc_z_abs_mean"] = np.nan
            row["phi_exc_used_by_training"] = "UNKNOWN"
    else:
        row["phi_exc_shape"] = "MISSING"
        row["phi_exc_shape_ok"] = 0
        row["phi_exc_z_abs_mean"] = np.nan
        row["phi_exc_used_by_training"] = "MISSING"

    if "modal_mass" in g:
        mass = np.asarray(g["modal_mass"][:], dtype=np.float64)
        row["modal_mass_mean"] = _safe_float(np.mean(mass))
        row["modal_mass_max_abs_err_from_1"] = _safe_float(np.max(np.abs(mass - 1.0)))
    else:
        row["modal_mass_mean"] = np.nan
        row["modal_mass_max_abs_err_from_1"] = np.nan

    if "modal_stiffness" in g and "modal_omega" in g and "modal_mass" in g:
        omega = np.asarray(g["modal_omega"][:], dtype=np.float64)
        mass = np.asarray(g["modal_mass"][:], dtype=np.float64)
        stiff = np.asarray(g["modal_stiffness"][:], dtype=np.float64)
        expected = omega ** 2 * mass
        row["modal_stiffness_rel_err_max"] = _safe_float(np.max(np.abs(stiff - expected) / (np.abs(expected) + 1e-12)))
    else:
        row["modal_stiffness_rel_err_max"] = np.nan

    if "point_features" in g:
        pf = np.asarray(g["point_features"][:])
        row["point_features_shape"] = _shape_text(pf)
        row["point_features_dim_ok"] = int(pf.ndim == 2 and pf.shape[1] == 7)
    else:
        row["point_features_shape"] = "MISSING"
        row["point_features_dim_ok"] = 0

    if "local_thickness_ratio" in g:
        ltr = np.asarray(g["local_thickness_ratio"][:], dtype=np.float64)
        row["local_thickness_min"] = _safe_float(np.min(ltr))
        row["local_thickness_mean"] = _safe_float(np.mean(ltr))
    else:
        row["local_thickness_min"] = np.nan
        row["local_thickness_mean"] = np.nan

    if "pocket_depth_ratio" in g:
        pdr = np.asarray(g["pocket_depth_ratio"][:], dtype=np.float64)
        row["pocket_depth_max"] = _safe_float(np.max(pdr))
        row["pocket_depth_mean"] = _safe_float(np.mean(pdr))
    else:
        row["pocket_depth_max"] = np.nan
        row["pocket_depth_mean"] = np.nan

    if "spring_k_xyz" in g and "spring_c_xyz" in g:
        sk = np.asarray(g["spring_k_xyz"][:], dtype=np.float64)
        sc = np.asarray(g["spring_c_xyz"][:], dtype=np.float64)
        row["spring_k_nonzero_nodes"] = int(np.sum(np.any(sk > 0, axis=1)))
        row["spring_c_nonzero_nodes"] = int(np.sum(np.any(sc > 0, axis=1)))
        row["spring_k_max"] = _safe_float(np.max(sk))
        row["spring_c_max"] = _safe_float(np.max(sc))
    else:
        row["spring_k_nonzero_nodes"] = 0
        row["spring_c_nonzero_nodes"] = 0
        row["spring_k_max"] = np.nan
        row["spring_c_max"] = np.nan

    if "excitation_index" in g:
        exc_idx = int(g["excitation_index"][()])
        row["excitation_index"] = exc_idx
        row["excitation_index_in_range"] = int(0 <= exc_idx < points.shape[0])
        if "excitation_coord" in g and 0 <= exc_idx < points.shape[0]:
            exc_coord = np.asarray(g["excitation_coord"][:], dtype=np.float64)
            row["excitation_coord_err"] = _safe_float(np.linalg.norm(exc_coord - points[exc_idx]))
        else:
            row["excitation_coord_err"] = np.nan
    else:
        row["excitation_index"] = -1
        row["excitation_index_in_range"] = 0
        row["excitation_coord_err"] = np.nan

    if "point_frf" in g:
        frf = np.asarray(g["point_frf"][:], dtype=np.float32)
        row["frf_shape"] = _shape_text(frf)
        row["frf_shape_ok"] = int(frf.ndim == 3 and frf.shape[0] == points.shape[0] and frf.shape[2] == 2 and frf.shape[1] == row.get("freq_grid_len", -1))
        amp = np.sqrt(frf[..., 0] ** 2 + frf[..., 1] ** 2)
        row["frf_amp_mean"] = _safe_float(np.mean(amp))
        row["frf_amp_p99"] = _safe_float(np.percentile(amp, 99))
        row["frf_amp_max"] = _safe_float(np.max(amp))
        row["frf_finite"] = int(np.all(np.isfinite(frf)))
    else:
        row["frf_shape"] = "MISSING"
        row["frf_shape_ok"] = 0
        row["frf_amp_mean"] = np.nan
        row["frf_amp_p99"] = np.nan
        row["frf_amp_max"] = np.nan
        row["frf_finite"] = 0

    bool_checks = [
        "freq_grid_strict", "modal_freq_increasing", "gap32_gt_200", "zeta_positive",
        "zeta_in_model_range", "phi_shape_ok", "phi_exc_shape_ok", "point_features_dim_ok",
        "excitation_index_in_range", "frf_shape_ok", "frf_finite",
    ]
    row["error_flags"] = ";".join([k for k in bool_checks if int(row.get(k, 0)) != 1] + (["missing_fields"] if missing else []))
    return row


def write_diagnostics(data_dir: str, out_names: Dict[str, str]) -> None:
    rows: List[Dict[str, object]] = []
    for split, name in out_names.items():
        path = os.path.join(data_dir, name)
        with h5py.File(path, "r") as f:
            for key in sample_keys(f):
                rows.append(inspect_sample(split, key, f[key]))

    metrics_path = os.path.join(data_dir, "stratified_split_metrics.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(metrics_path, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    report_path = os.path.join(data_dir, "stratified_split_report.txt")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("MeshGraphNet stratified split diagnostics\n")
        fp.write("=" * 80 + "\n\n")
        fp.write(f"metrics_csv: {metrics_path}\n")
        fp.write(f"n_samples: {len(rows)}\n\n")

        error_rows = [r for r in rows if str(r.get("error_flags", ""))]
        fp.write(f"ERROR_ROWS: {len(error_rows)}\n")
        if error_rows[:20]:
            fp.write("First error rows:\n")
            for r in error_rows[:20]:
                fp.write(f"  {r['split']}/{r['sample']}: {r['error_flags']}\n")
        fp.write("\n")

        for split in ["train", "val", "test", "ALL"]:
            subset = rows if split == "ALL" else [r for r in rows if r["split"] == split]
            if not subset:
                continue
            fp.write(f"[{split}] n={len(subset)}\n")
            for mode in [1, 2, 3]:
                counts = Counter(str(r.get(f"mode{mode}_dom", "NA")) for r in subset)
                fp.write(f"  mode{mode}_dominant_counts: {dict(counts)}\n")

            for key in [
                "n_nodes", "n_edges", "f1_Hz", "f2_Hz", "f3_Hz", "gap21_Hz", "gap32_Hz",
                "zeta1", "zeta2", "zeta3", "freq_df_min", "freq_df_median",
                "mode3_X_ratio", "mode3_Y_ratio", "mode3_Z_ratio", "phi_exc_z_abs_mean",
                "modal_mass_max_abs_err_from_1", "modal_stiffness_rel_err_max",
                "local_thickness_min", "pocket_depth_max", "spring_k_nonzero_nodes", "spring_c_nonzero_nodes",
                "frf_amp_mean", "frf_amp_p99", "frf_amp_max",
            ]:
                vals = np.asarray([r.get(key, np.nan) for r in subset], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                fp.write(
                    f"  {key:32s} min={np.min(vals):.6g} p5={np.percentile(vals,5):.6g} "
                    f"mean={np.mean(vals):.6g} p95={np.percentile(vals,95):.6g} max={np.max(vals):.6g}\n"
                )
            fp.write("\n")

        fp.write("Training-definition checks\n")
        fp.write("- modal_omega is stored in rad/s; training converts to Hz only inside frequency loss.\n")
        fp.write("- modal_zeta is [K] scalar damping per mode, not [K,3]. Log-zeta loss is appropriate.\n")
        fp.write("- modal_phi_xyz should be [N,3,3]. Training predicts full [N,K,3].\n")
        fp.write("- modal_phi_exc may be [3,3]; dataset/training uses its Z column for Z-direction FRF.\n")
        fp.write("- frequencies should be [60] strictly increasing; FRF should be [N,60,2].\n")
        fp.write("- modal_mass should be 1 and modal_stiffness should equal modal_omega**2 * modal_mass.\n")
        fp.write("- ZetaHead current output range is [0.001, 0.031]; zeta_in_model_range should be true for all samples.\n")

    print(f"diagnostics wrote: {metrics_path}")
    print(f"diagnostics wrote: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-place", action="store_true", help="overwrite train.h5/val.h5/test.h5 after making .sequential.bak backups")
    parser.add_argument("--no-diagnostics", action="store_true", help="skip stratified_split_metrics.csv and stratified_split_report.txt")
    args = parser.parse_args()

    data_dir = args.data_dir
    input_names = ["train.h5", "val.h5", "test.h5"]
    refs = collect_refs(data_dir, input_names)
    counts = {"train": args.train, "val": args.val, "test": args.test}
    splits = build_stratified_split(refs, counts, args.seed)

    print("Original mode-3 direction counts:", class_counts(refs))
    for split, split_refs in splits.items():
        print(f"{split:5s}: n={len(split_refs):3d}, mode3={class_counts(split_refs)}")

    if args.in_place:
        for name in input_names:
            path = os.path.join(data_dir, name)
            bak = path + ".sequential.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
        out_names = {"train": "train.h5", "val": "val.h5", "test": "test.h5"}
    else:
        out_names = {"train": "train_stratified.h5", "val": "val_stratified.h5", "test": "test_stratified.h5"}

    for split, out_name in out_names.items():
        write_split(os.path.join(data_dir, out_name), splits[split])
        print("wrote", os.path.join(data_dir, out_name))

    if not args.no_diagnostics:
        write_diagnostics(data_dir, out_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
