"""Filter an existing HDF5 graph dataset to Z-dominant modal samples.

This script creates a new dataset channel from the already generated ANSYS files.
It does not rerun ANSYS and does not modify ansys/data/*.h5.

Default input:
    ansys/data/train.h5, val.h5, test.h5

Default output:
    ansys/data_z_dominant/train.h5, val.h5, test.h5

Default criterion:
    Keep a sample only when all selected modes are Z-dominant by modal_phi_xyz energy.
    By default SELECT_MODES=1,2,3, meaning all three modes must be Z-dominant.

Environment variables:
    SRC_DIR=ansys/data
    DST_DIR=ansys/data_z_dominant
    SELECT_MODES=1,2,3      # one-based mode numbers; e.g. "3" keeps only mode3 Z-dominant
    MIN_Z_RATIO=0.0         # optional stricter threshold; e.g. 0.6 requires Z energy ratio >= 0.6

Recommended commands from project root:
    F:/pytorch_cuda12/python.exe -B ansys/filter_z_dominant_dataset.py

Loose mode3-only filter:
    set SELECT_MODES=3
    F:/pytorch_cuda12/python.exe -B ansys/filter_z_dominant_dataset.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(os.getenv("SRC_DIR", str(ROOT / "ansys" / "data"))).resolve()
DST_DIR = Path(os.getenv("DST_DIR", str(ROOT / "ansys" / "data_z_dominant"))).resolve()
SELECT_MODES = [int(x.strip()) - 1 for x in os.getenv("SELECT_MODES", "1,2,3").split(",") if x.strip()]
MIN_Z_RATIO = float(os.getenv("MIN_Z_RATIO", "0.0"))
SPLITS = ["train", "val", "test"]


def _modal_direction_and_ratio(phi_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return modal dominant direction and Z energy ratio.

    phi_xyz shape is [N, K, 3]. Direction ids: 0=X, 1=Y, 2=Z.
    """
    energy = np.sum(phi_xyz ** 2, axis=0)  # [K, 3]
    denom = np.sum(energy, axis=-1) + 1e-12
    ratio = energy / denom[:, None]
    direction = np.argmax(ratio, axis=-1)
    return direction.astype(np.int64), ratio[:, 2].astype(np.float64)


def _copy_attrs(src_obj, dst_obj) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def _copy_group(src_grp: h5py.Group, dst_grp: h5py.Group) -> None:
    _copy_attrs(src_grp, dst_grp)
    for name, item in src_grp.items():
        if isinstance(item, h5py.Dataset):
            compression = item.compression
            dst_grp.create_dataset(name, data=item[()], compression=compression)
        elif isinstance(item, h5py.Group):
            child = dst_grp.create_group(name)
            _copy_group(item, child)


def filter_split(split: str) -> dict[str, object]:
    src_path = SRC_DIR / f"{split}.h5"
    dst_path = DST_DIR / f"{split}.h5"
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    kept = []
    counts_all = {"X": 0, "Y": 0, "Z": 0}
    counts_kept = {"X": 0, "Y": 0, "Z": 0}
    names = ["X", "Y", "Z"]

    with h5py.File(src_path, "r") as fsrc:
        keys = sorted(fsrc.keys())
        for key in keys:
            phi = fsrc[key]["modal_phi_xyz"][:]
            direction, z_ratio = _modal_direction_and_ratio(phi)
            # Count mode3 distribution for diagnostics.
            counts_all[names[int(direction[2])]] += 1
            keep = True
            for mode_idx in SELECT_MODES:
                if mode_idx < 0 or mode_idx >= len(direction):
                    raise ValueError(f"SELECT_MODES contains invalid mode {mode_idx + 1}; available K={len(direction)}")
                if int(direction[mode_idx]) != 2:
                    keep = False
                    break
                if z_ratio[mode_idx] < MIN_Z_RATIO:
                    keep = False
                    break
            if keep:
                kept.append(key)
                counts_kept[names[int(direction[2])]] += 1

        DST_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_suffix(".tmp.h5")
        if tmp_path.exists():
            tmp_path.unlink()
        with h5py.File(tmp_path, "w") as fdst:
            _copy_attrs(fsrc, fdst)
            fdst.attrs["source_file"] = str(src_path)
            fdst.attrs["filter"] = "Z-dominant modal subset"
            fdst.attrs["select_modes_one_based"] = ",".join(str(i + 1) for i in SELECT_MODES)
            fdst.attrs["min_z_ratio"] = MIN_Z_RATIO
            fdst.attrs["n_source_samples"] = len(keys)
            fdst.attrs["n_kept_samples"] = len(kept)
            for out_idx, key in enumerate(kept):
                dst_grp = fdst.create_group(f"sample_{out_idx}")
                _copy_group(fsrc[key], dst_grp)
        shutil.move(str(tmp_path), str(dst_path))

    return {
        "split": split,
        "src": str(src_path),
        "dst": str(dst_path),
        "n_all": sum(counts_all.values()),
        "n_kept": len(kept),
        "mode3_all": counts_all,
        "mode3_kept": counts_kept,
    }


def main() -> int:
    print("=" * 80)
    print("Filter Z-dominant dataset")
    print("=" * 80)
    print(f"SRC_DIR={SRC_DIR}")
    print(f"DST_DIR={DST_DIR}")
    print(f"SELECT_MODES(one-based)={[i + 1 for i in SELECT_MODES]}")
    print(f"MIN_Z_RATIO={MIN_Z_RATIO}")

    summaries = [filter_split(split) for split in SPLITS]
    print("\nSummary:")
    for s in summaries:
        print(
            f"{s['split']}: kept {s['n_kept']}/{s['n_all']} | "
            f"mode3_all={s['mode3_all']} | mode3_kept={s['mode3_kept']} -> {s['dst']}"
        )

    if any(int(s["n_kept"]) == 0 for s in summaries):
        raise RuntimeError("At least one split has zero kept samples; loosen SELECT_MODES or MIN_Z_RATIO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
