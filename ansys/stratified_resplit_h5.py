"""Stratified train/val/test split for generated MeshGraphNet HDF5 files.

The generator intentionally does not store derived labels such as mode_type.
This utility derives the third-mode dominant direction from modal_phi_xyz only
for splitting, then rewrites train.h5 / val.h5 / test.h5 with a more balanced
mode-3 X/Y/Z distribution.

Run after ansys/generate_3d_test.py:
    python ansys/stratified_resplit_h5.py --in-place
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import h5py
import numpy as np


Ref = Tuple[str, str, int]


def mode3_direction(h5_group) -> int:
    """Return 0/1/2 for X/Y/Z dominant direction of mode 3."""
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

    energy = np.sum(phi[:, 2, :] ** 2, axis=0)
    return int(np.argmax(energy))


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

    # Enforce exact total counts.  This may slightly perturb class balance, but
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
    names = {0: "X", 1: "Y", 2: "Z"}
    counts = {"X": 0, "Y": 0, "Z": 0}
    for _, _, cls in refs:
        counts[names[cls]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-place", action="store_true", help="overwrite train.h5/val.h5/test.h5 after making .sequential.bak backups")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
