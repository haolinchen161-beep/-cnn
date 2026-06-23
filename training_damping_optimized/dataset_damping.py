# -*- coding: utf-8 -*-
"""Optimized HDF5 dataset loader for natural damping ratio training with memory caching."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

L_BASE = 0.160
W_BASE = 0.060

class DampingH5Dataset(Dataset):
    """Dataset for training the optimized modal damping ratio model with memory caching."""

    def __init__(self, h5_path: str | Path, target_modes: int = 3):
        self.h5_path = str(h5_path)
        self.target_modes = int(target_modes)
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
            
        with h5py.File(self.h5_path, "r") as h5:
            self.sample_keys: List[str] = sorted(
                [k for k in h5.keys() if k.startswith("sample_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            if not self.sample_keys:
                raise RuntimeError(f"No sample_* group in HDF5 file: {self.h5_path}")
                
            # Resolve and load precomputed frequency and shape norm priors
            current_dir = Path(__file__).resolve().parent
            priors_file = None
            if "train.h5" in self.h5_path:
                priors_file = current_dir / "train_priors.pt"
            elif "val.h5" in self.h5_path:
                priors_file = current_dir / "val_priors.pt"
            elif "test.h5" in self.h5_path:
                priors_file = current_dir / "test_priors.pt"
            
            priors_dict = {}
            if priors_file and priors_file.exists():
                print(f"Loading precomputed priors from {priors_file}...")
                priors_dict = torch.load(priors_file, map_location="cpu")
            else:
                print(f"Warning: Priors file {priors_file} not found. Fallback to zeros.")

            print(f"Caching features from {self.h5_path} into memory...")
            self.cached_data: List[Dict[str, torch.Tensor]] = []
            
            for key in self.sample_keys:
                g = h5[key]
                pocket = self._read_pocket_features(g)
                clamp = self._read_clamp_features(g)
                e_ratio, rho_ratio = self._read_material_features(g)
                layout_type = self._read_scalar(g, "layout_type", 0.0)
                coverage_code = self._read_scalar(g, "coverage_level_code", 0.0)
                clamp_code = self._read_scalar(g, "clamp_level_code", self._read_scalar(g, "clamp_model_code", 0.0))
                removed_volume_ratio = self._read_scalar(g, "removed_volume_ratio", 0.0)
                grid_jitter = self._read_scalar(g, "grid_jitter", 0.0)
                finished_count = self._read_scalar(g, "finished_count", 0.0)
                current_progress = self._read_scalar(g, "current_progress", 1.0)
                zeta = np.asarray(g["modal_zeta"], dtype=np.float32)[: self.target_modes]

                if zeta.shape[0] != self.target_modes:
                    raise RuntimeError(f"Incomplete modal_zeta for {key}: expected {self.target_modes}, got {zeta.shape[0]}")

                # 1. Normalize clamp features to align with shape predictor
                clamp = clamp.copy()
                clamp[:, 5:8] /= 12.0
                clamp[:, 8:11] /= 8.0

                # 2. Extract pocket and clamp 2D spatial centers
                pocket_centers = np.zeros((7, 2), dtype=np.float32)
                pocket_centers[:, 0] = (pocket[:, 0] + pocket[:, 1]) / 2.0
                pocket_centers[:, 1] = (pocket[:, 2] + pocket[:, 3]) / 2.0

                clamp_centers = np.zeros((7, 2), dtype=np.float32)
                clamp_centers[:, 0] = (clamp[:, 0] + clamp[:, 1]) / 2.0
                clamp_centers[:, 1] = (clamp[:, 2] + clamp[:, 3]) / 2.0

                # Get precomputed priors
                priors = priors_dict.get(key, None)
                if priors is not None:
                    omega_pred = priors["omega_pred"]  # [3]
                    phi_z_norm_pred = priors["phi_z_norm_pred"]  # [3]
                else:
                    omega_pred = np.zeros(3, dtype=np.float32)
                    phi_z_norm_pred = np.zeros(3, dtype=np.float32)

                # 3. Global features: base layout features (7) + predicted physical priors (6) = 13 dimensions
                global_features = np.asarray(
                    [
                        layout_type / 7.0,
                        coverage_code / 2.0,
                        clamp_code / 2.0,
                        removed_volume_ratio,
                        grid_jitter,
                        finished_count / 7.0,
                        current_progress,
                    ],
                    dtype=np.float32,
                )
                global_features = np.concatenate([global_features, omega_pred, phi_z_norm_pred]).astype(np.float32)

                # 4. Material physical scaling factor for boundary damping
                log_material_scale_damping = -0.5 * np.log(e_ratio * rho_ratio)

                self.cached_data.append({
                    "pocket_features": torch.from_numpy(pocket),
                    "pocket_centers": torch.from_numpy(pocket_centers),
                    "clamp_features": torch.from_numpy(clamp),
                    "clamp_centers": torch.from_numpy(clamp_centers),
                    "global_features": torch.from_numpy(global_features),
                    "zeta": torch.from_numpy(zeta),
                    "log_material_scale_damping": torch.tensor(log_material_scale_damping, dtype=torch.float32),
                })
            print(f"Successfully cached {len(self.cached_data)} samples into memory.")

    def __len__(self) -> int:
        return len(self.sample_keys)

    @staticmethod
    def _read_scalar(group: h5py.Group, key: str, default: float = 0.0) -> float:
        if key not in group:
            return float(default)
        arr = np.asarray(group[key])
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])

    @staticmethod
    def _build_pocket_features_from_legacy(group: h5py.Group) -> np.ndarray:
        feats = np.zeros((7, 8), dtype=np.float32)
        if "cell_bounds" not in group or "cell_depth_ratio" not in group:
            return feats
        cell_bounds = np.asarray(group["cell_bounds"], dtype=np.float32)
        cell_depth_ratio = np.asarray(group["cell_depth_ratio"], dtype=np.float32)
        n = min(7, cell_bounds.shape[0])
        for i in range(n):
            bounds = cell_bounds[i]
            if np.any(np.isnan(bounds)):
                feats[i, 4] = 0.0
                continue
            depth = float(cell_depth_ratio[i]) if i < len(cell_depth_ratio) and np.isfinite(cell_depth_ratio[i]) else 0.0
            x0, x1, y0, y1 = bounds
            feats[i, 0] = x0 / L_BASE
            feats[i, 1] = x1 / L_BASE
            feats[i, 2] = y0 / W_BASE
            feats[i, 3] = y1 / W_BASE
            feats[i, 4] = 1.0
            feats[i, 5] = 1.0 if depth > 1e-6 else 0.0
            feats[i, 6] = depth
            feats[i, 7] = 1.0 - depth
        return feats

    @staticmethod
    def _read_pocket_features(group: h5py.Group) -> np.ndarray:
        if "pocket_features" in group:
            arr = np.asarray(group["pocket_features"], dtype=np.float32)
            if arr.shape == (7, 8):
                return arr
            fixed = np.zeros((7, 8), dtype=np.float32)
            fixed[: min(7, arr.shape[0]), : min(8, arr.shape[1])] = arr[: min(7, arr.shape[0]), : min(8, arr.shape[1])]
            return fixed
        return DampingH5Dataset._build_pocket_features_from_legacy(group)

    @staticmethod
    def _read_clamp_features(group: h5py.Group) -> np.ndarray:
        if "clamp_features" in group:
            arr = np.asarray(group["clamp_features"], dtype=np.float32)
            if arr.shape == (7, 11):
                return arr
            fixed = np.zeros((7, 11), dtype=np.float32)
            fixed[: min(7, arr.shape[0]), : min(11, arr.shape[1])] = arr[: min(7, arr.shape[0]), : min(11, arr.shape[1])]
            return fixed
        return np.zeros((7, 11), dtype=np.float32)

    @staticmethod
    def _read_material_features(group: h5py.Group) -> Tuple[float, float]:
        if "point_features" not in group:
            return 1.0, 1.0
        pf = np.asarray(group["point_features"], dtype=np.float32)
        if pf.ndim != 2 or pf.shape[0] == 0 or pf.shape[1] < 3:
            return 1.0, 1.0
        return float(pf[0, 0]), float(pf[0, 2])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Shallow copy to avoid mutating cache but still fast O(1) memory retrieve
        item = self.cached_data[int(index)].copy()
        item["sample_index"] = torch.tensor(index, dtype=torch.long)
        return item
