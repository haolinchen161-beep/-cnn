# -*- coding: utf-8 -*-
"""HDF5 固有频率数据集读取。

约定：
- 输入优先使用生成程序保存的 pocket_features、clamp_features。
- 如果旧数据没有 pocket_features，则由 cell_bounds + cell_depth_ratio 临时构造。
- 固有频率目标使用 modal_omega[:target_modes]，单位 rad/s。
- 激励点不参与固有频率预测，因为固有频率是结构自身全局属性。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


L_BASE = 0.160
W_BASE = 0.060


class FrequencyH5Dataset(Dataset):
    """读取 train.h5 / val.h5 / test.h5 中的全局频率训练样本。"""

    def __init__(self, h5_path: str | Path, target_modes: int = 3):
        self.h5_path = str(h5_path)
        self.target_modes = int(target_modes)
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as h5:
            self.sample_keys: List[str] = sorted(
                [k for k in h5.keys() if k.startswith("sample_")],
                key=lambda x: int(x.split("_")[-1]),
            )
        if not self.sample_keys:
            raise RuntimeError(f"HDF5 文件中没有 sample_* group: {self.h5_path}")

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
        """兼容旧数据：由 cell_bounds 和 cell_depth_ratio 构造 [7,8] pocket token。"""
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
        return FrequencyH5Dataset._build_pocket_features_from_legacy(group)

    @staticmethod
    def _read_clamp_features(group: h5py.Group) -> np.ndarray:
        if "clamp_features" in group:
            arr = np.asarray(group["clamp_features"], dtype=np.float32)
            if arr.shape == (7, 11):
                return arr
            fixed = np.zeros((7, 11), dtype=np.float32)
            fixed[: min(7, arr.shape[0]), : min(11, arr.shape[1])] = arr[: min(7, arr.shape[0]), : min(11, arr.shape[1])]
            return fixed
        # 旧数据没有 clamp_features 时，用 0 占位；此时模型仍可用 pocket/material 预测频率 baseline。
        return np.zeros((7, 11), dtype=np.float32)

    @staticmethod
    def _read_material_features(group: h5py.Group) -> tuple[float, float]:
        """从 point_features 读取 E/E0 和 rho/rho0。"""
        if "point_features" not in group:
            return 1.0, 1.0
        pf = np.asarray(group["point_features"], dtype=np.float32)
        if pf.ndim != 2 or pf.shape[0] == 0 or pf.shape[1] < 3:
            return 1.0, 1.0
        return float(pf[0, 0]), float(pf[0, 2])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        key = self.sample_keys[int(index)]
        with h5py.File(self.h5_path, "r") as h5:
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
            omega = np.asarray(g["modal_omega"], dtype=np.float32)[: self.target_modes]

        if omega.shape[0] != self.target_modes:
            raise RuntimeError(f"{self.h5_path}/{key}: modal_omega 阶数不足，得到 {omega.shape[0]}, 需要 {self.target_modes}")

        # Normalize clamp boundary features
        clamp_norm = clamp.copy()
        clamp_norm[:, 5:8] /= 12.0
        clamp_norm[:, 8:11] /= 8.0

        # 全局标量特征
        global_features = np.asarray(
            [
                e_ratio,
                rho_ratio,
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

        return {
            "pocket_features": torch.from_numpy(pocket),
            "clamp_features": torch.from_numpy(clamp_norm),
            "global_features": torch.from_numpy(global_features),
            "omega": torch.from_numpy(omega),
        }
