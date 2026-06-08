"""
dataset.py — MeshGraphNet/GNN 数据集。

旧版会把 3D 节点投影成 [C, 60, 160] 的 2.5D 图像；这不再适用于
GNN / MeshGraphNet。新版直接返回有限元节点图：

    points, node_features, edge_index, edge_attr, batch

HDF5 推荐格式：
    /sample_i/points          (N, 3)      节点坐标, 单位 m
    /sample_i/point_features  (N, F)      节点物理特征
    /sample_i/edge_index      (2, E)      可选, FE 单元拓扑边
    /sample_i/point_frf       (N, T, 2)   FRF [Re, Im]
    /sample_i/frequencies     (T,)        频率 Hz
    /sample_i/modal_omega     (K,)        圆频率 rad/s
    /sample_i/modal_zeta      (K,)        阻尼比
    /sample_i/modal_phi       (N, K)      Z 向振型
    /sample_i/modal_phi_exc   (K,)        激励点 Z 向振型

如果旧数据里没有 edge_index，Dataset 会用 kNN 图兜底。正式训练建议先运行：
    python ansys/prepare_graph_h5.py
把 edge_index 写入 *_graph.h5，避免每轮动态建图。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


L_BASE = 0.160
W_BASE = 0.060
H_BASE = 0.010
OMEGA_MAX_DEFAULT = 25000.0


class GraphHDF5Dataset(Dataset):
    """per-sample-group HDF5 → variable-size mesh graph batch."""

    def __init__(self, data_paths: Iterable[str], config: Dict, data_dir: str = ".",
                 test: bool = False, normalization: bool = True):
        self.config = config
        self.normalization = normalization
        self.test = test
        self.freq_min = float(config.get("freq_min", 1.0))
        self.freq_max = float(config.get("freq_max", 5000.0))
        self.omega_max = float(config.get("omega_max", OMEGA_MAX_DEFAULT))
        self.knn_k = int(config.get("graph", {}).get("knn_k", 12))
        self._samples: List[tuple[str, str]] = []

        full_paths = [os.path.join(data_dir, p) for p in data_paths]
        self._load_index(full_paths)

    def _load_index(self, full_paths: List[str]) -> None:
        for fp in full_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)
            with h5py.File(fp, "r") as f:
                keys = [k for k in f.keys() if k.startswith("sample_")]
                keys = sorted(keys, key=lambda k: int(k.split("_")[-1]))
                for key in keys:
                    self._samples.append((fp, key))
        if not self._samples:
            raise RuntimeError(f"No per-sample-group data found in: {full_paths}")

    def __len__(self) -> int:
        return len(self._samples)

    def undo_normalize(self, frf: torch.Tensor) -> torch.Tensor:
        # 兼容旧 trainer/evaluate 接口；当前 FRF 默认保持物理量，不再 asinh。
        return frf

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp, grp_name = self._samples[idx]
        with h5py.File(fp, "r") as f:
            grp = f[grp_name]
            points = torch.from_numpy(grp["points"][:]).float()
            point_features = torch.from_numpy(grp["point_features"][:]).float()
            point_frf = torch.from_numpy(grp["point_frf"][:]).float()
            frequencies = torch.from_numpy(grp["frequencies"][:]).float()

            edge_index = None
            if "edge_index" in grp:
                edge_index = torch.from_numpy(grp["edge_index"][:]).long()
                if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                    edge_index = edge_index.t().contiguous()

            out: Dict[str, torch.Tensor] = {}
            for key in ["modal_omega", "modal_zeta", "modal_phi", "modal_phi_exc", "modal_phi_xyz"]:
                if key in grp:
                    out[key] = torch.from_numpy(grp[key][:]).float()

            if "excitation_index" in grp:
                out["excitation_index"] = torch.tensor(int(grp["excitation_index"][()]), dtype=torch.long)

        if edge_index is None or edge_index.numel() == 0:
            edge_index = build_knn_edge_index(points, k=self.knn_k)

        if self.normalization:
            frequencies = (frequencies - self.freq_min) / (self.freq_max - self.freq_min) * 2.0 - 1.0

        if "modal_omega" in out:
            out["modal_omega_phys"] = out["modal_omega"].clone()
            out["modal_omega_norm"] = out.pop("modal_omega") / self.omega_max

        coords_norm = normalize_points(points)
        node_features = torch.cat([coords_norm, sanitize_features(point_features)], dim=-1)
        edge_attr = build_edge_attr(points, edge_index)

        result: Dict[str, torch.Tensor] = {
            "points": points,
            "query_coords": coords_norm,   # 兼容旧脚本命名；现在是 3D normalized coords
            "point_features": point_features,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "point_frf": point_frf,
            "frequencies": frequencies,
        }
        result.update(out)
        return result


# 向后兼容旧入口名，但语义已改为图数据。
GeometricHDF5Dataset = GraphHDF5Dataset


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    return points / scale * 2.0 - 1.0


def sanitize_features(features: torch.Tensor) -> torch.Tensor:
    """把旧 point_features 中的缺省 logK/logC 等安全化。"""
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return features


def build_knn_edge_index(points: torch.Tensor, k: int = 12) -> torch.Tensor:
    """无外部依赖的 kNN 兜底建图。

    仅建议用于兼容旧数据；正式训练应把 FE 拓扑 edge_index 写入 HDF5。
    """
    n = int(points.shape[0])
    if n <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    k = max(1, min(k, n - 1))
    with torch.no_grad():
        dist = torch.cdist(points, points)
        nn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]
        src = torch.arange(n, dtype=torch.long).repeat_interleave(k)
        dst = nn_idx.reshape(-1).long()
        edge_index = torch.stack([src, dst], dim=0)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
    return edge_index.contiguous()


def build_edge_attr(points: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(0, 4, dtype=points.dtype)
    src, dst = edge_index
    delta = points[dst] - points[src]
    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    delta_norm = delta / scale
    dist = torch.linalg.norm(delta_norm, dim=-1, keepdim=True)
    return torch.cat([delta_norm, dist], dim=-1).float()


def collate_geometry_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """把多个变节点数样本拼成一个 disjoint graph batch。"""
    node_features, points, query_coords, point_features = [], [], [], []
    edge_indices, edge_attrs = [], []
    point_frf = []
    batch_vec = []
    node_offset = 0

    for i, item in enumerate(batch):
        n_i = int(item["points"].shape[0])
        node_features.append(item["node_features"])
        points.append(item["points"])
        query_coords.append(item["query_coords"])
        point_features.append(item["point_features"])
        point_frf.append(item["point_frf"])
        batch_vec.append(torch.full((n_i,), i, dtype=torch.long))

        ei = item["edge_index"].long() + node_offset
        edge_indices.append(ei)
        edge_attrs.append(item.get("edge_attr", build_edge_attr(item["points"], item["edge_index"])))
        node_offset += n_i

    f_lens = [item["frequencies"].shape[0] for item in batch]
    if all(f == f_lens[0] for f in f_lens):
        frequencies = torch.stack([item["frequencies"] for item in batch])
        frf_out = torch.cat(point_frf, dim=0)
    else:
        frequencies = [item["frequencies"] for item in batch]
        frf_out = point_frf

    out: Dict[str, torch.Tensor] = {
        "node_features": torch.cat(node_features, dim=0),
        "points": torch.cat(points, dim=0),
        "query_coords": torch.cat(query_coords, dim=0),
        "point_features": torch.cat(point_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "frequencies": frequencies,
        "point_frf": frf_out,
    }

    modal = _stack_modal(batch)
    if modal:
        out.update(modal)
    return out


def _stack_modal(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    required = ["modal_omega_norm", "modal_zeta", "modal_phi"]
    if any(key not in batch[0] for key in required):
        return {}

    result: Dict[str, torch.Tensor] = {
        "modal_omega_norm": torch.stack([item["modal_omega_norm"] for item in batch]),
        "modal_zeta": torch.stack([item["modal_zeta"] for item in batch]),
        "modal_phi": torch.cat([item["modal_phi"] for item in batch], dim=0),
    }
    for key in ["modal_phi_exc", "modal_omega_phys"]:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    if "modal_phi_xyz" in batch[0]:
        result["modal_phi_xyz"] = torch.cat([item["modal_phi_xyz"] for item in batch], dim=0)
    if "excitation_index" in batch[0]:
        result["excitation_index"] = torch.stack([item["excitation_index"] for item in batch])
    return result
