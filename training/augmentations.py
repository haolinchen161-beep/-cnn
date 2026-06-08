"""
augmentations.py — MeshGraphNet 25D 图数据增强。

当前主数据结构来自 data.dataset.collate_geometry_batch：
    node_features, points, edge_index, edge_attr, batch, point_frf, frequencies

25D node_features 布局：
    0:3    normalized xyz
    3:10   point_features
    10:13  spring_k_xyz log normalized
    13:16  spring_c_xyz log normalized
    16:21  node_type one-hot
    21     pocket_bottom_mask
    22     cut_region_mask
    23     excitation_flag
    24     normalized distance to excitation

默认训练脚本未启用增强；该模块仅作为可选工具。
"""

from __future__ import annotations

import torch


L_BASE, W_BASE, H_BASE = 0.160, 0.060, 0.010


class GraphBatchAugmenter:
    """对 GNN 图 batch 做保守增强。"""

    def __init__(self,
                 coord_noise: float = 1e-4,
                 material_noise: float = 0.002,
                 log_spring_noise: float = 0.003,
                 freq_subsample: int | None = None,
                 enabled: bool = True):
        self.coord_noise = coord_noise
        self.material_noise = material_noise
        self.log_spring_noise = log_spring_noise
        self.freq_subsample = freq_subsample
        self.enabled = enabled
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, batch):
        if not self.enabled or not self.training:
            return batch
        batch = dict(batch)
        batch = self._augment_coords(batch)
        batch = self._augment_features(batch)
        batch = self._augment_frequencies(batch)
        return batch

    def _augment_coords(self, batch):
        if 'points' not in batch or 'node_features' not in batch:
            return batch
        points = batch['points']
        noise = torch.randn_like(points) * self.coord_noise
        if 'point_features' in batch and batch['point_features'].shape[-1] >= 4:
            is_bc = batch['point_features'][:, 3] > 0
            noise[is_bc] *= 0.1
        batch['points'] = points + noise
        scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
        batch['node_features'] = batch['node_features'].clone()
        batch['node_features'][:, :3] = batch['points'] / scale * 2.0 - 1.0
        return batch

    def _augment_features(self, batch):
        if 'node_features' not in batch:
            return batch
        nf = batch['node_features'].clone()
        if nf.shape[-1] < 25:
            batch['node_features'] = nf
            return batch

        # 仅扰动连续物理量，不扰动 one-hot/mask/excitation。
        # point_features: E ratio, rho ratio, logK/logC/ZH 等。
        cont_idx = [3, 5, 7, 8, 9]
        nf[:, cont_idx] += torch.randn_like(nf[:, cont_idx]) * self.material_noise
        # normalized spring log features。
        nf[:, 10:16] += torch.randn_like(nf[:, 10:16]) * self.log_spring_noise
        batch['node_features'] = nf
        return batch

    def _augment_frequencies(self, batch):
        if self.freq_subsample is None or self.freq_subsample <= 0:
            return batch
        if 'frequencies' not in batch or 'point_frf' not in batch:
            return batch
        freqs = batch['frequencies']
        frf = batch['point_frf']
        if not torch.is_tensor(freqs) or freqs.ndim != 2:
            return batch
        f_total = freqs.shape[-1]
        if self.freq_subsample >= f_total:
            return batch
        idx = torch.sort(torch.randperm(f_total, device=freqs.device)[:self.freq_subsample]).values
        batch['frequencies'] = freqs[:, idx]
        batch['point_frf'] = frf[:, idx, :]
        return batch


# 向后兼容旧导入名。
GeometryAugmenter = GraphBatchAugmenter
