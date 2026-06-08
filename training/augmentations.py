"""
augmentations.py — MeshGraphNet 图数据增强。

当前主数据结构是 collate_geometry_batch 返回的 disjoint graph batch：
    node_features, points, edge_index, edge_attr, batch, point_frf, frequencies

默认训练脚本暂未启用增强。该模块保留为可选工具，避免旧 geometry 容器接口残留。
"""

from __future__ import annotations

import torch


class GraphBatchAugmenter:
    """对 GNN 图 batch 做轻量增强。

    注意：节点 dropout 会改变 edge_index/edge_attr/point_frf/modal_phi，对物理一致性要求较高。
    因此当前默认只做坐标和节点特征微扰，不做节点删除。
    """

    def __init__(self,
                 coord_noise: float = 1e-4,
                 feat_noise_scale: float = 0.005,
                 freq_subsample: int | None = None,
                 enabled: bool = True):
        self.coord_noise = coord_noise
        self.feat_noise_scale = feat_noise_scale
        self.freq_subsample = freq_subsample
        self.enabled = enabled
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, batch):
        if not self.enabled or not self.training:
            return batch
        batch = self._augment_coords(batch)
        batch = self._augment_node_features(batch)
        batch = self._augment_frequencies(batch)
        return batch

    def _augment_coords(self, batch):
        if 'points' not in batch:
            return batch
        points = batch['points']
        noise = torch.randn_like(points) * self.coord_noise
        if 'point_features' in batch and batch['point_features'].shape[-1] >= 4:
            # 保持装夹节点几何更稳定。
            is_bc = batch['point_features'][:, 3] > 0
            noise[is_bc] *= 0.1
        batch['points'] = points + noise
        # node_features 前三维是归一化坐标，保持同步近似更新。
        if 'node_features' in batch and batch['node_features'].shape[-1] >= 3:
            scale = torch.tensor([0.160, 0.060, 0.010], dtype=points.dtype, device=points.device)
            batch['node_features'][:, :3] = batch['points'] / scale * 2.0 - 1.0
        return batch

    def _augment_node_features(self, batch):
        if 'node_features' not in batch:
            return batch
        nf = batch['node_features']
        noise = torch.zeros_like(nf)
        # 前3维是坐标，已由 _augment_coords 处理；后7维来自 point_features。
        if nf.shape[-1] >= 10:
            scales = torch.tensor([
                0.0, 0.0, 0.0,   # xyz normalized
                0.005, 0.0, 0.003, 0.0, 0.05, 0.05, 0.005,
            ], dtype=nf.dtype, device=nf.device)
            noise = torch.randn_like(nf) * scales * self.feat_noise_scale
        batch['node_features'] = nf + noise
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
