"""Transolver-Modal HDF5 数据集。

本 Dataset 同时兼容两类 HDF5：
1. 主分支 CNN 生成器 ansys/generate_3d_test.py 写出的 data_2/*.h5；
2. trainsolver 分支扩展生成器写出的 transolver_point_features / element_node_indices 格式。

主分支 CNN 数据不是图像本身，而是 HDF5 中已经包含完整节点信息：
    points, point_features, modal_omega, modal_zeta, modal_phi=[N,K,3],
    modal_phi_exc=[K,3], frequencies, point_frf
因此可以准确用于 Transolver-Modal。关键兼容处理：
    - modal_phi 若为 [N,K,3]，直接作为 modal_phi_xyz；
    - 若没有 excitation_index，则用 modal_phi_exc 与 modal_phi_xyz 精确匹配反推激励节点；
    - 7 维 CNN point_features 会映射为 Transolver 兼容特征索引。
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_FEATURE_NAMES = [
    'x_norm', 'y_norm', 'z_norm',
    'E_over_E0', 'rho_over_rho0', 'PRXY',
    'pocket_active_flag', 'pocket_bottom_flag', 'cutting_band_flag',
    'pocket_id_norm', 'pocket_depth_frac', 'remaining_thickness_ratio',
    'distance_to_pocket_edge_norm',
    'fixture_corner_flag', 'fixture_side_flag',
    'log10_Kx', 'log10_Ky', 'log10_Kz',
    'log10_Cx', 'log10_Cy', 'log10_Cz',
    'distance_to_excitation_norm', 'excitation_flag',
    'free_surface_flag', 'top_surface_flag', 'workpiece_bottom_flag',
    'external_side_surface_flag', 'pocket_sidewall_flag',
]

_DIR_MAP = {'X': 0, 'Y': 1, 'Z': 2}


def _read_attr_str(h5: h5py.File, key: str, default: str = '') -> str:
    raw = h5.attrs.get(key, default)
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return str(raw)


def _read_attr_int(h5: h5py.File, key: str, default: int = 0) -> int:
    val = h5.attrs.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _elements_to_edges(element_node_indices: np.ndarray) -> torch.Tensor:
    if element_node_indices is None or element_node_indices.size == 0:
        return torch.empty(2, 0, dtype=torch.long)
    edges = set()
    for elem in element_node_indices:
        nodes = [int(n) for n in elem if int(n) >= 0]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if a == b:
                    continue
                edges.add((a, b))
                edges.add((b, a))
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def _cnn7_to_transolver_features(points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
    """把主分支 CNN 的 7 维 point_features 映射到 Transolver 兼容索引。

    CNN 7维定义：
        [E_ratio, PRXY, rho_ratio, is_fixed, log10K, log10C, Z/H]
    Transolver 兼容索引保持：
        3:E, 4:rho, 5:PRXY, 15..17:logK, 18..20:logC。
    """
    n = points.shape[0]
    out = torch.zeros(n, len(DEFAULT_FEATURE_NAMES), dtype=torch.float32)
    out[:, 0] = points[:, 0] / 0.160 * 2.0 - 1.0
    out[:, 1] = points[:, 1] / 0.060 * 2.0 - 1.0
    out[:, 2] = points[:, 2] / 0.010 * 2.0 - 1.0

    if point_features is None or point_features.numel() == 0:
        out[:, 3] = 1.0
        out[:, 4] = 1.0
        out[:, 5] = 0.33
        out[:, 11] = points[:, 2] / 0.010
        return out

    pf = point_features.float()
    if pf.ndim == 1:
        pf = pf.unsqueeze(0).expand(n, -1)
    if pf.shape[1] < 7:
        pad = torch.zeros(n, 7 - pf.shape[1], dtype=pf.dtype)
        pf = torch.cat([pf, pad], dim=-1)

    e_ratio = pf[:, 0]
    prxy = pf[:, 1]
    rho_ratio = pf[:, 2]
    is_fixed = pf[:, 3]
    logk = pf[:, 4]
    logc = pf[:, 5]
    z_h = pf[:, 6]

    out[:, 3] = e_ratio
    out[:, 4] = rho_ratio
    out[:, 5] = prxy
    out[:, 11] = z_h
    out[:, 13] = (is_fixed > 0.75).float()
    out[:, 14] = ((is_fixed > 0.25) & (is_fixed <= 0.75)).float()
    out[:, 15] = logk
    out[:, 16] = logk
    out[:, 17] = logk
    out[:, 18] = logc
    out[:, 19] = logc
    out[:, 20] = logc
    out[:, 23] = 1.0
    out[:, 24] = (points[:, 2] > 0.0095).float()
    out[:, 25] = (points[:, 2] < 0.0005).float()
    return out


def _infer_excitation_index(modal_phi_xyz: torch.Tensor, modal_phi_exc: torch.Tensor) -> int:
    """主分支 CNN 数据没有保存 excitation_index，但保存了 modal_phi_exc。

    生成器中 modal_phi_exc = phi_3d_safe[exc_idx,:,:]，因此用 [K,3] 振型签名
    与每个节点的 modal_phi_xyz 精确匹配即可恢复 exc_idx。
    """
    if modal_phi_exc is None:
        return 0
    target = modal_phi_exc.view(1, *modal_phi_exc.shape)
    err = torch.sum((modal_phi_xyz - target) ** 2, dim=(1, 2))
    return int(torch.argmin(err).item())


class TransolverModalDataset(Dataset):
    def __init__(self,
                 data_paths: Sequence[str],
                 data_dir: str = '.',
                 use_edges: bool = True,
                 require_frf: bool = True,
                 min_k2_k3_gap_hz: float = 200.0):
        self.data_dir = data_dir
        self.use_edges = use_edges
        self.require_frf = require_frf
        self.min_k2_k3_gap_hz = min_k2_k3_gap_hz
        self.samples: List[Tuple[str, str]] = []
        self.feature_names: List[str] = DEFAULT_FEATURE_NAMES
        self._file_attrs: Dict[str, object] = {}
        self.ram_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._load_index([os.path.join(data_dir, p) for p in data_paths])

    def _load_index(self, paths: Iterable[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with h5py.File(path, 'r') as h5:
                for attr_key in ['response_direction', 'force_direction', 'response_dir_index', 'force_dir_index']:
                    val = h5.attrs.get(attr_key)
                    if val is not None:
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        self._file_attrs[attr_key] = val
                keys = sorted([k for k in h5.keys() if k.startswith('sample_')], key=lambda k: int(k.split('_')[-1]))
                filtered = 0
                for key in keys:
                    if self.min_k2_k3_gap_hz > 0 and 'modal_omega' in h5[key]:
                        omega = h5[key]['modal_omega'][:]
                        if len(omega) >= 3:
                            gap_hz = omega[2] / (2 * np.pi) - omega[1] / (2 * np.pi)
                            if gap_hz < self.min_k2_k3_gap_hz:
                                filtered += 1
                                continue
                    self.samples.append((path, key))
                if filtered > 0:
                    print(f'[数据过滤] {os.path.basename(path)}: 剔除 {filtered} 个 2-3 阶间隔过小样本')
        if not self.samples:
            raise RuntimeError('HDF5 文件中没有可用 sample_* 数据。')

    @property
    def response_direction(self) -> str:
        return str(self._file_attrs.get('response_direction', 'Z'))

    @property
    def force_direction(self) -> str:
        return str(self._file_attrs.get('force_direction', 'Z'))

    @property
    def response_dir_index(self) -> int:
        val = self._file_attrs.get('response_dir_index')
        if val is not None:
            return int(val)
        return _DIR_MAP.get(self.response_direction, 2)

    @property
    def force_dir_index(self) -> int:
        val = self._file_attrs.get('force_dir_index')
        if val is not None:
            return int(val)
        return _DIR_MAP.get(self.force_direction, 2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index in self.ram_cache:
            return self.ram_cache[index]

        path, group_name = self.samples[index]
        with h5py.File(path, 'r') as h5:
            group = h5[group_name]
            points = torch.from_numpy(group['points'][:]).float()

            raw_pf = None
            if 'transolver_point_features' in group:
                node_features = torch.from_numpy(group['transolver_point_features'][:]).float()
            elif 'point_features' in group:
                raw_pf = torch.from_numpy(group['point_features'][:]).float()
                node_features = _cnn7_to_transolver_features(points, raw_pf)
            else:
                node_features = _cnn7_to_transolver_features(points, None)

            modal_omega = torch.from_numpy(group['modal_omega'][:]).float()
            modal_zeta = torch.from_numpy(group['modal_zeta'][:]).float()

            if 'modal_phi_xyz' in group:
                modal_phi_xyz = torch.from_numpy(group['modal_phi_xyz'][:]).float()
            else:
                phi_raw = torch.from_numpy(group['modal_phi'][:]).float()
                if phi_raw.ndim == 3 and phi_raw.shape[-1] == 3:
                    modal_phi_xyz = phi_raw
                else:
                    modal_phi_xyz = torch.zeros(phi_raw.shape[0], phi_raw.shape[1], 3)
                    modal_phi_xyz[..., 2] = phi_raw

            if 'modal_phi_exc' in group:
                phi_exc_raw = torch.from_numpy(group['modal_phi_exc'][:]).float()
                if phi_exc_raw.ndim == 2 and phi_exc_raw.shape[-1] == 3:
                    modal_phi_exc = phi_exc_raw
                else:
                    modal_phi_exc = torch.zeros(modal_phi_xyz.shape[1], 3)
                    modal_phi_exc[:, 2] = phi_exc_raw.reshape(-1)
            else:
                modal_phi_exc = None

            if 'excitation_index' in group:
                excitation_index = int(np.asarray(group['excitation_index']))
            else:
                excitation_index = _infer_excitation_index(modal_phi_xyz, modal_phi_exc)

            frequencies = torch.from_numpy(group['frequencies'][:]).float()
            if self.require_frf and 'point_frf' in group:
                point_frf = torch.from_numpy(group['point_frf'][:]).float()
            else:
                point_frf = torch.zeros(points.shape[0], frequencies.shape[0], 2)

            if 'boundary_c_xyz' in group:
                boundary_c_xyz = torch.from_numpy(group['boundary_c_xyz'][:]).float()
            else:
                boundary_c_xyz = torch.zeros(points.shape[0], 3)
            if 'boundary_k_xyz' in group:
                boundary_k_xyz = torch.from_numpy(group['boundary_k_xyz'][:]).float()
            else:
                boundary_k_xyz = torch.zeros(points.shape[0], 3)

            if 'element_node_indices' in group and self.use_edges:
                elem_np = group['element_node_indices'][:]
                edge_index = _elements_to_edges(elem_np)
                element_node_indices = torch.from_numpy(elem_np).long()
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                element_node_indices = torch.empty(0, 0, dtype=torch.long)

        resp_idx = self.response_dir_index
        force_idx = self.force_dir_index
        modal_phi_response = modal_phi_xyz[..., resp_idx]
        modal_phi_force = modal_phi_xyz[..., force_idx]
        if modal_phi_exc is None:
            modal_phi_exc = modal_phi_xyz[excitation_index]

        result = {
            'points': points,
            'node_features': node_features,
            'edge_index': edge_index,
            'element_node_indices': element_node_indices,
            'boundary_c_xyz': boundary_c_xyz,
            'boundary_k_xyz': boundary_k_xyz,
            'excitation_index': torch.tensor(excitation_index, dtype=torch.long),
            'modal_omega': modal_omega,
            'modal_zeta': modal_zeta,
            'modal_phi_xyz': modal_phi_xyz,
            'modal_phi_response': modal_phi_response,
            'modal_phi_force': modal_phi_force,
            'modal_phi_exc': modal_phi_exc,
            'modal_phi_z': modal_phi_xyz[..., 2],
            'response_dir_index': torch.tensor(resp_idx, dtype=torch.long),
            'force_dir_index': torch.tensor(force_idx, dtype=torch.long),
            'frequencies': frequencies,
            'point_frf': point_frf,
            'sample_path': path,
            'sample_group': group_name,
        }
        self.ram_cache[index] = result
        return result


GeometricHDF5Dataset = TransolverModalDataset


def collate_mesh_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    node_offsets = []
    running = 0
    for item in batch:
        node_offsets.append(running)
        running += item['points'].shape[0]

    points = torch.cat([item['points'] for item in batch], dim=0)
    node_features = torch.cat([item['node_features'] for item in batch], dim=0)
    boundary_c_xyz = torch.cat([item['boundary_c_xyz'] for item in batch], dim=0)
    boundary_k_xyz = torch.cat([item['boundary_k_xyz'] for item in batch], dim=0)
    modal_phi_xyz = torch.cat([item['modal_phi_xyz'] for item in batch], dim=0)
    modal_phi_response = torch.cat([item['modal_phi_response'] for item in batch], dim=0)
    modal_phi_force = torch.cat([item['modal_phi_force'] for item in batch], dim=0)
    point_frf = torch.cat([item['point_frf'] for item in batch], dim=0)

    batch_index = torch.cat([
        torch.full((item['points'].shape[0],), i, dtype=torch.long)
        for i, item in enumerate(batch)
    ], dim=0)

    edge_parts, elem_parts = [], []
    for item, offset in zip(batch, node_offsets):
        if item['edge_index'].numel() > 0:
            edge_parts.append(item['edge_index'] + offset)
        elem = item['element_node_indices']
        if elem.numel() > 0:
            e = elem.clone()
            e[e >= 0] += offset
            elem_parts.append(e)
    edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.empty(2, 0, dtype=torch.long)
    element_node_indices = torch.cat(elem_parts, dim=0) if elem_parts else torch.empty(0, 0, dtype=torch.long)

    frequencies = torch.stack([item['frequencies'] for item in batch], dim=0)
    modal_omega = torch.stack([item['modal_omega'] for item in batch], dim=0)
    modal_zeta = torch.stack([item['modal_zeta'] for item in batch], dim=0)
    modal_phi_exc = torch.stack([item['modal_phi_exc'] for item in batch], dim=0)
    excitation_index = torch.stack([
        item['excitation_index'] + offset for item, offset in zip(batch, node_offsets)
    ], dim=0)
    response_dir_index = torch.stack([item['response_dir_index'] for item in batch], dim=0)
    force_dir_index = torch.stack([item['force_dir_index'] for item in batch], dim=0)

    return {
        'points': points,
        'node_features': node_features,
        'edge_index': edge_index,
        'element_node_indices': element_node_indices,
        'batch': batch_index,
        'boundary_c_xyz': boundary_c_xyz,
        'boundary_k_xyz': boundary_k_xyz,
        'modal_omega': modal_omega,
        'modal_zeta': modal_zeta,
        'modal_phi_xyz': modal_phi_xyz,
        'modal_phi_response': modal_phi_response,
        'modal_phi_force': modal_phi_force,
        'modal_phi_exc': modal_phi_exc,
        'modal_phi_z': modal_phi_xyz[..., 2],
        'response_dir_index': response_dir_index,
        'force_dir_index': force_dir_index,
        'frequencies': frequencies,
        'point_frf': point_frf,
        'excitation_index': excitation_index,
        'node_counts': [item['points'].shape[0] for item in batch],
        'num_graphs': len(batch),
        'sample_path': [item['sample_path'] for item in batch],
        'sample_group': [item['sample_group'] for item in batch],
    }


collate_geometry_batch = collate_mesh_batch
