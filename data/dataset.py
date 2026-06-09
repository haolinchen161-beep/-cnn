"""Transolver 网格 HDF5 数据集。

本加载器消费 ``ansys/generate_3d_test.py`` 在 ``transolver-modal-dataset``
分支上生成的 ANSYS 逐样本 HDF5 文件。不将 3D 网格投影到 2.5D 图像，
每个样本保持为非结构化节点云，附带可选的单元导出边。

每样本主要张量：
    points                     (N, 3)     节点坐标
    transolver_point_features   (N, C)     Transolver 节点特征
    boundary_c_xyz              (N, 3)     边界阻尼系数
    modal_omega                 (K,)       固有圆频率
    modal_zeta                  (K,)       阻尼比
    modal_phi_xyz               (N, K, 3)  三向模态振型
    frequencies                 (F,)       频率 Hz
    point_frf                   (N, F, 2)  FRF [实部, 虚部]
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


def _read_feature_names(h5: h5py.File) -> List[str]:
    raw = h5.attrs.get('transolver_feature_names', '')
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    names = [x for x in str(raw).split(',') if x]
    return names or DEFAULT_FEATURE_NAMES


def _elements_to_edges(element_node_indices: np.ndarray) -> torch.Tensor:
    """将填充的单元连接关系转换为无向 COO 边。"""
    if element_node_indices is None or element_node_indices.size == 0:
        return torch.empty(2, 0, dtype=torch.long)
    edges = set()
    for elem in element_node_indices:
        nodes = [int(n) for n in elem if int(n) >= 0]
        m = len(nodes)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                if a == b:
                    continue
                edges.add((a, b))
                edges.add((b, a))
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    edge_arr = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return edge_arr


def _read_attr_str(h5: h5py.File, key: str, default: str = "") -> str:
    """安全读取 HDF5 字符串属性。"""
    raw = h5.attrs.get(key, default)
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return str(raw)


def _read_attr_int(h5: h5py.File, key: str, default: int = 0) -> int:
    """安全读取 HDF5 整数属性。"""
    val = h5.attrs.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class TransolverModalDataset(Dataset):
    """逐样本 HDF5 数据集，用于 Transolver 模态-FRF 训练。

    自动识别 boolean 完成态数据和 ekill 过程数据。
    支持从 HDF5 attrs 中读取方向配置。
    """

    def __init__(self,
                 data_paths: Sequence[str],
                 data_dir: str = '.',
                 use_edges: bool = True,
                 require_frf: bool = True):
        self.data_dir = data_dir
        self.use_edges = use_edges
        self.require_frf = require_frf
        self.samples: List[Tuple[str, str]] = []
        self.feature_names: List[str] = DEFAULT_FEATURE_NAMES

        # 文件级属性缓存（取最后一个文件的属性）
        self._file_attrs: Dict[str, object] = {}
        self.ram_cache: Dict[int, Dict[str, torch.Tensor]] = {}  # 内存缓存，第 2 epoch 起极速
        self._load_index([os.path.join(data_dir, p) for p in data_paths])

    def _load_index(self, paths: Iterable[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with h5py.File(path, 'r') as h5:
                self.feature_names = _read_feature_names(h5)

                # 缓存文件级属性
                for attr_key in ['response_direction', 'force_direction',
                                 'response_dir_index', 'force_dir_index',
                                 'frequency_grid_mode']:
                    val = h5.attrs.get(attr_key)
                    if val is not None:
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        self._file_attrs[attr_key] = val

                keys = [k for k in h5.keys() if k.startswith('sample_')]
                keys = sorted(keys, key=lambda k: int(k.split('_')[-1]))
                for key in keys:
                    self.samples.append((path, key))
        if not self.samples:
            raise RuntimeError('HDF5 文件中未找到 sample_* 分组。')

    @property
    def response_direction(self) -> str:
        """文件级响应方向（如 "Y"）。"""
        return str(self._file_attrs.get('response_direction', 'Z'))

    @property
    def force_direction(self) -> str:
        """文件级激励方向（如 "Y"）。"""
        return str(self._file_attrs.get('force_direction', 'Z'))

    @property
    def response_dir_index(self) -> int:
        """文件级响应方向索引。"""
        val = self._file_attrs.get('response_dir_index')
        if val is not None:
            return int(val)
        # 兼容旧文件：从 frf_direction / excitation_direction 推断
        frf_dir = str(self._file_attrs.get('frf_direction', 'Z'))
        return {"X": 0, "Y": 1, "Z": 2}.get(frf_dir, 2)

    @property
    def force_dir_index(self) -> int:
        """文件级激励方向索引。"""
        val = self._file_attrs.get('force_dir_index')
        if val is not None:
            return int(val)
        exc_dir = str(self._file_attrs.get('excitation_direction', 'Z'))
        return {"X": 0, "Y": 1, "Z": 2}.get(exc_dir, 2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # 内存缓存命中 → 直接返回（第 2+ epoch 极速）
        if index in self.ram_cache:
            return self.ram_cache[index]

        path, group_name = self.samples[index]
        with h5py.File(path, 'r') as h5:
            group = h5[group_name]

            # --- 基础几何与特征 ---
            points = torch.from_numpy(group['points'][:]).float()
            if 'transolver_point_features' in group:
                node_features = torch.from_numpy(group['transolver_point_features'][:]).float()
            else:
                node_features = torch.from_numpy(group['point_features'][:]).float()

            # --- 模态参数 ---
            modal_omega = torch.from_numpy(group['modal_omega'][:]).float()
            modal_zeta = torch.from_numpy(group['modal_zeta'][:]).float()

            # --- 三向振型（主目标） ---
            if 'modal_phi_xyz' in group:
                modal_phi_xyz = torch.from_numpy(group['modal_phi_xyz'][:]).float()
            else:
                # 向后兼容：旧文件只有 Z 向振型
                modal_phi_z = torch.from_numpy(group['modal_phi'][:]).float()
                modal_phi_xyz = torch.zeros(modal_phi_z.shape[0], modal_phi_z.shape[1], 3)
                modal_phi_xyz[..., 2] = modal_phi_z

            # --- 方向感知振型投影 ---
            resp_idx = self.response_dir_index
            force_idx = self.force_dir_index
            modal_phi_response = modal_phi_xyz[..., resp_idx]  # (N, K)
            modal_phi_force = modal_phi_xyz[..., force_idx]    # (N, K)

            # --- 频率与 FRF ---
            frequencies = torch.from_numpy(group['frequencies'][:]).float()
            if self.require_frf and 'point_frf' in group:
                point_frf = torch.from_numpy(group['point_frf'][:]).float()
            else:
                point_frf = torch.empty(points.shape[0], frequencies.shape[0], 2)

            # --- 边界条件 ---
            boundary_c_xyz = torch.from_numpy(
                group.get('boundary_c_xyz',
                          np.zeros((points.shape[0], 3), np.float32))[:]
            ).float()
            boundary_k_xyz = torch.from_numpy(
                group.get('boundary_k_xyz',
                          np.zeros((points.shape[0], 3), np.float32))[:]
            ).float()
            fixture_type = torch.from_numpy(
                group.get('fixture_type',
                          np.zeros((points.shape[0],), np.int8))[:]
            ).long()
            surface_flags = torch.from_numpy(
                group.get('surface_flags',
                          np.zeros((points.shape[0], 0), np.float32))[:]
            ).float()

            # --- 激励点 ---
            excitation_index = int(np.asarray(group.get('excitation_index', 0)))

            # --- 刀触点 / 过程字段（带 fallback） ---
            contact_node_index = int(np.asarray(
                group.get('contact_node_index', excitation_index)))
            tool_position = torch.from_numpy(
                group.get('tool_position',
                          points[excitation_index].numpy())[:]
            ).float()
            force_direction_vector = torch.from_numpy(
                group.get('force_direction_vector',
                          np.array([0., 0., 1.], np.float32))[:]
            ).float()
            response_direction_vector = torch.from_numpy(
                group.get('response_direction_vector',
                          np.array([0., 0., 1.], np.float32))[:]
            ).float()
            active_pocket_id = int(np.asarray(group.get('active_pocket_id', -1)))
            process_step = float(np.asarray(group.get('process_step', 1.0)))
            removed_volume_ratio = float(np.asarray(
                group.get('removed_volume_ratio', 0.0)))

            # --- ekill 过程字段（带 fallback） ---
            node_active_flag = group.get('node_active_flag')
            element_active_flag = group.get('element_active_flag')
            removed_element_flag = group.get('removed_element_flag')

            # --- 网格连接关系 ---
            if 'element_node_indices' in group and self.use_edges:
                element_node_indices_np = group['element_node_indices'][:]
                edge_index = _elements_to_edges(element_node_indices_np)
                element_node_indices = torch.from_numpy(element_node_indices_np).long()
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                element_node_indices = torch.empty(0, 0, dtype=torch.long)

        # --- 特征增强（ekill 字段追加） ---
        extra_feats = []
        if node_active_flag is not None:
            naf = torch.from_numpy(node_active_flag[:]).float()
            if naf.dim() == 0 or naf.shape[0] != node_features.shape[0]:
                naf = naf.unsqueeze(-1) if naf.dim() == 1 else naf
            else:
                naf = naf.unsqueeze(-1)
            # 确保形状匹配
            if naf.shape[0] == node_features.shape[0]:
                extra_feats.append(naf)

        if tool_position is not None and points.shape[0] > 0:
            diag = float(np.sqrt(0.160**2 + 0.060**2 + 0.010**2))
            dist_to_tool = torch.norm(points - tool_position.unsqueeze(0), dim=1, keepdim=True)
            dist_to_tool_norm = dist_to_tool / max(diag, 1e-8)
            extra_feats.append(dist_to_tool_norm)

        if extra_feats:
            node_features = torch.cat([node_features] + extra_feats, dim=-1)

        result = {
            'points': points,
            'node_features': node_features,
            'edge_index': edge_index,
            'element_node_indices': element_node_indices,
            'boundary_c_xyz': boundary_c_xyz,
            'boundary_k_xyz': boundary_k_xyz,
            'fixture_type': fixture_type,
            'surface_flags': surface_flags,
            'excitation_index': torch.tensor(excitation_index, dtype=torch.long),
            'modal_omega': modal_omega,
            'modal_zeta': modal_zeta,
            'modal_phi_xyz': modal_phi_xyz,
            # 方向感知字段
            'modal_phi_response': modal_phi_response,
            'modal_phi_force': modal_phi_force,
            'response_dir_index': torch.tensor(resp_idx, dtype=torch.long),
            'force_dir_index': torch.tensor(force_idx, dtype=torch.long),
            'response_direction': self.response_direction,
            'force_direction': self.force_direction,
            # 向后兼容
            'modal_phi_z': modal_phi_xyz[..., 2],
            # FRF
            'frequencies': frequencies,
            'point_frf': point_frf,
            # 刀触点 / 过程字段
            'contact_node_index': torch.tensor(contact_node_index, dtype=torch.long),
            'tool_position': tool_position,
            'force_direction_vector': force_direction_vector,
            'response_direction_vector': response_direction_vector,
            'active_pocket_id': torch.tensor(active_pocket_id, dtype=torch.long),
            'process_step': torch.tensor(process_step, dtype=torch.float32),
            'removed_volume_ratio': torch.tensor(removed_volume_ratio, dtype=torch.float32),
            # 元数据
            'sample_path': path,
            'sample_group': group_name,
        }
        self.ram_cache[index] = result
        return result


# 向后兼容别名
GeometricHDF5Dataset = TransolverModalDataset


def collate_mesh_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """将变长网格拼接为批次：节点拼接 + 边偏移。"""
    node_offsets = []
    running = 0
    for item in batch:
        node_offsets.append(running)
        running += item['points'].shape[0]

    # --- 节点级张量：直接拼接 ---
    points = torch.cat([item['points'] for item in batch], dim=0)
    node_features = torch.cat([item['node_features'] for item in batch], dim=0)
    boundary_c_xyz = torch.cat([item['boundary_c_xyz'] for item in batch], dim=0)
    boundary_k_xyz = torch.cat([item['boundary_k_xyz'] for item in batch], dim=0)
    fixture_type = torch.cat([item['fixture_type'] for item in batch], dim=0)
    modal_phi_xyz = torch.cat([item['modal_phi_xyz'] for item in batch], dim=0)
    point_frf = torch.cat([item['point_frf'] for item in batch], dim=0)

    # --- 方向感知振型 ---
    modal_phi_response = torch.cat([item['modal_phi_response'] for item in batch], dim=0)
    modal_phi_force = torch.cat([item['modal_phi_force'] for item in batch], dim=0)

    # --- 批次索引 ---
    batch_index = torch.cat([
        torch.full((item['points'].shape[0],), i, dtype=torch.long)
        for i, item in enumerate(batch)
    ], dim=0)

    # --- 边与单元（带偏移） ---
    edge_parts = []
    element_parts = []
    for item, offset in zip(batch, node_offsets):
        edge_index = item['edge_index']
        if edge_index.numel() > 0:
            edge_parts.append(edge_index + offset)
        elems = item['element_node_indices']
        if elems.numel() > 0:
            e = elems.clone()
            e[e >= 0] += offset
            element_parts.append(e)
    edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.empty(2, 0, dtype=torch.long)
    element_node_indices = torch.cat(element_parts, dim=0) if element_parts else torch.empty(0, 0, dtype=torch.long)

    # --- 图级张量：stack ---
    frequencies = torch.stack([item['frequencies'] for item in batch], dim=0)
    modal_omega = torch.stack([item['modal_omega'] for item in batch], dim=0)
    modal_zeta = torch.stack([item['modal_zeta'] for item in batch], dim=0)
    excitation_index = torch.stack([
        item['excitation_index'] + offset for item, offset in zip(batch, node_offsets)
    ], dim=0)
    response_dir_index = torch.stack([item['response_dir_index'] for item in batch], dim=0)
    force_dir_index = torch.stack([item['force_dir_index'] for item in batch], dim=0)

    # --- 刀触点 / 过程字段 ---
    contact_node_index = torch.stack([
        item['contact_node_index'] + offset for item, offset in zip(batch, node_offsets)
    ], dim=0)

    result = {
        'points': points,
        'node_features': node_features,
        'edge_index': edge_index,
        'element_node_indices': element_node_indices,
        'batch': batch_index,
        'boundary_c_xyz': boundary_c_xyz,
        'boundary_k_xyz': boundary_k_xyz,
        'fixture_type': fixture_type,
        'modal_omega': modal_omega,
        'modal_zeta': modal_zeta,
        'modal_phi_xyz': modal_phi_xyz,
        # 方向感知
        'modal_phi_response': modal_phi_response,
        'modal_phi_force': modal_phi_force,
        'response_dir_index': response_dir_index,
        'force_dir_index': force_dir_index,
        # 向后兼容
        'modal_phi_z': modal_phi_xyz[..., 2],
        # FRF
        'frequencies': frequencies,
        'point_frf': point_frf,
        # 激励点
        'excitation_index': excitation_index,
        'contact_node_index': contact_node_index,
        # 元数据
        'num_graphs': len(batch),
        'sample_path': [item['sample_path'] for item in batch],
        'sample_group': [item['sample_group'] for item in batch],
    }
    return result


# 向后兼容别名
collate_geometry_batch = collate_mesh_batch
