"""
dataset.py — 几何数据 Dataset + DataLoader (2.5D 投影版)。

将 3D 节点投影为 160×60 像素的 6 通道物理图像。
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

W_MM, L_MM = 60, 160  # Y=60mm, X=160mm (1mm/pixel)

# 输入图像逐通道归一化 (train.h5 统计, 避免 CNN 第一层输入不平衡)
IMG_MEAN = torch.tensor([0.2142, 0.0044, 0.0315, 0.0005, 1.0001, 1.0000]).view(6, 1, 1)
IMG_STD  = torch.tensor([0.3508, 0.0620, 0.4267, 0.0121, 0.0284, 0.0171]).view(6, 1, 1)


class GeometricHDF5Dataset(Dataset):
    """HDF5 数据集 (2.5D 投影 + per-sample-group 格式)。"""

    def __init__(self, data_paths, config, data_dir=".",
                 test=False, normalization=True):
        self.config = config
        self.normalization = normalization
        self.test = test
        self.freq_min = config.get('freq_min', 1.0)
        self.freq_max = config.get('freq_max', 5000.0)
        self._samples = []

        full_paths = [os.path.join(data_dir, p) for p in data_paths]
        self._load_index(full_paths)

    def _load_index(self, full_paths):
        for fp in full_paths:
            with h5py.File(fp, 'r') as f:
                for key in sorted(f.keys(), key=lambda k: int(k.split('_')[-1])):
                    if key.startswith('sample_'):
                        # 过滤极端间隙: 太小→模态混淆, 太大→离群
                        omega = f[key]['modal_omega'][:]
                        fhz = omega / (2.0 * np.pi)
                        g32 = fhz[2] - fhz[1]
                        if g32 < 200.0 or g32 > 900.0:
                            continue
                        self._samples.append((fp, key))
        if len(self._samples) == 0:
            raise RuntimeError(f"No per-sample-group data: {full_paths}")

    def undo_normalize(self, frf):
        return torch.sinh(frf)

    def _build_node_xyz(self, points):
        """归一化节点坐标到 [-1,1]"""
        return torch.stack([
            points[:, 0] / 0.160 * 2 - 1,
            points[:, 1] / 0.060 * 2 - 1,
            points[:, 2] / 0.010 * 2 - 1,
        ], dim=-1).float()

    def _build_global_features(self, points, point_features):
        """从节点特征构建全局特征向量 [G]"""
        if point_features is None:
            return torch.zeros(20, dtype=torch.float32)

        E_ratio = point_features[0, 0]
        prxy = point_features[0, 1]
        rho_ratio = point_features[0, 2]
        is_fixed = point_features[:, 3]
        logK = point_features[:, 4]
        logC = point_features[:, 5]
        z_h = point_features[:, 6]

        spring_mask = logK > 0
        fixed_mask = is_fixed > 0
        corner_mask = is_fixed > 0.75
        side_mask = (is_fixed > 0.25) & (is_fixed < 0.75)

        def safe_mean(x):
            return x.mean() if x.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
        def safe_std(x):
            return x.std(unbiased=False) if x.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
        def safe_min(x):
            return x.min() if x.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
        def safe_max(x):
            return x.max() if x.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)

        spring_logK = logK[spring_mask]
        spring_logC = logC[spring_mask]

        return torch.stack([
            E_ratio, prxy, rho_ratio,
            z_h.mean(), z_h.std(unbiased=False), z_h.min(), z_h.max(),
            spring_mask.float().mean(),
            fixed_mask.float().mean(),
            corner_mask.float().mean(),
            side_mask.float().mean(),
            safe_mean(spring_logK), safe_std(spring_logK), safe_min(spring_logK), safe_max(spring_logK),
            safe_mean(spring_logC), safe_std(spring_logC), safe_min(spring_logC), safe_max(spring_logC),
            torch.tensor(float(points.shape[0]), dtype=torch.float32) / 10000.0,
        ]).float()

    def _project_to_image(self, points, point_features):
        """将 3D 节点投影为 [6, 60, 160] 物理图像。

        Ch0: Z/H (局部厚度, max per pixel)
        Ch1: is_fixed (边界条件掩码, 0/0.5/1.0)
        Ch2: log10(K) (弹簧刚度, -1=无弹簧)
        Ch3: log10(C) (弹簧阻尼, -1=无弹簧)
        Ch4: E/E_base (全局材料)
        Ch5: rho/rho_base (全局材料)
        """
        X = points[:, 0]  # [0, 0.160] m
        Y = points[:, 1]  # [0, 0.060] m
        Z_H = point_features[:, 6]  # Z/H
        is_fixed = point_features[:, 3]
        logK = point_features[:, 4]
        logC = point_features[:, 5]
        E_val = point_features[0, 0]
        rho_val = point_features[0, 2]

        # 像素坐标
        X_pix = (X * 1000).long().clamp(0, L_MM - 1)
        Y_pix = (Y * 1000).long().clamp(0, W_MM - 1)
        pix_idx = Y_pix * L_MM + X_pix  # 1D index

        img = torch.zeros(6, W_MM * L_MM, dtype=torch.float32)

        # Ch0: Z/H max per pixel (向量化scatter)
        img[0].scatter_reduce_(0, pix_idx, Z_H, reduce='amax')
        # Ch1: is_fixed max per pixel
        img[1].scatter_reduce_(0, pix_idx, is_fixed, reduce='amax')
        # Ch2-3: 仅弹簧节点 (logK>0)
        spring_mask = logK > 0
        if spring_mask.any():
            img[2].scatter_reduce_(0, pix_idx[spring_mask], logK[spring_mask], reduce='amax')
            img[3].scatter_reduce_(0, pix_idx[spring_mask], logC[spring_mask], reduce='amax')

        # Ch4-5: 全局材料
        img[4] = E_val
        img[5] = rho_val

        return img.view(6, W_MM, L_MM)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        fp, grp_name = self._samples[idx]
        with h5py.File(fp, 'r') as f:
            grp = f[grp_name]
            points = torch.from_numpy(grp['points'][:]).float()
            freqs = torch.from_numpy(grp['frequencies'][:]).float()
            frf = torch.from_numpy(grp['point_frf'][:]).float()

            point_feat = None
            if 'point_features' in grp:
                gf = torch.from_numpy(grp['point_features'][:]).float()
                point_feat = gf if gf.ndim > 1 else gf.unsqueeze(0).expand(points.shape[0], -1)

            out = {}
            for key in ['modal_omega', 'modal_zeta', 'modal_phi', 'modal_phi_exc']:
                if key in grp:
                    val = torch.from_numpy(grp[key][:]).float()
                    # modal_phi_exc 新版为 [K,3]，取 Z 向分量用于 FRF 激励
                    if key == 'modal_phi_exc' and val.ndim == 2 and val.shape[1] == 3:
                        val = val[:, 2]  # [K, 3] → [K]
                    out[key] = val

            # 高级物理标签 (effm/pfact)，兼容老数据
            for h5_key, out_key in [('modal_effm', 'modal_effm'), ('modal_pfact', 'modal_pfact')]:
                if h5_key in grp:
                    out[out_key] = torch.from_numpy(grp[h5_key][:]).float()

        # ω 归一化到 [0,1] (sigmoid 输出空间)
        OMEGA_MAX = 25000.0
        if 'modal_omega' in out:
            out['modal_omega_norm'] = out['modal_omega'].clone() / OMEGA_MAX
            out['modal_omega_target'] = out.pop('modal_omega')  # 保留物理值用于评估

        # 2.5D 投影 + 逐通道归一化
        image_tensor = self._project_to_image(points, point_feat)
        image_tensor = (image_tensor - IMG_MEAN) / (IMG_STD + 1e-8)

        # 坐标归一化到 [-1, 1]
        query_coords = torch.stack([
            points[:, 0] / 0.160 * 2 - 1,  # X [-1, 1]
            points[:, 1] / 0.060 * 2 - 1,  # Y [-1, 1]
        ], dim=-1)

        # 归一化
        if self.normalization:
            freqs = (freqs - self.freq_min) / (self.freq_max - self.freq_min) * 2 - 1
            # FRF保持线性物理量, dB/CDF Loss自行处理量级

        result = {
            'image_tensor': image_tensor,
            'query_coords': query_coords,
            'points': points,  # (N,3) 物理坐标, 用于可视化
            'point_frf': frf,
            'frequencies': freqs,
        }
        for key, val in out.items():
            result[key] = val
        # modal_omega 为归一化值(训练), modal_omega_phys 为物理值(评估)
        if 'modal_omega_target' in out:
            result['modal_omega_phys'] = out.pop('modal_omega_target')

        # ---- CNN-ModalV2 新增字段 ----
        result['node_xyz'] = self._build_node_xyz(points)
        result['node_features'] = point_feat.float() if point_feat is not None else torch.zeros(points.shape[0], 7)
        result['global_features'] = self._build_global_features(points, point_feat)

        if 'modal_omega_phys' in result:
            result['modal_freq_hz'] = result['modal_omega_phys'] / (2.0 * torch.pi)

        if 'modal_zeta' in result:
            result['modal_log_zeta'] = torch.log(torch.clamp(result['modal_zeta'], min=1e-8))

        return result


def collate_geometry_batch(batch):
    """批次整理: 图像 stack, 坐标/frf cat, 可变F→list。"""
    f_lens = [item['frequencies'].shape[0] for item in batch]
    all_same_f = all(f == f_lens[0] for f in f_lens)

    if all_same_f:
        frequencies = torch.stack([item['frequencies'] for item in batch])
        images = torch.stack([item['image_tensor'] for item in batch])
        point_frf = torch.cat([item['point_frf'] for item in batch], dim=0)
        coords = torch.cat([item['query_coords'] for item in batch], dim=0)
        batch_tensor = torch.cat([
            torch.full((item['query_coords'].shape[0],), i, dtype=torch.long)
            for i, item in enumerate(batch)
        ], dim=0)
    else:
        frequencies = [item['frequencies'] for item in batch]
        point_frf = [item['point_frf'] for item in batch]
        images = torch.stack([item['image_tensor'] for item in batch])
        coords = torch.cat([item['query_coords'] for item in batch], dim=0)
        batch_tensor = torch.cat([
            torch.full((item['query_coords'].shape[0],), i, dtype=torch.long)
            for i, item in enumerate(batch)
        ], dim=0)

    out = {
        'image_tensor': images,
        'query_coords': coords,
        'point_frf': point_frf,
        'frequencies': frequencies,
        'batch': batch_tensor,
    }
    # CNN-ModalV2 新增字段
    out['node_xyz'] = torch.cat([item['node_xyz'] for item in batch], dim=0)
    out['node_features'] = torch.cat([item['node_features'] for item in batch], dim=0)
    out['global_features'] = torch.stack([item['global_features'] for item in batch])
    modal = _stack_modal(batch)
    if modal:
        out.update(modal)
    return out


def _stack_modal(batch):
    for key in ['modal_omega_norm', 'modal_zeta', 'modal_phi']:
        if key not in batch[0] or batch[0][key] is None:
            return {}
    result = {}
    for key in ['modal_omega_norm', 'modal_zeta', 'modal_phi_exc']:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.stack([item[key] for item in batch])
    result['modal_phi'] = torch.cat([item['modal_phi'] for item in batch], dim=0)
    if 'modal_omega_phys' in batch[0]:
        result['modal_omega_phys'] = torch.stack([item['modal_omega_phys'] for item in batch])
    if 'modal_freq_hz' in batch[0] and batch[0]['modal_freq_hz'] is not None:
        result['modal_freq_hz'] = torch.stack([item['modal_freq_hz'] for item in batch])
    if 'modal_log_zeta' in batch[0] and batch[0]['modal_log_zeta'] is not None:
        result['modal_log_zeta'] = torch.stack([item['modal_log_zeta'] for item in batch])
    for key in ['modal_effm', 'modal_pfact']:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.stack([item[key] for item in batch])
    return result
