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
                        self._samples.append((fp, key))
        if len(self._samples) == 0:
            raise RuntimeError(f"No per-sample-group data: {full_paths}")

    def undo_normalize(self, frf):
        return torch.sinh(frf)

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

        # 像素坐标: X_pix = int(X * 1000), Y_pix = int(Y * 1000)
        X_pix = (X * 1000).long().clamp(0, L_MM - 1)
        Y_pix = (Y * 1000).long().clamp(0, W_MM - 1)

        # 初始化 6 通道图像
        img = torch.zeros(6, W_MM, L_MM, dtype=torch.float32)

        # Ch0: Z/H, 每像素取最大值
        img_ch0 = torch.full((W_MM, L_MM), -1.0, dtype=torch.float32)
        for n in range(len(points)):
            xp, yp = X_pix[n].item(), Y_pix[n].item()
            if Z_H[n] > img_ch0[yp, xp]:
                img_ch0[yp, xp] = Z_H[n]
        img[0] = img_ch0.clamp(0, 1)

        # Ch1: is_fixed, per-pixel max
        for n in range(len(points)):
            xp, yp = X_pix[n].item(), Y_pix[n].item()
            img[1, yp, xp] = max(img[1, yp, xp].item(), is_fixed[n].item())

        # Ch2: log10(K), spring nodes only
        for n in range(len(points)):
            if logK[n] > 0:
                xp, yp = X_pix[n].item(), Y_pix[n].item()
                img[2, yp, xp] = max(img[2, yp, xp].item(), logK[n].item())

        # Ch3: log10(C), spring nodes only
        for n in range(len(points)):
            if logC[n] > 0:
                xp, yp = X_pix[n].item(), Y_pix[n].item()
                img[3, yp, xp] = max(img[3, yp, xp].item(), logC[n].item())

        # Ch4-5: 全局材料常数
        img[4] = E_val
        img[5] = rho_val

        return img

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
                    out[key] = val

        # ω 归一化到 [0,1] (sigmoid 输出空间)
        OMEGA_MAX = 25000.0
        if 'modal_omega' in out:
            out['modal_omega_norm'] = out['modal_omega'].clone() / OMEGA_MAX
            out['modal_omega_target'] = out.pop('modal_omega')  # 保留物理值用于评估

        # 2.5D 投影
        image_tensor = self._project_to_image(points, point_feat)

        # 坐标归一化到 [-1, 1]
        query_coords = torch.stack([
            points[:, 0] / 0.160 * 2 - 1,  # X [-1, 1]
            points[:, 1] / 0.060 * 2 - 1,  # Y [-1, 1]
        ], dim=-1)

        # 归一化
        if self.normalization:
            freqs = (freqs - self.freq_min) / (self.freq_max - self.freq_min) * 2 - 1
            frf = torch.asinh(frf)

        result = {
            'image_tensor': image_tensor,
            'query_coords': query_coords,
            'point_frf': frf,
            'frequencies': freqs,
        }
        for key, val in out.items():
            result[key] = val
        # modal_omega 为归一化值(训练), modal_omega_phys 为物理值(评估)
        if 'modal_omega_target' in out:
            result['modal_omega_phys'] = out.pop('modal_omega_target')
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
    return result
