"""
unet_physics_model.py — 2.5D CNN-UNet 模态参数预测。

架构: CNN Encoder + Macro MLP Decoder + Micro Upsample Decoder + PhysicsDecoder

  image_tensor [B,6,60,160]
       │
  ┌────┴────┐
  │ CNN Enc │ 4层Conv → [B,512]
  └────┬────┘
       │
  ┌────┴──────────┐
  │               │
  │ Macro MLP     │ Micro Upsample
  │ → ω[B,K]      │ → mode_maps [B,K,60,160]
  │ → ζ[B,K]      │        │
  │               │   grid_sample(query_coords)
  │               │   → phi [B,N,K]
  └────┬──────────┘
       │
  ┌────┴──────────┐
  │ PhysicsDecoder │ H=Σφ_kφ_k/(ω²-ω²+j2ζωω)
  │ → FRF (asinh)  │
  └───────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import PhysicsDecoder


class CNNEncoder(nn.Module):
    """4层 CNN 编码器 + UNet 跳连"""

    def __init__(self, in_ch=6, hidden=512):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, hidden)

    def forward(self, x):
        f1 = self.conv1(x)  # [B,32,30,80]
        f2 = self.conv2(f1) # [B,64,15,40]
        f3 = self.conv3(f2) # [B,128,8,20]
        f4 = self.conv4(f3) # [B,256,4,10]
        latent = self.fc(self.pool(f4).flatten(1))
        return latent, (f1, f2, f3, f4)


class MacroDecoder(nn.Module):
    """宏观参数 MLP: 512 → ω[B,K] + ζ[B,K]
    ω 输出归一化 [0,1], 使用时 ×omega_max 还原。
    """

    def __init__(self, hidden=512, n_modes=3, omega_max=25000.0):
        super().__init__()
        self.n_modes = n_modes
        self.omega_max = omega_max
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_modes * 2),
        )
        nn.init.constant_(self.mlp[-1].bias[:n_modes], -2.0)  # sigmoid(-2)≈0.12→3000Hz

    def forward(self, latent):
        out = self.mlp(latent)
        omega_norm = torch.sigmoid(out[:, :self.n_modes])      # [0,1]
        zeta = torch.sigmoid(out[:, self.n_modes:]) * 0.05 + 1e-4
        return omega_norm, zeta


class MicroDecoder(nn.Module):
    """UNet 上采样解码器: 512 + skips → mode_maps [B,K,60,160]"""

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.fc_up = nn.Linear(hidden, 256 * 4 * 10)  # -> f4 尺寸 [B,256,4,10]

        self.up3 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, n_modes, 3, padding=1)

    def forward(self, latent, skips):
        f1, f2, f3, f4 = skips
        x = self.fc_up(latent).view(-1, 256, 4, 10)

        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))

        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))

        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))

        x = F.interpolate(x, size=(60, 160), mode='bilinear', align_corners=False)
        return self.final(x)


class UNetPhysicsModel(nn.Module):
    """2.5D CNN-UNet 模态参数预测模型。

    Args:
        in_ch:      输入图像通道数 (默认 6)
        hidden:     隐向量维度
        n_modes:    模态阶数 K
        amp_scale:  FRF 幅值缩放
        freq_min/max: 频率范围
    """

    def __init__(self, in_ch=6, hidden=512, n_modes=3,
                 amp_scale=500000.0, freq_min=1.0, freq_max=5000.0):
        super().__init__()
        self.n_modes = n_modes

        self.encoder = CNNEncoder(in_ch, hidden)
        self.macro_decoder = MacroDecoder(hidden, n_modes)
        self.micro_decoder = MicroDecoder(hidden, n_modes)
        self.physics = PhysicsDecoder(amp_scale, freq_min, freq_max)

    def forward(self, image_tensor, query_coords, frequencies=None,
                phi_exc=None, batch=None, alpha=1.0):
        """
        Args:
            image_tensor: [B, C, 60, 160] 物理场图像
            query_coords: [total_N, 2] 或 [B, N, 2] 归一化坐标 [-1,1]
            frequencies:  [B, F] 归一化频率 或 None
            phi_exc:      [B, K] 激励点振型
            batch:        [total_N,] 变N时批次索引 或 None
        Returns:
            frf, omega, zeta, phi
        """
        # 处理可变 N
        if query_coords.ndim == 3:
            B, N_max, _ = query_coords.shape
            coords = query_coords
            var_n_flag = False
        else:
            B = int(batch.max().item()) + 1
            var_n_flag = True

        latent, skips = self.encoder(image_tensor)
        omega_norm, zeta = self.macro_decoder(latent)
        mode_maps = self.micro_decoder(latent, skips)

        # grid_sample: 从连续振型场采样到离散节点
        if var_n_flag:
            phi_list = []
            for b in range(B):
                mask = batch == b
                coords_b = query_coords[mask].unsqueeze(0).unsqueeze(0)  # [1,1,N_b,2]
                maps_b = mode_maps[b:b+1]
                phi_b = F.grid_sample(maps_b, coords_b, mode='bilinear',
                                      align_corners=True, padding_mode='border')
                phi_list.append(phi_b.squeeze(2).transpose(1, 2).squeeze(0))  # [N_b, K]
            phi = torch.cat(phi_list, dim=0)  # [total_N, K]
        else:
            coords = query_coords.unsqueeze(1)  # [B, 1, N, 2]
            phi = F.grid_sample(mode_maps, coords, mode='bilinear',
                                align_corners=True, padding_mode='border')
            phi = phi.squeeze(2).transpose(1, 2)  # [B, N, K]
            phi = phi.reshape(-1, self.n_modes)  # [B*N, K]

        omega_phys = omega_norm * self.macro_decoder.omega_max
        if frequencies is not None:
            frf = self.physics(phi, omega_phys, zeta, frequencies, phi_exc,
                               batch_idx=batch if var_n_flag else None, alpha=alpha)
        else:
            frf = None

        return frf, omega_norm, zeta, phi  # ω_norm ∈ [0,1], target ∈ [0,1]
