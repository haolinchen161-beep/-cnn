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
        self.pool = nn.AdaptiveAvgPool2d((2, 5))  # 保留 2×5 空间网格 (匹配板 160:60≈2.67:1)
        self.fc = nn.Linear(256 * 10, hidden)     # 256 通道 × 10 个空间位置

    def forward(self, x):
        f1 = self.conv1(x)  # [B,32,30,80]
        f2 = self.conv2(f1) # [B,64,15,40]
        f3 = self.conv3(f2) # [B,128,8,20]
        f4 = self.conv4(f3) # [B,256,4,10]
        latent = self.fc(self.pool(f4).flatten(1))
        return latent, (f1, f2, f3, f4)


class MacroDecoder(nn.Module):
    """宏观参数 MLP: latent+skip_pools → ω[B,K] + ζ[B,K]"""

    def __init__(self, hidden=512, n_modes=3, omega_max=25000.0):
        super().__init__()
        self.n_modes = n_modes
        self.omega_max = omega_max
        input_dim = hidden  # 512
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.1),   # 加宽：256→512
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),          # 加深一层
            nn.Linear(256, 128), nn.GELU(),
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
        C = 2  # 通道倍数
        self.fc_up = nn.Linear(hidden, 256 * C * 8 * 20)

        self.up3 = nn.Sequential(nn.Conv2d(256*C + 128, 128*C, 3, padding=1), nn.BatchNorm2d(128*C), nn.GELU(), nn.Dropout2d(0.1))
        self.up2 = nn.Sequential(nn.Conv2d(128*C + 64, 64*C, 3, padding=1), nn.BatchNorm2d(64*C), nn.GELU(), nn.Dropout2d(0.1))
        self.up1 = nn.Sequential(nn.Conv2d(64*C + 32, 32*C, 3, padding=1), nn.BatchNorm2d(32*C), nn.GELU())
        self.final = nn.Conv2d(32*C, n_modes, 3, padding=1)
        self.C = C

    def forward(self, latent, skips):
        f1, f2, f3, f4 = skips
        C = self.C
        x = self.fc_up(latent).view(-1, 256*C, 8, 20)

        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))

        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))

        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))

        x = F.interpolate(x, size=(60, 160), mode='bilinear', align_corners=False)
        return self.final(x) * 30.0  # phi_true std=1~2, final输出std≈0.05→×30≈1.5


class OmegaHead(nn.Module):
    """单调间隙频率预测器。softplus 保证 w1<w2<w3，无需 hardcode 统计量。"""

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_modes),
        )

    def forward(self, latent):
        out = self.mlp(latent)
        # 过滤后统计: f1≈948Hz(5957), gap21≈1399Hz(8790), gap32≈498Hz(3130)
        w1 = F.softplus(out[:, 0:1]) * 8000.0 + 500.0       # → ~6044 rad/s ≈ 962Hz
        gap21 = F.softplus(out[:, 1:2]) * 12000.0 + 500.0   # → ~8816 rad/s ≈ 1403Hz
        gap32 = F.softplus(out[:, 2:3]) * 4200.0 + 200.0    # → ~3111 rad/s ≈ 495Hz
        w2 = w1 + gap21
        w3 = w2 + gap32
        return torch.cat([w1, w2, w3], dim=-1)  # [B, 3] rad/s


class ZetaHead(nn.Module):
    """对数域阻尼预测器。exp(bias)≈0.02，clamp 防发散。"""

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, n_modes),
        )
        # 过滤后真实 ζ: mode1≈0.0029, mode2≈0.0062, mode3≈0.0074
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([-5.85, -5.08, -4.91]))

    def forward(self, latent):
        log_zeta = self.mlp(latent)
        return torch.clamp(log_zeta, min=-9.0, max=-0.1)  # 1.2e-4 ~ 0.9 (收紧下限)


class NodePhiRefiner(nn.Module):
    """轻量节点级振型修正。2D map 为主，xyz/node_features 做 residual。"""

    def __init__(self, hidden=512, node_feat_dim=7, n_modes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + 3 + node_feat_dim + n_modes, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_modes),
        )

    def forward(self, phi_base, latent, node_xyz, node_features, batch_idx):
        if node_xyz is None or node_features is None:
            return phi_base
        latent_n = latent[batch_idx]
        x = torch.cat([latent_n, node_xyz, node_features, phi_base], dim=-1)
        delta = 0.25 * torch.tanh(self.net(x))
        return phi_base + delta


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
        self.omega_head = OmegaHead(hidden, n_modes)
        self.zeta_head = ZetaHead(hidden, n_modes)
        self.micro_decoder = MicroDecoder(hidden, n_modes)
        self.phi_refiner = NodePhiRefiner(hidden, node_feat_dim=7, n_modes=n_modes)
        self.physics = PhysicsDecoder(amp_scale, freq_min, freq_max)

    def forward(self, image_tensor, query_coords, frequencies=None,
                phi_exc=None, batch=None, alpha=1.0,
                node_xyz=None, node_features=None):
        """
        Returns:
            frf, omega_phys, log_zeta, zeta, phi
            omega_phys: [B, K] rad/s (单调递增保证)
            log_zeta:   [B, K] 对数阻尼
            zeta:       [B, K] 物理阻尼 = exp(log_zeta)
            phi:        [total_N, K] 振型
        """
        B = image_tensor.shape[0]
        if query_coords.ndim == 3:
            coords_flat = query_coords.reshape(-1, 2)
            batch = batch if batch is not None else torch.arange(B, device=image_tensor.device).repeat_interleave(query_coords.shape[1])
        else:
            coords_flat = query_coords
            batch = batch if batch is not None else torch.zeros(coords_flat.shape[0], dtype=torch.long, device=image_tensor.device)

        latent, skips = self.encoder(image_tensor)

        # 频率: 直接输出物理 rad/s (单调保证)
        omega_phys = self.omega_head(latent)
        # 阻尼: 对数域输出
        log_zeta = self.zeta_head(latent)
        zeta = torch.exp(log_zeta)

        # 振型: 2D map → grid_sample
        mode_maps = self.micro_decoder(latent, skips)
        phi_list = []
        for b in range(B):
            mask = batch == b
            coords_b = coords_flat[mask].unsqueeze(0).unsqueeze(0)
            maps_b = mode_maps[b:b+1]
            phi_b = F.grid_sample(maps_b, coords_b, mode='bilinear',
                                  align_corners=True, padding_mode='border')
            phi_list.append(phi_b.squeeze(2).transpose(1, 2).squeeze(0))
        phi = torch.cat(phi_list, dim=0)

        # 轻量节点级修正 (若提供了 node 信息)
        if node_xyz is not None and node_features is not None:
            phi = self.phi_refiner(phi, latent, node_xyz, node_features, batch)

        # FRF 重建
        if frequencies is not None:
            frf = self.physics(phi, omega_phys, zeta, frequencies, phi_exc,
                               batch_idx=batch, alpha=alpha)
        else:
            frf = None

        return frf, omega_phys, log_zeta, zeta, phi
