"""
unet_physics_model.py — 2.5D CNN-UNet 模态参数预测。

架构: ResNet+SE Encoder + OmegaHead + ZetaHead + MicroDecoder + PhysicsDecoder

  image_tensor [B,6,60,160]
       │
  ┌────┴──────────┐
  │ ResNet+SE Enc │ 含 U-Net 跳连 → [B,512]
  └────┬──────────┘
       │
  ┌────┴──────────┐
  │               │
  │ OmegaHead     │ MicroDecoder (U-Net)
  │ → ω[B,K]      │ → mode_maps [B,K,60,160]
  │ ZetaHead      │        │
  │ → ζ[B,K]      │   grid_sample(query_coords)
  │               │   + PhiScaleHead (形数解耦)
  └────┬──────────┘   → phi [B,N,K]
       │
  ┌────┴──────────┐
  │ PhysicsDecoder │ H=Σφ_kφ_k/(ω²-ω²+j2ζωω)
  │ → FRF          │
  └───────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_decoder import PhysicsDecoder


# ============================================================
# SE + ResNet 基础模块
# ============================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResSEBlock(nn.Module):
    """带 SE 通道注意力的残差块，支持下采样"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.act = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.act(out)


class ImprovedCNNEncoder(nn.Module):
    """ResNet+SE Encoder: 深度残差 + 通道注意力 + U-Net 跳连。

    跳连输出通道: f1=64, f2=128, f3=256, f4=512 (逐层翻倍)
    """

    def __init__(self, in_ch=6, hidden=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )  # → [B, 64, 30, 80]

        self.layer1 = ResSEBlock(64, 128, stride=2)   # → [B, 128, 15, 40]
        self.layer2 = ResSEBlock(128, 256, stride=2)   # → [B, 256, 8, 20]
        self.layer3 = ResSEBlock(256, 512, stride=2)   # → [B, 512, 4, 10]

        self.pool = nn.AdaptiveAvgPool2d((2, 5))
        self.fc = nn.Linear(512 * 10, hidden)

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        latent = self.fc(self.pool(f4).flatten(1))
        return latent, (f1, f2, f3, f4)


class MicroDecoder(nn.Module):
    """U-Net 上采样解码器: latent + skips → mode_maps [B,K,60,160]。

    跳连通道: f1=64, f2=128, f3=256, f4=512 (来自 ImprovedCNNEncoder)。
    """

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        # fc_up 产生初始特征图 [B, 512, 8, 20]
        self.fc_up = nn.Linear(hidden, 512 * 8 * 20)

        # 逐层上采样 + 跳连: (当前通道 + 跳连通道) → 输出通道
        self.up3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.GELU(), nn.Dropout2d(0.1))
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.GELU(), nn.Dropout2d(0.1))
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.GELU())
        self.final = nn.Conv2d(64, n_modes * 3, 3, padding=1)  # 每模态输出 XYZ 三向

    def forward(self, latent, skips):
        f1, f2, f3, f4 = skips
        x = self.fc_up(latent).view(-1, 512, 8, 20)      # [B, 512, 8, 20]

        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))           # cat(512, 256) → 256

        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))           # cat(256, 128) → 128

        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))           # cat(128, 64) → 64

        x = F.interpolate(x, size=(60, 160), mode='bilinear', align_corners=False)
        return self.final(x)


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
        self.n_modes = n_modes
        self.net = nn.Sequential(
            nn.Linear(hidden + 3 + node_feat_dim + n_modes * 3, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_modes * 3),
        )

    def forward(self, phi_base, latent, node_xyz, node_features, batch_idx):
        if node_xyz is None or node_features is None:
            return phi_base
        latent_n = latent[batch_idx]
        # phi_base [N, K, 3] → flatten to [N, K*3]
        phi_flat = phi_base.reshape(phi_base.shape[0], -1)
        x = torch.cat([latent_n, node_xyz, node_features, phi_flat], dim=-1)
        delta = 0.25 * torch.tanh(self.net(x))
        return phi_base + delta.view(-1, self.n_modes, 3)


class PhiScaleHead(nn.Module):
    """模态幅值标量预测器。解耦形状与幅值：UNet 输出纯形状(unit std)，MLP 预测物理尺度。"""

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_modes),
        )
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0]))  # exp(0)=1.0

    def forward(self, latent):
        return torch.exp(self.mlp(latent))  # [B, K] > 0


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

        self.encoder = ImprovedCNNEncoder(in_ch, hidden)
        self.omega_head = OmegaHead(hidden, n_modes)
        self.zeta_head = ZetaHead(hidden, n_modes)
        self.micro_decoder = MicroDecoder(hidden, n_modes)
        self.phi_refiner = NodePhiRefiner(hidden, node_feat_dim=7, n_modes=n_modes)
        self.phi_scale_head = PhiScaleHead(hidden, n_modes)
        self.physics = PhysicsDecoder(amp_scale, freq_min, freq_max)

    def forward(self, image_tensor, query_coords, frequencies=None,
                phi_exc=None, batch=None, alpha=1.0,
                node_xyz=None, node_features=None,
                omega_true=None, teacher_alpha=0.0):
        """
        Returns:
            frf, omega_phys, log_zeta, zeta, phi
            omega_phys: [B, K] rad/s (单调递增保证)
            log_zeta:   [B, K] 对数阻尼
            zeta:       [B, K] 物理阻尼 = exp(log_zeta)
            phi:        [total_N, K, 3] 三维振型 (X, Y, Z)

        Teacher-Forced Omega (Phase2):
            teacher_alpha=1.0 → FRF 全用 ω_true (峰位置完美对齐，只训 φ/ζ)
            teacher_alpha=0.0 → FRF 全用 ω_pred (端到端推理模式)
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

        # 振型: 2D map → grid_sample → [N, K, 3]
        # 形数解耦：UNet 输出纯形状(unit std)，PhiScaleHead 预测物理幅值
        mode_maps = self.micro_decoder(latent, skips)                # [B, K*3, H, W]

        # 拆分为 [B, K, 3, H, W] 以计算三维整体能量 Std
        mode_maps_3d = mode_maps.view(B, self.n_modes, 3, mode_maps.shape[-2], mode_maps.shape[-1])
        maps_flat = mode_maps_3d.reshape(B, self.n_modes, -1)       # [B, K, 3*H*W]
        maps_std = torch.std(maps_flat, dim=2) + 1e-8               # [B, K]
        normalized_maps = mode_maps_3d / maps_std.view(B, self.n_modes, 1, 1, 1)
        scale = self.phi_scale_head(latent)                          # [B, K]
        mode_maps_3d = normalized_maps * scale.view(B, self.n_modes, 1, 1, 1)

        # 合并回 [B, K*3, H, W] 送入 grid_sample
        mode_maps = mode_maps_3d.reshape(B, self.n_modes * 3, mode_maps.shape[-2], mode_maps.shape[-1])
        phi_list = []
        for b in range(B):
            mask = batch == b
            coords_b = coords_flat[mask].unsqueeze(0).unsqueeze(0)
            maps_b = mode_maps[b:b+1]
            phi_b = F.grid_sample(maps_b, coords_b, mode='bilinear',
                                  align_corners=True, padding_mode='border')
            # phi_b: [1, K*3, 1, N] → squeeze → [K*3, N] → T → [N, K*3] → view [N, K, 3]
            phi_b = phi_b.squeeze(2).transpose(1, 2).squeeze(0)     # [N, K*3]
            phi_list.append(phi_b.view(-1, self.n_modes, 3))
        phi = torch.cat(phi_list, dim=0)                             # [total_N, K, 3]

        # 轻量节点级修正 (若提供了 node 信息)
        if node_xyz is not None and node_features is not None:
            phi = self.phi_refiner(phi, latent, node_xyz, node_features, batch)

        # FRF 重建 (Teacher-Forced: Z 向 FRF，从三维振型提取 dim=2)
        if frequencies is not None:
            phi_z = phi[..., 2]                                      # [total_N, K] Z 向分量
            if omega_true is not None and teacher_alpha > 0.0:
                omega_used = teacher_alpha * omega_true + (1.0 - teacher_alpha) * omega_phys
            else:
                omega_used = omega_phys
            frf = self.physics(phi_z, omega_used, zeta, frequencies, phi_exc,
                               batch_idx=batch, alpha=alpha)
        else:
            frf = None

        return frf, omega_phys, log_zeta, zeta, phi
