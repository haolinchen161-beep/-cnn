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
        # fc_up 产生初始特征图 [B, 256, 4, 10]
        self.fc_up = nn.Linear(hidden, 256 * 4 * 10)

        # 逐层上采样 + 跳连: (当前通道 + 跳连通道) → 输出通道
        self.up3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), nn.BatchNorm2d(256),
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
        x = self.fc_up(latent).view(-1, 256, 4, 10)      # [B, 256, 4, 10]

        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))           # cat(256, 256) → 256

        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))           # cat(256, 128) → 128

        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))           # cat(128, 64) → 64

        x = F.interpolate(x, size=(60, 160), mode='bilinear', align_corners=False)
        return self.final(x)


class SineAct(nn.Module):
    def __init__(self, w0=20.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class CoordinatePhiResidual(nn.Module):
    """CNN map φ + 坐标连续场 residual (SIREN)。"""

    def __init__(self, hidden=256, node_feat_dim=7, n_modes=3, mode_dim=32):
        super().__init__()
        self.n_modes = n_modes
        self.mode_emb = nn.Parameter(torch.randn(n_modes, mode_dim) * 0.02)

        in_dim = hidden + 3 + node_feat_dim + mode_dim + 3

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            SineAct(w0=15.0),
            nn.Linear(256, 256),
            SineAct(w0=15.0),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi_base, latent, node_xyz, node_features, batch_idx):
        if node_xyz is None or node_features is None:
            return phi_base

        n_nodes = phi_base.shape[0]
        k = self.n_modes

        latent_n = latent[batch_idx]                            # [N, H]
        latent_e = latent_n.unsqueeze(1).expand(n_nodes, k, -1)
        xyz_e = node_xyz.unsqueeze(1).expand(n_nodes, k, -1)
        feat_e = node_features.unsqueeze(1).expand(n_nodes, k, -1)
        mode_e = self.mode_emb.unsqueeze(0).expand(n_nodes, k, -1)

        x = torch.cat([latent_e, xyz_e, feat_e, mode_e, phi_base], dim=-1)

        delta = self.net(x)

        # 残差幅度不要太大，防止一开始破坏已有 map 解码器
        return phi_base + 0.20 * torch.tanh(delta)


class PhysicsPriorOmegaHead(nn.Module):
    """物理先验频率头：global物理量给粗预测，CNN latent 给残差修正。"""

    def __init__(self, hidden=256, n_modes=3, phys_dim=22):
        super().__init__()
        self.n_modes = n_modes

        self.prior_mlp = nn.Sequential(
            nn.Linear(phys_dim, 128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, n_modes),
        )

        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden + phys_dim, 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, n_modes),
        )

        # f1, gap21, gap32 的物理范围
        self.f1_min, self.f1_max = 700.0, 1250.0
        self.g21_min, self.g21_max = 700.0, 2600.0
        self.g32_min, self.g32_max = 150.0, 1000.0

        self.f1_span = self.f1_max - self.f1_min
        self.g21_span = self.g21_max - self.g21_min
        self.g32_span = self.g32_max - self.g32_min

        def inv_sigmoid(p):
            p = torch.tensor(p).clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1.0 - p))

        # 初始化到当前数据均值附近
        b1 = inv_sigmoid((957.0 - self.f1_min) / self.f1_span)
        b2 = inv_sigmoid((1632.0 - self.g21_min) / self.g21_span)
        b3 = inv_sigmoid((388.0 - self.g32_min) / self.g32_span)

        with torch.no_grad():
            self.prior_mlp[-1].bias.copy_(torch.tensor([b1, b2, b3]))
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(self, latent, phys_features):
        prior_raw = self.prior_mlp(phys_features)

        # delta 只允许做有限修正，防止 CNN latent 直接盖过物理先验
        delta_raw = 0.35 * torch.tanh(
            self.delta_mlp(torch.cat([latent, phys_features], dim=-1))
        )

        raw = prior_raw + delta_raw
        s = torch.sigmoid(raw)

        f1 = self.f1_min + self.f1_span * s[:, 0:1]
        g21 = self.g21_min + self.g21_span * s[:, 1:2]
        g32 = self.g32_min + self.g32_span * s[:, 2:3]

        f2 = f1 + g21
        f3 = f2 + g32
        f_hz = torch.cat([f1, f2, f3], dim=-1)

        return f_hz * (2.0 * torch.pi)


class ZetaHead(nn.Module):
    """物理驱动阻尼预测器: latent + ω + 边界 C/K 特征 → ζ。"""

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden + n_modes + 2, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, n_modes),
        )

    def forward(self, zeta_input):
        out = self.mlp(zeta_input)
        zeta = torch.sigmoid(out) * 0.030 + 0.001  # [0.001, 0.031]
        return torch.log(zeta), zeta


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


class DirectionBranchHead(nn.Module):
    """辅助分类头: 预测每阶模态 X/Y/Z 方向能量比例 (soft label)。

    输出 log_softmax，用于 KL 散度监督。概率拼接后喂给 PhiScaleHead 做条件特征。
    """

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, n_modes * 3),
        )

    def forward(self, latent):
        logits = self.mlp(latent).view(latent.shape[0], -1, 3)  # [B, K, 3]
        return F.log_softmax(logits, dim=-1)


class PhiScaleHead(nn.Module):
    """模态幅值预测器。接收 latent + 方向概率 [B, 512+K*3]，预测物理 joint scale。

    DirectionBranchHead 输出的方向比例作为"条件小抄"，让 scale 头知道每阶是 Z主导/XY主导。
    """

    def __init__(self, hidden=512, n_modes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden + n_modes * 3, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_modes),
        )
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.zeros(n_modes))

    def forward(self, conditioned_latent):
        return torch.exp(self.mlp(conditioned_latent))  # [B, K]


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
        self.omega_head = PhysicsPriorOmegaHead(hidden, n_modes, phys_dim=22)
        self.zeta_head = ZetaHead(hidden, n_modes)
        self.micro_decoder = MicroDecoder(hidden, n_modes)
        self.phi_refiner = NodePhiRefiner(hidden, node_feat_dim=7, n_modes=n_modes)
        self.coord_phi_residual = CoordinatePhiResidual(hidden, node_feat_dim=7, n_modes=n_modes)
        self.phi_scale_head = PhiScaleHead(hidden, n_modes)
        self.branch_head = DirectionBranchHead(hidden, n_modes)
        self.physics = PhysicsDecoder(amp_scale, freq_min, freq_max)

    def forward(self, image_tensor, query_coords, frequencies=None,
                phi_exc=None, batch=None, alpha=1.0,
                node_xyz=None, node_features=None,
                omega_true=None, global_features=None):
        """
        Returns:
            frf, omega_phys, log_zeta, zeta, phi
            omega_phys: [B, K] rad/s, zeta: [B, K], phi: [total_N, K, 3]
        """
        B = image_tensor.shape[0]
        if query_coords.ndim == 3:
            coords_flat = query_coords.reshape(-1, 2)
            batch = batch if batch is not None else torch.arange(B, device=image_tensor.device).repeat_interleave(query_coords.shape[1])
        else:
            coords_flat = query_coords
            batch = batch if batch is not None else torch.zeros(coords_flat.shape[0], dtype=torch.long, device=image_tensor.device)

        latent, skips = self.encoder(image_tensor)

        # 1. 首先提取全图边界 C/K 特征 (给频率和阻尼做联合小抄)
        c_k_features = torch.zeros(B, 2, device=latent.device)
        if node_features is not None and batch is not None:
            for b_idx in range(B):
                mask = (batch == b_idx) & (node_features[:, 4] > 0)  # logK>0 = 弹簧节点
                if mask.any():
                    c_k_features[b_idx, 0] = node_features[mask, 4].mean()  # log10(K)
                    c_k_features[b_idx, 1] = node_features[mask, 5].mean()  # log10(C)

        # 防御: global_features 取 [E/E_base, ρ/ρ_base], 跳过硬编码的 prxy
        if global_features is None:
            mat_feat = torch.zeros(B, 20, device=latent.device)
        else:
            mat_feat = global_features  # [B, 20]

        # 2. 频率预测: 物理先验频率头 (global物理量给粗预测, CNN latent 给残差)
        phys_features = torch.cat([mat_feat, c_k_features], dim=-1)  # [B, 22]
        omega_phys = self.omega_head(latent, phys_features)

        # 3. 阻尼预测: latent + ω + C/K
        zeta_input = torch.cat([
            latent,
            omega_phys.detach() / 5000.0,
            c_k_features,
        ], dim=-1)

        # 4. 阻尼预测
        log_zeta, zeta = self.zeta_head(zeta_input)

        # 方向分类头: 预测每模态 XYZ 能量比例, 作为 PhiScaleHead 的"条件小抄"
        self.branch_log_probs = self.branch_head(latent)             # [B, K, 3]
        branch_probs = torch.exp(self.branch_log_probs).view(B, -1)  # [B, K*3]
        conditioned_latent = torch.cat([latent, branch_probs], dim=-1)  # [B, 512+K*3]

        # 振型: 2D map → grid_sample → [N, K, 3]
        mode_maps = self.micro_decoder(latent, skips)                # [B, K*3, H, W]

        # 联合归一化 + 共享 scale (有条件方向信息)
        mode_maps_3d = mode_maps.view(B, self.n_modes, 3, mode_maps.shape[-2], mode_maps.shape[-1])
        maps_flat = mode_maps_3d.reshape(B, self.n_modes, -1)       # [B, K, 3*H*W]
        maps_std = torch.std(maps_flat, dim=2) + 1e-8               # [B, K] 联合 std
        normalized_maps = mode_maps_3d / maps_std.view(B, self.n_modes, 1, 1, 1)
        scale = self.phi_scale_head(conditioned_latent)              # [B, K]
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
            phi = self.coord_phi_residual(phi, latent, node_xyz, node_features, batch)

        # FRF 重建: 若传入 omega_true 则使用 teacher forcing，否则使用 omega_phys
        if frequencies is not None:
            phi_z = phi[..., 2]                                      # [total_N, K] Z 向分量
            omega_used = omega_true if omega_true is not None else omega_phys
            # ω/ζ detach: FRF 梯度只流向 φ, 不干扰频率和阻尼
            frf = self.physics(phi_z, omega_used.detach(), zeta.detach(),
                               frequencies, phi_exc, batch_idx=batch, alpha=alpha)
        else:
            frf = None

        return frf, omega_phys, log_zeta, zeta, phi
