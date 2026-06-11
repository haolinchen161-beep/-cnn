# geometric_frf_groove — 基于几何的频响函数预测

输入 3D 几何 + 边界条件 → UNetPhysicsModel (2.5D ResNet+SE UNet + PhysicsDecoder) → 模态参数 (ω, ζ, φ) → 物理公式重建 FRF。

## 1. 目录结构

```
├── models/
│   ├── unet_physics_model.py   主模型: ResNet+SE Encoder + OmegaHead + ZetaHead + MicroDecoder
│   ├── frf_model.py             模型工厂 build_geometric_model()
│   ├── physics_decoder.py       无参数物理解码器 (模态叠加→FRF)
│   ├── geometry_data.py         GeometryData 数据容器
├── data/
│   └── dataset.py               HDF5 数据集 (per-sample-group) + 2.5D 投影 + collate
├── training/
│   ├── losses.py                modal_loss (ω/ζ/φ) + frf_loss (dB + CDF)
│   ├── trainer.py               三阶段训练循环 + 动态退火 + 评估
│   └── augmentations.py         数据增强 (未接入训练流程)
├── ansys/
│   ├── generate_3d_test.py      ANSYS MAPDL 数据生成 (凹槽工件)
│   ├── data/                    train/val/test.h5
│   └── mesh_viz/                网格截图 + FRF 可视化
└── sample/
    ├── run_validation.py        训练入口 + 超参数配置
    ├── evaluate.py              评估 + 保存 final_results.npz
    ├── 对比图.py                预测 vs 真实对比图 (幅值+实部+虚部+振型)
    ├── 对比图db.py              dB 对比图 (真实 vs 预测分列)
    ├── 测试.py                  查看原始 FRF 数据
    └── output/                  checkpoint + 图表 + npz
```

## 2. 架构

```
输入: 6ch 物理场图像 [B,6,60,160] + query_coords [N,2]
                    │
    ┌───────────────┴───────────────┐
    │  ResNet+SE Encoder             │
    │  Stem(64) + 3×ResSEBlock      │
    │  f1=64 → f2=128 → f3=256      │
    │  → f4=512 → latent [B,512]    │
    │  含 U-Net 跳连 (f1~f4)        │
    └───────────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────┴────┐          ┌─────┴──────┐
    │OmegaHead│          │MicroDecoder│
    │→ ω[B,K]│          │→ mode_maps │ [B,K,60,160]
    │ZetaHead │          │    ↓        │
    │→ ζ[B,K]│          │ PhiScaleHead│ (形数解耦)
    └────┬────┘          │→ mode_maps │ × scale
         │               │    ↓        │
         │               │ grid_sample │
         │               │ → φ [N,K]   │
         │               └─────┬──────┘
         │                     │
    ┌────┴─────────────────────┴──────┐
    │  PhysicsDecoder (无参数)         │
    │  H=Σφ_kφ_k/(ω_k²-ω²+j2ζ_kω_kω) │
    │  → FRF (N,F,2)                  │
    └─────────────────────────────────┘
```

### 形数解耦 (Shape-Amplitude Decoupling)

- **MicroDecoder (UNet)**: 输出纯形状 (unit std 归一化)，专注空间分布
- **PhiScaleHead (MLP)**: 从 latent 预测物理幅值标量 `scale [B,K]`，专注模态质量

```
mode_maps_raw = MicroDecoder(latent, skips)
normalized = mode_maps_raw / std  (unit std)
scale = exp(PhiScaleHead(latent))
final_maps = normalized × scale
```

## 3. 数据

### ANSYS 凹槽工件

| 参数 | 值 |
|------|-----|
| 工件尺寸 | 160×60×10mm (固定), 铝7075 |
| 材料范围 | E±5% (Sobol序列), ρ±3% |
| 凹槽布局 | 5凹槽(4×3格局) / 6凹槽(4×3) / 7凹槽(5×3), 等概率 |
| 凹槽加工 | 随机选择1~N个凹槽, 每个深度30~60%×H (3~6mm) |
| 结构筋/边界 | 6mm 绝对宽度 |
| 装夹 | 4角螺栓 (XYZ三向弹簧) + 3侧顶杆 (Y向弹簧) |
| 弹簧刚度 K | 角点 K_c∈[5e6,1e8] N/m, 侧面 K_s∈[1e6,3e7] N/m |
| 弹簧阻尼 C | C=2ζ√(K·M_ref), ζ_joint∈[0.005,0.05], M_ref=0.01kg |
| 材料阻尼 | ζ_material=0.002 |
| 总阻尼 | ζ_k = 0.002 + Σ(C·φ²)/(2ω_k) (三维耗散求和) |
| 网格 | SOLID187 四面体, 6mm, 自由划分 |
| 模态求解 | Block Lanczos (LANB), 质量归一化 (nrmkey=ON) |
| 模态方向 | 前3阶, Z向面外振型 |
| 激励点 | 凹槽底面几何中心最近节点, Z向激励 |
| 频率网格 | 60点, 每峰±3×半功率带宽线性采样 (≥15点), 间隙对数补点 |

### 2.5D 投影物理场 (dataset.py 在线生成)

HDF5 存储原始 3D 节点数据。`__getitem__` 时投影为 6 通道 160×60 像素图像：

| 通道 | 内容 | 投影方式 |
|------|------|---------|
| Ch0 | 局部厚度 Z/H | scatter_reduce(max) 每像素 |
| Ch1 | is_fixed 边界条件 | 0/0.5/1.0 |
| Ch2 | log10(K) 弹簧刚度 | 仅弹簧节点 |
| Ch3 | log10(C) 弹簧阻尼 | 仅弹簧节点 |
| Ch4 | E/E_base 弹性模量 | 全局常数 |
| Ch5 | ρ/ρ_base 密度 | 全局常数 |

坐标归一化到 [-1,1] (X/0.160×2-1, Y/0.060×2-1)。
数据划分: 1200 训练 / 150 验证 / 150 测试。

## 4. 训练

### 三阶段 + Teacher-Forced Omega

| 阶段 | Epoch | 策略 | 损失 | 目的 |
|------|-------|------|------|------|
| 1: modal warmup | 0 ~ phase2_min_epoch | ω/ζ/φ 各自逼近真值 | modal_loss | ω 收敛到 <1%, MAC > 0.95 |
| 2: Teacher-Forced FRF | unlock ~ teacher_anneal | ω_used = α·ω_true + (1-α)·ω_pred, α: 1.0→0.0 | modal + FRF(warmup) | 峰位对齐时打磨 φ/ζ, 不背频率误差的锅 |
| 3: 端到端 | teacher_α=0 后 | 完全自主预测 | modal + FRF | 真实推理场景 |

**Teacher-Forced Omega 原理**: 共振峰极窄，ω 偏 1% 就会导致峰位错开、FRF Loss 爆炸。使用 ω_true 混合后，FRF Loss 只教 φ 调幅值、ζ 调宽度，各司其职。

### 超参数

| 参数 | 值 |
|------|-----|
| 模型 | UNetPhysicsModel (~52.8M params) |
| Encoder | ImprovedCNNEncoder (ResNet+SE, 64→128→256→512) |
| hidden / n_modes | 512 / 3 |
| dropout | 0.1~0.2 |
| 损失权重 | omega=1.0, zeta=10.0 (前40轮=0), phi=3.0 |
| FRF 损失 | dB MSE + 10×CDF L1, weight=0.5 (warmup) |
| optimizer | AdamW, lr=1e-3, wd=3e-4, CosineAnnealingLR |
| 梯度裁剪 | encoder=3.0, micro=5.0, omega=2.0, zeta=2.0, phi_refiner=2.0, phi_scale_head=2.0 |
| batch_size | 8 |

### 损失函数

| 损失项 | 空间 | 公式 |
|--------|------|------|
| ω | Hz 空间 | smooth_l1(f_pred, f_true) + 0.1×峰值敏感项 |
| ζ | log 空间 | smooth_l1(log_ζ_pred, log_ζ_true) |
| φ (MSE) | 归一化 | MSE(φ_pred/std, φ_true/std) |
| φ (MAC) | 尺度无关 | 1 - MAC, 按图平均 |
| φ (Std) | log 空间 | smooth_l1(log(std_pred), log(std_true)) |
| FRF (dB) | dB 空间 | MSE(20·log10(amp_pred), 20·log10(amp_true)) |
| FRF (CDF) | 累积分布 | L1(cumsum(amp_norm_pred), cumsum(amp_norm_true)) |

### 训练监控指标

```
Epoch   50 | w=[1.8/2.3/2.2]% z=[7/22/24]% φn=[3.9/30.9/43.5]% φa=[3.9/120.0/147.9]% MAC=[0.998/0.963/0.908] | w63% z0% ph37% | loss=69.9
```

- **w%**: 频率相对误差 (越小越好)
- **z%**: 阻尼相对误差 (越小越好)
- **φn%**: 振型 NRMSE (越小越好，形状+幅值综合)
- **φa%**: 振型范数幅值误差 (越小越好，致命指标)
- **MAC**: 模态置信准则 (越接近 1 越好，纯形状)

## 5. 快速开始

```bash
# 生成数据 (需 ANSYS MAPDL license)
F:/pytorch_cuda12/python.exe ansys/generate_3d_test.py

# 查看原始 FRF
F:/pytorch_cuda12/python.exe sample/测试.py

# 训练
F:/pytorch_cuda12/python.exe sample/run_validation.py

# 评估
F:/pytorch_cuda12/python.exe sample/evaluate.py

# 对比图
F:/pytorch_cuda12/python.exe sample/对比图.py
F:/pytorch_cuda12/python.exe sample/对比图db.py
```
