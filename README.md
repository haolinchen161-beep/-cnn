# geometric_frf_groove — 基于几何的频响函数预测

输入 3D 几何 + 边界条件 → UNetPhysicsModel (2.5D CNN-UNet + PhysicsDecoder) → 模态参数 (ω, ζ, φ) → 物理公式重建 FRF。

## 1. 目录结构

```
├── models/
│   ├── unet_physics_model.py   主模型: 2.5D CNN-UNet
│   ├── frf_model.py             模型工厂 build_geometric_model()
│   ├── physics_decoder.py       无参数物理解码器 (模态叠加→FRF)
│   ├── geometry_data.py         GeometryData 数据容器
├── data/
│   └── dataset.py               HDF5 数据集 (per-sample-group) + collate
├── training/
│   ├── losses.py                modal_loss + frf_loss
│   ├── trainer.py               两阶段训练循环 + 评估
│   └── augmentations.py         数据增强 (坐标/特征噪声, 节点dropout, 频率子采样)
├── ansys/
│   ├── generate_3d_test.py      ANSYS MAPDL 数据生成 (凹槽工件)
│   ├── data/                    train/val/test.h5
│   └── mesh_viz/                网格截图
└── sample/
    ├── run_validation.py        训练入口
    ├── evaluate.py              评估 + 保存 final_results.npz
    ├── 测试.py                  查看原始 FRF
    ├── 对比图.py                预测 vs 真实对比图
    ├── predict.py               推理
    └── output/                  checkpoint + 图表 + npz
```

## 2. 架构

```
输入: 6ch 物理场图像 [B,6,60,160] + query_coords [N,2]
                    │
    ┌───────────────┴───────────────┐
    │  CNN Encoder (4层Conv+UNet跳连) │ → latent [B,512]
    └───────────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────┴────┐          ┌─────┴──────┐
    │Macro MLP│          │Micro UNet  │
    │→ ω[B,K]│          │→ mode_maps │ [B,K,60,160]
    │→ ζ[B,K]│          │   grid_sample(query_coords)
    └────┬────┘          │   → φ [N,K]│
         │               └─────┬──────┘
         │                     │
    ┌────┴─────────────────────┴──────┐
    │  PhysicsDecoder (无参数)         │
    │  H=Σφ_kφ_k/(ω_k²-ω²+j2ζ_kω_kω) │
    │  → FRF (N,F,2)                  │
    └─────────────────────────────────┘
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
| 弹簧刚度 K | 角点 K_c∈[5e6,1e8] N/m, 侧面 K_s∈[1e6,3e7] N/m, 每个装夹区独立采样 |
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

### HDF5 格式 (per-sample-group)

```
/sample_0/
├── points         (N₀, 3)        3D 节点坐标 [x, y, z] (m)
├── point_frf      (N₀, F₀, 2)   复数 FRF [Re, Im] (Z向, 物理空间)
├── frequencies    (F₀,)          频率采样点 (Hz)
├── point_features (N₀, 7)        逐节点: [E/E_base, PRXY, ρ/ρ_base, is_fixed, logK, logC, Z/H]
├── modal_omega    (K,)           固有圆频率 (rad/s)
├── modal_zeta     (K,)           阻尼比
├── modal_phi      (N₀, K)        Z向模态振型
└── modal_phi_exc  (K,)           激励点振型值
```

## 4. 训练

### 两阶段策略

| 阶段 | Epoch | 动作 | 损失 | 目的 |
|------|-------|------|------|------|
| 1: modal warmup | dynamic unlock | sorted loss + phi x 100 | modal_loss | omega < 0.5%% or epoch > 1000 |
| 2: FRF joint | unlock ~ 2000 | FRF warmup x 0.05 | modal + FRF(warmup) | damping anneal alpha: 10 -> 1 |

### 超参数

| 参数 | 值 |
|------|-----|
| 模型 | UNetPhysicsModel (~6.5M params) |
| hidden / n_modes | 512 / 3 |
| Conv layers / UNet skips | 4 / 3 |
| dropout | 0.2 |
| loss weights | rel_omega x 200 + rel_zeta x 10 + phi signMSE x 100 + FRF dB x 1 + CDF x 10 |
| optimizer | AdamW, lr=1e-3, wd=3e-4, CosineAnnealingLR |
| 梯度裁剪 | encoder=3.0, micro=5.0, macro=2.0 |
| batch_size | 8 |

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
```
