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
| 工件尺寸 | 160×60×10mm (铝7075, E=71.7GPa, ρ=2810) |
| 凹槽 | 5/6/7 方案, 深度随机 30-60% 厚度 |
| 装夹 | 4角 XYZ 弹簧 + 3侧面 Y 向弹簧 (COMBIN14) |
| 弹簧刚度 | K∈[1e6, 1e8] N/m, 随机 |
| 弹簧阻尼 | C∈[10, 500] N·s/m, 随机 |
| 阻尼 | ζ=材料0.002 + 边界耗散 (物理公式) |
| 模态 | 前3阶, Z向振型 |
| 网格 | 6mm 自由四面体 (SOLID187), ~4k 节点/样本 |
| 频率 | 80点自适应网格 |

### HDF5 格式 (per-sample-group)

```
/sample_0/
├── points         (N₀, 3)       节点坐标 [x, y, z] (m)
├── point_frf      (N₀, F₀, 2)  复数频响函数 [Re, Im]
├── frequencies    (F₀,)         频率采样点 (Hz)
├── point_features (N₀, 7)       逐节点特征:
│                                [E/E_base, PRXY, ρ/ρ_base, is_fixed, log10(K), log10(C), Z/H]
├── modal_omega    (K,)          固有圆频率 (rad/s)
├── modal_zeta     (K,)          阻尼比
├── modal_phi      (N₀, K)       模态振型 φ_k(x)
└── modal_phi_exc  (K,)          激励点振型值
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
