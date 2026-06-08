# geometric_frf_groove — MeshGraphNet 模态 FRF 预测

本项目已从旧的 2.5D CNN-UNet 方案迁移为 **GNN / MeshGraphNet + 物理模态叠加层**。

当前主流程：

```text
3D 几何 + ANSYS FE mesh + 材料 + 装夹边界
        ↓
GraphHDF5Dataset
        ↓
MeshGraphFRFModel
        ↓
ω, ζ, φ_z
        ↓
PhysicsDecoder 模态叠加
        ↓
FRF(Re, Im)
```

CNN/UNet 模块已废弃，不再作为训练、评估或预测入口。

---

## 1. 目录结构

```text
├── ansys/
│   ├── generate_3d_test.py       ANSYS MAPDL 数据生成，直接导出图数据
│   ├── data/                     train.h5 / val.h5 / test.h5
│   └── mesh_viz/                 网格与 FRF 可视化
├── data/
│   ├── __init__.py
│   └── dataset.py                GraphHDF5Dataset + 图 batch collate
├── models/
│   ├── __init__.py
│   ├── frf_model.py              build_geometric_model()
│   ├── meshgraphnet_frf_model.py MeshGraphNet 主模型
│   ├── physics_decoder.py        模态叠加 FRF 物理解码器
│   └── geometry_data.py          可选图数据容器
├── training/
│   ├── __init__.py
│   ├── losses.py                 modal_loss + frf_loss
│   ├── trainer.py                两阶段训练循环
│   └── augmentations.py          可选图 batch 增强
└── sample/
    ├── run_validation.py         训练入口
    ├── evaluate.py               模态/FRF 评估并保存 final_results.npz
    ├── predict.py                checkpoint 推理
    ├── 测试.py                   原始 FRF 数据可视化
    ├── 对比图.py                 预测 vs 真实对比图
    └── output/                   checkpoint、日志和 npz 结果
```

---

## 2. 数据生成

运行：

```bash
F:/pytorch_cuda12/python.exe ansys/generate_3d_test.py
```

生成：

```text
ansys/data/train.h5
ansys/data/val.h5
ansys/data/test.h5
```

### 保持不变的数据分布

| 参数 | 值 |
|---|---|
| 工件尺寸 | 160×60×10mm，铝7075 |
| 材料范围 | E±5%，ρ±3%，Sobol 序列 |
| 凹槽布局 | 5凹槽 / 6凹槽 / 7凹槽，等概率 |
| 凹槽加工 | 随机选择 1~N 个凹槽，每个深度 30~60%×H |
| 结构筋/边界 | 6mm 绝对宽度 |
| 装夹 | 4角螺栓 XYZ 三向弹簧 + 3侧顶杆 Y 向弹簧 |
| 弹簧刚度 | 角点 Kc∈[5e6,1e8] N/m；侧面 Ks∈[1e6,3e7] N/m |
| 弹簧阻尼 | C=2ζ√(K·M_ref)，ζ_joint∈[0.005,0.05]，M_ref=0.01kg |
| 材料阻尼 | ζ_material=0.002 |
| 总阻尼 | ζ_k=0.002+Σ(Cxφx²+Cyφy²+Czφz²)/(2ω_k) |
| 网格 | SOLID187 四面体，6mm，自由划分 |
| 模态求解 | Block Lanczos，质量归一化 nrmkey=ON |
| 模态阶数 | 前3阶 |
| 激励点 | 凹槽底面几何中心最近节点，Z向激励 |
| 频率网格 | 60点，每峰±3×半功率带宽线性采样，间隙对数补点 |

### 新 HDF5 字段

```text
/sample_i/
├── points              (N, 3)       节点坐标，m
├── edge_index          (2, E)       FE mesh 拓扑边
├── edge_attr           (E, 4)       [dx/L, dy/W, dz/H, length]
├── point_features      (N, 7)       旧物理特征，兼容保留
├── spring_k_xyz        (N, 3)       每节点三向弹簧刚度
├── spring_c_xyz        (N, 3)       每节点三向弹簧阻尼
├── node_type           (N,)         0普通/1槽底/2切削区/3侧顶杆/4角点夹持
├── pocket_bottom_mask  (N,)         凹槽底面节点 mask
├── cut_region_mask     (N,)         切削区节点 mask
├── point_frf           (N, F, 2)    复数 FRF [Re, Im]
├── frequencies         (F,)         频率 Hz
├── modal_omega         (K,)         固有圆频率 rad/s
├── modal_zeta          (K,)         阻尼比
├── modal_phi           (N, K)       Z向振型
├── modal_phi_xyz       (N, K, 3)    XYZ三向振型
├── modal_phi_exc       (K,)         激励节点 Z向振型
├── excitation_index    scalar       激励节点索引
└── excitation_coord    (3,)         激励点坐标
```

---

## 3. 模型架构

```text
node_features + edge_index + edge_attr + batch
        ↓
node encoder / edge encoder
        ↓
MeshGraphNet message passing blocks
        ↓
全局池化
        ↓
┌────────────────────────┬────────────────────────┐
│ global modal head      │ node modal head         │
│ ω1,ω2,ω3               │ φ_z(node, mode)         │
│ ζ1,ζ2,ζ3               │                         │
└────────────────────────┴────────────────────────┘
        ↓
PhysicsDecoder
        ↓
FRF(node, frequency, Re/Im)
```

模型入口：

```python
from models import build_geometric_model
net = build_geometric_model(
    encoder_kwargs={
        'node_in_dim': 10,
        'edge_in_dim': 4,
        'hidden': 256,
        'n_layers': 8,
        'n_modes': 3,
        'omega_max': 25000.0,
        'amp_scale': 500000.0,
        'freq_min': 1.0,
        'freq_max': 5000.0,
    },
    decoder_kwargs={},
)
```

---

## 4. 训练

运行：

```bash
F:/pytorch_cuda12/python.exe sample/run_validation.py
```

训练入口使用：

```text
GraphHDF5Dataset → collate_geometry_batch → MeshGraphFRFModel → trainer.train
```

两阶段训练：

| 阶段 | 目标 | 损失 |
|---|---|---|
| Phase 1 | 先学习模态参数 | Lω + Lζ + Lφ |
| Phase 2 | 加入 FRF 物理重建 | Lω + Lζ + Lφ + LFRF |

输出：

```text
sample/output/checkpoint_last
sample/output/checkpoint_best
sample/output/loss_log.csv
```

---

## 5. 评估、预测和画图

```bash
# 查看原始真实 FRF
F:/pytorch_cuda12/python.exe sample/测试.py

# 评估 checkpoint，生成 final_results.npz
F:/pytorch_cuda12/python.exe sample/evaluate.py

# 推理，生成 predictions.npz
F:/pytorch_cuda12/python.exe sample/predict.py

# 预测 vs 真实对比图
F:/pytorch_cuda12/python.exe sample/对比图.py
```

---

## 6. 当前推荐工作流

```text
1. 运行 ansys/generate_3d_test.py 重新生成 train/val/test.h5
2. 运行 sample/run_validation.py 训练 MeshGraphNet
3. 运行 sample/evaluate.py 计算模态和 FRF 指标
4. 运行 sample/对比图.py 生成论文用对比图
```

旧 CNN 数据投影和 UNet 训练路径已移除。当前仓库以 MeshGraphNet/GNN 为主线。