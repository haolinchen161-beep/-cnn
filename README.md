# MeshGraphNet 模态参数与 FRF 预测

本分支 `gnn-meshgraphnet-refactor` 是面向论文最终实验的 GNN / MeshGraphNet 主线。

目标不是直接黑箱预测 FRF，而是保持当前研究路线：

```text
ANSYS 生成真实数据
3D 几何 + FE mesh + 材料 + 装夹边界
        ↓
MeshGraphNet
        ↓
模态参数: ω, ζ, φ_xyz
        ↓
PhysicsDecoder 模态叠加
        ↓
Z 向 FRF(Re, Im)
```

当前分支已经废弃早期 Z-only GNN 写法，改为完整三维振型形式：

```text
phi = [total_N, K, 3]
K = 3
```

FRF 重建仍与 ANSYS 数据生成一致，只使用 Z 向响应：

```text
phi_z = phi[..., 2]
```

---

## 1. 当前目录结构

```text
├── ansys/
│   ├── generate_3d_test.py        质量归一化 ANSYS 数据生成脚本
│   ├── stratified_resplit_h5.py   按 mode3 主方向临时分层划分 train/val/test
│   ├── data/                      train.h5 / val.h5 / test.h5
│   └── mesh_viz/                  网格与 FRF 可视化
├── data/
│   ├── __init__.py
│   └── dataset.py                 GraphHDF5Dataset + graph batch collate
├── models/
│   ├── __init__.py
│   ├── frf_model.py               build_geometric_model()
│   ├── meshgraphnet_frf_model.py  MeshGraphNet 主模型
│   └── physics_decoder.py         模态叠加 FRF 物理解码器
├── training/
│   ├── __init__.py
│   ├── losses.py                  modal_loss + branch_loss + frf_loss
│   └── trainer.py                 分阶段训练循环
└── sample/
    ├── run_validation.py          训练入口
    ├── evaluate.py                评估并保存 final_results.npz
    └── output_meshgraphnet/       checkpoint、loss_log、npz 结果
```

---

## 2. 数据生成

运行：

```bash
F:/pytorch_cuda12/python.exe -u ansys/generate_3d_test.py
```

默认生成：

```text
N_SAMPLES = 300
N_TRAIN   = 240
N_VAL     = 30
N_TEST    = 30
```

输出：

```text
ansys/data/train.h5
ansys/data/val.h5
ansys/data/test.h5
ansys/data/sample_log.csv
```

可以用环境变量临时修改样本数：

```bash
set N_SAMPLES=600
set N_TRAIN=480
set N_VAL=60
set N_TEST=60
F:/pytorch_cuda12/python.exe -u ansys/generate_3d_test.py
```

---

## 3. 数据生成的物理设置

当前生成脚本采用更适合最终论文数据的设置：

| 项目 | 当前设置 |
|---|---|
| 工件尺寸 | 160×60×10mm，铝7075 |
| 材料范围 | E±5%，ρ±3%，Sobol 序列 |
| 凹槽布局 | 5凹槽 / 6凹槽 / 7凹槽，等概率 |
| 凹槽加工 | 随机选择 1~N 个凹槽，每个深度 30%~60%×H |
| 结构筋/边界 | 6mm 绝对宽度 |
| 网格 | SOLID187 四面体，6mm，自由划分 |
| 装夹 | 4角螺栓 XYZ 三向弹簧 + 3侧顶杆 Y 向弹簧 |
| 弹簧刚度 | 角点 Kc∈[5e6,1e8] N/m；侧面 Ks∈[1e6,3e7] N/m |
| 弹簧阻尼 | C=2ζ√(K·M_ref)，ζ_joint∈[0.005,0.05]，M_ref=0.01kg |
| 材料阻尼 | ζ_material=0.002 |
| 质量矩阵 | 默认一致质量矩阵，`USE_LUMPED_MASS = False` |
| 模态归一化 | 质量归一化，`USE_MASS_NORMALIZATION = True`，`nrmkey='OFF'` |
| 模态阶数 | 前 3 阶 |
| 激励点 | 凹槽底面/切削区中心最近节点，Z 向激励 |
| 频率网格 | 60 点，每峰附近加密，间隙对数补点 |

质量归一化后，当前阻尼和 FRF 公式更物理一致：

```text
ζ_k = ζ_material + Σ(Cx φx² + Cy φy² + Cz φz²) / (2ω_k)

H_z(x, xf, ω) = Σ φz_k(x) φz_k(xf) /
                (ω_k² - ω² + j 2ζ_kω_kω)
```

代码中仍乘以 `AMPLITUDE_SCALE = 500000.0`，用于保持 FRF 数值量级与训练稳定性。

---

## 4. HDF5 字段

每个样本结构：

```text
/sample_i/
├── points                  (N, 3)       节点物理坐标，m
├── edge_index              (2, E)       FE mesh 拓扑边
├── edge_attr               (E, 4)       [dx/L, dy/W, dz/H, length]
├── point_features          (N, 7)       [E/E0, PRXY, rho/rho0, is_fixed, logK, logC, local_thickness/H]
├── spring_k_xyz            (N, 3)       每节点三方向弹簧刚度
├── spring_c_xyz            (N, 3)       每节点三方向弹簧阻尼
├── node_type               (N,)         0普通/1槽底/2切削区/3侧顶杆/4角点夹持
├── pocket_bottom_mask      (N,)         凹槽底面节点 mask
├── cut_region_mask         (N,)         切削区节点 mask
├── local_thickness_ratio   (N,)         局部残余厚度 / H
├── pocket_depth_ratio      (N,)         当前 XY 区域加工深度 / H
├── point_frf               (N, F, 2)    Z 向复数 FRF [Re, Im]
├── frequencies             (F,)         频率 Hz
├── modal_omega             (K,)         固有圆频率 rad/s
├── modal_zeta              (K,)         阻尼比
├── modal_phi               (N, K, 3)    XYZ 三向振型，兼容字段
├── modal_phi_xyz           (N, K, 3)    XYZ 三向振型，训练优先使用
├── modal_phi_exc           (K, 3)       激励节点 XYZ 三向振型
├── modal_mass              (K,)         质量归一化下为 1
├── modal_stiffness         (K,)         ω² × modal_mass
├── modal_effm              (K, 3)       ANSYS 有效质量
├── modal_pfact             (K, 3)       ANSYS 参与系数
├── excitation_index        scalar       激励节点索引
└── excitation_coord        (3,)         激励点坐标
```

不额外保存 `mode_type` 或 `direction_energy_ratio`。它们可由 `modal_phi_xyz` 临时计算，只用于诊断或分层划分，不作为真实物理输入字段。

---

## 5. 分层划分 train / val / test

生成脚本默认先按顺序切分。由于之前诊断显示三阶模态存在 X/Y/Z 分型不均衡，建议生成完成后运行：

```bash
F:/pytorch_cuda12/python.exe -u ansys/stratified_resplit_h5.py --in-place
```

该脚本会：

```text
1. 从 modal_phi_xyz 临时计算 mode3 主方向；
2. 按 mode3 的 X/Y/Z 主方向重新分层划分 train/val/test；
3. 备份原始顺序切分文件为 *.sequential.bak；
4. 重写 train.h5 / val.h5 / test.h5。
```

脚本不会把 `mode_type` 写进 HDF5。

---

## 6. 模型架构

```text
node_features + edge_index + edge_attr + batch
        ↓
node encoder / edge encoder
        ↓
MeshGraphNet message passing blocks
        ↓
global mean pool + global max pool
        ↓
PhysicsPriorOmegaHead       →  ω1,ω2,ω3
ZetaHead                    →  ζ1,ζ2,ζ3
DirectionBranchHead         →  每阶 XYZ 分支概率 [B,K,3]
ModeTokenPhiDecoder         →  φ_raw [N,K,3]
PhiScaleHead                →  每阶每方向 scale [B,K,3]
        ↓
φ_xyz [N,K,3]
        ↓
PhysicsDecoder 使用 φ_z 重建 FRF
```

模型入口：

```python
from models import build_geometric_model
from data.dataset import NODE_FEATURE_DIM

net = build_geometric_model(
    encoder_kwargs={
        "node_in_dim": NODE_FEATURE_DIM,  # 当前为 26
        "edge_in_dim": 4,
        "hidden": 128,
        "n_layers": 4,
        "n_modes": 3,
        "amp_scale": 500000.0,
        "freq_min": 1.0,
        "freq_max": 5000.0,
        "dropout": 0.05,
    },
    decoder_kwargs={},
)
```

---

## 7. 节点特征维度说明

`GraphHDF5Dataset` 当前构建 **26 维**节点特征：

| 维度范围 | 含义 |
|---|---|
| 0:3 | 归一化 xyz 坐标 |
| 3:10 | point_features = [E/E0, PRXY, rho/rho0, is_fixed, logK, logC, local_thickness/H] |
| 10:13 | 归一化 log spring_k_xyz |
| 13:16 | 归一化 log spring_c_xyz |
| 16:21 | node_type one-hot，5 类 |
| 21 | pocket_bottom_mask |
| 22 | cut_region_mask |
| 23 | pocket_depth_ratio |
| 24 | excitation_flag |
| 25 | 归一化到激励点距离 |

这个改动把凹槽深度作为显式节点输入，符合当前样本生成形式，也更有利于频率和振型预测。

---

## 8. 训练

运行：

```bash
F:/pytorch_cuda12/python.exe -u sample/run_validation.py
```

训练输出目录：

```text
sample/output_meshgraphnet/
├── checkpoint_last
├── checkpoint_best
├── checkpoint_best_modal
├── loss_log.csv
└── final_results.npz
```

当前训练阶段：

| 阶段 | 默认 epoch | 目标 |
|---|---:|---|
| Phase0a | 0~19 | 只训练 omega prior MLP |
| Phase0b | 20~39 | 解锁图编码器和 omega delta，仍只训频率 |
| Phase1 | 40~159 | 训练 ω / ζ / φ / branch |
| Phase2a | 160~199 | 冻结 φ 相关模块，弱 FRF 调 ω/ζ |
| Phase2b | 200~299 | 全模型弱 FRF 联调 |

默认配置在 `sample/run_validation.py` 中：

```text
hidden = 128
n_layers = 4
batch_size = 1
epochs = 300
fp16 = True
frf_loss_weight = 0.005
```

---

## 9. 评估

运行：

```bash
F:/pytorch_cuda12/python.exe -u sample/evaluate.py
```

默认加载：

```text
sample/output_meshgraphnet/checkpoint_best_modal
```

输出：

```text
sample/output_meshgraphnet/final_results.npz
```

主要保存：

```text
pred_freq_hz / true_freq_hz
pred_zeta / true_zeta
pred_phi / true_phi
phi_mac / phi_nrmse / phi_a
predicted_frf / target_frf
peak_shift_hz / peak_amp_rel
```

---

## 10. 推荐完整工作流

```text
1. git checkout gnn-meshgraphnet-refactor
2. F:/pytorch_cuda12/python.exe -u ansys/generate_3d_test.py
3. F:/pytorch_cuda12/python.exe -u ansys/stratified_resplit_h5.py --in-place
4. 删除旧 checkpoint 和 loss_log.csv
5. F:/pytorch_cuda12/python.exe -u sample/run_validation.py
6. F:/pytorch_cuda12/python.exe -u sample/evaluate.py
```

需要删除的旧训练文件：

```text
sample/output_meshgraphnet/checkpoint_last
sample/output_meshgraphnet/checkpoint_best
sample/output_meshgraphnet/checkpoint_best_modal
sample/output_meshgraphnet/loss_log.csv
```

当前分支以 MeshGraphNet + 质量归一化模态参数 + 物理 FRF 重建为主线。
