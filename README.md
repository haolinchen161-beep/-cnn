# Mesh Modal Lite：轻量 MeshGraphNet 模态预测

本分支 `mesh-modal-lite` 是在旧的 `gnn-meshgraphnet-refactor` 基础上重写的轻量版本。

目标从复杂的 “ω / ζ / φ / FRF 多任务训练” 简化为第一阶段最关键的问题：

```text
ANSYS 生成的 FE mesh + 几何/材料/装夹刚度特征
        ↓
MeshGraphNet
        ↓
前 3 阶固有圆频率 ω1,ω2,ω3
全节点三向质量归一化振型 φ_xyz
```

当前版本**不训练阻尼 ζ，不重建 FRF，不使用 PhysicsDecoder，不使用 ZetaHead，不使用 DirectionBranchHead，也不做多阶段 FRF 微调**。  
阻尼和 FRF 后续可以在模态预测稳定后再作为独立物理层加入。

---

## 1. 数据集

直接使用现有 `ansys/generate_3d_test.py` 生成的 HDF5：

```text
ansys/data/train.h5
ansys/data/val.h5
ansys/data/test.h5
```

每个样本至少需要：

```text
points                (N, 3)
edge_index            (2, E)
edge_attr             (E, 4)
point_features        (N, 7)
spring_k_xyz          (N, 3)
node_type             (N,)
pocket_bottom_mask    (N,)
cut_region_mask       (N,)
local_thickness_ratio (N,)
pocket_depth_ratio    (N,)
excitation_index      scalar
excitation_coord      (3,)
modal_omega           (3,)
modal_phi_xyz         (N, 3, 3)
```

`modal_zeta`、`spring_c_xyz`、`point_frf`、`frequencies` 可以存在于 HDF5 中，但本分支训练时会忽略它们。

---

## 2. 节点特征

`GraphHDF5Dataset` 构建 21 维节点特征：

| 维度 | 含义 |
|---|---|
| 0:3 | 归一化坐标 `x/L,y/W,z/H -> [-1,1]` |
| 3:7 | `E/E0, rho/rho0, PRXY, local_thickness_ratio` |
| 7:10 | `log10(1 + spring_k_xyz) / 8` |
| 10 | 是否存在弹簧刚度 |
| 11:16 | `node_type` one-hot：普通/槽底/切削区/浮动装夹/角点装夹 |
| 16 | `pocket_bottom_mask` |
| 17 | `cut_region_mask` |
| 18 | `pocket_depth_ratio` |
| 19 | 激励点 flag |
| 20 | 到激励点的归一化距离 |

阻尼相关输入不再进入节点特征。

---

## 3. 模型结构

```text
node_features + edge_index + edge_attr + batch
        ↓
node encoder / edge encoder
        ↓
MeshGraphNet message passing blocks
        ↓
global mean pool + global max pool
        ↓
OmegaHead      →  ω1,ω2,ω3
PhiDecoder     →  φ_xyz [total_N, 3, 3]
```

保留并简化旧分支中有用的部分：

```text
1. 变节点数 graph batch 拼接
2. MeshGraphNet residual message passing
3. 每图每模态符号对齐
4. 3D MAC 指标
5. 振型标准差一致性约束
6. 关键区域加权振型损失
```

删除旧分支中当前不需要的部分：

```text
1. ZetaHead
2. PhysicsDecoder / FRF 重建
3. DirectionBranchHead
4. PhiScaleHead
5. FRF loss / branch loss / zeta loss
6. 多阶段训练 schedule
```

---

## 4. 训练

运行：

```bash
python -u sample/run_validation.py
```

默认配置：

```text
data_dir = ansys/data
epochs = 200
batch_size = 1
hidden = 128
n_layers = 6
lr = 3e-4
```

输出目录：

```text
sample/output_mesh_modal_lite/
├── checkpoint_last.pt
├── checkpoint_best.pt
├── loss_log.csv
└── final_results.npz
```

---

## 5. 评估

训练结束会自动在 test 集上评估。也可以单独运行：

```bash
python -u sample/evaluate.py
```

主要指标：

```text
freq_mae_hz
freq_mape_percent
phi_nrmse
mac_mode1 / mac_mode2 / mac_mode3
```

其中 MAC 和振型误差会先做逐图逐模态符号对齐，因此不会把整体反号的正确振型误判为错误。

---

## 6. 当前研究定位

本分支只解决第一阶段：

```text
几何 + 装夹刚度边界 → 固有频率 + 全节点三向振型
```

建议后续路线：

```text
阶段 1：训练本分支，确认 ω 和 φ 可预测
阶段 2：用预测 φ 在装夹面计算阻尼 ζ
阶段 3：用 ω,ζ,φ 显式模态叠加重建 FRF
阶段 4：必要时加入 FRF 物理损失或 sparse K/M 物理残差
```
