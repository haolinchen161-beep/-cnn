# Transolver Modal-FRF — 基于 ANSYS 网格的模态参数预测

当前分支：`transolver-modal-dataset`

本分支已从旧的 CNN / UNet / 2.5D 图像投影路线迁移为 **非结构化 mesh Transolver + 模态叠加物理层**。

目标不是直接黑箱预测所有节点的 FRF，而是学习：

```text
3D 几何 + 材料参数 + 装夹边界 + 激励点/刀触点 + 查询节点
        ↓
TransolverModalFRF
        ↓
前三阶模态参数: ω, ζ, φ
        ↓
ModalFRFDecoder 物理模态叠加
        ↓
方向一致的 FRF: H_ab（a=响应方向, b=激励方向）
```

## 方向约定

项目已从硬编码 Z 向改造为**可配置的方向链路**。核心概念：

```text
H_ab(x, x_f, ω) = Σ_k φ_k^a(x)·φ_k^b(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·Ω)

其中:
  a = 响应方向（response direction），测量位移的方向
  b = 激励方向（force direction），施加力的方向
```

默认配置：**H_YY**（Y 向激励 + Y 向响应），适用于铣削切削平面方向。

通过 `--response-dir` 和 `--force-dir` 参数可切换为 X、Y、Z 任意组合。

---

## 1. 当前目录结构

```text
├── utils/
│   └── direction.py                 统一方向工具模块（DIRECTION_TO_INDEX 等）
│
├── ansys/
│   ├── generate_3d_test.py          ANSYS MAPDL boolean 数据生成（完成态，保留兼容）
│   ├── generate_ekill_process_dataset.py  EKILL 过程数据生成（推荐新路径）
│   ├── data/                        boolean 数据输出 (train.h5 / val.h5 / test.h5)
│   ├── data_ekill/                  ekill 数据输出
│   └── mesh_viz/                    ANSYS 网格与装夹/激励点可视化
│
├── data/
│   └── dataset.py                   TransolverModalDataset + collate_mesh_batch
│
├── models/
│   ├── transolver_modal_model.py    TransolverModalFRF 主模型（方向感知 + 振型加权池化）
│   ├── physics_decoder.py           ModalFRFDecoder，无参数方向性模态叠加 FRF 层
│   ├── frf_model.py                 build_geometric_model() 模型工厂
│   ├── geometry_data.py             TransolverMeshBatch 数据容器
│   └── __init__.py                  模型导出
│
├── training/
│   ├── losses.py                    modal_loss / frf_loss / total_loss（数值稳定版）
│   ├── trainer.py                   TransolverTrainer + train/evaluate 入口
│   └── augmentations.py            可选 mesh 节点/特征扰动
│
└── sample/
    ├── run_validation.py            训练入口（支持 --response-dir / --force-dir）
    ├── evaluate.py                  评估并保存 final_results.npz / eval_summary.txt
    ├── predict.py                   单样本推理（支持 --query-node / --tool-position）
    ├── 测试.py                      Transolver FRF 快速可视化
    ├── 对比图.py                    预测 vs ANSYS 真实 FRF 对比图
    └── output/                      checkpoint 与评估结果
```

---

## 2. 模型结构

当前主模型为：

```text
models/transolver_modal_model.py::TransolverModalFRF
```

整体流程：

```text
points(N,3) + transolver_point_features(N,C)
        │
        ├── optional mesh edge stem
        │       使用 element_node_indices 生成 edge_index
        │       可用 --no-edges 关闭
        │
        ├── SliceTransolverBlock × L
        │       节点 → learned physics-aware slices
        │       slice token attention
        │       slice → 节点回写
        │
        ├── global attention pooling
        │       → modal_omega: (B,K)
        │
        ├── mode-weighted boundary pooling (新)
        │       模态能量 + 边界强度 → 每模态独立 context
        │       → zeta_residual → modal_zeta: (B,K)
        │
        ├── node modal head
        │       → modal_phi_xyz: (total_N,K,3)
        │       → phi_response = phi_xyz[..., response_dir_index]
        │       → phi_force = phi_xyz[..., force_dir_index]
        │
        └── ModalFRFDecoder, optional during training/eval
                → point_frf: (total_N,F,2)  [H_ab]
```

### 2.1 方向感知设计

模型在 `__init__` 时接受 `response_direction` 和 `force_direction`（默认均为 `"Y"`），
内部自动映射为笛卡尔轴索引（X=0, Y=1, Z=2）。`forward()` 中：

- `phi_response = phi_xyz[..., self.response_dir_index]` — 响应方向振型
- `phi_force = phi_xyz[..., self.force_dir_index]` — 激励方向振型
- 解码器使用 `phi_response` 和 `phi_force_exc` 重建方向性 FRF

### 2.2 振型加权 zeta 残差

新设计的 `mode_weighted_pool()` 替代了旧的全局 zeta 残差预测头：

```text
modal_energy_i,k = Σ_d φ_xyz[i,k,d]²
boundary_strength_i = log1p(Σ_d |C_i,d|)
score_i,k = MLP([latent_i, modal_energy_i,k, boundary_strength_i])
           + log(modal_energy_i,k + ε)
           + 0.1 · boundary_strength_i
weight_i,k = softmax(score over nodes of same graph)
context_g,k = Σ_i weight_i,k · latent_i
zeta_residual_k = MLP(context_g,k)
zeta_k = zeta_phys_k · exp(0.1 · tanh(zeta_residual_k))
```

每个模态都有独立的 context，增强了局部装夹边界对阻尼的影响。

### 2.3 可选 mesh edge stem

数据生成器保存 ANSYS SOLID187 单元连接关系。模型中的 `GraphEdgeConv` 是可选局部几何 stem。
训练时可用 `--no-edges` 关闭。

### 2.4 可选 FRF 物理层与 FRF loss

`ModalFRFDecoder` 是无参数物理层，使用方向性模态叠加重建 FRF。
训练时可联合 FRF loss（默认启用）。

---

## 3. 数据生成

### 3.1 主路径：Boolean 完成态数据（推荐）

```bash
python ansys/generate_3d_test.py
```

特点：
- 每样本独立建模（block + vsbv 布尔挖槽 → vmesh），凹槽边界光滑
- 结构筋 + 外围边界保证几何合理性
- 输出目录：`ansys/data/`
- 模型本质上对拓扑不完全一致容忍度较高（Transolver 是点云 attention）

### 3.2 实验性路径：EKILL 过程数据

```bash
python ansys/generate_ekill_process_dataset.py
```

特点：
- 一次性 vmesh 完整长方体 → 每样本 EKILL 去除单元
- 固定拓扑，保存 `element_active_flag`、`node_active_flag` 等过程字段
- **注意**：6mm 粗四面体网格下 EKILL 边界锯齿严重，物理不可靠，仅作实验参考
- 输出目录：`ansys/data_ekill/`

### 3.3 方向与频率网格配置

两种生成器均支持：
- `RESPONSE_DIRECTION` / `FORCE_DIRECTION`：默认 Y/Y
- `FREQ_GRID_MODE`：`"fixed"`（固定网格，训练主路径） / `"adaptive"`（自适应峰网格） / `"both"`
- 固定网格确保 batch 内频率一致，训练统计更稳定

### 3.4 数据生成逻辑

| 项目 | 当前设置 |
|---|---|
| 工件尺寸 | 160 × 60 × 10 mm，铝 7075 |
| 材料扰动 | E ±5%，ρ ±3%，Sobol 序列 |
| 凹槽布局 | 5/6/7 凹槽，等概率 |
| 凹槽加工 | 随机选择 1~N 个凹槽，深度 30%~60% × H |
| 结构筋/边界 | 6 mm 绝对宽度 |
| 装夹 | 4 角螺栓 XYZ 三向弹簧 + 3 侧顶杆 Y 向弹簧 |
| 单元 | SOLID187 四面体，mesh size 6 mm |
| 模态求解 | Block Lanczos，质量归一化 |
| 模态阶数 | 前 3 阶 |
| 激励点 | 凹槽底面中心最近节点 |
| 频率网格 | 固定 128 点 hybrid（训练主路径）+ 自适应 60 点（可选） |

---

## 4. HDF5 数据格式

每个 HDF5 文件使用 per-sample group：

```text
/sample_0/
├── points                              (N,3)       节点坐标, m
├── point_features                      (N,7)       原始旧字段，保留兼容
├── transolver_point_features           (N,C)       当前模型主要输入特征
├── point_frf                           (N,F,2)     FRF [Re, Im]（固定网格）
├── frequencies                         (F,)        Hz（固定网格）
│
├── modal_omega                         (K,)        固有圆频率, rad/s
├── modal_zeta                          (K,)        阻尼比
├── modal_phi                           (N,K)       响应方向振型
├── modal_phi_exc                       (K,)        激励点激励方向振型
├── modal_phi_xyz                       (N,K,3)     XYZ 三向振型，当前主目标
├── modal_phi_exc_xyz                   (K,3)       激励点 XYZ 三向振型
│
├── boundary_k_xyz / boundary_c_xyz     (N,3)       节点三向弹簧刚度/阻尼
├── fixture_type                        (N,)        0 无装夹, 1 角点, 2 侧顶杆
│
├── element_node_indices               (Ne,M)       单元节点 0-based index
│
├── excitation_index / excitation_point              激励点信息
│
├── tool_position                       (3,)        刀具位置（当前=激励点）
├── contact_node_index                               刀触节点索引
├── force_direction_vector / response_direction_vector  方向 one-hot
├── active_pocket_id / process_step / removed_volume_ratio  过程字段
│
├── element_active_flag (ekill) / node_active_flag (ekill)   单元/节点活性标记
│
├── frequencies_adaptive (可选) / point_frf_adaptive (可选)   自适应网格 FRF
```

文件级 attrs 包含：
```text
format = "modal_frf_transolver_v2" 或 "modal_frf_transolver_ekill_v1"
response_direction / force_direction / frf_definition
frequency_grid_mode / n_freqs_fixed
```

---

## 5. 训练目标与损失

当前训练目标：

```text
input  = points + transolver_point_features + optional edge_index
         + boundary_c_xyz + excitation_index
output = modal_omega + modal_zeta + modal_phi_xyz
frf    = ModalFRFDecoder(phi_response, phi_force_exc, omega, zeta)
```

损失位于 `training/losses.py`：

- `loss_omega`: relative_l1（数值稳定）
- `loss_zeta`: 0.5·log_l1 + 0.5·relative_l1（混合）
- `loss_phi_resp`: RMS 归一化 + sign-invariant MSE（方向感知）
- `loss_phi_xyz`: RMS 归一化 + sign-invariant MSE（完整三向）
- `loss_mac`: 1-MAC 损失
- `loss_frf`: complex L1 + log-amplitude + dB（可选）

---

## 6. 快速开始

### 6.1 生成数据

```bash
# 推荐：ekill 过程数据
python ansys/generate_ekill_process_dataset.py

# 兼容：boolean 完成态数据
python ansys/generate_3d_test.py
```

### 6.2 训练

```bash
# 默认 H_YY 训练
python sample/run_validation.py \
  --data-dir ansys/data_ekill \
  --epochs 300 \
  --batch-size 1

# 其他方向
python sample/run_validation.py \
  --data-dir ansys/data \
  --response-dir X \
  --force-dir Y \
  --epochs 300

# 关闭 edge stem
python sample/run_validation.py --data-dir ansys/data --no-edges
```

### 6.3 评估

```bash
python sample/evaluate.py \
  --data-dir ansys/data \
  --response-dir Y --force-dir Y \
  --checkpoint sample/output/checkpoint_best
```

### 6.4 推理

```bash
# 预测单样本
python sample/predict.py \
  --data-dir ansys/data \
  --sample-index 0 \
  --response-dir Y --force-dir Y

# 指定查询节点
python sample/predict.py \
  --sample-index 0 --query-node 10

# 指定刀触点
python sample/predict.py \
  --sample-index 0 \
  --tool-position 0.08 0.03 0.005
```

### 6.5 可视化

```bash
python sample/测试.py --response-dir Y --force-dir Y
python sample/对比图.py --response-dir Y --force-dir Y
```

---

## 7. 设计取舍与注意事项

### 推荐路径
- **训练数据**：ekill fixed mesh + fixed frequency grid + H_YY
- **Boolean 完成态**：保留兼容路径，与 ekill 数据**不建议直接混训**
- **固定频率网格**：训练主路径，确保 batch 内频率统计一致
- **自适应峰网格**：可选评估/诊断路径

### 必选字段
`points`, `transolver_point_features`, `modal_omega`, `modal_zeta`, `modal_phi_xyz`, `excitation_index`, `boundary_c_xyz`

### 可选字段
`element_node_indices` / `edge_index`（edge stem）、`point_frf` / `frequencies`（FRF loss）、`node_active_flag` / `element_active_flag`（ekill 过程特征）

### 废弃路线
CNN 图像投影、image_tensor、query_coords 2D grid_sample、UNetPhysicsModel
