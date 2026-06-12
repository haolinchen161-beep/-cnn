# geometric_frf_groove — 基于几何与物理模态的 FRF 预测

输入 3D 几何 + 边界条件 → `UNetPhysicsModel` 预测模态参数 `ω / ζ / φ` → `PhysicsDecoder` 通过模态叠加公式重建节点 Z 向 FRF。

当前实现重点是：固定 160×60×10 mm 凹槽薄板，在随机材料、随机凹槽、随机装夹刚度/阻尼条件下，预测前三阶三维振型、固有频率与阻尼，并在 Phase2 中用物理 FRF 损失联合约束。

---

## 1. 当前任务定义

```text
输入:
  6 通道 2.5D 几何/边界物理图像 [B,6,60,160]
  节点二维查询坐标 query_coords [total_N,2]
  节点三维坐标 node_xyz [total_N,3]
  节点物理特征 node_features [total_N,7]

网络输出:
  omega_phys [B,3]        前三阶固有圆频率 rad/s, 单调递增
  zeta       [B,3]        前三阶阻尼比
  phi        [total_N,3,3]前三阶三维振型, XYZ 三方向

物理解码:
  H(ω) = Σ_k φ_exc,k · φ_node,k / (ω_k² - ω² + j 2 ζ_k ω_k ω)
```

FRF 当前使用 Z 向激励与 Z 向响应；模态振型本身按三维 `X/Y/Z` 共同训练。

---

## 2. 目录结构

```text
├── models/
│   ├── unet_physics_model.py   主模型: ResNet+SE Encoder + Ω/ζ/φ heads + PhysicsDecoder
│   ├── frf_model.py             模型工厂 build_geometric_model()
│   ├── physics_decoder.py       无参数物理解码器, 模态叠加重建 FRF
│   └── geometry_data.py         GeometryData 数据容器
├── data/
│   └── dataset.py               HDF5 per-sample 数据集 + 2.5D 投影 + collate
├── training/
│   ├── losses.py                modal_loss + branch_loss + frf_loss
│   ├── trainer.py               Phase1/Phase2 训练、验证、日志与 checkpoint
│   └── augmentations.py         数据增强, 当前未启用
├── ansys/
│   ├── generate_3d_test.py      ANSYS MAPDL 数据生成脚本
│   └── data_2/                  train/val/test.h5
└── sample/
    ├── run_validation.py        当前训练入口与超参数配置
    ├── evaluate.py              评估与保存 final_results.npz
    ├── 对比图.py                预测 vs 真实对比图
    ├── 对比图db.py              dB 对比图
    └── output/                  checkpoint + loss_log.csv + 图表
```

---

## 3. 模型结构

```text
image_tensor [B,6,60,160]
        │
        ▼
ImprovedCNNEncoder
  Stem + ResSEBlock×3
  输出 latent [B,hidden] 与 U-Net skip features
        │
        ├── OmegaHead
        │     softplus gap 参数化
        │     w1, gap21, gap32 → [w1,w2,w3]
        │
        ├── ZetaHead
        │     log ζ 输出, exp 后得到阻尼比
        │
        ├── DirectionBranchHead
        │     每阶模态 XYZ 能量比例 log_softmax [B,3,3]
        │
        └── MicroDecoder + PhiScaleHead + NodePhiRefiner
              MicroDecoder 输出三维 mode map [B,3,3,60,160]
              joint std 归一化
              PhiScaleHead 接收 latent + branch_probs, 输出 scale [B,3]
              grid_sample 到节点
              NodePhiRefiner 用 node_xyz/node_features 做轻量 residual
        │
        ▼
phi [total_N,3,3]
        │
        ▼
PhysicsDecoder → FRF [N,F,2]
```

### 3.1 固有频率头

`OmegaHead` 不直接输出三个无序频率，而是输出单调间隙：

```text
w1    = softplus(o1) * 8000  + 500
gap21 = softplus(o2) * 12000 + 500
gap32 = softplus(o3) * 4200  + 200
w2 = w1 + gap21
w3 = w2 + gap32
```

这样保证 `w1 < w2 < w3`，避免后处理排序破坏模态对应关系。

### 3.2 三维振型输出

当前振型不是单独 Z 分量，而是完整三维：

```text
phi[...,0] = X 向振型
phi[...,1] = Y 向振型
phi[...,2] = Z 向振型
```

`MicroDecoder` 输出空间形状，`PhiScaleHead` 输出物理幅值尺度，`NodePhiRefiner` 进一步利用节点三维坐标与节点特征修正局部误差。

### 3.3 方向分支头

`DirectionBranchHead` 预测每阶模态在 XYZ 三方向上的能量比例。该分支主要服务于 Mode2/Mode3 的物理分支识别，例如：

```text
Mode2 可能是 Z 主导、X 主导、Y 主导或 X/Z 混合；
仅用连续回归容易学成平均态，因此需要显式方向比例监督。
```

方向概率 `branch_probs` 同时作为 `PhiScaleHead` 的条件输入，帮助 scale head 区分不同物理分支下的模态幅值。

---

## 4. 数据与物理设置

| 项目 | 当前设置 |
|---|---|
| 工件尺寸 | 160×60×10 mm, 铝 7075 |
| 材料变化 | E ±5%, ρ ±3% |
| 凹槽布局 | 5/6/7 凹槽, 多布局随机 |
| 凹槽深度 | 3~6 mm |
| 边界/装夹 | 4 角螺栓 XYZ 弹簧 + 3 侧顶杆 Y 弹簧 |
| 弹簧阻尼 | `C = 2 ζ sqrt(K M_ref)` |
| 材料阻尼 | `ζ_material = 0.002` |
| 总阻尼 | `ζ_k = 0.002 + Σ(C_i φ_i,k²)/(2ω_k)` |
| 网格 | SOLID187 四面体, 约 6 mm |
| 模态 | 前三阶, 保存三维节点振型 |
| 激励 | 凹槽底面几何中心最近节点, Z 向激励 |
| 频率网格 | 每峰附近采样, FRF 用线性物理量训练 |

### 4.1 Dataset 输入通道

`dataset.py` 在线将 3D 节点投影为 6 通道 160×60 图像：

| 通道 | 内容 | 说明 |
|---|---|---|
| Ch0 | 局部厚度 `Z/H` | 每像素取 max |
| Ch1 | `is_fixed` | 0 / 0.5 / 1.0 |
| Ch2 | `log10(K)` | 弹簧刚度, 仅弹簧节点 |
| Ch3 | `log10(C)` | 弹簧阻尼, 仅弹簧节点 |
| Ch4 | `E/E_base` | 全局材料常数 |
| Ch5 | `rho/rho_base` | 全局材料常数 |

图像逐通道使用 train 统计量归一化。节点坐标 `node_xyz` 归一化到 `[-1,1]`，用于 `NodePhiRefiner`。

### 4.2 数据过滤

当前 dataset 会过滤极端三阶间隙样本：

```text
f3 - f2 < 200 Hz 或 f3 - f2 > 900 Hz 的样本会被跳过
```

目的是减少三阶间隔过小造成的模态混淆，以及明显离群样本对训练的影响。

---

## 5. 损失函数

当前 `modal_loss` 包含频率、阻尼、振型三部分。

### 5.1 频率损失

```text
f_pred = omega_pred / 2π
f_true = omega_true / 2π
loss_freq = SmoothL1(f_pred, f_true)
rel = |omega_pred - omega_true| / omega_true
peak_sensitive = clamp(rel / zeta_true, max=100)
loss_omega = (loss_freq + 0.1 * mean(peak_sensitive)) * omega_weight
```

频率在 Hz 空间监督，同时加入阻尼相关的峰值敏感项。窄峰样本对频率误差更敏感。

### 5.2 阻尼损失

阻尼在 log 空间监督：

```text
loss_zeta = SmoothL1(log_zeta_pred, log(zeta_true)) * zeta_weight
```

训练前 `zeta_warmup_epochs=40` 轮将 `zeta_weight=0`，避免早期阻尼 spike 干扰频率和振型。

### 5.3 振型损失

当前振型损失是修复 Mode2 幅值问题后的版本：

```text
loss_phi = 10 * raw_phi_mse
         + 40 * loss_mac
         + 20 * loss_std
         + 10 * loss_dir_norm
```

其中：

| 项 | 作用 |
|---|---|
| 逐图 sign 对齐 | 每个样本、每阶模态单独对齐 ANSYS 随机符号 |
| 逐图 std | 每个样本独立归一化, 避免 batch 尺度平均掩盖单样本错误 |
| 方向能量加权 MSE | 按真实 XYZ 能量比例强调主导方向 |
| MAC | 约束模态空间形状, 尺度无关 |
| `loss_std` | 约束每样本、每阶模态总尺度 |
| `loss_dir_norm` | 约束每样本、每阶模态、每方向 XYZ 范数 |

`loss_dir_norm` 的默认模态权重为：

```text
[Mode1, Mode2, Mode3] = [0.2, 5.0, 0.5]
```

Mode2 权重最高，因为 Mode2 在当前数据中存在明显方向分支：Z 主导、X/Y 主导和混合态。

### 5.4 方向分支 KL

`branch_loss` 不再依赖外部 `modal_effm`，而是直接用真实三维振型计算方向能量比例：

```text
energy_k,dir = Σ_nodes phi_true²
prob_k,dir = energy_k,dir / Σ_dir energy_k,dir
KL(branch_log_probs, prob)
```

默认权重：

```text
[Mode1, Mode2, Mode3] = [0.1, 5.0, 0.5]
```

训练总损失：

```text
Phase1: loss = modal_loss + 20 * branch_loss
Phase2: loss = modal_loss + frf_weight * frf_loss + 20 * branch_loss
```

### 5.5 FRF 损失

`frf_loss` 使用 dB MSE + CDF 形状项：

```text
loss_db  = MSE(20log10(|H_pred|), 20log10(|H_true|))
loss_cdf = L1(cumsum(norm_amp_pred), cumsum(norm_amp_true))
loss_frf = loss_db + 10 * loss_cdf
```

---

## 6. 训练流程

当前训练入口：

```bash
F:/pytorch_cuda12/python.exe sample/run_validation.py
```

### 6.1 当前超参数

| 参数 | 值 |
|---|---|
| epochs | 2000 |
| validation_frequency | 5 |
| batch_size | 8 |
| hidden | 768 |
| n_modes | 3 |
| optimizer | AdamW |
| lr | 1e-3 |
| weight_decay | 1e-3 |
| scheduler | CosineAnnealingWarmRestarts, `T_0=400`, `eta_min=1e-6` |
| omega_loss_weight | 1.0 |
| zeta_loss_weight | 10.0, 前 40 轮为 0 |
| phi_loss_weight | 3.0 |
| frf_loss_weight | 0.5 |
| frf_warmup_epochs | 50 |
| phase2_min_epoch | 200 |
| teacher_anneal_epochs | 200 |
| fp16 | False |

### 6.2 Phase1: 纯模态训练

Phase1 从 epoch 0 到 `phase2_min_epoch` 前，训练：

```text
omega / zeta / phi / branch
```

FRF 不参与，验证阶段只输出 `ω_MAE (rad/s)`。

### 6.3 Phase2: FRF 联合训练

Phase2 从 `phase2_min_epoch` 开启：

```text
loss = modal_loss + current_frf_w * frf_loss + 20 * branch_loss
```

FRF 权重 warmup，避免刚进入 Phase2 时破坏已经收敛的模态参数。

### 6.4 Teacher-Forced Omega

Phase2 初期使用真实频率帮助 FRF 峰位对齐：

```text
omega_used = teacher_alpha * omega_true + (1 - teacher_alpha) * omega_pred
teacher_alpha: 1.0 → 0.0
```

这样 FRF loss 先主要训练振型幅值和阻尼宽度，而不是被峰位偏差主导。

---

## 7. 日志说明

当前训练日志示例：

```text
Epoch    1 | w=[5.2/8.6/7.4]% z=[8/36/46]% φn=[22.4/14.7/24.7]% φa=[6.7/4.1/5.7]% MAC=[0.935/0.782/0.612] | w71z0ph26 | kl=0.315 dir2=96% dir3=75% | loss=217.1
```

| 字段 | 含义 |
|---|---|
| `w=[w1/w2/w3]%` | 三阶固有频率相对误差 |
| `z=[z1/z2/z3]%` | 三阶阻尼相对误差 |
| `φn=[...]%` | 振型 NRMSE, 反映空间形状与幅值综合误差 |
| `φa=[...]%` | 振型范数幅值误差, 当前 Mode2 核心修复指标 |
| `MAC=[...]` | 模态置信准则, 只看形状相关性, 不看尺度 |
| `w/z/ph` | 频率、阻尼、振型损失占总损失比例 |
| `kl` | 方向分支 KL 损失 |
| `dir2/dir3` | Mode2/Mode3 主方向分类准确率 |
| `loss` | 当前 epoch 平均训练损失 |

### 7.1 当前诊断重点

振型修复后，重点观察：

```text
Mode2:
  dir2 是否保持 90%+
  φa2 是否保持低位, 例如 <10%
  MAC2 是否逐步升到 0.95+
  φn2 是否继续下降

Mode3:
  dir3 是否继续从 70%~80% 提升到 90%+
  MAC3 是否升到 0.90+
  φn3 是否下降

频率:
  w2/w3 是否逐步接近 w1
  若 w3 长期偏高, 优先考虑 gap32=f3-f2 的间隔约束, 不建议直接大幅提高 w3 权重
```

---

## 8. 当前已解决的问题

### 8.1 Mode2 幅值不收敛

旧问题：

```text
dir 高, MAC2 高, 但 φa2 长期卡在 40%~50%
```

根因：旧 loss 用 batch 级 std 和总尺度约束，无法保证每个样本、每个方向的真实范数。

当前修复：

```text
逐图 sign 对齐
逐图 std
逐方向 norm loss
Mode2 强权重 branch KL
branch-conditioned scale
```

修复后早期日志已经显示：

```text
φa2: epoch0 6.8% → epoch1 4.1%
```

说明 Mode2 的物理幅值尺度已被有效约束。

### 8.2 Mode2/Mode3 方向分支不可见

旧日志只有整体 `dir%`，无法确认 Mode2 是否单独分对。

当前日志已拆分：

```text
dir2=...% dir3=...%
```

用于直接观察 Mode2/Mode3 的方向分类效果。

---

## 9. 可选后续优化方向

以下不是当前必改项，而是后续诊断触发条件。

### 9.1 三阶频率优化

如果 Phase1 到 30~50 轮后 `w3` 仍显著高于其他阶，可在频率 loss 中加入间隔约束：

```text
gap_pred = f_pred[:,1:] - f_pred[:,:-1]
gap_true = f_true[:,1:] - f_true[:,:-1]
loss_gap = SmoothL1(gap_pred, gap_true)
```

优先加强 `gap32 = f3 - f2`，而不是直接大幅提高 `w3` 单项权重。

### 9.2 Mode3 分支增强

如果 `dir3` 长期低、`φa3/φn3` 不降，可将 Mode3 权重从 `0.5` 逐步提高到 `1.0~1.5`：

```text
loss_dir_norm weights: [0.2, 5.0, 1.0]
branch_loss weights:   [0.1, 5.0, 1.0]
```

### 9.3 结构级幅值增强

如果 `dir2` 很高、`φa2` 仍重新升高，说明 loss 已经知道分支，但输出表达不够，可考虑：

```text
PhiScaleHead: scale [B,K] → [B,K,3]
Mode2 gate / mixture-of-experts
```

当前从早期训练看暂不需要立刻启用。

---

## 10. 快速开始

```bash
# 生成数据, 需 ANSYS MAPDL license
F:/pytorch_cuda12/python.exe ansys/generate_3d_test.py

# 训练
F:/pytorch_cuda12/python.exe sample/run_validation.py

# 评估
F:/pytorch_cuda12/python.exe sample/evaluate.py

# 可视化对比
F:/pytorch_cuda12/python.exe sample/对比图.py
F:/pytorch_cuda12/python.exe sample/对比图db.py
```

训练过程会在 `sample/output/` 下保存：

```text
checkpoint_last
checkpoint_best
loss_log.csv
```

重新运行 `run_validation.py` 时，如果存在 `checkpoint_last`，会自动恢复训练。
