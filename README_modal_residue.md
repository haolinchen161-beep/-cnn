# Modal-residue FRF surrogate workflow

本分支用于三维铣削凹槽工件的模态留数 FRF 代理模型研究。核心思路不是直接预测整条频响曲线，而是先预测低维模态参数，再通过模态叠加公式重构 Z-Z FRF。

---

## 1. 研究目标

本研究面向加工过程中带凹槽薄板/工件的动态特性预测，目标是建立一个从几何、材料、加工状态和装夹边界条件到模态 FRF 的快速代理模型。

具体目标如下：

1. 使用 ANSYS/MAPDL 生成带不同凹槽布局、加工深度和装夹状态的三维有限元样本。
2. 对每个样本进行质量归一化模态分析，提取前 10 阶模态角频率 `modal_omega` 和模态振型 `modal_phi_xyz`。
3. 根据激励点位置计算 Z 向模态留数：

   \[
   A_r(x)=\phi_{r,z}(x)\phi_{r,z}(x_f)
   \]

4. 训练图神经网络预测：

   ```text
   modal_omega
   modal_residue_z = A_r(x)
   ```

5. 最终通过模态叠加公式重构 Z-Z FRF：

   \[
   H_z(x,\omega)=\sum_{r=1}^{10}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
   \]

其中 `x` 为响应节点，`x_f` 为激励节点，`r` 为模态阶次。

---

## 2. 基本物理参数

### 2.1 工件几何

| 参数 | 数值 | 说明 |
|---|---:|---|
| `L_BASE` | `0.160 m` | 工件长度 |
| `W_BASE` | `0.060 m` | 工件宽度 |
| `H_BASE` | `0.010 m` | 工件厚度 |
| `MESH_SIZE` | `0.006 m` | 默认网格尺寸 |

### 2.2 材料参数

材料按 Al7075 近似设置：

| 参数 | 数值 | 说明 |
|---|---:|---|
| `E_BASE` | `71.7e9 Pa` | 弹性模量 |
| `RHO_BASE` | `2810 kg/m^3` | 密度 |
| `PRXY_BASE` | `0.33` | 泊松比 |
| `E_RANGE` | `0.95 ~ 1.05` | 弹性模量扰动比例 |
| `RHO_RANGE` | `0.97 ~ 1.03` | 密度扰动比例 |

### 2.3 模态与 FRF 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `N_MODES` | `10` | 预测/保存前 10 阶模态 |
| `N_FREQS` | `120` | FRF 频率采样点数 |
| `FREQ_MIN_HZ` | `1.0` | 频率下限 |
| `FREQ_MAX_HZ` | 空 | 默认不固定上限，由第 10 阶频率自适应确定 |
| `ZETA_MATERIAL` | `0.002` | 材料阻尼比基值 |
| `FRF_OUTPUT_SCALE` | `1.0` | FRF 默认保存物理量 `m/N` |
| `MIN_RELATIVE_MODE_GAP` | `0.01` | 近频模态过滤阈值 |

模态分析使用质量归一化振型：

```text
USE_MASS_NORMALIZATION = True
USE_LUMPED_MASS = False
```

因此当前模态留数公式中不再额外除以模态质量。

---

## 3. 加工状态与凹槽参数

数据集通过不同凹槽布局和加工深度构造工件几何变化。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `layout_type` | `5 / 6 / 7` | 凹槽布局数量 |
| `coverage_level` | `low / medium / high` | 加工覆盖程度 |
| `TARGET_DEPTH_RANGE` | `0.25 ~ 0.60` | 凹槽目标深度比例 |
| `TARGET_DEPTH_MODE` | `0.42` | 三角分布众数 |
| `CURRENT_PROGRESS_RANGE` | `0.25 ~ 1.00` | 当前加工进度范围 |
| `GAP_ABS` | `0.006 m` | 凹槽间间隔 |
| `BORDER_ABS` | `0.006 m` | 边界保留距离 |
| `GRID_JITTER_RANGE` | `0.08 ~ 0.15` | 凹槽网格扰动 |

节点特征中会保存局部厚度、凹槽深度、切削区标记和凹槽底面标记，用于后续图神经网络输入。

---

## 4. 边界条件选择

### 4.1 为什么采用弹性装夹边界

最开始的问题中，工件不是理想完全固支，而是由工装、压板或螺钉夹紧。实际加工中装夹刚度有限，并且存在小幅预紧误差。因此本数据集不采用全边界刚性固定，而采用离散弹簧模拟装夹约束。

这样做的目的：

1. 比完全固定边界更接近真实工装约束；
2. 保留装夹刚度对低阶模态和 FRF 峰值的影响；
3. 避免把 soft / normal / hard 三类边界做成过大的工况跳变；
4. 使用固定基准刚度加小扰动，模拟实际制造和拧紧误差。

### 4.2 装夹区域

当前装夹区域包括：

1. 四个角部装夹区；
2. 若干侧边装夹区；
3. 角部装夹同时约束 `UX / UY / UZ`；
4. 侧边装夹主要约束 `UY`。

角部装夹长度：

```text
clamp_len = 0.010 m
```

角部区域选取：

```text
(0, clamp_len) × y边界
(L-clamp_len, L) × y边界
```

侧边装夹在避开角部后随机布置，最小间距约为：

```text
min_gap = 2 * H
```

### 4.3 装夹刚度参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `K_CORNER_BASE` | `3.0e7 N/m` | 角部装夹总刚度基准 |
| `K_SIDE_BASE` | `8.0e6 N/m` | 侧边装夹总刚度基准 |
| `K_CORNER_JITTER` | `0.10` | 角部刚度扰动 |
| `K_SIDE_JITTER` | `0.15` | 侧边刚度扰动 |
| `ZETA_JOINT_BASE` | `0.015` | 装夹/接触阻尼基值 |
| `ZETA_JOINT_JITTER` | `0.20` | 装夹阻尼扰动 |

对每个装夹区域，先选择区域内的有限元节点，再将区域总刚度平均分配到节点：

```text
K_each = K_this / n_selected
C_each = C_this / n_selected
```

每个被装夹节点通过 `COMBIN14` 弹簧连接到一个全约束虚拟节点。

---

## 5. 数据集设置

默认生成：

| split | 样本数 |
|---|---:|
| train | 240 |
| val | 30 |
| test | 30 |
| total | 300 |

默认输出目录：

```text
modal_residue/data_modal_residue_fixedclamp300
```

每个样本保存主要数组：

```text
points
edge_index
edge_attr
point_features
spring_k_xyz
spring_c_xyz
node_type
modal_omega
modal_zeta
modal_phi_xyz
modal_residue_z
modal_phi_exc
excitation_index
excitation_coord
```

其中 `modal_residue_z` 是当前模型训练的主要节点级目标。

---

## 6. ANSYS 到 A 的计算流程

ANSYS/MAPDL 负责求解模态问题：

\[
K\phi_r=\omega_r^2M\phi_r
\]

程序从 ANSYS 提取：

```text
modal_omega
modal_phi_xyz
```

然后 Python 计算：

```text
modal_residue_z[i, r] = phi_z[i, r] * phi_z[excitation_index, r]
```

因此：

```text
ANSYS 输出 omega 和 phi；
Python 根据模态留数公式计算 A；
神经网络预测 omega 和 A；
FRF 由 omega、zeta、A 重构。
```

---

## 7. 训练目标

当前训练目标为：

```text
Y = asinh(A / s_mode)
```

其中 `s_mode` 为每阶模态的固定尺度，优先使用数据诊断得到的推荐尺度；若不存在，则使用训练集每个样本模态 RMS 的中位数。

当前损失包括：

```text
omega log-MSE
full signed-asinh residue loss
top-|A| physical auxiliary loss
node-dominant physical auxiliary loss
```

训练输出目录：

```text
runs/modal_residue_asinh_fixedclamp300
```

小样本过拟合诊断输出目录：

```text
runs/modal_residue_asinh_fixedclamp300_debug1
```

---

## 8. 运行命令

生成数据集：

```powershell
F:/pytorch_cuda12/python.exe -B modal_residue/generate_modal_residue_dataset_filtered_v2.py
```

训练模型：

```powershell
F:/pytorch_cuda12/python.exe run_meshgraph_modal.py
```

如果 `run_meshgraph_modal.py` 中：

```python
DEBUG_TRAIN_SAMPLES = 1
```

则为小样本过拟合诊断；恢复全数据训练时改为：

```python
DEBUG_TRAIN_SAMPLES = 0
DEBUG_VAL_SAMPLES = 0
DEBUG_TEST_SAMPLES = 0
DEBUG_VAL_TEST_FROM_TRAIN = False
```
