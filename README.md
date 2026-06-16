# Mesh Modal Lite Clean：Z 向模态预测

当前分支 `mesh-modal-lite-clean` 是轻量模态训练分支。

本阶段目标为：

```text
有限元网格 + 几何参数 + 材料参数 + 装夹刚度边界
        → 前 K 阶固有圆频率 omega
        → 全节点 Z 向振型 phi_z
```

默认 `K = 3`。

当前模型不预测全场 `phi_x` 和 `phi_y`。数据加载器仍会读取 `modal_phi_xyz`，但只是为了计算每阶模态的 Z 向能量比例 `dir_z_ratio`，用于损失加权。

## 1. 为什么改成 Z-only

当前研究对象是 Z 向激励、Z 向响应的 FRF。模态叠加中，Z-Z FRF 的分子主要依赖：

```text
phi_response_z * phi_excitation_z
```

因此第一阶段优先训练：

```text
几何 + 材料 + 装夹刚度 → omega + phi_z
```

阻尼和 FRF 重建放到后续物理层处理。

## 2. 数据集字段

默认使用：

```text
ansys/data/train.h5
ansys/data/val.h5
ansys/data/test.h5
```

至少需要：

```text
points                [N, 3]
edge_index            [2, E]
edge_attr             [E, 4]
point_features        [N, 7]
spring_k_xyz          [N, 3]
node_type             [N]
pocket_bottom_mask    [N]
cut_region_mask       [N]
local_thickness_ratio [N]
pocket_depth_ratio    [N]
excitation_index      scalar
excitation_coord      [3]
modal_omega           [>=K]
modal_phi_xyz         [N, >=K, 3]
```

实际训练标签：

```text
modal_omega[:K]
modal_phi_xyz[:, :K, 2]
```

`modal_zeta`、`spring_c_xyz`、`point_frf`、`frequencies` 可以保留在 HDF5 中，但本阶段训练会忽略它们。

## 3. 模型输出

```text
omega: [B, K]
phi_z: [total_N, K]
```

## 4. 损失函数

频率损失不按方向加权。

Z 向振型损失包含：

```text
1. 符号对齐后的 phi_z MSE
2. phi_z 尺度 loss
3. phi_z MAC loss
```

每阶模态的振型损失权重为：

```text
w_k = min_mode_weight + (1 - min_mode_weight) * dir_z_ratio_k
```

默认：

```text
min_mode_weight = 0.2
```

这表示：非 Z 主导模态不删除，但它的 Z 向振型损失权重更小。模型仍然学习“该模态在 Z 向上的投影较小”这一事实。

## 5. 训练

```bash
python -u modal_run.py
```

或显式指定：

```bash
python -u modal_run.py --data_dir ansys/data --out_dir sample/output_modal_zonly --n_modes 3
```

旧入口也可用，它现在会转到同一套训练流程：

```bash
python -u sample/run_validation.py
```

## 6. 评估

```bash
python -u sample/evaluate.py --data_dir ansys/data --out_dir sample/output_modal_zonly
```

主要指标：

```text
freq_mae_hz
freq_mape_percent
phi_z_mse
phi_z_scale
phi_z_mac
phi_z_mac_mode1 / mode2 / mode3
dir_z_ratio_mode1 / mode2 / mode3
mode_weight_mode1 / mode2 / mode3
```

## 7. 当前研究定位

当前分支只验证：

```text
复杂加工几何 + 等效装夹刚度边界 → 固有频率 + Z 向振型
```

建议后续路线：

```text
阶段 1：训练当前 Z-only 模态模型，确认 omega 和 phi_z 可预测
阶段 2：使用预测 omega + phi_z 与数据集 modal_zeta 重建低频 Z-Z FRF
阶段 3：如果前三阶不足，再扩展到前 6 阶 phi_z
阶段 4：阻尼模型单独验证，可采用 modal_zeta、经验阻尼或 FRF/实验反标定
阶段 5：必要时再补装夹区域三向振型，用于计算装夹阻尼
```

如果 HDF5 中已经包含至少 6 阶模态，可直接运行：

```bash
python -u modal_run.py --n_modes 6 --data_dir ansys/data_20modes
```
