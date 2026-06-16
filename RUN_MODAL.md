# 运行 Z 向模态训练

当前分支：

```bash
git checkout mesh-modal-lite-clean
```

直接训练：

```bash
python -u modal_run.py
```

## 当前训练目标

本阶段只训练：

```text
网格 + 几何 + 材料 + 装夹刚度边界 → 固有频率 omega + 全节点 Z 向振型 phi_z
```

默认输出为：

```text
omega: [B, 3]
phi_z: [total_N, 3]
```

也就是前三阶固有频率和前三阶全节点 Z 向振型。

## 为什么不训练三向振型

当前研究先关注 Z 向激励和 Z 向响应 FRF。模态叠加中，Z-Z FRF 的分子主要依赖：

```text
响应点 phi_z × 激励点 phi_z
```

因此第一阶段只预测 `phi_z`，暂时不预测全场 `phi_x` 和 `phi_y`。

HDF5 数据中的 `modal_phi_xyz` 仍会被读取，但它只用于计算每阶模态的 Z 向能量比例 `dir_z_ratio`，从而给 `phi_z loss` 加权。网络本身不输出 `phi_x` 和 `phi_y`。

## 损失函数

保留的损失为：

```text
1. 固有频率 log loss
2. 符号对齐后的 Z 向振型 MSE
3. Z 向振型尺度 loss
4. Z 向 MAC loss
5. 每阶模态的 Z 向主导程度加权
```

每阶模态的振型损失权重为：

```text
w_k = min_mode_weight + (1 - min_mode_weight) * dir_z_ratio_k
```

默认：

```text
min_mode_weight = 0.2
```

含义是：非 Z 主导模态不会被删除，它的 `phi_z` 投影仍然参与训练，但权重更小。

## 训练命令

```bash
python -u modal_run.py --data_dir ansys/data --out_dir sample/output_modal_zonly
```

快速测试 3 个 epoch：

```bash
python -u modal_run.py --epochs 3 --batch_size 1
```

## 评估命令

```bash
python -u sample/evaluate.py --data_dir ansys/data --out_dir sample/output_modal_zonly
```

## 后续扩展到前 6 阶

如果 HDF5 中已经包含 6 阶或更多模态，可以直接改参数：

```bash
python -u modal_run.py --n_modes 6 --data_dir ansys/data_20modes
```

## 当前不做的事情

本分支暂时不训练：

```text
1. 阻尼 zeta
2. FRF
3. 全场 phi_x / phi_y
4. FRF loss
5. zeta loss
```

后续重建 Z-Z FRF 时，先使用预测的 `omega + phi_z`，阻尼可暂时使用数据集中的 `modal_zeta` 或后续标定的阻尼模型。
