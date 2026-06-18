# 模态留数 FRF 预测流程

该分支只保留当前阶段所需的模态留数 FRF 预测相关文件。

## 文件结构

```text
README.md
下一步实验说明.md
run_meshgraph_modal.py                         # 训练入口，只调用正式 R3 程序

modal_residue/
├── generate_modal_residue_dataset_filtered_v2.py  # 数据生成程序
├── train_modal_residue_model.py                   # 原基础训练程序，保留作对照
├── train_modal_residue_bottom_model.py            # 原底面训练程序，保留作对照
└── train_r3_per_mode_bottom.py                    # 当前正式下一步程序：R=3、每阶 A-head、每阶 loss

评价与误差分析/
├── 检查数据集质量.py
├── 分析留数数据集.py
├── 诊断留数模态匹配.py
├── 查看振型和留数大小.py
└── 检测困难样本.py
```

## 当前训练入口

运行：

```text
F:/pytorch_cuda12/python.exe -B run_meshgraph_modal.py
```

当前入口调用：

```text
modal_residue/train_r3_per_mode_bottom.py
```

不再使用临时 monkey patch。新程序内部正式包含：

```text
1. R=3 标签截取；
2. 底面查询点训练；
3. PerModeResidueNet；
4. 每阶独立 A-head；
5. 每阶独立 omega/A/top/dominant loss；
6. checkpoint 中记录 model_type 和 loss_type。
```

当前配置：

```text
N_MODES_USED = 3
TARGET_REGION = bottom
HIDDEN = 96
GNN_LAYERS = 3
EPOCHS = 150
```

训练时只读取原始数据中的前三阶：

```text
modal_omega[:3]
modal_residue_z[:, :3]
```

模型结构：

```text
shared MeshGraph encoder
omega head -> omega_1, omega_2, omega_3
A head 1 -> A_q1
A head 2 -> A_q2
A head 3 -> A_q3
```

loss 结构：

```text
L = L_omega_per_mode
  + L_A_asinh_per_mode
  + L_A_top_per_mode
  + L_A_dominant_per_mode
```

输出目录：

```text
runs/下一步_R3_每阶A头_bottom/
```

## 评价与误差分析脚本

```text
F:/pytorch_cuda12/python.exe -B 评价与误差分析/检查数据集质量.py --data-dir modal_residue/data_modal_residue_fixedclamp300
F:/pytorch_cuda12/python.exe -B 评价与误差分析/分析留数数据集.py --data-dir modal_residue/data_modal_residue_fixedclamp300
F:/pytorch_cuda12/python.exe -B 评价与误差分析/诊断留数模态匹配.py
F:/pytorch_cuda12/python.exe -B 评价与误差分析/查看振型和留数大小.py
F:/pytorch_cuda12/python.exe -B 评价与误差分析/检测困难样本.py
```

## 目标

先验证前三阶模态留数 A 是否能比原 10 阶模型更稳定。如果前三阶 A 仍然学不好，就先检查底面网格、标签噪声、统一参考点和 clean dataset，而不是继续扩到 10 阶。
