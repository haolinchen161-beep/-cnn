# 模态留数 FRF 预测流程

该分支保留当前阶段所需的模态留数 FRF 预测相关文件。

## 文件结构

```text
README.md
run_meshgraph_modal.py              # 正式训练入口：所有常用参数集中写在这里
modal_residue/
├── generate_modal_residue_dataset_filtered_v2.py   # 数据生成程序，如果已上传则保留在这里
├── train_modal_residue_model.py                    # MeshGraph 模型、损失、训练循环、验证、测试
└── validate_dataset.py                             # 检查本地 HDF5 数据集质量
```

生成的 ANSYS/HDF5 数据集体积较大，保存在本地，不提交到 GitHub。

默认数据目录为：

```text
data_modal_residue_filtered/
├── train.h5
├── val.h5
└── test.h5
```

## 数据集检查

在仓库根目录运行：

```powershell
F:/pytorch_cuda12/python.exe -B modal_residue/validate_dataset.py --data-dir data_modal_residue_filtered
```

检查内容包括：

```text
1. HDF5 文件是否存在；
2. 每个样本是否包含图结构 edge_index / edge_attr；
3. 每个样本是否包含节点特征、装夹特征和材料/几何特征；
4. 每个样本是否包含 modal_omega 与 modal_residue_z；
5. 模态频率是否递增；
6. 近频过滤是否满足最小相对间隔要求；
7. 如果存在 modal_phi_xyz，则检查 modal_residue_z 是否满足 A_r(x)=phi_r,z(x)*phi_r,z(x_f)。
```

## 推荐训练入口

直接运行根目录的正式训练入口：

```powershell
F:/pytorch_cuda12/python.exe -B run_meshgraph_modal.py
```

所有常用参数都集中放在 `run_meshgraph_modal.py` 顶部，例如：

```text
DATA_DIR
OUT_DIR
EPOCHS
QUERY_NODES
EVAL_QUERY_NODES
HIDDEN
GNN_LAYERS
LEARNING_RATE
WEIGHT_DECAY
GRAD_CLIP_NORM
OMEGA_LOSS_WEIGHT
PHI_LOSS_WEIGHT
LOG_EVERY
SEED
DEVICE
FP16
```

## 底层训练脚本

底层训练脚本为：

```text
modal_residue/train_modal_residue_model.py
```

该文件内部包含：

```text
1. argparse 参数设置；
2. HDF5 数据读取；
3. 节点输入特征构造；
4. edge_index / edge_attr 图消息传递；
5. MeshGraphNet 风格模型；
6. modal_omega 损失；
7. modal_residue_z 损失；
8. 训练、验证、测试与结果保存。
```

当前训练不依赖 `point_frf`，也不使用 FRF loss。FRF 后续由预测出的 `modal_omega` 和 `modal_residue_z` 按物理公式重建。

## 训练输出

训练输出默认保存在：

```text
runs/modal_residue_meshgraph/
├── best_model.pt
├── last_model.pt
├── normalization_stats.npz
├── training_log.csv
├── history.csv
├── val_metrics.csv
├── test_metrics.csv
└── summary.json
```

## 模型预测目标

模型预测两个核心量：

```text
modal_omega       # 10 阶模态角频率
modal_residue_z   # Z 向模态留数
```

其中模态留数定义为：

```math
A_r(x)=\phi_{r,z}(x)\phi_{r,z}(x_f)
```

FRF 由物理模态叠加公式重建：

```math
H_z(x,\omega)=\sum_{r=1}^{10}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
```

## 当前阶段说明

30 个样本主要用于验证 MeshGraph 模型结构、标签读取、训练日志和保存逻辑。正式泛化能力需要继续扩展到更多样本。
