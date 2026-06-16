# 模态留数 FRF 预测流程

该分支已经清理为当前阶段所需的最小代码结构，只保留模态留数 FRF 预测相关文件。

## 文件结构

```text
README.md
modal_residue/
├── train_modal_residue_model.py   # 训练模态频率与模态留数预测模型
└── validate_dataset.py            # 检查本地 HDF5 数据集质量
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
2. 每个样本是否包含必要字段；
3. 是否为 10 阶模态；
4. 模态频率是否递增；
5. 频率网格是否覆盖第 10 阶模态；
6. 近频过滤是否满足最小相对间隔要求；
7. modal_residue_z 是否满足 A_r(x)=phi_r,z(x)*phi_r,z(x_f)；
8. point_frf 是否满足模态叠加公式。
```

## 训练基线模型

在仓库根目录运行：

```powershell
F:/pytorch_cuda12/python.exe -B modal_residue/train_modal_residue_model.py `
  --data-dir data_modal_residue_filtered `
  --out-dir runs/modal_residue_baseline `
  --epochs 300 `
  --query-nodes 512 `
  --eval-query-nodes 1024 `
  --frf-loss-weight 0.05
```

训练输出默认保存在：

```text
runs/modal_residue_baseline/
├── best_model.pt
├── normalization_stats.npz
├── history.csv
├── val_metrics.csv
├── test_metrics.csv
└── summary.json
```

## 模型预测目标

模型第一版预测两个核心量：

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

该分支用于验证“几何/装夹条件 → 模态频率与模态留数 → FRF 重建”的最小训练流程。当前版本不是最终模型结构，主要用于确认数据集、标签和训练闭环是否正常。
