# 模态留数 FRF 预测流程

该分支只保留当前阶段所需的模态留数 FRF 预测相关文件。

当前下一步实验已经改成：

```text
MeshGraph 输入结构图 + 激励点 + 查询点
→ 预测前 3 阶 modal_omega
→ 预测前 3 阶 Z 向模态留数 A_qr
→ 后续用模态叠加重建 FRF
```

不再继续硬训原来的 10 阶 A 模型；先验证低阶残量是否能稳定学出来。

---

## 1. 文件结构

```text
README.md
下一步实验说明.md
run_meshgraph_modal.py                         # 正式训练入口：R=3、每阶独立 A-head

modal_residue/
├── generate_modal_residue_dataset_filtered_v2.py  # 数据生成程序
├── train_modal_residue_model.py                   # 基础 MeshGraph 模型、损失、训练循环
└── train_modal_residue_bottom_model.py            # 底面区域训练版本

评价与误差分析/
└── 检查数据集质量.py                              # 原 validate_dataset.py，数据质量/残量公式检查
```

生成的 ANSYS/HDF5 数据集体积较大，保存在本地，不提交到 GitHub。

默认数据目录为：

```text
modal_residue/data_modal_residue_fixedclamp300/
├── train.h5
├── val.h5
└── test.h5
```

---

## 2. 数据集检查

在仓库根目录运行：

```powershell
F:/pytorch_cuda12/python.exe -B 评价与误差分析/检查数据集质量.py --data-dir modal_residue/data_modal_residue_fixedclamp300
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

---

## 3. 推荐训练入口

直接运行根目录训练入口：

```powershell
F:/pytorch_cuda12/python.exe -B run_meshgraph_modal.py
```

`run_meshgraph_modal.py` 当前已经改成下一步实验配置：

```text
N_MODES_USED = 3              # 只取前 3 阶
TARGET_REGION = "bottom"     # 只训练凹槽底面区域
KEY_QUERY_NODES = 256         # 训练时抽 256 个底面点
EVAL_QUERY_NODES = 0          # 验证/测试使用全部底面点
HIDDEN = 96
GNN_LAYERS = 3
EPOCHS = 150
```

训练入口会在读取样本时只截取：

```text
modal_omega[:3]
modal_residue_z[:, :3]
```

所以不需要重新生成 HDF5 数据集。

---

## 4. 这次程序形态的变化

旧目标：

```text
一次输出 10 阶 A
```

新目标：

```text
先只输出前 3 阶 A
每一阶 A 使用独立 residue head
```

也就是：

```text
shared MeshGraph encoder
        ↓
omega head → ω1, ω2, ω3
A head 1 → A_q1
A head 2 → A_q2
A head 3 → A_q3
```

这样做的目的不是最终只做 3 阶，而是先判断：

```text
低阶 A 是否能明显比原 10 阶模型更稳定
```

如果前 3 阶 A 都学不好，继续扩到 10 阶没有意义；如果前 3 阶能显著改善，再扩到前 5 阶、前 10 阶。

---

## 5. 底层训练脚本

底层训练脚本仍为：

```text
modal_residue/train_modal_residue_model.py
modal_residue/train_modal_residue_bottom_model.py
```

其中训练入口 `run_meshgraph_modal.py` 会安装一个下一步实验用的模型版本：

```text
PerModeResidueNet
```

它把原来单个 residue head 改为每阶一个独立 residue head。

---

## 6. 训练输出

当前训练输出默认保存在：

```text
runs/下一步_R3_每阶A头_bottom/
├── best_model.pt
├── last_model.pt
├── normalization_stats.npz
├── training_log.csv
├── history.csv
├── val_metrics.csv
├── test_metrics.csv
└── summary.json
```

---

## 7. 模型预测目标

当前模型预测两个核心量：

```text
modal_omega       # 前 3 阶模态角频率
modal_residue_z   # 前 3 阶 Z 向模态留数
```

其中模态留数定义为：

```math
A_r(x)=\phi_{r,z}(x)\phi_{r,z}(x_f)
```

FRF 后续由物理模态叠加公式重建：

```math
H_z(x,\omega)=\sum_{r=1}^{R}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
```

当前下一步先取：

```text
R = 3
```
