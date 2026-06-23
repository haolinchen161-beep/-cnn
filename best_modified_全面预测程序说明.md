# best_modified 分支：频率、振型、阻尼全面预测程序说明

本文档说明 `best_modified` 分支中用于 **固有频率、模态振型、阻尼比与 FRF 重建** 的完整程序结构、数据流、训练目标和需要注意的问题。

当前分支相比原始主分支，删除了旧的 CNN/UNet/FRF 端到端程序，改为三阶段物理分解预测：

```text
结构参数 / 夹具参数 / 材料参数
        │
        ├── 频率模型：预测前三阶固有角频率 omega_1:3
        ├── 振型模型：预测前三阶 z 向模态振型 phi_z(q, r)
        └── 阻尼模型：预测前三阶阻尼比 zeta_1:3

A_r(q,p) = phi_z(q,r) * phi_z(p,r)

H(q,p,Omega) = sum_r A_r(q,p) / (omega_r^2 - Omega^2 + j*2*zeta_r*omega_r*Omega)
```

其中 `p` 是激励点，`q` 是响应点。频率、振型、阻尼分别预测，最后在根目录脚本 `1.py` 中组合成 FRF。

---

## 1. 分支文件结构

当前新增的主要程序包括：

```text
1.py
training_frequency_optimized/
    dataset_frequency.py
    model_frequency.py
    trainer_frequency_v2.py
    train_frequency.py
    evaluate_frequency.py

training_shape_optimized/
    dataset.py
    model.py
    loss.py
    train.py

training_damping_optimized/
    dataset_damping.py
    model_damping.py
    trainer_damping.py
    train_damping.py
    precompute_damping_priors.py
    damping_training_report.md
```

整体含义如下：

| 模块 | 作用 |
|---|---|
| `training_frequency_optimized` | 训练固有频率预测模型 |
| `training_shape_optimized` | 训练 z 向模态振型预测模型，并可组合出模态残量 A |
| `training_damping_optimized` | 训练阻尼比预测模型 |
| `1.py` | 加载三类最优模型，重建并绘制 FRF |

---

## 2. 频率预测模块：`training_frequency_optimized`

### 2.1 输入与目标

频率模型预测前三阶固有角频率：

```text
modal_omega[:3]
```

输入为结构级特征，不包含激励点，因为固有频率是结构自身属性：

```text
pocket_features: [7, 8]
clamp_features: [7, 11]
global_features: [9]
```

`global_features` 包括：

```text
E_ratio, rho_ratio,
layout_type / 7,
coverage_code / 2,
clamp_code / 2,
removed_volume_ratio,
grid_jitter,
finished_count / 7,
current_progress
```

`clamp_features` 中刚度和阻尼的 log 特征会被缩放：

```text
clamp[:, 5:8]  /= 12
clamp[:, 8:11] /= 8
```

### 2.2 模型结构

模型文件：

```text
training_frequency_optimized/model_frequency.py
```

模型类：

```text
FrequencyTokenMLP
```

结构：

```text
pocket_encoder -> mean/max pooling
clamp_encoder  -> mean/max pooling
global_encoder
concat -> fusion MLP -> linear head -> 3 modes
```

它不是 GNN，也不是 CNN，而是结构 token 的 MLP + mean/max pooling。该结构适合当前参数化工件数据，因为几何和夹具主要已经被压缩成结构化 token。

### 2.3 训练目标

训练器中对真实频率做自然对数：

```text
logw = log(omega)
y = (logw - mean) / std
```

模型输出标准化后的 `log(omega)`，反归一化为：

```text
omega_pred = exp(pred * omega_log_std + omega_log_mean)
```

损失函数：

```text
SmoothL1(pred, target) + order_loss
```

其中 order loss 用于保持前三阶频率递增。

### 2.4 输出文件

训练脚本：

```bash
cd training_frequency_optimized
python train_frequency.py --preset full --data-dir <data_dir>
```

默认输出：

```text
training_frequency_optimized/runs/
    checkpoints/best_frequency_model.pt
    checkpoints/last_frequency_model.pt
    logs/frequency_train_log.csv
    logs/test_metrics.json
```

评估脚本：

```bash
python evaluate_frequency.py
```

会读取 `runs/checkpoints/best_frequency_model.pt` 并输出频率误差报告。

---

## 3. 振型预测模块：`training_shape_optimized`

### 3.1 输入与目标

振型模型预测前三阶 z 向振型：

```text
phi_z(q, r), r = 1,2,3
```

数据集返回：

```text
pocket_features
clamp_features
global_features
q_coord, q_node_features
p_coord, p_node_features
target_phi_z
target_residue
modal_omega
```

其中：

```text
target_phi_z(q,r) = modal_phi_z[q,r]
target_residue(q,r) = modal_residue_z[q,r]
                   = phi_z(q,r) * phi_z(p,r)
```

### 3.2 训练时如何得到 A

训练脚本把所有查询点 q 和激励点 p 拼到一起：

```text
all_coords = concat(q_coord, p_coord)
all_nodes  = concat(q_node_features, p_node_features)
```

模型一次前向输出：

```text
phi_pred_all: [B, Q+1, 3]
```

再拆分：

```text
phi_pred_q = phi_pred_all[:, :Q, :]
phi_pred_p = phi_pred_all[:, Q:, :]
```

并组合出模态残量：

```text
A_pred(q,p,r) = phi_pred_q(q,r) * phi_pred_p(p,r)
```

因此当前振型模型已经不仅仅是纯 phi 训练，它在 loss 和指标中都引入了 A。

### 3.3 模型结构

模型文件：

```text
training_shape_optimized/model.py
```

模型类：

```text
SymmetricSymlogModalOperator
```

核心结构：

```text
frozen frequency model -> normalized omega
pocket_encoder
clamp_encoder
global_encoder
omega_encoder
context_encoder
FourierPositionEncoding(q_coord)
node_encoder
cross attention: q node attends to pocket/clamp tokens
mode-specific query heads
phi_scale_head
```

关键设计：

1. **冻结频率模型**：使用频率模型输出的 `normalized_omega` 作为模态条件。
2. **Fourier 坐标编码**：增强高阶模态的空间表达能力。
3. **Cross Attention**：响应节点特征通过 attention 读取 pocket/clamp token。
4. **每阶独立预测头**：减少一阶、二阶、三阶之间的梯度干扰。
5. **scale head**：对标准化形状进行整体幅值缩放。

### 3.4 频率分支的特殊处理

当前 `model.py` 中保留了两个频率分支：

```text
normalized_omega_bug
normalized_omega_correct
```

其中：

- `normalized_omega_bug` 用于 shape branch 的上下文编码，目的是保持已经训练出来的权重兼容。
- `normalized_omega_correct` 用恢复后的 clamp 特征计算真实物理频率输出。

物理频率反归一化已经采用正确形式：

```text
physical_omega = exp(logw)
```

该处理说明当前模型是一个“兼容旧权重 + 修正物理输出”的版本。后续如果重新训练完整振型模型，建议统一频率分支，避免 bug/correct 两套逻辑长期并存。

### 3.5 损失函数

文件：

```text
training_shape_optimized/loss.py
```

当前 loss 名称仍为 `OrderedSymlogLoss`，但实际已经不再使用 symlog。核心损失包括：

```text
1. 符号不敏感 phi L1 loss
2. MAC loss
3. 模态残量 A loss
4. 二阶、三阶加权 mode weights
```

振型符号不唯一，因此每阶损失取：

```text
min(|pred_phi - true_phi|, |pred_phi + true_phi|)
```

A loss：

```text
mean(|A_pred - A_true|)
```

当前组合：

```text
total_loss_per_mode = shape_loss + 1.0 * mac_loss + 1.5 * residue_loss
mode_weight = [1.0, 1.5, 2.0]
```

### 3.6 指标

当前日志同时记录 raw 和 strong 两类指标。

raw 指标：

```text
所有样本、直接按 mode 序号计算
```

strong 指标：

```text
只统计 target shape norm >= 50 的强模态，并使用 Hungarian matching 做模态匹配
```

A 指标：

```text
A_nMAE_raw_i
A_corr_raw_i
A_nMAE_strong_i
A_corr_strong_i
```

其中 A_nMAE 使用：

```text
mean(abs(A_pred - A_true)) / max(mean(abs(A_true)), 0.05)
```

### 3.7 训练入口

```bash
cd training_shape_optimized
python train.py --mode train
```

微调：

```bash
python train.py --mode ft --lr 1e-4 --epochs 60
```

默认设置：

```text
train: 180 epochs, lr=5e-4, weight_decay=2e-4, patience=25
ft:     60 epochs, lr=1e-4, patience=15
```

输出：

```text
training_shape_optimized/checkpoints/best_model.pth
training_shape_optimized/checkpoints/latest_model.pth
training_shape_optimized/logs/training_log.csv
```

---

## 4. 阻尼预测模块：`training_damping_optimized`

### 4.1 目标

阻尼模型预测前三阶模态阻尼比：

```text
modal_zeta[:3]
```

阻尼比由材料阻尼和边界阻尼组成：

```text
zeta_k = zeta_material + zeta_boundary,k
zeta_material = 0.002
```

训练器将边界阻尼部分写成：

```text
boundary = zeta - 0.002
```

并引入材料物理标度：

```text
log_material_scale_damping = -0.5 * log(E_ratio * rho_ratio)
```

训练目标为：

```text
y = normalize(log(zeta - 0.002) - log_material_scale_damping)
```

预测反变换：

```text
zeta_pred = exp(y_pred * std + mean + log_material_scale_damping) + 0.002
```

### 4.2 阻尼先验 precompute

阻尼数据集需要预先计算两个物理先验：

```text
omega_pred:         [3]
phi_z_norm_pred:    [3]
```

预计算脚本：

```bash
cd training_damping_optimized
python precompute_damping_priors.py
```

它会加载：

```text
training_shape_optimized/checkpoints/best_model.pth
training_frequency_optimized/runs/best_frequency_model.pt
```

然后对 `train.h5 / val.h5 / test.h5` 生成：

```text
train_priors.pt
val_priors.pt
test_priors.pt
```

这些先验会在 `dataset_damping.py` 中读入；如果不存在，则退化为 0。

### 4.3 输入特征

阻尼模型输入包括：

```text
pocket_features: [7, 8]
clamp_features: [7, 11]
pocket_centers: [7, 2]
clamp_centers:  [7, 2]
global_features: [13]
```

其中 `global_features` 为：

```text
layout_type / 7
coverage_code / 2
clamp_code / 2
removed_volume_ratio
grid_jitter
finished_count / 7
current_progress
omega_pred[0:3]
phi_z_norm_pred[0:3]
```

### 4.4 模型结构

模型文件：

```text
training_damping_optimized/model_damping.py
```

模型类：

```text
DampingTokenMLP
```

核心结构：

```text
pocket_encoder + 2D Fourier PE
clamp_encoder  + 2D Fourier PE
learnable CLS token
2-layer TransformerEncoder
concat(CLS, global_token)
fusion MLP
linear head -> zeta_1:3
```

阻尼模型不是单纯 MLP，它使用 Transformer token 交互建模 pocket/clamp 空间耦合。

### 4.5 训练损失

训练器采用标准化后的 log-boundary-damping 作为目标，损失为 Smooth L1：

```text
SmoothL1(pred, target)
```

并对二阶、三阶加权：

```text
mode_weight = [1.0, 2.0, 2.0]
```

训练包含 10 epoch warmup + cosine scheduler。

### 4.6 训练入口

```bash
cd training_damping_optimized
python train_damping.py --preset full --data-dir <data_dir>
```

输出：

```text
training_damping_optimized/runs/best_damping_model.pt
training_damping_optimized/runs/last_damping_model.pt
training_damping_optimized/runs/damping_train_log.csv
training_damping_optimized/runs/test_metrics.json
```

---

## 5. FRF 全流程重建脚本：`1.py`

根目录 `1.py` 是完整预测与 FRF 可视化脚本。

加载模型：

```text
shape_model:   training_shape_optimized/checkpoints/best_model.pth
freq_model:    training_frequency_optimized/runs/best_frequency_model.pt
damping_model: training_damping_optimized/runs/best_damping_model.pt
```

预测流程：

```text
1. 读取 test.h5
2. 对每个样本使用 shape model 预测 phi_z(q) 和 phi_z(p)
3. 使用 frequency model 预测 omega_1:3
4. 使用 damping model 预测 zeta_1:3
5. 组合 A_pred(q,p,r) = phi_q(q,r) * phi_p(p,r)
6. 计算预测 FRF
7. 使用真实前三阶模态参数重建 3-mode true FRF 作为公平对比
8. 绘制幅值 dB 和 unwrap 后的相位图
```

FRF 计算公式：

```text
H(q,p,Omega) = sum_r A_r(q,p) / (omega_r^2 - Omega^2 + j*2*zeta_r*omega_r*Omega)
```

脚本中使用实部、虚部分解：

```text
dw = omega_r^2 - Omega^2
gm = 2*zeta_r*omega_r*Omega
real += A * dw / (dw^2 + gm^2)
imag += -A * gm / (dw^2 + gm^2)
```

相位图已经使用：

```text
np.unwrap(np.angle(H))
```

这避免了 `+180°/-180°` 包裹造成的视觉跳变误判。

---

## 6. 推荐运行顺序

推荐完整流程如下：

```bash
# 1. 训练频率模型
cd training_frequency_optimized
python train_frequency.py --preset full --data-dir <data_dir>

# 2. 训练振型模型
cd ../training_shape_optimized
python train.py --mode train

# 3. 预计算阻尼模型需要的频率/振型先验
cd ../training_damping_optimized
python precompute_damping_priors.py

# 4. 训练阻尼模型
python train_damping.py --preset full --data-dir <data_dir>

# 5. 重建 FRF 并绘图
cd ..
python 1.py
```

---

## 7. 当前程序的主要优点

1. **物理分解清晰**：频率、振型、阻尼分别预测，比端到端 FRF 更容易分析误差来源。
2. **A 的物理一致性更强**：A 不是直接回归，而是由预测振型乘积得到。
3. **振型 loss 处理了符号不唯一**：`phi` 和 `-phi` 被视为同一物理振型。
4. **阻尼加入物理标度律**：对材料参数 E、rho 的影响做了显式解耦。
5. **日志指标更完整**：振型模块已经记录 phi raw/strong 指标和 A raw/strong 指标。
6. **FRF 可视化更合理**：相位已经使用 unwrap，避免相位包裹误判。

---

## 8. 需要重点注意的问题

### 8.1 频率 checkpoint 路径不一致

频率训练器保存路径为：

```text
training_frequency_optimized/runs/checkpoints/best_frequency_model.pt
```

但以下程序默认读取：

```text
training_frequency_optimized/runs/best_frequency_model.pt
```

涉及文件：

```text
training_shape_optimized/model.py
training_damping_optimized/precompute_damping_priors.py
1.py
```

因此实际运行前需要确认 checkpoint 是否已经复制到对应路径，或者统一修改为：

```text
training_frequency_optimized/runs/checkpoints/best_frequency_model.pt
```

否则可能出现找不到频率模型、或者加载了非预期模型的问题。

### 8.2 `training_shape_optimized/model.py` 的导入路径风险

当前振型模型中导入频率模型时写的是：

```python
from training_frequency.model_frequency import FrequencyTokenMLP
```

但当前分支新增的是：

```text
training_frequency_optimized/model_frequency.py
```

如果运行环境中没有旧的 `training_frequency` 包，则该导入会失败，并导致 `FrequencyTokenMLP` 未定义。

建议修改为更稳健的形式：

```python
try:
    from training_frequency_optimized.model_frequency import FrequencyTokenMLP
except ImportError:
    from model_frequency import FrequencyTokenMLP
```

或在 `sys.path` 中明确加入 `training_frequency_optimized` 后直接：

```python
from model_frequency import FrequencyTokenMLP
```

### 8.3 振型 CSV 中的 `MSE` 实际不是平方误差

振型 loss 使用的是 L1/MAE：

```text
mean(abs(pred_phi - target_phi))
```

但 CSV header 仍写成：

```text
Train_phi_MSE_i
Val_phi_MSE_i
```

建议改名为：

```text
Train_phi_MAE_i
Val_phi_MAE_i
```

避免论文或图表中误写。

### 8.4 `OrderedSymlogLoss` 名字已经不准确

当前 loss 已经移除了 symlog，但类名仍为 `OrderedSymlogLoss`。

建议后续改名：

```text
ModalShapeResidueLoss
```

或：

```text
SignInvariantShapeResidueLoss
```

### 8.5 shape 模型中的 bug/correct 双频率分支需要最终统一

当前 `model.py` 明确保留：

```text
normalized_omega_bug     -> 用于 shape branch，兼容旧权重
normalized_omega_correct -> 用于 physical omega 输出
```

这是为了权重兼容可以接受，但如果后续重新从头训练，建议统一使用 correct branch，避免长期存在两套频率语义。

### 8.6 FRF 评价当前主要是可视化，还缺全局数值表

`1.py` 已经能画单节点 FRF 对比图，但建议后续增加全测试集指标：

```text
FRF_complex_relative_error
FRF_amp_dB_MAE
A_nMAE_mean
A_corr_mean
phase_circular_error
```

这样可以和训练日志中的 A 指标闭环。

---

## 9. 当前推荐定位

当前 `best_modified` 分支可以作为论文中的“全面预测路线”：

```text
频率模型：结构级全局参数 -> omega
振型模型：结构级参数 + 节点局部特征 -> phi_z，并组合 A
阻尼模型：结构级参数 + 频率/振型先验 -> zeta
FRF 重建：omega + zeta + A -> H(q,p,Omega)
```

与之前直接预测 A 或 factorized A 的路线相比，本分支的特点是：

```text
1. 物理解释更强；
2. 可分别评估频率、振型、阻尼误差；
3. A 由 phi 乘积构造，更符合模态理论；
4. 适合作为最终 FRF 预测框架或论文主线候选。
```

但在正式作为最终主线前，建议先修正第 8 节中的路径、命名和评价闭环问题。
