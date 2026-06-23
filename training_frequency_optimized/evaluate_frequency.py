# -*- coding: utf-8 -*-
"""Evaluation script for Baseline MLP natural frequency model."""
from __future__ import annotations

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_frequency import FrequencyH5Dataset
from model_frequency import FrequencyTokenMLP

DATA_DIR = r"f:\毕业论文\stage1-modal-residue-dataset\data\data_modal_residue_stage1500"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    # predictions: [N, 3], targets: [N, 3] in Hz
    errors = predictions - targets
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / np.maximum(targets, 1e-8)
    
    metrics = {}
    for i in range(3):
        metrics[f"mae_m{i+1}"] = float(abs_errors[:, i].mean())
        metrics[f"mape_m{i+1}"] = float(rel_errors[:, i].mean())
        metrics[f"max_err_m{i+1}"] = float(abs_errors[:, i].max())
        metrics[f"max_mape_m{i+1}"] = float(rel_errors[:, i].max())
        
    metrics["mae_mean"] = float(abs_errors.mean())
    metrics["mape_mean"] = float(rel_errors.mean())
    metrics["max_mape_overall"] = float(rel_errors.max())
    return metrics

def run_evaluation():
    device = get_device()
    print(f"Using device for evaluation: {device}")
    
    runs_dir = Path(CURRENT_DIR) / "runs"
    ckpt_path = runs_dir / "checkpoints" / "best_frequency_model.pt"
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found at: {ckpt_path}")
        print("Please train the model first by running: python train_frequency.py")
        return
        
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config_dict = ckpt["config"]
    stats_raw = ckpt["stats"]
    
    # Reconstruct stats tensors
    stats = {k: v.to(device) for k, v in stats_raw.items()}
    
    # Initialize model
    model = FrequencyTokenMLP(
        pocket_dim=config_dict.get("pocket_dim", 8),
        clamp_dim=config_dict.get("clamp_dim", 11),
        global_dim=config_dict.get("global_dim", 9),
        token_dim=config_dict.get("token_dim", 96),
        hidden_dim=config_dict.get("hidden_dim", 192),
        fusion_dim=config_dict.get("fusion_dim", 256),
        out_modes=config_dict.get("target_modes", 3),
        token_layers=config_dict.get("token_layers", 3),
        fusion_layers=config_dict.get("fusion_layers", 4),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model weights successfully loaded.")
    
    # Load dataset
    val_h5 = config_dict.get("val_h5", str(Path(DATA_DIR) / "val.h5"))
    test_h5 = config_dict.get("test_h5", str(Path(DATA_DIR) / "test.h5"))
    
    val_set = FrequencyH5Dataset(val_h5, target_modes=3)
    test_set = FrequencyH5Dataset(test_h5, target_modes=3)
    
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    # Evaluate Validation set
    val_preds, val_targets = evaluate_set(model, val_loader, stats, device)
    val_metrics = compute_metrics(val_preds, val_targets)
    
    # Evaluate Test set
    test_preds, test_targets = evaluate_set(model, test_loader, stats, device)
    test_metrics = compute_metrics(test_preds, test_targets)
    
    # Output report
    report_path = runs_dir / "logs" / "frequency_evaluation_report_mlp.md"
    write_markdown_report(report_path, val_metrics, test_metrics, val_preds, val_targets, test_preds, test_targets)
    print(f"\nEvaluation finished. Report saved to: {report_path}")
    
    # Also save to test_metrics.json for programmatic access
    with open(runs_dir / "logs" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

def evaluate_set(model, loader, stats, device) -> tuple[np.ndarray, np.ndarray]:
    all_preds = []
    all_targets = []
    
    eps = 1e-8
    with torch.no_grad():
        for batch in loader:
            p_feats = batch["pocket_features"].float().to(device)
            c_feats = batch["clamp_features"].float().to(device)
            g = batch["global_features"].float().to(device)
            w = batch["omega"].float().numpy()
            
            # Normalize inputs using stats
            p_feats_norm = (p_feats - stats["pocket_features_mean"].view(1, 1, -1)) / stats["pocket_features_std"].view(1, 1, -1).clamp_min(eps)
            c_feats_norm = (c_feats - stats["clamp_features_mean"].view(1, 1, -1)) / stats["clamp_features_std"].view(1, 1, -1).clamp_min(eps)
            g_norm = (g - stats["global_features_mean"].view(1, -1)) / stats["global_features_std"].view(1, -1).clamp_min(eps)
            
            # Model prediction
            pred_norm = model(p_feats_norm, c_feats_norm, g_norm)
            
            # De-normalize to physical frequencies (rad/s -> Hz by dividing by 2*pi)
            logw = pred_norm * stats["omega_log_std"].view(1, -1) + stats["omega_log_mean"].view(1, -1)
            pred_omega_rad = torch.exp(logw).clamp_min(eps).cpu().numpy()
            pred_omega_hz = pred_omega_rad / (2 * np.pi)
            target_omega_hz = w / (2 * np.pi)
            
            all_preds.append(pred_omega_hz)
            all_targets.append(target_omega_hz)
            
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)

def pct(x: float) -> str:
    return f"{x * 100.0:.3f}%"

def write_markdown_report(path, val_m, test_m, val_preds, val_targets, test_preds, test_targets):
    val_ranges = [f"{val_targets[:, i].min():.1f} ~ {val_targets[:, i].max():.1f}" for i in range(3)]
    test_ranges = [f"{test_targets[:, i].min():.1f} ~ {test_targets[:, i].max():.1f}" for i in range(3)]
    
    content = f"""# 📊 Baseline MLP Natural Frequency Prediction Evaluation Report

本报告展示了训练好的 Baseline MLP 频率模型在验证集 (`val.h5`) 和测试集 (`test.h5`) 上的预测精度。

---

## 1. 固有频率预测误差统计表

### 验证集 (`val.h5`) 误差统计
| 模态阶数 (Mode) | 真实频率范围 (Hz) | 平均绝对误差 (MAE) | 平均相对百分误差 (MAPE) | 最大绝对误差 (Max Err) | 最大相对百分误差 (Max MAPE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **第一阶 (Mode 1)** | {val_ranges[0]} | **{val_m['mae_m1']:.3f} Hz** | **{pct(val_m['mape_m1'])}** | {val_m['max_err_m1']:.3f} Hz | {pct(val_m['max_mape_m1'])} |
| **第二阶 (Mode 2)** | {val_ranges[1]} | **{val_m['mae_m2']:.3f} Hz** | **{pct(val_m['mape_m2'])}** | {val_m['max_err_m2']:.3f} Hz | {pct(val_m['max_mape_m2'])} |
| **第三阶 (Mode 3)** | {val_ranges[2]} | **{val_m['mae_m3']:.3f} Hz** | **{pct(val_m['mape_m3'])}** | {val_m['max_err_m3']:.3f} Hz | {pct(val_m['max_mape_m3'])} |
| **平均值 (Mean)** | - | **{val_m['mae_mean']:.3f} Hz** | **{pct(val_m['mape_mean'])}** | - | - |

### 测试集 (`test.h5`) 误差统计
| 模态阶数 (Mode) | 真实频率范围 (Hz) | 平均绝对误差 (MAE) | 平均相对百分误差 (MAPE) | 最大绝对误差 (Max Err) | 最大相对百分误差 (Max MAPE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **第一阶 (Mode 1)** | {test_ranges[0]} | **{test_m['mae_m1']:.3f} Hz** | **{pct(test_m['mape_m1'])}** | {test_m['max_err_m1']:.3f} Hz | {pct(test_m['max_mape_m1'])} |
| **第二阶 (Mode 2)** | {test_ranges[1]} | **{test_m['mae_m2']:.3f} Hz** | **{pct(test_m['mape_m2'])}** | {test_m['max_err_m2']:.3f} Hz | {pct(test_m['max_mape_m2'])} |
| **第三阶 (Mode 3)** | {test_ranges[2]} | **{test_m['mae_m3']:.3f} Hz** | **{pct(test_m['mape_m3'])}** | {test_m['max_err_m3']:.3f} Hz | {pct(test_m['max_mape_m3'])} |
| **平均值 (Mean)** | - | **{test_m['mae_mean']:.3f} Hz** | **{pct(test_m['mape_mean'])}** | - | - |

---

## 2. 物理合理性与计算效率分析

> [!NOTE]
> **基线 MLP 模型特性**
> 1. **无损特征输入**：使用解析的 1D 口袋与夹具参数直接作为输入，消除了空间网格光栅化离散精度限制（$1.25\text{ mm}$），在小样本几何模板拟合中拥有极高的自适应逼近精度。
> 2. **直接绝对值拟合**：以 Smooth L1 损失函数对频率绝对值进行直接优化，不包含畸变的比例约束，保证求解梯度方向直指 MAPE 最小值。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    run_evaluation()
