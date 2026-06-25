# -*- coding: utf-8 -*-
"""固有频率训练入口。

当前训练固有频率 baseline：
    pocket_features + clamp_features + material/global scalars -> modal_omega[:3]
"""
from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass, replace
from pathlib import Path

# Add current folder to path to import local trainer and model correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer_frequency_v2 import train_frequency_model

# Data directory
DATA_DIR = r"f:\毕业论文\stage1-modal-residue-dataset\data\data_modal_residue_stage1500"
RUN_PRESET = "full"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class FrequencyTrainConfig:
    # Path settings
    train_h5: str = str(Path(DATA_DIR) / "train.h5")
    val_h5: str = str(Path(DATA_DIR) / "val.h5")
    test_h5: str = str(Path(DATA_DIR) / "test.h5")
    output_dir: str = os.path.join(CURRENT_DIR, "runs")

    # Target settings
    target_modes: int = 3

    # Input dimensions
    pocket_dim: int = 8
    clamp_dim: int = 11
    global_dim: int = 9

    # Model settings
    token_dim: int = 96
    hidden_dim: int = 192
    fusion_dim: int = 256
    token_layers: int = 3
    fusion_layers: int = 4
    dropout: float = 0.05

    # Optimizer settings
    epochs: int = 350
    batch_size: int = 16
    eval_batch_size: int = 64
    stat_batch_size: int = 128
    learning_rate: float = 5.0e-4
    min_learning_rate: float = 1.0e-6
    weight_decay: float = 5.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip_norm: float = 1.0

    # Loss settings
    smooth_l1_beta: float = 0.25
    order_loss_weight: float = 0.02
    order_log_margin: float = 1.0e-4
    eps: float = 1.0e-8

    # Reporting and checkpoint saving
    val_interval: int = 1
    log_interval: int = 1
    save_last_interval: int = 5

    # General control
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    early_stop_patience: int = 120
    eval_test: bool = True

PRESETS = {
    "debug10": dict(
        epochs=15,
        batch_size=2,
        eval_batch_size=2,
        stat_batch_size=10,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-5,
        weight_decay=1.0e-4,
        val_interval=1,
        log_interval=1,
        save_last_interval=1,
        early_stop_patience=15,
        output_dir=os.path.join(CURRENT_DIR, "runs_debug10"),
    ),
    "small": dict(
        epochs=100,
        batch_size=8,
        eval_batch_size=16,
        stat_batch_size=64,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-5,
        weight_decay=1.0e-4,
        val_interval=1,
        log_interval=1,
        save_last_interval=5,
        early_stop_patience=30,
        output_dir=os.path.join(CURRENT_DIR, "runs_small"),
    ),
    "full": dict(
        epochs=350,
        batch_size=16,
        eval_batch_size=64,
        stat_batch_size=128,
        learning_rate=5.0e-4,
        min_learning_rate=1.0e-6,
        weight_decay=5.0e-4,
        val_interval=1,
        log_interval=1,
        save_last_interval=5,
        early_stop_patience=120,
        output_dir=os.path.join(CURRENT_DIR, "runs"),
    ),
}

def build_config(preset: str, data_dir: str | None = None) -> FrequencyTrainConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")

    cfg = FrequencyTrainConfig()
    cfg = replace(cfg, **PRESETS[preset])

    if data_dir is not None:
        d = Path(data_dir)
    else:
        d = Path(DATA_DIR)

    cfg.train_h5 = str(d / "train.h5")
    cfg.val_h5 = str(d / "val.h5")
    cfg.test_h5 = str(d / "test.h5")
    return cfg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Baseline MLP natural frequency model")
    parser.add_argument("--preset", default=RUN_PRESET, choices=list(PRESETS.keys()), help="Training preset")
    parser.add_argument("--data-dir", default=None, help="Dataset folder path containing train.h5/val.h5/test.h5")
    parser.add_argument("--device", default=None, help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Force override training epochs")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = build_config(args.preset, args.data_dir)
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs
    print(f"Using preset: {args.preset}")
    print(f"Dataset path: {Path(cfg.train_h5).parent}")
    print(f"Output directory: {cfg.output_dir}")
    train_frequency_model(cfg)

if __name__ == "__main__":
    main()
