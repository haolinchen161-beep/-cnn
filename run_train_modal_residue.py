# -*- coding: utf-8 -*-
"""
模态留数 FRF 训练入口。

用法：
    F:/pytorch_cuda12/python.exe -B run_train_modal_residue.py

说明：
1. 所有常用训练参数集中放在本文件顶部；
2. 不使用 config.yaml / json 配置文件；
3. 先执行数据集检查，再启动训练；
4. 真正的模型、损失、训练循环仍在 modal_residue/train_modal_residue_model.py 中。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ===================== 路径参数 =====================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_modal_residue_filtered"
OUT_DIR = ROOT_DIR / "runs" / "modal_residue_baseline"

TRAIN_SCRIPT = ROOT_DIR / "modal_residue" / "train_modal_residue_model.py"
VALIDATE_SCRIPT = ROOT_DIR / "modal_residue" / "validate_dataset.py"


# ===================== 是否先检查数据 =====================
VALIDATE_BEFORE_TRAIN = True
MIN_RELATIVE_GAP = 0.03


# ===================== 训练参数 =====================
EPOCHS = 300
QUERY_NODES = 512
EVAL_QUERY_NODES = 1024

HIDDEN = 192
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

FRF_LOSS_WEIGHT = 0.05
LOG_EVERY = 10
SEED = 42

# DEVICE 可选："cuda"、"cpu"。
# 如果写成 "auto"，会交给 train_modal_residue_model.py 自动选择。
DEVICE = "cuda"


# ===================== 辅助函数 =====================
def run_cmd(cmd: list[str]) -> None:
    print("\n>>> 执行命令:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"找不到训练脚本: {TRAIN_SCRIPT}")
    if VALIDATE_BEFORE_TRAIN and not VALIDATE_SCRIPT.exists():
        raise FileNotFoundError(f"找不到数据检查脚本: {VALIDATE_SCRIPT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if VALIDATE_BEFORE_TRAIN:
        validate_cmd = [
            sys.executable,
            "-B",
            str(VALIDATE_SCRIPT),
            "--data-dir",
            str(DATA_DIR),
            "--min-relative-gap",
            str(MIN_RELATIVE_GAP),
        ]
        run_cmd(validate_cmd)

    train_cmd = [
        sys.executable,
        "-B",
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(DATA_DIR),
        "--out-dir",
        str(OUT_DIR),
        "--epochs",
        str(EPOCHS),
        "--query-nodes",
        str(QUERY_NODES),
        "--eval-query-nodes",
        str(EVAL_QUERY_NODES),
        "--hidden",
        str(HIDDEN),
        "--lr",
        str(LEARNING_RATE),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--frf-loss-weight",
        str(FRF_LOSS_WEIGHT),
        "--grad-clip-norm",
        str(GRAD_CLIP_NORM),
        "--log-every",
        str(LOG_EVERY),
        "--seed",
        str(SEED),
    ]

    if DEVICE != "auto":
        train_cmd += ["--device", DEVICE]

    run_cmd(train_cmd)


if __name__ == "__main__":
    main()
