# -*- coding: utf-8 -*-
"""
模态留数训练入口。

用法：
    F:/pytorch_cuda12/python.exe -B run_train_modal_residue.py

说明：
1. 所有常用训练参数集中放在本文件顶部；
2. 不使用 config.yaml / json 配置文件；
3. 先执行数据集检查，再启动训练；
4. 训练目标只有 modal_omega 与 modal_residue_z，不依赖 point_frf。
"""
from __future__ import annotations

import os
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

OMEGA_LOSS_WEIGHT = 1.0
PHI_LOSS_WEIGHT = 1.0

LOG_EVERY = 10
SEED = 42

# DEVICE 可选："cuda"、"auto"、"cpu"。
DEVICE = "cuda"

# 与 gnn-meshgraphnet-refactor 分支一致，默认开启 CUDA AMP。
FP16 = True


# ===================== CUDA DLL 路径修复 =====================
def prepare_cuda_dll_path() -> None:
    py_root = Path(sys.executable).resolve().parent
    candidates = [
        py_root,
        py_root / "Library" / "bin",
        py_root / "Lib" / "site-packages" / "torch" / "lib",
        py_root / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
        py_root / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
    ]

    existing = [str(p) for p in candidates if p.exists()]
    if existing:
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(existing + [old_path])
        if hasattr(os, "add_dll_directory"):
            for p in existing:
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass


# ===================== 辅助函数 =====================
def run_cmd(cmd: list[str]) -> None:
    print("\n>>> 执行命令:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    prepare_cuda_dll_path()

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
        "--omega-loss-weight",
        str(OMEGA_LOSS_WEIGHT),
        "--phi-loss-weight",
        str(PHI_LOSS_WEIGHT),
        "--grad-clip-norm",
        str(GRAD_CLIP_NORM),
        "--log-every",
        str(LOG_EVERY),
        "--seed",
        str(SEED),
    ]

    if FP16:
        train_cmd += ["--fp16"]
    if DEVICE != "auto":
        train_cmd += ["--device", DEVICE]

    run_cmd(train_cmd)


if __name__ == "__main__":
    main()
