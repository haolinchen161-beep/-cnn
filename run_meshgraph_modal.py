# -*- coding: utf-8 -*-
"""MeshGraph 模态频率/模态留数训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_modal_residue_filtered"
OUT_DIR = ROOT_DIR / "runs" / "modal_residue_meshgraph"

EPOCHS = 300
QUERY_NODES = 512
EVAL_QUERY_NODES = 1024
HIDDEN = 192
GNN_LAYERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
OMEGA_LOSS_WEIGHT = 1.0
PHI_LOSS_WEIGHT = 1.0
LOG_EVERY = 10
SEED = 42
DEVICE = "cuda"
FP16 = True


def main() -> int:
    from modal_residue.train_modal_residue_model import main as train_main

    argv = [
        "train_modal_residue_model.py",
        "--data-dir", str(DATA_DIR),
        "--out-dir", str(OUT_DIR),
        "--epochs", str(EPOCHS),
        "--query-nodes", str(QUERY_NODES),
        "--eval-query-nodes", str(EVAL_QUERY_NODES),
        "--hidden", str(HIDDEN),
        "--gnn-layers", str(GNN_LAYERS),
        "--lr", str(LEARNING_RATE),
        "--weight-decay", str(WEIGHT_DECAY),
        "--omega-loss-weight", str(OMEGA_LOSS_WEIGHT),
        "--phi-loss-weight", str(PHI_LOSS_WEIGHT),
        "--grad-clip-norm", str(GRAD_CLIP_NORM),
        "--log-every", str(LOG_EVERY),
        "--seed", str(SEED),
    ]
    if FP16:
        argv.append("--fp16")
    if DEVICE != "auto":
        argv += ["--device", DEVICE]

    old_argv = sys.argv
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
