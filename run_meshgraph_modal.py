# -*- coding: utf-8 -*-
"""MeshGraph 模态频率/模态留数训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "modal_residue" / "data_modal_residue_fixedclamp300"
OUT_DIR = ROOT_DIR / "runs" / "modal_residue_asinh_fixedclamp300"

EPOCHS = 300
# 0 表示全节点训练/评估。若显存不够，可改成 1024 或 2048。
QUERY_NODES = 0
EVAL_QUERY_NODES = 0
HIDDEN = 64
GNN_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

OMEGA_LOSS_WEIGHT = 1.0
RESIDUE_FULL_LOSS_WEIGHT = 1.0
TOP_AUX_LOSS_WEIGHT = 0.2
NODE_DOMINANT_LOSS_WEIGHT = 0.1
TOP_NODE_FRAC = 0.10
NODE_DOMINANT_K = 1
BEST_A_WEIGHT = 0.01
RESIDUE_VISIBLE_REL = 1e-3

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
        "--residue-full-loss-weight", str(RESIDUE_FULL_LOSS_WEIGHT),
        "--top-aux-loss-weight", str(TOP_AUX_LOSS_WEIGHT),
        "--node-dominant-loss-weight", str(NODE_DOMINANT_LOSS_WEIGHT),
        "--top-node-frac", str(TOP_NODE_FRAC),
        "--node-dominant-k", str(NODE_DOMINANT_K),
        "--best-a-weight", str(BEST_A_WEIGHT),
        "--residue-visible-rel", str(RESIDUE_VISIBLE_REL),
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
