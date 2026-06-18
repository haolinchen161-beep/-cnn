from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "modal_residue" / "data_modal_residue_fixedclamp300"
OUT_DIR = ROOT_DIR / "runs/下一步_R3_每阶A头_bottom"

# 下一步正式实验：R=3、底面点、每阶独立 A-head、每阶独立损失。
N_MODES_USED = 3
TARGET_REGION = "bottom"
KEY_QUERY_NODES = 256
EVAL_QUERY_NODES = 0

EPOCHS = 150
HIDDEN = 96
GNN_LAYERS = 3
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

OMEGA_LOSS_WEIGHT = 1.0
RESIDUE_FULL_LOSS_WEIGHT = 1.0
TOP_AUX_LOSS_WEIGHT = 0.25
NODE_DOMINANT_LOSS_WEIGHT = 0.10
TOP_NODE_FRAC = 0.10
NODE_DOMINANT_K = 1
RESIDUE_VISIBLE_REL = 1e-3
SIGN_VISIBLE_REL = 1e-4

LOG_EVERY = 1
SEED = 42
DEVICE = "cuda"
FP16 = True
PRELOAD = True

# 调试用；正式训练保持 0。
DEBUG_TRAIN_SAMPLES = 0
DEBUG_VAL_SAMPLES = 0
DEBUG_TEST_SAMPLES = 0


def main() -> int:
    from modal_residue.train_r3_per_mode_bottom import main as train_main

    argv = [
        "train_r3_per_mode_bottom.py",
        "--data-dir", str(DATA_DIR),
        "--out-dir", str(OUT_DIR),
        "--epochs", str(EPOCHS),
        "--n-modes-used", str(N_MODES_USED),
        "--target-region", TARGET_REGION,
        "--query-nodes", str(KEY_QUERY_NODES),
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
        "--residue-visible-rel", str(RESIDUE_VISIBLE_REL),
        "--sign-visible-rel", str(SIGN_VISIBLE_REL),
        "--grad-clip-norm", str(GRAD_CLIP_NORM),
        "--log-every", str(LOG_EVERY),
        "--seed", str(SEED),
        "--debug-train-samples", str(DEBUG_TRAIN_SAMPLES),
        "--debug-val-samples", str(DEBUG_VAL_SAMPLES),
        "--debug-test-samples", str(DEBUG_TEST_SAMPLES),
    ]
    if FP16:
        argv.append("--fp16")
    if PRELOAD:
        argv.append("--preload")
    else:
        argv.append("--no-preload")
    if DEVICE != "auto":
        argv += ["--device", DEVICE]

    print(">>> 启动下一步正式实验：R=3 + 每阶独立 A-head + 每阶独立 loss")
    old_argv = sys.argv
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
