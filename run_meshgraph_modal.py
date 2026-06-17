from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "modal_residue" / "data_modal_residue_fixedclamp300"

#  恢复全数据训练时将下面DEBUG_TRAIN_SAMPLES改成0 
DEBUG_TRAIN_SAMPLES = 1
DEBUG_VAL_SAMPLES = 1
DEBUG_TEST_SAMPLES = 1
DEBUG_VAL_TEST_FROM_TRAIN = True

OUT_DIR = ROOT_DIR / (
    "runs/modal_residue_asinh_fixedclamp300_debug1"
    if DEBUG_TRAIN_SAMPLES and DEBUG_TRAIN_SAMPLES > 0
    else "runs/modal_residue_asinh_fixedclamp300"
)

EPOCHS = 300
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

LOG_EVERY = 1
SEED = 42
DEVICE = "cuda"
FP16 = True


def residue_first_modal_score(metrics, best_a_weight):
    y_triplet = metrics.get("Y_smooth_l1_triplet")
    if y_triplet is None:
        w_rms = float(metrics.get("w10_triplet", (0.0, 0.0, 0.0))[2])
        a_vis_rms = float(metrics.get("A_vis_triplet", (0.0, 0.0, 0.0))[2])
        return float(w_rms + float(best_a_weight) * a_vis_rms)
    y_rms = float(y_triplet[2])
    w_rms = float(metrics.get("w10_triplet", (0.0, 0.0, 0.0))[2])
    a_vis_mean = float(metrics.get("A_vis_triplet", (0.0, 0.0, 0.0))[0])
    return float(y_rms + 0.05 * w_rms + 0.001 * a_vis_mean)


def install_debug_split(trainer) -> None:
    if not DEBUG_TRAIN_SAMPLES or DEBUG_TRAIN_SAMPLES <= 0:
        return

    base_split = trainer.H5Split

    class LimitedH5Split(base_split):
        def __init__(self, data_dir, split: str):
            source_split = "train" if DEBUG_VAL_TEST_FROM_TRAIN and split in {"val", "test"} else split
            super().__init__(data_dir, source_split)
            if split == "train":
                limit = DEBUG_TRAIN_SAMPLES
            elif split == "val":
                limit = DEBUG_VAL_SAMPLES
            elif split == "test":
                limit = DEBUG_TEST_SAMPLES
            else:
                limit = 0
            if limit and limit > 0:
                self.keys = self.keys[: min(int(limit), len(self.keys))]

    trainer.H5Split = LimitedH5Split


def main() -> int:
    import modal_residue.train_modal_residue_model as trainer

    trainer.modal_score = residue_first_modal_score
    install_debug_split(trainer)
    train_main = trainer.main

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
