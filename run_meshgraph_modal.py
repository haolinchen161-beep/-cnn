from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "modal_residue" / "data_modal_residue_fixedclamp300"

# 全数据训练。需要快速小样本排查时，再把下面三个值改成 10 或 3。
DEBUG_TRAIN_SAMPLES = 0
DEBUG_VAL_SAMPLES = 0
DEBUG_TEST_SAMPLES = 0
DEBUG_VAL_TEST_FROM_TRAIN = False

# 关键加速：把 train/val/test 需要读取的 HDF5 样本一次性加载到内存。
# 之前每个 epoch、每个样本都会重新打开 h5 + gzip 解压，所以看起来“卡住不动”。
PRELOAD_SPLITS = True

TARGET_REGION = "bottom"
KEY_QUERY_NODES = 256          # 训练时最多抽 256 个凹槽底面点；底面少于 256 时使用全部底面点，不重复补点。
EVAL_QUERY_NODES = 0           # 验证/测试用全部凹槽底面点。

OUT_DIR = ROOT_DIR / (
    f"runs/modal_residue_bottom_asinh_fixedclamp300_debug{DEBUG_TRAIN_SAMPLES}"
    if DEBUG_TRAIN_SAMPLES and DEBUG_TRAIN_SAMPLES > 0
    else "runs/modal_residue_bottom_asinh_fixedclamp300"
)

EPOCHS = 300
HIDDEN = 64                    # 先用轻量配置跑通底面目标；128 会明显更慢。
GNN_LAYERS = 2                 # 2 层优先，确认有效后再试 96x2 或 128x3。
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
SIGN_VISIBLE_REL = 1e-4

LOG_EVERY = 5
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
    sign_mean = float(metrics.get("A_sign_triplet", (0.0, 0.0, 0.0))[0])
    return float(y_rms + 0.05 * w_rms + 0.001 * a_vis_mean + 0.0002 * max(0.0, 100.0 - sign_mean))


def _limit_for_requested_split(split: str) -> int:
    if split == "train":
        return int(DEBUG_TRAIN_SAMPLES or 0)
    if split == "val":
        return int(DEBUG_VAL_SAMPLES or 0)
    if split == "test":
        return int(DEBUG_TEST_SAMPLES or 0)
    return 0


def install_cached_split(trainer) -> None:
    base_split = trainer.H5Split

    class CachedH5Split(base_split):
        def __init__(self, data_dir, split: str):
            requested_split = split
            source_split = "train" if DEBUG_VAL_TEST_FROM_TRAIN and split in {"val", "test"} else split
            super().__init__(data_dir, source_split)

            limit = _limit_for_requested_split(requested_split)
            if limit and limit > 0:
                self.keys = self.keys[: min(limit, len(self.keys))]

            self._cache = None
            if PRELOAD_SPLITS:
                total = len(self.keys)
                print(f">>> preload {requested_split} from {source_split}: {total} samples ...", flush=True)
                cache = []
                for i in range(total):
                    cache.append(base_split.__getitem__(self, i))
                    if (i + 1) % 20 == 0 or (i + 1) == total:
                        print(f"    loaded {requested_split}: {i + 1}/{total}", flush=True)
                self._cache = cache

        def __getitem__(self, i: int):
            if self._cache is not None:
                return self._cache[i]
            return super().__getitem__(i)

    trainer.H5Split = CachedH5Split


def main() -> int:
    import modal_residue.train_modal_residue_bottom_model as trainer

    trainer.modal_score = residue_first_modal_score
    install_cached_split(trainer)
    train_main = trainer.main

    argv = [
        "train_modal_residue_bottom_model.py",
        "--data-dir", str(DATA_DIR),
        "--out-dir", str(OUT_DIR),
        "--epochs", str(EPOCHS),
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
        "--best-a-weight", str(BEST_A_WEIGHT),
        "--residue-visible-rel", str(RESIDUE_VISIBLE_REL),
        "--sign-visible-rel", str(SIGN_VISIBLE_REL),
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
