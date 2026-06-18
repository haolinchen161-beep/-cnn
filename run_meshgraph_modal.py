from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "modal_residue" / "data_modal_residue_fixedclamp300"

# 下一步实验：先只学前 3 阶，验证 A 是否能在低阶模态上稳定下降。
N_MODES_USED = 3

# 需要快速小样本排查时，把下面三个值改成 10 或 3。
DEBUG_TRAIN_SAMPLES = 0
DEBUG_VAL_SAMPLES = 0
DEBUG_TEST_SAMPLES = 0
DEBUG_VAL_TEST_FROM_TRAIN = False

# 读入内存，避免每个 epoch 重复打开 HDF5 和 gzip 解压。
PRELOAD_SPLITS = True

TARGET_REGION = "bottom"
KEY_QUERY_NODES = 256
EVAL_QUERY_NODES = 0

OUT_DIR = ROOT_DIR / (
    f"runs/下一步_R{N_MODES_USED}_每阶A头_bottom_debug{DEBUG_TRAIN_SAMPLES}"
    if DEBUG_TRAIN_SAMPLES and DEBUG_TRAIN_SAMPLES > 0
    else f"runs/下一步_R{N_MODES_USED}_每阶A头_bottom"
)

# 不再继续原 10 阶模型硬训；先用 100~150 epoch 验证低阶 A 是否可学。
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
BEST_A_WEIGHT = 0.01
RESIDUE_VISIBLE_REL = 1e-3
SIGN_VISIBLE_REL = 1e-4

LOG_EVERY = 1
SEED = 42
DEVICE = "cuda"
FP16 = True


def residue_first_modal_score(metrics, best_a_weight):
    """更偏向 A 的验证分数：先看 signed-asinh 的 Y loss，再看频率和可见 A。"""
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


def _trim_modes(sample: dict, n_modes: int) -> dict:
    """训练入口层面只取前 n_modes 阶，不重写 HDF5 数据集。"""
    if n_modes <= 0:
        return sample
    if "modal_omega" in sample:
        sample["modal_omega"] = sample["modal_omega"][:n_modes]
    if "modal_residue_z" in sample:
        sample["modal_residue_z"] = sample["modal_residue_z"][:, :n_modes]
    if "modal_phi_xyz" in sample:
        sample["modal_phi_xyz"] = sample["modal_phi_xyz"][:, :n_modes, :]
    if "modal_phi" in sample:
        sample["modal_phi"] = sample["modal_phi"][:, :n_modes, ...]
    if "modal_zeta" in sample:
        sample["modal_zeta"] = sample["modal_zeta"][:n_modes]
    return sample


def install_cached_limited_split(trainer) -> None:
    base_split = trainer.H5Split

    class CachedLimitedH5Split(base_split):
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
                    item = _trim_modes(base_split.__getitem__(self, i), N_MODES_USED)
                    cache.append(item)
                    if (i + 1) % 20 == 0 or (i + 1) == total:
                        print(f"    loaded {requested_split}: {i + 1}/{total}", flush=True)
                self._cache = cache

        def __getitem__(self, i: int):
            if self._cache is not None:
                return self._cache[i]
            return _trim_modes(super().__getitem__(i), N_MODES_USED)

    trainer.H5Split = CachedLimitedH5Split


def install_per_mode_residue_model(base) -> None:
    """把原来一个 head 输出所有 A，改成每阶一个独立 A-head。"""

    class PerModeResidueNet(nn.Module):
        def __init__(self, node_in_dim: int, edge_in_dim: int, n_modes: int, hidden: int = 96, gnn_layers: int = 3):
            super().__init__()
            self.n_modes = int(n_modes)
            self.node_encoder = base.mlp(node_in_dim, hidden, hidden, layers=3)
            self.edge_encoder = base.mlp(edge_in_dim, hidden, hidden, layers=3)
            self.blocks = nn.ModuleList([base.MeshGraphBlock(hidden) for _ in range(gnn_layers)])
            self.global_mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            )
            self.omega_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.n_modes)
            )
            head_in = 3 * hidden + 6
            self.residue_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(self.n_modes)
            ])

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                    coords_norm: torch.Tensor, q: torch.Tensor, exc_idx: torch.Tensor):
            h = self.node_encoder(x)
            e = self.edge_encoder(edge_attr).to(dtype=h.dtype)
            for block in self.blocks:
                h, e = block(h, e, edge_index)

            g_raw = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0)
            g = self.global_mlp(g_raw)
            omega_norm = self.omega_head(g)

            exc_i = torch.clamp(exc_idx.long(), 0, h.shape[0] - 1)
            hq = h[q]
            he = h[exc_i].view(1, -1).expand(hq.shape[0], -1)
            gg = g.view(1, -1).expand(hq.shape[0], -1)
            q_xyz = coords_norm[q].to(dtype=h.dtype)
            rel_xyz = q_xyz - coords_norm[exc_i].view(1, 3).to(dtype=h.dtype)
            residue_input = torch.cat([hq, gg, he, q_xyz, rel_xyz], dim=-1)
            residue_y = torch.cat([head(residue_input) for head in self.residue_heads], dim=-1)
            return omega_norm, residue_y

    base.MeshGraphModalResidueNet = PerModeResidueNet


def main() -> int:
    import modal_residue.train_modal_residue_model as base
    import modal_residue.train_modal_residue_bottom_model as trainer

    install_per_mode_residue_model(base)
    trainer.modal_score = residue_first_modal_score
    install_cached_limited_split(trainer)
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

    print(f">>> 下一步实验: 仅使用前 {N_MODES_USED} 阶；每阶独立 A-head；target_region={TARGET_REGION}")
    old_argv = sys.argv
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
