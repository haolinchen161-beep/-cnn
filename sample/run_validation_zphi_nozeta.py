"""MeshGraphNet training entrypoint for Z-dominant + phi_z-only + no-zeta supervision.

Pipeline:
    1) Build ansys/data_z_dominant with ansys/filter_z_dominant_dataset.py.
    2) Run this script to train only omega and Z-projection mode shapes.

This channel intentionally does NOT supervise damping and does NOT supervise
phi_x/phi_y. It is intended for Z-direction FRF studies where the physically
important modal quantities are omega_k and phi_z(node,k).

Recommended command:
    F:/pytorch_cuda12/python.exe -B sample/run_validation_zphi_nozeta.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM, collate_geometry_batch
from models import build_geometric_model
from training import train, evaluate, modal_loss_z_only


CONFIG = {
    "epochs": 160,
    "validation_frequency": 5,
    # Z-dominant + phi_z-only is simpler; graph encoder can learn from epoch 0.
    "omega_pretrain_epochs": 15,
    "omega_prior_only_epochs": 0,
    # Modal-only first. FRF metrics are disabled because zeta is not trained here.
    "phase2_min_epoch": 9999,
    "phase2_omega_tune_epochs": 0,
    "phase2a_lr": 1e-4,
    "frf_teacher_epochs": 0,
    "omega_loss_weight": 1.0,
    "zeta_loss_weight": 0.0,
    "phi_loss_weight": 3.0,
    "branch_loss_weight": 0.0,
    "frf_loss_weight": 0.0,
    "frf_warmup_epochs": 20,
    "phi_loss_mode": "z_only",
    "disable_zeta_training": True,
    "modal_score_zeta_weight": 0.0,
    "evaluate_frf": False,
    "freq_min": 1.0,
    "freq_max": 5000.0,
    "omega_max": 32000.0,
    "data_path_train": ["train.h5"],
    "data_path_val": ["val.h5"],
    "data_path_test": ["test.h5"],
    "filter_g32": False,
    "graph": {"knn_k": 12},
    "optimizer": {
        "name": "AdamW",
        "kwargs": {"lr": 5e-4, "weight_decay": 0.01, "betas": (0.9, 0.999)},
        "gradient_clip": 2.0,
    },
    "plateau_patience": 6,
    "plateau_factor": 0.5,
}

MODEL_CFG = {
    "encoder_kwargs": {
        "node_in_dim": NODE_FEATURE_DIM,
        "edge_in_dim": 4,
        "hidden": 128,
        "n_layers": 4,
        "n_modes": 3,
        "amp_scale": 500000.0,
        "freq_min": 1.0,
        "freq_max": 5000.0,
        "dropout": 0.05,
    },
    "decoder_kwargs": {},
}


class SimpleArgs:
    def __init__(self):
        self.batch_size = 1
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = True
        self.dir = os.path.join(os.path.dirname(__file__), "output_meshgraphnet_zphi_nozeta")
        self.debug = False


def main():
    print("=" * 80)
    print("FEM-aware MeshGraphNet | Z-dominant, phi_z-only, no zeta supervision")
    print("=" * 80)
    args = SimpleArgs()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data_z_dominant")
    print(f"Device: {args.device}, batch_size={args.batch_size}, fp16={args.fp16}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {args.dir}")
    print("Training target: omega + phi_z only. Zeta and phi_x/phi_y are not supervised.")

    trainset = GraphHDF5Dataset(CONFIG["data_path_train"], CONFIG, data_dir=data_dir, normalization=True, test=False)
    valset = GraphHDF5Dataset(CONFIG["data_path_val"], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset = GraphHDF5Dataset(CONFIG["data_path_test"], CONFIG, data_dir=data_dir, normalization=True, test=True)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_geometry_batch, generator=gen,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, drop_last=False, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_geometry_batch,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, drop_last=False, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_geometry_batch,
    )

    batch = next(iter(trainloader))
    print(f"Train/Val/Test: {len(trainset)} / {len(valset)} / {len(testset)} samples")
    print(f"Nodes: {tuple(batch['node_features'].shape)}, Edges: {tuple(batch['edge_index'].shape)}")
    print(f"edge_attr: {tuple(batch['edge_attr'].shape)}, node_dim={batch['node_features'].shape[-1]}")
    print(f"FRF: {tuple(batch['point_frf'].shape)}, frequencies: {tuple(batch['frequencies'].shape)}")
    print(f"modal_phi: {tuple(batch['modal_phi'].shape)}, modal_omega_phys: {tuple(batch['modal_omega_phys'].shape)}")

    net = build_geometric_model(MODEL_CFG["encoder_kwargs"], MODEL_CFG["decoder_kwargs"]).to(args.device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Params: {total_params:,}")

    net.eval()
    with torch.no_grad():
        batch_dev = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        frf_p, omega_p, log_zeta_p, zeta_p, phi_p = net(
            batch_dev["node_features"], batch_dev["edge_index"], batch_dev["edge_attr"], batch_dev["batch"],
            frequencies=batch_dev["frequencies"],
            phi_exc=batch_dev.get("modal_phi_exc"),
            excitation_index_global=batch_dev.get("excitation_index_global"),
        )
    print(f"FRF={tuple(frf_p.shape)}, omega={tuple(omega_p.shape)}, zeta={tuple(zeta_p.shape)}, phi={tuple(phi_p.shape)}")

    with torch.no_grad():
        init_loss, l_w, l_z, l_p, mac = modal_loss_z_only(
            omega_p, batch_dev["modal_omega_phys"],
            log_zeta_p, batch_dev["modal_zeta"],
            phi_p, batch_dev["modal_phi"],
            batch_idx=batch_dev["batch"],
            omega_weight=CONFIG["omega_loss_weight"],
            zeta_weight=CONFIG["zeta_loss_weight"],
            phi_weight=CONFIG["phi_loss_weight"],
        )
    print(f"Init loss={init_loss.item():.4e}, omega={l_w.item():.4e}, zeta={l_z.item():.4e}, phi_z={l_p.item():.4e}, MAC_z={mac.detach().cpu().numpy()}")

    optimizer = torch.optim.AdamW(net.parameters(), **CONFIG["optimizer"]["kwargs"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CONFIG["plateau_patience"], factor=CONFIG["plateau_factor"], min_lr=1e-6,
    )

    start_epoch = 0
    ckpt_path = os.path.join(args.dir, "checkpoint_last")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resume from epoch {start_epoch}")

    t0 = time.time()
    net = train(args, CONFIG, MODEL_CFG, net, trainloader, optimizer, valloader,
                scheduler=scheduler, logger=None, start_epoch=start_epoch)
    elapsed = time.time() - t0

    best_path = os.path.join(args.dir, "checkpoint_best_modal")
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=args.device)["model_state_dict"])
    results = evaluate(args, CONFIG, net, testloader, verbose=True)

    print("\n" + "=" * 80)
    print(f"Done | Params={total_params:,} | Train time={elapsed:.0f}s")
    print(f"Test modal score={results.get('val_modal_score', -1):.6g}")
    print(f"Test w={results.get('val_w', [])}")
    print(f"Test z(ignored)={results.get('val_z', [])}")
    print(f"Test MAC_z={results.get('val_mac', [])}")
    print(f"Test phiN_z={results.get('val_phi_n', [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
