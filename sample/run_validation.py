from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM, collate_geometry_batch
from models import build_geometric_model
from training import evaluate, train


CONFIG = {
    "epochs": 300,
    "validation_frequency": 5,
    "progress_interval": 0,
    "omega_pretrain_epochs": 30,
    "omega_loss_weight": 1.0,
    "phi_loss_weight": 3.0,
    "mac_weight": 5.0,
    "scale_weight": 1.0,
    "n_modes": 3,
    "graph": {"knn_k": 12},
    "gradient_clip": 2.0,
    "optimizer": {
        "name": "AdamW",
        "kwargs": {"lr": 5e-4, "weight_decay": 0.01, "betas": (0.9, 0.999)},
    },
}


MODEL_CFG = {
    "encoder_kwargs": {
        "node_in_dim": NODE_FEATURE_DIM,
        "edge_in_dim": 4,
        "hidden": 128,
        "n_layers": 4,
        "n_modes": 3,
        "dropout": 0.05,
    },
    "decoder_kwargs": {},
}


class Args:
    def __init__(self):
        self.batch_size = 1
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = True
        self.dir = os.path.join(os.path.dirname(__file__), "output_meshgraphnet_zonly")


def make_loader(dataset, batch_size, shuffle, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_geometry_batch,
        generator=gen if shuffle else None,
    )


def main():
    print("=" * 80, flush=True)
    print("FEM-aware MeshGraphNet Z-only modal training", flush=True)
    print("Target: omega + node-wise phi_z; no zeta, no FRF, no branch loss", flush=True)
    print("=" * 80, flush=True)

    args = Args()
    torch.manual_seed(args.seed)
    data_dir = os.path.join(ROOT, "ansys", "data")
    os.makedirs(args.dir, exist_ok=True)

    print(f"Device: {args.device}, batch_size={args.batch_size}, fp16={args.fp16}", flush=True)
    print(f"Data dir: {data_dir}", flush=True)
    print(f"Output dir: {args.dir}", flush=True)
    print(f"Config: hidden={MODEL_CFG['encoder_kwargs']['hidden']}, layers={MODEL_CFG['encoder_kwargs']['n_layers']}, knn_k={CONFIG['graph']['knn_k']}", flush=True)
    print(f"Schedule: first {CONFIG['omega_pretrain_epochs']} epochs omega-only, then omega+phi_z", flush=True)
    print("Progress: epoch-only logging; batch progress logs disabled", flush=True)

    trainset = GraphHDF5Dataset(["train.h5"], CONFIG, data_dir=data_dir, normalization=True, test=False)
    valset = GraphHDF5Dataset(["val.h5"], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset = GraphHDF5Dataset(["test.h5"], CONFIG, data_dir=data_dir, normalization=True, test=True)

    trainloader = make_loader(trainset, args.batch_size, True, args.seed)
    valloader = make_loader(valset, 1, False, args.seed)
    testloader = make_loader(testset, 1, False, args.seed)

    batch = next(iter(trainloader))
    print(f"Train/Val/Test: {len(trainset)} / {len(valset)} / {len(testset)} samples", flush=True)
    print(f"Nodes: {tuple(batch['node_features'].shape)}, Edges: {tuple(batch['edge_index'].shape)}", flush=True)
    print(f"edge_attr: {tuple(batch['edge_attr'].shape)}, node_dim={batch['node_features'].shape[-1]}", flush=True)
    print(f"omega: {tuple(batch['modal_omega_phys'].shape)}, phi_z: {tuple(batch['modal_phi_z'].shape)}, phi_xyz: {tuple(batch['modal_phi_xyz'].shape)}", flush=True)

    net = build_geometric_model(MODEL_CFG["encoder_kwargs"], MODEL_CFG["decoder_kwargs"]).to(args.device)
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}", flush=True)

    with torch.no_grad():
        batch_dev = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = net(batch_dev["node_features"], batch_dev["edge_index"], batch_dev["edge_attr"], batch_dev["batch"], compute_phi=True)
    print(f"Init forward: omega={tuple(out['omega'].shape)}, phi_z={tuple(out['phi_z'].shape)}", flush=True)

    optimizer = torch.optim.AdamW(net.parameters(), **CONFIG["optimizer"]["kwargs"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    net = train(args, CONFIG, MODEL_CFG, net, trainloader, optimizer, valloader, scheduler=scheduler)
    evaluate(args, CONFIG, net, testloader, compute_phi=True, verbose=True)
    return 0


if __name__ == "__main__":
    main()
