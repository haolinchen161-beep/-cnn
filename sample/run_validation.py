from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM, collate_geometry_batch
from models import build_geometric_model
from training import evaluate_modal, train_modal


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Z-only MeshGraphNet 模态模型")
    parser.add_argument("--data_dir", default=os.path.join(ROOT, "ansys", "data"))
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "output_modal_zonly"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_modes", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--freq_weight", type=float, default=1.0)
    parser.add_argument("--phi_weight", type=float, default=1.0)
    parser.add_argument("--mac_weight", type=float, default=5.0)
    parser.add_argument("--scale_weight", type=float, default=1.0)
    parser.add_argument("--min_mode_weight", type=float, default=0.2)
    parser.add_argument("--knn_k", type=int, default=12)
    return parser.parse_args()


def build_config(args):
    return {
        "n_modes": args.n_modes,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freq_weight": args.freq_weight,
        "phi_weight": args.phi_weight,
        "mac_weight": args.mac_weight,
        "scale_weight": args.scale_weight,
        "min_mode_weight": args.min_mode_weight,
        "gradient_clip": 2.0,
        "graph": {"knn_k": args.knn_k},
    }


def make_loader(dataset, batch_size, shuffle, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_geometry_batch,
        generator=gen if shuffle else None,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = build_config(args)
    run_args = SimpleNamespace(device=args.device, dir=args.out_dir)

    print("=" * 80)
    print("Z 向模态 MeshGraphNet 训练")
    print("训练目标：固有频率 omega + 全节点 Z 向振型 phi_z")
    print("本阶段不训练阻尼和 FRF。")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.out_dir}")

    trainset = GraphHDF5Dataset(["train.h5"], cfg, data_dir=args.data_dir, normalization=True, test=False)
    valset = GraphHDF5Dataset(["val.h5"], cfg, data_dir=args.data_dir, normalization=True, test=True)
    testset = GraphHDF5Dataset(["test.h5"], cfg, data_dir=args.data_dir, normalization=True, test=True)

    trainloader = make_loader(trainset, args.batch_size, True, args.seed)
    valloader = make_loader(valset, 1, False, args.seed)
    testloader = make_loader(testset, 1, False, args.seed)

    batch0 = next(iter(trainloader))
    print(f"训练/验证/测试样本数: {len(trainset)} / {len(valset)} / {len(testset)}")
    print(f"节点特征: {tuple(batch0['node_features'].shape)}")
    print(f"图边索引: {tuple(batch0['edge_index'].shape)}")
    print(f"频率标签: {tuple(batch0['modal_omega_phys'].shape)}")
    print(f"Z 向振型标签: {tuple(batch0['modal_phi_z'].shape)}")
    print(f"完整三向振型仅用于加权: {tuple(batch0['modal_phi_xyz'].shape)}")

    model = build_geometric_model(
        encoder_kwargs={
            "node_in_dim": NODE_FEATURE_DIM,
            "edge_in_dim": 4,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "n_modes": args.n_modes,
            "dropout": args.dropout,
        },
        decoder_kwargs={},
    )
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    model = train_modal(run_args, cfg, model, trainloader, valloader)
    metrics = evaluate_modal(run_args, cfg, model, testloader, verbose=True)

    print("最终测试指标:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6g}")
    return 0


if __name__ == "__main__":
    main()
