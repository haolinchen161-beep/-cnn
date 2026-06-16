from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import GraphHDF5Dataset, NODE_FEATURE_DIM, collate_geometry_batch
from models import build_geometric_model
from training import evaluate_modal


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate z-only modal MeshGraphNet")
    parser.add_argument("--data_dir", default=os.path.join(ROOT, "ansys", "data"))
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "output_modal_zonly"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_modes", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--knn_k", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {
        "n_modes": args.n_modes,
        "graph": {"knn_k": args.knn_k},
        "freq_weight": 1.0,
        "phi_weight": 1.0,
        "mac_weight": 5.0,
        "scale_weight": 1.0,
        "min_mode_weight": 0.2,
    }
    run_args = SimpleNamespace(device=args.device, dir=args.out_dir)
    ckpt_path = args.checkpoint or os.path.join(args.out_dir, "checkpoint_best.pt")

    testset = GraphHDF5Dataset(["test.h5"], cfg, data_dir=args.data_dir, normalization=True, test=True)
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_geometry_batch,
    )

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
    ).to(args.device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics = evaluate_modal(run_args, cfg, model, testloader, verbose=True)

    pred_omega, true_omega = [], []
    pred_phi_z, true_phi_z = [], []
    sample_names = []
    with torch.no_grad():
        for batch in testloader:
            batch_dev = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch_dev["node_features"], batch_dev["edge_index"], batch_dev["edge_attr"], batch_dev["batch"])
            pred_omega.append(out["omega"].detach().cpu().numpy())
            true_omega.append(batch["modal_omega_phys"].numpy())
            pred_phi_z.append(out["phi_z"].detach().cpu().numpy())
            true_phi_z.append(batch["modal_phi_z"].numpy())
            sample_names.append(str(batch.get("sample_name", "sample")))

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "zonly_eval_results.npz"),
        pred_omega=np.array(pred_omega, dtype=object),
        true_omega=np.array(true_omega, dtype=object),
        pred_phi_z=np.array(pred_phi_z, dtype=object),
        true_phi_z=np.array(true_phi_z, dtype=object),
        sample_names=np.array(sample_names, dtype=object),
        metrics=np.array(metrics, dtype=object),
    )

    print("Final metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6g}")
    print(f"Saved: {os.path.join(args.out_dir, 'zonly_eval_results.npz')}")


if __name__ == "__main__":
    main()
