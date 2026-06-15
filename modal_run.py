import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import NODE_FEATURE_DIM, GraphHDF5Dataset, collate_geometry_batch
from models import build_geometric_model
from training.modal_trainer_simple import evaluate_modal, train_modal

class Args:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir = os.path.join("sample", "output_mesh_modal_lite")

def main():
    cfg = {"epochs": 200, "lr": 3e-4, "weight_decay": 1e-2, "gradient_clip": 2.0, "freq_weight": 1.0, "phi_weight": 1.0, "mac_weight": 5.0, "graph": {"knn_k": 12}, "omega_scale": 2.0 * np.pi * 5000.0}
    data_dir = os.path.join("ansys", "data")
    train_set = GraphHDF5Dataset(["train.h5"], cfg, data_dir=data_dir)
    val_set = GraphHDF5Dataset(["val.h5"], cfg, data_dir=data_dir)
    test_set = GraphHDF5Dataset(["test.h5"], cfg, data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_geometry_batch)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_geometry_batch)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_geometry_batch)
    model = build_geometric_model({"node_in_dim": NODE_FEATURE_DIM, "edge_in_dim": 4, "hidden": 128, "n_layers": 6, "n_modes": 3, "dropout": 0.05})
    args = Args()
    model = train_modal(args, cfg, model, train_loader, val_loader)
    result = evaluate_modal(args, cfg, model, test_loader, verbose=True)
    os.makedirs(args.dir, exist_ok=True)
    np.savez(os.path.join(args.dir, "final_results.npz"), **{k: np.array(v) for k, v in result.items()})

if __name__ == "__main__":
    main()
