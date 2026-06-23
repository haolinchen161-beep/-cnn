# -*- coding: utf-8 -*-
"""Precompute frequency and shape norm priors for damping training."""
import sys
import os
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Setup paths to import local modules
CURRENT_DIR = Path(__file__).resolve().parent
BEST_MODIFIED_DIR = CURRENT_DIR.parent
PROJECT_ROOT = BEST_MODIFIED_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BEST_MODIFIED_DIR / "training_shape_optimized"))
sys.path.insert(0, str(BEST_MODIFIED_DIR / "training_frequency_optimized"))

from dataset import SymmetricSymlogModalDataset
from model import SymmetricSymlogModalOperator
from model_frequency import FrequencyTokenMLP as FrequencyModel

def precompute_file_priors(h5_path: Path, output_pt_path: Path, shape_model, freq_model, freq_stats, device):
    print(f"Loading dataset from {h5_path}...")
    # Load with query_per_sample=-1 to get all nodes for accurate shape L2 norms
    dataset = SymmetricSymlogModalDataset(h5_path=h5_path, target_modes=3, query_per_sample=-1, random_query=False)
    
    priors_dict = {}
    eps = 1e-6
    
    # Enable evaluation mode
    shape_model.eval()
    freq_model.eval()
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Predicting {h5_path.name}"):
            item = dataset[idx]
            sample_key = dataset.sample_keys[idx]
            
            # Prepare inputs
            pocket = item["pocket_features"].unsqueeze(0).to(device)
            clamp = item["clamp_features"].unsqueeze(0).to(device)
            gf = item["global_features"].unsqueeze(0).to(device)
            q_coord = item["q_coord"].unsqueeze(0).to(device)
            q_node = item["q_node_features"].unsqueeze(0).to(device)
            p_coord = item["p_coord"].to(device).unsqueeze(0).unsqueeze(1)
            p_node = item["p_node_features"].to(device).unsqueeze(0).unsqueeze(1)
            
            all_coords = torch.cat([q_coord, p_coord], dim=1)
            all_nodes = torch.cat([q_node, p_node], dim=1)
            
            # 1. Predict Shape & Freq from Shape Model (we only need the shape norm)
            outputs = shape_model(pocket, clamp, gf, all_coords, all_nodes)
            phi_all_pred = outputs["symlog_phi_z"].squeeze(0).cpu().numpy() # [N+1, 3]
            phi_z_norm_pred = np.linalg.norm(phi_all_pred, axis=0) # [3]
            
            # 2. Predict Freq from Dedicated Frequency Model (higher accuracy)
            p_norm = (pocket - freq_stats["pocket_features_mean"].view(1, 1, -1)) / freq_stats["pocket_features_std"].view(1, 1, -1).clamp_min(eps)
            c_norm = (clamp - freq_stats["clamp_features_mean"].view(1, 1, -1)) / freq_stats["clamp_features_std"].view(1, 1, -1).clamp_min(eps)
            g_norm = (gf - freq_stats["global_features_mean"].view(1, -1)) / freq_stats["global_features_std"].view(1, -1).clamp_min(eps)
            
            pred_f_norm = freq_model(p_norm, c_norm, g_norm)
            logw = pred_f_norm * freq_stats["omega_log_std"].view(1, -1) + freq_stats["omega_log_mean"].view(1, -1)
            omega_pred = torch.exp(logw).clamp_min(eps).squeeze(0).cpu().numpy() # [3] rad/s
            
            priors_dict[sample_key] = {
                "omega_pred": omega_pred,
                "phi_z_norm_pred": phi_z_norm_pred
            }
            
    # Save the dictionary
    print(f"Saving priors to {output_pt_path}...")
    torch.save(priors_dict, output_pt_path)
    print("Done!")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    shape_ckpt = BEST_MODIFIED_DIR / "training_shape_optimized" / "checkpoints" / "best_model.pth"
    freq_ckpt = BEST_MODIFIED_DIR / "training_frequency_optimized" / "runs" / "best_frequency_model.pt"
    
    if not shape_ckpt.exists():
        print(f"Error: Shape model checkpoint not found at {shape_ckpt}")
        sys.exit(1)
    if not freq_ckpt.exists():
        print(f"Error: Frequency model checkpoint not found at {freq_ckpt}")
        sys.exit(1)
        
    # Load frequency model
    print("Loading frequency model...")
    ckpt_f = torch.load(freq_ckpt, map_location=device)
    freq_model = FrequencyModel(
        pocket_dim=8, clamp_dim=11, global_dim=9,
        token_dim=96, hidden_dim=192, fusion_dim=256,
        out_modes=3, token_layers=3, fusion_layers=4, dropout=0.05
    ).to(device)
    freq_model.load_state_dict(ckpt_f["model_state"], strict=False)
    freq_stats = {k: v.to(device) for k, v in ckpt_f["stats"].items()}
    
    # Load shape model
    print("Loading shape model...")
    shape_model = SymmetricSymlogModalOperator(target_modes=3).to(device)
    shape_model.load_state_dict(torch.load(shape_ckpt, map_location=device))
    
    # Data directory
    data_dir = PROJECT_ROOT / "data" / "data_modal_residue_stage1500"
    
    # Precompute for train, val, test
    files = ["train.h5", "val.h5", "test.h5"]
    for f_name in files:
        h5_path = data_dir / f_name
        if h5_path.exists():
            out_pt_path = CURRENT_DIR / f"{f_name.split('.')[0]}_priors.pt"
            precompute_file_priors(h5_path, out_pt_path, shape_model, freq_model, freq_stats, device)
        else:
            print(f"Warning: {f_name} not found at {h5_path}")

if __name__ == "__main__":
    main()
