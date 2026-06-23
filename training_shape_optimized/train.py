# -*- coding: utf-8 -*-
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from torch.utils.data import DataLoader
from dataset import SymmetricSymlogModalDataset
from model import SymmetricSymlogModalOperator
from loss import OrderedSymlogLoss
import csv

def train_and_validate(mode="train", lr_override=None, epochs_override=None, patience_override=None):
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Mode: {mode}")
    
    # Paths & Logging
    base_dir = r'f:\毕业论文\stage1-modal-residue-dataset\data\data_modal_residue_stage1500'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    csv_path = os.path.join(log_dir, 'training_log.csv')
    
    k_modes = 3
    # Setup CSV Headers
    headers = ['Epoch', 'TrainLoss', 'ValLoss']
    for i in range(k_modes):
        headers.extend([
            f'Train_phi_MSE_{i+1}', 
            f'Train_phi_MAPE_raw_{i+1}', f'Train_phi_MAPE_strong_{i+1}',
            f'Train_phi_MAC_raw_{i+1}', f'Train_phi_MAC_strong_{i+1}',
            f'Val_phi_MSE_{i+1}', 
            f'Val_phi_MAPE_raw_{i+1}', f'Val_phi_MAPE_strong_{i+1}',
            f'Val_phi_MAC_raw_{i+1}', f'Val_phi_MAC_strong_{i+1}',
            f'Val_A_nMAE_raw_{i+1}', f'Val_A_nMAE_strong_{i+1}',
            f'Val_A_corr_raw_{i+1}', f'Val_A_corr_strong_{i+1}'
        ])
    headers.append('LR')
    
    # Overwrite log only when training from scratch, append for fine-tuning
    write_mode = 'w' if mode == "train" else 'a'
    if mode == "train" or not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    train_path = os.path.join(base_dir, 'train.h5')
    val_path = os.path.join(base_dir, 'val.h5')
    
    print(f"Loading Train Dataset: {train_path}")
    train_dataset = SymmetricSymlogModalDataset(h5_path=train_path, target_modes=k_modes, random_query=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"Loading Val Dataset: {val_path}")
    val_dataset = SymmetricSymlogModalDataset(h5_path=val_path, target_modes=k_modes, random_query=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # Configure hyper-parameters based on mode
    if mode == "ft":
        num_epochs = 60
        initial_lr = 1e-4
        early_stop_patience = 15
    else:
        num_epochs = 180
        initial_lr = 5e-4
        early_stop_patience = 25
        
    # Overrides
    if epochs_override is not None:
        num_epochs = epochs_override
    if lr_override is not None:
        initial_lr = lr_override
    if patience_override is not None:
        early_stop_patience = patience_override
        
    model = SymmetricSymlogModalOperator(target_modes=k_modes).to(device)
    
    # Load pretrained weights for fine-tuning
    best_pth = os.path.join(checkpoint_dir, 'best_model.pth')
    if mode == "ft":
        if os.path.exists(best_pth):
            print(f"[*] Loading pretrained weights for fine-tuning from: {best_pth}")
            model.load_state_dict(torch.load(best_pth, map_location=device))
        else:
            print(f"[*] Warning: Pretrained weights not found at {best_pth}. Starting from scratch.")
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = OrderedSymlogLoss()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    def init_metrics():
        m = {"loss": 0.0, "mse_loss_mean": 0.0, "mac_loss_mean": 0.0}
        for i in range(k_modes):
            m[f"shape_loss_phi{i+1}"] = 0.0
            m[f"shape_mape_raw_phi{i+1}"] = 0.0
            m[f"shape_mape_strong_phi{i+1}"] = 0.0
            m[f"shape_mac_raw_phi{i+1}"] = 0.0
            m[f"shape_mac_strong_phi{i+1}"] = 0.0
            m[f"A_nMAE_raw_{i+1}"] = 0.0
            m[f"A_nMAE_strong_{i+1}"] = 0.0
            m[f"A_corr_raw_{i+1}"] = 0.0
            m[f"A_corr_strong_{i+1}"] = 0.0
        return m

    def accumulate_metrics(accum, batch_metrics):
        for k, v in batch_metrics.items():
            if k in accum:
                accum[k] += v.item() if isinstance(v, torch.Tensor) else v
            else:
                accum[k] = v.item() if isinstance(v, torch.Tensor) else v

    def average_metrics(accum, num_batches):
        return {k: v / num_batches for k, v in accum.items()}
    
    print(f"Starting Training/Fine-Tuning Loop for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        train_metrics = init_metrics()
        
        for i, batch in enumerate(train_loader):
            pocket_features = batch["pocket_features"].to(device)
            clamp_features = batch["clamp_features"].to(device)
            global_features = batch["global_features"].to(device)
            q_coord = batch["q_coord"].to(device)
            q_node_features = batch["q_node_features"].to(device)
            target_phi_z = batch["target_phi_z"].to(device)
            target_omega_raw = batch["target_omega_raw"].to(device)
            
            p_coord = batch["p_coord"].to(device)
            p_node_features = batch["p_node_features"].to(device)
            target_residue = batch["target_residue"].to(device)
            
            optimizer.zero_grad()
            
            # Concatenate q and p coordinates/features along node dimension to run forward pass only once
            qn = q_coord.shape[1]
            p_coord_exp = p_coord.unsqueeze(1) # [B, 1, 3]
            p_node_exp = p_node_features.unsqueeze(1) # [B, 1, C]
            
            all_coords = torch.cat([q_coord, p_coord_exp], dim=1) # [B, N+1, 3]
            all_nodes = torch.cat([q_node_features, p_node_exp], dim=1) # [B, N+1, C]
            
            outputs = model(pocket_features, clamp_features, global_features, all_coords, all_nodes)
            phi_pred_all = outputs["symlog_phi_z"] # [B, N+1, K]
            
            # Split predictions back
            phi_pred_q = phi_pred_all[:, :qn, :]
            phi_pred_p = phi_pred_all[:, qn:, :] # [B, 1, K]
            
            # Calculate predicted Modal Residue A
            pred_A = phi_pred_q * phi_pred_p # [B, N, K]
            
            loss_dict = criterion(outputs["omega"], phi_pred_q, target_omega_raw, target_phi_z, pred_A=pred_A, target_A=target_residue, q_node_features=q_node_features)
            loss = loss_dict["loss"]
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            accumulate_metrics(train_metrics, loss_dict)
            
            if i % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
                
        avg_train = average_metrics(train_metrics, len(train_loader))
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_metrics = init_metrics()
        
        with torch.no_grad():
            for batch in val_loader:
                pocket_features = batch["pocket_features"].to(device)
                clamp_features = batch["clamp_features"].to(device)
                global_features = batch["global_features"].to(device)
                q_coord = batch["q_coord"].to(device)
                q_node_features = batch["q_node_features"].to(device)
                target_phi_z = batch["target_phi_z"].to(device)
                target_omega_raw = batch["target_omega_raw"].to(device)
                
                # New A targets
                p_coord = batch["p_coord"].to(device)
                p_node_features = batch["p_node_features"].to(device)
                target_residue = batch["target_residue"].to(device)
                
                # Concatenate q and p coordinates/features along node dimension to run forward pass only once
                qn = q_coord.shape[1]
                p_coord_exp = p_coord.unsqueeze(1) # [B, 1, 3]
                p_node_exp = p_node_features.unsqueeze(1) # [B, 1, C]
                
                all_coords = torch.cat([q_coord, p_coord_exp], dim=1) # [B, N+1, 3]
                all_nodes = torch.cat([q_node_features, p_node_exp], dim=1) # [B, N+1, C]
                
                outputs = model(pocket_features, clamp_features, global_features, all_coords, all_nodes)
                phi_pred_all = outputs["symlog_phi_z"] # [B, N+1, K]
                
                # Split predictions back
                phi_pred_q = phi_pred_all[:, :qn, :]
                phi_pred_p = phi_pred_all[:, qn:, :] # [B, 1, K]
                
                # Calculate predicted Modal Residue A
                pred_A = phi_pred_q * phi_pred_p # [B, N, K]
                
                loss_dict = criterion(outputs["omega"], phi_pred_q, target_omega_raw, target_phi_z, pred_A=pred_A, target_A=target_residue, q_node_features=q_node_features)
                
                accumulate_metrics(val_metrics, loss_dict)
                
        avg_val = average_metrics(val_metrics, len(val_loader))
        
        # ---------------- METRICS & CHECKPOINTS ----------------
        current_lr = optimizer.param_groups[0]['lr']
        
        # Prepare row for CSV
        row = [epoch, avg_train['loss'], avg_val['loss']]
        for i in range(k_modes):
            phi_mse = f"shape_loss_phi{i+1}"
            phi_mape_raw = f"shape_mape_raw_phi{i+1}"
            phi_mape_strong = f"shape_mape_strong_phi{i+1}"
            phi_mac_raw = f"shape_mac_raw_phi{i+1}"
            phi_mac_strong = f"shape_mac_strong_phi{i+1}"
            a_nmae_raw = f"A_nMAE_raw_{i+1}"
            a_nmae_strong = f"A_nMAE_strong_{i+1}"
            a_corr_raw = f"A_corr_raw_{i+1}"
            a_corr_strong = f"A_corr_strong_{i+1}"
            row.extend([
                avg_train[phi_mse], 
                avg_train[phi_mape_raw], avg_train[phi_mape_strong],
                avg_train[phi_mac_raw], avg_train[phi_mac_strong],
                avg_val[phi_mse], 
                avg_val[phi_mape_raw], avg_val[phi_mape_strong],
                avg_val[phi_mac_raw], avg_val[phi_mac_strong],
                avg_val.get(a_nmae_raw, 0.0), avg_val.get(a_nmae_strong, 0.0),
                avg_val.get(a_corr_raw, 0.0), avg_val.get(a_corr_strong, 0.0)
            ])
        row.append(current_lr)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Scheduler step per epoch for CosineAnnealingLR
        scheduler.step()
        
        # Save Last Model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'latest_model.pth'))
        
        # Save Best Model
        if avg_val['loss'] < best_val_loss:
            best_val_loss = avg_val['loss']
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"[*] Best model saved at epoch {epoch} with Val Loss: {best_val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Calculate loss component proportions
        total_train_loss = avg_train['loss']
        unweighted_total = avg_train['mse_loss_mean'] + avg_train['mac_loss_mean']
        mse_pct = (avg_train['mse_loss_mean'] / unweighted_total) * 100.0 if unweighted_total > 0 else 0
        mac_pct = (avg_train['mac_loss_mean'] / unweighted_total) * 100.0 if unweighted_total > 0 else 0
        
        print(f"==> Epoch [{epoch}/{num_epochs}] Summary | Train Loss: {total_train_loss:.4f} (MSE {mse_pct:.1f}% : MAC {mac_pct:.1f}%) | Val Loss: {avg_val['loss']:.4f} | LR: {current_lr:.2e}")
        print(f"    [Val] 振型MAPE -> phi1: {avg_val['shape_mape_strong_phi1']:.2f}% ({avg_val['shape_mape_raw_phi1']:.2f}%) | phi2: {avg_val['shape_mape_strong_phi2']:.2f}% ({avg_val['shape_mape_raw_phi2']:.2f}%) | phi3: {avg_val['shape_mape_strong_phi3']:.2f}% ({avg_val['shape_mape_raw_phi3']:.2f}%)")
        print(f"    [Val] 振型MAC  -> phi1: {avg_val['shape_mac_strong_phi1']:.4f} ({avg_val['shape_mac_raw_phi1']:.4f}) | phi2: {avg_val['shape_mac_strong_phi2']:.4f} ({avg_val['shape_mac_raw_phi2']:.4f}) | phi3: {avg_val['shape_mac_strong_phi3']:.4f} ({avg_val['shape_mac_raw_phi3']:.4f})")
        print(f"    [Val] A_nMAE   -> A1: {avg_val.get('A_nMAE_strong_1', 0):.2f}% ({avg_val.get('A_nMAE_raw_1', 0):.2f}%) | A2: {avg_val.get('A_nMAE_strong_2', 0):.2f}% ({avg_val.get('A_nMAE_raw_2', 0):.2f}%) | A3: {avg_val.get('A_nMAE_strong_3', 0):.2f}% ({avg_val.get('A_nMAE_raw_3', 0):.2f}%)")
        
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
            break

    print("Training Complete! Best Val Loss:", best_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or fine-tune shape model")
    parser.add_argument("--mode", default="train", choices=["train", "ft"], help="train (scratch) or ft (fine-tune)")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    parser.add_argument("--patience", type=int, default=None, help="override early stop patience")
    args = parser.parse_args()
    train_and_validate(mode=args.mode, lr_override=args.lr, epochs_override=args.epochs, patience_override=args.patience)
