import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

# Removed symlog and inv_symlog because phi_z is small ([-5, 5]) and log transformation ruins the amplitude precision.

class OrderedSymlogLoss(torch.nn.Module):
    def __init__(self, symlog_eps: float = 1e-6):
        super().__init__()
        self.eps = float(symlog_eps)

    def forward(
        self,
        pred_omega: torch.Tensor,       # [Batch, K] - Physical frequency from perfect frozen model
        pred_phi: torch.Tensor,         # [Batch, Nodes, K] - Linear prediction (no symlog)
        target_omega: torch.Tensor,     # [Batch, K]
        target_phi: torch.Tensor,       # [Batch, Nodes, K]
        pred_A: torch.Tensor = None,
        target_A: torch.Tensor = None,
        q_node_features: torch.Tensor = None, # [Batch, Nodes, C]
    ) -> dict:
        batch_size, nodes, k = pred_phi.shape
        device = pred_phi.device
        
        # 1. No SymLog Space Transform Needed for Phi
        # 2. Ordered L1 Loss directly on Physical Space with Clamp Weighting
        # Calculate element-wise absolute errors for sign alignment
        diff_pos = torch.abs(pred_phi - target_phi) # [Batch, Nodes, K]
        diff_neg = torch.abs(pred_phi + target_phi) # [Batch, Nodes, K]
        
        mse_pos = torch.mean(diff_pos, dim=1) # [Batch, K]
        mse_neg = torch.mean(diff_neg, dim=1) # [Batch, K]
        
        # Sign mask to align target sign to match pred sign for each mode
        sign_mask = (mse_pos < mse_neg).float().unsqueeze(1) # [Batch, 1, K]
        aligned_diff = diff_pos * sign_mask + diff_neg * (1.0 - sign_mask) # [Batch, Nodes, K]
        
        # Extract clamp node weighting mask if node features are provided
        if q_node_features is not None:
            # Columns 8 to 13 contain spring_k and spring_c features
            clamped_nodes = (q_node_features[:, :, 8:14].max(dim=-1).values > 1e-6).float() # [Batch, Nodes]
            clamp_weight = 1.0 + 2.0 * clamped_nodes # [Batch, Nodes]
            shape_loss_per_mode = torch.mean(aligned_diff * clamp_weight.unsqueeze(-1), dim=1) # [Batch, K]
        else:
            shape_loss_per_mode = torch.minimum(mse_pos, mse_neg) # [Batch, K]
        
        # Calculate L2 norm for each target mode to detect weak/in-plane modes
        target_norms = torch.norm(target_phi, dim=1)
        mac_mask = (target_norms >= 50.0).float()
        
        # Calculate MAC for each mode (MAC is inherently sign-invariant due to squaring)
        mac_loss_per_mode = []
        for i in range(k):
            p = pred_phi[:, :, i]
            t = target_phi[:, :, i]
            num = torch.sum(p * t, dim=1)**2
            den = torch.sum(p * p, dim=1) * torch.sum(t * t, dim=1)
            mac = num / torch.clamp(den, min=1e-12) # [Batch]
            mac_loss_per_mode.append(1.0 - mac)
            
        mac_loss_per_mode = torch.stack(mac_loss_per_mode, dim=1) # [Batch, K]
        mac_loss_per_mode = mac_loss_per_mode * mac_mask
        
        # 3. Calculate Modal Residue A Loss (directly supervises shape products and sign consistency)
        if pred_A is not None and target_A is not None:
            residue_diff = torch.abs(pred_A - target_A) # [Batch, Nodes, K]
            if q_node_features is not None:
                clamped_nodes = (q_node_features[:, :, 8:14].max(dim=-1).values > 1e-6).float()
                clamp_weight = 1.0 + 2.0 * clamped_nodes
                residue_loss_per_mode = torch.mean(residue_diff * clamp_weight.unsqueeze(-1), dim=1)
            else:
                residue_loss_per_mode = torch.mean(residue_diff, dim=1)
        else:
            residue_loss_per_mode = torch.zeros_like(shape_loss_per_mode)
        
        # Combine shape loss, MAC loss, and Residue loss
        total_loss_per_mode = shape_loss_per_mode + 1.0 * mac_loss_per_mode + 1.5 * residue_loss_per_mode
        
        # Apply per-mode weights to focus on the harder higher-order modes (Mode 2 and 3)
        # 独立预测头已隔离了顶层干扰，加大权重可以驱动底层的 Attention 更聚焦二三阶
        mode_weight = torch.tensor([1.0, 1.5, 2.0], device=device).view(1, -1)
        total_loss_per_mode = total_loss_per_mode * mode_weight
        
        # Total scalar loss is the mean over batch and modes
        total_loss = torch.mean(total_loss_per_mode)
        
        # 3. Calculate metrics
        metrics = {
            "loss": total_loss,
            "mse_loss_mean": torch.mean(shape_loss_per_mode).item(),
            "mac_loss_mean": torch.mean(1.0 * mac_loss_per_mode).item()
        }
        
        with torch.no_grad():
            pred_phi_np = pred_phi.cpu().numpy() # [Batch, Nodes, K]
            target_phi_np = target_phi.cpu().numpy() # [Batch, Nodes, K]
            
            # Arrays to accumulate raw (direct, all samples) metrics
            raw_mac_sums = [0.0, 0.0, 0.0]
            raw_mape_sums = [0.0, 0.0, 0.0]
            
            # Arrays to accumulate strong (Hungarian matched, norm >= 50) metrics
            strong_mac_sums = [0.0, 0.0, 0.0]
            strong_mape_sums = [0.0, 0.0, 0.0]
            strong_counts = [0, 0, 0]
            
            # Residue A metrics
            pred_A_np = pred_A.cpu().numpy() if pred_A is not None else None
            target_A_np = target_A.cpu().numpy() if target_A is not None else None
            
            raw_a_nmae_sums = [0.0, 0.0, 0.0]
            raw_a_corr_sums = [0.0, 0.0, 0.0]
            strong_a_nmae_sums = [0.0, 0.0, 0.0]
            strong_a_corr_sums = [0.0, 0.0, 0.0]
            strong_a_counts = [0, 0, 0]
            
            for b in range(batch_size):
                p_sample = pred_phi_np[b] # [Nodes, K]
                t_sample = target_phi_np[b] # [Nodes, K]
                
                # --- 1. Shape Raw (Direct) Metrics ---
                for i in range(k):
                    p_shape_dir = p_sample[:, i]
                    t_shape_dir = t_sample[:, i]
                    
                    if np.dot(p_shape_dir, t_shape_dir) < 0:
                        p_shape_dir = -p_shape_dir
                        
                    num_raw = np.dot(p_shape_dir, t_shape_dir) ** 2
                    den_raw = np.dot(p_shape_dir, p_shape_dir) * np.dot(t_shape_dir, t_shape_dir)
                    mac_raw = num_raw / max(den_raw, 1e-12)
                    raw_mac_sums[i] += mac_raw
                    
                    t_norm_dir = np.linalg.norm(t_shape_dir)
                    l2_err_dir = np.linalg.norm(p_shape_dir - t_shape_dir)
                    mape_raw = (l2_err_dir / max(t_norm_dir, 50.0)) * 100.0
                    raw_mape_sums[i] += mape_raw
                    
                # --- 2. Shape Strong & Hungarian Metrics ---
                cost_matrix = np.zeros((k, k))
                for t_idx in range(k):
                    for p_idx in range(k):
                        t_shape = t_sample[:, t_idx]
                        p_shape = p_sample[:, p_idx]
                        err_pos = np.mean(np.abs(p_shape - t_shape))
                        err_neg = np.mean(np.abs(p_shape + t_shape))
                        cost_matrix[t_idx, p_idx] = min(err_pos, err_neg)
                        
                target_indices, pred_indices = linear_sum_assignment(cost_matrix)
                matching = {t: p for t, p in zip(target_indices, pred_indices)}
                
                for i in range(k):
                    matched_p_idx = matching[i]
                    t_shape = t_sample[:, i]
                    p_shape = p_sample[:, matched_p_idx]
                    
                    if np.dot(p_shape, t_shape) < 0:
                        p_shape = -p_shape
                        
                    t_norm = np.linalg.norm(t_shape)
                    
                    if t_norm >= 50.0:
                        num_str = np.dot(p_shape, t_shape) ** 2
                        den_str = np.dot(p_shape, p_shape) * np.dot(t_shape, t_shape)
                        mac_str = num_str / max(den_str, 1e-12)
                        strong_mac_sums[i] += mac_str
                        
                        l2_err_str = np.linalg.norm(p_shape - t_shape)
                        mape_str = (l2_err_str / t_norm) * 100.0
                        strong_mape_sums[i] += mape_str
                        
                        strong_counts[i] += 1
                        
                # --- 3. Residue A Metrics (Computed inside the sample loop to use matching) ---
                if pred_A_np is not None and target_A_np is not None:
                    for i in range(k):
                        matched_p_idx = matching[i]
                        ap_val = pred_A_np[b, :, matched_p_idx]
                        at_val = target_A_np[b, :, i]
                        
                        # Raw (Direct) A Metrics
                        ap_raw = pred_A_np[b, :, i]
                        at_raw = target_A_np[b, :, i]
                        
                        at_mean_raw = np.mean(np.abs(at_raw))
                        mae_raw = np.mean(np.abs(ap_raw - at_raw))
                        nmae_raw = (mae_raw / max(at_mean_raw, 0.05)) * 100.0
                        raw_a_nmae_sums[i] += nmae_raw
                        
                        ap_cent_raw = ap_raw - np.mean(ap_raw)
                        at_cent_raw = at_raw - np.mean(at_raw)
                        den_raw = np.sqrt(np.sum(ap_cent_raw**2) * np.sum(at_cent_raw**2))
                        corr_raw = np.sum(ap_cent_raw * at_cent_raw) / max(den_raw, 1e-12)
                        raw_a_corr_sums[i] += corr_raw
                        
                        # Strong (Hungarian matched, target_mean >= 0.05) A Metrics
                        at_mean_str = np.mean(np.abs(at_val))
                        if at_mean_str >= 0.05:
                            mae_str = np.mean(np.abs(ap_val - at_val))
                            nmae_str = (mae_str / at_mean_str) * 100.0
                            strong_a_nmae_sums[i] += nmae_str
                            
                            ap_cent_str = ap_val - np.mean(ap_val)
                            at_cent_str = at_val - np.mean(at_val)
                            den_str = np.sqrt(np.sum(ap_cent_str**2) * np.sum(at_cent_str**2))
                            corr_str = np.sum(ap_cent_str * at_cent_str) / max(den_str, 1e-12)
                            strong_a_corr_sums[i] += corr_str
                            
                            strong_a_counts[i] += 1
                            
            # Compute Batch-level Averages
            for i in range(k):
                metrics[f"shape_loss_phi{i+1}"] = torch.mean(shape_loss_per_mode[:, i]).item()
                
                # Shape MAPE & MAC
                metrics[f"shape_mape_raw_phi{i+1}"] = raw_mape_sums[i] / batch_size
                metrics[f"shape_mac_raw_phi{i+1}"] = raw_mac_sums[i] / batch_size
                
                if strong_counts[i] > 0:
                    metrics[f"shape_mape_strong_phi{i+1}"] = strong_mape_sums[i] / strong_counts[i]
                    metrics[f"shape_mac_strong_phi{i+1}"] = strong_mac_sums[i] / strong_counts[i]
                else:
                    metrics[f"shape_mape_strong_phi{i+1}"] = 0.0
                    metrics[f"shape_mac_strong_phi{i+1}"] = 1.0
                    
                # Residue A Metrics
                if pred_A_np is not None and target_A_np is not None:
                    metrics[f"A_nMAE_raw_{i+1}"] = raw_a_nmae_sums[i] / batch_size
                    metrics[f"A_corr_raw_{i+1}"] = raw_a_corr_sums[i] / batch_size
                    
                    if strong_a_counts[i] > 0:
                        metrics[f"A_nMAE_strong_{i+1}"] = strong_a_nmae_sums[i] / strong_a_counts[i]
                        metrics[f"A_corr_strong_{i+1}"] = strong_a_corr_sums[i] / strong_a_counts[i]
                    else:
                        metrics[f"A_nMAE_strong_{i+1}"] = 0.0
                        metrics[f"A_corr_strong_{i+1}"] = 1.0

        return metrics
