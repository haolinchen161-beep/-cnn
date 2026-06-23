"""Set Prediction Model for Symmetric SymLog Modal Shapes."""
import math
import sys
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from training_frequency.model_frequency import FrequencyTokenMLP
except ImportError:
    pass


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        last = int(in_dim)
        for _ in range(max(0, int(num_layers) - 1)):
            layers.append(nn.Linear(last, int(hidden_dim)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            last = int(hidden_dim)
        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierPositionEncoding(nn.Module):
    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands
        frequencies = math.pi * torch.pow(2.0, torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = [coords]
        for freq in self.frequencies:
            features.append(torch.sin(coords * freq))
            features.append(torch.cos(coords * freq))
        return torch.cat(features, dim=-1)


class SymmetricSymlogModalOperator(nn.Module):
    """
    Architecture strictly following ResidueQueryMLP.
    Simultaneously predicts all target_modes by concatenating all context and outputting K values at once.
    """

    def __init__(
        self,
        pocket_dim: int = 8,
        clamp_dim: int = 11,
        global_dim: int = 9,
        node_local_dim: int = 18,
        target_modes: int = 3,
        token_dim: int = 96,
        node_dim: int = 96,
        context_dim: int = 192,
        hidden_dim: int = 256,
        token_layers: int = 3,
        query_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.target_modes = int(target_modes)
        
        # ------------------------------------------
        # 1. PERFECT FREQUENCY PREDICTION MODULE
        # ------------------------------------------
        self.freq_model = FrequencyTokenMLP(
            pocket_dim=pocket_dim, clamp_dim=clamp_dim, global_dim=global_dim,
            token_dim=96, hidden_dim=192, fusion_dim=256,
            out_modes=target_modes, token_layers=3, fusion_layers=4, dropout=dropout
        )
        freq_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training_frequency_optimized", "runs", "best_frequency_model.pt"))
        if not os.path.exists(freq_ckpt):
            # Fallback to workspace root frequency_full
            freq_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs", "frequency_full", "best_frequency_model.pt"))
        if os.path.exists(freq_ckpt):
            ckpt = torch.load(freq_ckpt, map_location="cpu")
            if "model_state" in ckpt:
                self.freq_model.load_state_dict(ckpt["model_state"], strict=False)
            elif "model" in ckpt:
                self.freq_model.load_state_dict(ckpt["model"], strict=False)
            
            if "stats" in ckpt:
                for k, v in ckpt["stats"].items():
                    if k.endswith("_mean") or k.endswith("_std"):
                        if "pocket" in k or "clamp" in k:
                            self.register_buffer(k, v.view(1, 1, -1))
                        else:
                            self.register_buffer(k, v.view(1, -1))
            else:
                raise ValueError("Normalization stats missing in frequency model checkpoint!")
                
        # Freeze frequency model
        for param in self.freq_model.parameters():
            param.requires_grad = False
            
        # ------------------------------------------
        # 2. JOINT MODE SHAPE PREDICTOR
        # ------------------------------------------
        self.pocket_encoder = MLP(pocket_dim, hidden_dim, token_dim, token_layers, dropout)
        self.clamp_encoder = MLP(clamp_dim, hidden_dim, token_dim, token_layers, dropout)
        self.global_encoder = MLP(global_dim, hidden_dim, token_dim, token_layers, dropout)
        self.omega_encoder = MLP(self.target_modes, hidden_dim, token_dim, token_layers, dropout)
        
        # pocket mean/max + clamp mean/max + global + omega = 6*token_dim
        self.context_encoder = MLP(6 * token_dim, hidden_dim, context_dim, query_layers, dropout)
        
        self.pe = FourierPositionEncoding(num_bands=4)
        self.node_input_dim = (3 + 2 * self.pe.num_bands * 3) + int(node_local_dim)
        self.node_encoder = MLP(self.node_input_dim, hidden_dim, node_dim, token_layers, dropout)
        
        # Query Head takes Context + Node + Attention
        query_in_dim = context_dim + node_dim + node_dim
        # Decoupled heads for each mode to prevent gradient interference
        self.query_heads = nn.ModuleList([
            MLP(query_in_dim, hidden_dim, 1, query_layers, dropout)
            for _ in range(self.target_modes)
        ])
        
        self.phi_scale_head = MLP(context_dim, hidden_dim, self.target_modes, num_layers=2, dropout=dropout)
        
        # Cross Attention: Node (Query) attending to Pockets and Clamps (Key/Value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, 
            kdim=token_dim, 
            vdim=token_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=dropout
        )

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def encode_context(
        self,
        pocket_features: torch.Tensor,
        clamp_features: torch.Tensor,
        global_features: torch.Tensor,
        omega_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_tok = self.pocket_encoder(pocket_features)       # [B,7,C]
        c_tok = self.clamp_encoder(clamp_features)         # [B,7,C]
        g_tok = self.global_encoder(global_features)       # [B,C]
        w_tok = self.omega_encoder(omega_norm)             # [B,C]
        fused = torch.cat([
            self._pool_tokens(p_tok),
            self._pool_tokens(c_tok),
            g_tok,
            w_tok,
        ], dim=-1)
        ctx = self.context_encoder(fused)                 # [B,context_dim]
        return ctx, p_tok, c_tok

    def forward(
        self,
        pocket_features: torch.Tensor,
        clamp_features: torch.Tensor,
        global_features: torch.Tensor,
        q_coord: torch.Tensor,
        q_node_features: torch.Tensor,
        shape_std: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, qn, _ = q_coord.shape
        
        # 1. Extract Frequencies for Shape Branch (keep legacy bug for 100% weight consistency)
        eps = 1e-6
        p_norm = (pocket_features - self.pocket_features_mean) / self.pocket_features_std.clamp_min(eps)
        c_norm_bug = (clamp_features - self.clamp_features_mean) / self.clamp_features_std.clamp_min(eps)
        g_norm = (global_features - self.global_features_mean) / self.global_features_std.clamp_min(eps)
        
        self.freq_model.eval()
        with torch.no_grad():
            normalized_omega_bug = self.freq_model(p_norm, c_norm_bug, g_norm) # [B, target_modes]
            
        # 2. Extract Correct Frequencies for physical output (omega)
        clamp_features_restored = clamp_features.clone()
        clamp_features_restored[:, :, 5:8] *= 12.0
        clamp_features_restored[:, :, 8:11] *= 8.0
        c_norm_correct = (clamp_features_restored - self.clamp_features_mean) / self.clamp_features_std.clamp_min(eps)
        
        with torch.no_grad():
            normalized_omega_correct = self.freq_model(p_norm, c_norm_correct, g_norm) # [B, target_modes]
            
        logw = normalized_omega_correct * self.omega_log_std + self.omega_log_mean
        physical_omega = torch.exp(logw).clamp_min(1e-6) # [B, target_modes]
            
        # 3. Encode Context using the legacy normalized omega to match trained weights
        ctx, p_tok, c_tok = self.encode_context(pocket_features, clamp_features, global_features, normalized_omega_bug)
        ctx_q = ctx.unsqueeze(1).expand(-1, qn, -1) # [B, N, context_dim]
        
        # 3. Encode Nodes
        q_coord_pe = self.pe(q_coord)
        q_in = torch.cat([q_coord_pe, q_node_features], dim=-1)
        q_emb = self.node_encoder(q_in.reshape(bsz * qn, -1)).reshape(bsz, qn, -1) # [B, N, node_dim]
        
        # 4. Cross Attention (Nodes looking at Pockets and Clamps)
        local_tokens = torch.cat([p_tok, c_tok], dim=1) # [B, 14, token_dim]
        attn_out, _ = self.cross_attn(query=q_emb, key=local_tokens, value=local_tokens) # [B, N, node_dim]
        
        # 5. Joint Prediction (Decoupled)
        query = torch.cat([ctx_q, q_emb, attn_out], dim=-1) # [B, N, context_dim + node_dim + node_dim]
        q_flat = query.reshape(bsz * qn, -1)
        phis = [head(q_flat) for head in self.query_heads] # list of [B*N, 1]
        raw_shape = torch.cat(phis, dim=-1).reshape(bsz, qn, self.target_modes) # [B, N, target_modes]
        
        if shape_std is None:
            if qn > 1:
                shape_std = torch.std(raw_shape, dim=1, keepdim=True) + 1e-8 # [B, 1, target_modes]
            else:
                shape_std = torch.ones(bsz, 1, self.target_modes, device=raw_shape.device)
                
        normalized_shape = raw_shape / shape_std
        
        scale_multiplier = torch.nn.functional.softplus(self.phi_scale_head(ctx)) # [B, target_modes]
        scale_multiplier = scale_multiplier.unsqueeze(1) # [B, 1, target_modes]
        
        symlog_phi_z = normalized_shape * scale_multiplier # [B, N, target_modes]

        return {
            "omega": physical_omega,
            "symlog_phi_z": symlog_phi_z,
            "shape_std": shape_std,
        }
