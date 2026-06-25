# -*- coding: utf-8 -*-
"""Mean/Max Pooling with Parallel Independent Heads for Frequency Prediction."""
from __future__ import annotations

import torch
import torch.nn as nn

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

class FrequencyTokenMLP(nn.Module):
    """Frequency Predictor with Mean/Max Pooling and Parallel Independent Heads."""

    def __init__(
        self,
        pocket_dim: int = 8,
        clamp_dim: int = 11,
        global_dim: int = 9,
        token_dim: int = 96,
        hidden_dim: int = 192,
        fusion_dim: int = 256,
        out_modes: int = 3,
        token_layers: int = 3,
        fusion_layers: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.out_modes = int(out_modes)
        
        # 1. Feature Encoders
        self.pocket_encoder = MLP(pocket_dim, hidden_dim, token_dim, token_layers, dropout)
        self.clamp_encoder = MLP(clamp_dim, hidden_dim, token_dim, token_layers, dropout)
        self.global_encoder = MLP(global_dim, hidden_dim, token_dim, token_layers, dropout)
        
        # 2. Fusion MLP (pocket mean/max + clamp mean/max + global = 5 * token_dim)
        self.fusion = MLP(5 * token_dim, fusion_dim, fusion_dim, fusion_layers, dropout)
        
        # 3. Independent parallel heads (directly output y1, y2, y3)
        self.head = nn.Linear(fusion_dim, self.out_modes)

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B,T,C] -> [B,2C], mean + max."""
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(
        self,
        pocket_features: torch.Tensor,
        clamp_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        # Encode features
        p_tok = self.pocket_encoder(pocket_features)  # [B, 7, C]
        c_tok = self.clamp_encoder(clamp_features)    # [B, 7, C]
        g_tok = self.global_encoder(global_features)  # [B, C]
        
        # 1. Dual-channel Set pooling (Mean + Max)
        p_pool = self._pool_tokens(p_tok)             # [B, 2*C]
        c_pool = self._pool_tokens(c_tok)             # [B, 2*C]
        
        # 2. Concatenate all pooled tokens (5 * C total dimensions)
        fused = torch.cat([p_pool, c_pool, g_tok], dim=-1)  # [B, 5*C]
        h = self.fusion(fused)  # [B, fusion_dim]
        
        # 3. Predict y1, y2, y3 completely independently
        y_pred = self.head(h)  # [B, 3]
        return y_pred
