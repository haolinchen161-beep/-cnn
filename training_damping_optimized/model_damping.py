# -*- coding: utf-8 -*-
"""Optimized modal damping ratio prediction model architecture."""
from __future__ import annotations

import math
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

class FourierPositionEncoding2D(nn.Module):
    """Fourier positional encoding for 2D coordinates."""
    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands
        frequencies = math.pi * torch.pow(2.0, torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 2]
        features = [coords]
        for freq in self.frequencies:
            features.append(torch.sin(coords * freq))
            features.append(torch.cos(coords * freq))
        return torch.cat(features, dim=-1) # [B, N, 18]

class DampingTokenMLP(nn.Module):
    """Optimized Modal Damping Predictor with Self-Attention."""

    def __init__(
        self,
        pocket_dim: int = 8,
        clamp_dim: int = 11,
        global_dim: int = 13,  # Includes base layout features (7) + predicted physical priors (6)
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
        
        # 2. Fourier Positional Encoding for 2D coordinates
        self.pe = FourierPositionEncoding2D(num_bands=4)
        self.pe_projection = nn.Linear(18, token_dim)
        
        # 3. Learnable global CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        
        # 4. Transformer Encoder for Slot-Clamp interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 5. Fusion MLP
        self.fusion = MLP(2 * token_dim, fusion_dim, fusion_dim, fusion_layers, dropout)
        self.head = nn.Linear(fusion_dim, self.out_modes)

    def forward(
        self,
        pocket_features: torch.Tensor,
        pocket_centers: torch.Tensor,
        clamp_features: torch.Tensor,
        clamp_centers: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        bsz = pocket_features.shape[0]
        
        # Encode features
        p_tok = self.pocket_encoder(pocket_features) # [B, 7, C]
        c_tok = self.clamp_encoder(clamp_features)   # [B, 7, C]
        g_tok = self.global_encoder(global_features)   # [B, C]
        
        # Apply Positional Encoding to Centers
        p_pos = self.pe_projection(self.pe(pocket_centers)) # [B, 7, C]
        c_pos = self.pe_projection(self.pe(clamp_centers))   # [B, 7, C]
        
        p_tok = p_tok + p_pos
        c_tok = c_tok + c_pos
        
        # Prepend learnable CLS token
        cls_tokens = self.cls_token.expand(bsz, -1, -1) # [B, 1, C]
        tokens = torch.cat([cls_tokens, p_tok, c_tok], dim=1) # [B, 15, C]
        
        # Self-Attention Interaction
        tokens_out = self.transformer(tokens) # [B, 15, C]
        cls_out = tokens_out[:, 0] # [B, C]
        
        # Concatenate CLS token with Global features
        fused = torch.cat([cls_out, g_tok], dim=-1) # [B, 2*C]
        h = self.fusion(fused) # [B, fusion_dim]
        
        # Predict normalized outputs directly
        y_pred = self.head(h) # [B, 3]
        return y_pred
