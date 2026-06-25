# -*- coding: utf-8 -*-
"""Transformer + Fourier positional encoding frequency predictor.

The public class name remains ``FrequencyTokenMLP`` so existing training and
inference scripts can keep importing the same symbol.  Internally the model no
longer collapses pocket/clamp tokens with mean+max pooling before any
interaction.  It first adds 2D Fourier position embeddings to pocket/clamp
centers, runs a small Transformer encoder with a CLS token, and then regresses
modal frequencies from the global structural context.
"""
from __future__ import annotations

import math
from typing import Optional

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
    """2D Fourier features for normalized x/y token centers."""

    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = int(num_bands)
        freqs = math.pi * torch.pow(2.0, torch.arange(self.num_bands, dtype=torch.float32))
        self.register_buffer("frequencies", freqs)
        self.out_dim = 2 + 2 * 2 * self.num_bands

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        centers = torch.nan_to_num(centers.float(), nan=0.0, posinf=1.0, neginf=0.0)
        features = [centers]
        for freq in self.frequencies:
            x = centers * freq
            features.append(torch.sin(x))
            features.append(torch.cos(x))
        return torch.cat(features, dim=-1)


class FrequencyTokenMLP(nn.Module):
    """Frequency predictor with token Transformer and 2D Fourier PE.

    Inputs:
        pocket_features: [B, 7, 8]
        clamp_features:  [B, 7, 11]
        global_features: [B, 9]
        pocket_centers: optional [B, 7, 2], raw normalized x/y centers
        clamp_centers:  optional [B, 7, 2], raw normalized x/y centers

    Output:
        normalized log-frequency targets [B, out_modes].
    """

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
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        pe_num_bands: int = 4,
    ):
        super().__init__()
        self.out_modes = int(out_modes)
        self.token_dim = int(token_dim)

        self.pocket_encoder = MLP(pocket_dim, hidden_dim, token_dim, token_layers, dropout)
        self.clamp_encoder = MLP(clamp_dim, hidden_dim, token_dim, token_layers, dropout)
        self.global_encoder = MLP(global_dim, hidden_dim, token_dim, token_layers, dropout)

        self.pos_encoder = FourierPositionEncoding2D(num_bands=pe_num_bands)
        self.pos_proj = nn.Linear(self.pos_encoder.out_dim, token_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.type_embedding = nn.Embedding(4, token_dim)  # 0=CLS, 1=pocket, 2=clamp, 3=global

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(transformer_heads),
            dim_feedforward=int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(transformer_layers))
        self.norm = nn.LayerNorm(token_dim)

        # CLS + pocket mean/max + clamp mean/max + global token = 6 * token_dim
        self.fusion = MLP(6 * token_dim, fusion_dim, fusion_dim, fusion_layers, dropout)
        self.head = nn.Linear(fusion_dim, self.out_modes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    @staticmethod
    def _centers_from_bounds(features: torch.Tensor) -> torch.Tensor:
        # Fallback only.  Training code should pass raw centers before standardization.
        x = 0.5 * (features[..., 0] + features[..., 1])
        y = 0.5 * (features[..., 2] + features[..., 3])
        return torch.stack([x, y], dim=-1)

    def _add_position_and_type(self, tokens: torch.Tensor, centers: torch.Tensor, type_id: int) -> torch.Tensor:
        pos = self.pos_proj(self.pos_encoder(centers))
        type_idx = torch.full((tokens.shape[0], tokens.shape[1]), int(type_id), dtype=torch.long, device=tokens.device)
        return tokens + pos + self.type_embedding(type_idx)

    def forward(
        self,
        pocket_features: torch.Tensor,
        clamp_features: torch.Tensor,
        global_features: torch.Tensor,
        pocket_centers: Optional[torch.Tensor] = None,
        clamp_centers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = pocket_features.shape[0]
        device = pocket_features.device

        if pocket_centers is None:
            pocket_centers = self._centers_from_bounds(pocket_features)
        if clamp_centers is None:
            clamp_centers = self._centers_from_bounds(clamp_features)

        p_tok = self.pocket_encoder(pocket_features)
        c_tok = self.clamp_encoder(clamp_features)
        g_tok = self.global_encoder(global_features).unsqueeze(1)

        p_tok = self._add_position_and_type(p_tok, pocket_centers.to(device), type_id=1)
        c_tok = self._add_position_and_type(c_tok, clamp_centers.to(device), type_id=2)

        g_type = torch.full((bsz, 1), 3, dtype=torch.long, device=device)
        g_tok = g_tok + self.type_embedding(g_type)

        cls = self.cls_token.expand(bsz, -1, -1)
        cls_type = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        cls = cls + self.type_embedding(cls_type)

        tokens = torch.cat([cls, p_tok, c_tok, g_tok], dim=1)  # [B, 16, C]
        tokens = self.norm(self.transformer(tokens))

        p_count = p_tok.shape[1]
        c_count = c_tok.shape[1]
        cls_out = tokens[:, 0, :]
        p_out = tokens[:, 1:1 + p_count, :]
        c_out = tokens[:, 1 + p_count:1 + p_count + c_count, :]
        g_out = tokens[:, -1, :]

        fused = torch.cat([
            cls_out,
            self._pool_tokens(p_out),
            self._pool_tokens(c_out),
            g_out,
        ], dim=-1)
        h = self.fusion(fused)
        return self.head(h)
