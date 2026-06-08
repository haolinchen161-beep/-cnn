"""Optional augmentations for Transolver mesh batches.

These augmentations operate directly on concatenated node tensors and keep the
modal/FRF targets aligned. They are intentionally conservative; the default
configuration disables augmentation unless explicitly enabled.
"""
from __future__ import annotations

from typing import Dict

import torch


class GeometryAugmenter:
    def __init__(self,
                 coord_noise: float = 0.0,
                 feature_noise: float = 0.0,
                 enabled: bool = False):
        self.coord_noise = coord_noise
        self.feature_noise = feature_noise
        self.enabled = enabled
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, batch: Dict) -> Dict:
        if not self.enabled or not self.training:
            return batch
        if self.coord_noise > 0 and 'points' in batch:
            noise = torch.randn_like(batch['points']) * self.coord_noise
            if 'fixture_type' in batch:
                noise[batch['fixture_type'] > 0] *= 0.1
            batch['points'] = batch['points'] + noise
        if self.feature_noise > 0 and 'node_features' in batch:
            batch['node_features'] = batch['node_features'] + torch.randn_like(batch['node_features']) * self.feature_noise
        return batch


def create_augmenter(config):
    aug = config.get('augmentation', {})
    return GeometryAugmenter(
        coord_noise=aug.get('coord_noise', 0.0),
        feature_noise=aug.get('feature_noise', 0.0),
        enabled=aug.get('enabled', False),
    )
