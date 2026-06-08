"""Training package exports for MeshGraphNet/GNN FRF surrogate."""

from .trainer import train, evaluate, save_model
from .losses import modal_loss, frf_loss, mac_loss, sign_aligned_mse, zeta_physics_loss
from .augmentations import GraphBatchAugmenter, GeometryAugmenter

__all__ = [
    "train",
    "evaluate",
    "save_model",
    "modal_loss",
    "frf_loss",
    "mac_loss",
    "sign_aligned_mse",
    "zeta_physics_loss",
    "GraphBatchAugmenter",
    "GeometryAugmenter",
]
