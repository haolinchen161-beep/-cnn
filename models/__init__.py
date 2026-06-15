"""Model package exports for FEM-aware MeshGraphNet modal surrogate."""

from .frf_model import build_geometric_model
from .meshgraphnet_frf_model import MeshGraphFRFModel
from .physics_decoder import PhysicsDecoder

__all__ = [
    "build_geometric_model",
    "MeshGraphFRFModel",
    "PhysicsDecoder",
]
