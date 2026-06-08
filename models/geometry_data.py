"""Geometry containers for Transolver modal-FRF training.

The old CNN branch used this file as a loose geometry holder.  The Transolver
branch uses it to carry a concatenated variable-size mesh batch produced by
``data.dataset.collate_mesh_batch``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TransolverMeshBatch:
    """Batched unstructured mesh data for modal-FRF prediction.

    Attributes:
        points: Node coordinates, concatenated across graphs, shape ``(total_N, 3)``.
        node_features: Per-node Transolver features, shape ``(total_N, C)``.
        batch: Graph id of each node, shape ``(total_N,)``.
        edge_index: Optional mesh edge COO tensor, shape ``(2, E)``.
        boundary_c_xyz: Per-node damping coefficients, shape ``(total_N, 3)``.
        boundary_k_xyz: Per-node stiffness coefficients, shape ``(total_N, 3)``.
        excitation_index: Global node indices of excitation nodes, shape ``(B,)``.
        frequencies: Per-sample frequency grids in Hz, shape ``(B, F)``.
    """

    points: torch.Tensor
    node_features: torch.Tensor
    batch: torch.Tensor
    edge_index: Optional[torch.Tensor] = None
    boundary_c_xyz: Optional[torch.Tensor] = None
    boundary_k_xyz: Optional[torch.Tensor] = None
    excitation_index: Optional[torch.Tensor] = None
    frequencies: Optional[torch.Tensor] = None

    @property
    def num_graphs(self) -> int:
        if self.batch.numel() == 0:
            return 0
        return int(self.batch.max().item()) + 1

    def to(self, device: torch.device | str) -> "TransolverMeshBatch":
        self.points = self.points.to(device)
        self.node_features = self.node_features.to(device)
        self.batch = self.batch.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.boundary_c_xyz is not None:
            self.boundary_c_xyz = self.boundary_c_xyz.to(device)
        if self.boundary_k_xyz is not None:
            self.boundary_k_xyz = self.boundary_k_xyz.to(device)
        if self.excitation_index is not None:
            self.excitation_index = self.excitation_index.to(device)
        if self.frequencies is not None:
            self.frequencies = self.frequencies.to(device)
        return self


# Backward-compatible alias for imports that still reference GeometryData.
GeometryData = TransolverMeshBatch
