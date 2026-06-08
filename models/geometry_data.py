"""
geometry_data.py — MeshGraphNet 图数据容器。

主训练流程直接使用 data.dataset.collate_geometry_batch 返回的 dict，
本 dataclass 仅作为可选的类型化容器，方便后续模块化重构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GeometryData:
    """统一图几何数据容器。"""

    points: torch.Tensor                    # [total_N, 3]
    node_features: torch.Tensor             # [total_N, F]
    edge_index: torch.Tensor                # [2, total_E]
    edge_attr: torch.Tensor                 # [total_E, Fe]
    batch: torch.Tensor                     # [total_N]
    point_features: Optional[torch.Tensor] = None
    spring_k_xyz: Optional[torch.Tensor] = None
    spring_c_xyz: Optional[torch.Tensor] = None
    node_type: Optional[torch.Tensor] = None
    pocket_bottom_mask: Optional[torch.Tensor] = None
    cut_region_mask: Optional[torch.Tensor] = None
    excitation_index: Optional[torch.Tensor] = None
    excitation_coord: Optional[torch.Tensor] = None

    def to(self, device):
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    @classmethod
    def from_batch(cls, batch_dict: dict) -> "GeometryData":
        return cls(
            points=batch_dict["points"],
            node_features=batch_dict["node_features"],
            edge_index=batch_dict["edge_index"],
            edge_attr=batch_dict["edge_attr"],
            batch=batch_dict["batch"],
            point_features=batch_dict.get("point_features"),
            spring_k_xyz=batch_dict.get("spring_k_xyz"),
            spring_c_xyz=batch_dict.get("spring_c_xyz"),
            node_type=batch_dict.get("node_type"),
            pocket_bottom_mask=batch_dict.get("pocket_bottom_mask"),
            cut_region_mask=batch_dict.get("cut_region_mask"),
            excitation_index=batch_dict.get("excitation_index"),
            excitation_coord=batch_dict.get("excitation_coord"),
        )
