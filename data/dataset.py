from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

L_BASE, W_BASE, H_BASE = 0.160, 0.060, 0.010
E_BASE, RHO_BASE, PRXY_BASE = 71.7e9, 2810.0, 0.33

NODE_FEATURE_DIM = 21
DEFAULT_N_MODES = 3


class GraphHDF5Dataset(Dataset):
    """HDF5 graph dataset for z-only modal MeshGraphNet training.

    The first-stage target is only:
        geometry + stiffness boundary -> omega + full-node z mode shapes.

    Full xyz mode shapes are still loaded as modal_phi_xyz because the loss uses
    them to compute the per-mode z-dominance ratio. Damping/FRF fields may exist
    in the HDF5 files, but are ignored by this training dataset.
    """

    def __init__(self,
                 data_paths: Iterable[str],
                 config: Dict | None = None,
                 data_dir: str = ".",
                 normalization: bool = True,
                 test: bool = False):
        self.config = config or {}
        self.data_dir = data_dir
        self.normalization = normalization
        self.test = test
        self.n_modes = int(self.config.get("n_modes", DEFAULT_N_MODES))
        self.knn_k = int(self.config.get("graph", {}).get("knn_k", 12))
        self.omega_scale = float(self.config.get("omega_scale", 2.0 * torch.pi * 5000.0))
        self.samples: List[Tuple[str, str]] = []
        self._load_index([os.path.join(data_dir, p) for p in data_paths])

    def _load_index(self, paths: List[str]) -> None:
        for fp in paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)
            with h5py.File(fp, "r") as f:
                keys = sorted(
                    [k for k in f.keys() if k.startswith("sample_")],
                    key=lambda x: int(x.split("_")[-1]),
                )
                for key in keys:
                    g = f[key]
                    required = ["points", "modal_omega"]
                    if not all(name in g for name in required):
                        continue
                    if "modal_phi_xyz" not in g and "modal_phi" not in g:
                        continue
                    self.samples.append((fp, key))
        if not self.samples:
            raise RuntimeError(f"No valid modal graph samples found in {paths}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp, grp_name = self.samples[idx]
        with h5py.File(fp, "r") as f:
            g = f[grp_name]
            points = torch.from_numpy(g["points"][:]).float()
            n_nodes = points.shape[0]

            point_features = _read_point_features(g, n_nodes)
            spring_k_xyz = _read_optional_node_array(g, "spring_k_xyz", n_nodes, 3)
            node_type = _read_optional_1d(g, "node_type", n_nodes, torch.long)
            pocket_bottom_mask = _read_optional_1d(g, "pocket_bottom_mask", n_nodes, torch.float32)
            cut_region_mask = _read_optional_1d(g, "cut_region_mask", n_nodes, torch.float32)
            local_thickness_ratio = _read_optional_1d(g, "local_thickness_ratio", n_nodes, torch.float32)
            pocket_depth_ratio = _read_optional_1d(g, "pocket_depth_ratio", n_nodes, torch.float32)

            if "edge_index" in g:
                edge_index = torch.from_numpy(g["edge_index"][:]).long()
                if edge_index.ndim != 2:
                    raise ValueError(f"{grp_name}/edge_index must be 2D")
                if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                    edge_index = edge_index.t().contiguous()
            else:
                edge_index = build_knn_edge_index(points, self.knn_k)

            if "edge_attr" in g:
                edge_attr = torch.from_numpy(g["edge_attr"][:]).float()
            else:
                edge_attr = build_edge_attr(points, edge_index)

            if edge_attr.shape[0] != edge_index.shape[1]:
                edge_attr = build_edge_attr(points, edge_index)

            omega = torch.from_numpy(g["modal_omega"][:]).float()[:self.n_modes]
            phi_xyz = _read_phi_xyz(g, n_nodes, self.n_modes)
            phi_z = phi_xyz[..., 2]

            if "excitation_index" in g:
                excitation_index = torch.tensor(int(g["excitation_index"][()]), dtype=torch.long)
            else:
                excitation_index = torch.tensor(0, dtype=torch.long)
            if "excitation_coord" in g:
                excitation_coord = torch.from_numpy(g["excitation_coord"][:]).float()
            else:
                excitation_coord = points[torch.clamp(excitation_index, 0, n_nodes - 1)].clone()

        point_features[:, 6] = local_thickness_ratio

        coords_norm = normalize_points(points)
        node_features = build_node_features(
            points=points,
            coords_norm=coords_norm,
            point_features=point_features,
            spring_k_xyz=spring_k_xyz,
            node_type=node_type,
            pocket_bottom_mask=pocket_bottom_mask,
            cut_region_mask=cut_region_mask,
            pocket_depth_ratio=pocket_depth_ratio,
            excitation_index=excitation_index,
            excitation_coord=excitation_coord,
        )
        node_weight = build_node_weights(
            node_type=node_type,
            pocket_bottom_mask=pocket_bottom_mask,
            cut_region_mask=cut_region_mask,
            excitation_index=excitation_index,
            n_nodes=n_nodes,
        )

        return {
            "points": points,
            "query_coords": coords_norm,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "spring_k_xyz": spring_k_xyz,
            "node_type": node_type,
            "pocket_bottom_mask": pocket_bottom_mask,
            "cut_region_mask": cut_region_mask,
            "local_thickness_ratio": local_thickness_ratio,
            "pocket_depth_ratio": pocket_depth_ratio,
            "node_weight": node_weight,
            "modal_omega_phys": omega,
            "modal_omega_norm": omega / self.omega_scale,
            "modal_freq_hz": omega / (2.0 * torch.pi),
            "modal_phi_z": phi_z,
            # Backward-compatible alias: modal_phi is now z-only [N,K].
            "modal_phi": phi_z,
            # Keep full xyz only for computing z-dominance weights/diagnostics.
            "modal_phi_xyz": phi_xyz,
            "excitation_index": excitation_index,
            "excitation_coord": excitation_coord,
            "sample_name": grp_name,
        }


GeometricHDF5Dataset = GraphHDF5Dataset


def _read_point_features(g, n_nodes: int) -> torch.Tensor:
    if "point_features" in g:
        pf = torch.from_numpy(g["point_features"][:]).float()
        if pf.ndim == 1:
            pf = pf.unsqueeze(0).expand(n_nodes, -1).clone()
    else:
        pf = torch.zeros(n_nodes, 7, dtype=torch.float32)
        pf[:, 0] = 1.0
        pf[:, 1] = PRXY_BASE
        pf[:, 2] = 1.0
        pf[:, 6] = 1.0
    if pf.shape[1] < 7:
        pf = torch.cat([pf, torch.zeros(n_nodes, 7 - pf.shape[1], dtype=pf.dtype)], dim=-1)
    return torch.nan_to_num(pf[:, :7].float(), nan=0.0, posinf=0.0, neginf=0.0)


def _read_phi_xyz(g, n_nodes: int, n_modes: int) -> torch.Tensor:
    if "modal_phi_xyz" in g:
        phi = torch.from_numpy(g["modal_phi_xyz"][:]).float()
    else:
        raw = torch.from_numpy(g["modal_phi"][:]).float()
        if raw.ndim == 3 and raw.shape[-1] == 3:
            phi = raw
        elif raw.ndim == 2:
            phi = torch.zeros(raw.shape[0], raw.shape[1], 3, dtype=torch.float32)
            phi[..., 2] = raw
        else:
            raise ValueError(f"Unsupported modal_phi shape: {tuple(raw.shape)}")
    if phi.shape[0] != n_nodes:
        raise ValueError(f"modal_phi node count mismatch: {phi.shape[0]} vs {n_nodes}")
    if phi.shape[1] < n_modes:
        raise ValueError(f"modal_phi has only {phi.shape[1]} modes, requested {n_modes}")
    return torch.nan_to_num(phi[:, :n_modes, :3].float(), nan=0.0, posinf=0.0, neginf=0.0)


def _read_optional_node_array(g, key: str, n_nodes: int, width: int) -> torch.Tensor:
    if key in g:
        arr = torch.from_numpy(g[key][:]).float()
        if arr.ndim == 1:
            arr = arr.unsqueeze(-1)
        if arr.shape[1] < width:
            arr = torch.cat([arr, torch.zeros(n_nodes, width - arr.shape[1])], dim=-1)
        return torch.nan_to_num(arr[:, :width].float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.zeros(n_nodes, width, dtype=torch.float32)


def _read_optional_1d(g, key: str, n_nodes: int, dtype=torch.float32) -> torch.Tensor:
    if key in g:
        arr = torch.from_numpy(g[key][:]).to(dtype=dtype)
        if arr.numel() != n_nodes:
            arr = arr.reshape(-1)[:n_nodes]
        return arr
    return torch.zeros(n_nodes, dtype=dtype)


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    return points / scale * 2.0 - 1.0


def normalize_log_stiffness(k_xyz: torch.Tensor) -> torch.Tensor:
    k_xyz = torch.clamp(torch.nan_to_num(k_xyz.float(), nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    return torch.log10(1.0 + k_xyz) / 8.0


def build_node_features(points: torch.Tensor,
                        coords_norm: torch.Tensor,
                        point_features: torch.Tensor,
                        spring_k_xyz: torch.Tensor,
                        node_type: torch.Tensor,
                        pocket_bottom_mask: torch.Tensor,
                        cut_region_mask: torch.Tensor,
                        pocket_depth_ratio: torch.Tensor,
                        excitation_index: torch.Tensor,
                        excitation_coord: torch.Tensor) -> torch.Tensor:
    e_ratio = point_features[:, 0:1]
    prxy = point_features[:, 1:2]
    rho_ratio = point_features[:, 2:3]
    local_thickness = point_features[:, 6:7]
    material_geom = torch.cat([e_ratio, rho_ratio, prxy, local_thickness], dim=-1)

    spring_k_norm = normalize_log_stiffness(spring_k_xyz)
    spring_flag = (spring_k_xyz.sum(dim=-1, keepdim=True) > 0).float()

    node_type_oh = F.one_hot(torch.clamp(node_type.long(), min=0, max=4), num_classes=5).float()
    bottom = pocket_bottom_mask.float().unsqueeze(-1)
    cut = cut_region_mask.float().unsqueeze(-1)
    depth = pocket_depth_ratio.float().unsqueeze(-1)

    excitation_flag = torch.zeros(points.shape[0], 1, dtype=points.dtype, device=points.device)
    ei = int(excitation_index.item())
    if 0 <= ei < points.shape[0]:
        excitation_flag[ei, 0] = 1.0

    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    exc = excitation_coord.to(points.device, dtype=points.dtype)
    dist_to_exc = torch.linalg.norm((points - exc.unsqueeze(0)) / scale, dim=-1, keepdim=True)

    node_features = torch.cat([
        coords_norm,
        material_geom,
        spring_k_norm,
        spring_flag,
        node_type_oh,
        bottom,
        cut,
        depth,
        excitation_flag,
        dist_to_exc,
    ], dim=-1).float()

    if node_features.shape[-1] != NODE_FEATURE_DIM:
        raise RuntimeError(f"NODE_FEATURE_DIM mismatch: got {node_features.shape[-1]}, expected {NODE_FEATURE_DIM}")
    return torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)


def build_node_weights(node_type: torch.Tensor,
                       pocket_bottom_mask: torch.Tensor,
                       cut_region_mask: torch.Tensor,
                       excitation_index: torch.Tensor,
                       n_nodes: int) -> torch.Tensor:
    w = torch.ones(n_nodes, dtype=torch.float32)
    w = w + 1.0 * pocket_bottom_mask.float()
    w = w + 1.0 * cut_region_mask.float()
    w = w + 2.0 * ((node_type == 3) | (node_type == 4)).float()
    ei = int(excitation_index.item())
    if 0 <= ei < n_nodes:
        w[ei] += 4.0
    return w


def build_knn_edge_index(points: torch.Tensor, k: int = 12) -> torch.Tensor:
    n = int(points.shape[0])
    if n <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    k = max(1, min(k, n - 1))
    with torch.no_grad():
        scaled = points / torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
        dist = torch.cdist(scaled, scaled)
        nn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]
        src = torch.arange(n, dtype=torch.long, device=points.device).repeat_interleave(k)
        dst = nn_idx.reshape(-1).long()
        edge_index = torch.stack([src, dst], dim=0)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
    return edge_index.cpu().contiguous()


def build_edge_attr(points: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(0, 4, dtype=points.dtype)
    src, dst = edge_index.cpu()
    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    delta = (points[dst] - points[src]) / scale
    length = torch.linalg.norm(delta, dim=-1, keepdim=True)
    return torch.cat([delta, length], dim=-1).float()


def collate_geometry_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    node_features, points, query_coords, edge_indices, edge_attrs, batch_vec = [], [], [], [], [], []
    node_weight = []
    modal_phi_z = []
    modal_phi_xyz = []
    node_offset = 0
    excitation_index_local, excitation_index_global, excitation_coord = [], [], []
    passthrough_cat = {k: [] for k in [
        "spring_k_xyz", "node_type", "pocket_bottom_mask", "cut_region_mask",
        "local_thickness_ratio", "pocket_depth_ratio",
    ]}

    for graph_id, item in enumerate(batch):
        n_i = int(item["points"].shape[0])
        node_features.append(item["node_features"])
        points.append(item["points"])
        query_coords.append(item["query_coords"])
        node_weight.append(item["node_weight"])
        modal_phi_z.append(item["modal_phi_z"])
        modal_phi_xyz.append(item["modal_phi_xyz"])
        edge_indices.append(item["edge_index"].long() + node_offset)
        edge_attrs.append(item.get("edge_attr", build_edge_attr(item["points"], item["edge_index"])))
        batch_vec.append(torch.full((n_i,), graph_id, dtype=torch.long))

        for key in passthrough_cat:
            if key in item:
                passthrough_cat[key].append(item[key])

        local_idx = item.get("excitation_index", torch.tensor(0, dtype=torch.long)).long()
        excitation_index_local.append(local_idx)
        excitation_index_global.append(local_idx + node_offset)
        excitation_coord.append(item.get("excitation_coord", item["points"][local_idx]))

        node_offset += n_i

    phi_z_cat = torch.cat(modal_phi_z, dim=0)
    phi_xyz_cat = torch.cat(modal_phi_xyz, dim=0)

    out = {
        "node_features": torch.cat(node_features, dim=0),
        "points": torch.cat(points, dim=0),
        "query_coords": torch.cat(query_coords, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "node_weight": torch.cat(node_weight, dim=0),
        "modal_phi_z": phi_z_cat,
        # Backward-compatible alias: modal_phi is now z-only [total_N,K].
        "modal_phi": phi_z_cat,
        "modal_phi_xyz": phi_xyz_cat,
        "modal_omega_phys": torch.stack([item["modal_omega_phys"] for item in batch]),
        "modal_omega_norm": torch.stack([item["modal_omega_norm"] for item in batch]),
        "modal_freq_hz": torch.stack([item["modal_freq_hz"] for item in batch]),
        "excitation_index": torch.stack(excitation_index_local),
        "excitation_index_global": torch.stack(excitation_index_global),
        "excitation_coord": torch.stack(excitation_coord),
    }

    for key, values in passthrough_cat.items():
        if len(values) == len(batch):
            out[key] = torch.cat(values, dim=0)

    return out
