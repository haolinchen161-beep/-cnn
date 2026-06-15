"""
dataset.py — FEM-aware MeshGraphNet graph dataset.

This branch expects per-sample HDF5 groups generated from ANSYS/FEM data:

    /sample_i/points              [N,3]
    /sample_i/edge_index          [2,E] or [E,2]        optional but recommended
    /sample_i/edge_attr           [E,4]                 optional
    /sample_i/point_features      [N,7] or [7]
        [E_ratio, PRXY, rho_ratio, is_fixed, logK, logC, Z/H]
    /sample_i/spring_k_xyz        [N,3]                 optional
    /sample_i/spring_c_xyz        [N,3]                 optional
    /sample_i/node_type           [N]                   optional
    /sample_i/pocket_bottom_mask  [N]                   optional
    /sample_i/cut_region_mask     [N]                   optional
    /sample_i/point_frf           [N,F,2]
    /sample_i/frequencies         [F]
    /sample_i/modal_omega         [K]       rad/s
    /sample_i/modal_zeta          [K]
    /sample_i/modal_phi_xyz       [N,K,3]   preferred
    /sample_i/modal_phi           [N,K]     legacy Z-only fallback

Returned batches are disjoint graphs with concatenated node tensors and
edge indices shifted by node offsets.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


L_BASE = 0.160
W_BASE = 0.060
H_BASE = 0.010
OMEGA_MAX_DEFAULT = 32000.0
NODE_FEATURE_DIM = 25
DEFAULT_FORCE_VECTOR = [0.0, 0.0, 1.0]


class GraphHDF5Dataset(Dataset):
    """per-sample-group HDF5 -> variable-size FEM graph batch."""

    def __init__(self, data_paths: Iterable[str], config: Dict, data_dir: str = ".",
                 test: bool = False, normalization: bool = True,
                 force_vector: List[float] | None = None):
        self.config = config
        self.normalization = normalization
        self.test = test
        self.freq_min = float(config.get("freq_min", 1.0))
        self.freq_max = float(config.get("freq_max", 5000.0))
        self.omega_max = float(config.get("omega_max", OMEGA_MAX_DEFAULT))
        self.knn_k = int(config.get("graph", {}).get("knn_k", 12))
        self.filter_g32 = bool(config.get("filter_g32", False))
        self.g32_min = float(config.get("g32_min", 200.0))
        self.g32_max = float(config.get("g32_max", 900.0))
        self.force_vector = torch.tensor(force_vector or DEFAULT_FORCE_VECTOR, dtype=torch.float32)
        self._samples: List[tuple[str, str]] = []

        full_paths = [os.path.join(data_dir, p) for p in data_paths]
        self._load_index(full_paths)

    def _load_index(self, full_paths: List[str]) -> None:
        for fp in full_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)
            with h5py.File(fp, "r") as f:
                keys = [k for k in f.keys() if k.startswith("sample_")]
                keys = sorted(keys, key=lambda k: int(k.split("_")[-1]))
                for key in keys:
                    if self.filter_g32 and "modal_omega" in f[key]:
                        omega = f[key]["modal_omega"][:]
                        fhz = omega / (2.0 * 3.141592653589793)
                        if len(fhz) >= 3:
                            g32 = float(fhz[2] - fhz[1])
                            if g32 < self.g32_min or g32 > self.g32_max:
                                continue
                    self._samples.append((fp, key))
        if not self._samples:
            raise RuntimeError(f"No per-sample-group data found in: {full_paths}")

    def __len__(self) -> int:
        return len(self._samples)

    def undo_normalize(self, frf: torch.Tensor) -> torch.Tensor:
        return frf

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp, grp_name = self._samples[idx]
        with h5py.File(fp, "r") as f:
            grp = f[grp_name]
            points = torch.from_numpy(grp["points"][:]).float()
            point_features = _read_point_features(grp, points.shape[0])
            point_frf = torch.from_numpy(grp["point_frf"][:]).float()
            frequencies = torch.from_numpy(grp["frequencies"][:]).float()

            edge_index = None
            if "edge_index" in grp:
                edge_index = torch.from_numpy(grp["edge_index"][:]).long()
                if edge_index.ndim != 2:
                    raise ValueError(f"{grp_name}/edge_index must be 2D, got {tuple(edge_index.shape)}")
                if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                    edge_index = edge_index.t().contiguous()

            edge_attr = None
            if "edge_attr" in grp:
                edge_attr = torch.from_numpy(grp["edge_attr"][:]).float()

            n_nodes = points.shape[0]
            spring_k_xyz = _read_optional_node_array(grp, "spring_k_xyz", n_nodes, 3)
            spring_c_xyz = _read_optional_node_array(grp, "spring_c_xyz", n_nodes, 3)
            node_type = _read_optional_1d(grp, "node_type", n_nodes, dtype=torch.long)
            pocket_bottom_mask = _read_optional_1d(grp, "pocket_bottom_mask", n_nodes, dtype=torch.float32)
            cut_region_mask = _read_optional_1d(grp, "cut_region_mask", n_nodes, dtype=torch.float32)

            out: Dict[str, torch.Tensor] = {}
            if "modal_omega" in grp:
                modal_omega = torch.from_numpy(grp["modal_omega"][:]).float()
                out["modal_omega_phys"] = modal_omega
                out["modal_omega_norm"] = modal_omega / self.omega_max
                out["modal_freq_hz"] = modal_omega / (2.0 * torch.pi)
            if "modal_zeta" in grp:
                out["modal_zeta"] = torch.from_numpy(grp["modal_zeta"][:]).float()
                out["modal_log_zeta"] = torch.log(torch.clamp(out["modal_zeta"], min=1e-8))

            phi_xyz = None
            if "modal_phi_xyz" in grp:
                phi_xyz = torch.from_numpy(grp["modal_phi_xyz"][:]).float()
            elif "modal_phi" in grp:
                phi_legacy = torch.from_numpy(grp["modal_phi"][:]).float()
                if phi_legacy.ndim == 3 and phi_legacy.shape[-1] == 3:
                    phi_xyz = phi_legacy
                else:
                    phi_xyz = torch.zeros(phi_legacy.shape[0], phi_legacy.shape[1], 3, dtype=torch.float32)
                    phi_xyz[..., 2] = phi_legacy
            if phi_xyz is not None:
                out["modal_phi"] = phi_xyz
                out["modal_phi_xyz"] = phi_xyz

            if "modal_phi_exc" in grp:
                phi_exc = torch.from_numpy(grp["modal_phi_exc"][:]).float()
                if phi_exc.ndim == 2 and phi_exc.shape[-1] == 3:
                    phi_exc = phi_exc[:, 2]
                out["modal_phi_exc"] = phi_exc

            if "excitation_index" in grp:
                excitation_index = torch.tensor(int(grp["excitation_index"][()]), dtype=torch.long)
            else:
                excitation_index = torch.tensor(0, dtype=torch.long)
            out["excitation_index"] = excitation_index

            if "excitation_coord" in grp:
                excitation_coord = torch.from_numpy(grp["excitation_coord"][:]).float()
            else:
                excitation_coord = points[excitation_index].clone()
            out["excitation_coord"] = excitation_coord

        if "modal_phi_exc" not in out and "modal_phi" in out:
            ei = int(out["excitation_index"].item())
            if 0 <= ei < out["modal_phi"].shape[0]:
                out["modal_phi_exc"] = out["modal_phi"][ei, :, 2].clone()

        if edge_index is None or edge_index.numel() == 0:
            edge_index = build_knn_edge_index(points, k=self.knn_k)
        if edge_attr is None or edge_attr.shape[0] != edge_index.shape[1]:
            edge_attr = build_edge_attr(points, edge_index)

        if self.normalization:
            frequencies = (frequencies - self.freq_min) / (self.freq_max - self.freq_min) * 2.0 - 1.0

        coords_norm = normalize_points(points)
        node_features = build_node_features(
            points=points,
            coords_norm=coords_norm,
            point_features=point_features,
            spring_k_xyz=spring_k_xyz,
            spring_c_xyz=spring_c_xyz,
            node_type=node_type,
            pocket_bottom_mask=pocket_bottom_mask,
            cut_region_mask=cut_region_mask,
            excitation_index=out["excitation_index"],
            excitation_coord=out["excitation_coord"],
        )

        result: Dict[str, torch.Tensor] = {
            "points": points,
            "query_coords": coords_norm,
            "point_features": point_features,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "spring_k_xyz": spring_k_xyz,
            "spring_c_xyz": spring_c_xyz,
            "node_type": node_type,
            "pocket_bottom_mask": pocket_bottom_mask,
            "cut_region_mask": cut_region_mask,
            "point_frf": point_frf,
            "frequencies": frequencies,
            "force_vector": self.force_vector,
        }
        result.update(out)
        return result


GeometricHDF5Dataset = GraphHDF5Dataset


def _read_point_features(grp, n_nodes: int) -> torch.Tensor:
    if "point_features" in grp:
        pf = torch.from_numpy(grp["point_features"][:]).float()
        if pf.ndim == 1:
            pf = pf.unsqueeze(0).expand(n_nodes, -1).clone()
    else:
        pf = torch.zeros(n_nodes, 7, dtype=torch.float32)
        pf[:, 0] = 1.0
        pf[:, 1] = 0.33
        pf[:, 2] = 1.0
        pf[:, 6] = torch.clamp(torch.zeros(n_nodes), min=0.0)

    if pf.shape[1] < 7:
        pad = torch.zeros(n_nodes, 7 - pf.shape[1], dtype=pf.dtype)
        pf = torch.cat([pf, pad], dim=-1)
    return pf[:, :7].float()


def _read_optional_node_array(grp, key: str, n_nodes: int, width: int) -> torch.Tensor:
    if key in grp:
        arr = torch.from_numpy(grp[key][:]).float()
        if arr.ndim == 1:
            arr = arr.unsqueeze(-1)
        if arr.shape[1] < width:
            arr = torch.cat([arr, torch.zeros(n_nodes, width - arr.shape[1])], dim=-1)
        return arr[:, :width]
    return torch.zeros(n_nodes, width, dtype=torch.float32)


def _read_optional_1d(grp, key: str, n_nodes: int, dtype=torch.float32) -> torch.Tensor:
    if key in grp:
        arr = torch.from_numpy(grp[key][:])
        return arr.to(dtype=dtype)
    return torch.zeros(n_nodes, dtype=dtype)


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    return points / scale * 2.0 - 1.0


def sanitize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_log_positive(x: torch.Tensor, max_log: float = 8.0) -> torch.Tensor:
    x = torch.clamp(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    return torch.log10(1.0 + x) / max_log


def build_node_features(points: torch.Tensor,
                        coords_norm: torch.Tensor,
                        point_features: torch.Tensor,
                        spring_k_xyz: torch.Tensor,
                        spring_c_xyz: torch.Tensor,
                        node_type: torch.Tensor,
                        pocket_bottom_mask: torch.Tensor,
                        cut_region_mask: torch.Tensor,
                        excitation_index: torch.Tensor,
                        excitation_coord: torch.Tensor) -> torch.Tensor:
    point_features = sanitize_features(point_features)
    spring_k_norm = normalize_log_positive(spring_k_xyz, max_log=8.0)
    spring_c_norm = normalize_log_positive(spring_c_xyz, max_log=4.0)

    node_type = torch.clamp(node_type.long(), min=0, max=4)
    node_type_onehot = F.one_hot(node_type, num_classes=5).float()

    bottom = pocket_bottom_mask.float().unsqueeze(-1)
    cut = cut_region_mask.float().unsqueeze(-1)

    excitation_flag = torch.zeros(points.shape[0], 1, dtype=points.dtype, device=points.device)
    ei = int(excitation_index.item())
    if 0 <= ei < points.shape[0]:
        excitation_flag[ei, 0] = 1.0

    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    exc = excitation_coord.to(points.device, dtype=points.dtype)
    dist_to_exc = torch.linalg.norm((points - exc.unsqueeze(0)) / scale, dim=-1, keepdim=True)

    node_features = torch.cat([
        coords_norm,
        point_features,
        spring_k_norm,
        spring_c_norm,
        node_type_onehot,
        bottom,
        cut,
        excitation_flag,
        dist_to_exc,
    ], dim=-1).float()

    if node_features.shape[-1] != NODE_FEATURE_DIM:
        raise RuntimeError(f"NODE_FEATURE_DIM mismatch: got {node_features.shape[-1]}, expected {NODE_FEATURE_DIM}")
    return node_features


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
    edge_index_cpu = edge_index.cpu()
    src, dst = edge_index_cpu
    delta = points[dst] - points[src]
    scale = torch.tensor([L_BASE, W_BASE, H_BASE], dtype=points.dtype, device=points.device)
    delta_norm = delta / scale
    dist = torch.linalg.norm(delta_norm, dim=-1, keepdim=True)
    return torch.cat([delta_norm, dist], dim=-1).float()


def collate_geometry_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    node_features, points, query_coords, point_features = [], [], [], []
    edge_indices, edge_attrs = [], []
    point_frf = []
    batch_vec = []
    node_offset = 0

    passthrough_cat = {
        "spring_k_xyz": [],
        "spring_c_xyz": [],
        "node_type": [],
        "pocket_bottom_mask": [],
        "cut_region_mask": [],
    }
    excitation_index_local, excitation_index_global, excitation_coord = [], [], []
    force_vectors = []

    for i, item in enumerate(batch):
        n_i = int(item["points"].shape[0])
        node_features.append(item["node_features"])
        points.append(item["points"])
        query_coords.append(item["query_coords"])
        point_features.append(item["point_features"])
        point_frf.append(item["point_frf"])
        batch_vec.append(torch.full((n_i,), i, dtype=torch.long))

        edge_indices.append(item["edge_index"].long() + node_offset)
        edge_attrs.append(item.get("edge_attr", build_edge_attr(item["points"], item["edge_index"])))

        for key in passthrough_cat:
            if key in item:
                passthrough_cat[key].append(item[key])
        if "excitation_index" in item:
            local_idx = item["excitation_index"].long()
            excitation_index_local.append(local_idx)
            excitation_index_global.append(local_idx + node_offset)
        if "excitation_coord" in item:
            excitation_coord.append(item["excitation_coord"])
        if "force_vector" in item:
            force_vectors.append(item["force_vector"])

        node_offset += n_i

    f_lens = [item["frequencies"].shape[0] for item in batch]
    if all(f == f_lens[0] for f in f_lens):
        frequencies = torch.stack([item["frequencies"] for item in batch])
        frf_out = torch.cat(point_frf, dim=0)
    else:
        frequencies = [item["frequencies"] for item in batch]
        frf_out = point_frf

    out: Dict[str, torch.Tensor] = {
        "node_features": torch.cat(node_features, dim=0),
        "points": torch.cat(points, dim=0),
        "query_coords": torch.cat(query_coords, dim=0),
        "point_features": torch.cat(point_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "frequencies": frequencies,
        "point_frf": frf_out,
    }

    for key, values in passthrough_cat.items():
        if len(values) == len(batch):
            out[key] = torch.cat(values, dim=0)
    if len(excitation_index_local) == len(batch):
        out["excitation_index"] = torch.stack(excitation_index_local)
        out["excitation_index_global"] = torch.stack(excitation_index_global)
    if len(excitation_coord) == len(batch):
        out["excitation_coord"] = torch.stack(excitation_coord)
    if len(force_vectors) == len(batch):
        out["force_vector"] = torch.stack(force_vectors)

    modal = _stack_modal(batch)
    if modal:
        out.update(modal)
    return out


def _stack_modal(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    required = ["modal_omega_phys", "modal_zeta", "modal_phi"]
    if any(key not in batch[0] for key in required):
        return {}

    result: Dict[str, torch.Tensor] = {
        "modal_omega_phys": torch.stack([item["modal_omega_phys"] for item in batch]),
        "modal_omega_norm": torch.stack([item["modal_omega_norm"] for item in batch]),
        "modal_zeta": torch.stack([item["modal_zeta"] for item in batch]),
        "modal_phi": torch.cat([item["modal_phi"] for item in batch], dim=0),
    }
    for key in ["modal_phi_exc", "modal_freq_hz", "modal_log_zeta"]:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    if "modal_phi_xyz" in batch[0]:
        result["modal_phi_xyz"] = torch.cat([item["modal_phi_xyz"] for item in batch], dim=0)
    return result
