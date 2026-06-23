import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Union

from training_residue_query.dataset_residue_query import ResidueQueryH5Dataset, COORD_SCALE

class SymmetricSymlogModalDataset(ResidueQueryH5Dataset):
    """
    Dataset for Symmetric Symlog Set Prediction.
    Returns the raw modal_phi_z and modal_omega directly.
    The neural network is supervised entirely on the full-field shapes.
    """
    def __init__(
        self,
        h5_path: Union[str, Path],
        target_modes: int = 3,
        query_per_sample: int = 4096,
        random_query: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(
            h5_path=h5_path,
            target_modes=target_modes,
            query_per_sample=query_per_sample, 
            random_query=random_query,
            seed=seed,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        key = self.sample_keys[int(index)]
        with h5py.File(self.h5_path, "r") as h5:
            g = h5[key]
            points = np.asarray(g["points"], dtype=np.float32)
            n_nodes = int(points.shape[0])
            points = points[:, :3]
            coords = points / COORD_SCALE

            pocket = self._read_pocket_features(g)
            clamp = self._read_clamp_features(g)
            
            # Normalize clamp features
            clamp[:, 5:8] /= 12.0
            clamp[:, 8:11] /= 8.0
            
            e_ratio, rho_ratio = self._read_material_features(g)
            layout_type = self._read_scalar(g, "layout_type", 0.0)
            coverage_code = self._read_scalar(g, "coverage_level_code", 0.0)
            clamp_code = self._read_scalar(g, "clamp_level_code", self._read_scalar(g, "clamp_model_code", 0.0))
            removed_volume_ratio = self._read_scalar(g, "removed_volume_ratio", 0.0)
            grid_jitter = self._read_scalar(g, "grid_jitter", 0.0)
            finished_count = self._read_scalar(g, "finished_count", 0.0)
            current_progress = self._read_scalar(g, "current_progress", 1.0)

            omega = np.asarray(g["modal_omega"], dtype=np.float32)[: self.target_modes]
            residue = np.asarray(g["modal_residue_z"], dtype=np.float32)[:, : self.target_modes]
            
            exc_idx = int(np.asarray(g["excitation_index"]).reshape(-1)[0]) if "excitation_index" in g else 0
            exc_idx = max(0, min(exc_idx, n_nodes - 1))
            node_local = self._build_node_local_features(g, n_nodes)
            
            # Load raw modal phi_z [Nodes, Modes]
            phi_z_full = np.asarray(g["modal_phi_z"][:, : self.target_modes], dtype=np.float32)
            omega_raw = np.asarray(g["modal_omega"][: self.target_modes], dtype=np.float32)
            omega_log = np.log10(np.maximum(omega_raw, 1.0))

        q_idx = self._select_query_indices(n_nodes, index)
        p_coord = coords[exc_idx].astype(np.float32)
        p_node = node_local[exc_idx].astype(np.float32)
        q_coord = coords[q_idx].astype(np.float32)
        q_node = node_local[q_idx].astype(np.float32)
        
        rel_xyz = q_coord - p_coord.reshape(1, 3)
        rel_dist = np.linalg.norm(rel_xyz, axis=1, keepdims=True).astype(np.float32)
        rel = np.concatenate([rel_xyz, rel_dist], axis=1).astype(np.float32)
        target = residue[q_idx].astype(np.float32)
        phi_z_sampled = phi_z_full[q_idx]

        global_features = np.asarray([
            e_ratio, rho_ratio, layout_type / 7.0, coverage_code / 2.0,
            clamp_code / 2.0, removed_volume_ratio, grid_jitter,
            finished_count / 7.0, current_progress
        ], dtype=np.float32)

        return {
            "pocket_features": torch.from_numpy(pocket),
            "clamp_features": torch.from_numpy(clamp),
            "global_features": torch.from_numpy(global_features),
            "omega": torch.from_numpy(omega.astype(np.float32)),
            "p_coord": torch.from_numpy(p_coord),
            "p_node_features": torch.from_numpy(p_node),
            "q_coord": torch.from_numpy(q_coord),
            "q_node_features": torch.from_numpy(q_node),
            "rel_features": torch.from_numpy(rel),
            "target_residue": torch.from_numpy(target),
            "query_index": torch.from_numpy(q_idx),
            "excitation_index": torch.tensor(exc_idx, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "target_phi_z": torch.from_numpy(phi_z_sampled),
            "target_omega": torch.from_numpy(omega_log),
            "target_omega_raw": torch.from_numpy(omega_raw),
        }
