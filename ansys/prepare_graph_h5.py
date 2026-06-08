"""
prepare_graph_h5.py — 把旧 ANSYS HDF5 转换为 GNN/MeshGraphNet 友好的图数据。

用途：
    旧 generate_3d_test.py 已经能生成 points / point_features / modal_phi / FRF，
    但没有保存 edge_index / edge_attr / spring_k_xyz / spring_c_xyz 等 GNN 需要的字段。
    本脚本在不删除旧数据的前提下，生成 *_graph.h5。

注意：
    这里的 edge_index 默认由 kNN 构建，是兼容旧数据的过渡方案；
    最佳方案仍是在 generate_3d_test.py 里直接从 MAPDL 单元连接关系保存 FE 拓扑。
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import h5py
import numpy as np


L_BASE = 0.160
W_BASE = 0.060
H_BASE = 0.010


def build_knn_edge_index(points: np.ndarray, k: int = 12) -> np.ndarray:
    """用节点坐标构建无向 kNN 图。返回 shape=(2, E)。"""
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    k = max(1, min(k, n - 1))

    # 坐标归一化后再算距离，避免 X/Y/Z 尺度不一致。
    scale = np.array([L_BASE, W_BASE, H_BASE], dtype=np.float32)
    p = points / scale
    # N≈5000，完整距离矩阵约 100MB float32，可接受；如未来 N 很大再改分块/FAISS。
    diff = p[:, None, :] - p[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    nn_idx = np.argpartition(dist2, kth=k + 1, axis=1)[:, 1:k + 1]

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = nn_idx.reshape(-1).astype(np.int64)
    edges = np.stack([src, dst], axis=0)
    rev = edges[::-1]
    edges = np.concatenate([edges, rev], axis=1)
    # 去重
    edges = np.unique(edges.T, axis=0).T.astype(np.int64)
    return edges


def build_edge_attr(points: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """edge_attr=[dx/L, dy/W, dz/H, normalized_length]。"""
    if edge_index.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    src, dst = edge_index
    scale = np.array([L_BASE, W_BASE, H_BASE], dtype=np.float32)
    delta = (points[dst] - points[src]) / scale
    length = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.concatenate([delta, length], axis=-1).astype(np.float32)


def infer_spring_xyz(point_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从旧 7 维 point_features 近似恢复三向弹簧 K/C。

    旧特征：
        [E/E_base, PRXY, rho/rho_base, is_fixed, log10(K), log10(C), Z/H]
    约定：
        is_fixed == 1.0: 角点，XYZ 三向弹簧
        is_fixed == 0.5: 侧顶杆，Y 向弹簧
        else: 无弹簧
    """
    n = point_features.shape[0]
    spring_k = np.zeros((n, 3), dtype=np.float32)
    spring_c = np.zeros((n, 3), dtype=np.float32)
    if point_features.shape[1] < 6:
        return spring_k, spring_c

    fixed = point_features[:, 3]
    logk = point_features[:, 4]
    logc = point_features[:, 5]
    valid = logk > 0
    k_val = np.zeros(n, dtype=np.float32)
    c_val = np.zeros(n, dtype=np.float32)
    k_val[valid] = np.power(10.0, logk[valid]).astype(np.float32)
    c_val[valid] = np.power(10.0, logc[valid]).astype(np.float32)

    corner = valid & (fixed >= 0.9)
    side = valid & (fixed > 0.1) & (fixed < 0.9)
    spring_k[corner, :] = k_val[corner, None]
    spring_c[corner, :] = c_val[corner, None]
    spring_k[side, 1] = k_val[side]
    spring_c[side, 1] = c_val[side]
    return spring_k, spring_c


def infer_node_type(point_features: np.ndarray) -> np.ndarray:
    """粗略节点类型：0普通，1侧顶杆，2角点夹持。"""
    node_type = np.zeros((point_features.shape[0],), dtype=np.int64)
    if point_features.shape[1] >= 4:
        fixed = point_features[:, 3]
        node_type[(fixed > 0.1) & (fixed < 0.9)] = 1
        node_type[fixed >= 0.9] = 2
    return node_type


def copy_dataset(src_grp: h5py.Group, dst_grp: h5py.Group) -> None:
    for key in src_grp.keys():
        dst_grp.create_dataset(key, data=src_grp[key][:])


def convert_file(src_path: str, dst_path: str, k: int = 12, overwrite: bool = False) -> None:
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"{dst_path} already exists; use --overwrite")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        sample_keys = sorted([key for key in src.keys() if key.startswith("sample_")],
                             key=lambda x: int(x.split("_")[-1]))
        for i, key in enumerate(sample_keys):
            sg = src[key]
            dg = dst.create_group(key)
            copy_dataset(sg, dg)

            points = sg["points"][:].astype(np.float32)
            point_features = sg["point_features"][:].astype(np.float32)
            edge_index = build_knn_edge_index(points, k=k)
            edge_attr = build_edge_attr(points, edge_index)
            spring_k_xyz, spring_c_xyz = infer_spring_xyz(point_features)
            node_type = infer_node_type(point_features)

            if "edge_index" not in dg:
                dg.create_dataset("edge_index", data=edge_index, compression="gzip")
            if "edge_attr" not in dg:
                dg.create_dataset("edge_attr", data=edge_attr, compression="gzip")
            if "spring_k_xyz" not in dg:
                dg.create_dataset("spring_k_xyz", data=spring_k_xyz, compression="gzip")
            if "spring_c_xyz" not in dg:
                dg.create_dataset("spring_c_xyz", data=spring_c_xyz, compression="gzip")
            if "node_type" not in dg:
                dg.create_dataset("node_type", data=node_type, compression="gzip")

            if i % 25 == 0:
                print(f"  {os.path.basename(src_path)}: {i+1}/{len(sample_keys)} samples")
    print(f"saved: {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--files", nargs="+", default=["train.h5", "val.h5", "test.h5"])
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in args.files:
        src = os.path.join(args.data_dir, name)
        base, ext = os.path.splitext(name)
        dst = os.path.join(args.data_dir, f"{base}_graph{ext}")
        convert_file(src, dst, k=args.k, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
