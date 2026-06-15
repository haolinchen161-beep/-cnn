"""
ANSYS 凹槽工件数据集生成 — MeshGraphNet 物理一致版本。

本脚本用于重新生成 GNN / MeshGraphNet 数据集，核心约束：
1. 使用质量归一化振型：MODOPT(..., nrmkey='OFF')。
2. 默认使用一致质量矩阵，不开启 LUMPM。
3. FRF 与阻尼公式按质量归一化模态使用。
4. 保存 MeshGraphNet 需要的 FEM 图字段和必要物理量。
5. 生成阶段剔除三阶与二阶固有频率间隔小于 MIN_GAP32_HZ 的样本。
6. 频率网格在 float32 保存后仍保证严格递增，避免重复频率点。

默认生成 300 个有效样本，保存到 ansys/data/train.h5、val.h5、test.h5。
"""
from __future__ import annotations

import csv
import os
import random
import time

os.environ["PYVISTA_OFF_SCREEN"] = "true"

import h5py
import numpy as np
from ansys.mapdl.core import launch_mapdl
from scipy.stats import qmc


# ===================== 基本配置 =====================
SEED = int(os.getenv("DATASET_SEED", "2"))
np.random.seed(SEED)
random.seed(SEED)

N_SAMPLES = int(os.getenv("N_SAMPLES", "300"))
N_TRAIN = int(os.getenv("N_TRAIN", "240"))
N_VAL = int(os.getenv("N_VAL", "30"))
N_TEST = int(os.getenv("N_TEST", "30"))
assert N_TRAIN + N_VAL + N_TEST == N_SAMPLES, "N_TRAIN + N_VAL + N_TEST must equal N_SAMPLES"

N_MODES = 3
N_FREQS = 60
FREQ_MIN, FREQ_MAX = 1.0, 5000.0
FREQ_GRID_MIN_STEP_HZ = float(os.getenv("FREQ_GRID_MIN_STEP_HZ", "0.01"))
MIN_GAP32_HZ = float(os.getenv("MIN_GAP32_HZ", "200.0"))
MESH_SIZE = 0.006
ZETA_MATERIAL = 0.002
AMPLITUDE_SCALE = 500000.0

# 推荐：质量归一化 + 一致质量矩阵。
USE_MASS_NORMALIZATION = True
USE_LUMPED_MASS = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "mesh_viz")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


# ===================== 固定工件与随机范围 =====================
E_BASE, RHO_BASE, PRXY_BASE = 71.7e9, 2810.0, 0.33
L_BASE, W_BASE, H_BASE = 0.160, 0.060, 0.010
E_RANGE = (0.95, 1.05)
RHO_RANGE = (0.97, 1.03)

GRID_JITTER = 0.15
GAP_ABS = 0.006
BORDER_ABS = 0.006
POCKET_DEPTH_RANGE = (0.30, 0.60)

K_CORNER_RANGE = (5e6, 1e8)
K_SIDE_RANGE = (1e6, 3e7)
ZETA_JOINT_RANGE = (0.005, 0.05)
M_REF = 0.01


# ===================== 凹槽区域定义 =====================
def generate_region_division(n_cols, n_rows, L, W, jitter=GRID_JITTER,
                             gap=GAP_ABS, border=BORDER_ABS):
    n_gaps_x = n_cols - 1
    n_gaps_y = n_rows - 1
    available_x = L - 2 * border - n_gaps_x * gap
    available_y = W - 2 * border - n_gaps_y * gap

    weights_x = np.array([1.0 + np.random.uniform(-jitter, jitter) for _ in range(n_cols)])
    weights_y = np.array([1.0 + np.random.uniform(-jitter, jitter) for _ in range(n_rows)])
    weights_x = weights_x / weights_x.sum() * available_x
    weights_y = weights_y / weights_y.sum() * available_y

    x_pockets, y_pockets = [], []
    cur = border
    for w in weights_x:
        x_pockets.append((cur / L, (cur + w) / L))
        cur += w + gap

    cur = border
    for h in weights_y:
        y_pockets.append((cur / W, (cur + h) / W))
        cur += h + gap

    return x_pockets, y_pockets


def get_pocket_from_cells(x_pockets, y_pockets, cell_indices, n_cols):
    rows = [(idx - 1) // n_cols for idx in cell_indices]
    cols = [(idx - 1) % n_cols for idx in cell_indices]
    xmin = min(x_pockets[c][0] for c in cols)
    xmax = max(x_pockets[c][1] for c in cols)
    ymin = min(y_pockets[r][0] for r in rows)
    ymax = max(y_pockets[r][1] for r in rows)
    return xmin, xmax, ymin, ymax


POCKET_CELLS_5 = [
    [1, 2, 5, 6, 9, 10],
    [3, 4],
    [8, 12],
    [7],
    [11],
]
POCKET_CELLS_6 = [
    [1, 5, 9],
    [2],
    [3, 7],
    [4, 8, 12],
    [6],
    [10, 11],
]
POCKET_CELLS_7 = [
    [1, 6, 11],
    [4, 9, 14],
    [5, 10],
    [15],
    [2, 3],
    [7, 8],
    [12, 13],
]


# ===================== 图边特征 =====================
def build_knn_edge_index(points, k=12):
    n = len(points)
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    k = max(1, min(k, n - 1))
    scale = np.array([L_BASE, W_BASE, H_BASE], dtype=np.float32)
    pts = points / scale
    dist2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    nn_idx = np.argpartition(dist2, kth=k + 1, axis=1)[:, 1:k + 1]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = nn_idx.reshape(-1).astype(np.int64)
    edge_index = np.concatenate([np.stack([src, dst], axis=0), np.stack([dst, src], axis=0)], axis=1)
    return np.unique(edge_index.T, axis=0).T.astype(np.int64)


def build_edge_attr(points, edge_index):
    if edge_index.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    src, dst = edge_index
    scale = np.array([L_BASE, W_BASE, H_BASE], dtype=np.float32)
    delta = (points[dst] - points[src]) / scale
    length = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.concatenate([delta, length], axis=-1).astype(np.float32)


def build_fe_edge_index_from_grid(grid, n_nodes_total):
    """从 FE cell connectivity 提取边；失败时返回 None，由 kNN 兜底。"""
    edge_set = set()
    try:
        cells = np.asarray(grid.cells, dtype=np.int64)
        ptr = 0
        while ptr < len(cells):
            n_cell_nodes = int(cells[ptr])
            ids = cells[ptr + 1:ptr + 1 + n_cell_nodes]
            ptr += 1 + n_cell_nodes
            ids = [int(i) for i in ids if 0 <= int(i) < n_nodes_total]
            if len(ids) < 2:
                continue
            for a in range(len(ids)):
                ia = ids[a]
                for b in range(a + 1, len(ids)):
                    ib = ids[b]
                    edge_set.add((ia, ib))
                    edge_set.add((ib, ia))
    except Exception as exc:
        print(f"  警告: FE拓扑提取失败，将使用kNN兜底: {exc}")
        return None

    if not edge_set:
        return None
    return np.array(sorted(edge_set), dtype=np.int64).T


# ===================== 辅助函数 =====================
def try_get_mode_quantity(mapdl, mode_id, item, dire):
    try:
        mapdl.run(f"*GET, tmp_q, MODE, {mode_id}, {item}, , DIRE, {dire}")
        return float(mapdl.parameters["tmp_q"])
    except Exception:
        return np.nan


def compute_local_thickness(points, pocket_records):
    """根据加工凹槽 XY footprint 给每个节点赋局部残余厚度和加工深度比例。"""
    local_thickness = np.ones(points.shape[0], dtype=np.float32)
    pocket_depth = np.zeros(points.shape[0], dtype=np.float32)
    x, y = points[:, 0], points[:, 1]
    for rec in pocket_records:
        xmin, xmax, ymin, ymax = rec["xmin"], rec["xmax"], rec["ymin"], rec["ymax"]
        residual = rec["bottom_z"] / H_BASE
        depth = rec["depth_frac"]
        inside_xy = (x >= xmin - 1e-8) & (x <= xmax + 1e-8) & (y >= ymin - 1e-8) & (y <= ymax + 1e-8)
        local_thickness[inside_xy] = np.minimum(local_thickness[inside_xy], residual)
        pocket_depth[inside_xy] = np.maximum(pocket_depth[inside_xy], depth)
    return local_thickness, pocket_depth


def _dedup_with_min_step(freqs, min_step):
    freqs = np.asarray(freqs, dtype=np.float64)
    freqs = freqs[np.isfinite(freqs)]
    freqs = freqs[(freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)]
    if freqs.size == 0:
        freqs = np.array([FREQ_MIN, FREQ_MAX], dtype=np.float64)
    freqs = np.sort(freqs)

    cleaned = [float(freqs[0])]
    for f in freqs[1:]:
        if f > cleaned[-1] + min_step:
            cleaned.append(float(f))
    return np.asarray(cleaned, dtype=np.float64)


def _fill_frequency_grid(freqs, target_n, min_step):
    freqs = _dedup_with_min_step(freqs, min_step)
    if freqs[0] > FREQ_MIN + min_step:
        freqs = np.insert(freqs, 0, FREQ_MIN)
    if freqs[-1] < FREQ_MAX - min_step:
        freqs = np.append(freqs, FREQ_MAX)

    while len(freqs) < target_n:
        gaps = np.diff(freqs)
        if len(gaps) == 0:
            new_f = min(FREQ_MAX, freqs[0] + min_step)
            freqs = np.append(freqs, new_f)
            continue
        order = np.argsort(-gaps)
        inserted = False
        for j in order:
            lo, hi = freqs[j], freqs[j + 1]
            if hi - lo <= 2.2 * min_step:
                continue
            if lo > 0:
                candidate = np.sqrt(lo * hi)
            else:
                candidate = 0.5 * (lo + hi)
            if not (lo + min_step < candidate < hi - min_step):
                candidate = 0.5 * (lo + hi)
            freqs = np.insert(freqs, j + 1, candidate)
            inserted = True
            break
        if not inserted:
            raise RuntimeError("Cannot fill frequency grid with strict minimum spacing.")
    return np.sort(freqs)


def _trim_frequency_grid(freqs, target_n, protected_mask):
    freqs = np.asarray(freqs, dtype=np.float64)
    keep = np.ones(len(freqs), dtype=bool)
    n_remove = len(freqs) - target_n
    if n_remove <= 0:
        return freqs

    removable = np.where(~protected_mask)[0]
    while n_remove > 0 and len(removable) > 0:
        kept_idx = np.where(keep)[0]
        removable = np.array([idx for idx in kept_idx if not protected_mask[idx]], dtype=np.int64)
        if len(removable) == 0:
            break
        # 优先删除局部间距最小的非峰值保护点。
        costs = []
        for idx in removable:
            pos = np.where(kept_idx == idx)[0][0]
            left_gap = freqs[kept_idx[pos]] - freqs[kept_idx[pos - 1]] if pos > 0 else np.inf
            right_gap = freqs[kept_idx[pos + 1]] - freqs[kept_idx[pos]] if pos < len(kept_idx) - 1 else np.inf
            costs.append(min(left_gap, right_gap))
        remove_idx = removable[int(np.argmin(costs))]
        keep[remove_idx] = False
        n_remove -= 1

    if n_remove > 0:
        kept_idx = np.where(keep)[0]
        # 保护点不够删时，均匀删点，但保留首尾。
        candidates = kept_idx[1:-1]
        if len(candidates) < n_remove:
            raise RuntimeError("Cannot trim frequency grid to target length.")
        remove_pos = np.linspace(0, len(candidates) - 1, n_remove, dtype=int)
        keep[candidates[remove_pos]] = False
    return freqs[keep]


def finalize_frequency_grid(freqs, omega_k, zeta_k):
    """保证 N_FREQS 个 float32 频率点严格递增，且尽量保护模态峰附近频率点。"""
    min_step = float(FREQ_GRID_MIN_STEP_HZ)
    freqs = _dedup_with_min_step(freqs, min_step)

    protected = np.zeros(len(freqs), dtype=bool)
    for fk, zk in zip(omega_k / (2.0 * np.pi), zeta_k):
        bw_half = max(2.0 * float(zk) * float(fk), min_step * 10.0)
        protected |= (freqs >= fk - bw_half) & (freqs <= fk + bw_half)

    if len(freqs) > N_FREQS:
        freqs = _trim_frequency_grid(freqs, N_FREQS, protected)
    if len(freqs) < N_FREQS:
        freqs = _fill_frequency_grid(freqs, N_FREQS, min_step)

    freqs = np.sort(np.asarray(freqs, dtype=np.float64))[:N_FREQS]

    # 前向/后向安全处理：确保转 float32 后仍严格递增。
    for i in range(1, len(freqs)):
        if freqs[i] <= freqs[i - 1] + min_step:
            freqs[i] = freqs[i - 1] + min_step
    if freqs[-1] > FREQ_MAX:
        freqs[-1] = FREQ_MAX
        for i in range(len(freqs) - 2, -1, -1):
            if freqs[i] >= freqs[i + 1] - min_step:
                freqs[i] = freqs[i + 1] - min_step
    if freqs[0] < FREQ_MIN:
        freqs += (FREQ_MIN - freqs[0])

    freqs32 = freqs.astype(np.float32)
    if len(freqs32) != N_FREQS or not np.all(np.diff(freqs32.astype(np.float64)) > 0.0):
        raise RuntimeError("frequency grid is not strictly increasing after float32 conversion")
    if freqs32[0] < FREQ_MIN - 1e-6 or freqs32[-1] > FREQ_MAX + 1e-6:
        raise RuntimeError("frequency grid is out of configured frequency range")
    return freqs32


def make_frequency_grid(omega_k, zeta_k):
    freqs_parts = []
    prev = FREQ_MIN
    for idx_k, f_k in enumerate(omega_k / (2.0 * np.pi)):
        bw = max(2.0 * float(zeta_k[idx_k]) * float(f_k), FREQ_GRID_MIN_STEP_HZ * 20.0)
        lo = max(FREQ_MIN, f_k - 3.0 * bw)
        hi = min(FREQ_MAX, f_k + 3.0 * bw)
        if prev < lo:
            freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(lo),
                                           max(2, int(5 * (lo - prev) / FREQ_MAX)), endpoint=False))
        # 峰附近多给候选点，后面再统一裁剪到 60。
        freqs_parts.append(np.linspace(lo, hi, max(25, int(40 * (hi - lo) / FREQ_MAX)), endpoint=True))
        freqs_parts.append(np.array([max(FREQ_MIN, f_k - bw), f_k, min(FREQ_MAX, f_k + bw)], dtype=np.float64))
        prev = max(prev, hi)
    if prev < FREQ_MAX:
        freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(FREQ_MAX),
                                       max(2, int(5 * (FREQ_MAX - prev) / FREQ_MAX)), endpoint=True))

    freqs = np.concatenate(freqs_parts)
    return finalize_frequency_grid(freqs, omega_k, zeta_k)


def save_h5(name, idx_slice, arrays):
    idxs = list(idx_slice)
    path = os.path.join(OUT_DIR, name)
    with h5py.File(path, "w") as f:
        f.attrs["modal_normalization"] = "mass"
        f.attrs["mass_matrix"] = "lumped" if USE_LUMPED_MASS else "consistent"
        f.attrs["n_modes"] = N_MODES
        f.attrs["n_freqs"] = N_FREQS
        f.attrs["min_gap32_hz"] = MIN_GAP32_HZ
        f.attrs["freq_grid_min_step_hz"] = FREQ_GRID_MIN_STEP_HZ
        f.attrs["description"] = "MeshGraphNet dataset generated by mass-normalized modal analysis."
        for i, idx in enumerate(idxs):
            grp = f.create_group(f"sample_{i}")
            grp.attrs["modal_normalization"] = "mass"
            grp.attrs["mass_matrix"] = "lumped" if USE_LUMPED_MASS else "consistent"
            for key, values in arrays.items():
                compression = "gzip" if key in {
                    "edge_index", "edge_attr", "modal_phi_xyz", "spring_k_xyz", "spring_c_xyz",
                    "node_type", "pocket_bottom_mask", "cut_region_mask",
                    "local_thickness_ratio", "pocket_depth_ratio",
                } else None
                grp.create_dataset(key, data=values[idx], compression=compression)
    print(f"  保存: {name} ({len(idxs)}样本) -> {path}")


# ===================== 主流程 =====================
print(">>> 生成 Sobol 低偏差序列... ")
SOBOL_BUFFER = 500
sampler = qmc.Sobol(d=2, scramble=True, seed=SEED)
sobol_samples = sampler.random(n=N_SAMPLES + SOBOL_BUFFER)
scaled_sobol = qmc.scale(sobol_samples, [E_RANGE[0], RHO_RANGE[0]], [E_RANGE[1], RHO_RANGE[1]])

print(f"配置: {N_SAMPLES}样本, train/val/test={N_TRAIN}/{N_VAL}/{N_TEST}, {N_MODES}阶模态")
print(f"模态归一化: mass, 质量矩阵: {'lumped' if USE_LUMPED_MASS else 'consistent'}")
print(f"样本过滤: f3 - f2 > {MIN_GAP32_HZ:.1f} Hz")
print(f"频率网格: {N_FREQS}点, float32后严格递增, min_step={FREQ_GRID_MIN_STEP_HZ:.4f} Hz")
print(f"网格: SOLID187, MESH_SIZE={MESH_SIZE*1000:.1f} mm")

print("\n>>> 连接 ANSYS MAPDL...")
mapdl = launch_mapdl(override=True)
print(f">>> 连接成功: {mapdl.version}\n")

arrays = {
    "points": [],
    "edge_index": [],
    "edge_attr": [],
    "point_frf": [],
    "frequencies": [],
    "modal_omega": [],
    "modal_zeta": [],
    "modal_phi": [],
    "modal_phi_xyz": [],
    "modal_phi_exc": [],
    "modal_mass": [],
    "modal_stiffness": [],
    "modal_effm": [],
    "modal_pfact": [],
    "point_features": [],
    "spring_k_xyz": [],
    "spring_c_xyz": [],
    "node_type": [],
    "pocket_bottom_mask": [],
    "cut_region_mask": [],
    "local_thickness_ratio": [],
    "pocket_depth_ratio": [],
    "excitation_index": [],
    "excitation_coord": [],
}

csv_path = os.path.join(OUT_DIR, "sample_log.csv")
csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "sample", "n_nodes", "n_edges", "n_pockets_to_machine", "n_pocket_scheme",
    "depth_range_%", "cut_nodes/bottom_nodes", "exc_x_mm", "exc_y_mm", "exc_z_mm",
    "n_spring_areas", "n_spring_nodes", "modal_norm", "mass_matrix",
    "zeta1", "zeta2", "zeta3", "f1_Hz", "f2_Hz", "f3_Hz", "gap32_Hz",
    "E_ratio", "rho_ratio", "n_cols", "n_rows",
])

valid_samples = 0
attempt_count = 0
skip_gap32_count = 0
t0 = time.time()

while valid_samples < N_SAMPLES:
    attempt_count += 1
    sobol_idx = (attempt_count - 1) % len(scaled_sobol)
    print(f"[有效样本 {valid_samples + 1}/{N_SAMPLES}] 尝试 {attempt_count}", end=" ", flush=True)

    try:
        mapdl.clear()
        mapdl.prep7()
    except Exception:
        print("(reconnect)", end=" ", flush=True)
        try:
            mapdl.exit()
        except Exception:
            pass
        time.sleep(2)
        mapdl = launch_mapdl(override=True)
        mapdl.clear()
        mapdl.prep7()

    try:
        # ---------- 1. 采样参数 ----------
        E = E_BASE * scaled_sobol[sobol_idx, 0]
        rho = RHO_BASE * scaled_sobol[sobol_idx, 1]
        L, W, H = L_BASE, W_BASE, H_BASE

        K_corners, C_corners, K_sides, C_sides = [], [], [], []
        for _ in range(4):
            kc = 10 ** random.uniform(np.log10(K_CORNER_RANGE[0]), np.log10(K_CORNER_RANGE[1]))
            zc = random.uniform(*ZETA_JOINT_RANGE)
            K_corners.append(kc)
            C_corners.append(2.0 * zc * np.sqrt(kc * M_REF))
        for _ in range(3):
            ks = 10 ** random.uniform(np.log10(K_SIDE_RANGE[0]), np.log10(K_SIDE_RANGE[1]))
            zs = random.uniform(*ZETA_JOINT_RANGE)
            K_sides.append(ks)
            C_sides.append(2.0 * zs * np.sqrt(ks * M_REF))

        mapdl.mp("EX", 1, E)
        mapdl.mp("PRXY", 1, PRXY_BASE)
        mapdl.mp("DENS", 1, rho)

        # ---------- 2. 几何与凹槽 ----------
        num_machined = random.choice([5, 6, 7])
        if num_machined == 5:
            pocket_cells, n_cols, n_rows = POCKET_CELLS_5, 4, 3
        elif num_machined == 6:
            pocket_cells, n_cols, n_rows = POCKET_CELLS_6, 4, 3
        else:
            pocket_cells, n_cols, n_rows = POCKET_CELLS_7, 5, 3

        x_pockets, y_pockets = generate_region_division(n_cols, n_rows, L, W)
        n_pockets_to_machine = random.randint(1, num_machined)
        pockets_to_machine = random.sample(range(num_machined), n_pockets_to_machine)

        mapdl.btol(0.0001)
        mapdl.block(0, L, 0, W, 0, H)
        wk_vol = int(mapdl.geometry.vnum[0])

        pocket_records = []
        pocket_depth_fracs = []
        bool_ok = True
        for pocket_idx in pockets_to_machine:
            depth_frac = random.uniform(*POCKET_DEPTH_RANGE)
            pocket_depth_fracs.append(depth_frac)
            pocket_zmin = H - depth_frac * H
            cells = pocket_cells[pocket_idx]
            xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(x_pockets, y_pockets, cells, n_cols)
            xmin_p, xmax_p = max(xmin_frac * L, 0.0), min(xmax_frac * L, L)
            ymin_p, ymax_p = max(ymin_frac * W, 0.0), min(ymax_frac * W, W)
            if xmax_p <= xmin_p or ymax_p <= ymin_p or pocket_zmin >= H:
                continue
            pocket_records.append({
                "xmin": xmin_p, "xmax": xmax_p,
                "ymin": ymin_p, "ymax": ymax_p,
                "bottom_z": pocket_zmin,
                "depth_frac": depth_frac,
            })

            mapdl.allsel()
            old_vols = set(mapdl.geometry.vnum)
            mapdl.block(xmin_p, xmax_p, ymin_p, ymax_p, pocket_zmin, H + 0.001)
            new_vols = set(mapdl.geometry.vnum) - old_vols
            if not new_vols:
                continue
            pk_vol = int(list(new_vols)[0])
            try:
                mapdl.vsbv(wk_vol, pk_vol)
            except Exception as exc:
                print(f"  VSBV失败: {exc}")
                bool_ok = False
                break
            mapdl.allsel()
            remaining_vols = mapdl.geometry.vnum
            if len(remaining_vols) > 0:
                wk_vol = int(remaining_vols[0])
            else:
                bool_ok = False
                break
        if not bool_ok:
            print("  跳过: 布尔运算失败")
            continue

        # ---------- 3. 网格 ----------
        mapdl.et(1, "SOLID187")
        mapdl.mshape(1, "3D")
        mapdl.mshkey(0)
        mapdl.esize(MESH_SIZE)
        try:
            mapdl.vmesh("ALL")
        except Exception:
            mapdl.smrtsize(4)
            mapdl.vmesh("ALL")

        all_node_ids = mapdl.mesh.nnum
        all_node_coords = np.array(mapdl.mesh.nodes, dtype=np.float32)
        n_nodes_total = len(all_node_ids)
        node_id_to_idx = {int(nid): idx for idx, nid in enumerate(all_node_ids)}

        try:
            edge_index = build_fe_edge_index_from_grid(mapdl.mesh._grid, n_nodes_total)
        except Exception as exc:
            print(f"  警告: 获取FE grid失败: {exc}")
            edge_index = None
        if edge_index is None:
            edge_index = build_knn_edge_index(all_node_coords, k=12)
        edge_attr = build_edge_attr(all_node_coords, edge_index)

        local_thickness_ratio, pocket_depth_ratio = compute_local_thickness(all_node_coords, pocket_records)

        point_features = np.zeros((n_nodes_total, 7), dtype=np.float32)
        point_features[:, 0] = E / E_BASE
        point_features[:, 1] = PRXY_BASE
        point_features[:, 2] = rho / RHO_BASE
        point_features[:, 3] = 0.0
        point_features[:, 4] = -1.0
        point_features[:, 5] = -1.0
        point_features[:, 6] = local_thickness_ratio

        spring_k_xyz = np.zeros((n_nodes_total, 3), dtype=np.float32)
        spring_c_xyz = np.zeros((n_nodes_total, 3), dtype=np.float32)
        node_type = np.zeros((n_nodes_total,), dtype=np.int64)
        pocket_bottom_mask = np.zeros((n_nodes_total,), dtype=np.uint8)
        cut_region_mask = np.zeros((n_nodes_total,), dtype=np.uint8)

        # ---------- 4. 凹槽底面/切削区节点 ----------
        pocket_cut_indices = []
        pocket_bottom_any_indices = []
        tool_r = MESH_SIZE / 2.0
        cut_band = MESH_SIZE * 0.6

        mapdl.allsel()
        for rec in pocket_records:
            xmin_p, xmax_p = rec["xmin"], rec["xmax"]
            ymin_p, ymax_p = rec["ymin"], rec["ymax"]
            pocket_bottom_z = rec["bottom_z"]
            mapdl.nsel("S", "LOC", "Z", pocket_bottom_z, pocket_bottom_z + 1e-6)
            margin = 1e-4
            mapdl.nsel("R", "LOC", "X", xmin_p + margin, xmax_p - margin)
            mapdl.nsel("R", "LOC", "Y", ymin_p + margin, ymax_p - margin)
            bottom_nids = set(int(nid) for nid in mapdl.mesh.nnum)
            for nid in bottom_nids:
                if nid not in node_id_to_idx:
                    continue
                idx = node_id_to_idx[nid]
                if idx not in pocket_bottom_any_indices:
                    pocket_bottom_any_indices.append(idx)
                x, y = all_node_coords[idx, 0], all_node_coords[idx, 1]
                dist_to_wall = min(x - xmin_p, xmax_p - x, y - ymin_p, ymax_p - y)
                if abs(dist_to_wall - tool_r) < cut_band and idx not in pocket_cut_indices:
                    pocket_cut_indices.append(idx)
        mapdl.allsel()

        if pocket_bottom_any_indices:
            pocket_bottom_mask[np.array(pocket_bottom_any_indices, dtype=np.int64)] = 1
            node_type[np.array(pocket_bottom_any_indices, dtype=np.int64)] = 1
        if pocket_cut_indices:
            cut_region_mask[np.array(pocket_cut_indices, dtype=np.int64)] = 1
            node_type[np.array(pocket_cut_indices, dtype=np.int64)] = 2

        # ---------- 5. 柔性装夹 ----------
        mapdl.et(2, "COMBIN14"); mapdl.keyopt(2, 2, 1)  # UX
        mapdl.et(3, "COMBIN14"); mapdl.keyopt(3, 2, 2)  # UY
        mapdl.et(4, "COMBIN14"); mapdl.keyopt(4, 2, 3)  # UZ

        clamp_len = 0.010
        all_clamp_areas = [
            (0, clamp_len, 0, 1e-4),
            (L - clamp_len, L, 0, 1e-4),
            (0, clamp_len, W - 1e-4, W),
            (L - clamp_len, L, W - 1e-4, W),
        ]
        corner_excl = clamp_len + H / 2.0
        x_min, x_max = corner_excl, L - corner_excl
        side_choices = [0, 0, 1] if random.random() < 0.5 else [1, 1, 0]
        sides_y = [0, W]
        for side_idx in (0, 1):
            n_on_side = sum(1 for s in side_choices if s == side_idx)
            if n_on_side == 0:
                continue
            xs = []
            min_gap = 2 * H
            for _ in range(n_on_side):
                for _attempt in range(100):
                    x_try = random.uniform(x_min, x_max)
                    if all(abs(x_try - x_exist) >= min_gap for x_exist in xs):
                        xs.append(x_try)
                        break
                else:
                    xs.append(x_min + len(xs) * H)
            cy = sides_y[side_idx]
            for x_c in xs:
                all_clamp_areas.append((x_c - H / 2, x_c + H / 2, cy - 1e-4, cy + 1e-4))

        max_node_id = int(all_node_ids.max())
        spring_info = []  # [(original_nid, Cx, Cy, Cz)]
        spring_node_set = set()
        real_const_num = 2
        for idx_area, (xmin, xmax, ymin, ymax) in enumerate(all_clamp_areas):
            mapdl.nsel("S", "LOC", "X", xmin, xmax)
            mapdl.nsel("R", "LOC", "Y", ymin, ymax)
            mapdl.nsel("R", "LOC", "Z", 0, H)
            n_selected = mapdl.mesh.n_node
            if n_selected <= 0:
                continue
            clamp_nodes = mapdl.mesh.nnum
            is_corner = idx_area < 4
            K_this = K_corners[idx_area] if is_corner else K_sides[idx_area - 4]
            C_this = C_corners[idx_area] if is_corner else C_sides[idx_area - 4]
            K_each = K_this / n_selected
            C_each = C_this / n_selected
            mapdl.r(real_const_num, K_each, 0.0)

            for n1 in clamp_nodes:
                n1_int = int(n1)
                if n1_int not in node_id_to_idx or n1_int in spring_node_set:
                    continue
                spring_node_set.add(n1_int)
                idx_n1 = node_id_to_idx[n1_int]
                x1, y1, z1 = all_node_coords[idx_n1]
                max_node_id += 1
                n2 = max_node_id
                mapdl.n(n2, x1, y1, z1)
                mapdl.d(n2, "ALL")

                if is_corner:
                    mapdl.type(2); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                    mapdl.type(3); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                    mapdl.type(4); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                    point_features[idx_n1, 3] = 1.0
                    spring_info.append((n1_int, C_each, C_each, C_each))
                    spring_k_xyz[idx_n1, :] = K_each
                    spring_c_xyz[idx_n1, :] = C_each
                    node_type[idx_n1] = 4
                else:
                    mapdl.type(3); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                    point_features[idx_n1, 3] = 0.5
                    spring_info.append((n1_int, 0.0, C_each, 0.0))
                    spring_k_xyz[idx_n1, 1] = K_each
                    spring_c_xyz[idx_n1, 1] = C_each
                    node_type[idx_n1] = 3
                point_features[idx_n1, 4] = np.log10(K_each)
                point_features[idx_n1, 5] = np.log10(C_each)
            real_const_num += 1
        mapdl.allsel()

        # ---------- 6. 质量归一化模态分析 ----------
        mapdl.slashsolu()
        mapdl.antype("MODAL")
        if USE_LUMPED_MASS:
            mapdl.lumpm("ON")
        else:
            try:
                mapdl.lumpm("OFF")
            except Exception:
                pass
        nrmkey = "OFF" if USE_MASS_NORMALIZATION else "ON"
        mapdl.modopt("LANB", N_MODES, nrmkey=nrmkey)
        mapdl.solve()

        # ---------- 7. 结果提取 ----------
        mapdl.post1()
        current_nnum = mapdl.mesh.nnum
        omega_k = np.zeros(N_MODES, dtype=np.float32)
        phi_x = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
        phi_y = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
        phi_z = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
        effm_k = np.full((N_MODES, 3), np.nan, dtype=np.float32)
        pfact_k = np.full((N_MODES, 3), np.nan, dtype=np.float32)

        for k in range(1, N_MODES + 1):
            mapdl.set(1, k)
            f_hz = mapdl.post_processing.freq
            omega_k[k - 1] = 2.0 * np.pi * f_hz
            disp = np.array(mapdl.post_processing.nodal_displacement("ALL"), dtype=np.float32)
            for idx_curr, nid in enumerate(current_nnum):
                nid_int = int(nid)
                if nid_int in node_id_to_idx:
                    idx_orig = node_id_to_idx[nid_int]
                    phi_x[idx_orig, k - 1] = disp[idx_curr, 0]
                    phi_y[idx_orig, k - 1] = disp[idx_curr, 1]
                    phi_z[idx_orig, k - 1] = disp[idx_curr, 2]

            for j, dire in enumerate(["X", "Y", "Z"]):
                effm_k[k - 1, j] = try_get_mode_quantity(mapdl, k, "EFFM", dire)
                pfact_k[k - 1, j] = try_get_mode_quantity(mapdl, k, "PFACT", dire)

        freq_hz = omega_k / (2.0 * np.pi)
        if not np.all(np.diff(freq_hz.astype(np.float64)) > 0.0):
            print(f"  跳过: 模态频率未递增 f={freq_hz}")
            continue
        gap32_hz = float(freq_hz[2] - freq_hz[1])
        if gap32_hz <= MIN_GAP32_HZ:
            skip_gap32_count += 1
            print(f"  跳过: f3-f2={gap32_hz:.2f} Hz <= {MIN_GAP32_HZ:.1f} Hz")
            continue

        phi_xyz = np.stack([phi_x, phi_y, phi_z], axis=-1).astype(np.float32)
        modal_mass = np.ones(N_MODES, dtype=np.float32) if USE_MASS_NORMALIZATION else np.full(N_MODES, np.nan, dtype=np.float32)
        modal_stiffness = (omega_k ** 2 * modal_mass).astype(np.float32)

        # ---------- 8. 激励点 ----------
        if pocket_cut_indices:
            cut_coords = all_node_coords[pocket_cut_indices]
            center = cut_coords.mean(axis=0)
            dists = np.linalg.norm(cut_coords - center, axis=1)
            exc_idx = pocket_cut_indices[int(np.argmin(dists))]
        elif pocket_bottom_any_indices:
            any_coords = all_node_coords[pocket_bottom_any_indices]
            center = any_coords.mean(axis=0)
            dists = np.linalg.norm(any_coords - center, axis=1)
            exc_idx = pocket_bottom_any_indices[int(np.argmin(dists))]
        else:
            exc_idx = np.random.randint(0, n_nodes_total)
        phi_exc_xyz = phi_xyz[exc_idx, :, :].copy()
        exc_actual = all_node_coords[exc_idx]

        # ---------- 9. 阻尼比与 FRF ----------
        zeta_k = np.zeros(N_MODES, dtype=np.float32)
        for k in range(N_MODES):
            wk = omega_k[k]
            zeta_boundary = 0.0
            for ansys_nid, cx, cy, cz in spring_info:
                if ansys_nid in node_id_to_idx:
                    idx_orig = node_id_to_idx[ansys_nid]
                    diss = cx * phi_x[idx_orig, k] ** 2 + cy * phi_y[idx_orig, k] ** 2 + cz * phi_z[idx_orig, k] ** 2
                    zeta_boundary += diss / (2.0 * wk)
            zeta_k[k] = ZETA_MATERIAL + zeta_boundary

        freqs = make_frequency_grid(omega_k, zeta_k)
        if len(freqs) != N_FREQS or not np.all(np.diff(freqs.astype(np.float64)) > 0.0):
            raise RuntimeError("make_frequency_grid returned invalid frequency grid")
        omega_q = 2.0 * np.pi * freqs
        frf = np.zeros((n_nodes_total, len(freqs), 2), dtype=np.float32)
        for k in range(N_MODES):
            wk = omega_k[k]
            zk = zeta_k[k]
            pk_z = phi_z[:, k] * phi_exc_xyz[k, 2]
            dw = wk ** 2 - omega_q ** 2
            gm = 2.0 * zk * wk * omega_q
            denom = np.maximum(dw ** 2 + gm ** 2 + 1e-6, 1.0)
            frf[:, :, 0] += np.outer(pk_z, AMPLITUDE_SCALE * dw / denom)
            frf[:, :, 1] += np.outer(pk_z, -AMPLITUDE_SCALE * gm / denom)

        # ---------- 10. 保存到内存 ----------
        arrays["points"].append(all_node_coords)
        arrays["edge_index"].append(edge_index)
        arrays["edge_attr"].append(edge_attr)
        arrays["point_frf"].append(frf)
        arrays["frequencies"].append(freqs)
        arrays["modal_omega"].append(omega_k)
        arrays["modal_zeta"].append(zeta_k)
        arrays["modal_phi"].append(phi_xyz)
        arrays["modal_phi_xyz"].append(phi_xyz)
        arrays["modal_phi_exc"].append(phi_exc_xyz)
        arrays["modal_mass"].append(modal_mass)
        arrays["modal_stiffness"].append(modal_stiffness)
        arrays["modal_effm"].append(effm_k)
        arrays["modal_pfact"].append(pfact_k)
        arrays["point_features"].append(point_features)
        arrays["spring_k_xyz"].append(spring_k_xyz)
        arrays["spring_c_xyz"].append(spring_c_xyz)
        arrays["node_type"].append(node_type)
        arrays["pocket_bottom_mask"].append(pocket_bottom_mask)
        arrays["cut_region_mask"].append(cut_region_mask)
        arrays["local_thickness_ratio"].append(local_thickness_ratio)
        arrays["pocket_depth_ratio"].append(pocket_depth_ratio)
        arrays["excitation_index"].append(np.array(exc_idx, dtype=np.int64))
        arrays["excitation_coord"].append(exc_actual.astype(np.float32))

        depth_min = min(pocket_depth_fracs) * 100 if pocket_depth_fracs else 0.0
        depth_max = max(pocket_depth_fracs) * 100 if pocket_depth_fracs else 0.0
        n_cut = len(pocket_cut_indices)
        n_bottom = len(pocket_bottom_any_indices)
        min_df = float(np.min(np.diff(freqs.astype(np.float64))))
        print(
            f"N={n_nodes_total}, E={edge_index.shape[1]}, 加工{n_pockets_to_machine}/{num_machined}, "
            f"深度{depth_min:.0f}~{depth_max:.0f}%, cut/bottom={n_cut}/{n_bottom}, "
            f"f=[{freq_hz[0]:.1f}, {freq_hz[1]:.1f}, {freq_hz[2]:.1f}]Hz, "
            f"gap32={gap32_hz:.1f}Hz, df_min={min_df:.4f}Hz, "
            f"zeta=[{zeta_k[0]:.4f}, {zeta_k[1]:.4f}, {zeta_k[2]:.4f}]"
        )

        csv_writer.writerow([
            valid_samples + 1, n_nodes_total, edge_index.shape[1], n_pockets_to_machine, num_machined,
            f"{depth_min:.1f}~{depth_max:.1f}", f"{n_cut}/{n_bottom}",
            f"{exc_actual[0]*1000:.2f}", f"{exc_actual[1]*1000:.2f}", f"{exc_actual[2]*1000:.2f}",
            len(all_clamp_areas), len(spring_info), "mass", "lumped" if USE_LUMPED_MASS else "consistent",
            f"{zeta_k[0]:.6f}", f"{zeta_k[1]:.6f}", f"{zeta_k[2]:.6f}",
            f"{freq_hz[0]:.2f}", f"{freq_hz[1]:.2f}", f"{freq_hz[2]:.2f}", f"{gap32_hz:.2f}",
            f"{E/E_BASE:.4f}", f"{rho/RHO_BASE:.4f}", n_cols, n_rows,
        ])
        csv_file.flush()

        # 可视化只保存前 5 个样本，避免 300 个样本生成过慢。
        if valid_samples < 5:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import pyvista as pv
                mapdl.allsel()
                grid = mapdl.mesh._grid
                plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
                plotter.add_mesh(grid, color="lightblue", show_edges=True, edge_color="gray", line_width=0.3, opacity=0.8)
                if pocket_bottom_any_indices:
                    plotter.add_points(all_node_coords[pocket_bottom_any_indices], color="red", point_size=5,
                                       render_points_as_spheres=True)
                plotter.add_points(exc_actual.reshape(1, -1), color="green", point_size=15,
                                   render_points_as_spheres=True)
                plotter.add_text(
                    f"Sample {valid_samples + 1}: mass-normalized, gap32={gap32_hz:.1f}Hz",
                    font_size=10,
                )
                plotter.camera_position = "iso"
                plotter.screenshot(os.path.join(VIZ_DIR, f"sample_{valid_samples:03d}_mesh.png"))
                plotter.close()
            except Exception as exc:
                print(f"  可视化失败: {exc}")

        valid_samples += 1
        time.sleep(0.2)

    except Exception as exc:
        print(f"  跳过本次尝试: {exc}")
        try:
            mapdl.clear()
        except Exception:
            pass
        continue

csv_file.close()
try:
    mapdl.exit()
except Exception:
    pass

elapsed = time.time() - t0
print(f"\n生成完成, 有效样本={N_SAMPLES}, 总尝试={attempt_count}, gap32过滤={skip_gap32_count}, 耗时={elapsed:.0f}s")
print(f"CSV日志: {csv_path}")

print("\n保存 HDF5...")
save_h5("train.h5", range(N_TRAIN), arrays)
save_h5("val.h5", range(N_TRAIN, N_TRAIN + N_VAL), arrays)
save_h5("test.h5", range(N_TRAIN + N_VAL, N_SAMPLES), arrays)

# 简单 FRF 可视化
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords0 = arrays["points"][0]
    frf0 = arrays["point_frf"][0]
    freqs0 = arrays["frequencies"][0]
    amp0 = np.sqrt(frf0[..., 0] ** 2 + frf0[..., 1] ** 2)
    n_nodes0 = len(coords0)
    selected_idx = np.linspace(0, n_nodes0 - 1, num=min(5, n_nodes0), dtype=int)

    fig = plt.figure(figsize=(18, 12))
    for i, idx in enumerate(selected_idx):
        ax = fig.add_subplot(len(selected_idx), 1, i + 1)
        amp_db = 20 * np.log10(amp0[idx] + 1e-12)
        ax.semilogx(freqs0, amp_db, linewidth=1.0)
        for k in range(N_MODES):
            ax.axvline(arrays["modal_omega"][0][k] / (2.0 * np.pi), linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"node {idx}\nmag(dB)")
        ax.grid(alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    fig.suptitle("Mass-normalized modal FRF check")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "sample_000_frf.png"), dpi=150)
    plt.close()
    print(f"可视化保存: {VIZ_DIR}")
except Exception as exc:
    print(f"FRF可视化失败: {exc}")
