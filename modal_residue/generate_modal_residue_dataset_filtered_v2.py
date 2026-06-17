# -*- coding: utf-8 -*-
"""
ANSYS 凹槽工件模态留数数据集生成程序。

用途：
1. 批量生成 3D 几何 + 弹性装夹边界条件下的质量归一化模态参数。
2. 保存前 N_MODES 阶固有角频率、振型、模态留数：A_r(x)=phi_r,z(x)*phi_r,z(x_f)。
3. 使用 Python 模态叠加公式生成 Z-Z 复数 FRF 标签，默认保存物理量 m/N。
4. 后续神经网络建议预测 modal_omega 与 modal_residue_z，再由公式重建 FRF。

核心约束：
1. 使用质量归一化振型：MODOPT(..., nrmkey='OFF')。
2. 默认使用一致质量矩阵，不开启 LUMPM。
3. 激励点来自随机已加工凹槽的底面中心附近节点，且不贴凹槽边缘。
4. 频率上限默认按第 N_MODES 阶自适应，不再固定 5000 Hz；频率网格在 float32 保存后仍严格递增，并优先保护各模态峰附近频点。

默认生成 300 个有效样本，保存到 ./data_modal_residue_filtered300/train.h5、val.h5、test.h5。
本版采用受控随机数据集：clamp_level × coverage_level 分层，保留 5/6/7 凹槽布局和边界扰动。
默认加入一个简单近频过滤：任意相邻模态相对间隔 < MIN_RELATIVE_MODE_GAP 时跳过，默认 0.03。
可通过环境变量覆盖：N_SAMPLES、N_TRAIN、N_VAL、N_TEST、N_MODES、N_FREQS、MIN_RELATIVE_MODE_GAP、FRF_OUTPUT_SCALE、SAVE_POINT_FRF。
"""
from __future__ import annotations

import csv
import math
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

N_MODES = int(os.getenv("N_MODES", "10"))
N_FREQS = int(os.getenv("N_FREQS", "120"))
FREQ_MIN = float(os.getenv("FREQ_MIN_HZ", "1.0"))
# 默认不再固定 5000 Hz。若确实需要固定上限，可设置环境变量 FREQ_MAX_HZ。
# 默认模式下，每个样本的频率上限由第 N_MODES 阶频率自适应决定。
FREQ_MAX_FIXED_ENV = os.getenv("FREQ_MAX_HZ", "").strip()
FREQ_MAX_FIXED = float(FREQ_MAX_FIXED_ENV) if FREQ_MAX_FIXED_ENV else None
FREQ_MAX_MARGIN_RATIO = float(os.getenv("FREQ_MAX_MARGIN_RATIO", "0.05"))
FREQ_MAX_MARGIN_MIN_HZ = float(os.getenv("FREQ_MAX_MARGIN_MIN_HZ", "100.0"))
FREQ_MAX_MARGIN_BW_MULT = float(os.getenv("FREQ_MAX_MARGIN_BW_MULT", "4.0"))
FREQ_MAX_HARD_HZ = float(os.getenv("FREQ_MAX_HARD_HZ", "50000.0"))
FREQ_GRID_MIN_STEP_HZ = float(os.getenv("FREQ_GRID_MIN_STEP_HZ", "0.01"))
MESH_SIZE = 0.006
ZETA_MATERIAL = float(os.getenv("ZETA_MATERIAL", "0.002"))
ZETA_JOINT_BASE = float(os.getenv("ZETA_JOINT_BASE", "0.015"))
ZETA_JOINT_JITTER = float(os.getenv("ZETA_JOINT_JITTER", "0.20"))
# FRF 默认保存为物理量 m/N；如确需数值缩放，可设置 FRF_OUTPUT_SCALE。
FRF_OUTPUT_SCALE = float(os.getenv("FRF_OUTPUT_SCALE", "1.0"))
SAVE_POINT_FRF = os.getenv("SAVE_POINT_FRF", "0").strip().lower() in {"1", "true", "yes", "y"}
SAVE_POINT_FRF_QC_COUNT = int(os.getenv("SAVE_POINT_FRF_QC_COUNT", "5"))
# 简单近频过滤：若任意相邻模态的相对间隔 (f_{r+1}-f_r)/f_r 小于该阈值，则跳过样本。
# 设置为 0 可关闭过滤。默认 0.03，即 3%。
MIN_RELATIVE_MODE_GAP = float(os.getenv("MIN_RELATIVE_MODE_GAP", "0.03"))

# 推荐：质量归一化 + 一致质量矩阵。
USE_MASS_NORMALIZATION = True
USE_LUMPED_MASS = False

OUT_DIR = os.getenv("OUT_DIR", os.path.join(os.path.dirname(__file__), "data_modal_residue_filtered300"))
VIZ_DIR = os.getenv("VIZ_DIR", os.path.join(os.path.dirname(__file__), "mesh_viz_modal_residue_filtered300"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


# ===================== 固定工件与随机范围 =====================
E_BASE, RHO_BASE, PRXY_BASE = 71.7e9, 2810.0, 0.33
L_BASE, W_BASE, H_BASE = 0.160, 0.060, 0.010
E_RANGE = (0.95, 1.05)
RHO_RANGE = (0.97, 1.03)

GRID_JITTER_RANGE = (float(os.getenv("GRID_JITTER_MIN", "0.08")), float(os.getenv("GRID_JITTER_MAX", "0.15")))
GAP_ABS = 0.006
BORDER_ABS = 0.006
TARGET_DEPTH_RANGE = (float(os.getenv("TARGET_DEPTH_MIN", "0.25")), float(os.getenv("TARGET_DEPTH_MAX", "0.65")))
TARGET_DEPTH_MODE = float(os.getenv("TARGET_DEPTH_MODE", "0.45"))
CURRENT_PROGRESS_RANGE = (float(os.getenv("CURRENT_PROGRESS_MIN", "0.25")), float(os.getenv("CURRENT_PROGRESS_MAX", "1.00")))

CLAMP_LEVELS = {
    "soft": {"K_corner_base": 1.5e7, "K_side_base": 3.0e6},
    "normal": {"K_corner_base": 3.0e7, "K_side_base": 8.0e6},
    "hard": {"K_corner_base": 6.0e7, "K_side_base": 1.6e7},
}
CLAMP_WEIGHTS = {"soft": 0.25, "normal": 0.50, "hard": 0.25}
COVERAGE_WEIGHTS = {"low": 0.25, "medium": 0.50, "high": 0.25}
CLAMP_LEVEL_CODE = {"soft": 0, "normal": 1, "hard": 2}
COVERAGE_LEVEL_CODE = {"low": 0, "medium": 1, "high": 2}
K_CORNER_JITTER = float(os.getenv("K_CORNER_JITTER", "0.20"))
K_SIDE_JITTER = float(os.getenv("K_SIDE_JITTER", "0.30"))
M_REF = 0.01


# ===================== 凹槽区域定义 =====================
def generate_region_division(n_cols, n_rows, L, W, jitter=None,
                             gap=GAP_ABS, border=BORDER_ABS):
    if jitter is None:
        jitter = random.uniform(GRID_JITTER_RANGE[0], GRID_JITTER_RANGE[1])
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


# ===================== 受控随机采样计划 =====================
def _allocate_counts(total, labels, weights):
    """按权重把 total 个样本分配给 labels，使用 largest remainder 保证总数严格相等。"""
    raw = np.array([float(weights[label]) for label in labels], dtype=np.float64)
    raw = raw / raw.sum() * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {label: int(counts[i]) for i, label in enumerate(labels)}


def _balanced_layouts(n):
    """在一个 clamp×coverage 组内部尽量均匀安排 5/6/7 布局。"""
    layouts = [5, 6, 7]
    out = [layouts[i % len(layouts)] for i in range(int(n))]
    random.shuffle(out)
    return out


def build_sample_plan(n_train, n_val, n_test):
    """
    生成固定长度的样本计划。

    主分层只用 clamp_level × coverage_level = 3×3。
    每个 split 内按相同权重分配，保证 train/val/test 都覆盖 soft/normal/hard 和 low/medium/high。
    layout_type=5/6/7 不作为硬分层，只在每个组内部尽量均匀出现。
    """
    plan = []
    split_specs = [("train", int(n_train)), ("val", int(n_val)), ("test", int(n_test))]
    clamp_labels = ["soft", "normal", "hard"]
    coverage_labels = ["low", "medium", "high"]
    combo_labels = [(c, g) for c in clamp_labels for g in coverage_labels]
    combo_weights = {
        (c, g): CLAMP_WEIGHTS[c] * COVERAGE_WEIGHTS[g]
        for c, g in combo_labels
    }
    for split, n_split in split_specs:
        combo_counts = _allocate_counts(n_split, combo_labels, combo_weights)
        for (clamp_level, coverage_level), n_combo in combo_counts.items():
            for layout_type in _balanced_layouts(n_combo):
                plan.append({
                    "split": split,
                    "clamp_level": clamp_level,
                    "coverage_level": coverage_level,
                    "layout_type": int(layout_type),
                })
    random.shuffle(plan)
    assert len(plan) == int(n_train + n_val + n_test)
    return plan


def sample_clamp_parameters(clamp_level):
    """样本级装夹强弱 + 样本内部小扰动。"""
    cfg = CLAMP_LEVELS[clamp_level]
    k_corner_base = float(cfg["K_corner_base"])
    k_side_base = float(cfg["K_side_base"])
    K_corners, C_corners, zeta_corners = [], [], []
    K_sides, C_sides, zeta_sides = [], [], []

    for _ in range(4):
        kc = k_corner_base * random.uniform(1.0 - K_CORNER_JITTER, 1.0 + K_CORNER_JITTER)
        zc = ZETA_JOINT_BASE * random.uniform(1.0 - ZETA_JOINT_JITTER, 1.0 + ZETA_JOINT_JITTER)
        K_corners.append(kc)
        C_corners.append(2.0 * zc * np.sqrt(kc * M_REF))
        zeta_corners.append(zc)
    for _ in range(3):
        ks = k_side_base * random.uniform(1.0 - K_SIDE_JITTER, 1.0 + K_SIDE_JITTER)
        zs = ZETA_JOINT_BASE * random.uniform(1.0 - ZETA_JOINT_JITTER, 1.0 + ZETA_JOINT_JITTER)
        K_sides.append(ks)
        C_sides.append(2.0 * zs * np.sqrt(ks * M_REF))
        zeta_sides.append(zs)
    return (
        K_corners, C_corners, K_sides, C_sides,
        np.asarray(zeta_corners + zeta_sides, dtype=np.float32),
        k_corner_base, k_side_base,
    )


def choose_layout(layout_type):
    if int(layout_type) == 5:
        return POCKET_CELLS_5, 4, 3
    if int(layout_type) == 6:
        return POCKET_CELLS_6, 4, 3
    if int(layout_type) == 7:
        return POCKET_CELLS_7, 5, 3
    raise ValueError(f"unknown layout_type={layout_type}")


def sample_active_count(num_cells, coverage_level):
    """coverage 只控制加工区域数量，不控制深度。"""
    n = int(num_cells)
    if coverage_level == "low":
        lo, hi = 1, max(1, int(math.ceil(0.30 * n)))
    elif coverage_level == "medium":
        lo = max(1, int(math.floor(0.30 * n)))
        hi = max(lo, int(math.ceil(0.65 * n)))
    elif coverage_level == "high":
        lo = max(1, int(math.floor(0.65 * n)))
        hi = n
    else:
        raise ValueError(f"unknown coverage_level={coverage_level}")
    return random.randint(lo, hi)


def sample_machining_state(num_cells, coverage_level):
    """
    返回某一加工时刻的状态。
    已完成区域 depth=target_depth；当前加工区域 depth=target_depth*current_progress；未加工区域 depth=0。
    """
    order = random.sample(range(int(num_cells)), int(num_cells))
    active_count = sample_active_count(num_cells, coverage_level)
    target_depth = random.triangular(TARGET_DEPTH_RANGE[0], TARGET_DEPTH_RANGE[1], TARGET_DEPTH_MODE)

    # 大部分样本保留一个“当前加工”区域；高覆盖且全部区域激活时，少量样本允许全部完成。
    all_finished = (active_count == int(num_cells)) and (random.random() < 0.20)
    if all_finished:
        finished_count = active_count
        current_cell = -1
        current_progress = 1.0
    else:
        finished_count = max(0, active_count - 1)
        current_cell = int(order[finished_count])
        current_progress = random.uniform(CURRENT_PROGRESS_RANGE[0], CURRENT_PROGRESS_RANGE[1])

    depth_by_cell = np.zeros(int(num_cells), dtype=np.float32)
    for cell in order[:finished_count]:
        depth_by_cell[int(cell)] = target_depth
    if current_cell >= 0:
        depth_by_cell[current_cell] = target_depth * current_progress

    return {
        "pocket_order": order,
        "active_count": int(active_count),
        "finished_count": int(finished_count),
        "current_cell": int(current_cell),
        "current_progress": float(current_progress),
        "target_depth_ratio": float(target_depth),
        "depth_by_cell": depth_by_cell,
    }


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


def select_bottom_center_excitation(points, pocket_records):
    """
    随机选择一个已加工凹槽，然后在该凹槽底面中心附近选最近节点。
    要求节点在凹槽底面且不贴边；若目标凹槽无合格点，则尝试其它已加工凹槽。
    不使用切削边缘点，不使用真实振型做选择，避免标签泄露。
    """
    if not pocket_records:
        return None, None, None

    order = list(range(len(pocket_records)))
    random.shuffle(order)
    z_tols = [1e-6, 3e-6, 1e-5]

    for pocket_id in order:
        rec = pocket_records[pocket_id]
        xmin, xmax = rec["xmin"], rec["xmax"]
        ymin, ymax = rec["ymin"], rec["ymax"]
        z0 = rec["bottom_z"]
        width, height = xmax - xmin, ymax - ymin
        if width <= 0 or height <= 0:
            continue
        center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, z0], dtype=np.float32)
        min_size = max(min(width, height), 1e-6)
        margin_list = [
            min(0.20 * min_size, 0.45 * MESH_SIZE),
            min(0.15 * min_size, 0.35 * MESH_SIZE),
            min(0.10 * min_size, 0.25 * MESH_SIZE),
            min(0.05 * min_size, 0.15 * MESH_SIZE),
            min(0.03 * min_size, 0.10 * MESH_SIZE),
        ]
        margin_list = [max(float(m), 1e-5) for m in margin_list]

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        for z_tol in z_tols:
            on_bottom = np.abs(z - z0) <= z_tol
            for margin in margin_list:
                if xmax - xmin <= 2.0 * margin or ymax - ymin <= 2.0 * margin:
                    continue
                mask = (
                    on_bottom &
                    (x > xmin + margin) & (x < xmax - margin) &
                    (y > ymin + margin) & (y < ymax - margin)
                )
                candidates = np.where(mask)[0]
                if candidates.size == 0:
                    continue
                dists = np.linalg.norm(points[candidates] - center.reshape(1, 3), axis=1)
                exc_idx = int(candidates[int(np.argmin(dists))])
                return exc_idx, int(pocket_id), float(margin)

    return None, None, None


def compute_adaptive_frequency_max(omega_k, zeta_k):
    """
    根据最高保留模态自适应确定 FRF 频率上限。

    默认覆盖第 N_MODES 阶模态峰，并在最高阶峰后留出安全余量：
        f_max = f_N + max(ratio*f_N, bw_mult*(2*zeta_N*f_N), min_margin)

    若设置环境变量 FREQ_MAX_HZ，则使用固定上限。
    """
    if FREQ_MAX_FIXED is not None:
        return float(FREQ_MAX_FIXED)

    freq_hz = np.asarray(omega_k, dtype=np.float64) / (2.0 * np.pi)
    zeta = np.asarray(zeta_k, dtype=np.float64)
    finite = np.isfinite(freq_hz) & np.isfinite(zeta) & (freq_hz > FREQ_MIN)
    if not np.any(finite):
        return max(FREQ_MIN + 100.0, 100.0)

    f_last = float(freq_hz[finite][-1])
    z_last = float(max(zeta[finite][-1], 0.0))
    bw_last = max(2.0 * z_last * f_last, FREQ_GRID_MIN_STEP_HZ * 20.0)
    margin = max(
        FREQ_MAX_MARGIN_RATIO * f_last,
        FREQ_MAX_MARGIN_BW_MULT * bw_last,
        FREQ_MAX_MARGIN_MIN_HZ,
    )
    fmax = f_last + margin
    fmax = min(float(fmax), float(FREQ_MAX_HARD_HZ))
    if fmax <= FREQ_MIN + 10.0 * FREQ_GRID_MIN_STEP_HZ:
        fmax = FREQ_MIN + 10.0 * FREQ_GRID_MIN_STEP_HZ
    return fmax


def _dedup_with_min_step(freqs, min_step, fmin, fmax):
    freqs = np.asarray(freqs, dtype=np.float64)
    freqs = freqs[np.isfinite(freqs)]
    freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]
    if freqs.size == 0:
        freqs = np.array([fmin, fmax], dtype=np.float64)
    freqs = np.sort(freqs)

    cleaned = [float(freqs[0])]
    for f in freqs[1:]:
        if f > cleaned[-1] + min_step:
            cleaned.append(float(f))
    return np.asarray(cleaned, dtype=np.float64)


def _fill_frequency_grid(freqs, target_n, min_step, fmin, fmax):
    freqs = _dedup_with_min_step(freqs, min_step, fmin, fmax)
    if freqs[0] > fmin + min_step:
        freqs = np.insert(freqs, 0, fmin)
    if freqs[-1] < fmax - min_step:
        freqs = np.append(freqs, fmax)

    while len(freqs) < target_n:
        gaps = np.diff(freqs)
        order = np.argsort(-gaps)
        inserted = False
        for j in order:
            lo, hi = freqs[j], freqs[j + 1]
            if hi - lo <= 2.2 * min_step:
                continue
            candidate = np.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
            if not (lo + min_step < candidate < hi - min_step):
                candidate = 0.5 * (lo + hi)
            freqs = np.insert(freqs, j + 1, candidate)
            inserted = True
            break
        if not inserted:
            raise RuntimeError("Cannot fill frequency grid with strict minimum spacing.")
    return np.sort(freqs)


def _trim_frequency_grid(freqs, target_n, protected_mask):
    """
    裁剪候选频率点。优先删除非保护点，并倾向于删除局部过密的点。
    若保护点数量本身超过 target_n，则保留端点和每个峰中心附近点后继续裁剪。
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    keep = np.ones(len(freqs), dtype=bool)
    n_remove = len(freqs) - target_n
    if n_remove <= 0:
        return freqs

    while n_remove > 0:
        kept_idx = np.where(keep)[0]
        removable = np.array([idx for idx in kept_idx if not protected_mask[idx]], dtype=np.int64)
        if len(removable) == 0:
            # 保护点过多时，不删端点，删除局部最密集的内部点。
            removable = kept_idx[1:-1]
        if len(removable) == 0:
            raise RuntimeError("Cannot trim frequency grid to target length.")

        costs = []
        for idx in removable:
            pos = np.where(kept_idx == idx)[0][0]
            left_gap = freqs[kept_idx[pos]] - freqs[kept_idx[pos - 1]] if pos > 0 else np.inf
            right_gap = freqs[kept_idx[pos + 1]] - freqs[kept_idx[pos]] if pos < len(kept_idx) - 1 else np.inf
            costs.append(min(left_gap, right_gap))
        keep[removable[int(np.argmin(costs))]] = False
        n_remove -= 1
    return freqs[keep]


def finalize_frequency_grid(freqs, protected_freqs, target_n, fmin, fmax):
    """保证 target_n 个 float32 频率点严格递增，且严格落在 [fmin, fmax] 内。

    修复点：旧版在末端/首端二次修正时，极少数情况下会先把末端压回 fmax，
    又因为首端小于 fmin 而整体平移，导致最后一个频点略大于 fmax，触发
    "frequency grid is out of configured frequency range"。这里改成前后双向约束，
    不再整体平移破坏上界。
    """
    min_step = float(FREQ_GRID_MIN_STEP_HZ)
    target_n = int(target_n)
    fmin = float(fmin)
    fmax = float(fmax)

    if target_n < 2:
        raise RuntimeError("target_n must be >= 2")
    if fmax <= fmin + (target_n - 1) * min_step:
        raise RuntimeError(
            f"frequency range too narrow for strict grid: fmin={fmin}, fmax={fmax}, "
            f"target_n={target_n}, min_step={min_step}"
        )

    freqs = _dedup_with_min_step(freqs, min_step, fmin, fmax)

    # 强制加入端点。
    if freqs.size == 0:
        freqs = np.array([fmin, fmax], dtype=np.float64)
    if abs(freqs[0] - fmin) > min_step:
        freqs = np.insert(freqs, 0, fmin)
    else:
        freqs[0] = fmin
    if abs(freqs[-1] - fmax) > min_step:
        freqs = np.append(freqs, fmax)
    else:
        freqs[-1] = fmax
    freqs = np.sort(freqs)

    protected_mask = np.zeros(len(freqs), dtype=bool)
    protected_freqs = np.asarray(protected_freqs, dtype=np.float64)
    protected_freqs = protected_freqs[np.isfinite(protected_freqs)]
    if protected_freqs.size:
        for pf in protected_freqs:
            if pf < fmin - 1e-9 or pf > fmax + 1e-9:
                continue
            j = int(np.argmin(np.abs(freqs - pf)))
            if abs(freqs[j] - pf) <= max(5.0 * min_step, 1e-5 * max(abs(pf), 1.0)):
                protected_mask[j] = True
    protected_mask[0] = True
    protected_mask[-1] = True

    if len(freqs) > target_n:
        freqs = _trim_frequency_grid(freqs, target_n, protected_mask)
    if len(freqs) < target_n:
        freqs = _fill_frequency_grid(freqs, target_n, min_step, fmin, fmax)

    freqs = np.sort(np.asarray(freqs, dtype=np.float64))

    # 裁剪后如果仍多一点，保留端点和排序后的前 target_n 个会丢 fmax；
    # 因此用均匀下标抽稀兜底，保证首尾端点存在。
    if len(freqs) > target_n:
        idx = np.linspace(0, len(freqs) - 1, target_n).round().astype(int)
        idx[0] = 0
        idx[-1] = len(freqs) - 1
        freqs = freqs[np.unique(idx)]
        while len(freqs) < target_n:
            freqs = _fill_frequency_grid(freqs, target_n, min_step, fmin, fmax)
        freqs = np.sort(freqs[:target_n])

    # 端点锁定 + 前向约束。
    freqs[0] = fmin
    freqs[-1] = fmax
    for i in range(1, len(freqs) - 1):
        if freqs[i] <= freqs[i - 1] + min_step:
            freqs[i] = freqs[i - 1] + min_step

    # 若前向约束把内部点推过上界，则从后往前压回；不再整体平移。
    freqs[-1] = fmax
    for i in range(len(freqs) - 2, 0, -1):
        if freqs[i] >= freqs[i + 1] - min_step:
            freqs[i] = freqs[i + 1] - min_step

    # 最后检查首端是否仍满足下界。正常情况下必定满足，因为范围远大于 min_step*N。
    if freqs[1] <= fmin + min_step:
        # 极端兜底：直接生成一套 log/linear 混合网格，保证不失败。
        base = np.logspace(np.log10(max(fmin, 1e-6)), np.log10(fmax), target_n)
        base[0] = fmin
        base[-1] = fmax
        freqs = base
        for i in range(1, len(freqs)):
            if freqs[i] <= freqs[i - 1] + min_step:
                freqs[i] = freqs[i - 1] + min_step
        if freqs[-1] > fmax:
            freqs = np.linspace(fmin, fmax, target_n)

    freqs = np.clip(freqs, fmin, fmax)
    freqs[0] = fmin
    freqs[-1] = fmax

    freqs32 = freqs.astype(np.float32)
    # float32 端点可能出现微小上溢，重新锁一下。
    freqs32[0] = np.float32(fmin)
    freqs32[-1] = np.float32(fmax)

    if len(freqs32) != target_n or not np.all(np.diff(freqs32.astype(np.float64)) > 0.0):
        raise RuntimeError("frequency grid is not strictly increasing after float32 conversion")
    # 容许 float32 的极小舍入误差。
    tol = max(1e-3, 1e-6 * max(abs(fmax), 1.0))
    if freqs32[0] < fmin - tol or freqs32[-1] > fmax + tol:
        raise RuntimeError(
            f"frequency grid is out of configured frequency range: "
            f"[{freqs32[0]}, {freqs32[-1]}] not in [{fmin}, {fmax}]"
        )
    return freqs32

def _mode_peak_offsets(mode_strength_ratio):
    """根据模态可见性分配峰附近采样点。"""
    if mode_strength_ratio >= 0.50:
        return np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    if mode_strength_ratio >= 0.15:
        return np.array([-2.5, -1.25, -0.5, 0.0, 0.5, 1.25, 2.5], dtype=np.float64)
    return np.array([-2.0, -0.75, 0.0, 0.75, 2.0], dtype=np.float64)


def make_frequency_grid(omega_k, zeta_k, phi_z=None, exc_idx=None):
    """
    生成自适应频率网格。

    改进点：
    1. 频率上限不再固定为 5000 Hz，而是默认覆盖第 N_MODES 阶模态峰。
    2. 所有落在 [FREQ_MIN, f_max] 内的模态都分配峰值附近点。
    3. 峰值附近点数按 Z-Z 可见性分配：强峰 9 点、中峰 7 点、弱峰 5 点。
    4. 剩余点用于全局 log 背景和峰间补点。
    """
    fmin = float(FREQ_MIN)
    fmax = compute_adaptive_frequency_max(omega_k, zeta_k)
    target_n = int(N_FREQS)
    min_step = float(FREQ_GRID_MIN_STEP_HZ)

    fk_all = np.asarray(omega_k, dtype=np.float64) / (2.0 * np.pi)
    zeta = np.asarray(zeta_k, dtype=np.float64)

    # 模态可见性：激励点 Z 向可激发程度 × 全局 Z 向 RMS 响应。
    if phi_z is not None and exc_idx is not None:
        phi_z_arr = np.asarray(phi_z, dtype=np.float64)
        phi_exc_z = np.abs(phi_z_arr[int(exc_idx), :])
        phi_rms_z = np.sqrt(np.mean(phi_z_arr ** 2, axis=0))
        visibility = phi_exc_z * phi_rms_z
        vmax = float(np.max(visibility)) if visibility.size else 0.0
        if vmax <= 0 or not np.isfinite(vmax):
            strength_ratio = np.ones_like(fk_all, dtype=np.float64)
        else:
            strength_ratio = visibility / (vmax + 1e-30)
    else:
        strength_ratio = np.ones_like(fk_all, dtype=np.float64)

    freqs_parts = [np.array([fmin, fmax], dtype=np.float64)]
    protected_parts = [np.array([fmin, fmax], dtype=np.float64)]

    # 全局背景点。数量随总频点数轻微缩放，避免峰点过少。
    n_background = min(max(24, target_n // 4), max(target_n // 2, 24))
    freqs_parts.append(np.logspace(np.log10(max(fmin, 1e-6)), np.log10(fmax), n_background))

    # 峰值附近自适应加密。
    peak_band_edges = []
    for idx_k, f_k in enumerate(fk_all):
        if not np.isfinite(f_k) or f_k < fmin or f_k > fmax:
            continue
        bw = max(2.0 * float(max(zeta[idx_k], 0.0)) * float(f_k), min_step * 20.0)
        offsets = _mode_peak_offsets(float(strength_ratio[idx_k]))
        peak_pts = f_k + offsets * bw
        peak_pts = np.clip(peak_pts, fmin, fmax)
        freqs_parts.append(peak_pts)
        protected_parts.append(peak_pts)
        peak_band_edges.extend([max(fmin, f_k - 3.5 * bw), min(fmax, f_k + 3.5 * bw)])

    # 峰间补点：按区间长度自适应分配，保留反共振/背景趋势。
    anchors = np.unique(np.clip(np.asarray([fmin, fmax] + peak_band_edges, dtype=np.float64), fmin, fmax))
    for lo, hi in zip(anchors[:-1], anchors[1:]):
        gap = hi - lo
        if gap <= max(20.0, 20.0 * min_step):
            continue
        # 较大间隙给更多点，但不让候选点爆炸。
        n_gap = int(np.clip(np.ceil(gap / max(fmax / 45.0, 50.0)), 1, 5))
        pts = np.logspace(np.log10(max(lo, fmin)), np.log10(hi), n_gap + 2, endpoint=True)[1:-1]
        if pts.size:
            freqs_parts.append(pts)

    freqs = np.concatenate(freqs_parts)
    protected_freqs = np.concatenate(protected_parts)
    return finalize_frequency_grid(freqs, protected_freqs, target_n, fmin, fmax)


def save_h5(name, idx_slice, arrays):
    idxs = list(idx_slice)
    path = os.path.join(OUT_DIR, name)
    with h5py.File(path, "w") as f:
        f.attrs["modal_normalization"] = "mass"
        f.attrs["mass_matrix"] = "lumped" if USE_LUMPED_MASS else "consistent"
        f.attrs["n_modes"] = N_MODES
        f.attrs["n_freqs"] = N_FREQS
        f.attrs["frequency_grid"] = "adaptive_to_highest_extracted_mode"
        f.attrs["freq_min_hz"] = FREQ_MIN
        f.attrs["freq_max_fixed_hz"] = np.nan if FREQ_MAX_FIXED is None else FREQ_MAX_FIXED
        f.attrs["freq_max_margin_ratio"] = FREQ_MAX_MARGIN_RATIO
        f.attrs["freq_max_margin_min_hz"] = FREQ_MAX_MARGIN_MIN_HZ
        f.attrs["freq_max_margin_bw_mult"] = FREQ_MAX_MARGIN_BW_MULT
        f.attrs["target_label"] = "modal_residue_z"
        f.attrs["frf_formula"] = "sum_r modal_residue_z_r/(omega_r^2-omega^2+2j*zeta_r*omega_r*omega)"
        f.attrs["frf_unit"] = "m/N when FRF_OUTPUT_SCALE=1.0"
        f.attrs["frf_output_scale"] = FRF_OUTPUT_SCALE
        f.attrs["min_relative_mode_gap"] = MIN_RELATIVE_MODE_GAP
        f.attrs["freq_grid_min_step_hz"] = FREQ_GRID_MIN_STEP_HZ
        f.attrs["sampling_strategy"] = "stratified clamp_level x coverage_level; balanced layout_type 5/6/7 inside each group"
        f.attrs["save_point_frf"] = int(SAVE_POINT_FRF)
        f.attrs["save_point_frf_qc_count"] = SAVE_POINT_FRF_QC_COUNT
        f.attrs["zeta_material"] = ZETA_MATERIAL
        f.attrs["zeta_joint_base"] = ZETA_JOINT_BASE
        f.attrs["zeta_joint_jitter"] = ZETA_JOINT_JITTER
        f.attrs["description"] = "Modal-residue dataset generated by mass-normalized modal analysis and Python modal superposition."
        for i, idx in enumerate(idxs):
            grp = f.create_group(f"sample_{i}")
            grp.attrs["modal_normalization"] = "mass"
            grp.attrs["mass_matrix"] = "lumped" if USE_LUMPED_MASS else "consistent"
            for key, values in arrays.items():
                compression = "gzip" if key in {
                    "edge_index", "edge_attr", "point_frf", "modal_phi_xyz", "modal_residue_z", "spring_k_xyz", "spring_c_xyz",
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

SAMPLE_PLAN = build_sample_plan(N_TRAIN, N_VAL, N_TEST)

print(f"配置: {N_SAMPLES}样本, train/val/test={N_TRAIN}/{N_VAL}/{N_TEST}, {N_MODES}阶模态, {N_FREQS}个频率点")
print(f"模态归一化: mass, 质量矩阵: {'lumped' if USE_LUMPED_MASS else 'consistent'}")
if MIN_RELATIVE_MODE_GAP > 0:
    print(f"样本过滤: 简单近频过滤 min_relative_gap >= {MIN_RELATIVE_MODE_GAP:.3f}；另过滤频率非递增、激励点无效、ANSYS求解失败等异常样本")
else:
    print("样本过滤: 近频过滤关闭；仅过滤频率非递增、激励点无效、ANSYS求解失败等异常样本")
print(f"激励点: 随机已加工凹槽的底面中心附近节点，排除凹槽边缘")
print(f"频率网格: {N_FREQS}点, adaptive-to-mode-{N_MODES}, visible-peak-first, float32后严格递增, min_step={FREQ_GRID_MIN_STEP_HZ:.4f} Hz")
if FREQ_MAX_FIXED is None:
    print(f"频率上限: 自适应，覆盖第{N_MODES}阶；margin=max({FREQ_MAX_MARGIN_RATIO:g}*fN, {FREQ_MAX_MARGIN_BW_MULT:g}*BW_N, {FREQ_MAX_MARGIN_MIN_HZ:g}Hz)")
else:
    print(f"频率上限: 固定 FREQ_MAX_HZ={FREQ_MAX_FIXED:g} Hz")
print(f"FRF输出缩放: FRF_OUTPUT_SCALE={FRF_OUTPUT_SCALE:g}，默认物理单位 m/N")
print(f"FRF保存: SAVE_POINT_FRF={int(SAVE_POINT_FRF)}, QC_COUNT={SAVE_POINT_FRF_QC_COUNT}")
print("采样: clamp_level×coverage_level 分层；layout 5/6/7 组内均衡；深度独立三角分布；边界 jitter 受控随机")
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
    "frequency_max_hz": [],
    "highest_mode_frequency_hz": [],
    "modal_omega": [],
    "modal_zeta": [],
    "modal_phi": [],
    "modal_phi_xyz": [],
    "modal_residue_z": [],
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
    "sample_id_global": [],
    "split_code": [],
    "clamp_level_code": [],
    "coverage_level_code": [],
    "layout_type": [],
    "grid_jitter": [],
    "target_depth_ratio": [],
    "current_progress": [],
    "finished_count": [],
    "current_cell": [],
    "cell_depth_ratio": [],
    "pocket_order": [],
    "cell_bounds": [],
    "spring_k_summary": [],
    "near_mode_summary": [],
    "removed_volume_ratio": [],
}

csv_path = os.path.join(OUT_DIR, "sample_log.csv")
csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
csv_writer = csv.writer(csv_file)
csv_header = [
    "sample", "split", "plan_index",
    "clamp_level", "coverage_level", "layout_type", "n_cols", "n_rows",
    "n_nodes", "n_edges", "n_pockets_to_machine", "n_pocket_scheme",
    "target_depth_ratio", "depth_range_%", "current_progress", "finished_count", "current_cell", "pocket_order",
    "grid_jitter", "removed_volume_ratio", "cut_region_fraction", "pocket_bottom_fraction", "cut_nodes/bottom_nodes",
    "exc_x_mm", "exc_y_mm", "exc_z_mm", "exc_pocket_id", "exc_margin_mm",
    "n_spring_areas", "n_spring_nodes",
    "K_corner_base", "K_side_base",
    "K_corner_mean", "K_corner_min", "K_corner_max",
    "K_side_mean", "K_side_min", "K_side_max",
    "spring_k_sum", "spring_k_mean_nonzero", "spring_k_max",
    "zeta_joint_base", "zeta_joint_mean", "zeta_joint_min", "zeta_joint_max",
    "modal_norm", "mass_matrix", "n_modes", "n_freqs",
    "freq_min_Hz", "freq_max_Hz", "df_min_Hz",
    "fN_Hz", "min_mode_gap_Hz", "min_relative_gap", "min_relative_gap_pair",
    "near_mode_flag", "very_near_mode_flag",
    "E_ratio", "rho_ratio", "frf_output_scale", "save_point_frf",
]
csv_header += [f"K_corner_{i+1}" for i in range(4)]
csv_header += [f"K_side_{i+1}" for i in range(3)]
csv_header += [f"cell_depth_{i+1:02d}" for i in range(7)]
for i in range(7):
    csv_header += [f"cell_{i+1:02d}_x0_mm", f"cell_{i+1:02d}_x1_mm", f"cell_{i+1:02d}_y0_mm", f"cell_{i+1:02d}_y1_mm"]
csv_header += [f"f{k+1:02d}_Hz" for k in range(N_MODES)]
csv_header += [f"zeta{k+1:02d}" for k in range(N_MODES)]
csv_writer.writerow(csv_header)

valid_samples = 0
attempt_count = 0
skip_excitation_count = 0
skip_close_mode_count = 0
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
        plan_rec = SAMPLE_PLAN[valid_samples]
        split_name = plan_rec["split"]
        clamp_level = plan_rec["clamp_level"]
        coverage_level = plan_rec["coverage_level"]
        layout_type = int(plan_rec["layout_type"])

        E = E_BASE * scaled_sobol[sobol_idx, 0]
        rho = RHO_BASE * scaled_sobol[sobol_idx, 1]
        L, W, H = L_BASE, W_BASE, H_BASE

        (
            K_corners, C_corners, K_sides, C_sides, zeta_joint_values,
            K_corner_base, K_side_base,
        ) = sample_clamp_parameters(clamp_level)

        mapdl.mp("EX", 1, E)
        mapdl.mp("PRXY", 1, PRXY_BASE)
        mapdl.mp("DENS", 1, rho)

        # ---------- 2. 几何与凹槽 ----------
        num_machined = layout_type
        pocket_cells, n_cols, n_rows = choose_layout(layout_type)
        grid_jitter = random.uniform(GRID_JITTER_RANGE[0], GRID_JITTER_RANGE[1])
        x_pockets, y_pockets = generate_region_division(n_cols, n_rows, L, W, jitter=grid_jitter)

        machining_state = sample_machining_state(num_machined, coverage_level)
        pocket_order = machining_state["pocket_order"]
        depth_by_cell = machining_state["depth_by_cell"]
        target_depth_ratio = float(machining_state["target_depth_ratio"])
        current_progress = float(machining_state["current_progress"])
        finished_count = int(machining_state["finished_count"])
        current_cell = int(machining_state["current_cell"])

        pockets_to_machine = [int(i) for i in range(num_machined) if float(depth_by_cell[i]) > 1e-6]
        n_pockets_to_machine = len(pockets_to_machine)

        # 记录所有 cell 的实际边界与深度：不存在的 cell 用 NaN，存在但未加工 depth=0。
        cell_depth_ratio = np.full(7, np.nan, dtype=np.float32)
        cell_bounds = np.full((7, 4), np.nan, dtype=np.float32)
        for cell_idx in range(num_machined):
            xmin_f, xmax_f, ymin_f, ymax_f = get_pocket_from_cells(x_pockets, y_pockets, pocket_cells[cell_idx], n_cols)
            cell_depth_ratio[cell_idx] = float(depth_by_cell[cell_idx])
            cell_bounds[cell_idx] = np.array([xmin_f * L, xmax_f * L, ymin_f * W, ymax_f * W], dtype=np.float32)

        pocket_order_arr = np.full(7, -1, dtype=np.int64)
        pocket_order_arr[:num_machined] = np.asarray(pocket_order, dtype=np.int64)

        mapdl.btol(0.0001)
        mapdl.block(0, L, 0, W, 0, H)
        wk_vol = int(mapdl.geometry.vnum[0])

        pocket_records = []
        pocket_depth_fracs = []
        bool_ok = True
        for pocket_idx in pockets_to_machine:
            depth_frac = float(depth_by_cell[pocket_idx])
            if depth_frac <= 1e-6:
                continue
            pocket_depth_fracs.append(depth_frac)
            pocket_zmin = H - depth_frac * H
            cells = pocket_cells[pocket_idx]
            xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(x_pockets, y_pockets, cells, n_cols)
            xmin_p, xmax_p = max(xmin_frac * L, 0.0), min(xmax_frac * L, L)
            ymin_p, ymax_p = max(ymin_frac * W, 0.0), min(ymax_frac * W, W)
            if xmax_p <= xmin_p or ymax_p <= ymin_p or pocket_zmin >= H:
                continue
            pocket_records.append({
                "pocket_idx": int(pocket_idx),
                "xmin": xmin_p, "xmax": xmax_p,
                "ymin": ymin_p, "ymax": ymax_p,
                "bottom_z": pocket_zmin,
                "depth_frac": depth_frac,
                "is_current": int(pocket_idx == current_cell),
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
        if not pocket_records:
            print("  跳过: 无有效凹槽记录")
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
        mode_gaps_hz = np.diff(freq_hz.astype(np.float64))
        min_mode_gap_hz = float(np.min(mode_gaps_hz))
        relative_gaps = mode_gaps_hz / np.maximum(freq_hz[:-1].astype(np.float64), 1e-12)
        min_relative_gap = float(np.min(relative_gaps))
        # 1-based pair index: 例如 4 表示第4阶和第5阶最近
        min_relative_gap_pair = int(np.argmin(relative_gaps) + 1)

        if MIN_RELATIVE_MODE_GAP > 0.0 and min_relative_gap < MIN_RELATIVE_MODE_GAP:
            skip_close_mode_count += 1
            print(
                f"  跳过: 近频模态 pair {min_relative_gap_pair}-{min_relative_gap_pair + 1}, "
                f"min_rel_gap={min_relative_gap:.4f} < {MIN_RELATIVE_MODE_GAP:.4f}, "
                f"df={mode_gaps_hz[min_relative_gap_pair - 1]:.2f}Hz"
            )
            continue

        phi_xyz = np.stack([phi_x, phi_y, phi_z], axis=-1).astype(np.float32)
        modal_mass = np.ones(N_MODES, dtype=np.float32) if USE_MASS_NORMALIZATION else np.full(N_MODES, np.nan, dtype=np.float32)
        modal_stiffness = (omega_k ** 2 * modal_mass).astype(np.float32)

        # ---------- 8. 激励点 ----------
        exc_idx, excitation_pocket_id, excitation_margin = select_bottom_center_excitation(
            all_node_coords,
            pocket_records,
        )
        if exc_idx is None:
            skip_excitation_count += 1
            print("  跳过: 没有找到合格的凹槽底面中心附近激励点")
            continue
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

        # ---------- 9.1 模态留数与 Python 模态叠加 FRF ----------
        # modal_residue_z[i, k] = phi_k,z(query_node_i) * phi_k,z(excitation_node)
        # 这是 Z向单位力输入、Z向位移输出 FRF 公式中的分子项 A_k(x)。
        modal_residue_z = (phi_z * phi_exc_xyz[np.newaxis, :, 2]).astype(np.float32)

        freqs = make_frequency_grid(omega_k, zeta_k, phi_z=phi_z, exc_idx=exc_idx)
        if len(freqs) != N_FREQS or not np.all(np.diff(freqs.astype(np.float64)) > 0.0):
            raise RuntimeError("make_frequency_grid returned invalid frequency grid")
        omega_q = (2.0 * np.pi * freqs).astype(np.float64)
        frf = np.zeros((n_nodes_total, len(freqs), 2), dtype=np.float32)
        for k in range(N_MODES):
            wk = float(omega_k[k])
            zk = float(zeta_k[k])
            ak_z = modal_residue_z[:, k].astype(np.float64)
            dw = wk ** 2 - omega_q ** 2
            gm = 2.0 * zk * wk * omega_q
            denom = dw ** 2 + gm ** 2 + 1e-30
            frf[:, :, 0] += np.outer(ak_z, FRF_OUTPUT_SCALE * dw / denom).astype(np.float32)
            frf[:, :, 1] += np.outer(ak_z, -FRF_OUTPUT_SCALE * gm / denom).astype(np.float32)

        # ---------- 10. 保存到内存 ----------
        arrays["points"].append(all_node_coords)
        arrays["edge_index"].append(edge_index)
        arrays["edge_attr"].append(edge_attr)
        if SAVE_POINT_FRF or valid_samples < SAVE_POINT_FRF_QC_COUNT:
            arrays["point_frf"].append(frf)
        else:
            arrays["point_frf"].append(np.zeros((0, 0, 2), dtype=np.float32))
        arrays["frequencies"].append(freqs)
        arrays["frequency_max_hz"].append(np.array(freqs[-1], dtype=np.float32))
        arrays["highest_mode_frequency_hz"].append(np.array(freq_hz[-1], dtype=np.float32))
        arrays["modal_omega"].append(omega_k)
        arrays["modal_zeta"].append(zeta_k)
        arrays["modal_phi"].append(phi_xyz)
        arrays["modal_phi_xyz"].append(phi_xyz)
        arrays["modal_residue_z"].append(modal_residue_z)
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

        # 受控随机元数据，后续可用于分层检查和误差诊断。
        split_code = {"train": 0, "val": 1, "test": 2}[split_name]
        spring_k_nonzero = spring_k_xyz[np.linalg.norm(spring_k_xyz, axis=1) > 0]
        spring_k_sum = float(np.sum(spring_k_xyz))
        spring_k_mean_nonzero = float(np.mean(spring_k_nonzero)) if spring_k_nonzero.size else 0.0
        spring_k_max = float(np.max(spring_k_xyz)) if spring_k_xyz.size else 0.0
        removed_volume_ratio = float(sum((rec["xmax"] - rec["xmin"]) * (rec["ymax"] - rec["ymin"]) * rec["depth_frac"] for rec in pocket_records) / (L * W))
        near_mode_flag = int(min_relative_gap < 0.04)
        very_near_mode_flag = int(min_relative_gap < 0.035)

        arrays["sample_id_global"].append(np.array(valid_samples, dtype=np.int64))
        arrays["split_code"].append(np.array(split_code, dtype=np.int64))
        arrays["clamp_level_code"].append(np.array(CLAMP_LEVEL_CODE[clamp_level], dtype=np.int64))
        arrays["coverage_level_code"].append(np.array(COVERAGE_LEVEL_CODE[coverage_level], dtype=np.int64))
        arrays["layout_type"].append(np.array(layout_type, dtype=np.int64))
        arrays["grid_jitter"].append(np.array(grid_jitter, dtype=np.float32))
        arrays["target_depth_ratio"].append(np.array(target_depth_ratio, dtype=np.float32))
        arrays["current_progress"].append(np.array(current_progress, dtype=np.float32))
        arrays["finished_count"].append(np.array(finished_count, dtype=np.int64))
        arrays["current_cell"].append(np.array(current_cell, dtype=np.int64))
        arrays["cell_depth_ratio"].append(cell_depth_ratio.astype(np.float32))
        arrays["pocket_order"].append(pocket_order_arr.astype(np.int64))
        arrays["cell_bounds"].append(cell_bounds.astype(np.float32))
        arrays["spring_k_summary"].append(np.array([spring_k_sum, spring_k_mean_nonzero, spring_k_max, len(spring_info)], dtype=np.float32))
        arrays["near_mode_summary"].append(np.array([near_mode_flag, very_near_mode_flag, min_relative_gap_pair, min_relative_gap], dtype=np.float32))
        arrays["removed_volume_ratio"].append(np.array(removed_volume_ratio, dtype=np.float32))

        depth_min = min(pocket_depth_fracs) * 100 if pocket_depth_fracs else 0.0
        depth_max = max(pocket_depth_fracs) * 100 if pocket_depth_fracs else 0.0
        n_cut = len(pocket_cut_indices)
        n_bottom = len(pocket_bottom_any_indices)
        min_df = float(np.min(np.diff(freqs.astype(np.float64))))
        freq_short = ", ".join(f"{v:.0f}" for v in freq_hz[:min(4, N_MODES)])
        zeta_short = ", ".join(f"{v:.4f}" for v in zeta_k[:min(4, N_MODES)])
        cut_region_fraction = float(np.mean(cut_region_mask.astype(bool)))
        pocket_bottom_fraction = float(np.mean(pocket_bottom_mask.astype(bool)))
        zeta_joint_mean = float(np.mean(zeta_joint_values))
        zeta_joint_min = float(np.min(zeta_joint_values))
        zeta_joint_max = float(np.max(zeta_joint_values))

        print(
            f"[{split_name}] clamp={clamp_level}, coverage={coverage_level}, layout={layout_type}, "
            f"N={n_nodes_total}, E={edge_index.shape[1]}, 加工{n_pockets_to_machine}/{num_machined}, "
            f"target_depth={target_depth_ratio*100:.0f}%, 实际深度{depth_min:.0f}~{depth_max:.0f}%, "
            f"progress={current_progress:.2f}, cut/bottom={n_cut}/{n_bottom}, "
            f"exc_pocket={excitation_pocket_id}, margin={excitation_margin*1000:.2f}mm, "
            f"modes={N_MODES}, f1..={freq_short}Hz, fN={freq_hz[-1]:.1f}Hz, "
            f"f_grid=[{freqs[0]:.1f},{freqs[-1]:.1f}]Hz/{N_FREQS}点, df_min={min_df:.4f}Hz, "
            f"min_gap={min_mode_gap_hz:.1f}Hz, min_rel_gap={min_relative_gap:.3f}"
            f"(pair {min_relative_gap_pair}-{min_relative_gap_pair+1}), "
            f"zeta1..={zeta_short}"
        )

        pocket_order_str = " ".join(str(int(x)) for x in pocket_order_arr if int(x) >= 0)
        csv_row = [
            valid_samples + 1, split_name, valid_samples,
            clamp_level, coverage_level, layout_type, n_cols, n_rows,
            n_nodes_total, edge_index.shape[1], n_pockets_to_machine, num_machined,
            f"{target_depth_ratio:.6f}", f"{depth_min:.1f}~{depth_max:.1f}", f"{current_progress:.6f}", finished_count, current_cell, pocket_order_str,
            f"{grid_jitter:.6f}", f"{removed_volume_ratio:.6f}", f"{cut_region_fraction:.6f}", f"{pocket_bottom_fraction:.6f}", f"{n_cut}/{n_bottom}",
            f"{exc_actual[0]*1000:.2f}", f"{exc_actual[1]*1000:.2f}", f"{exc_actual[2]*1000:.2f}",
            excitation_pocket_id, f"{excitation_margin*1000:.3f}",
            len(all_clamp_areas), len(spring_info),
            f"{K_corner_base:.6g}", f"{K_side_base:.6g}",
            f"{np.mean(K_corners):.6g}", f"{np.min(K_corners):.6g}", f"{np.max(K_corners):.6g}",
            f"{np.mean(K_sides):.6g}", f"{np.min(K_sides):.6g}", f"{np.max(K_sides):.6g}",
            f"{spring_k_sum:.6g}", f"{spring_k_mean_nonzero:.6g}", f"{spring_k_max:.6g}",
            f"{ZETA_JOINT_BASE:.6f}", f"{zeta_joint_mean:.6f}", f"{zeta_joint_min:.6f}", f"{zeta_joint_max:.6f}",
            "mass", "lumped" if USE_LUMPED_MASS else "consistent", N_MODES, N_FREQS,
            f"{freqs[0]:.2f}", f"{freqs[-1]:.2f}", f"{min_df:.5f}",
            f"{freq_hz[-1]:.2f}", f"{min_mode_gap_hz:.2f}", f"{min_relative_gap:.6f}", min_relative_gap_pair,
            near_mode_flag, very_near_mode_flag,
            f"{E/E_BASE:.4f}", f"{rho/RHO_BASE:.4f}", f"{FRF_OUTPUT_SCALE:g}", int(SAVE_POINT_FRF or valid_samples < SAVE_POINT_FRF_QC_COUNT),
        ]
        csv_row += [f"{v:.6g}" for v in K_corners]
        csv_row += [f"{v:.6g}" for v in K_sides]
        csv_row += ["" if np.isnan(v) else f"{float(v):.6f}" for v in cell_depth_ratio]
        for bounds in cell_bounds:
            csv_row += ["" if np.isnan(v) else f"{float(v)*1000.0:.3f}" for v in bounds]
        csv_row += [f"{v:.2f}" for v in freq_hz]
        csv_row += [f"{v:.6f}" for v in zeta_k]
        csv_writer.writerow(csv_row)
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
                    f"Sample {valid_samples + 1}: exc_pocket={excitation_pocket_id}, fN={freq_hz[-1]:.1f}Hz, fmax={freqs[-1]:.1f}Hz",
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
print(
    f"\n生成完成, 有效样本={N_SAMPLES}, 总尝试={attempt_count}, "
    f"激励点过滤={skip_excitation_count}, 近频过滤={skip_close_mode_count}, 耗时={elapsed:.0f}s"
)
print(f"CSV日志: {csv_path}")

print("\n保存 HDF5...")
train_indices = [i for i, rec in enumerate(SAMPLE_PLAN) if rec["split"] == "train"]
val_indices = [i for i, rec in enumerate(SAMPLE_PLAN) if rec["split"] == "val"]
test_indices = [i for i, rec in enumerate(SAMPLE_PLAN) if rec["split"] == "test"]
assert len(train_indices) == N_TRAIN and len(val_indices) == N_VAL and len(test_indices) == N_TEST
save_h5("train.h5", train_indices, arrays)
save_h5("val.h5", val_indices, arrays)
save_h5("test.h5", test_indices, arrays)

# 简单 FRF 可视化
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords0 = arrays["points"][0]
    frf0 = arrays["point_frf"][0]
    if frf0.size == 0:
        raise RuntimeError("sample_000 未保存 point_frf；如需 FRF 图，设置 SAVE_POINT_FRF=1 或 SAVE_POINT_FRF_QC_COUNT>=1")
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
    fig.suptitle("Mass-normalized modal-residue FRF dataset check")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "sample_000_frf.png"), dpi=150)
    plt.close()
    print(f"可视化保存: {VIZ_DIR}")
except Exception as exc:
    print(f"FRF可视化失败: {exc}")
