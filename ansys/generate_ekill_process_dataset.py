"""
ANSYS EKILL 过程数据集生成 — 固定拓扑 + EKILL 材料去除。

与 boolean 生成器（generate_3d_test.py）的区别：
    - boolean 生成器：每个样本独立建模（block + vsbv 挖槽 → vmesh），拓扑不固定。
    - ekill 生成器：一次性 vmesh 完整长方体 → 提取固定拓扑 → 每样本 EKILL 去除单元。

优势：
    1. 固定拓扑保证节点编号和单元连接关系一致，便于"过程"数据对齐。
    2. EKILL 模拟铣削材料逐层去除，更贴近真实加工过程。
    3. element_active_flag / node_active_flag 可直接作为模型输入特征。

严禁事项：
    - 不使用 vsbv 重新挖槽。
    - 不允许每个样本重新 vmesh（基础 points/elements 必须一致）。
    - 随机凹槽边界导致的 active mask 不同是允许的。

方向可配置的位移频响函数：
    H_ab(x, x_f, ω) = Σ_k φ_k^a(x)·φ_k^b(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·Ω)
    默认 H_YY。
"""
import random
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
from ansys.mapdl.core import launch_mapdl
from scipy.stats import qmc
import numpy as np
import h5py
import time
import csv

# ============ 全局随机种子 ============
SEED = 2
np.random.seed(SEED)
random.seed(SEED)

# ============ 配置 ============
N_SAMPLES = 300
N_TRAIN = 150
N_VAL = 50
N_TEST = 50
N_MODES = 3
N_FREQS = 60
FREQ_MIN, FREQ_MAX = 1.0, 5000.0
MESH_SIZE = 0.006
ZETA_MATERIAL = 0.002
AMPLITUDE_SCALE = 500000.0

# 输出目录（与 boolean 数据分开）
OUT_DIR = os.path.join(os.path.dirname(__file__), "data_ekill")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "mesh_viz_ekill")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# ============ 方向配置 ============
RESPONSE_DIRECTION = "Y"
FORCE_DIRECTION = "Y"
_DIR_MAP = {"X": 0, "Y": 1, "Z": 2}
RESPONSE_DIR_INDEX = _DIR_MAP[RESPONSE_DIRECTION]
FORCE_DIR_INDEX = _DIR_MAP[FORCE_DIRECTION]

# ============ 频率网格配置 ============
FREQ_GRID_MODE = "both"      # "fixed" / "adaptive" / "both"
N_FREQS_FIXED = 128

# ============ 固定工件尺寸 ============
E_BASE, RHO_BASE, PRXY_BASE = 71.7e9, 2810.0, 0.33
L_BASE, W_BASE, H_BASE = 0.160, 0.060, 0.010

# ============ 材料参数随机范围 ============
E_RANGE = (0.95, 1.05)
RHO_RANGE = (0.97, 1.03)

# ============ 凹槽参数 ============
GRID_JITTER = 0.15
GAP_ABS = 0.006
BORDER_ABS = 0.006
POCKET_DEPTH_RANGE = (0.30, 0.60)

# ============ 弹簧阻尼器参数 ============
K_CORNER_RANGE = (5e6, 1e8)
K_SIDE_RANGE = (1e6, 3e7)
ZETA_JOINT_RANGE = (0.005, 0.05)
M_REF = 0.01

# ============ 表面标记名称 ============
SURFACE_FLAG_NAMES = [
    'free_surface_flag', 'top_surface_flag', 'workpiece_bottom_flag',
    'external_side_surface_flag', 'pocket_sidewall_flag'
]
TRANSOLVER_FEATURE_NAMES = [
    'x_norm', 'y_norm', 'z_norm',
    'E_over_E0', 'rho_over_rho0', 'PRXY',
    'pocket_active_flag', 'pocket_bottom_flag', 'cutting_band_flag',
    'pocket_id_norm', 'pocket_depth_frac', 'remaining_thickness_ratio',
    'distance_to_pocket_edge_norm',
    'fixture_corner_flag', 'fixture_side_flag',
    'log10_Kx', 'log10_Ky', 'log10_Kz',
    'log10_Cx', 'log10_Cy', 'log10_Cz',
    'distance_to_excitation_norm', 'excitation_flag',
    *SURFACE_FLAG_NAMES
]
POINT_FEATURE_NAMES = ['E_over_E0', 'PRXY', 'rho_over_rho0', 'is_fixed', 'log10_K', 'log10_C', 'Z_over_H']

# ============ 凹槽区域定义（复用 boolean 生成器的布局） ============
POCKET_CELLS_5 = [
    [1, 2, 5, 6, 9, 10], [3, 4], [8, 12], [7], [11],
]
POCKET_CELLS_6 = [
    [1, 5, 9], [2], [3, 7], [4, 8, 12], [6], [10, 11],
]
POCKET_CELLS_7 = [
    [1, 6, 11], [4, 9, 14], [5, 10], [15], [2, 3], [7, 8], [12, 13],
]


def safe_log10(values):
    """对正值取 log10；0 表示无弹簧，保留为 -1。"""
    out = np.full_like(values, -1.0, dtype=np.float32)
    mask = values > 0.0
    out[mask] = np.log10(values[mask]).astype(np.float32)
    return out


def build_fixed_frequency_grid(freq_min, freq_max, n_freqs, mode="hybrid"):
    """构建统一固定频率网格（hybrid: 40% 线性 + 60% 对数）。"""
    if mode == "linspace":
        return np.linspace(freq_min, freq_max, n_freqs, dtype=np.float32)
    if mode == "log":
        return np.logspace(np.log10(max(freq_min, 0.1)), np.log10(freq_max),
                           n_freqs, dtype=np.float32)
    split_hz = 1000.0
    n_low = max(2, int(n_freqs * 0.4))
    n_high = n_freqs - n_low
    low_part = np.linspace(freq_min, split_hz, n_low, endpoint=False, dtype=np.float32)
    high_part = np.logspace(np.log10(split_hz), np.log10(freq_max),
                            n_high, dtype=np.float32)
    grid = np.concatenate([low_part, high_part])
    return np.unique(np.sort(grid)).astype(np.float32)


def generate_region_division(n_cols, n_rows, L, W, jitter=GRID_JITTER,
                              gap=GAP_ABS, border=BORDER_ABS):
    """生成区域划分（复用 boolean 生成器逻辑）。"""
    n_gaps_x = n_cols - 1
    n_gaps_y = n_rows - 1
    available_x = L - 2 * border - n_gaps_x * gap
    available_y = W - 2 * border - n_gaps_y * gap

    weights_x = np.array([1.0 + np.random.uniform(-jitter, jitter) for _ in range(n_cols)])
    weights_x = weights_x / weights_x.sum() * available_x
    weights_y = np.array([1.0 + np.random.uniform(-jitter, jitter) for _ in range(n_rows)])
    weights_y = weights_y / weights_y.sum() * available_y

    x_pockets, y_pockets = [], []
    current_x = border
    for i in range(n_cols):
        col_w = weights_x[i]
        x_pockets.append((current_x / L, (current_x + col_w) / L))
        current_x += col_w + gap
    current_y = border
    for i in range(n_rows):
        row_h = weights_y[i]
        y_pockets.append((current_y / W, (current_y + row_h) / W))
        current_y += row_h + gap
    return x_pockets, y_pockets


def get_pocket_from_cells(x_pockets, y_pockets, cell_indices, n_cols):
    """根据区域单元索引计算凹槽边界（合并多个单元）。"""
    rows = [(idx - 1) // n_cols for idx in cell_indices]
    cols = [(idx - 1) % n_cols for idx in cell_indices]
    xmin = min(x_pockets[c][0] for c in cols)
    xmax = max(x_pockets[c][1] for c in cols)
    ymin = min(y_pockets[r][0] for r in rows)
    ymax = max(y_pockets[r][1] for r in rows)
    return (xmin, xmax, ymin, ymax)


def extract_solid_element_data_ekill(mapdl, all_node_ids, all_node_coords):
    """提取 SOLID187 网格连接关系（ekill 文件内联版本）。

    失败时返回空数组，不影响原数据生成流程。
    """
    try:
        grid = mapdl.mesh._grid
        raw_cells = np.asarray(grid.cells, dtype=np.int64)
        celltypes = np.asarray(getattr(grid, 'celltypes', np.zeros(0, dtype=np.int32)), dtype=np.int32)
        conn = []
        ptr = 0
        while ptr < len(raw_cells):
            n = int(raw_cells[ptr])
            ids = raw_cells[ptr + 1: ptr + 1 + n]
            ptr += n + 1
            if n >= 4 and np.all(ids >= 0) and np.all(ids < len(all_node_ids)):
                conn.append(ids.astype(np.int64))
        if not conn:
            return (np.zeros((0, 0), dtype=np.int64),
                    np.zeros((0, 0), dtype=np.int64),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        max_nodes = max(len(c) for c in conn)
        elem_node_indices = -np.ones((len(conn), max_nodes), dtype=np.int64)
        elem_node_ids = -np.ones((len(conn), max_nodes), dtype=np.int64)
        elem_centers = np.zeros((len(conn), 3), dtype=np.float32)
        for ei, ids in enumerate(conn):
            elem_node_indices[ei, :len(ids)] = ids
            elem_node_ids[ei, :len(ids)] = all_node_ids[ids]
            elem_centers[ei] = all_node_coords[ids].mean(axis=0)
        if len(celltypes) >= len(conn):
            elem_types = celltypes[:len(conn)].astype(np.int32)
        else:
            elem_types = np.full((len(conn),), -1, dtype=np.int32)
        try:
            elem_ids = np.asarray(mapdl.mesh.enum, dtype=np.int64)
            if len(elem_ids) != len(conn):
                elem_ids = np.arange(1, len(conn) + 1, dtype=np.int64)
        except Exception:
            elem_ids = np.arange(1, len(conn) + 1, dtype=np.int64)
        return elem_node_indices, elem_node_ids, elem_types, elem_centers, elem_ids
    except Exception as e:
        print(f"  警告: 单元连接关系提取失败: {e}")
        return (np.zeros((0, 0), dtype=np.int64),
                np.zeros((0, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))


def kill_elements_by_ids(mapdl, element_ids):
    """通过 APDL EKILL 命令杀死指定单元。

    原理：ESEL 按单元号选中 → EKILL 杀死 → ALLSEL 恢复全选。
    EKILL 后的单元刚度矩阵乘以 1e-6，对整体刚度贡献可忽略。

    注意：每样本 EKILL 后必须 ALLSEL，否则后续模态求解选择集不正确。
    """
    if len(element_ids) == 0:
        return
    # 批量选中所有待杀单元
    for eid in element_ids:
        try:
            mapdl.esel("S", "ELEM", "", int(eid))
            mapdl.ekill("ALL")
        except Exception as e:
            print(f"    警告: EKILL 单元 {eid} 失败: {e}")
    mapdl.allsel()


def determine_removed_elements(element_centers, L, W, H,
                                pocket_cells, x_pockets, y_pockets,
                                pockets_to_machine, pocket_depth_fracs, n_cols):
    """根据凹槽几何判断哪些 element_center 落入被去除区域。

    返回:
        removed_mask: (Ne,) bool 数组，True 表示该单元被去除。
        removed_volume_ratio: 去除体积占比估算值。
    """
    Ne = element_centers.shape[0]
    removed = np.zeros(Ne, dtype=bool)

    for pocket_i, pocket_idx in enumerate(pockets_to_machine):
        if pocket_i >= len(pocket_depth_fracs):
            continue
        pocket_depth_k = pocket_depth_fracs[pocket_i] * H
        pocket_bottom_z = H - pocket_depth_k

        cells = pocket_cells[pocket_idx]
        xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(
            x_pockets, y_pockets, cells, n_cols)
        xmin_p, xmax_p = xmin_frac * L, xmax_frac * L
        ymin_p, ymax_p = ymin_frac * W, ymax_frac * W

        x = element_centers[:, 0]
        y = element_centers[:, 1]
        z = element_centers[:, 2]

        xy_in = (x >= xmin_p) & (x <= xmax_p) & (y >= ymin_p) & (y <= ymax_p)
        z_removed = z >= pocket_bottom_z
        removed |= xy_in & z_removed

    total_vol = L * W * H
    removed_vol = 0.0
    for pocket_i, pocket_idx in enumerate(pockets_to_machine):
        cells = pocket_cells[pocket_idx]
        xf_min, xf_max, yf_min, yf_max = get_pocket_from_cells(
            x_pockets, y_pockets, cells, n_cols)
        pk_depth = pocket_depth_fracs[pocket_i] * H
        removed_vol += (xf_max - xf_min) * L * (yf_max - yf_min) * W * pk_depth
    removed_volume_ratio = min(removed_vol / max(total_vol, 1e-12), 1.0)

    return removed, removed_volume_ratio


def compute_node_active_from_elements(element_node_indices, element_active_flag, n_nodes):
    """从单元活性推断节点活性：某节点只要连接至少一个 active 单元即为 active。"""
    node_active = np.zeros(n_nodes, dtype=np.int8)
    for ei, elem_nodes in enumerate(element_node_indices):
        if element_active_flag[ei]:
            for ni in elem_nodes:
                if ni >= 0 and ni < n_nodes:
                    node_active[ni] = 1
    return node_active


# ======================================================================
# 主流程
# ======================================================================
print(">>> 正在生成 Sobol 低偏差序列 (材料参数)...")
SOBOL_BUFFER = 50
sampler = qmc.Sobol(d=2, scramble=True, seed=SEED)
sobol_samples = sampler.random(n=N_SAMPLES + SOBOL_BUFFER)
l_bounds = [E_RANGE[0], RHO_RANGE[0]]
u_bounds = [E_RANGE[1], RHO_RANGE[1]]
scaled_sobol = qmc.scale(sobol_samples, l_bounds, u_bounds)

print(f"配置: {N_SAMPLES}样本, EKILL 过程数据生成")
print(f"工件: {L_BASE*1000:.0f}×{W_BASE*1000:.0f}×{H_BASE*1000:.0f}mm (固定拓扑)")
print(f"方向: 响应={RESPONSE_DIRECTION}, 激励={FORCE_DIRECTION}")

print("\n>>> 正在连接 ANSYS 求解器...")
mapdl = launch_mapdl(override=True)
print(f">>> 连接成功! 版本: {mapdl.version}\n")

# ======================================================================
# 第一步：一次性创建完整工件并划分网格（固定拓扑）
# ======================================================================
print(">>> 创建完整工件并一次性划分网格（固定拓扑）...")
mapdl.clear()
mapdl.prep7()
mapdl.btol(0.0001)

# 材料定义（使用基准值，后续每样本重新设置）
mapdl.mp("EX", 1, E_BASE)
mapdl.mp("PRXY", 1, PRXY_BASE)
mapdl.mp("DENS", 1, RHO_BASE)

# 创建完整长方体（无凹槽）
mapdl.block(0, L_BASE, 0, W_BASE, 0, H_BASE)

# 一次性划分网格
mapdl.et(1, "SOLID187")
mapdl.mshape(1, "3D")
mapdl.mshkey(0)
mapdl.esize(MESH_SIZE)
try:
    mapdl.vmesh("ALL")
except Exception:
    mapdl.smrtsize(4)
    mapdl.vmesh("ALL")

# 提取固定拓扑信息
all_node_ids_base = mapdl.mesh.nnum
all_node_coords_base = np.array(mapdl.mesh.nodes, dtype=np.float32)
n_nodes_total = len(all_node_ids_base)
node_id_to_idx_base = {int(nid): idx for idx, nid in enumerate(all_node_ids_base)}
max_node_id_base = int(all_node_ids_base.max())

print(f"  固定拓扑: {n_nodes_total} 节点, 节点ID范围 1~{max_node_id_base}")

# 提取 SOLID187 单元连接关系（固定拓扑，使用内联版本避免触发 generate_3d_test 的 MAPDL 启动）
element_node_indices_base, element_node_ids_base, element_types_base, \
    element_centers_base, element_ids_base = extract_solid_element_data_ekill(
        mapdl, all_node_ids_base, all_node_coords_base)
n_elements = element_node_indices_base.shape[0]
print(f"  单元数: {n_elements}")

mapdl.save("ekill_base_mesh", "db")
print("  基础网格已保存: ekill_base_mesh.db\n")

# ======================================================================
# 循环生成样本
# ======================================================================
# 预分配
all_points, all_frf, all_freqs = [], [], []
all_omega, all_zeta, all_phi, all_phi_exc = [], [], [], []
all_transolver_features, all_phi_xyz, all_phi_exc_xyz = [], [], []
all_zeta_boundary_components_xyz = []
all_boundary_k_xyz, all_boundary_c_xyz, all_fixture_type = [], [], []
all_surface_flags = []
all_element_node_indices, all_element_node_ids = [], []
all_element_types, all_element_centers, all_element_ids = [], [], []
all_excitation_index, all_excitation_node_id, all_excitation_point = [], [], []
# 方向 / 刀触点 / 过程字段
all_frf_adaptive, all_freqs_adaptive = [], []
all_tool_position, all_contact_node_index = [], []
all_force_direction_vector, all_response_direction_vector = [], []
all_active_pocket_id, all_process_step, all_removed_volume_ratio = [], [], []
# EKILL 特有字段
all_element_active_flag, all_node_active_flag = [], []
all_removed_element_flag = []

# CSV 日志
csv_path = os.path.join(OUT_DIR, "sample_log_ekill.csv")
csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    '样本编号', '节点总数', '单元总数', '去除单元数', '去除体积比',
    '加工凹槽数', '凹槽方案总数',
    '激励点X(mm)', '激励点Y(mm)', '激励点Z(mm)',
    '阻尼比ζ₁', '阻尼比ζ₂', '阻尼比ζ₃',
    '固有频率f₁(Hz)', '固有频率f₂(Hz)', '固有频率f₃(Hz)',
    '弹性模量比', '密度比',
])

t0 = time.time()
valid_samples = 0
attempt_count = 0

while valid_samples < N_SAMPLES:
    attempt_count += 1
    sobol_idx = (attempt_count - 1) % len(scaled_sobol)
    print(f"[有效样本 {valid_samples+1}/{N_SAMPLES}] (尝试第{attempt_count}次)", end=" ", flush=True)

    try:
        # 恢复基础网格
        mapdl.clear()
        mapdl.resume("ekill_base_mesh", "db")
        mapdl.prep7()
    except Exception:
        print("(reconnect)", end=" ", flush=True)
        mapdl.exit()
        time.sleep(2)
        mapdl = launch_mapdl(override=True)
        mapdl.resume("ekill_base_mesh", "db")
        mapdl.prep7()

    # 1. 采样材料参数
    E = E_BASE * scaled_sobol[sobol_idx, 0]
    rho = RHO_BASE * scaled_sobol[sobol_idx, 1]
    L, W, H = L_BASE, W_BASE, H_BASE

    # 重新设置材料（网格已存在，仅更新材料属性）
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, PRXY_BASE)
    mapdl.mp("DENS", 1, rho)

    # 2. 随机弹簧参数
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

    # 3. 凹槽方案选择
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

    pocket_depth_fracs = []
    for _ in pockets_to_machine:
        pocket_depth_fracs.append(random.uniform(*POCKET_DEPTH_RANGE))

    # 4. 判断落入去除区域的单元（纯 numpy，不涉及 ANSYS）
    removed_mask, removed_volume_ratio = determine_removed_elements(
        element_centers_base, L, W, H,
        pocket_cells, x_pockets, y_pockets,
        pockets_to_machine, pocket_depth_fracs, n_cols)

    removed_elem_ids = element_ids_base[removed_mask]

    element_active_flag = (~removed_mask).astype(np.int8)
    removed_element_flag = removed_mask.astype(np.int8)

    # 5. 计算节点活性（纯 numpy）
    node_active_flag = compute_node_active_from_elements(
        element_node_indices_base, element_active_flag, n_nodes_total)

    # 6. 识别凹槽底面节点并选激励点
    pocket_cut_indices, pocket_bottom_any_indices = [], []
    mapdl.allsel()
    for pocket_i, pocket_idx in enumerate(pockets_to_machine):
        pocket_depth_k = pocket_depth_fracs[pocket_i] * H
        pocket_bottom_z = H - pocket_depth_k
        cells = pocket_cells[pocket_idx]
        xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(
            x_pockets, y_pockets, cells, n_cols)
        xmin_p, xmax_p = xmin_frac * L, xmax_frac * L
        ymin_p, ymax_p = ymin_frac * W, ymax_frac * W

        mapdl.nsel("S", "LOC", "Z", pocket_bottom_z, pocket_bottom_z + 1e-6)
        margin = 1e-4
        mapdl.nsel("R", "LOC", "X", xmin_p + margin, xmax_p - margin)
        mapdl.nsel("R", "LOC", "Y", ymin_p + margin, ymax_p - margin)
        for nid in mapdl.mesh.nnum:
            nid_int = int(nid)
            if nid_int not in node_id_to_idx_base:
                continue
            idx = node_id_to_idx_base[nid_int]
            if idx not in pocket_bottom_any_indices:
                pocket_bottom_any_indices.append(idx)
                x, y = all_node_coords_base[idx, 0], all_node_coords_base[idx, 1]
                dist_to_wall = min(x - xmin_p, xmax_p - x, y - ymin_p, ymax_p - y)
                tool_r = MESH_SIZE / 2
                cut_band = MESH_SIZE * 0.6
                if abs(dist_to_wall - tool_r) < cut_band:
                    pocket_cut_indices.append(idx)
    mapdl.allsel()

    # 选择激励点
    if len(pocket_cut_indices) > 0:
        cut_coords = all_node_coords_base[pocket_cut_indices]
        center = cut_coords.mean(axis=0)
        dists = np.linalg.norm(cut_coords - center, axis=1)
        exc_idx = pocket_cut_indices[int(np.argmin(dists))]
    elif len(pocket_bottom_any_indices) > 0:
        any_coords = all_node_coords_base[pocket_bottom_any_indices]
        center = any_coords.mean(axis=0)
        dists = np.linalg.norm(any_coords - center, axis=1)
        exc_idx = pocket_bottom_any_indices[int(np.argmin(dists))]
    else:
        exc_idx = np.random.randint(0, n_nodes_total)

    exc_actual = all_node_coords_base[exc_idx]
    exc_node_id = int(all_node_ids_base[exc_idx])

    # 7. 施加 COMBIN14 弹簧装夹
    mapdl.et(2, "COMBIN14"); mapdl.keyopt(2, 2, 1)
    mapdl.et(3, "COMBIN14"); mapdl.keyopt(3, 2, 2)
    mapdl.et(4, "COMBIN14"); mapdl.keyopt(4, 2, 3)

    clamp_len = 0.010
    all_clamp_areas = [
        (0, clamp_len, 0, 1e-4),
        (L - clamp_len, L, 0, 1e-4),
        (0, clamp_len, W - 1e-4, W),
        (L - clamp_len, L, W - 1e-4, W),
    ]
    CORNER_EXCL = clamp_len + H / 2
    x_min, x_max = CORNER_EXCL, L - CORNER_EXCL
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
            all_clamp_areas.append((x_c - H/2, x_c + H/2, cy - 1e-4, cy + 1e-4))

    # 初始化边界数组
    boundary_k_xyz = np.zeros((n_nodes_total, 3), dtype=np.float32)
    boundary_c_xyz = np.zeros((n_nodes_total, 3), dtype=np.float32)
    fixture_type = np.zeros(n_nodes_total, dtype=np.int8)
    spring_info = []
    spring_node_set = set()

    current_max_node_id = max_node_id_base
    real_const_num = 2

    for idx_area, (xmin, xmax, ymin, ymax) in enumerate(all_clamp_areas):
        mapdl.nsel("S", "LOC", "X", xmin, xmax)
        mapdl.nsel("R", "LOC", "Y", ymin, ymax)
        mapdl.nsel("R", "LOC", "Z", 0, H)
        n_selected = mapdl.mesh.n_node
        if n_selected > 0:
            clamp_nodes = list(mapdl.mesh.nnum)
            is_corner = (idx_area < 4)
            K_this = K_corners[idx_area] if is_corner else K_sides[idx_area - 4]
            C_this = C_corners[idx_area] if is_corner else C_sides[idx_area - 4]
            K_each = K_this / n_selected
            C_each = C_this / n_selected

            mapdl.r(real_const_num, K_each, 0.0)
            for n1 in clamp_nodes:
                n1_int = int(n1)
                if n1_int in node_id_to_idx_base and n1_int not in spring_node_set:
                    spring_node_set.add(n1_int)
                    idx_n1 = node_id_to_idx_base[n1_int]
                    x1, y1, z1 = all_node_coords_base[idx_n1]

                    current_max_node_id += 1
                    n2 = current_max_node_id
                    mapdl.n(n2, x1, y1, z1)
                    mapdl.d(n2, "ALL")

                    if is_corner:
                        mapdl.type(2); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        mapdl.type(3); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        mapdl.type(4); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        spring_info.append((n1_int, C_each, C_each, C_each))
                        boundary_k_xyz[idx_n1, :] = K_each
                        boundary_c_xyz[idx_n1, :] = C_each
                        fixture_type[idx_n1] = 1
                    else:
                        mapdl.type(3); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        spring_info.append((n1_int, 0.0, C_each, 0.0))
                        boundary_k_xyz[idx_n1, 1] = K_each
                        boundary_c_xyz[idx_n1, 1] = C_each
                        fixture_type[idx_n1] = 2
            real_const_num += 1
    mapdl.allsel()

    # 8. 模态分析（EKILL 必须在 /SOLU 中、NROPT,FULL 之后、SOLVE 之前执行）
    mapdl.slashsolu()
    mapdl.nropt("FULL")          # 生死单元功能强制要求
    # EKILL：杀死落入去除区域的单元
    if len(removed_elem_ids) > 0:
        kill_elements_by_ids(mapdl, removed_elem_ids)
    mapdl.antype("MODAL")
    mapdl.modopt("LANB", N_MODES, nrmkey="ON")
    try:
        mapdl.solve()
    except Exception as e:
        print(f"  模态求解失败: {e}")
        mapdl.clear()
        continue

    # 9. 提取模态结果
    mapdl.post1()
    current_nnum = mapdl.mesh.nnum
    current_id_to_idx = {int(nid): idx for idx, nid in enumerate(current_nnum)}

    omega_k = np.zeros(N_MODES, dtype=np.float32)
    phi_x_safe = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
    phi_y_safe = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
    phi_z_safe = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)

    for k in range(1, N_MODES + 1):
        mapdl.set(1, k)
        f_hz = mapdl.post_processing.freq
        omega_k[k - 1] = 2.0 * np.pi * f_hz
        disp = np.array(mapdl.post_processing.nodal_displacement("ALL"), dtype=np.float32)
        for idx_curr, nid in enumerate(current_nnum):
            nid_int = int(nid)
            if nid_int in node_id_to_idx_base:
                idx_orig = node_id_to_idx_base[nid_int]
                phi_x_safe[idx_orig, k - 1] = disp[idx_curr, 0]
                phi_y_safe[idx_orig, k - 1] = disp[idx_curr, 1]
                phi_z_safe[idx_orig, k - 1] = disp[idx_curr, 2]

    phi_xyz_safe = np.stack([phi_x_safe, phi_y_safe, phi_z_safe], axis=-1).astype(np.float32)

    # 10. 阻尼比计算
    zeta_k = np.zeros(N_MODES, dtype=np.float32)
    zeta_boundary_components_xyz = np.zeros((N_MODES, 3), dtype=np.float32)
    for k in range(N_MODES):
        wk = omega_k[k]
        zeta_boundary_k = 0.0
        for ansys_nid, cx, cy, cz in spring_info:
            if ansys_nid in node_id_to_idx_base:
                idx_orig = node_id_to_idx_base[ansys_nid]
                phi_x = phi_x_safe[idx_orig, k]
                phi_y = phi_y_safe[idx_orig, k]
                phi_z = phi_z_safe[idx_orig, k]
                dissipation = cx * (phi_x ** 2) + cy * (phi_y ** 2) + cz * (phi_z ** 2)
                zeta_boundary_k += dissipation / (2.0 * wk)
                zeta_boundary_components_xyz[k, 0] += cx * (phi_x ** 2) / (2.0 * wk)
                zeta_boundary_components_xyz[k, 1] += cy * (phi_y ** 2) / (2.0 * wk)
                zeta_boundary_components_xyz[k, 2] += cz * (phi_z ** 2) / (2.0 * wk)
        zeta_k[k] = ZETA_MATERIAL + zeta_boundary_k

    # 11. 频率网格
    if FREQ_GRID_MODE in ("fixed", "both"):
        freqs_fixed = build_fixed_frequency_grid(FREQ_MIN, FREQ_MAX, N_FREQS_FIXED, mode="hybrid")
    else:
        freqs_fixed = None

    if FREQ_GRID_MODE in ("adaptive", "both"):
        # 简化的自适应网格生成
        freqs_parts = []
        prev = FREQ_MIN
        for idx_k, f_k in enumerate(omega_k / (2 * np.pi)):
            bw = 2.0 * zeta_k[idx_k] * f_k
            lo = max(FREQ_MIN, f_k - 3.0 * bw)
            hi = min(FREQ_MAX, f_k + 3.0 * bw)
            if prev < lo:
                freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(lo),
                                    max(2, int(5 * (lo - prev) / FREQ_MAX)), endpoint=False))
            freqs_parts.append(np.linspace(lo, hi, max(15, int(20 * (hi - lo) / FREQ_MAX)), endpoint=True))
            prev = hi
        if prev < FREQ_MAX:
            freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(FREQ_MAX),
                                max(2, int(5 * (FREQ_MAX - prev) / FREQ_MAX)), endpoint=True))
        freqs_adaptive = np.unique(np.sort(np.concatenate(freqs_parts)))[:N_FREQS].astype(np.float32)
    else:
        freqs_adaptive = None

    if FREQ_GRID_MODE == "fixed" or (FREQ_GRID_MODE == "both" and freqs_fixed is not None):
        freqs_main = freqs_fixed
    else:
        freqs_main = freqs_adaptive

    # 12. 方向感知 FRF 计算
    def _compute_frf_for_grid(freq_grid, phi_resp, phi_exc_force):
        if freq_grid is None:
            return None
        omega_q = 2.0 * np.pi * freq_grid
        frf = np.zeros((n_nodes_total, len(freq_grid), 2), dtype=np.float32)
        for k in range(N_MODES):
            wk = omega_k[k]
            zk = zeta_k[k]
            pk = phi_resp[:, k] * phi_exc_force[k]
            dw = wk**2 - omega_q**2
            gm = 2.0 * zk * wk * omega_q
            D = np.maximum(dw**2 + gm**2 + 1e-6, 1.0)
            frf[:, :, 0] += np.outer(pk, AMPLITUDE_SCALE * dw / D)
            frf[:, :, 1] += np.outer(pk, -AMPLITUDE_SCALE * gm / D)
        return frf

    phi_response = phi_xyz_safe[:, :, RESPONSE_DIR_INDEX]
    phi_force_exc = phi_xyz_safe[exc_idx, :, FORCE_DIR_INDEX]

    frf_main = _compute_frf_for_grid(freqs_main, phi_response, phi_force_exc)
    frf_adaptive = _compute_frf_for_grid(freqs_adaptive, phi_response, phi_force_exc)

    # 13. 刀触点字段
    tool_position = exc_actual.copy().astype(np.float32)
    contact_node_index = exc_idx
    force_direction_vector = np.zeros(3, dtype=np.float32)
    force_direction_vector[FORCE_DIR_INDEX] = 1.0
    response_direction_vector = np.zeros(3, dtype=np.float32)
    response_direction_vector[RESPONSE_DIR_INDEX] = 1.0

    active_pocket_id = -1
    for pocket_i, pocket_idx in enumerate(pockets_to_machine):
        cells = pocket_cells[pocket_idx]
        xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(
            x_pockets, y_pockets, cells, n_cols)
        xmin_p, xmax_p = xmin_frac * L, xmax_frac * L
        ymin_p, ymax_p = ymin_frac * W, ymax_frac * W
        if xmin_p <= exc_actual[0] <= xmax_p and ymin_p <= exc_actual[1] <= ymax_p:
            active_pocket_id = pocket_i
            break
    process_step = float(np.sum(removed_mask)) / max(float(n_elements), 1.0)

    # 14. 数据保存
    all_points.append(all_node_coords_base)
    all_frf.append(frf_main)
    all_freqs.append(freqs_main)
    all_omega.append(omega_k)
    all_zeta.append(zeta_k)
    all_phi.append(phi_response)
    all_phi_exc.append(phi_force_exc)
    all_phi_xyz.append(phi_xyz_safe)
    all_phi_exc_xyz.append(phi_xyz_safe[exc_idx, :, :].copy())
    all_zeta_boundary_components_xyz.append(zeta_boundary_components_xyz)
    all_boundary_k_xyz.append(boundary_k_xyz)
    all_boundary_c_xyz.append(boundary_c_xyz)
    all_fixture_type.append(fixture_type)
    all_surface_flags.append(np.zeros((n_nodes_total, 5), dtype=np.float32))  # 简化 surface flags
    all_element_node_indices.append(element_node_indices_base)
    all_element_node_ids.append(element_node_ids_base)
    all_element_types.append(element_types_base)
    all_element_centers.append(element_centers_base)
    all_element_ids.append(element_ids_base)
    all_excitation_index.append(np.array(exc_idx, dtype=np.int64))
    all_excitation_node_id.append(np.array(exc_node_id, dtype=np.int64))
    all_excitation_point.append(exc_actual.astype(np.float32))
    # 方向 / 刀触点 / 过程
    all_frf_adaptive.append(frf_adaptive if frf_adaptive is not None else np.zeros((0, 0, 2), dtype=np.float32))
    all_freqs_adaptive.append(freqs_adaptive if freqs_adaptive is not None else np.zeros((0,), dtype=np.float32))
    all_tool_position.append(tool_position)
    all_contact_node_index.append(np.array(contact_node_index, dtype=np.int64))
    all_force_direction_vector.append(force_direction_vector)
    all_response_direction_vector.append(response_direction_vector)
    all_active_pocket_id.append(np.array(active_pocket_id, dtype=np.int64))
    all_process_step.append(np.array(process_step, dtype=np.float32))
    all_removed_volume_ratio.append(np.array(removed_volume_ratio, dtype=np.float32))
    # EKILL 特有
    all_element_active_flag.append(element_active_flag)
    all_node_active_flag.append(node_active_flag)
    all_removed_element_flag.append(removed_element_flag)

    # 简易 Transolver 特征（ekill 版本可后续扩展）
    transolver_feats = np.zeros((n_nodes_total, len(TRANSOLVER_FEATURE_NAMES)), dtype=np.float32)
    transolver_feats[:, 0] = all_node_coords_base[:, 0] / L
    transolver_feats[:, 1] = all_node_coords_base[:, 1] / W
    transolver_feats[:, 2] = all_node_coords_base[:, 2] / H
    transolver_feats[:, 3] = E / E_BASE
    transolver_feats[:, 4] = rho / RHO_BASE
    transolver_feats[:, 5] = PRXY_BASE
    transolver_feats[:, 13] = (fixture_type == 1).astype(np.float32)
    transolver_feats[:, 14] = (fixture_type == 2).astype(np.float32)
    transolver_feats[:, 15:18] = safe_log10(boundary_k_xyz)
    transolver_feats[:, 18:21] = safe_log10(boundary_c_xyz)
    diag = np.sqrt(L ** 2 + W ** 2 + H ** 2)
    transolver_feats[:, 21] = np.linalg.norm(all_node_coords_base - exc_actual[None, :], axis=1) / diag
    transolver_feats[exc_idx, 22] = 1.0
    all_transolver_features.append(transolver_feats)

    n_removed = int(np.sum(removed_mask))
    print(f"N={n_nodes_total}, Ne={n_elements}, 去除={n_removed}/{n_elements}, "
          f"去除体积比={removed_volume_ratio:.3f}, "
          f"f₁={omega_k[0]/(2*np.pi):.1f}Hz, f₂={omega_k[1]/(2*np.pi):.1f}Hz")

    csv_writer.writerow([
        valid_samples + 1, n_nodes_total, n_elements, n_removed, f"{removed_volume_ratio:.4f}",
        n_pockets_to_machine, num_machined,
        f"{exc_actual[0]*1000:.2f}", f"{exc_actual[1]*1000:.2f}", f"{exc_actual[2]*1000:.2f}",
        f"{zeta_k[0]:.6f}", f"{zeta_k[1]:.6f}", f"{zeta_k[2]:.6f}" if N_MODES >= 3 else "",
        f"{omega_k[0]/(2*np.pi):.2f}", f"{omega_k[1]/(2*np.pi):.2f}",
        f"{omega_k[2]/(2*np.pi):.2f}" if N_MODES >= 3 else "",
        f"{E/E_BASE:.4f}", f"{rho/RHO_BASE:.4f}",
    ])
    csv_file.flush()

    # 网格可视化（每样本）
    try:
        import matplotlib as _mpl
        _mpl.use('Agg')
        import pyvista as pv

        mapdl.allsel()
        grid = mapdl.mesh._grid

        # 只显示存活单元（过滤掉被 EKILL 杀死的单元）
        alive_cell_indices = np.where(element_active_flag == 1)[0]
        if len(alive_cell_indices) > 0 and len(alive_cell_indices) < grid.n_cells:
            grid_alive = grid.extract_cells(alive_cell_indices)
        else:
            grid_alive = grid

        plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
        plotter.add_mesh(grid_alive, color='lightblue', show_edges=True,
                         edge_color='gray', line_width=0.3, opacity=0.9)

        # 被杀死的单元中心（红色标记）
        if n_removed > 0:
            killed_centers = element_centers_base[removed_mask]
            if killed_centers.shape[0] > 0:
                plotter.add_points(killed_centers, color='red', point_size=3,
                                   render_points_as_spheres=True, opacity=0.5)

        # 激励点
        plotter.add_points(exc_actual.reshape(1, -1), color='green', point_size=15,
                           render_points_as_spheres=True, label='Excitation')

        # 装夹点
        clamp_points_plot = []
        for (xmin, xmax, ymin, ymax) in all_clamp_areas:
            mapdl.nsel("S", "LOC", "X", xmin, xmax)
            mapdl.nsel("R", "LOC", "Y", ymin, ymax)
            mapdl.nsel("R", "LOC", "Z", 0, H)
            for nid in mapdl.mesh.nnum:
                nid_int = int(nid)
                if nid_int in node_id_to_idx_base:
                    clamp_points_plot.append(all_node_coords_base[node_id_to_idx_base[nid_int]])
        mapdl.allsel()
        if clamp_points_plot:
            clamp_points_plot = np.array(clamp_points_plot)
            plotter.add_points(clamp_points_plot, color='yellow', point_size=8,
                               render_points_as_spheres=True, label='Clamp')

        plotter.add_text(
            f'[EKILL] Sample {valid_samples+1}: {n_pockets_to_machine}/{num_machined} pockets, '
            f'killed={n_removed}/{n_elements}\n'
            f'Blue=Mesh, Red=Killed, Green=Excitation, Yellow=Clamp',
            font_size=10)
        plotter.camera_position = 'iso'
        plotter.screenshot(os.path.join(VIZ_DIR, f'sample_{valid_samples:03d}_mesh.png'))
        plotter.close()
    except Exception as e:
        print(f"  可视化失败: {e}")

    time.sleep(0.3)
    valid_samples += 1

csv_file.close()
print(f"CSV 日志已保存: {csv_path}")

mapdl.exit()
elapsed = time.time() - t0
print(f"\n生成完成, 耗时 {elapsed:.0f}s")
print(f"总样本数: {N_SAMPLES}个 (总尝试{attempt_count}次)")


# ============ 保存 HDF5 ============
def save_h5(name, idx_slice):
    idxs = list(idx_slice)
    with h5py.File(os.path.join(OUT_DIR, name), 'w') as f:
        # 文件级元数据
        f.attrs['format'] = 'modal_frf_transolver_ekill_v1'
        f.attrs['geometry_method'] = 'ekill'
        f.attrs['point_feature_names'] = ','.join(POINT_FEATURE_NAMES)
        f.attrs['transolver_feature_names'] = ','.join(TRANSOLVER_FEATURE_NAMES)
        f.attrs['surface_flag_names'] = ','.join(SURFACE_FLAG_NAMES)
        f.attrs['n_modes'] = N_MODES
        f.attrs['mesh_size_m'] = MESH_SIZE
        f.attrs['mass_normalized_modes'] = 'true'
        # 方向配置
        f.attrs['response_direction'] = RESPONSE_DIRECTION
        f.attrs['force_direction'] = FORCE_DIRECTION
        f.attrs['response_dir_index'] = RESPONSE_DIR_INDEX
        f.attrs['force_dir_index'] = FORCE_DIR_INDEX
        f.attrs['frf_definition'] = f'H_{RESPONSE_DIRECTION}{FORCE_DIRECTION}'
        f.attrs['frf_direction'] = RESPONSE_DIRECTION
        f.attrs['excitation_direction'] = FORCE_DIRECTION
        # 频率网格
        f.attrs['frequency_grid_mode'] = FREQ_GRID_MODE
        f.attrs['n_freqs_fixed'] = N_FREQS_FIXED if FREQ_GRID_MODE in ('fixed', 'both') else 0
        f.attrs['frequency_grid_definition'] = 'hybrid: 40% linear 0-1000Hz + 60% log 1000-5000Hz'
        f.attrs['element_connectivity'] = 'element_node_indices are 0-based indices; -1 means padding'
        f.attrs['ekill_note'] = 'element_active_flag: 1=alive, 0=killed; node_active_flag inferred from adjacent alive elements'
        f.attrs['zeta_formula'] = 'zeta_k = zeta_material + sum(C_xyz * phi_xyz^2)/(2*omega_k)'
        for i, idx in enumerate(idxs):
            grp = f.create_group(f'sample_{i}')
            grp.create_dataset('points', data=all_points[idx])
            grp.create_dataset('point_frf', data=all_frf[idx])
            grp.create_dataset('frequencies', data=all_freqs[idx])
            grp.create_dataset('modal_omega', data=all_omega[idx])
            grp.create_dataset('modal_zeta', data=all_zeta[idx])
            grp.create_dataset('modal_phi', data=all_phi[idx])
            grp.create_dataset('modal_phi_exc', data=all_phi_exc[idx])
            grp.create_dataset('transolver_point_features', data=all_transolver_features[idx])
            grp.create_dataset('modal_phi_xyz', data=all_phi_xyz[idx])
            grp.create_dataset('modal_phi_exc_xyz', data=all_phi_exc_xyz[idx])
            grp.create_dataset('modal_zeta_material', data=np.full(N_MODES, ZETA_MATERIAL, dtype=np.float32))
            grp.create_dataset('modal_zeta_boundary_components_xyz', data=all_zeta_boundary_components_xyz[idx])
            grp.create_dataset('boundary_k_xyz', data=all_boundary_k_xyz[idx])
            grp.create_dataset('boundary_c_xyz', data=all_boundary_c_xyz[idx])
            grp.create_dataset('fixture_type', data=all_fixture_type[idx])
            grp.create_dataset('surface_flags', data=all_surface_flags[idx])
            grp.create_dataset('element_node_indices', data=all_element_node_indices[idx])
            grp.create_dataset('element_node_ids', data=all_element_node_ids[idx])
            grp.create_dataset('element_types', data=all_element_types[idx])
            grp.create_dataset('element_centers', data=all_element_centers[idx])
            grp.create_dataset('element_ids', data=all_element_ids[idx])
            grp.create_dataset('excitation_index', data=all_excitation_index[idx])
            grp.create_dataset('excitation_node_id', data=all_excitation_node_id[idx])
            grp.create_dataset('excitation_point', data=all_excitation_point[idx])
            # 方向 / 刀触点 / 过程
            grp.create_dataset('tool_position', data=all_tool_position[idx])
            grp.create_dataset('contact_node_index', data=all_contact_node_index[idx])
            grp.create_dataset('force_direction_vector', data=all_force_direction_vector[idx])
            grp.create_dataset('response_direction_vector', data=all_response_direction_vector[idx])
            grp.create_dataset('active_pocket_id', data=all_active_pocket_id[idx])
            grp.create_dataset('process_step', data=all_process_step[idx])
            grp.create_dataset('removed_volume_ratio', data=all_removed_volume_ratio[idx])
            # EKILL 特有
            grp.create_dataset('element_active_flag', data=all_element_active_flag[idx])
            grp.create_dataset('node_active_flag', data=all_node_active_flag[idx])
            grp.create_dataset('removed_element_flag', data=all_removed_element_flag[idx])
            # 可选自适应 FRF
            if FREQ_GRID_MODE == 'both' and all_freqs_adaptive[idx].size > 0:
                grp.create_dataset('frequencies_adaptive', data=all_freqs_adaptive[idx])
                grp.create_dataset('point_frf_adaptive', data=all_frf_adaptive[idx])
    print(f"  保存: {name} ({len(idxs)}样本)")


save_h5('train.h5', range(N_TRAIN))
save_h5('val.h5', range(N_TRAIN, N_TRAIN + N_VAL))
save_h5('test.h5', range(N_TRAIN + N_VAL, N_SAMPLES))
print("完成! 数据已保存到:", OUT_DIR)

# ============ FRF 可视化 ============
frf_label = f'H_{RESPONSE_DIRECTION}{FORCE_DIRECTION}'
print(f"\n生成 FRF 可视化 ({frf_label})...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    coords0 = all_points[0]
    frf0 = all_frf[0]
    freqs0 = all_freqs[0]
    amp0 = np.sqrt(frf0[..., 0]**2 + frf0[..., 1]**2)
    n_nodes0 = len(coords0)

    # 随机选5个节点
    np.random.seed(42)
    selected_idx = np.random.choice(n_nodes0, size=min(5, n_nodes0), replace=False)

    fig = plt.figure(figsize=(20, 14))
    for i, idx_node in enumerate(selected_idx):
        coord_str = (f'({coords0[idx_node, 0]*1000:.0f},'
                     f'{coords0[idx_node, 1]*1000:.0f},'
                     f'{coords0[idx_node, 2]*1000:.0f})mm')
        amp_node = amp0[idx_node]

        # 左侧: dB 幅值
        ax_db = fig.add_subplot(5, 2, 2*i + 1)
        amp_db = 20 * np.log10(amp_node + 1e-12)
        ax_db.semilogx(freqs0, amp_db, 'b-', linewidth=1.0)
        for k in range(N_MODES):
            fk = all_omega[0][k] / (2*np.pi)
            zk = all_zeta[0][k]
            ax_db.axvline(fk, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                          label=f'f{k+1}={fk:.0f}Hz (ζ={zk:.4f})' if i == 0 else '')
        ax_db.set_ylabel(f'Point{i+1}\n{coord_str}\nMagnitude (dB)', fontsize=8)
        db_max = amp_db.max()
        ax_db.set_ylim(db_max - 60, db_max + 5)
        ax_db.grid(alpha=0.3)
        if i == 0:
            ax_db.legend(fontsize=7, loc='upper right')

        # 右侧: 线性幅值
        ax_lin = fig.add_subplot(5, 2, 2*i + 2)
        ax_lin.semilogx(freqs0, amp_node, 'b-', linewidth=1.0)
        for k in range(N_MODES):
            fk = all_omega[0][k] / (2*np.pi)
            ax_lin.axvline(fk, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_lin.set_ylabel(f'Point{i+1}\nMagnitude (lin)', fontsize=8)
        ax_lin.grid(alpha=0.3)
        p95 = np.percentile(amp_node, 95)
        p99 = np.percentile(amp_node, 99.9)
        ax_lin.set_ylim(0, min(p95 * 3, p99 * 0.8))

    axes = fig.get_axes()
    axes[-2].set_xlabel('Frequency (Hz)')
    axes[-1].set_xlabel('Frequency (Hz)')

    fig.suptitle(f'[EKILL] Grooved Workpiece FRF ({frf_label}) — {n_nodes0} nodes, '
                 f'f₁={all_omega[0][0]/(2*np.pi):.0f}Hz, '
                 f'f₂={all_omega[0][1]/(2*np.pi):.0f}Hz'
                 + (f', f₃={all_omega[0][2]/(2*np.pi):.0f}Hz' if N_MODES >= 3 else ''),
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'sample_000_frf.png'), dpi=150)
    plt.close()
    print(f"FRF 可视化保存: {VIZ_DIR}/")

    # 同时保存第一个样本的 FRF 数据供后续使用
    np.savez(os.path.join(VIZ_DIR, 'sample_000_frf_data.npz'),
             frequencies=freqs0, frf_re=frf0[..., 0], frf_im=frf0[..., 1],
             amp=amp0, omega=all_omega[0], zeta=all_zeta[0],
             points=coords0)
except Exception as e:
    print(f"  FRF 可视化失败: {e}")
