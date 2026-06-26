# -*- coding: utf-8 -*-
"""
基于 generate_modal_residue_dataset_stage1_fullbase_no_knn_skip.py 的真实装夹修正版。

只改动三类装夹相关内容：
1. 将装夹刚度三档改为“正常拧紧工况”下的低/中/高预紧范围，不再包含未拧紧/松夹状态；
2. 将三个浮动装夹由单 Y 向弹簧改为 X/Y/Z 三向弹簧；
3. 采用各向异性方向刚度：侧边接触法向 Y 最大，X/Z 为较弱等效切向/面外约束。

其余逻辑均从原始脚本读取并执行，不在本文件中重写。
"""
from __future__ import annotations

from pathlib import Path


THIS_FILE = Path(__file__).resolve()
BASE_FILE = THIS_FILE.with_name("generate_modal_residue_dataset_stage1_fullbase_no_knn_skip.py")


def _replace_exact(src: str, old: str, new: str, name: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"Patch '{name}' expected exactly 1 match, found {count}. Base file may have changed.")
    return src.replace(old, new, 1)


source = BASE_FILE.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# 1) 刚度范围：保留原有 weak/medium/strong 三档接口，
#    但 weak 表示正常拧紧下的低预紧，不表示松夹/未拧紧。
# -----------------------------------------------------------------------------
source = _replace_exact(
    source,
    '''# 装夹刚度三档 + 小扰动：比固定基准更能学习边界影响，但仍比全连续大范围随机更容易训练。
CLAMP_LEVELS = ["weak", "medium", "strong"]
CLAMP_LEVEL_CODE = {"weak": 0, "medium": 1, "strong": 2}
K_CORNER_LEVEL_BASE = {
    "weak": float(os.getenv("K_CORNER_WEAK", "1.0e7")),
    "medium": float(os.getenv("K_CORNER_MEDIUM", "3.0e7")),
    "strong": float(os.getenv("K_CORNER_STRONG", "8.0e7")),
}
K_SIDE_LEVEL_BASE = {
    "weak": float(os.getenv("K_SIDE_WEAK", "2.0e6")),
    "medium": float(os.getenv("K_SIDE_MEDIUM", "8.0e6")),
    "strong": float(os.getenv("K_SIDE_STRONG", "2.0e7")),
}
COVERAGE_WEIGHTS = {"low": 0.25, "medium": 0.50, "high": 0.25}
LAYOUT_WEIGHTS = {5: 1.0, 6: 1.0, 7: 1.0}
COVERAGE_LEVEL_CODE = {"low": 0, "medium": 1, "high": 2}
K_CORNER_JITTER = float(os.getenv("K_CORNER_JITTER", "0.10"))
K_SIDE_JITTER = float(os.getenv("K_SIDE_JITTER", "0.15"))
M_REF = 0.01
''',
    '''# 装夹刚度三档 + 小扰动：真实螺栓已拧紧后的正常工况。
# 说明：为兼容原有 sample plan，仍保留 weak/medium/strong 名称；
# 这里 weak 表示“正常拧紧下的偏低预紧/偏低接触刚度”，不表示松夹或未拧紧。
CLAMP_LEVELS = ["weak", "medium", "strong"]
CLAMP_LEVEL_CODE = {"weak": 0, "medium": 1, "strong": 2}
K_CORNER_LEVEL_BASE = {
    "weak": float(os.getenv("K_CORNER_WEAK", "4.0e7")),
    "medium": float(os.getenv("K_CORNER_MEDIUM", "6.0e7")),
    "strong": float(os.getenv("K_CORNER_STRONG", "8.0e7")),
}
K_SIDE_LEVEL_BASE = {
    "weak": float(os.getenv("K_SIDE_WEAK", "1.0e7")),
    "medium": float(os.getenv("K_SIDE_MEDIUM", "1.5e7")),
    "strong": float(os.getenv("K_SIDE_STRONG", "2.0e7")),
}
COVERAGE_WEIGHTS = {"low": 0.25, "medium": 0.50, "high": 0.25}
LAYOUT_WEIGHTS = {5: 1.0, 6: 1.0, 7: 1.0}
COVERAGE_LEVEL_CODE = {"low": 0, "medium": 1, "high": 2}
K_CORNER_JITTER = float(os.getenv("K_CORNER_JITTER", "0.10"))
K_SIDE_JITTER = float(os.getenv("K_SIDE_JITTER", "0.10"))
# 侧边接触面法向为 Y；X/Z 为等效切向/面外连接刚度。
# 四角拧入连接较强，浮动贯穿孔凸台三向均有约束但 X/Z 弱于 Y。
CORNER_K_DIRECTION_FACTORS = np.asarray([0.70, 1.00, 0.70], dtype=np.float64)
SIDE_K_DIRECTION_FACTORS = np.asarray([0.50, 1.00, 0.50], dtype=np.float64)
M_REF = 0.01
''',
    "normal_tightened_clamp_stiffness",
)

# -----------------------------------------------------------------------------
# 2) clamp token：侧边/浮动装夹也写入三向 K/C，且三向各向异性。
# -----------------------------------------------------------------------------
source = _replace_exact(
    source,
    '''        if is_corner:
            K = float(K_corners[i])
            C = float(C_corners[i])
            Kx, Ky, Kz = K, K, K
            Cx, Cy, Cz = C, C, C
            ctype = 1.0
        else:
            j = i - 4
            K = float(K_sides[j])
            C = float(C_sides[j])
            Kx, Ky, Kz = 0.0, K, 0.0
            Cx, Cy, Cz = 0.0, C, 0.0
            ctype = 0.5
''',
    '''        if is_corner:
            K = float(K_corners[i])
            C = float(C_corners[i])
            factors = CORNER_K_DIRECTION_FACTORS
            ctype = 1.0
        else:
            j = i - 4
            K = float(K_sides[j])
            C = float(C_sides[j])
            factors = SIDE_K_DIRECTION_FACTORS
            ctype = 0.5
        Kx, Ky, Kz = (K * factors).tolist()
        # 沿用原阻尼生成方式，但令各方向阻尼与对应方向刚度一致：C_d ∝ sqrt(K_d)。
        Cx, Cy, Cz = (C * np.sqrt(np.maximum(factors, 0.0))).tolist()
''',
    "clamp_feature_directional_xyz",
)

# -----------------------------------------------------------------------------
# 3) FE 弹簧：所有七个装夹区均施加 X/Y/Z 三向 COMBIN14；
#    三个方向使用不同 real constant，对应各向异性刚度。
# -----------------------------------------------------------------------------
source = _replace_exact(
    source,
    '''            K_this = K_corners[idx_area] if is_corner else K_sides[idx_area - 4]
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
''',
    '''            K_this = K_corners[idx_area] if is_corner else K_sides[idx_area - 4]
            C_this = C_corners[idx_area] if is_corner else C_sides[idx_area - 4]
            direction_factors = CORNER_K_DIRECTION_FACTORS if is_corner else SIDE_K_DIRECTION_FACTORS
            K_each_xyz = (float(K_this) * direction_factors / n_selected).astype(np.float64)
            # 沿用原阻尼生成方式，但令各方向阻尼与对应方向刚度一致：C_d ∝ sqrt(K_d)。
            C_each_xyz = (float(C_this) * np.sqrt(np.maximum(direction_factors, 0.0)) / n_selected).astype(np.float64)

            real_x, real_y, real_z = real_const_num, real_const_num + 1, real_const_num + 2
            mapdl.r(real_x, float(K_each_xyz[0]), 0.0)
            mapdl.r(real_y, float(K_each_xyz[1]), 0.0)
            mapdl.r(real_z, float(K_each_xyz[2]), 0.0)

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

                mapdl.type(2); mapdl.real(real_x); mapdl.e(n1_int, n2)
                mapdl.type(3); mapdl.real(real_y); mapdl.e(n1_int, n2)
                mapdl.type(4); mapdl.real(real_z); mapdl.e(n1_int, n2)
                point_features[idx_n1, 3] = 1.0 if is_corner else 0.5
                spring_info.append((n1_int, float(C_each_xyz[0]), float(C_each_xyz[1]), float(C_each_xyz[2])))
                spring_k_xyz[idx_n1, :] = K_each_xyz.astype(np.float32)
                spring_c_xyz[idx_n1, :] = C_each_xyz.astype(np.float32)
                node_type[idx_n1] = 4 if is_corner else 3
                # 兼容旧版 point_features 的单标量 K/C：使用侧边接触法向 Y 的等效值。
                point_features[idx_n1, 4] = np.log10(float(K_each_xyz[1]))
                point_features[idx_n1, 5] = np.log10(float(C_each_xyz[1]))
            real_const_num += 3
''',
    "floating_clamp_three_directional_springs",
)

# 使用当前文件名编译，便于报错定位到这个修正版脚本。
exec_globals = {
    "__name__": "__main__",
    "__file__": str(THIS_FILE),
    "__package__": None,
}
exec(compile(source, str(THIS_FILE), "exec"), exec_globals)
