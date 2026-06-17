# -*- coding: utf-8 -*-
"""
固定装夹刚度版数据集生成入口。

用途：
    保留 generate_modal_residue_dataset_filtered_v2.py 的求解主流程，
    但在运行前把采样策略改成：

    1. 装夹刚度固定基准 + 小扰动，不再 soft/normal/hard 分层；
    2. 主分层改为 coverage_level × layout_type，即加工覆盖范围 × 5/6/7 布局；
    3. 其他凹槽逻辑、边界 jitter、深度独立随机、FRF 保存开关等保持上一版逻辑。

运行：
    python -B modal_residue/generate_modal_residue_dataset_fixed_clamp.py

常用环境变量：
    N_SAMPLES, N_TRAIN, N_VAL, N_TEST
    K_CORNER_BASE, K_SIDE_BASE
    K_CORNER_JITTER, K_SIDE_JITTER
    OUT_DIR, SAVE_POINT_FRF, SAVE_POINT_FRF_QC_COUNT
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "generate_modal_residue_dataset_filtered_v2.py"
RUNTIME_SCRIPT = THIS_DIR / "_runtime_fixed_clamp_generator.py"


def _must_replace(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        raise RuntimeError(f"固定装夹补丁失败：未找到片段 {label}")
    return src.replace(old, new, 1)


def build_fixed_clamp_source(src: str) -> str:
    """把上一版生成器源码改成固定装夹 + coverage×layout 分层。"""
    src = _must_replace(
        src,
        "本版采用受控随机数据集：clamp_level × coverage_level 分层，保留 5/6/7 凹槽布局和边界扰动。",
        "本版采用受控随机数据集：装夹刚度固定基准 + 小扰动；coverage_level × layout_type 分层，保留 5/6/7 凹槽布局和边界扰动。",
        "docstring sampling strategy",
    )

    src = _must_replace(
        src,
        '''CLAMP_LEVELS = {
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
M_REF = 0.01''',
        '''# 装夹物理上视为固定工装/螺丝预紧状态；只保留小幅制造与拧紧误差。
# 不再把装夹刚度作为 soft/normal/hard 三类工况分层。
K_CORNER_BASE = float(os.getenv("K_CORNER_BASE", "3.0e7"))
K_SIDE_BASE = float(os.getenv("K_SIDE_BASE", "8.0e6"))
COVERAGE_WEIGHTS = {"low": 0.25, "medium": 0.50, "high": 0.25}
COVERAGE_LEVEL_CODE = {"low": 0, "medium": 1, "high": 2}
LAYOUT_WEIGHTS = {5: 1.0, 6: 1.0, 7: 1.0}
K_CORNER_JITTER = float(os.getenv("K_CORNER_JITTER", "0.10"))
K_SIDE_JITTER = float(os.getenv("K_SIDE_JITTER", "0.15"))
M_REF = 0.01''',
        "clamp config",
    )

    src = _must_replace(
        src,
        '''def _balanced_layouts(n):
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
    K_sides, C_sides, zeta_sides = [], [], []''',
        '''def build_sample_plan(n_train, n_val, n_test):
    """
    生成固定长度的样本计划。

    装夹刚度不再作为主分层变量：物理上视为同一固定装夹方案，只保留小扰动。
    主分层改为 coverage_level × layout_type = 3×3。
    每个 split 内按相同权重分配，保证 train/val/test 都覆盖 low/medium/high 和 5/6/7 布局。
    """
    plan = []
    split_specs = [("train", int(n_train)), ("val", int(n_val)), ("test", int(n_test))]
    coverage_labels = ["low", "medium", "high"]
    layout_labels = [5, 6, 7]
    combo_labels = [(g, l) for g in coverage_labels for l in layout_labels]
    combo_weights = {
        (g, l): COVERAGE_WEIGHTS[g] * LAYOUT_WEIGHTS[l]
        for g, l in combo_labels
    }
    for split, n_split in split_specs:
        combo_counts = _allocate_counts(n_split, combo_labels, combo_weights)
        for (coverage_level, layout_type), n_combo in combo_counts.items():
            for _ in range(int(n_combo)):
                plan.append({
                    "split": split,
                    "coverage_level": coverage_level,
                    "layout_type": int(layout_type),
                })
    random.shuffle(plan)
    assert len(plan) == int(n_train + n_val + n_test)
    return plan


def sample_clamp_parameters():
    """固定装夹基准刚度 + 样本内部小扰动。"""
    k_corner_base = float(K_CORNER_BASE)
    k_side_base = float(K_SIDE_BASE)
    K_corners, C_corners, zeta_corners = [], [], []
    K_sides, C_sides, zeta_sides = [], [], []''',
        "sample plan and clamp sampler",
    )

    replacements = [
        ('f.attrs["sampling_strategy"] = "stratified clamp_level x coverage_level; balanced layout_type 5/6/7 inside each group"',
         'f.attrs["sampling_strategy"] = "stratified coverage_level x layout_type; fixed clamp stiffness with small jitter"'),
        ('print("采样: clamp_level×coverage_level 分层；layout 5/6/7 组内均衡；深度独立三角分布；边界 jitter 受控随机")',
         'print("采样: 固定装夹刚度+小扰动；coverage_level×layout_type 分层；深度独立三角分布；边界 jitter 受控随机")'),
        ('"clamp_level_code": [],', '"clamp_model_code": [],'),
        ('"clamp_level", "coverage_level", "layout_type", "n_cols", "n_rows",',
         '"clamp_model", "coverage_level", "layout_type", "n_cols", "n_rows",'),
        ('''        split_name = plan_rec["split"]
        clamp_level = plan_rec["clamp_level"]
        coverage_level = plan_rec["coverage_level"]
        layout_type = int(plan_rec["layout_type"])''',
         '''        split_name = plan_rec["split"]
        clamp_model = "fixed"
        coverage_level = plan_rec["coverage_level"]
        layout_type = int(plan_rec["layout_type"])'''),
        (') = sample_clamp_parameters(clamp_level)', ') = sample_clamp_parameters()'),
        ('arrays["clamp_level_code"].append(np.array(CLAMP_LEVEL_CODE[clamp_level], dtype=np.int64))',
         'arrays["clamp_model_code"].append(np.array(0, dtype=np.int64))'),
        ('f"[{split_name}] clamp={clamp_level}, coverage={coverage_level}, layout={layout_type}, "',
         'f"[{split_name}] clamp=fixed, coverage={coverage_level}, layout={layout_type}, "'),
        ('            clamp_level, coverage_level, layout_type, n_cols, n_rows,',
         '            clamp_model, coverage_level, layout_type, n_cols, n_rows,'),
    ]
    for old, new in replacements:
        src = _must_replace(src, old, new, old[:60])

    # 清理残留描述，避免日志或 HDF 中继续出现“装夹分层”的误导说法。
    src = src.replace("clamp_level × coverage_level", "coverage_level × layout_type")
    src = src.replace("clamp_level×coverage_level", "coverage_level×layout_type")
    src = src.replace("clamp_level x coverage_level", "coverage_level x layout_type")
    src = src.replace("clamp×coverage", "coverage×layout")

    forbidden = ["CLAMP_LEVELS", "CLAMP_WEIGHTS", "CLAMP_LEVEL_CODE", "plan_rec[\"clamp_level\"]"]
    remain = [token for token in forbidden if token in src]
    if remain:
        raise RuntimeError(f"固定装夹补丁失败：源码仍包含 {remain}")
    return src


def main() -> None:
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"找不到基础生成器：{BASE_SCRIPT}")

    src = BASE_SCRIPT.read_text(encoding="utf-8")
    patched = build_fixed_clamp_source(src)
    RUNTIME_SCRIPT.write_text(patched, encoding="utf-8")
    print(f">>> 使用固定装夹刚度版运行：{RUNTIME_SCRIPT}")
    print(">>> 装夹刚度：K_CORNER_BASE/K_SIDE_BASE 固定基准 + K_CORNER_JITTER/K_SIDE_JITTER 小扰动")
    print(">>> 主分层：coverage_level × layout_type，不再按装夹刚度分层")

    # 用 run_path 执行生成器，使 __file__ 指向 runtime 文件，OUT_DIR 默认仍在 modal_residue 下。
    runpy.run_path(str(RUNTIME_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
