"""
Generate a 20-mode diagnostic ANSYS dataset.

This script intentionally does not overwrite ansys/generate_3d_test.py.
It reuses the original generator source and patches only:

1. N_MODES: default 20 instead of 3.
2. OUT_DIR: ansys/data_20modes instead of ansys/data.
3. VIZ_DIR: ansys/mesh_viz_20modes instead of ansys/mesh_viz.

Run from the repository root:

    python -u ansys/generate_20_modes.py

Optional environment overrides:

    set N_SAMPLES=30
    set N_TRAIN=24
    set N_VAL=3
    set N_TEST=3
    set MODAL_EXPORT_N_MODES=20
    set MODAL_EXPORT_OUT_SUBDIR=data_20modes
    python -u ansys/generate_20_modes.py

The output HDF5 files keep the same field names as the 3-mode dataset, but
modal_omega, modal_zeta, modal_phi_xyz, modal_effm, modal_pfact, modal_mass,
and modal_stiffness contain 20 modes by default.
"""
from __future__ import annotations

import os
from pathlib import Path


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence of {old!r}, found {count}.")
    return source.replace(old, new, 1)


def main() -> None:
    here = Path(__file__).resolve().parent
    base_script = here / "generate_3d_test.py"
    if not base_script.exists():
        raise FileNotFoundError(f"Base generator not found: {base_script}")

    n_modes = int(os.getenv("MODAL_EXPORT_N_MODES", "20"))
    out_subdir = os.getenv("MODAL_EXPORT_OUT_SUBDIR", "data_20modes")
    viz_subdir = os.getenv("MODAL_EXPORT_VIZ_SUBDIR", "mesh_viz_20modes")

    source = base_script.read_text(encoding="utf-8")
    source = source.replace(
        "ANSYS 凹槽工件数据集生成 — MeshGraphNet 物理一致版本。",
        "ANSYS 前20阶模态诊断数据集生成 — MeshGraphNet 物理一致版本。",
        1,
    )
    source = source.replace(
        "默认生成 300 个有效样本，保存到 ansys/data/train.h5、val.h5、test.h5。",
        f"默认导出前 {n_modes} 阶模态，保存到 ansys/{out_subdir}/train.h5、val.h5、test.h5。",
        1,
    )
    source = _replace_once(source, "N_MODES = 3", f"N_MODES = {n_modes}")
    source = _replace_once(
        source,
        'OUT_DIR = os.path.join(os.path.dirname(__file__), "data")',
        f'OUT_DIR = os.path.join(os.path.dirname(__file__), "{out_subdir}")',
    )
    source = _replace_once(
        source,
        'VIZ_DIR = os.path.join(os.path.dirname(__file__), "mesh_viz")',
        f'VIZ_DIR = os.path.join(os.path.dirname(__file__), "{viz_subdir}")',
    )

    print("=" * 80)
    print("Running 20-mode diagnostic generator")
    print(f"Base script : {base_script}")
    print(f"N_MODES     : {n_modes}")
    print(f"Output dir  : {here / out_subdir}")
    print(f"Viz dir     : {here / viz_subdir}")
    print("=" * 80)

    namespace = {
        "__name__": "__main__",
        "__file__": str(base_script),
        "__package__": None,
    }
    exec(compile(source, str(base_script), "exec"), namespace)


if __name__ == "__main__":
    main()
