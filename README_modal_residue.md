# Modal-residue FRF surrogate workflow

This branch contains the first modal-residue FRF workflow for the grooved workpiece study.

## Research goal and physical setting

The goal is to build a fast surrogate model for the Z-direction FRF of a machined Al7075 workpiece. Instead of directly predicting the full FRF curve, the workflow predicts modal parameters and reconstructs FRF through modal superposition.

The learning targets are:

```text
modal_omega        # first 10 natural angular frequencies
modal_residue_z    # A_r(x) = phi_r,z(x) * phi_r,z(x_f)
```

The reconstructed Z-Z FRF is:

\[
H_z(x,\omega)=\sum_{r=1}^{10}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
\]

where:

```text
x    = response node
x_f  = excitation node
r    = modal index
A_r  = Z-direction modal residue
```

ANSYS/MAPDL is used to solve the modal problem and export the mass-normalized modal shapes. Python then computes the modal residue from the exported mode shapes.

## Main model/data parameters

| Item | Value / setting | Notes |
|---|---:|---|
| Workpiece material | Al7075 approximation | used for the machining workpiece |
| `E_BASE` | `71.7e9 Pa` | Young's modulus |
| `RHO_BASE` | `2810 kg/m^3` | density |
| `PRXY_BASE` | `0.33` | Poisson's ratio |
| `L_BASE` | `0.160 m` | length |
| `W_BASE` | `0.060 m` | width |
| `H_BASE` | `0.010 m` | thickness |
| `MESH_SIZE` | `0.006 m` | default FE mesh size |
| `N_MODES` | `10` | first 10 modes are saved/predicted |
| `N_FREQS` | `120` | FRF frequency points |
| `MIN_RELATIVE_MODE_GAP` | `0.01` | near-mode filtering threshold |
| `USE_MASS_NORMALIZATION` | `True` | modal mass is treated as 1 |
| `USE_LUMPED_MASS` | `False` | consistent mass matrix |

Material and density are slightly perturbed in the dataset:

```text
E_RANGE   = 0.95 ~ 1.05
RHO_RANGE = 0.97 ~ 1.03
```

## Machining state parameters

The geometry contains different pocket layouts and machining depths.

| Parameter | Setting | Notes |
|---|---:|---|
| `layout_type` | `5 / 6 / 7` | number/layout of machined pocket regions |
| `coverage_level` | `low / medium / high` | machining coverage level |
| `TARGET_DEPTH_RANGE` | `0.25 ~ 0.60` | target pocket depth ratio |
| `TARGET_DEPTH_MODE` | `0.42` | triangular-distribution mode |
| `CURRENT_PROGRESS_RANGE` | `0.25 ~ 1.00` | current machining progress |
| `GAP_ABS` | `0.006 m` | gap between pocket regions |
| `BORDER_ABS` | `0.006 m` | reserved border distance |
| `GRID_JITTER_RANGE` | `0.08 ~ 0.15` | random grid perturbation |

The node features include geometric position, material ratio, local thickness, pocket depth, cut-region information, spring stiffness/damping information, and node type.

## Boundary-condition choice

The boundary condition is modeled as elastic clamping rather than ideal fully fixed support.

This choice follows the physical setup discussed at the beginning of the project: the workpiece is held by fixtures/bolts/clamps, so the support stiffness is finite and has small tightening/manufacturing variations. A perfectly fixed boundary would over-constrain the workpiece and can shift low-order modal frequencies and FRF peaks too strongly.

The current boundary strategy is:

1. Use discrete `COMBIN14` spring elements to connect selected clamp nodes to fully constrained virtual nodes.
2. Use four corner clamp regions plus several side clamp regions.
3. Corner clamp regions constrain `UX / UY / UZ`.
4. Side clamp regions mainly constrain `UY`.
5. Use fixed baseline stiffness with small random jitter, instead of large soft/normal/hard categories.

Clamp-region parameters:

```text
clamp_len = 0.010 m
corner clamp: near four corner-edge regions
side clamp: placed along side edges, away from corners
min side-clamp spacing ≈ 2 * H
```

Spring stiffness and damping parameters:

| Parameter | Value | Notes |
|---|---:|---|
| `K_CORNER_BASE` | `3.0e7 N/m` | total baseline corner clamp stiffness |
| `K_SIDE_BASE` | `8.0e6 N/m` | total baseline side clamp stiffness |
| `K_CORNER_JITTER` | `0.10` | corner stiffness jitter |
| `K_SIDE_JITTER` | `0.15` | side stiffness jitter |
| `ZETA_JOINT_BASE` | `0.015` | joint/contact damping baseline |
| `ZETA_JOINT_JITTER` | `0.20` | joint damping jitter |

For a selected clamp region, the total stiffness/damping is distributed to all selected nodes:

```text
K_each = K_this / n_selected
C_each = C_this / n_selected
```

Corner nodes receive springs in `UX / UY / UZ`; side clamp nodes receive the side-direction spring currently implemented as `UY`.

## Files

- `scripts/first_step_modal_frf_check.py`  
  Single-sample ANSYS/Python closed-loop check. It verifies that mass-normalized modal parameters exported from MAPDL can reconstruct the same complex FRF as ANSYS MSUP harmonic response.

- `scripts/generate_modal_residue_dataset_filtered_v2.py`  
  Dataset generator. It uses 10 mass-normalized modes, an adaptive frequency upper bound that covers the 10th mode, simple near-mode filtering, and saves `modal_residue_z`.

- `scripts/check_modal_residue_dataset.py`  
  Dataset quality checker. It verifies HDF5 structure, modal frequency order, near-mode filtering, adaptive frequency coverage, modal-residue formula consistency, and FRF formula consistency.

- `train_modal_residue_model.py`  
  Minimal PyTorch baseline. It predicts `modal_omega` and `modal_residue_z(x)` and reconstructs FRF using the modal-superposition formula.

## Dataset target

For each sample and node:

\[
A_r(x) = \phi_{r,z}(x)\phi_{r,z}(x_f)
\]

Because the mode shapes are mass-normalized, modal mass is 1 and the residue does not need an additional division by modal mass.

The reconstructed Z-Z FRF is:

\[
H_z(x,\omega)=\sum_{r=1}^{10}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
\]

## Local data generation

Run from `F:\毕业论文\new` or the project root where MAPDL/PyMAPDL is configured:

```powershell
F:/pytorch_cuda12/python.exe -B scripts/generate_modal_residue_dataset_filtered_v2.py
```

For a 3-sample smoke test:

```powershell
$env:N_SAMPLES="3"
$env:N_TRAIN="1"
$env:N_VAL="1"
$env:N_TEST="1"
F:/pytorch_cuda12/python.exe -B scripts/generate_modal_residue_dataset_filtered_v2.py
```

## Quality check

```powershell
F:/pytorch_cuda12/python.exe -B scripts/check_modal_residue_dataset.py --data-dir data_modal_residue_filtered
```

Expected first small dataset result:

- 30 samples checked
- OK = 30, WARN = 0, ERROR = 0
- `modal_residue_z` formula relative error: 0
- `point_frf` formula relative error around `1e-5` or lower

## Training

The HDF5 data are local and are not committed to GitHub. Train locally:

```powershell
F:/pytorch_cuda12/python.exe run_meshgraph_modal.py
```

Current training target transform:

```text
Y = asinh(A / s_mode)
```

Current loss components:

```text
omega log-MSE
full signed-asinh residue loss
top-|A| physical auxiliary loss
node-dominant physical auxiliary loss
```

Outputs:

```text
runs/modal_residue_asinh_fixedclamp300/
├── best_model.pt
├── last_model.pt
├── normalization_stats.npz
├── training_log.csv
├── history.csv
├── val_metrics.csv
├── test_metrics.csv
└── summary.json
```

For small-sample overfit diagnostics, use the debug switches in `run_meshgraph_modal.py`:

```python
DEBUG_TRAIN_SAMPLES = 1
DEBUG_VAL_SAMPLES = 1
DEBUG_TEST_SAMPLES = 1
DEBUG_VAL_TEST_FROM_TRAIN = True
```

For full-data training, set:

```python
DEBUG_TRAIN_SAMPLES = 0
DEBUG_VAL_SAMPLES = 0
DEBUG_TEST_SAMPLES = 0
DEBUG_VAL_TEST_FROM_TRAIN = False
```
