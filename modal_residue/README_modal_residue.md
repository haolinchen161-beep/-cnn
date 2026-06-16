# Modal-residue FRF surrogate workflow

This branch contains the first modal-residue FRF workflow for the grooved workpiece study.

## Files

- `scripts/first_step_modal_frf_check.py`  
  Single-sample ANSYS/Python closed-loop check. It verifies that mass-normalized modal parameters exported from MAPDL can reconstruct the same complex FRF as ANSYS MSUP harmonic response.

- `scripts/generate_modal_residue_dataset_filtered_v2.py`  
  Dataset generator. It uses 10 mass-normalized modes, an adaptive frequency upper bound that covers the 10th mode, simple near-mode filtering, and saves `modal_residue_z`.

- `scripts/check_modal_residue_dataset.py`  
  Dataset quality checker. It verifies HDF5 structure, modal frequency order, near-mode filtering, adaptive frequency coverage, modal-residue formula consistency, and FRF formula consistency.

- `train_modal_residue_model.py`  
  Minimal PyTorch baseline. It predicts `modal_omega` and `modal_residue_z(x)` and reconstructs FRF using the modal-superposition formula.

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
F:/pytorch_cuda12/python.exe -B train_modal_residue_model.py `
    --data-dir data_modal_residue_filtered `
    --epochs 300 `
    --query-nodes 512 `
    --frf-loss-weight 0.05
```

Outputs:

```text
runs/modal_residue_baseline/
├── best_model.pt
├── normalization_stats.npz
├── history.csv
├── val_metrics.csv
├── test_metrics.csv
└── summary.json
```

## Dataset target

For each sample and node:

```math
A_r(x) = \phi_{r,z}(x)\phi_{r,z}(x_f)
```

The reconstructed Z-Z FRF is:

```math
H_z(x,\omega)=\sum_{r=1}^{10}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
```
