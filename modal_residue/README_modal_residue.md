# Modal-residue FRF workflow

This branch is cleaned to keep only the modal-residue workflow files.

## Files

```text
modal_residue/
├── README_modal_residue.md
├── train_modal_residue_model.py
└── validate_dataset.py
```

The generated ANSYS HDF5 data are local and are not committed to GitHub.

## Validate local dataset

Run from the repository root, where `data_modal_residue_filtered/` exists:

```powershell
F:/pytorch_cuda12/python.exe -B modal_residue/validate_dataset.py --data-dir data_modal_residue_filtered
```

## Train baseline model

```powershell
F:/pytorch_cuda12/python.exe -B modal_residue/train_modal_residue_model.py `
  --data-dir data_modal_residue_filtered `
  --out-dir runs/modal_residue_baseline `
  --epochs 300 `
  --query-nodes 512 `
  --eval-query-nodes 1024 `
  --frf-loss-weight 0.05
```

The model predicts:

```text
modal_omega       # 10 modal angular frequencies
modal_residue_z   # A_r(x)=phi_r,z(x)*phi_r,z(x_f)
```

FRF reconstruction uses:

```math
H_z(x,\omega)=\sum_{r=1}^{10}\frac{A_r(x)}{\omega_r^2-\omega^2+2j\zeta_r\omega_r\omega}
```
