# Mesh Modal Lite Clean: Z-only MeshGraphNet modal prediction

This branch is the current lightweight training branch.

The first-stage target is now simplified to:

```text
FE mesh + geometry + material + stiffness boundary
        -> first K natural circular frequencies omega
        -> full-node z-direction mode shapes phi_z
```

Default K is 3.

The model no longer predicts full-field `phi_x` and `phi_y` in this stage. The HDF5 loader still reads `modal_phi_xyz`, but only to compute the per-mode z-dominance ratio used by the loss weighting.

## Why z-only

The current task is Z-direction excitation and Z-direction response FRF. The modal numerator mainly uses the z-direction projection of each mode:

```text
phi_response_z * phi_excitation_z
```

Therefore, the first version should learn `omega + phi_z` before adding damping or FRF reconstruction.

## Dataset

The default dataset remains:

```text
ansys/data/train.h5
ansys/data/val.h5
ansys/data/test.h5
```

Required modal fields:

```text
modal_omega      [at least K]
modal_phi_xyz    [N, at least K, 3]
```

The actual training labels are:

```text
modal_omega[:K]
modal_phi_xyz[:, :K, 2]
```

Damping and FRF fields may exist in the HDF5 files, but this branch ignores them during training.

## Model output

```text
omega: [B, K]
phi_z: [total_N, K]
```

## Loss

Frequency loss is not direction-weighted.

Mode-shape loss is z-only and uses:

```text
w_k = min_mode_weight + (1 - min_mode_weight) * dir_z_ratio_k
```

Default:

```text
min_mode_weight = 0.2
```

This keeps all samples. Non-z-dominant modes are not removed; their z projection is still learned, but with lower shape-loss weight.

## Train

```bash
python -u modal_run.py
```

Equivalent explicit command:

```bash
python -u modal_run.py --data_dir ansys/data --out_dir sample/output_modal_zonly --n_modes 3
```

The old sample entrypoint now redirects to the same training flow:

```bash
python -u sample/run_validation.py
```

## Evaluate

```bash
python -u sample/evaluate.py --data_dir ansys/data --out_dir sample/output_modal_zonly
```

Main metrics:

```text
freq_mae_hz
freq_mape_percent
phi_z_mse
phi_z_scale
phi_z_mac
phi_z_mac_mode1 / mode2 / mode3
dir_z_ratio_mode1 / mode2 / mode3
mode_weight_mode1 / mode2 / mode3
```

## Current research scope

This branch only validates:

```text
complex machined geometry + equivalent clamping stiffness -> omega + phi_z
```

Damping and FRF reconstruction should be added later as a separate physical layer. For early FRF checks, use predicted `omega + phi_z` together with existing `modal_zeta` or a calibrated damping model.

If three modes are insufficient for the target frequency band, the same code can be run with six modes using HDF5 files that contain at least six modes:

```bash
python -u modal_run.py --n_modes 6 --data_dir ansys/data_20modes
```
