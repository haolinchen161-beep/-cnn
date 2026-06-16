# Generate 20-mode diagnostic ANSYS data

This is for modal-family inspection, not for the current lightweight 3-mode training run.

## Run

From the repository root:

```bash
git checkout mesh-modal-lite-clean
python -u ansys/generate_20_modes.py
```

By default it writes:

```text
ansys/data_20modes/train.h5
ansys/data_20modes/val.h5
ansys/data_20modes/test.h5
ansys/mesh_viz_20modes/
```

The field names remain compatible with the original dataset. The modal fields now contain 20 modes by default:

```text
modal_omega      [20]
modal_zeta       [20]
modal_phi_xyz    [N, 20, 3]
modal_effm       [20, 3]
modal_pfact      [20, 3]
modal_mass       [20]
modal_stiffness  [20]
```

## Recommended quick test

Generate a small diagnostic set first:

```bash
set N_SAMPLES=30
set N_TRAIN=24
set N_VAL=3
set N_TEST=3
python -u ansys/generate_20_modes.py
```

PowerShell equivalent:

```powershell
$env:N_SAMPLES="30"
$env:N_TRAIN="24"
$env:N_VAL="3"
$env:N_TEST="3"
python -u ansys/generate_20_modes.py
```

## Optional overrides

```bash
set MODAL_EXPORT_N_MODES=20
set MODAL_EXPORT_OUT_SUBDIR=data_20modes
set MODAL_EXPORT_VIZ_SUBDIR=mesh_viz_20modes
```

## Inspect after generation

Run the H5 inspection script on the new folder:

```bash
python inspect_modal_h5.py --data_dir "ansys/data_20modes"
```

On your Windows path, for example:

```bash
python inspect_modal_h5.py --data_dir "F:\毕业论文\-cnn-gnn-meshgraphnet-refactor\ansys\data_20modes"
```
