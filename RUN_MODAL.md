# Run z-only modal training

Use the current lightweight branch with the existing generated HDF5 files:

```bash
git checkout mesh-modal-lite-clean
python -u modal_run.py
```

Current objective:

```text
mesh + geometry + stiffness boundary -> omega + full-node z-direction mode shapes
```

The model predicts:

```text
omega: [B, 3]
phi_z: [total_N, 3]
```

The dataset still loads `modal_phi_xyz`, but only for computing the z-dominance weighting in the loss. The network itself does not predict `phi_x` or `phi_y` in this stage.

Losses kept:

```text
1. natural-frequency log loss
2. sign-aligned z-direction mode-shape MSE
3. z-direction scale loss
4. z-direction MAC loss
5. per-mode z-dominance weighting
```

The mode-shape loss uses:

```text
w_k = min_mode_weight + (1 - min_mode_weight) * dir_z_ratio_k
```

Default:

```text
min_mode_weight = 0.2
```

This means non-z-dominant modes are not removed. Their `phi_z` projection is still learned, but their shape loss is weaker.

Damping and FRF are not trained in this branch. For the first-stage validation, use predicted `omega + phi_z`; use `modal_zeta` or a calibrated damping model later when reconstructing Z-Z FRF.

## Useful commands

Train:

```bash
python -u modal_run.py --data_dir ansys/data --out_dir sample/output_modal_zonly
```

Evaluate:

```bash
python -u sample/evaluate.py --data_dir ansys/data --out_dir sample/output_modal_zonly
```

Quick debug run:

```bash
python -u modal_run.py --epochs 3 --batch_size 1
```

Future expansion to six modes can use the same code path if the HDF5 files contain at least six modes:

```bash
python -u modal_run.py --n_modes 6 --data_dir ansys/data_20modes
```
