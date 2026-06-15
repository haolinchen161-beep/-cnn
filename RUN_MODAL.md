# Run modal-only training

Use the lightweight branch with the existing generated HDF5 files:

```bash
git checkout mesh-modal-lite-clean
python -u modal_run.py
```

Current objective:

```text
mesh + geometry + stiffness boundary -> omega + full-node xyz mode shapes
```

Only three losses are kept:

```text
1. natural-frequency log loss
2. sign-aligned mode-shape MSE
3. 3D MAC loss
```

Damping and FRF are not trained in this branch.
