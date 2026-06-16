# Run from repository root or F:\毕业论文\new after data generation.
# HDF5 data are local and are not committed to GitHub.

$ErrorActionPreference = "Stop"

$PythonExe = "F:/pytorch_cuda12/python.exe"
$DataDir = "data_modal_residue_filtered"
$OutDir = "runs/modal_residue_baseline"

& $PythonExe -B modal_residue/scripts/check_modal_residue_dataset.py --data-dir $DataDir

& $PythonExe -B modal_residue/train_modal_residue_model.py `
  --data-dir $DataDir `
  --out-dir $OutDir `
  --epochs 300 `
  --query-nodes 512 `
  --eval-query-nodes 1024 `
  --frf-loss-weight 0.05
