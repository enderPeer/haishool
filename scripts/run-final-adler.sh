#!/usr/bin/env bash
# Dedicated, restartable Haishool training job; leaves existing chat services alone.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p runs
exec 9>runs/final-v5.lock
if ! flock -n 9; then
  echo "Haishool final pipeline already holds the training lock." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${HAISHOOL_PYTHON:-/home/ender/homunculi/.venv/bin/python}"
"$python_bin" -u -m haishool.final_run pipeline \
  --root runs/final-v5 --data data/final-v5 --workers 12 "$@"
