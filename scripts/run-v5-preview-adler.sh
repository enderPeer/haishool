#!/usr/bin/env bash
# Separate LAN preview with v5 maths/science routing and the legacy fact fallback.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p runs
exec 9>runs/v5-preview.lock
flock -n 9 || { echo 'The v5 preview is already running.' >&2; exit 1; }
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
exec /home/ender/homunculi/.venv/bin/python -u -m haishool.v5_app \
  --model /home/ender/haishool-final-20261001/exports/final-v5/haishool-v5-8x512-selected.pt \
  --records data/records-r3-all.jsonl --hops data/hops-v4b \
  --host 192.168.178.171 --port 8652 --device cpu
