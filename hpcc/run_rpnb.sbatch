#!/usr/bin/env bash
#SBATCH --job-name=rpnb
#SBATCH --output=rpnb_%j.out
#SBATCH --error=rpnb_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ "${1:-}" == "--resume" ]]; then
  RESUME_DIR="${2:?Usage: sbatch hpcc/run_rpnb.sbatch --resume RUN_DIR [CHECKPOINT_INTERVAL]}"
  CHECKPOINT_INTERVAL="${3:-10}"
  DATA=""
  SPEC=""
  OUT=""
else
  DATA=${1:?Usage: sbatch hpcc/run_rpnb.sbatch DATA.csv MODEL.yaml OUT_DIR [CHECKPOINT_INTERVAL]}
  SPEC=${2:?Usage: sbatch hpcc/run_rpnb.sbatch DATA.csv MODEL.yaml OUT_DIR [CHECKPOINT_INTERVAL]}
  OUT=${3:-rpnb_runs}
  CHECKPOINT_INTERVAL="${4:-10}"
  RESUME_DIR=""
fi

source .venv-rpnb/bin/activate
if [[ -n "${RESUME_DIR}" ]]; then
  python -m rpnb.cli fit --resume "$RESUME_DIR" --checkpoint-interval "$CHECKPOINT_INTERVAL"
else
  python -m rpnb.cli fit \
    --data "$DATA" \
    --spec "$SPEC" \
    --out "$OUT" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
fi
