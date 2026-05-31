#!/usr/bin/env bash
#SBATCH --job-name=rpnb
#SBATCH --output=rpnb_%j.out
#SBATCH --error=rpnb_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

DATA=${1:?Usage: sbatch hpcc/run_rpnb.sbatch DATA.csv MODEL.yaml OUT_DIR}
SPEC=${2:?Usage: sbatch hpcc/run_rpnb.sbatch DATA.csv MODEL.yaml OUT_DIR}
OUT=${3:-rpnb_runs}

source .venv-rpnb/bin/activate
python -m rpnb.cli fit --data "$DATA" --spec "$SPEC" --out "$OUT"
