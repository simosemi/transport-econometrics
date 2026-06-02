#!/usr/bin/env bash
#
# SLURM template for running rpopit on MSU ICER HPCC.
#
# Submit:
#
#   sbatch hpcc/run_rpopit.sbatch /path/to/crashes.csv /path/to/model.yaml /path/to/output [checkpoint_interval]
#
# Resume after walltime interruption:
#
#   sbatch hpcc/run_rpopit.sbatch --resume /path/to/output/rpopit_YYYYMMDD_HHMMSS_microseconds [checkpoint_interval]
#
# Scale #SBATCH --cpus-per-task, --mem, and --time using the recommendations
# in docs/icer_hpcc.md and README.md.

#SBATCH --job-name=rpopit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

if [[ "${1:-}" == "--resume" ]]; then
  RESUME_DIR="${2:?Usage: sbatch hpcc/run_rpopit.sbatch --resume RUN_DIR [CHECKPOINT_INTERVAL]}"
  CHECKPOINT_INTERVAL="${3:-10}"
  DATA_CSV=""
  MODEL_YAML=""
  OUTPUT_DIR=""
else
  DATA_CSV="${1:?Usage: sbatch hpcc/run_rpopit.sbatch DATA_CSV MODEL_YAML OUTPUT_DIR [CHECKPOINT_INTERVAL]}"
  MODEL_YAML="${2:?Usage: sbatch hpcc/run_rpopit.sbatch DATA_CSV MODEL_YAML OUTPUT_DIR [CHECKPOINT_INTERVAL]}"
  OUTPUT_DIR="${3:?Usage: sbatch hpcc/run_rpopit.sbatch DATA_CSV MODEL_YAML OUTPUT_DIR [CHECKPOINT_INTERVAL]}"
  CHECKPOINT_INTERVAL="${4:-10}"
  RESUME_DIR=""
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
ENV_DIR="${RPOPIT_ENV_DIR:-$HOME/.venvs/rpopit}"
PYTHON_MODULE="${RPOPIT_PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
if [[ -n "${OUTPUT_DIR}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node list: ${SLURM_JOB_NODELIST:-unknown}"
echo "Submit dir: ${PROJECT_ROOT}"
echo "Data CSV: ${DATA_CSV}"
echo "Model YAML: ${MODEL_YAML}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Resume dir: ${RESUME_DIR}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-unknown} MB"
date

module purge
module load "${PYTHON_MODULE}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

# Keep BLAS/OpenMP libraries from oversubscribing the allocated CPU cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PROJECT_ROOT}"

if [[ -n "${RESUME_DIR}" ]]; then
  python -m rpopit.cli fit \
    --resume "${RESUME_DIR}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
else
  python -m rpopit.cli fit \
    --data "${DATA_CSV}" \
    --spec "${MODEL_YAML}" \
    --out "${OUTPUT_DIR}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
fi

date
