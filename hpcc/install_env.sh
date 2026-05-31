#!/usr/bin/env bash
#
# Create a Python virtual environment for rpopit on MSU ICER HPCC.
#
# Run from the rpopit project root after cloning/copying the repository:
#
#   bash hpcc/install_env.sh
#
# Optional environment variables:
#   RPOPIT_ENV_DIR       Path to virtual environment. Default: $HOME/.venvs/rpopit
#   RPOPIT_PYTHON_MODULE Python module to load. Default: Python/3.11.3-GCCcore-12.3.0

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${RPOPIT_ENV_DIR:-$HOME/.venvs/rpopit}"
PYTHON_MODULE="${RPOPIT_PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

echo "Project root: ${PROJECT_ROOT}"
echo "Environment:  ${ENV_DIR}"
echo "Python module: ${PYTHON_MODULE}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE}"
else
  echo "Warning: module command not found; using python from PATH."
fi

python --version
python -m venv "${ENV_DIR}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${PROJECT_ROOT}"

python - <<'PY'
import numpy
import pandas
import scipy
import yaml
import openpyxl
import rpopit

print("Validated imports:")
print("  numpy", numpy.__version__)
print("  pandas", pandas.__version__)
print("  scipy", scipy.__version__)
print("  pyyaml", yaml.__version__)
print("  openpyxl", openpyxl.__version__)
print("  rpopit", rpopit.__version__)
PY

cat <<EOF

Done.

Activate this environment in future sessions with:

  module purge
  module load ${PYTHON_MODULE}
  source ${ENV_DIR}/bin/activate

Submit an example job with:

  sbatch hpcc/run_rpopit.sbatch /path/to/crashes.csv /path/to/model.yaml /path/to/output
EOF
