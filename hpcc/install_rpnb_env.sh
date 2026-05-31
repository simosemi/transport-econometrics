#!/usr/bin/env bash
set -euo pipefail

module purge || true
module load Python/3.11 || module load python/3.11 || true

python -m venv .venv-rpnb
source .venv-rpnb/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python - <<'PY'
import rpnb
print("Installed RPNB", rpnb.__version__)
PY
