# Running rpopit on MSU ICER HPCC

This guide gives a conservative SLURM workflow for running `rpopit` on MSU
ICER HPCC. ICER uses SLURM for batch jobs and provides Python through the module
system. If the default Python module in these scripts is not available on your
assigned node type, run `module spider Python` and set `RPOPIT_PYTHON_MODULE`
to an available Python 3.10+ module.

## 1. Install the Environment

From the project root:

```bash
bash hpcc/install_env.sh
```

By default this creates:

```text
$HOME/.venvs/rpopit
```

To choose a different environment path or Python module:

```bash
export RPOPIT_ENV_DIR="$HOME/rpopit-env"
export RPOPIT_PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
bash hpcc/install_env.sh
```

## 2. Activate the Environment

For an interactive login or development session:

```bash
module purge
module load "${RPOPIT_PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
source "${RPOPIT_ENV_DIR:-$HOME/.venvs/rpopit}/bin/activate"
python -m rpopit.cli --help
```

## 3. Submit a Batch Job

```bash
sbatch hpcc/run_rpopit.sbatch \
  /mnt/research/YOUR_GROUP/crashes.csv \
  /mnt/research/YOUR_GROUP/model.yaml \
  /mnt/research/YOUR_GROUP/rpopit_runs
```

The batch script runs:

```bash
python -m rpopit.cli fit \
  --data /mnt/research/YOUR_GROUP/crashes.csv \
  --spec /mnt/research/YOUR_GROUP/model.yaml \
  --out /mnt/research/YOUR_GROUP/rpopit_runs \
  --checkpoint-interval 10
```

To resume a run after a walltime interruption:

```bash
sbatch hpcc/run_rpopit.sbatch \
  --resume /mnt/research/YOUR_GROUP/rpopit_runs/rpopit_YYYYMMDD_HHMMSS_microseconds
```

The resumed job reads `model_spec.yaml`, `run_metadata.yaml`, and
`checkpoints/checkpoint_latest.npz` from the previous run directory.

## 4. Recommended SLURM Settings

These settings assume one node, 200-500 simulation draws, 1-4 random
parameters, and `chunk_size: 10000`. Increase memory when using larger
`chunk_size`, more random parameters, many categories, or post-estimation
predicted probabilities for very large datasets.

| Observations | CPUs | Memory | Wall Time | Notes |
|---:|---:|---:|---:|---|
| 100,000 | 4 | 32G | 4-8 hr | Good default for model development and 200-500 draws. |
| 500,000 | 8 | 96G | 12-24 hr | Use `workers: 2-4` only after a serial benchmark; monitor memory. |
| 1,000,000 | 12-16 | 192G | 24-48 hr | Prefer larger wall time, checkpoint output folder, and smaller chunks if memory pressure appears. |

Recommended YAML settings for large runs:

```yaml
estimation:
  optimizer: bfgs
  multistart: 1
  multistart_seed: 12345
  multistart_scale: 0.25
  maxiter: 1000
  tolerance: 0.0001
  covariance: bfgs
  chunk_size: 10000
  workers: 1
  checkpoint_interval: 10
```

Supported optimizer values are `bfgs`, `lbfgsb`, `nelder-mead`, and `powell`.
BFGS is the default.

For difficult likelihood surfaces, increase `multistart` to run several seeded
local optimizations and select the best final log-likelihood.

Use `workers > 1` when your job has enough memory for several chunks in flight.
On some environments process creation may be restricted; `rpopit` falls back to
serial likelihood evaluation when multiprocessing cannot start.

## 5. Output Folder Structure

`rpopit` never modifies the raw CSV. Each fit creates a timestamped folder under
the output directory:

```text
rpopit_runs/
  rpopit_YYYYMMDD_HHMMSS_microseconds/
    rpopit.log
    model_spec.yaml
    run_metadata.yaml
    checkpoints/
      checkpoint_latest.npz
      checkpoint_latest.json
      checkpoint_iter_000010.npz
    coefficients.csv
    fit_statistics.csv
    convergence.csv
    timing.csv
    predicted_probabilities.csv
    marginal_effects.csv
    rpopit_results.xlsx
    rpopit_results.html
slurm_logs/
  rpopit_JOBID.out
  rpopit_JOBID.err
```

`timing.csv` is the first place to check for scalability. The most useful fields
are `average_objective_seconds`, `objective_calls`, and `optimization_seconds`.

Checkpoints store the current optimizer iteration, internal parameter vector,
current log-likelihood, and optimizer metadata. Set `checkpoint_interval: 0` or
pass `--checkpoint-interval 0` to disable checkpoint writing.
