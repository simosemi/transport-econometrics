# rpopit

`rpopit` is a research-oriented Random Parameters Ordered Probit estimator inspired by
NLOGIT-style crash severity workflows. It estimates ordered probit latent-variable
models with normally distributed random coefficients using simulated maximum
likelihood.

## Features

- CSV input and YAML model specification.
- Fixed and random covariates.
- Normal random parameter distributions.
- Independent or correlated random parameters through a Cholesky factor.
- Ordered threshold parameterization that enforces increasing cut-points.
- Optional panel/group likelihood by corridor, route, segment, year, or any group ID.
- Pseudo-random, Halton, and Sobol simulation draws.
- Stable ordered-probit log probabilities and log-sum-exp likelihood aggregation.
- BFGS estimation through `scipy.optimize.minimize`.
- Coefficient means, random parameter standard deviations, thresholds, standard
  errors, z-values, p-values, log likelihood, AIC, BIC, and convergence diagnostics.
- Predicted probabilities and average marginal effects.
- CSV, Excel, and HTML result export.
- Timestamped run directories with logs. Raw data files are read but never modified.

## Installation

```powershell
python -m pip install -e .
```

## YAML model specification

```yaml
model:
  dependent: severity
  fixed:
    - speed_limit
    - wet_road
  random:
    truck_share:
      distribution: normal
      start_mean: 0.0
      start_sd: 0.4
  group_id: corridor_id
  categories: [0, 1, 2, 3]
  correlated_random_parameters: false
  missing: drop

simulation:
  draws: 500
  draw_type: halton
  chunk_size: 10000
  workers: 1
  seed: 12345

estimation:
  maxiter: 1000
  tolerance: 0.0001
  covariance: bfgs

output:
  directory: runs
```

Do not include a constant column unless you have fixed thresholds externally. With
free ordered thresholds, an intercept is not separately identified.

## Command line use

```powershell
rpopit fit --data crashes.csv --spec model.yaml --out runs
```

Each run creates a timestamped directory containing `rpopit.log`, the resolved
model specification, and exported tables:

- `coefficients.csv`
- `fit_statistics.csv`
- `convergence.csv`
- `timing.csv`
- `predicted_probabilities.csv`
- `marginal_effects.csv`
- `rpopit_results.xlsx`
- `rpopit_results.html`

## Scalability controls

Random-parameter likelihood evaluation is vectorized within chunks. Use
`chunk_size` to control the maximum number of observations materialized in each
likelihood block. Smaller chunks reduce memory pressure; larger chunks reduce
chunk overhead. Use `workers` to request process-based chunk evaluation where
the operating system permits it.

```yaml
estimation:
  chunk_size: 10000
  workers: 1
```

For profiling one likelihood evaluation on synthetic data:

```powershell
python benchmarks/profile_likelihood.py --observations 100000 --draws 200 --chunk-size 10000 --profile
```

### Benchmark Estimates

These estimates are for one serial CPU job with group size 10, 3 severity
categories, `chunk_size=10000`, and Halton draws. The 1-random-parameter rows
are anchored to local measurements: 10k observations took about 0.40 seconds,
100k took about 3.7 seconds, and 500k took about 19.8 seconds for one likelihood
evaluation. Actual optimization time depends on the number of BFGS objective
evaluations; 100 evaluations is a convenient planning baseline.

| Observations | Draws | Random parameters | Est. LL eval | Est. 100 evals |
|---:|---:|---:|---:|---:|
| 10,000 | 200 | 1 | 0.4 sec | 0.7 min |
| 10,000 | 500 | 2 | 1.2 sec | 1.9 min |
| 10,000 | 500 | 4 | 1.5 sec | 2.4 min |
| 100,000 | 200 | 1 | 3.7 sec | 6.2 min |
| 100,000 | 200 | 2 | 4.3 sec | 7.1 min |
| 100,000 | 500 | 2 | 10.6 sec | 17.8 min |
| 100,000 | 500 | 4 | 13.4 sec | 22.4 min |
| 500,000 | 200 | 1 | 19.8 sec | 33.0 min |
| 500,000 | 200 | 2 | 22.8 sec | 38.0 min |
| 500,000 | 500 | 2 | 56.9 sec | 1.6 hr |
| 500,000 | 500 | 4 | 71.8 sec | 2.0 hr |
| 1,000,000 | 200 | 1 | 39.6 sec | 1.1 hr |
| 1,000,000 | 200 | 2 | 45.5 sec | 1.3 hr |
| 1,000,000 | 500 | 2 | 113.9 sec | 3.2 hr |
| 1,000,000 | 500 | 4 | 143.6 sec | 4.0 hr |

For 150 objective evaluations, multiply the final column by about 1.5. More
severity categories, dense correlated random parameters, expensive standard
error settings, and post-estimation probability output can add time and memory.

## MSU ICER HPCC

The `hpcc/` directory contains SLURM-ready scripts:

```bash
bash hpcc/install_env.sh
sbatch hpcc/run_rpopit.sbatch /path/to/crashes.csv /path/to/model.yaml /path/to/rpopit_runs
```

The sbatch script runs the CLI form:

```bash
python -m rpopit.cli fit --data /path/to/crashes.csv --spec /path/to/model.yaml --out /path/to/rpopit_runs
```

See [docs/icer_hpcc.md](docs/icer_hpcc.md) for environment activation,
recommended CPU/memory settings for 100k, 500k, and 1M observations, and the
expected output folder structure.

## Python use

```python
from rpopit import RandomParametersOrderedProbit

model = RandomParametersOrderedProbit(
    dependent="severity",
    fixed=["speed_limit", "wet_road"],
    random=["truck_share"],
    group_id="corridor_id",
    draws=500,
    draw_type="sobol",
    seed=2026,
)

results = model.fit("crashes.csv", export=True)
print(results.parameter_table)
print(results.fit_statistics)
```

## Simulated example

```powershell
python examples/run_example.py
```

## Real Data Validation

HSIS-style validation examples and scripts are included:

```powershell
python -m rpopit.validate_hsis_data --help
python -m rpopit.model_comparison_report --help
validate-hsis --help
compare-models --help
```

Example real-data commands:

```powershell
python -m rpopit.validate_hsis_data --data crashes.csv --schema examples/hsis_schema.yaml --out validation_runs/hsis_validation
python -m rpopit.model_comparison_report --data crashes.csv --schema examples/hsis_schema.yaml --spec examples/hsis_model.yaml --out validation_runs/hsis_model_comparison
```

The model comparison report exports LL, AIC, BIC, McFadden pseudo-R2,
coefficient tables, and random-parameter standard deviations for Ordered Probit
and Random Parameters Ordered Probit. See
[docs/real_data_validation.md](docs/real_data_validation.md).

The exact threshold parameterization and proof that `mu1 < mu2 < mu3` is
guaranteed are in
[docs/threshold_parameterization.md](docs/threshold_parameterization.md).

## NLOGIT Benchmark

Generate one simulated benchmark dataset and fit `rpopit`:

```powershell
python -m rpopit.benchmark_nlogit --out validation_runs/nlogit_benchmark
```

For four severity categories `[0, 1, 2, 3]` and three thresholds:

```powershell
python -m rpopit.benchmark_nlogit --out validation_runs/nlogit_benchmark_4cat --categories 4
```

After running NLOGIT on the generated CSV and filling
`nlogit_results_template.csv`, create the side-by-side report:

```powershell
python -m rpopit.benchmark_nlogit --out validation_runs/nlogit_benchmark --nlogit-results validation_runs/nlogit_benchmark/nlogit_results_template.csv
```

The report exports coefficients, thresholds, random-parameter means,
random-parameter SDs, log-likelihood, and highlighted differences. See
[docs/nlogit_benchmark.md](docs/nlogit_benchmark.md).

## Tests

```powershell
pytest
```

The test suite includes probability checks, draw checks, and a simulated-data
recovery test with known fixed and random parameters.
