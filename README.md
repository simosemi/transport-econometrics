# RPNB

`RPNB` is a research-grade Python package for Random Parameters Negative
Binomial crash-frequency models estimated by simulated maximum likelihood.

The model is NB2 with a log link and a log-offset whose coefficient is fixed at
1:

```text
y_i ~ NB(mu_i, alpha)
log(mu_i) = offset_i + X_fixed beta + Z_random b_g
b_g = mean + draw_g * sd
Var(y_i | mu_i) = mu_i + alpha * mu_i^2
```

Random parameters are normally distributed. Independent random parameters use
estimated standard deviations. Correlated random parameters use an estimated
Cholesky factor and report natural standard deviations and correlations.

## Features

- CSV input with YAML model specifications.
- Dependent count variable validation.
- Offset variable added to `log(mu)` with coefficient fixed at 1.
- Fixed and normally distributed random covariates.
- Optional group or panel likelihood.
- Pseudo-random, Halton, and Sobol simulation draws.
- Independent or correlated random parameters.
- Positive dispersion parameter `alpha`.
- Coefficients, random-parameter means, random-parameter SDs, correlations,
  `alpha`, log likelihood, AIC, BIC, standard errors, z-values, p-values, and
  incidence rate ratios.
- Predicted counts and expected crash frequencies.
- Average marginal effects on expected crash counts.
- CSV, Excel, and HTML exports.
- HPCC SLURM scripts.
- Simulated-data tests and fixed-only validation against statsmodels.
- Raw data files are read only; the package works on an internal copy.

## Installation

```powershell
python -m pip install -e .
```

For development validation:

```powershell
python -m pip install -e ".[dev]"
pytest
```

## YAML Model Specification

The offset must already be on the log scale, for example `log_exposure` or
`log_segment_length`. RPNB includes an intercept by default; set
`intercept: false` to disable it.

```yaml
model:
  dependent: crashes
  offset: log_exposure
  fixed:
    - speed_limit
    - wet_road
  random:
    truck_share:
      distribution: normal
      start_mean: 0.0
      start_sd: 0.25
  group_id: segment_id
  intercept: true
  correlated_random_parameters: false
  missing: drop

simulation:
  draws: 500
  draw_type: halton
  seed: 12345

estimation:
  maxiter: 1000
  tolerance: 0.00001
  covariance: bfgs
  chunk_size: 10000
  workers: 1
  start_alpha: 0.5

output:
  directory: runs
```

## Command Line Use

```powershell
rpnb fit --data crashes.csv --spec model.yaml --out runs
```

Each run creates a timestamped directory containing:

- `rpnb.log`
- `model_spec.yaml`
- `coefficients.csv`
- `fit_statistics.csv`
- `convergence.csv`
- `timing.csv`
- `predictions.csv`
- `marginal_effects.csv`
- `rpnb_results.xlsx`
- `rpnb_results.html`

## Python Use

```python
from rpnb import RandomParametersNegativeBinomial

model = RandomParametersNegativeBinomial(
    dependent="crashes",
    offset="log_exposure",
    fixed=["speed_limit", "wet_road"],
    random=["truck_share"],
    group_id="segment_id",
    draws=500,
    draw_type="sobol",
    seed=2026,
)

results = model.fit("crashes.csv", export=True)
print(results.parameter_table)
print(results.predictions.head())
print(results.marginal_effects)
```

## Examples

```powershell
python examples/run_rpnb_example.py
```

The example simulates data, writes a run-specific CSV, fits RPNB, and exports
results under `runs/`.

## HPCC

```bash
bash hpcc/install_rpnb_env.sh
sbatch hpcc/run_rpnb.sbatch /path/to/crashes.csv /path/to/model.yaml /path/to/rpnb_runs
```

See [docs/rpnb_hpcc.md](docs/rpnb_hpcc.md) for SLURM notes and resource
guidance.

## Validation

The test suite includes:

- Known-parameter simulated-data recovery for fixed and random parameters.
- Likelihood checks against direct NB2 calculations.
- Fixed-only negative binomial validation against statsmodels.

```powershell
pytest tests/test_rpnb_*.py
```
