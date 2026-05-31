# RPNB and NLOGIT Benchmark

This workflow creates a single simulated count dataset so RPNB and NLOGIT can be
estimated on identical observations.

## 1. Generate Data and Fit RPNB

```bash
python -m rpnb.benchmark_nlogit --out validation_runs/rpnb_nlogit_benchmark
```

This creates:

```text
validation_runs/rpnb_nlogit_benchmark/
  simulated_rpnb_nlogit_benchmark_data.csv
  simulated_truth.csv
  rpnb_benchmark_model.yaml
  nlogit_results_template.csv
  nlogit_predicted_means_template.csv
  nlogit_run_instructions.md
  rpnb_runs/
    rpnb_YYYYMMDD_HHMMSS_microseconds/
      coefficients.csv
      fit_statistics.csv
      predictions.csv
      marginal_effects.csv
      rpnb_results.xlsx
      rpnb_results.html
```

The simulated data include:

- `crashes`: non-negative count outcome.
- `log_exposure`: log offset added to `log(mu)` with coefficient fixed at 1.
- `x1`: fixed covariate.
- `z1`: normally distributed random-parameter covariate.
- `group`: panel/group ID.

## 2. Run NLOGIT on the Same CSV

Use `simulated_rpnb_nlogit_benchmark_data.csv` in NLOGIT.

Benchmark model:

- Negative binomial count model with NB2 dispersion.
- Log link.
- Offset: `log_exposure`, coefficient fixed at 1.
- Fixed coefficients: intercept and `x1`.
- Random parameter: normally distributed `z1`.
- Panel/group likelihood: `group`.
- Halton draws matching `rpnb_benchmark_model.yaml`.

The generated `nlogit_run_instructions.md` includes a command sketch to adapt
to your NLOGIT version. The critical requirement is that `log_exposure` enters
as an offset, not as an estimated RHS coefficient.

## 3. Fill NLOGIT Templates

Fill:

```text
nlogit_results_template.csv
nlogit_predicted_means_template.csv
```

Expected parameter rows:

```csv
component,variable,estimate,std_error
coefficient,Intercept,...,...
coefficient,x1,...,...
random_mean,z1,...,...
random_sd,z1,...,...
alpha,alpha,...,...
offset_coefficient,log_exposure,1,
log_likelihood,LL,...,
```

Expected prediction rows:

```csv
row_index,predicted_mean
0,...
1,...
2,...
```

Aliases are accepted for common names such as `coef`, `fixed`, `mean`, `sd`,
`dispersion`, `offset`, `mu`, and `loglikelihood`, but the canonical names above
are recommended.

## 4. Produce the Comparison Report

```bash
python -m rpnb.benchmark_nlogit \
  --out validation_runs/rpnb_nlogit_benchmark \
  --nlogit-results validation_runs/rpnb_nlogit_benchmark/nlogit_results_template.csv \
  --nlogit-predictions validation_runs/rpnb_nlogit_benchmark/nlogit_predicted_means_template.csv
```

The report exports:

```text
comparison_report/
  nlogit_rpnb_comparison.csv
  coefficients_comparison.csv
  random_parameter_means_comparison.csv
  random_parameter_sds_comparison.csv
  alpha_comparison.csv
  offset_handling_comparison.csv
  predicted_means_comparison.csv
  log_likelihood_comparison.csv
  nlogit_rpnb_comparison.xlsx
  nlogit_rpnb_comparison.html
```

The comparison checks:

- Coefficients.
- Random parameter mean and SD.
- `alpha` / dispersion.
- Offset handling through a fixed coefficient of 1.
- Observation-level predicted means.
- Simulated log likelihood.
