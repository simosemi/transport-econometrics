# NLOGIT Comparison Benchmark

The NLOGIT benchmark uses a single simulated dataset so `rpopit` and NLOGIT are
estimated on identical observations.

## 1. Generate the Benchmark Dataset and rpopit Results

```bash
python -m rpopit.benchmark_nlogit --out validation_runs/nlogit_benchmark
```

This creates:

```text
validation_runs/nlogit_benchmark/
  simulated_nlogit_benchmark_data.csv
  simulated_truth.csv
  rpopit_benchmark_model.yaml
  nlogit_results_template.csv
  nlogit_run_instructions.md
  rpopit_runs/
    rpopit_YYYYMMDD_HHMMSS_microseconds/
      coefficients.csv
      fit_statistics.csv
      rpopit_results.xlsx
      rpopit_results.html
```

## 2. Run NLOGIT on the Same CSV

Use `simulated_nlogit_benchmark_data.csv` in NLOGIT. The benchmark model is:

- Ordered probit severity outcome `severity`.
- Fixed coefficient: `x`.
- Normally distributed random coefficient: `z`.
- Panel/group ID: `group`.
- Severity categories: `0, 1, 2`.
- Two finite thresholds.
- Halton simulation draws matching `rpopit_benchmark_model.yaml`.

Export or manually enter NLOGIT results into:

```text
nlogit_results_template.csv
```

Expected format:

```csv
component,variable,estimate,std_error
coefficient,x,0.70,0.05
random_mean,z,-0.80,0.07
random_sd,z,0.45,0.10
threshold,threshold[1],-0.35,0.05
threshold,threshold[2],0.85,0.06
log_likelihood,LL,-395.0,
```

Aliases are accepted for common component names such as `coef`, `fixed`,
`cutpoint`, `mean`, `sd`, and `loglikelihood`, but the canonical names above are
recommended.

## 3. Produce the Comparison Report

```bash
python -m rpopit.benchmark_nlogit \
  --out validation_runs/nlogit_benchmark \
  --nlogit-results validation_runs/nlogit_benchmark/nlogit_results_template.csv
```

The report exports:

```text
comparison_report/
  nlogit_rpopit_parameter_comparison.csv
  coefficients_comparison.csv
  thresholds_comparison.csv
  random_parameter_means_comparison.csv
  random_parameter_sds_comparison.csv
  log_likelihood_comparison.csv
  nlogit_rpopit_comparison.xlsx
  nlogit_rpopit_comparison.html
```

The HTML report highlights the largest absolute differences between NLOGIT and
`rpopit`.
