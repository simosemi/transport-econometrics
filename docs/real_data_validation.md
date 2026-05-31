# Real Crash Severity Validation Suite

This suite is designed for HSIS-style crash severity CSVs. The repository does
not include raw HSIS data; point the scripts at your local, authorized extract.
The scripts never modify the raw CSV.

## Files

- `examples/hsis_model.yaml`: example Random Parameters Ordered Probit spec.
- `examples/hsis_schema.yaml`: machine-readable schema used by validation.
- `examples/hsis_csv_schema.csv`: tabular CSV schema for review/editing.
- `src/rpopit/validate_hsis_data.py`: schema and missing-value validation CLI.
- `src/rpopit/model_comparison_report.py`: Ordered Probit vs RPOPIT report CLI.
- `validation_suite/*.py`: compatibility wrappers for older path-based commands.
- `benchmarks/benchmark_hsis_large_sample.py`: large-sample likelihood benchmark.

## Validate Data

```bash
python -m rpopit.validate_hsis_data \
  --data /path/to/hsis_crashes.csv \
  --schema examples/hsis_schema.yaml \
  --out validation_runs/hsis_validation \
  --missing drop
```

After `python -m pip install -e .`, the console script form is also available:

```bash
validate-hsis \
  --data /path/to/hsis_crashes.csv \
  --schema examples/hsis_schema.yaml \
  --out validation_runs/hsis_validation
```

Outputs:

```text
validation_runs/hsis_validation/
  validation_summary.csv
  validation_issues.csv
  missing_summary.csv
```

Use `--missing error` to fail if required model columns contain missing values.
Use `--write-cleaned` only when you want a separate cleaned copy in the output
folder; the raw input file is never changed.

## Model Comparison Report

```bash
python -m rpopit.model_comparison_report \
  --data /path/to/hsis_crashes.csv \
  --schema examples/hsis_schema.yaml \
  --spec examples/hsis_model.yaml \
  --out validation_runs/hsis_model_comparison
```

Console script form:

```bash
compare-models \
  --data /path/to/hsis_crashes.csv \
  --schema examples/hsis_schema.yaml \
  --spec examples/hsis_model.yaml \
  --out validation_runs/hsis_model_comparison
```

The report fits:

- Ordered Probit: all fixed and random-candidate variables enter as fixed.
- Random Parameters Ordered Probit: random variables use the YAML random specs.
- Null Ordered Probit: thresholds only, used for McFadden pseudo-R2.

Exports:

```text
validation_runs/hsis_model_comparison/
  model_comparison_metrics.csv
  model_comparison_coefficients.csv
  random_parameter_sds.csv
  model_comparison_report.xlsx
  model_comparison_report.html
  validation/
    validation_summary.csv
    validation_issues.csv
    missing_summary.csv
```

Metrics include log-likelihood, AIC, BIC, and McFadden pseudo-R2:

```text
McFadden pseudo-R2 = 1 - LL_model / LL_null
```

## Large-Sample Benchmark

```bash
python benchmarks/benchmark_hsis_large_sample.py \
  --data /path/to/hsis_crashes.csv \
  --schema examples/hsis_schema.yaml \
  --spec examples/hsis_model.yaml \
  --sample-sizes 100000 500000 1000000 \
  --out validation_runs/hsis_benchmark.csv
```

The benchmark times one simulated likelihood evaluation at each sample size and
reports an estimated 100-evaluation optimization cost.

## Threshold Ordering

The exact threshold parameterization and proof that `mu1 < mu2 < mu3` is
guaranteed are in `docs/threshold_parameterization.md`.
