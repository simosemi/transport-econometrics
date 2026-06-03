# Generic Model Specifications

RPNB and rpopit can be configured with arbitrary dataset column names. The
packages do not hard-code HSIS variables.

See `examples/generic_rpnb_model.yaml` and `examples/generic_rpopit_model.yaml`
for standalone generic specifications using both continuous and categorical
random parameters.

## RPNB Count Model

```yaml
model:
  dependent: crashes
  offset: log_exposure

  fixed:
    continuous:
      - speed_mean
      - Log_Hourly_volume

    categorical:
      Hour:
        reference: 0
      Year:
        reference: 2017
      speed_std_cat:
        reference: "<5"

  derived_categorical:
    speed_std_cat:
      source: speed_std
      bins:
        - upper: 5
          label: "<5"
        - upper: 7
          label: "5-7"
        - label: ">=7"
    speed_std_quartile:
      source: speed_std
      method: quantile
      bins: 4

  random:
    continuous:
      speed_std:
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3
      RtPvdShldrWidth:
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3
      TruckPercent:
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3
    categorical:
      Interstate:
        reference: 1
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3
      speed_std_quartile:
        reference: Q1
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3

  group_id: UniqueID
  correlated_random_parameters: false
  missing: drop

estimation:
  optimizer: bfgs
  multistart: 1
  multistart_seed: 12345
  multistart_scale: 0.25
```

## rpopit Ordered Probit Model

```yaml
model:
  dependent: severity

  fixed:
    continuous:
      - speed_mean
      - Log_Hourly_volume

    categorical:
      Hour:
        reference: 0
      Year:
        reference: 2017
      speed_std_cat:
        reference: "<5"

  derived_categorical:
    speed_std_cat:
      source: speed_std
      bins:
        - upper: 5
          label: "<5"
        - upper: 7
          label: "5-7"
        - label: ">=7"
    speed_std_quartile:
      source: speed_std
      method: quantile
      bins: 4

  random:
    continuous:
      speed_std:
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3
    categorical:
      Interstate:
        reference: 1
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3
      speed_std_quartile:
        reference: Q1
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3

  group_id: UniqueID
  categories: [0, 1, 2, 3]
  correlated_random_parameters: false
  missing: drop

estimation:
  optimizer: bfgs
  multistart: 1
  multistart_seed: 12345
  multistart_scale: 0.25
```

## Validation Rules

- Every declared variable must exist in the input CSV.
- A variable may appear in only one model role.
- RPNB `offset` is optional. Omit it or set `offset: null` to use a zero offset.
  If an offset is supplied, that same variable cannot be listed as fixed or
  random.
- `fixed.continuous` declares non-random fixed effects.
- `fixed.categorical` declares non-random categorical/factor effects.
- `random.continuous` declares random continuous parameters. Each variable
  estimates both `beta_random_mean[variable]` and `beta_random_sd[variable]`.
- `random.categorical` declares random categorical/factor effects. Each source
  variable is dummy-coded, the reference category is dropped, and every
  generated non-reference dummy estimates both `beta_random_mean[variable_value]`
  and `beta_random_sd[variable_value]`.
- Do not duplicate a random variable under `fixed`. Its average effect is
  already estimated as the random parameter mean.
- In RPNB, do not include the offset as fixed or random. The offset coefficient
  is fixed at 1 by definition.
- Fixed and random categorical references must exist in the data after
  missing-row handling.
- Fixed and random categorical variables are dummy-coded automatically.
- `derived_categorical` creates factor variables from numeric source columns
  inside the working estimation data. The raw CSV is not modified.
- Explicit derived bins use increasing `upper` values and may end with one
  open-ended bin that has only a `label`.
- Quantile derived bins use `method: quantile` and `bins: 4` for quartiles or
  `bins: 5` for quintiles. Generated labels are `Q1`, `Q2`, and so on.
- The declared reference category is dropped.
- Dummy names use `variable_value`, for example `Hour_1` or `Year_2018`.
- Category ordering is deterministic; categories are sorted before dummy columns
  are generated.
- For a random factor `Interstate` with `reference: 1`, a binary input with
  observed values `0` and `1` generates `Interstate_0` only. Output includes
  `beta_random_mean[Interstate_0]`, the average effect relative to
  `Interstate=1`, and `beta_random_sd[Interstate_0]`, the heterogeneity in that
  relative effect.
- Raw CSV files are read only and are never modified.

## RPNB Missing Data Handling

For RPNB, missing-data checks are applied to every column required for
estimation:

- dependent count variable
- offset variable
- fixed continuous variables
- fixed categorical variables
- random continuous variables
- random categorical variables
- group ID, when declared

With `missing: drop`, RPNB removes rows with any of the following in those
columns:

- `NaN` / null values
- blank strings, including whitespace-only strings
- non-finite numeric values such as `inf` and `-inf`
- values in numeric model columns that cannot be converted to numbers

With any other `missing` value, RPNB raises an error instead of dropping rows.
The fit report exports sample accounting fields:

- `missing_checked_columns`
- `n_rows_original`
- `n_rows_removed_missing`
- `n_rows_final_estimation_sample`
- `n_observations`

## RPNB Preprocessing Summary

Before likelihood optimization, RPNB builds a preprocessing summary from the
declared model columns. Numeric statistics and categorical frequency tables use
the final estimation sample after missing-row handling. The `number_missing`
field counts invalid values in the raw input columns before rows are dropped.

The exported files are:

- `preprocessing_summary.csv`
- `preprocessing_summary.xlsx`
- `preprocessing_summary.html`

The summary includes:

- variable name and model role
- mean, standard deviation, minimum, and maximum for numeric model variables
- number of missing or invalid raw values
- number of unique values in the estimation sample
- categorical frequency tables
- fixed and random categorical reference categories
- generated dummy variable names, such as `Hour_1` and `Year_2018`

## Optimizer Diagnostics

RPNB and rpopit export optimizer diagnostics in `convergence.csv`, the
`convergence` Excel sheet, and the HTML report convergence section.

Supported `estimation.optimizer` values are:

- `bfgs`
- `lbfgsb`
- `nelder-mead`
- `powell`

BFGS is the default when `optimizer` is omitted.

Set `multistart` to an integer greater than 1 to run multiple local
optimizations. The first start is the supplied/default vector; remaining starts
are seeded perturbations around that vector using `multistart_seed` and
`multistart_scale`.

The diagnostics include:

- optimizer method
- convergence code and convergence message
- convergence quality (`converged_clean`, `near_converged`, `usable_warning`,
  or `not_converged`)
- gradient norm
- Hessian condition number
- largest and smallest absolute parameter magnitudes
- normalized termination reason

The termination reason is one of:

- `convergence`
- `max_iterations`
- `precision_loss`
- `singular_hessian`
- `line_search_failure`
- `other`

Boolean columns also report whether the run terminated due to each named reason.

When multi-start is enabled, RPNB and rpopit export:

- `multistart_summary.csv`, `multistart_summary.xlsx`, and
  `multistart_summary.html`, with starting log-likelihood, final
  log-likelihood, AIC, BIC, convergence status, convergence quality, gradient
  norm, optimizer, best-start flag, and starting/final parameter vectors
- `multistart_local_solutions.csv`, with natural parameter estimates for every
  local solution
- RPNB fit statistics additionally report `n_local_optima` and
  `multiple_local_optima_found`, based on distinct final log-likelihood values
  across starts
- `nlogit_style_report.txt`, a plain-text report with NLOGIT-style model
  metrics, random parameter mean and SD sections, dispersion section when
  applicable, and significance stars

For independent RPNB random parameters, `random_parameter_tests.csv` is also
exported. For each random parameter, RPNB fits a restricted model with that
parameter's SD fixed near zero and reports:

- full and restricted log-likelihood
- LR statistic and p-value
- delta LL, delta AIC, and delta BIC
- recommendation: `Keep Random` or `Treat as Fixed`

RPNB run directories can be compared after estimation with:

```powershell
python -m rpnb.compare_runs --runs run1 run2 run3 --out comparison_report
```

The comparison report exports CSV, Excel, and HTML files for LL, AIC, BIC,
alpha, convergence quality, random-SD significance, parameter count, random
parameter means, and random parameter SDs. Models are ranked automatically by
AIC, with LL and BIC ranks also included.

RPNB can also generate an NLOGIT audit note:

```powershell
python -m rpnb.audit_nlogit --out nlogit_audit_report.md
```

The audit covers the NB2 likelihood formulation, alpha parameterization, random
coefficient generation, Halton draws, random categorical variables, offset
handling, panel likelihood, and differences to check when validating against
NLOGIT.

Backward-compatible shorthand remains available for continuous-only models:

```yaml
model:
  dependent: crashes
  offset: log_exposure
  fixed: [x1, x2]
  random:
    z1:
      distribution: normal
      start_mean: 0.0
      start_sd: 0.3
```
