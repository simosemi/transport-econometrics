# Generic Model Specifications

RPNB and rpopit can be configured with arbitrary dataset column names. The
packages do not hard-code HSIS variables.

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

  group_id: UniqueID
  correlated_random_parameters: false
  missing: drop
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

  random:
    continuous:
      speed_std:
        distribution: normal
        start_mean: 0.0
        start_sd: 0.3

  group_id: UniqueID
  categories: [0, 1, 2, 3]
  correlated_random_parameters: false
  missing: drop
```

## Validation Rules

- Every declared variable must exist in the input CSV.
- A variable may appear in only one model role.
- Fixed categorical references must exist in the data after missing-row handling.
- Fixed categorical variables are dummy-coded automatically.
- The declared reference category is dropped.
- Dummy names use `variable_value`, for example `Hour_1` or `Year_2018`.
- Category ordering is deterministic; categories are sorted before dummy columns
  are generated.
- Raw CSV files are read only and are never modified.

## RPNB Missing Data Handling

For RPNB, missing-data checks are applied to every column required for
estimation:

- dependent count variable
- offset variable
- fixed continuous variables
- fixed categorical variables
- random continuous variables
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
- fixed categorical reference categories
- generated dummy variable names, such as `Hour_1` and `Year_2018`

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
