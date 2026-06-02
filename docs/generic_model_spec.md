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
