# Likelihood Math

This note describes the likelihood implemented by `rpopit` for Random
Parameters Ordered Probit models.

## Latent Variable Model

For crash observation `i` in group `g`, the ordered probit model starts from an
unobserved continuous severity index:

```text
y*_ig = x_ig beta + z_ig b_g + epsilon_ig
epsilon_ig ~ N(0, 1)
```

`x_ig` contains fixed-coefficient covariates and `z_ig` contains covariates with
random coefficients. The observed crash severity category is determined by
where the latent index falls relative to ordered cut-points:

```text
y_ig = 0  if y*_ig <= mu_1
y_ig = 1  if mu_1 < y*_ig <= mu_2
...
y_ig = J  if mu_J < y*_ig
```

There is no separately identified intercept when all thresholds are freely
estimated.

## Ordered Probit Category Probabilities

Let `eta_ig = x_ig beta + z_ig b_g`. With finite thresholds
`mu_1, ..., mu_J`, define `mu_0 = -inf` and `mu_{J+1} = inf`. The probability of
category `j` is:

```text
Pr(y_ig = j | b_g) =
  Phi(mu_{j+1} - eta_ig) - Phi(mu_j - eta_ig)
```

where `Phi(.)` is the standard normal CDF. The first and last categories use the
same expression with infinite bounds:

```text
Pr(y_ig = 0 | b_g) = Phi(mu_1 - eta_ig)
Pr(y_ig = J | b_g) = 1 - Phi(mu_J - eta_ig)
```

## Random Parameter Distributions

For independent random parameters, `rpopit` uses normally distributed
coefficients:

```text
b_gk = alpha_k + sigma_k v_gk
v_gk ~ N(0, 1)
```

For correlated random parameters, the vector form is:

```text
b_g = alpha + L v_g
v_g ~ N(0, I)
```

`L` is a lower-triangular Cholesky factor. The covariance matrix of the random
parameters is `L L'`.

## Simulated Likelihood

For a single non-panel observation, the unconditional probability integrates
over the random coefficients:

```text
Pr(y_i = j) = integral Pr(y_i = j | b) f(b) db
```

Because this integral is generally not closed-form, `rpopit` approximates it
with simulation draws:

```text
Pr(y_i = j) approx (1 / R) sum_r Pr(y_i = j | b_ir)
```

The simulated log likelihood is the sum of log simulated probabilities over
observations.

## Panel and Group Likelihood

When a `group_id` is supplied, random coefficients are drawn once per group and
shared across that group's observations. For group `g` with observation set
`I_g`, the conditional likelihood for draw `r` is:

```text
L_gr = product_{i in I_g} Pr(y_ig | b_gr)
```

The simulated group likelihood is:

```text
L_g approx (1 / R) sum_r L_gr
```

The full simulated log likelihood is:

```text
log L = sum_g log L_g
```

This is the panel likelihood used for crash corridors, routes, segments, years,
or any repeated-observation grouping.

## Threshold Parameterization

The optimizer works with unconstrained internal threshold parameters. Natural
thresholds are recovered as:

```text
mu_1 = theta_1
mu_2 = theta_1 + exp(theta_2)
mu_3 = theta_1 + exp(theta_2) + exp(theta_3)
...
```

The exponential increments guarantee:

```text
mu_1 < mu_2 < mu_3 < ...
```

This avoids invalid ordered probit probabilities during optimization.
See `docs/threshold_parameterization.md` for the exact internal mapping and a
short proof for the three-threshold case.

## Numerical Stability

`rpopit` avoids unstable probability and likelihood calculations in several
places:

- Ordered probit probabilities are computed in log space with `log_ndtr`, which
  is more stable than taking logs of normal CDF differences directly.
- Interior category probabilities use stable log-difference calculations.
- The upper-tail form is used when it is numerically safer than subtracting two
  nearly equal lower-tail CDF values.
- Simulated group likelihoods are aggregated with log-sum-exp:

```text
log((1 / R) sum_r exp(ell_gr))
  = logsumexp_r(ell_gr) - log(R)
```

where `ell_gr` is the conditional group log likelihood for draw `r`.
- A very small probability floor protects downstream logs from exact zero while
  leaving ordinary probabilities unchanged.
