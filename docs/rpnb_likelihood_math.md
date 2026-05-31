# RPNB Likelihood

RPNB uses the NB2 parameterization:

```text
E[y_i | mu_i] = mu_i
Var[y_i | mu_i] = mu_i + alpha * mu_i^2
```

The log mean is:

```text
eta_i = offset_i + x_i beta + z_i b_g
mu_i = exp(eta_i)
```

The offset is not estimated. Its coefficient is fixed at 1, so users should
provide a log-scale exposure variable such as `log_segment_length` or
`log_vehicle_miles`.

For dispersion `alpha > 0`, define `r = 1 / alpha`. The observation log PMF is:

```text
log f(y_i) =
  lgamma(y_i + r) - lgamma(r) - lgamma(y_i + 1)
  + r * [log(r) - log(r + mu_i)]
  + y_i * [log(mu_i) - log(r + mu_i)]
```

For random parameters, RPNB simulates group-level coefficients. With independent
normal random parameters:

```text
b_gr = mean + draw_gr * sd
```

With correlated random parameters:

```text
b_gr = mean + L draw_gr
Sigma = L L'
```

The panel/group contribution is:

```text
L_g = (1 / R) * sum_r product_{i in g} f(y_i | b_gr)
log L = sum_g log(L_g)
```

The implementation evaluates this on the log scale with log-sum-exp aggregation.
