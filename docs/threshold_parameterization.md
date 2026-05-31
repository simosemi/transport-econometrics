# Threshold Parameterization

`rpopit` estimates ordered probit cut-points through unconstrained internal
parameters. This lets BFGS search over all real numbers while the natural
thresholds remain strictly ordered.

## Internal Form

For three finite thresholds, the optimizer stores:

```text
theta_mu = [a1, a2, a3]
```

The natural thresholds are:

```text
mu1 = a1
mu2 = a1 + exp(a2)
mu3 = a1 + exp(a2) + exp(a3)
```

The implementation is in `src/rpopit/model.py`:

```python
def _unpack_thresholds(packed):
    thresholds[0] = packed[0]
    thresholds[1:] = packed[0] + np.cumsum(np.exp(packed[1:]))
```

The inverse packing used for starting values is:

```python
packed[0] = thresholds[0]
packed[1:] = np.log(np.diff(thresholds))
```

## Proof of Ordering

For every finite real number `x`:

```text
exp(x) > 0
```

Therefore:

```text
mu2 - mu1 = exp(a2) > 0
mu3 - mu2 = exp(a3) > 0
```

So:

```text
mu1 < mu2
mu2 < mu3
```

By transitivity:

```text
mu1 < mu2 < mu3
```

This holds for all possible optimizer values `a1`, `a2`, and `a3`.

## General Case

For `M` finite thresholds:

```text
mu1 = a1
mu_j = a1 + sum_{m=2}^{j} exp(a_m), j = 2, ..., M
```

Then:

```text
mu_j - mu_{j-1} = exp(a_j) > 0
```

so all thresholds are strictly increasing.
