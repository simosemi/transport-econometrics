"""Simulate negative binomial crash-frequency data for testing and examples."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def simulate_negative_binomial_data(
    n_groups: int = 100,
    observations_per_group: int = 3,
    intercept: float = -0.5,
    fixed_betas: Mapping[str, float] | None = None,
    random_means: Mapping[str, float] | None = None,
    random_sds: Mapping[str, float] | None = None,
    alpha: float = 0.6,
    seed: int | None = 12345,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate NB2 count data with a log offset and group-level random parameters."""

    if n_groups < 1:
        raise ValueError("n_groups must be at least 1.")
    if observations_per_group < 1:
        raise ValueError("observations_per_group must be at least 1.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    fixed_betas = dict(fixed_betas or {"x1": 0.35})
    random_means = dict(random_means or {})
    random_sds = dict(random_sds or {name: 0.3 for name in random_means})
    for name in random_means:
        if random_sds.get(name, 0.0) <= 0:
            raise ValueError(f"random_sds[{name!r}] must be positive.")

    rng = np.random.default_rng(seed)
    n = n_groups * observations_per_group
    group = np.repeat(np.arange(n_groups), observations_per_group)
    data: dict[str, np.ndarray] = {"group": group}
    exposure = rng.uniform(0.5, 2.5, size=n)
    data["log_exposure"] = np.log(exposure)

    eta = data["log_exposure"] + intercept
    for name, beta in fixed_betas.items():
        values = rng.normal(size=n)
        data[name] = values
        eta = eta + beta * values

    group_coefficients: dict[str, np.ndarray] = {}
    for name, mean in random_means.items():
        values = rng.normal(size=n)
        coeffs = rng.normal(mean, random_sds[name], size=n_groups)
        data[name] = values
        group_coefficients[name] = coeffs
        eta = eta + coeffs[group] * values

    mu = np.exp(np.clip(eta, -700.0, 700.0))
    gamma_shape = 1.0 / alpha
    gamma_scale = alpha * mu
    poisson_rate = rng.gamma(shape=gamma_shape, scale=gamma_scale)
    data["crashes"] = rng.poisson(poisson_rate)

    truth = {
        "intercept": intercept,
        "fixed_betas": fixed_betas,
        "random_means": random_means,
        "random_sds": random_sds,
        "group_coefficients": group_coefficients,
        "alpha": alpha,
    }
    return pd.DataFrame(data), truth
