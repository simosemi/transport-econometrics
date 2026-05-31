"""Simulation utilities for examples and tests."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def simulate_ordered_probit_data(
    n_groups: int = 200,
    observations_per_group: int = 3,
    fixed_betas: Mapping[str, float] | None = None,
    random_means: Mapping[str, float] | None = None,
    random_sds: Mapping[str, float] | None = None,
    thresholds: Sequence[float] = (-0.4, 0.9),
    seed: int = 202405,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate grouped ordered probit data with normally random coefficients."""

    fixed_betas = dict(fixed_betas or {"x": 0.7})
    random_means = dict(random_means or {"z": -0.8})
    random_sds = dict(random_sds or {"z": 0.45})
    thresholds = np.asarray(thresholds, dtype=float)

    rng = np.random.default_rng(seed)
    n_obs = n_groups * observations_per_group
    group = np.repeat(np.arange(n_groups), observations_per_group)
    data: dict[str, np.ndarray] = {"group": group}

    eta = np.zeros(n_obs, dtype=float)
    for name, beta in fixed_betas.items():
        values = rng.normal(size=n_obs)
        data[name] = values
        eta += beta * values

    for name, mean in random_means.items():
        values = rng.normal(size=n_obs)
        data[name] = values
        sd = random_sds[name]
        group_coefficients = rng.normal(loc=mean, scale=sd, size=n_groups)
        eta += values * group_coefficients[group]

    latent = eta + rng.normal(size=n_obs)
    data["severity"] = np.searchsorted(thresholds, latent, side="right")
    frame = pd.DataFrame(data)
    truth = {
        "fixed_betas": fixed_betas,
        "random_means": random_means,
        "random_sds": random_sds,
        "thresholds": thresholds,
    }
    return frame, truth
