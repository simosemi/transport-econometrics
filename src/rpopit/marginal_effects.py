"""Predicted probabilities and average marginal effects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from rpopit.likelihood import (
    ordered_probit_density_difference,
    ordered_probit_probabilities,
    random_coefficients,
)


def predicted_probabilities(
    beta_fixed: np.ndarray,
    random_means: np.ndarray,
    thresholds: np.ndarray,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    group_indices: Sequence[np.ndarray],
    draws: np.ndarray,
    categories: Sequence[Any],
    random_sds: np.ndarray | None = None,
    cholesky: np.ndarray | None = None,
) -> pd.DataFrame:
    """Average predicted probabilities over simulation draws."""

    n_obs = x_fixed.shape[0]
    n_categories = len(categories)
    q = np.asarray(random_means).size
    probabilities = np.empty((n_obs, n_categories), dtype=float)

    coeffs = (
        random_coefficients(draws, random_means, random_sds, cholesky) if q else None
    )
    for group_number, obs_idx in enumerate(group_indices):
        base_eta = x_fixed[obs_idx] @ beta_fixed if beta_fixed.size else np.zeros(obs_idx.size)
        if q:
            eta = base_eta[:, None] + x_random[obs_idx] @ coeffs[group_number].T
        else:
            eta = base_eta[:, None]
        probabilities[obs_idx] = ordered_probit_probabilities(eta, thresholds).mean(axis=1)

    return pd.DataFrame(
        probabilities,
        columns=[f"Pr({category})" for category in categories],
    )


def average_marginal_effects(
    beta_fixed: np.ndarray,
    random_means: np.ndarray,
    thresholds: np.ndarray,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    group_indices: Sequence[np.ndarray],
    draws: np.ndarray,
    fixed_names: Sequence[str],
    random_names: Sequence[str],
    categories: Sequence[Any],
    random_sds: np.ndarray | None = None,
    cholesky: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute average marginal effects for continuous variables."""

    q = np.asarray(random_means).size
    coeffs = (
        random_coefficients(draws, random_means, random_sds, cholesky) if q else None
    )
    totals: dict[tuple[str, str, Any], np.ndarray] = {}
    count = 0

    for group_number, obs_idx in enumerate(group_indices):
        base_eta = x_fixed[obs_idx] @ beta_fixed if beta_fixed.size else np.zeros(obs_idx.size)
        if q:
            eta = base_eta[:, None] + x_random[obs_idx] @ coeffs[group_number].T
            n_draws = eta.shape[1]
        else:
            eta = base_eta[:, None]
            n_draws = 1

        density_delta = ordered_probit_density_difference(eta, thresholds)
        count += obs_idx.size * n_draws

        for j, name in enumerate(fixed_names):
            effects = beta_fixed[j] * density_delta
            _accumulate_effects(totals, name, "fixed", categories, effects.sum(axis=(0, 1)))

        for j, name in enumerate(random_names):
            beta_draw = coeffs[group_number, :, j][None, :, None]
            effects = beta_draw * density_delta
            _accumulate_effects(totals, name, "random", categories, effects.sum(axis=(0, 1)))

    rows = []
    for (variable, parameter_type, category), value in totals.items():
        rows.append(
            {
                "variable": variable,
                "parameter_type": parameter_type,
                "category": category,
                "marginal_effect": float(value / count),
            }
        )
    return pd.DataFrame(rows)


def _accumulate_effects(
    totals: dict[tuple[str, str, Any], np.ndarray],
    variable: str,
    parameter_type: str,
    categories: Sequence[Any],
    values: np.ndarray,
) -> None:
    for category, value in zip(categories, values):
        key = (variable, parameter_type, category)
        totals[key] = totals.get(key, 0.0) + value
