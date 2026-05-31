"""Predicted counts and average marginal effects for RPNB."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpnb.likelihood import linear_predictor, mu_from_eta, random_coefficients


def predicted_counts(
    beta_fixed: np.ndarray,
    random_means: np.ndarray,
    alpha: float,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    offset: np.ndarray,
    group_codes: np.ndarray,
    draws: np.ndarray,
    random_sds: np.ndarray | None = None,
    cholesky: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return expected counts under fixed or simulated random parameters."""

    del alpha
    random_means = np.asarray(random_means, dtype=float)
    if random_means.size == 0:
        eta = linear_predictor(beta_fixed, None, x_fixed, x_random, offset)
        mu = mu_from_eta(eta)
        return pd.DataFrame(
            {
                "linear_predictor": eta,
                "predicted_count": mu,
                "expected_crash_frequency": mu,
            }
        )

    coeffs = random_coefficients(draws, random_means, random_sds, cholesky)
    eta_draws = linear_predictor(
        beta_fixed, coeffs, x_fixed, x_random, offset, group_codes
    )
    mu_draws = mu_from_eta(eta_draws)
    mu = mu_draws.mean(axis=1)
    return pd.DataFrame(
        {
            "linear_predictor_mean": eta_draws.mean(axis=1),
            "predicted_count": mu,
            "expected_crash_frequency": mu,
            "prediction_draw_sd": mu_draws.std(axis=1, ddof=0),
        }
    )


def average_marginal_effects(
    beta_fixed: np.ndarray,
    random_means: np.ndarray,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    offset: np.ndarray,
    group_codes: np.ndarray,
    draws: np.ndarray,
    fixed_names: tuple[str, ...],
    random_names: tuple[str, ...],
    random_sds: np.ndarray | None = None,
    cholesky: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute average marginal effects on expected crash counts."""

    rows: list[dict[str, float | str]] = []
    random_means = np.asarray(random_means, dtype=float)
    beta_fixed = np.asarray(beta_fixed, dtype=float)

    if random_means.size == 0:
        eta = linear_predictor(beta_fixed, None, x_fixed, x_random, offset)
        mu = mu_from_eta(eta)
        for name, beta in zip(fixed_names, beta_fixed):
            if name == "Intercept":
                continue
            rows.append(
                {
                    "variable": name,
                    "component": "fixed",
                    "average_marginal_effect": float(np.mean(beta * mu)),
                    "average_semi_elasticity": float(beta),
                }
            )
        return pd.DataFrame(rows)

    coeffs = random_coefficients(draws, random_means, random_sds, cholesky)
    eta_draws = linear_predictor(
        beta_fixed, coeffs, x_fixed, x_random, offset, group_codes
    )
    mu_draws = mu_from_eta(eta_draws)
    for name, beta in zip(fixed_names, beta_fixed):
        if name == "Intercept":
            continue
        rows.append(
            {
                "variable": name,
                "component": "fixed",
                "average_marginal_effect": float(np.mean(beta * mu_draws)),
                "average_semi_elasticity": float(beta),
            }
        )

    obs_coeffs = coeffs[np.asarray(group_codes, dtype=int)]
    for idx, name in enumerate(random_names):
        draw_effects = obs_coeffs[:, :, idx] * mu_draws
        rows.append(
            {
                "variable": name,
                "component": "random",
                "average_marginal_effect": float(np.mean(draw_effects)),
                "average_semi_elasticity": float(np.mean(obs_coeffs[:, :, idx])),
            }
        )
    return pd.DataFrame(rows)
