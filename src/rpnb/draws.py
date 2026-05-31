"""Simulation draw generation for random parameters."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.stats import qmc

VALID_DRAW_TYPES = {"pseudo", "random", "halton", "sobol"}


def generate_draws(
    n_groups: int,
    n_draws: int,
    n_random: int,
    draw_type: str = "halton",
    seed: int | None = None,
    scramble: bool = True,
) -> np.ndarray:
    """Return standard-normal draws with shape ``(n_groups, n_draws, n_random)``."""

    if n_groups < 1:
        raise ValueError("n_groups must be at least 1.")
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")
    if n_random < 0:
        raise ValueError("n_random cannot be negative.")
    if n_random == 0:
        return np.zeros((n_groups, 1, 0), dtype=float)

    kind = draw_type.lower()
    if kind not in VALID_DRAW_TYPES:
        raise ValueError(
            f"Unknown draw_type {draw_type!r}. Expected one of {sorted(VALID_DRAW_TYPES)}."
        )

    if kind in {"pseudo", "random"}:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_groups, n_draws, n_random))

    n_total = n_groups * n_draws
    if kind == "halton":
        engine = qmc.Halton(d=n_random, scramble=scramble, seed=seed)
        uniforms = engine.random(n_total)
    else:
        engine = qmc.Sobol(d=n_random, scramble=scramble, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            uniforms = engine.random(n_total)

    eps = np.finfo(float).eps
    uniforms = np.clip(uniforms, eps, 1.0 - eps)
    normals = stats.norm.ppf(uniforms)
    return normals.reshape(n_groups, n_draws, n_random)
