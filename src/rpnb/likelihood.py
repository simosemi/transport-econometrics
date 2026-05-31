"""Negative binomial probabilities and simulated log likelihood."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
from scipy.special import gammaln, logsumexp

MIN_MU = 1e-300
MAX_LOG_MU = 700.0
DEFAULT_CHUNK_SIZE = 10_000


def negative_binomial_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """NB2 log PMF with ``Var(y|mu)=mu+alpha*mu**2``."""

    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be positive and finite.")
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any(y < 0):
        raise ValueError("Negative binomial counts must be non-negative.")
    mu = np.maximum(mu, MIN_MU)
    size = 1.0 / alpha
    log_size_plus_mu = np.log(size + mu)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * (np.log(size) - log_size_plus_mu)
        + y * (np.log(mu) - log_size_plus_mu)
    )


def random_coefficients(
    draws: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray | None = None,
    cholesky: np.ndarray | None = None,
) -> np.ndarray:
    """Transform standard-normal draws into random coefficients."""

    means = np.asarray(means, dtype=float)
    if means.size == 0:
        return np.zeros(draws.shape, dtype=float)
    if cholesky is not None:
        return means + np.einsum("grq,kq->grk", draws, cholesky)
    if sds is None:
        raise ValueError("sds are required when cholesky is not provided.")
    return means + draws * np.asarray(sds, dtype=float)


def linear_predictor(
    beta_fixed: np.ndarray,
    random_coefficients_draws: np.ndarray | None,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    offset: np.ndarray,
    group_codes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute log(mu) for fixed-only or random-parameter draws."""

    beta_fixed = np.asarray(beta_fixed, dtype=float)
    base_eta = np.asarray(offset, dtype=float).copy()
    if beta_fixed.size:
        base_eta = base_eta + np.asarray(x_fixed, dtype=float) @ beta_fixed
    if random_coefficients_draws is None or random_coefficients_draws.shape[-1] == 0:
        return base_eta
    if group_codes is None:
        raise ValueError("group_codes are required for random-parameter prediction.")
    x_random = np.asarray(x_random, dtype=float)
    coeffs = random_coefficients_draws[np.asarray(group_codes, dtype=int)]
    eta = np.broadcast_to(base_eta[:, None], coeffs.shape[:2]).copy()
    for parameter_number in range(x_random.shape[1]):
        eta += x_random[:, parameter_number, None] * coeffs[:, :, parameter_number]
    return eta


def mu_from_eta(eta: np.ndarray) -> np.ndarray:
    """Convert a log mean to a finite mean."""

    return np.exp(np.clip(eta, np.log(MIN_MU), MAX_LOG_MU))


def simulated_log_likelihood(
    beta_fixed: np.ndarray,
    random_means: np.ndarray,
    alpha: float,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    group_indices: Sequence[np.ndarray],
    draws: np.ndarray,
    random_sds: np.ndarray | None = None,
    cholesky: np.ndarray | None = None,
    group_starts: np.ndarray | None = None,
    group_counts: np.ndarray | None = None,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    workers: int = 1,
    pool: Executor | None = None,
) -> float:
    """Compute fixed-only or panel simulated maximum likelihood for NB2."""

    beta_fixed = np.asarray(beta_fixed, dtype=float)
    random_means = np.asarray(random_means, dtype=float)
    x_fixed = np.asarray(x_fixed, dtype=float)
    x_random = np.asarray(x_random, dtype=float)
    offset = np.asarray(offset, dtype=float)
    y = np.asarray(y, dtype=float)

    q = random_means.size
    if q == 0:
        eta = linear_predictor(beta_fixed, None, x_fixed, x_random, offset)
        return float(np.sum(negative_binomial_logpmf(y, mu_from_eta(eta), alpha)))

    if group_starts is None or group_counts is None:
        order, group_starts, group_counts = group_structure_from_indices(
            group_indices, y.size
        )
        x_fixed = x_fixed[order]
        x_random = x_random[order]
        offset = offset[order]
        y = y[order]
    else:
        group_starts = np.asarray(group_starts, dtype=int)
        group_counts = np.asarray(group_counts, dtype=int)

    coeffs = random_coefficients(draws, random_means, random_sds, cholesky)
    chunks = list(_iter_group_chunks(group_starts, group_counts, chunk_size))

    if workers > 1:
        payloads = [
            _build_chunk_payload(
                chunk,
                beta_fixed,
                alpha,
                x_fixed,
                x_random,
                offset,
                y,
                group_starts,
                group_counts,
                coeffs,
            )
            for chunk in chunks
        ]
        if pool is not None:
            try:
                values = pool.map(_simulated_log_likelihood_chunk, payloads)
                return float(np.sum(list(values)))
            except (OSError, BrokenProcessPool):
                pass
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                values = executor.map(_simulated_log_likelihood_chunk, payloads)
                return float(np.sum(list(values)))
        except (OSError, BrokenProcessPool):
            pass

    total = 0.0
    for chunk in chunks:
        payload = _build_chunk_payload(
            chunk,
            beta_fixed,
            alpha,
            x_fixed,
            x_random,
            offset,
            y,
            group_starts,
            group_counts,
            coeffs,
        )
        total += _simulated_log_likelihood_chunk(payload)
    return float(total)


def group_structure_from_indices(
    group_indices: Sequence[np.ndarray], n_observations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observation order, group starts, and group counts."""

    counts = np.asarray([len(indices) for indices in group_indices], dtype=int)
    order = np.empty(n_observations, dtype=int)
    cursor = 0
    starts = np.empty(len(group_indices), dtype=int)
    for group_number, indices in enumerate(group_indices):
        starts[group_number] = cursor
        count = counts[group_number]
        order[cursor : cursor + count] = indices
        cursor += count
    return order, starts, counts


def _iter_group_chunks(
    group_starts: np.ndarray, group_counts: np.ndarray, chunk_size: int | None
) -> list[tuple[int, int]]:
    n_groups = group_counts.size
    if chunk_size is None or chunk_size <= 0:
        return [(0, n_groups)]

    chunks: list[tuple[int, int]] = []
    group_start = 0
    obs_count = 0
    for group_end, count in enumerate(group_counts, start=1):
        if obs_count and obs_count + count > chunk_size:
            chunks.append((group_start, group_end - 1))
            group_start = group_end - 1
            obs_count = 0
        obs_count += int(count)
    if group_start < n_groups:
        chunks.append((group_start, n_groups))
    return chunks


def _build_chunk_payload(
    chunk: tuple[int, int],
    beta_fixed: np.ndarray,
    alpha: float,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    group_starts: np.ndarray,
    group_counts: np.ndarray,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, ...]:
    first_group, last_group = chunk
    obs_start = group_starts[first_group]
    obs_end = group_starts[last_group - 1] + group_counts[last_group - 1]
    local_starts = group_starts[first_group:last_group] - obs_start
    local_counts = group_counts[first_group:last_group]
    return (
        beta_fixed,
        np.asarray([alpha], dtype=float),
        x_fixed[obs_start:obs_end],
        x_random[obs_start:obs_end],
        offset[obs_start:obs_end],
        y[obs_start:obs_end],
        local_starts,
        local_counts,
        coeffs[first_group:last_group],
    )


def _simulated_log_likelihood_chunk(payload: tuple[np.ndarray, ...]) -> float:
    (
        beta_fixed,
        alpha_array,
        x_fixed,
        x_random,
        offset,
        y,
        group_starts,
        group_counts,
        coeffs,
    ) = payload
    alpha = float(alpha_array[0])
    n_observations = y.size
    n_draws = coeffs.shape[1]
    group_codes = np.repeat(np.arange(group_counts.size), group_counts)
    eta = linear_predictor(beta_fixed, coeffs, x_fixed, x_random, offset, group_codes)
    log_probs = negative_binomial_logpmf(y[:, None], mu_from_eta(eta), alpha)
    group_draw_log_likelihood = np.add.reduceat(log_probs, group_starts, axis=0)
    return float(np.sum(logsumexp(group_draw_log_likelihood, axis=1) - np.log(n_draws)))
