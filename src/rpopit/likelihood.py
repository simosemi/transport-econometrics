"""Ordered probit probabilities and simulated log likelihood."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

MIN_PROBABILITY = 1e-300
DEFAULT_CHUNK_SIZE = 10_000


def stable_logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Small wrapper around SciPy's stable log-sum-exp implementation."""

    return logsumexp(values, axis=axis)


def ordered_probit_log_probabilities(
    eta: np.ndarray, thresholds: np.ndarray, min_probability: float = MIN_PROBABILITY
) -> np.ndarray:
    """Compute log probabilities for all ordered outcomes.

    Parameters
    ----------
    eta:
        Latent mean values. Can be any shape.
    thresholds:
        Finite ordered thresholds, length ``n_categories - 1``.
    min_probability:
        Probability floor used only after stable calculations to avoid log(0).
    """

    eta = np.asarray(eta, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be one-dimensional.")
    if thresholds.size and np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be strictly increasing.")

    cuts = np.concatenate(([-np.inf], thresholds, [np.inf]))
    out = np.empty(eta.shape + (cuts.size - 1,), dtype=float)
    floor = np.log(min_probability)

    for j in range(cuts.size - 1):
        lower = cuts[j] - eta
        upper = cuts[j + 1] - eta
        if np.isneginf(cuts[j]):
            logp = log_ndtr(upper)
        elif np.isposinf(cuts[j + 1]):
            logp = log_ndtr(-lower)
        else:
            logp_cdf = _logdiffexp(log_ndtr(upper), log_ndtr(lower))
            logp_sf = _logdiffexp(log_ndtr(-lower), log_ndtr(-upper))
            logp = np.where((lower + upper) <= 0.0, logp_cdf, logp_sf)
        out[..., j] = np.maximum(logp, floor)
    return out


def ordered_probit_probabilities(
    eta: np.ndarray, thresholds: np.ndarray, min_probability: float = MIN_PROBABILITY
) -> np.ndarray:
    """Compute ordered probit probabilities for all outcomes."""

    return np.exp(ordered_probit_log_probabilities(eta, thresholds, min_probability))


def ordered_probit_selected_log_probs(
    eta: np.ndarray, y_codes: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Return log probabilities for observed category codes."""

    eta = np.asarray(eta, dtype=float)
    y_codes = np.asarray(y_codes, dtype=int)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be one-dimensional.")
    if thresholds.size and np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be strictly increasing.")

    if eta.ndim == 1:
        cuts = np.concatenate(([-np.inf], thresholds, [np.inf]))
        lower_cut = cuts[y_codes]
        upper_cut = cuts[y_codes + 1]
        return _bounded_interval_logprob(lower_cut - eta, upper_cut - eta)

    cuts = np.concatenate(([-np.inf], thresholds, [np.inf]))
    lower = cuts[y_codes][:, None] - eta
    upper = cuts[y_codes + 1][:, None] - eta
    return _bounded_interval_logprob(lower, upper)


def ordered_probit_density_difference(eta: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return phi(lower - eta) - phi(upper - eta) for each outcome."""

    eta = np.asarray(eta, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    cuts = np.concatenate(([-np.inf], thresholds, [np.inf]))
    out = np.empty(eta.shape + (cuts.size - 1,), dtype=float)
    for j in range(cuts.size - 1):
        lower = cuts[j] - eta
        upper = cuts[j + 1] - eta
        lower_pdf = np.zeros_like(eta) if np.isneginf(cuts[j]) else norm.pdf(lower)
        upper_pdf = np.zeros_like(eta) if np.isposinf(cuts[j + 1]) else norm.pdf(upper)
        out[..., j] = lower_pdf - upper_pdf
    return out


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


def simulated_log_likelihood(
    beta_fixed: np.ndarray,
    random_means: np.ndarray,
    thresholds: np.ndarray,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    y_codes: np.ndarray,
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
    """Compute panel simulated maximum likelihood for ordered probit."""

    beta_fixed = np.asarray(beta_fixed, dtype=float)
    random_means = np.asarray(random_means, dtype=float)
    x_fixed = np.asarray(x_fixed, dtype=float)
    x_random = np.asarray(x_random, dtype=float)
    y_codes = np.asarray(y_codes, dtype=int)

    q = random_means.size
    if q == 0:
        eta = x_fixed @ beta_fixed if beta_fixed.size else np.zeros(y_codes.size)
        return float(np.sum(ordered_probit_selected_log_probs(eta, y_codes, thresholds)))

    if group_starts is None or group_counts is None:
        order, group_starts, group_counts = group_structure_from_indices(
            group_indices, y_codes.size
        )
        x_fixed = x_fixed[order]
        x_random = x_random[order]
        y_codes = y_codes[order]
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
                thresholds,
                x_fixed,
                x_random,
                y_codes,
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
            thresholds,
            x_fixed,
            x_random,
            y_codes,
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


def _bounded_interval_logprob(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    floor = np.log(MIN_PROBABILITY)
    if np.isscalar(lower) or getattr(lower, "ndim", 0) == 0:
        if np.isneginf(lower):
            return np.maximum(log_ndtr(upper), floor)
        if np.isposinf(upper):
            return np.maximum(log_ndtr(-lower), floor)

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    first = np.isneginf(lower)
    last = np.isposinf(upper)
    middle = ~(first | last)
    out = np.empty_like(lower, dtype=float)

    if np.any(first):
        out[first] = log_ndtr(upper[first])
    if np.any(last):
        out[last] = log_ndtr(-lower[last])
    if np.any(middle):
        lower_mid = lower[middle]
        upper_mid = upper[middle]
        logp_cdf = _logdiffexp(log_ndtr(upper_mid), log_ndtr(lower_mid))
        logp_sf = _logdiffexp(log_ndtr(-lower_mid), log_ndtr(-upper_mid))
        out[middle] = np.where((lower_mid + upper_mid) <= 0.0, logp_cdf, logp_sf)
    return np.maximum(out, floor)


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
    thresholds: np.ndarray,
    x_fixed: np.ndarray,
    x_random: np.ndarray,
    y_codes: np.ndarray,
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
        thresholds,
        x_fixed[obs_start:obs_end],
        x_random[obs_start:obs_end],
        y_codes[obs_start:obs_end],
        local_starts,
        local_counts,
        coeffs[first_group:last_group],
    )


def _simulated_log_likelihood_chunk(payload: tuple[np.ndarray, ...]) -> float:
    (
        beta_fixed,
        thresholds,
        x_fixed,
        x_random,
        y_codes,
        group_starts,
        group_counts,
        coeffs,
    ) = payload
    n_observations = y_codes.size
    n_draws = coeffs.shape[1]
    base_eta = x_fixed @ beta_fixed if beta_fixed.size else np.zeros(n_observations)
    group_codes = np.repeat(np.arange(group_counts.size), group_counts)

    eta = np.broadcast_to(base_eta[:, None], (n_observations, n_draws)).copy()
    for parameter_number in range(x_random.shape[1]):
        eta += (
            x_random[:, parameter_number, None]
            * coeffs[group_codes, :, parameter_number]
        )

    log_probs = ordered_probit_selected_log_probs(eta, y_codes, thresholds)
    group_draw_log_likelihood = np.add.reduceat(log_probs, group_starts, axis=0)
    return float(np.sum(logsumexp(group_draw_log_likelihood, axis=1) - np.log(n_draws)))


def _logdiffexp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Compute log(exp(log_a) - exp(log_b)) for log_a >= log_b."""

    ratio = np.exp(np.minimum(log_b - log_a, 0.0))
    ratio = np.clip(ratio, 0.0, 1.0 - 1e-15)
    return log_a + np.log1p(-ratio)
