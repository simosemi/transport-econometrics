"""Optimization and covariance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizationDiagnostics:
    params: np.ndarray
    objective_value: float
    log_likelihood: float
    converged: bool
    status: int
    message: str
    iterations: int | None
    function_evaluations: int | None
    gradient_norm: float | None
    hess_inv: np.ndarray | None


def estimate_mle(
    objective: Callable[[np.ndarray], float],
    start_params: np.ndarray,
    method: str = "BFGS",
    maxiter: int = 1000,
    tolerance: float = 1e-6,
    callback: Callable[[int, np.ndarray], None] | None = None,
    initial_iteration: int = 0,
) -> OptimizationDiagnostics:
    """Minimize a negative log likelihood with SciPy."""

    iteration = int(initial_iteration)

    def scipy_callback(params: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if callback is not None:
            callback(iteration, np.asarray(params, dtype=float))

    result = minimize(
        objective,
        np.asarray(start_params, dtype=float),
        method=method,
        callback=scipy_callback,
        options={"maxiter": maxiter, "gtol": tolerance, "disp": False},
    )
    gradient_norm = None
    if getattr(result, "jac", None) is not None:
        jac = np.asarray(result.jac, dtype=float)
        if jac.size:
            gradient_norm = float(np.linalg.norm(jac, ord=np.inf))

    hess_inv = hess_inv_to_array(getattr(result, "hess_inv", None))
    return OptimizationDiagnostics(
        params=np.asarray(result.x, dtype=float),
        objective_value=float(result.fun),
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=getattr(result, "nit", None),
        function_evaluations=getattr(result, "nfev", None),
        gradient_norm=gradient_norm,
        hess_inv=hess_inv,
    )


def hess_inv_to_array(hess_inv: object) -> np.ndarray | None:
    """Convert SciPy Hessian inverse objects to dense arrays."""

    if hess_inv is None:
        return None
    if hasattr(hess_inv, "todense"):
        return np.asarray(hess_inv.todense(), dtype=float)
    arr = np.asarray(hess_inv, dtype=float)
    return arr if arr.ndim == 2 else None


def finite_difference_hessian(
    objective: Callable[[np.ndarray], float], params: np.ndarray, step: float = 1e-4
) -> np.ndarray:
    """Central finite-difference Hessian for small to medium models."""

    x = np.asarray(params, dtype=float)
    n = x.size
    hessian = np.empty((n, n), dtype=float)
    f0 = float(objective(x))
    steps = step * np.maximum(np.abs(x), 1.0)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        f_plus = float(objective(x + ei))
        f_minus = float(objective(x - ei))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = steps[j]
            f_pp = float(objective(x + ei + ej))
            f_pm = float(objective(x + ei - ej))
            f_mp = float(objective(x - ei + ej))
            f_mm = float(objective(x - ei - ej))
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def covariance_from_hessian(hessian: np.ndarray) -> np.ndarray:
    """Invert an observed Hessian, falling back to a pseudo-inverse."""

    try:
        return np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(hessian)
