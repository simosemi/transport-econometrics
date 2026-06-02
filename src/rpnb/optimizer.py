"""Optimization and covariance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


SUPPORTED_OPTIMIZERS = {
    "bfgs": "BFGS",
    "lbfgsb": "L-BFGS-B",
    "nelder-mead": "Nelder-Mead",
    "powell": "Powell",
}


@dataclass
class OptimizationDiagnostics:
    params: np.ndarray
    objective_value: float
    log_likelihood: float
    method: str
    converged: bool
    status: int
    message: str
    iterations: int | None
    function_evaluations: int | None
    gradient_norm: float | None
    hessian_condition_number: float | None
    largest_parameter_magnitude: float
    smallest_parameter_magnitude: float
    hess_inv: np.ndarray | None


def estimate_mle(
    objective: Callable[[np.ndarray], float],
    start_params: np.ndarray,
    method: str = "bfgs",
    maxiter: int = 1000,
    tolerance: float = 1e-6,
    callback: Callable[[int, np.ndarray], None] | None = None,
    initial_iteration: int = 0,
) -> OptimizationDiagnostics:
    """Minimize a negative log likelihood with SciPy."""

    optimizer = normalize_optimizer(method)
    scipy_method = SUPPORTED_OPTIMIZERS[optimizer]
    iteration = int(initial_iteration)

    def scipy_callback(params: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if callback is not None:
            callback(iteration, np.asarray(params, dtype=float))

    result = minimize(
        objective,
        np.asarray(start_params, dtype=float),
        method=scipy_method,
        callback=scipy_callback,
        options=optimizer_options(optimizer, maxiter, tolerance),
    )
    gradient_norm = None
    if getattr(result, "jac", None) is not None:
        jac = np.asarray(result.jac, dtype=float)
        if jac.size:
            gradient_norm = float(np.linalg.norm(jac, ord=np.inf))
    if gradient_norm is None:
        gradient_norm = finite_difference_gradient_norm(objective, np.asarray(result.x, dtype=float))

    hess_inv = hess_inv_to_array(getattr(result, "hess_inv", None))
    params = np.asarray(result.x, dtype=float)
    parameter_magnitudes = np.abs(params)
    return OptimizationDiagnostics(
        params=params,
        objective_value=float(result.fun),
        log_likelihood=float(-result.fun),
        method=optimizer,
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=getattr(result, "nit", None),
        function_evaluations=getattr(result, "nfev", None),
        gradient_norm=gradient_norm,
        hessian_condition_number=matrix_condition_number(hess_inv),
        largest_parameter_magnitude=float(np.max(parameter_magnitudes))
        if parameter_magnitudes.size
        else np.nan,
        smallest_parameter_magnitude=float(np.min(parameter_magnitudes))
        if parameter_magnitudes.size
        else np.nan,
        hess_inv=hess_inv,
    )


def normalize_optimizer(method: str) -> str:
    """Normalize user-facing optimizer names to supported internal keys."""

    key = str(method).strip().lower().replace("_", "-")
    aliases = {
        "bfgs": "bfgs",
        "lbfgs": "lbfgsb",
        "l-bfgs-b": "lbfgsb",
        "lbfgsb": "lbfgsb",
        "neldermead": "nelder-mead",
        "nelder-mead": "nelder-mead",
        "powell": "powell",
    }
    optimizer = aliases.get(key)
    if optimizer not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(f"Unsupported optimizer {method!r}. Supported optimizers: {supported}.")
    return optimizer


def optimizer_options(optimizer: str, maxiter: int, tolerance: float) -> dict[str, float | int]:
    """Build SciPy minimize options for each supported optimizer."""

    if optimizer == "bfgs":
        return {"maxiter": int(maxiter), "gtol": float(tolerance)}
    if optimizer == "lbfgsb":
        return {
            "maxiter": int(maxiter),
            "maxfun": max(int(maxiter) * 20, int(maxiter)),
            "gtol": float(tolerance),
            "ftol": float(tolerance),
        }
    if optimizer == "nelder-mead":
        return {
            "maxiter": int(maxiter),
            "xatol": float(tolerance),
            "fatol": float(tolerance),
        }
    if optimizer == "powell":
        return {
            "maxiter": int(maxiter),
            "xtol": float(tolerance),
            "ftol": float(tolerance),
        }
    raise ValueError(f"Unsupported optimizer {optimizer!r}.")


def hess_inv_to_array(hess_inv: object) -> np.ndarray | None:
    """Convert SciPy Hessian inverse objects to dense arrays."""

    if hess_inv is None:
        return None
    if hasattr(hess_inv, "todense"):
        return np.asarray(hess_inv.todense(), dtype=float)
    arr = np.asarray(hess_inv, dtype=float)
    return arr if arr.ndim == 2 else None


def matrix_condition_number(matrix: np.ndarray | None) -> float | None:
    """Return the condition number of a Hessian or inverse-Hessian matrix."""

    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not np.isfinite(arr).all():
        return np.inf
    try:
        condition = float(np.linalg.cond(arr))
    except np.linalg.LinAlgError:
        return np.inf
    return condition if np.isfinite(condition) else np.inf


def optimizer_termination_report(
    converged: bool,
    status: int,
    message: str,
    hessian_condition_number: float | None,
) -> dict[str, bool | str]:
    """Classify why the optimizer stopped using SciPy status and diagnostics."""

    message_lower = message.lower()
    reason = "other"
    if converged:
        reason = "convergence"
    elif status == 1 or "maximum number of iterations" in message_lower:
        reason = "max_iterations"
    elif "singular" in message_lower or "ill-conditioned" in message_lower:
        reason = "singular_hessian"
    elif "line search" in message_lower or "line-search" in message_lower:
        reason = "line_search_failure"
    elif status == 2 or "precision loss" in message_lower:
        reason = "precision_loss"
    elif _condition_indicates_singular_hessian(hessian_condition_number):
        reason = "singular_hessian"

    return {
        "termination_reason": reason,
        "terminated_due_to_convergence": reason == "convergence",
        "terminated_due_to_max_iterations": reason == "max_iterations",
        "terminated_due_to_precision_loss": reason == "precision_loss",
        "terminated_due_to_singular_hessian": reason == "singular_hessian",
        "terminated_due_to_line_search_failure": reason == "line_search_failure",
    }


def convergence_quality(
    converged: bool,
    gradient_norm: float | None,
    tolerance: float,
) -> str:
    """Grade optimizer convergence quality from SciPy success and gradient size."""

    if gradient_norm is None or not np.isfinite(gradient_norm):
        return "not_converged"
    if converged and gradient_norm < tolerance:
        return "converged_clean"
    if (not converged) and gradient_norm < 0.01:
        return "near_converged"
    if gradient_norm < 0.10:
        return "usable_warning"
    return "not_converged"


def _condition_indicates_singular_hessian(condition: float | None) -> bool:
    if condition is None:
        return False
    return (not np.isfinite(condition)) or condition >= 1e12


def finite_difference_gradient_norm(
    objective: Callable[[np.ndarray], float],
    params: np.ndarray,
    step: float = 1e-6,
) -> float | None:
    """Return an infinity-norm finite-difference gradient estimate."""

    x = np.asarray(params, dtype=float)
    if x.size == 0:
        return 0.0
    gradient = np.empty(x.size, dtype=float)
    steps = step * np.maximum(np.abs(x), 1.0)
    try:
        for index in range(x.size):
            step_vector = np.zeros(x.size, dtype=float)
            step_vector[index] = steps[index]
            f_plus = float(objective(x + step_vector))
            f_minus = float(objective(x - step_vector))
            gradient[index] = (f_plus - f_minus) / (2.0 * steps[index])
    except (FloatingPointError, ValueError, OverflowError):
        return None
    if not np.isfinite(gradient).all():
        return None
    return float(np.linalg.norm(gradient, ord=np.inf))


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
