"""Random Parameters Ordered Probit estimator."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from rpopit.config import ModelSpec, RandomParameterSpec, load_model_spec
from rpopit.draws import generate_draws
from rpopit.likelihood import simulated_log_likelihood
from rpopit.marginal_effects import average_marginal_effects, predicted_probabilities
from rpopit.optimizer import (
    covariance_from_hessian,
    estimate_mle,
    finite_difference_hessian,
)
from rpopit.output import RPOpitResults, build_parameter_table


@dataclass
class ParameterState:
    fixed: np.ndarray
    random_means: np.ndarray
    thresholds: np.ndarray
    random_sds: np.ndarray | None = None
    cholesky: np.ndarray | None = None


class RandomParametersOrderedProbit:
    """Estimate ordered probit models with normally distributed random parameters."""

    def __init__(
        self,
        dependent: str,
        fixed: Sequence[str] | None = None,
        random: Sequence[str | RandomParameterSpec] | None = None,
        group_id: str | None = None,
        categories: Sequence[Any] | None = None,
        draws: int = 200,
        draw_type: str = "halton",
        correlated_random_parameters: bool = False,
        seed: int | None = 12345,
        maxiter: int = 1000,
        tolerance: float = 1e-4,
        covariance: str = "bfgs",
        chunk_size: int | None = 10_000,
        workers: int = 1,
        output_dir: str = "runs",
        missing: str = "drop",
    ) -> None:
        self.dependent = dependent
        self.fixed = tuple(fixed or ())
        self.random_specs = tuple(_coerce_random_specs(random or ()))
        self.random = tuple(item.name for item in self.random_specs)
        self.group_id = group_id
        self.categories = None if categories is None else tuple(categories)
        self.draws = int(draws)
        self.draw_type = draw_type
        self.correlated_random_parameters = bool(correlated_random_parameters)
        self.seed = seed
        self.maxiter = int(maxiter)
        self.tolerance = float(tolerance)
        self.covariance = covariance.lower()
        self.chunk_size = chunk_size
        self.workers = int(workers)
        self.output_dir = output_dir
        self.missing = missing.lower()

        if self.draws < 1:
            raise ValueError("draws must be at least 1.")
        if self.covariance not in {"bfgs", "hessian"}:
            raise ValueError("covariance must be 'bfgs' or 'hessian'.")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be positive or None.")
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "RandomParametersOrderedProbit":
        return cls(
            dependent=spec.dependent,
            fixed=spec.fixed,
            random=spec.random,
            group_id=spec.group_id,
            categories=spec.categories,
            draws=spec.draws,
            draw_type=spec.draw_type,
            correlated_random_parameters=spec.correlated_random_parameters,
            seed=spec.seed,
            maxiter=spec.maxiter,
            tolerance=spec.tolerance,
            covariance=spec.covariance,
            chunk_size=spec.chunk_size,
            workers=spec.workers,
            output_dir=spec.output_dir,
            missing=spec.missing,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RandomParametersOrderedProbit":
        return cls.from_spec(load_model_spec(path))

    def fit(
        self,
        data: str | Path | pd.DataFrame,
        save_run: bool = True,
        output_dir: str | Path | None = None,
        export: bool = False,
    ) -> RPOpitResults:
        """Fit the model using simulated maximum likelihood."""

        fit_start = time.perf_counter()
        timing: dict[str, float | int] = {}
        run_dir = _create_run_dir(output_dir or self.output_dir) if save_run else None
        logger = _run_logger(run_dir)
        logger.info("Starting rpopit estimation.")
        logger.info("Model specification: %s", self.to_spec_dict())

        prepare_start = time.perf_counter()
        frame = _load_dataframe(data)
        work = self._prepare_frame(frame, logger)
        y_codes, categories = self._encode_dependent(work[self.dependent])
        x_fixed = _matrix(work, self.fixed)
        x_random = _matrix(work, self.random)
        q = len(self.random)
        group_codes, group_labels, group_indices, order, group_starts, group_counts = (
            self._group_indices(work, q > 0)
        )
        n_groups = len(group_indices)
        x_fixed_likelihood = x_fixed[order]
        x_random_likelihood = x_random[order]
        y_codes_likelihood = y_codes[order]
        timing["data_preparation_seconds"] = time.perf_counter() - prepare_start

        draw_start = time.perf_counter()
        draws = generate_draws(n_groups, self.draws, q, self.draw_type, self.seed)
        timing["draw_generation_seconds"] = time.perf_counter() - draw_start

        logger.info(
            "Prepared %s observations, %s groups, %s categories, %s random parameters.",
            len(work),
            n_groups,
            len(categories),
            q,
        )

        start_params = self._start_params(y_codes, len(categories))
        objective_calls = 0
        objective_seconds = 0.0
        effective_workers = self.workers
        process_pool = None
        if self.workers > 1 and q > 0:
            try:
                process_pool = ProcessPoolExecutor(max_workers=self.workers)
            except OSError as exc:
                logger.warning(
                    "Could not start multiprocessing workers; falling back to serial: %s",
                    exc,
                )
                effective_workers = 1

        def objective(theta: np.ndarray) -> float:
            nonlocal objective_calls, objective_seconds
            objective_start = time.perf_counter()
            try:
                state = self._unpack_params(theta, len(categories) - 1)
                value = simulated_log_likelihood(
                    beta_fixed=state.fixed,
                    random_means=state.random_means,
                    random_sds=state.random_sds,
                    cholesky=state.cholesky,
                    thresholds=state.thresholds,
                    x_fixed=x_fixed_likelihood,
                    x_random=x_random_likelihood,
                    y_codes=y_codes_likelihood,
                    group_indices=group_indices,
                    group_starts=group_starts,
                    group_counts=group_counts,
                    draws=draws,
                    chunk_size=self.chunk_size,
                    workers=effective_workers,
                    pool=process_pool,
                )
            except (FloatingPointError, ValueError, OverflowError):
                value = -1e100
            finally:
                objective_calls += 1
                objective_seconds += time.perf_counter() - objective_start
            if not np.isfinite(value):
                return 1e100
            return -float(value)

        optimization_start = time.perf_counter()
        diagnostics = estimate_mle(
            objective,
            start_params,
            method="BFGS",
            maxiter=self.maxiter,
            tolerance=self.tolerance,
        )
        timing["optimization_seconds"] = time.perf_counter() - optimization_start
        timing["objective_calls"] = objective_calls
        timing["objective_seconds"] = objective_seconds
        timing["average_objective_seconds"] = (
            objective_seconds / objective_calls if objective_calls else 0.0
        )
        logger.info("Optimization finished: %s", diagnostics.message)

        post_start = time.perf_counter()
        final_state = self._unpack_params(diagnostics.params, len(categories) - 1)
        internal_covariance = diagnostics.hess_inv
        if self.covariance == "hessian":
            logger.info("Computing finite-difference Hessian covariance.")
            try:
                hessian = finite_difference_hessian(objective, diagnostics.params)
                internal_covariance = covariance_from_hessian(hessian)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Falling back to BFGS covariance: %s", exc)
        if process_pool is not None:
            process_pool.shutdown()

        names, components, variables, estimates = self._natural_parameters(
            diagnostics.params, categories
        )
        natural_covariance = None
        if internal_covariance is not None:
            jacobian = self._natural_jacobian(diagnostics.params, categories)
            natural_covariance = jacobian @ internal_covariance @ jacobian.T

        parameter_table = build_parameter_table(
            names, components, variables, estimates, natural_covariance
        )

        probabilities = predicted_probabilities(
            final_state.fixed,
            final_state.random_means,
            final_state.thresholds,
            x_fixed,
            x_random,
            group_indices,
            draws,
            categories,
            random_sds=final_state.random_sds,
            cholesky=final_state.cholesky,
        )
        probabilities.insert(0, "row_index", work.index.to_numpy())
        probabilities.insert(1, self.dependent, work[self.dependent].to_numpy())
        probabilities.insert(2, "_group_code", group_codes)
        probabilities.insert(3, "_group_label", group_labels[group_codes])

        effects = average_marginal_effects(
            final_state.fixed,
            final_state.random_means,
            final_state.thresholds,
            x_fixed,
            x_random,
            group_indices,
            draws,
            self.fixed,
            self.random,
            categories,
            random_sds=final_state.random_sds,
            cholesky=final_state.cholesky,
        )

        n_params = diagnostics.params.size
        log_likelihood = diagnostics.log_likelihood
        fit_statistics = {
            "dependent": self.dependent,
            "n_observations": int(len(work)),
            "n_groups": int(n_groups),
            "n_categories": int(len(categories)),
            "n_parameters": int(n_params),
            "log_likelihood": log_likelihood,
            "AIC": 2.0 * n_params - 2.0 * log_likelihood,
            "BIC": np.log(len(work)) * n_params - 2.0 * log_likelihood,
            "draw_type": self.draw_type,
            "draws": int(self.draws if q else 1),
            "correlated_random_parameters": self.correlated_random_parameters,
        }
        convergence = {
            "converged": diagnostics.converged,
            "status": diagnostics.status,
            "message": diagnostics.message,
            "iterations": diagnostics.iterations,
            "function_evaluations": diagnostics.function_evaluations,
            "gradient_norm": diagnostics.gradient_norm,
            "chunk_size": self.chunk_size,
            "workers_requested": self.workers,
            "workers_used": effective_workers,
        }
        timing["postestimation_seconds"] = time.perf_counter() - post_start
        timing["total_fit_seconds"] = time.perf_counter() - fit_start

        if run_dir is not None:
            with (run_dir / "model_spec.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.to_spec_dict(), handle, sort_keys=False)

        results = RPOpitResults(
            parameter_table=parameter_table,
            fit_statistics=fit_statistics,
            convergence=convergence,
            predicted_probabilities=probabilities,
            marginal_effects=effects,
            run_dir=run_dir,
            model_spec=self.to_spec_dict(),
            timing=timing,
        )
        if export:
            results.export(run_dir)
            logger.info("Exported results to %s.", run_dir)
        logger.info(results.summary())
        return results

    def to_spec_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "dependent": self.dependent,
                "fixed": list(self.fixed),
                "random": {
                    item.name: {
                        "distribution": item.distribution,
                        "start_mean": item.start_mean,
                        "start_sd": item.start_sd,
                    }
                    for item in self.random_specs
                },
                "group_id": self.group_id,
                "categories": None if self.categories is None else list(self.categories),
                "correlated_random_parameters": self.correlated_random_parameters,
                "missing": self.missing,
            },
            "simulation": {
                "draws": self.draws,
                "draw_type": self.draw_type,
                "seed": self.seed,
            },
            "estimation": {
                "maxiter": self.maxiter,
                "tolerance": self.tolerance,
                "covariance": self.covariance,
                "chunk_size": self.chunk_size,
                "workers": self.workers,
            },
            "output": {"directory": self.output_dir},
        }

    def _prepare_frame(self, frame: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
        columns = [self.dependent, *self.fixed, *self.random]
        if self.group_id is not None:
            columns.append(self.group_id)
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Data are missing required columns: {missing_columns}")

        work = frame.loc[:, columns].copy()
        missing_count = int(work.isna().any(axis=1).sum())
        if missing_count:
            if self.missing == "drop":
                work = work.dropna(axis=0)
                logger.info("Dropped %s rows with missing model data.", missing_count)
            else:
                raise ValueError(
                    f"Found {missing_count} rows with missing model data and missing!='drop'."
                )
        if len(work) == 0:
            raise ValueError("No usable observations remain after missing-data handling.")
        return work

    def _encode_dependent(self, series: pd.Series) -> tuple[np.ndarray, tuple[Any, ...]]:
        if self.categories is None:
            categories = _ordered_unique(series)
        else:
            categories = self.categories
        if len(categories) < 2:
            raise ValueError("Ordered probit requires at least two outcome categories.")
        lookup = {category: index for index, category in enumerate(categories)}
        try:
            codes = series.map(lookup).astype(int).to_numpy()
        except ValueError as exc:
            unknown = sorted(set(series.unique()) - set(categories))
            raise ValueError(f"Dependent variable contains categories not in spec: {unknown}") from exc
        return codes, tuple(categories)

    def _group_indices(
        self, work: pd.DataFrame, random_parameters: bool
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        n_observations = len(work)
        if self.group_id is None:
            if random_parameters:
                codes = np.arange(n_observations, dtype=int)
                labels = work.index.to_numpy()
                order = np.arange(n_observations, dtype=int)
                starts = np.arange(n_observations, dtype=int)
                counts = np.ones(n_observations, dtype=int)
                indices = [np.array([index], dtype=int) for index in range(n_observations)]
            else:
                codes = np.zeros(n_observations, dtype=int)
                labels = np.array(["all"])
                order = np.arange(n_observations, dtype=int)
                starts = np.array([0], dtype=int)
                counts = np.array([n_observations], dtype=int)
                indices = [order]
        else:
            codes, labels = pd.factorize(work[self.group_id], sort=False)
            order = np.argsort(codes, kind="stable")
            counts = np.bincount(codes, minlength=len(labels)).astype(int)
            starts = np.empty(len(labels), dtype=int)
            starts[0] = 0
            if len(labels) > 1:
                starts[1:] = np.cumsum(counts[:-1])
            indices = list(np.split(order, starts[1:]))
        return codes, np.asarray(labels), indices, order, starts, counts

    def _start_params(self, y_codes: np.ndarray, n_categories: int) -> np.ndarray:
        fixed = np.zeros(len(self.fixed), dtype=float)
        random_means = np.array([spec.start_mean for spec in self.random_specs], dtype=float)
        random_sds = np.array([spec.start_sd for spec in self.random_specs], dtype=float)
        pieces = [fixed, random_means]

        q = len(self.random_specs)
        if q:
            if self.correlated_random_parameters:
                chol = np.zeros((q, q), dtype=float)
                np.fill_diagonal(chol, random_sds)
                pieces.append(self._pack_cholesky(chol))
            else:
                pieces.append(np.log(random_sds))

        thresholds = _initial_thresholds(y_codes, n_categories)
        pieces.append(_pack_thresholds(thresholds))
        return np.concatenate(pieces)

    def _unpack_params(self, theta: np.ndarray, n_thresholds: int) -> ParameterState:
        theta = np.asarray(theta, dtype=float)
        cursor = 0
        k = len(self.fixed)
        q = len(self.random_specs)

        fixed = theta[cursor : cursor + k]
        cursor += k
        random_means = theta[cursor : cursor + q]
        cursor += q

        random_sds = None
        cholesky = None
        if q:
            if self.correlated_random_parameters:
                n_chol = q * (q + 1) // 2
                cholesky = self._unpack_cholesky(theta[cursor : cursor + n_chol], q)
                cursor += n_chol
            else:
                random_sds = np.exp(theta[cursor : cursor + q])
                cursor += q

        thresholds = _unpack_thresholds(theta[cursor : cursor + n_thresholds])
        return ParameterState(fixed, random_means, thresholds, random_sds, cholesky)

    def _pack_cholesky(self, cholesky: np.ndarray) -> np.ndarray:
        values = []
        for row in range(cholesky.shape[0]):
            for col in range(row + 1):
                value = cholesky[row, col]
                values.append(np.log(value) if row == col else value)
        return np.asarray(values, dtype=float)

    def _unpack_cholesky(self, packed: np.ndarray, q: int) -> np.ndarray:
        cholesky = np.zeros((q, q), dtype=float)
        cursor = 0
        for row in range(q):
            for col in range(row + 1):
                value = packed[cursor]
                cholesky[row, col] = np.exp(value) if row == col else value
                cursor += 1
        return cholesky

    def _natural_parameters(
        self, theta: np.ndarray, categories: Sequence[Any]
    ) -> tuple[list[str], list[str], list[str], np.ndarray]:
        state = self._unpack_params(theta, len(categories) - 1)
        names: list[str] = []
        components: list[str] = []
        variables: list[str] = []
        values: list[float] = []

        for name, value in zip(self.fixed, state.fixed):
            names.append(f"beta_fixed[{name}]")
            components.append("fixed_mean")
            variables.append(name)
            values.append(float(value))

        for name, value in zip(self.random, state.random_means):
            names.append(f"beta_random_mean[{name}]")
            components.append("random_mean")
            variables.append(name)
            values.append(float(value))

        if self.random:
            if self.correlated_random_parameters:
                covariance = state.cholesky @ state.cholesky.T
                sds = np.sqrt(np.diag(covariance))
                for name, value in zip(self.random, sds):
                    names.append(f"beta_random_sd[{name}]")
                    components.append("random_sd")
                    variables.append(name)
                    values.append(float(value))
                for i in range(len(self.random)):
                    for j in range(i):
                        corr = covariance[i, j] / (sds[i] * sds[j])
                        names.append(f"corr[{self.random[i]},{self.random[j]}]")
                        components.append("random_correlation")
                        variables.append(f"{self.random[i]},{self.random[j]}")
                        values.append(float(corr))
            else:
                for name, value in zip(self.random, state.random_sds):
                    names.append(f"beta_random_sd[{name}]")
                    components.append("random_sd")
                    variables.append(name)
                    values.append(float(value))

        for threshold_number, value in enumerate(state.thresholds, start=1):
            lower = categories[threshold_number - 1]
            upper = categories[threshold_number]
            names.append(f"threshold[{threshold_number}]")
            components.append("threshold")
            variables.append(f"{lower}|{upper}")
            values.append(float(value))

        return names, components, variables, np.asarray(values, dtype=float)

    def _natural_jacobian(self, theta: np.ndarray, categories: Sequence[Any]) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        base = self._natural_parameters(theta, categories)[3]
        jacobian = np.empty((base.size, theta.size), dtype=float)
        steps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(theta), 1.0)
        for column in range(theta.size):
            step_vector = np.zeros(theta.size, dtype=float)
            step_vector[column] = steps[column]
            plus = self._natural_parameters(theta + step_vector, categories)[3]
            minus = self._natural_parameters(theta - step_vector, categories)[3]
            jacobian[:, column] = (plus - minus) / (2.0 * steps[column])
        return jacobian


RPOpitModel = RandomParametersOrderedProbit


def _coerce_random_specs(
    random: Iterable[str | RandomParameterSpec],
) -> list[RandomParameterSpec]:
    specs: list[RandomParameterSpec] = []
    for item in random:
        if isinstance(item, RandomParameterSpec):
            specs.append(item)
        else:
            specs.append(RandomParameterSpec(name=str(item)))
    return specs


def _load_dataframe(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.read_csv(data)


def _matrix(work: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(work), 0), dtype=float)
    return work.loc[:, columns].astype(float).to_numpy()


def _ordered_unique(series: pd.Series) -> tuple[Any, ...]:
    unique = list(pd.unique(series))
    try:
        return tuple(sorted(unique))
    except TypeError:
        return tuple(unique)


def _initial_thresholds(y_codes: np.ndarray, n_categories: int) -> np.ndarray:
    counts = np.bincount(y_codes, minlength=n_categories)
    cumulative = np.cumsum(counts[:-1]) / counts.sum()
    cumulative = np.clip(cumulative, 0.02, 0.98)
    thresholds = norm.ppf(cumulative)
    for i in range(1, thresholds.size):
        if thresholds[i] <= thresholds[i - 1] + 0.05:
            thresholds[i] = thresholds[i - 1] + 0.05
    return thresholds


def _pack_thresholds(thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    packed = np.empty_like(thresholds)
    packed[0] = thresholds[0]
    if thresholds.size > 1:
        packed[1:] = np.log(np.diff(thresholds))
    return packed


def _unpack_thresholds(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=float)
    thresholds = np.empty_like(packed)
    thresholds[0] = packed[0]
    if packed.size > 1:
        thresholds[1:] = packed[0] + np.cumsum(np.exp(packed[1:]))
    return thresholds


def _create_run_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"rpopit_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _run_logger(run_dir: Path | None) -> logging.Logger:
    name = f"rpopit.run.{id(run_dir)}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if run_dir is not None:
        handler: logging.Handler = logging.FileHandler(
            run_dir / "rpopit.log", encoding="utf-8"
        )
    else:
        handler = logging.NullHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
