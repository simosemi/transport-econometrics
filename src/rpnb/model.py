"""Random Parameters Negative Binomial estimator."""

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

from rpnb.checkpoint import load_latest_checkpoint, save_checkpoint, write_run_metadata
from rpnb.config import (
    CategoricalVariableSpec,
    ModelSpec,
    RandomParameterSpec,
    load_model_spec,
)
from rpnb.draws import generate_draws
from rpnb.likelihood import simulated_log_likelihood
from rpnb.marginal_effects import average_marginal_effects, predicted_counts
from rpnb.optimizer import (
    covariance_from_hessian,
    estimate_mle,
    finite_difference_hessian,
)
from rpnb.output import RPNBResults, build_parameter_table


@dataclass
class ParameterState:
    fixed: np.ndarray
    random_means: np.ndarray
    alpha: float
    random_sds: np.ndarray | None = None
    cholesky: np.ndarray | None = None


@dataclass(frozen=True)
class MissingDataReport:
    checked_columns: tuple[str, ...]
    original_rows: int
    rows_removed: int
    final_rows: int


class RandomParametersNegativeBinomial:
    """Estimate NB2 count models with normally distributed random parameters."""

    def __init__(
        self,
        dependent: str,
        offset: str,
        fixed: Sequence[str] | None = None,
        fixed_categorical: Sequence[CategoricalVariableSpec | dict[str, Any]] | None = None,
        random: Sequence[str | RandomParameterSpec] | None = None,
        group_id: str | None = None,
        intercept: bool = True,
        draws: int = 200,
        draw_type: str = "halton",
        correlated_random_parameters: bool = False,
        seed: int | None = 12345,
        maxiter: int = 1000,
        tolerance: float = 1e-5,
        covariance: str = "bfgs",
        chunk_size: int | None = 10_000,
        workers: int = 1,
        checkpoint_interval: int = 10,
        start_alpha: float = 0.5,
        output_dir: str = "runs",
        missing: str = "drop",
    ) -> None:
        self.dependent = dependent
        self.offset = offset
        self.fixed = tuple(fixed or ())
        self.fixed_categorical_specs = tuple(
            _coerce_categorical_specs(fixed_categorical or ())
        )
        self.fixed_categorical = tuple(item.name for item in self.fixed_categorical_specs)
        self.random_specs = tuple(_coerce_random_specs(random or ()))
        self.random = tuple(item.name for item in self.random_specs)
        self.group_id = group_id
        self.intercept = bool(intercept)
        self.draws = int(draws)
        self.draw_type = draw_type
        self.correlated_random_parameters = bool(correlated_random_parameters)
        self.seed = seed
        self.maxiter = int(maxiter)
        self.tolerance = float(tolerance)
        self.covariance = covariance.lower()
        self.chunk_size = chunk_size
        self.workers = int(workers)
        self.checkpoint_interval = int(checkpoint_interval)
        self.start_alpha = float(start_alpha)
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
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative.")
        if self.start_alpha <= 0:
            raise ValueError("start_alpha must be positive.")
        duplicates = _duplicates(
            (
                self.dependent,
                self.offset,
                *self.fixed,
                *self.fixed_categorical,
                *self.random,
                *(() if self.group_id is None else (self.group_id,)),
            )
        )
        if duplicates:
            raise ValueError(f"Variables may not appear in multiple roles: {duplicates}")
        if self.intercept and "Intercept" in self.fixed:
            raise ValueError("Use intercept=True instead of including an 'Intercept' column.")

    @property
    def fixed_names(self) -> tuple[str, ...]:
        return (("Intercept",) if self.intercept else ()) + self.fixed

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "RandomParametersNegativeBinomial":
        return cls(
            dependent=spec.dependent,
            offset=spec.offset,
            fixed=spec.fixed,
            fixed_categorical=spec.fixed_categorical,
            random=spec.random,
            group_id=spec.group_id,
            intercept=spec.intercept,
            draws=spec.draws,
            draw_type=spec.draw_type,
            correlated_random_parameters=spec.correlated_random_parameters,
            seed=spec.seed,
            maxiter=spec.maxiter,
            tolerance=spec.tolerance,
            covariance=spec.covariance,
            chunk_size=spec.chunk_size,
            workers=spec.workers,
            checkpoint_interval=spec.checkpoint_interval,
            start_alpha=spec.start_alpha,
            output_dir=spec.output_dir,
            missing=spec.missing,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RandomParametersNegativeBinomial":
        return cls.from_spec(load_model_spec(path))

    def fit(
        self,
        data: str | Path | pd.DataFrame,
        save_run: bool = True,
        output_dir: str | Path | None = None,
        export: bool = False,
        resume_from: str | Path | None = None,
        spec_path: str | Path | None = None,
    ) -> RPNBResults:
        """Fit the model using simulated maximum likelihood."""

        fit_start = time.perf_counter()
        timing: dict[str, float | int] = {}
        run_dir = Path(resume_from) if resume_from is not None else (
            _create_run_dir(output_dir or self.output_dir) if save_run else None
        )
        logger = _run_logger(run_dir)
        logger.info("Starting RPNB estimation.")
        logger.info("Model specification: %s", self.to_spec_dict())
        resume_checkpoint = None
        resume_iteration = 0
        if resume_from is not None:
            resume_checkpoint = load_latest_checkpoint(resume_from)
            resume_iteration = resume_checkpoint.iteration
            logger.info(
                "Resuming from checkpoint %s at iteration %s with LL=%s.",
                resume_checkpoint.path,
                resume_checkpoint.iteration,
                resume_checkpoint.log_likelihood,
            )

        prepare_start = time.perf_counter()
        frame = _load_dataframe(data)
        work, missing_report = self._prepare_frame(frame, logger)
        preprocessing_summary = self._preprocessing_summary(frame, work)
        y = _count_array(work[self.dependent])
        offset = work[self.offset].astype(float).to_numpy()
        x_fixed, fixed_names = self._fixed_design(work)
        x_random = _matrix(work, self.random)
        q = len(self.random)
        group_codes, group_labels, group_indices, order, group_starts, group_counts = (
            self._group_indices(work, q > 0)
        )
        n_groups = len(group_indices)
        x_fixed_likelihood = x_fixed[order]
        x_random_likelihood = x_random[order]
        offset_likelihood = offset[order]
        y_likelihood = y[order]
        timing["data_preparation_seconds"] = time.perf_counter() - prepare_start

        draw_start = time.perf_counter()
        draws = generate_draws(n_groups, self.draws, q, self.draw_type, self.seed)
        timing["draw_generation_seconds"] = time.perf_counter() - draw_start

        logger.info(
            "Prepared %s observations, %s groups, %s random parameters.",
            len(work),
            n_groups,
            q,
        )

        start_params = self._start_params(y, offset, len(fixed_names))
        if resume_checkpoint is not None:
            if resume_checkpoint.params.size != start_params.size:
                raise ValueError(
                    "Checkpoint parameter vector length does not match this model "
                    f"({resume_checkpoint.params.size} != {start_params.size})."
                )
            start_params = resume_checkpoint.params.copy()
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
                state = self._unpack_params(theta, len(fixed_names))
                value = simulated_log_likelihood(
                    beta_fixed=state.fixed,
                    random_means=state.random_means,
                    random_sds=state.random_sds,
                    cholesky=state.cholesky,
                    alpha=state.alpha,
                    x_fixed=x_fixed_likelihood,
                    x_random=x_random_likelihood,
                    offset=offset_likelihood,
                    y=y_likelihood,
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

        if run_dir is not None:
            write_run_metadata(
                run_dir,
                {
                    "package": "rpnb",
                    "data_path": _data_path_for_metadata(data),
                    "spec_path": None if spec_path is None else str(Path(spec_path).resolve()),
                    "resume_from": None if resume_from is None else str(Path(resume_from).resolve()),
                    "checkpoint_interval": self.checkpoint_interval,
                },
            )

        def checkpoint_callback(iteration: int, params: np.ndarray) -> None:
            if run_dir is None or self.checkpoint_interval <= 0:
                return
            if iteration % self.checkpoint_interval != 0:
                return
            objective_value = objective(params)
            save_checkpoint(
                run_dir,
                iteration,
                params,
                objective_value,
                -objective_value,
                metadata={
                    "package": "rpnb",
                    "method": "BFGS",
                    "checkpoint_type": "iteration",
                    "function_evaluations": objective_calls,
                    "resumed_from_iteration": resume_iteration,
                },
            )
            logger.info("Saved checkpoint at iteration %s.", iteration)

        optimization_start = time.perf_counter()
        remaining_maxiter = max(self.maxiter - resume_iteration, 1)
        diagnostics = estimate_mle(
            objective,
            start_params,
            method="BFGS",
            maxiter=remaining_maxiter,
            tolerance=self.tolerance,
            callback=checkpoint_callback,
            initial_iteration=resume_iteration,
        )
        timing["optimization_seconds"] = time.perf_counter() - optimization_start
        timing["objective_calls"] = objective_calls
        timing["objective_seconds"] = objective_seconds
        timing["average_objective_seconds"] = (
            objective_seconds / objective_calls if objective_calls else 0.0
        )
        logger.info("Optimization finished: %s", diagnostics.message)
        total_iterations = (
            resume_iteration + diagnostics.iterations
            if diagnostics.iterations is not None
            else None
        )
        if run_dir is not None and self.checkpoint_interval > 0:
            save_checkpoint(
                run_dir,
                int(total_iterations or resume_iteration),
                diagnostics.params,
                diagnostics.objective_value,
                diagnostics.log_likelihood,
                metadata={
                    "package": "rpnb",
                    "method": "BFGS",
                    "checkpoint_type": "final",
                    "converged": diagnostics.converged,
                    "status": diagnostics.status,
                    "message": diagnostics.message,
                    "function_evaluations": objective_calls,
                    "resumed_from_iteration": resume_iteration,
                },
            )

        post_start = time.perf_counter()
        final_state = self._unpack_params(diagnostics.params, len(fixed_names))
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
            diagnostics.params, fixed_names
        )
        natural_covariance = None
        if internal_covariance is not None:
            jacobian = self._natural_jacobian(diagnostics.params, fixed_names)
            natural_covariance = jacobian @ internal_covariance @ jacobian.T

        parameter_table = build_parameter_table(
            names, components, variables, estimates, natural_covariance
        )

        predictions = predicted_counts(
            final_state.fixed,
            final_state.random_means,
            final_state.alpha,
            x_fixed,
            x_random,
            offset,
            group_codes,
            draws,
            random_sds=final_state.random_sds,
            cholesky=final_state.cholesky,
        )
        predictions.insert(0, "row_index", work.index.to_numpy())
        predictions.insert(1, self.dependent, y)
        predictions.insert(2, self.offset, offset)
        predictions.insert(3, "_group_code", group_codes)
        predictions.insert(4, "_group_label", group_labels[group_codes])

        effects = average_marginal_effects(
            final_state.fixed,
            final_state.random_means,
            x_fixed,
            x_random,
            offset,
            group_codes,
            draws,
            fixed_names,
            self.random,
            random_sds=final_state.random_sds,
            cholesky=final_state.cholesky,
        )

        n_params = diagnostics.params.size
        log_likelihood = diagnostics.log_likelihood
        fit_statistics = {
            "dependent": self.dependent,
            "offset": self.offset,
            "missing_policy": self.missing,
            "missing_checked_columns": ",".join(missing_report.checked_columns),
            "n_rows_original": int(missing_report.original_rows),
            "n_rows_removed_missing": int(missing_report.rows_removed),
            "n_rows_final_estimation_sample": int(missing_report.final_rows),
            "n_observations": int(len(work)),
            "n_groups": int(n_groups),
            "n_parameters": int(n_params),
            "log_likelihood": log_likelihood,
            "AIC": 2.0 * n_params - 2.0 * log_likelihood,
            "BIC": np.log(len(work)) * n_params - 2.0 * log_likelihood,
            "alpha": final_state.alpha,
            "draw_type": self.draw_type,
            "draws": int(self.draws if q else 1),
            "correlated_random_parameters": self.correlated_random_parameters,
            "intercept": self.intercept,
        }
        convergence = {
            "converged": diagnostics.converged,
            "status": diagnostics.status,
            "message": diagnostics.message,
            "iterations": total_iterations,
            "iterations_this_run": diagnostics.iterations,
            "resumed_from_iteration": resume_iteration,
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

        results = RPNBResults(
            parameter_table=parameter_table,
            fit_statistics=fit_statistics,
            convergence=convergence,
            preprocessing_summary=preprocessing_summary,
            predictions=predictions,
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
                "offset": self.offset,
                "fixed": _fixed_spec_for_output(
                    self.fixed,
                    self.fixed_categorical_specs,
                ),
                "random": {
                    item.name: {
                        "distribution": item.distribution,
                        "start_mean": item.start_mean,
                        "start_sd": item.start_sd,
                    }
                    for item in self.random_specs
                },
                "group_id": self.group_id,
                "intercept": self.intercept,
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
                "checkpoint_interval": self.checkpoint_interval,
                "start_alpha": self.start_alpha,
            },
            "output": {"directory": self.output_dir},
        }

    def _model_columns(self) -> tuple[str, ...]:
        columns = [
            self.dependent,
            self.offset,
            *self.fixed,
            *self.fixed_categorical,
            *self.random,
        ]
        if self.group_id is not None:
            columns.append(self.group_id)
        return tuple(dict.fromkeys(columns))

    def _column_roles(self) -> dict[str, str]:
        roles = {
            self.dependent: "dependent",
            self.offset: "offset",
        }
        roles.update({column: "fixed_continuous" for column in self.fixed})
        roles.update({column: "fixed_categorical" for column in self.fixed_categorical})
        roles.update({column: "random_continuous" for column in self.random})
        if self.group_id is not None:
            roles[self.group_id] = "group_id"
        return roles

    def _preprocessing_summary(
        self, raw_frame: pd.DataFrame, work: pd.DataFrame
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        roles = self._column_roles()
        categorical_specs = {spec.name: spec for spec in self.fixed_categorical_specs}
        numeric_roles = {
            "dependent",
            "offset",
            "fixed_continuous",
            "random_continuous",
        }

        for column in self._model_columns():
            role = roles[column]
            numeric_expected = role in numeric_roles
            raw_series = raw_frame[column]
            work_series = work[column]
            missing_count = int(_invalid_column_mask(raw_series, numeric_expected).sum())
            reference = ""
            dummy_names: list[str] = []

            if column in categorical_specs:
                spec = categorical_specs[column]
                reference = _format_report_value(spec.reference)
                dummy_names = _dummy_names_from_series(
                    work_series,
                    spec.name,
                    spec.reference,
                )

            row = _variable_summary_row(
                column=column,
                role=role,
                series=work_series,
                numeric_expected=numeric_expected,
                missing_count=missing_count,
                reference=reference,
                dummy_names=dummy_names,
            )
            rows.append(row)

            if column in categorical_specs:
                spec = categorical_specs[column]
                rows.extend(
                    _categorical_frequency_rows(
                        series=work_series,
                        variable=column,
                        role=role,
                        reference=spec.reference,
                        dummy_names=dummy_names,
                        missing_count=missing_count,
                    )
                )

        return pd.DataFrame(rows, columns=_PREPROCESSING_SUMMARY_COLUMNS)

    def _prepare_frame(
        self, frame: pd.DataFrame, logger: logging.Logger
    ) -> tuple[pd.DataFrame, MissingDataReport]:
        columns = list(self._model_columns())
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Data are missing required columns: {missing_columns}")

        work = frame.loc[:, columns].copy()
        invalid_mask = _invalid_model_data_mask(
            work,
            columns,
            numeric_columns=[self.dependent, self.offset, *self.fixed, *self.random],
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            if self.missing == "drop":
                work = work.loc[~invalid_mask].copy()
                logger.info("Dropped %s rows with missing or non-finite model data.", invalid_count)
            else:
                raise ValueError(
                    f"Found {invalid_count} rows with missing or non-finite model data "
                    "and missing!='drop'."
                )
        if len(work) == 0:
            raise ValueError("No usable observations remain after missing-data handling.")
        report = MissingDataReport(
            checked_columns=tuple(columns),
            original_rows=int(len(frame)),
            rows_removed=invalid_count,
            final_rows=int(len(work)),
        )

        _validate_count_series(work[self.dependent])
        numeric_columns = [self.offset, *self.fixed, *self.random]
        for column in numeric_columns:
            values = pd.to_numeric(work[column], errors="coerce")
            if values.isna().any():
                raise ValueError(f"Column {column!r} contains non-numeric values.")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Column {column!r} contains non-finite values.")
            work[column] = values
        for spec in self.fixed_categorical_specs:
            if not _series_contains_value(work[spec.name], spec.reference):
                raise ValueError(
                    f"Reference category {spec.reference!r} was not found in {spec.name!r}."
                )
        return work, report

    def _fixed_design(self, work: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
        pieces: list[np.ndarray] = []
        names: list[str] = []
        if self.intercept:
            pieces.append(np.ones((len(work), 1), dtype=float))
            names.append("Intercept")
        if self.fixed:
            pieces.append(_matrix(work, self.fixed))
            names.extend(self.fixed)
        for spec in self.fixed_categorical_specs:
            dummy_table, dummy_names = _dummy_code_categorical(
                work[spec.name],
                spec.name,
                spec.reference,
            )
            if dummy_table.size:
                pieces.append(dummy_table)
                names.extend(dummy_names)
        if not pieces:
            return np.zeros((len(work), 0), dtype=float), ()
        return np.column_stack(pieces), tuple(names)

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

    def _start_params(self, y: np.ndarray, offset: np.ndarray, n_fixed: int) -> np.ndarray:
        fixed = np.zeros(n_fixed, dtype=float)
        if self.intercept:
            mean_y = max(float(np.mean(y)), 1e-6)
            fixed[0] = np.log(mean_y) - float(np.mean(offset))
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

        alpha_start = _moment_alpha_start(y, self.start_alpha)
        pieces.append(np.array([np.log(alpha_start)], dtype=float))
        return np.concatenate(pieces)

    def _unpack_params(self, theta: np.ndarray, n_fixed: int) -> ParameterState:
        theta = np.asarray(theta, dtype=float)
        cursor = 0
        k = n_fixed
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

        alpha = float(np.exp(theta[cursor]))
        return ParameterState(fixed, random_means, alpha, random_sds, cholesky)

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
        self, theta: np.ndarray, fixed_names: Sequence[str]
    ) -> tuple[list[str], list[str], list[str], np.ndarray]:
        state = self._unpack_params(theta, len(fixed_names))
        names: list[str] = []
        components: list[str] = []
        variables: list[str] = []
        values: list[float] = []

        for name, value in zip(fixed_names, state.fixed):
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

        names.append("alpha")
        components.append("dispersion")
        variables.append("alpha")
        values.append(float(state.alpha))
        return names, components, variables, np.asarray(values, dtype=float)

    def _natural_jacobian(self, theta: np.ndarray, fixed_names: Sequence[str]) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        base = self._natural_parameters(theta, fixed_names)[3]
        jacobian = np.empty((base.size, theta.size), dtype=float)
        steps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(theta), 1.0)
        for column in range(theta.size):
            step_vector = np.zeros(theta.size, dtype=float)
            step_vector[column] = steps[column]
            plus = self._natural_parameters(theta + step_vector, fixed_names)[3]
            minus = self._natural_parameters(theta - step_vector, fixed_names)[3]
            jacobian[:, column] = (plus - minus) / (2.0 * steps[column])
        return jacobian


RPNBModel = RandomParametersNegativeBinomial


_PREPROCESSING_SUMMARY_COLUMNS = [
    "variable_name",
    "role",
    "section",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "number_missing",
    "number_unique_values",
    "reference_category",
    "generated_dummy_variables",
    "category_value",
    "category_count",
    "category_percent",
]


def _variable_summary_row(
    column: str,
    role: str,
    series: pd.Series,
    numeric_expected: bool,
    missing_count: int,
    reference: str,
    dummy_names: Sequence[str],
) -> dict[str, Any]:
    mean, std, minimum, maximum = _numeric_summary(series) if numeric_expected else (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
    return {
        "variable_name": column,
        "role": role,
        "section": "variable_summary",
        "mean": mean,
        "standard_deviation": std,
        "minimum": minimum,
        "maximum": maximum,
        "number_missing": int(missing_count),
        "number_unique_values": _number_unique_values(series, numeric_expected),
        "reference_category": reference,
        "generated_dummy_variables": ";".join(dummy_names),
        "category_value": "",
        "category_count": np.nan,
        "category_percent": np.nan,
    }


def _categorical_frequency_rows(
    series: pd.Series,
    variable: str,
    role: str,
    reference: Any,
    dummy_names: Sequence[str],
    missing_count: int,
) -> list[dict[str, Any]]:
    valid = _nonmissing_series(series, numeric_expected=False)
    categories = _ordered_categories(valid)
    total = len(valid)
    rows: list[dict[str, Any]] = []
    for category in categories:
        count = int(
            valid.map(lambda value, target=category: _values_equal(value, target)).sum()
        )
        rows.append(
            {
                "variable_name": variable,
                "role": role,
                "section": "categorical_frequency",
                "mean": np.nan,
                "standard_deviation": np.nan,
                "minimum": np.nan,
                "maximum": np.nan,
                "number_missing": int(missing_count),
                "number_unique_values": len(categories),
                "reference_category": _format_report_value(reference),
                "generated_dummy_variables": ";".join(dummy_names),
                "category_value": _format_report_value(category),
                "category_count": count,
                "category_percent": count / total if total else np.nan,
            }
        )
    return rows


def _dummy_names_from_series(
    series: pd.Series, variable: str, reference: Any
) -> list[str]:
    valid = _nonmissing_series(series, numeric_expected=False)
    categories = _ordered_categories(valid)
    return [
        f"{variable}_{_format_category_value(category)}"
        for category in categories
        if not _values_equal(category, reference)
    ]


def _numeric_summary(series: pd.Series) -> tuple[float, float, float, float]:
    values = _finite_numeric_values(series)
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    return (
        float(np.mean(values)),
        std,
        float(np.min(values)),
        float(np.max(values)),
    )


def _number_unique_values(series: pd.Series, numeric_expected: bool) -> int:
    if numeric_expected:
        return int(pd.Series(_finite_numeric_values(series)).nunique(dropna=True))
    valid = _nonmissing_series(series, numeric_expected=False)
    return int(valid.nunique(dropna=True))


def _finite_numeric_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _nonmissing_series(series: pd.Series, numeric_expected: bool) -> pd.Series:
    return series.loc[~_invalid_column_mask(series, numeric_expected)]


def _format_report_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


def _coerce_categorical_specs(
    categorical: Iterable[CategoricalVariableSpec | dict[str, Any]],
) -> list[CategoricalVariableSpec]:
    specs: list[CategoricalVariableSpec] = []
    for item in categorical:
        if isinstance(item, CategoricalVariableSpec):
            specs.append(item)
        elif isinstance(item, dict):
            if "name" in item and "reference" in item:
                specs.append(
                    CategoricalVariableSpec(
                        name=str(item["name"]),
                        reference=item["reference"],
                    )
                )
            elif len(item) == 1:
                name, config = next(iter(item.items()))
                if isinstance(config, dict):
                    if "reference" not in config:
                        raise ValueError(
                            f"Categorical variable {name!r} requires a reference value."
                        )
                    reference = config["reference"]
                else:
                    reference = config
                specs.append(CategoricalVariableSpec(name=str(name), reference=reference))
            else:
                raise ValueError(
                    "Categorical specs must include name/reference or one variable mapping."
                )
        else:
            raise ValueError("Categorical specs must be CategoricalVariableSpec or dict.")
    return specs


def _load_dataframe(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.read_csv(data)


def _data_path_for_metadata(data: str | Path | pd.DataFrame) -> str | None:
    if isinstance(data, pd.DataFrame):
        return None
    return str(Path(data).resolve())


def _fixed_matrix(work: pd.DataFrame, columns: Sequence[str], intercept: bool) -> np.ndarray:
    matrix = _matrix(work, columns)
    if not intercept:
        return matrix
    ones = np.ones((len(work), 1), dtype=float)
    if matrix.size == 0:
        return ones
    return np.column_stack([ones, matrix])


def _dummy_code_categorical(
    series: pd.Series, variable: str, reference: Any
) -> tuple[np.ndarray, list[str]]:
    categories = _ordered_categories(series)
    if not any(_values_equal(category, reference) for category in categories):
        raise ValueError(f"Reference category {reference!r} was not found in {variable!r}.")
    non_reference = [
        category for category in categories if not _values_equal(category, reference)
    ]
    if not non_reference:
        return np.zeros((len(series), 0), dtype=float), []
    columns = [
        series.map(lambda value, category=category: _values_equal(value, category))
        .astype(float)
        .to_numpy()
        for category in non_reference
    ]
    names = [f"{variable}_{_format_category_value(category)}" for category in non_reference]
    return np.column_stack(columns), names


def _ordered_categories(series: pd.Series) -> list[Any]:
    values = list(pd.unique(series.dropna()))
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _series_contains_value(series: pd.Series, reference: Any) -> bool:
    return any(_values_equal(value, reference) for value in pd.unique(series.dropna()))


def _values_equal(left: Any, right: Any) -> bool:
    return bool(left == right)


def _format_category_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    text = text.strip().replace(" ", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def _fixed_spec_for_output(
    fixed: Sequence[str], categorical: Sequence[CategoricalVariableSpec]
) -> list[str] | dict[str, Any]:
    if not categorical:
        return list(fixed)
    return {
        "continuous": list(fixed),
        "categorical": {
            item.name: {"reference": item.reference}
            for item in categorical
        },
    }


def _matrix(work: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(work), 0), dtype=float)
    return work.loc[:, columns].astype(float).to_numpy()


def _invalid_model_data_mask(
    work: pd.DataFrame,
    columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> pd.Series:
    invalid = pd.Series(False, index=work.index)
    numeric_column_set = set(numeric_columns)
    for column in columns:
        invalid = invalid | _invalid_column_mask(
            work[column],
            numeric_expected=column in numeric_column_set,
        )

    return invalid


def _invalid_column_mask(series: pd.Series, numeric_expected: bool) -> pd.Series:
    invalid = series.isna()
    invalid = invalid | series.map(
        lambda value: isinstance(value, str) and value.strip() == ""
    )
    invalid = invalid | series.map(_is_infinite_scalar)
    if numeric_expected:
        numeric = pd.to_numeric(series, errors="coerce")
        non_finite = pd.Series(
            ~np.isfinite(numeric.to_numpy(dtype=float)),
            index=series.index,
        )
        invalid = invalid | numeric.isna() | non_finite
    return invalid


def _is_infinite_scalar(value: Any) -> bool:
    try:
        return bool(np.isinf(value))
    except (TypeError, ValueError):
        return False


def _validate_count_series(series: pd.Series) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError("Dependent count variable contains non-numeric values.")
    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Dependent count variable contains non-finite values.")
    if np.any(arr < 0):
        raise ValueError("Dependent count variable must be non-negative.")
    if not np.allclose(arr, np.round(arr)):
        raise ValueError("Dependent count variable must contain integer counts.")


def _count_array(series: pd.Series) -> np.ndarray:
    _validate_count_series(series)
    return np.round(series.astype(float).to_numpy()).astype(int)


def _moment_alpha_start(y: np.ndarray, fallback: float) -> float:
    mean_y = float(np.mean(y))
    var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    if mean_y > 0:
        alpha = (var_y - mean_y) / (mean_y**2)
        if np.isfinite(alpha) and alpha > 1e-4:
            return float(np.clip(alpha, 1e-4, 10.0))
    return float(max(fallback, 1e-4))


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _create_run_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"rpnb_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _run_logger(run_dir: Path | None) -> logging.Logger:
    name = f"rpnb.run.{id(run_dir)}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if run_dir is not None:
        handler: logging.Handler = logging.FileHandler(
            run_dir / "rpnb.log", encoding="utf-8"
        )
    else:
        handler = logging.NullHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
