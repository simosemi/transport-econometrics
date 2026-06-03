"""Random Parameters Ordered Probit estimator."""

from __future__ import annotations

import logging
import json
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

from rpopit.checkpoint import (
    load_latest_checkpoint,
    save_checkpoint,
    write_run_metadata,
)
from rpopit.config import (
    CategoricalVariableSpec,
    DerivedCategoricalBinSpec,
    DerivedCategoricalSpec,
    ModelSpec,
    RandomCategoricalVariableSpec,
    RandomParameterSpec,
    load_model_spec,
)
from rpopit.draws import generate_draws
from rpopit.likelihood import simulated_log_likelihood
from rpopit.marginal_effects import average_marginal_effects, predicted_probabilities
from rpopit.optimizer import (
    convergence_quality,
    covariance_from_hessian,
    estimate_mle,
    finite_difference_hessian,
    matrix_condition_number,
    normalize_optimizer,
    optimizer_termination_report,
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
        fixed_categorical: Sequence[CategoricalVariableSpec | dict[str, Any]] | None = None,
        random: Sequence[str | RandomParameterSpec | dict[str, Any]] | None = None,
        random_categorical: Sequence[
            RandomCategoricalVariableSpec | dict[str, Any]
        ] | None = None,
        derived_categorical: Sequence[DerivedCategoricalSpec | dict[str, Any]] | None = None,
        group_id: str | None = None,
        categories: Sequence[Any] | None = None,
        draws: int = 200,
        draw_type: str = "halton",
        correlated_random_parameters: bool = False,
        seed: int | None = 12345,
        maxiter: int = 1000,
        tolerance: float = 1e-4,
        optimizer: str = "bfgs",
        multistart: int = 1,
        multistart_random_seed: int | None = 12345,
        multistart_scale: float = 0.25,
        covariance: str = "bfgs",
        chunk_size: int | None = 10_000,
        workers: int = 1,
        checkpoint_interval: int = 10,
        output_dir: str = "runs",
        missing: str = "drop",
    ) -> None:
        self.dependent = dependent
        self.fixed = tuple(fixed or ())
        self.fixed_categorical_specs = tuple(
            _coerce_categorical_specs(fixed_categorical or ())
        )
        self.fixed_categorical = tuple(item.name for item in self.fixed_categorical_specs)
        self.random_continuous_specs = tuple(_coerce_random_specs(random or ()))
        self.random_continuous = tuple(item.name for item in self.random_continuous_specs)
        self.random_categorical_specs = tuple(
            _coerce_random_categorical_specs(random_categorical or ())
        )
        self.random_categorical = tuple(item.name for item in self.random_categorical_specs)
        self.derived_categorical_specs = tuple(
            _coerce_derived_categorical_specs(derived_categorical or ())
        )
        self.derived_categorical = tuple(item.name for item in self.derived_categorical_specs)
        self.derived_source_columns = tuple(
            dict.fromkeys(item.source for item in self.derived_categorical_specs)
        )
        self.random_specs = self.random_continuous_specs
        self.random = tuple(item.name for item in self.random_specs)
        self.group_id = group_id
        self.categories = None if categories is None else tuple(categories)
        self.draws = int(draws)
        self.draw_type = draw_type
        self.correlated_random_parameters = bool(correlated_random_parameters)
        self.seed = seed
        self.maxiter = int(maxiter)
        self.tolerance = float(tolerance)
        self.optimizer = normalize_optimizer(optimizer)
        self.multistart = int(multistart)
        self.multistart_random_seed = multistart_random_seed
        self.multistart_scale = float(multistart_scale)
        self.covariance = covariance.lower()
        self.chunk_size = chunk_size
        self.workers = int(workers)
        self.checkpoint_interval = int(checkpoint_interval)
        self.output_dir = output_dir
        self.missing = missing.lower()

        if self.draws < 1:
            raise ValueError("draws must be at least 1.")
        if self.multistart < 1:
            raise ValueError("multistart must be at least 1.")
        if self.multistart_scale < 0:
            raise ValueError("multistart_scale must be non-negative.")
        if self.covariance not in {"bfgs", "hessian"}:
            raise ValueError("covariance must be 'bfgs' or 'hessian'.")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be positive or None.")
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative.")
        _validate_ordered_model_roles(
            dependent=self.dependent,
            fixed_names=self.fixed,
            fixed_categorical_names=self.fixed_categorical,
            random_names=self.random_continuous,
            random_categorical_names=self.random_categorical,
            group_id=self.group_id,
        )

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "RandomParametersOrderedProbit":
        return cls(
            dependent=spec.dependent,
            fixed=spec.fixed,
            fixed_categorical=spec.fixed_categorical,
            random=spec.random,
            random_categorical=spec.random_categorical,
            derived_categorical=spec.derived_categorical,
            group_id=spec.group_id,
            categories=spec.categories,
            draws=spec.draws,
            draw_type=spec.draw_type,
            correlated_random_parameters=spec.correlated_random_parameters,
            seed=spec.seed,
            maxiter=spec.maxiter,
            tolerance=spec.tolerance,
            optimizer=spec.optimizer,
            multistart=spec.multistart,
            multistart_random_seed=spec.multistart_random_seed,
            multistart_scale=spec.multistart_scale,
            covariance=spec.covariance,
            chunk_size=spec.chunk_size,
            workers=spec.workers,
            checkpoint_interval=spec.checkpoint_interval,
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
        resume_from: str | Path | None = None,
        spec_path: str | Path | None = None,
    ) -> RPOpitResults:
        """Fit the model using simulated maximum likelihood."""

        fit_start = time.perf_counter()
        timing: dict[str, float | int] = {}
        run_dir = Path(resume_from) if resume_from is not None else (
            _create_run_dir(output_dir or self.output_dir) if save_run else None
        )
        logger = _run_logger(run_dir)
        logger.info("Starting rpopit estimation.")
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
        work = self._prepare_frame(frame, logger)
        y_codes, categories = self._encode_dependent(work[self.dependent])
        x_fixed, fixed_names = self._fixed_design(work)
        x_random, random_names, random_specs = self._random_design(work)
        self.random = random_names
        self.random_specs = random_specs
        q = len(random_names)
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

        start_params = self._start_params(y_codes, len(categories), len(fixed_names))
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
                state = self._unpack_params(theta, len(fixed_names), len(categories) - 1)
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

        if run_dir is not None:
            write_run_metadata(
                run_dir,
                {
                    "package": "rpopit",
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
                    "package": "rpopit",
                    "method": self.optimizer,
                    "checkpoint_type": "iteration",
                    "function_evaluations": objective_calls,
                    "resumed_from_iteration": resume_iteration,
                },
            )
            logger.info("Saved checkpoint at iteration %s.", iteration)

        optimization_start = time.perf_counter()
        remaining_maxiter = max(self.maxiter - resume_iteration, 1)
        effective_multistart = 1 if resume_checkpoint is not None else self.multistart
        start_vectors = _multistart_vectors(
            start_params,
            effective_multistart,
            self.multistart_random_seed,
            self.multistart_scale,
        )
        multistart_records: list[dict[str, Any]] = []
        for start_id, candidate_start in enumerate(start_vectors, start=1):
            logger.info(
                "Optimizing start %s of %s with %s.",
                start_id,
                effective_multistart,
                self.optimizer,
            )
            starting_log_likelihood = -objective(candidate_start)
            start_diagnostics = estimate_mle(
                objective,
                candidate_start,
                method=self.optimizer,
                maxiter=remaining_maxiter,
                tolerance=self.tolerance,
                callback=checkpoint_callback if effective_multistart == 1 else None,
                initial_iteration=resume_iteration if resume_checkpoint is not None else 0,
            )
            multistart_records.append(
                {
                    "start_id": start_id,
                    "starting_params": candidate_start.copy(),
                    "starting_log_likelihood": starting_log_likelihood,
                    "diagnostics": start_diagnostics,
                }
            )

        best_record = max(
            multistart_records,
            key=lambda item: item["diagnostics"].log_likelihood,
        )
        diagnostics = best_record["diagnostics"]
        selected_start_id = int(best_record["start_id"])
        multistart_summary = _multistart_summary_table(
            multistart_records,
            selected_start_id,
            self.optimizer,
            n_parameters=diagnostics.params.size,
            n_observations=len(work),
            tolerance=self.tolerance,
        )
        local_solutions = self._local_solutions_table(
            multistart_records,
            selected_start_id,
            fixed_names,
            categories,
        )
        timing["optimization_seconds"] = time.perf_counter() - optimization_start
        timing["objective_calls"] = objective_calls
        timing["objective_seconds"] = objective_seconds
        timing["average_objective_seconds"] = (
            objective_seconds / objective_calls if objective_calls else 0.0
        )
        logger.info(
            "Optimization finished from start %s: %s",
            selected_start_id,
            diagnostics.message,
        )
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
                    "package": "rpopit",
                    "method": self.optimizer,
                    "checkpoint_type": "final",
                    "converged": diagnostics.converged,
                    "status": diagnostics.status,
                    "message": diagnostics.message,
                    "function_evaluations": objective_calls,
                    "resumed_from_iteration": resume_iteration,
                },
            )

        post_start = time.perf_counter()
        final_state = self._unpack_params(
            diagnostics.params, len(fixed_names), len(categories) - 1
        )
        internal_covariance = diagnostics.hess_inv
        hessian_condition_number = diagnostics.hessian_condition_number
        if self.covariance == "hessian":
            logger.info("Computing finite-difference Hessian covariance.")
            try:
                hessian = finite_difference_hessian(objective, diagnostics.params)
                hessian_condition_number = matrix_condition_number(hessian)
                internal_covariance = covariance_from_hessian(hessian)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Falling back to BFGS covariance: %s", exc)
        if process_pool is not None:
            process_pool.shutdown()

        names, components, variables, estimates = self._natural_parameters(
            diagnostics.params, fixed_names, categories
        )
        natural_covariance = None
        if internal_covariance is not None:
            jacobian = self._natural_jacobian(diagnostics.params, fixed_names, categories)
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
            fixed_names,
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
            "multistart_requested": self.multistart,
            "multistart_completed": int(len(multistart_records)),
            "selected_start_id": selected_start_id,
        }
        termination_report = optimizer_termination_report(
            diagnostics.converged,
            diagnostics.status,
            diagnostics.message,
            hessian_condition_number,
        )
        selected_convergence_quality = convergence_quality(
            diagnostics.converged,
            diagnostics.gradient_norm,
            self.tolerance,
        )
        convergence = {
            "converged": diagnostics.converged,
            "status": diagnostics.status,
            "message": diagnostics.message,
            "convergence_quality": selected_convergence_quality,
            "optimizer_method": diagnostics.method,
            "optimizer_status_code": diagnostics.status,
            "optimizer_message": diagnostics.message,
            "convergence_code": diagnostics.status,
            "convergence_message": diagnostics.message,
            "iterations": total_iterations,
            "iterations_this_run": diagnostics.iterations,
            "multistart_total_iterations": _sum_optional_ints(
                record["diagnostics"].iterations for record in multistart_records
            ),
            "selected_start_id": selected_start_id,
            "resumed_from_iteration": resume_iteration,
            "function_evaluations": diagnostics.function_evaluations,
            "gradient_norm": diagnostics.gradient_norm,
            "hessian_condition_number": hessian_condition_number,
            "largest_parameter_magnitude": diagnostics.largest_parameter_magnitude,
            "smallest_parameter_magnitude": diagnostics.smallest_parameter_magnitude,
            **termination_report,
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
            multistart_summary=multistart_summary,
            local_solutions=local_solutions,
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
                "fixed": _fixed_spec_for_output(
                    self.fixed,
                    self.fixed_categorical_specs,
                ),
                "random": _random_spec_for_output(
                    self.random_continuous_specs,
                    self.random_categorical_specs,
                ),
                "derived_categorical": _derived_spec_for_output(
                    self.derived_categorical_specs
                ),
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
                "optimizer": self.optimizer,
                "multistart": self.multistart,
                "random_seed": self.multistart_random_seed,
                "multistart_seed": self.multistart_random_seed,
                "multistart_scale": self.multistart_scale,
                "covariance": self.covariance,
                "chunk_size": self.chunk_size,
                "workers": self.workers,
                "checkpoint_interval": self.checkpoint_interval,
            },
            "output": {"directory": self.output_dir},
        }

    def _prepare_frame(self, frame: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
        columns = list(self._raw_model_columns())
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
        numeric_columns = [*self.fixed, *self.random_continuous, *self.derived_source_columns]
        for column in numeric_columns:
            values = pd.to_numeric(work[column], errors="coerce")
            if values.isna().any():
                raise ValueError(f"Column {column!r} contains non-numeric values.")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Column {column!r} contains non-finite values.")
            work[column] = values
        self._materialize_derived_categorical(work)
        for spec in self.fixed_categorical_specs:
            if not _series_contains_value(work[spec.name], spec.reference):
                raise ValueError(
                    f"Reference category {spec.reference!r} was not found in {spec.name!r}."
                )
        for spec in self.random_categorical_specs:
            if not _series_contains_value(work[spec.name], spec.reference):
                raise ValueError(
                    f"Reference category {spec.reference!r} was not found in {spec.name!r}."
                )
        return work

    def _raw_model_columns(self) -> tuple[str, ...]:
        derived_names = set(self.derived_categorical)
        columns = [
            self.dependent,
            *self.fixed,
            *(name for name in self.fixed_categorical if name not in derived_names),
            *self.random_continuous,
            *(name for name in self.random_categorical if name not in derived_names),
            *self.derived_source_columns,
        ]
        if self.group_id is not None:
            columns.append(self.group_id)
        return tuple(dict.fromkeys(columns))

    def _materialize_derived_categorical(self, work: pd.DataFrame) -> None:
        for spec in self.derived_categorical_specs:
            values = pd.to_numeric(work[spec.source], errors="coerce")
            if values.isna().any():
                raise ValueError(f"Derived categorical source {spec.source!r} contains non-numeric values.")
            work[spec.name] = _derive_categorical_series(values, spec)

    def _fixed_design(self, work: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
        pieces: list[np.ndarray] = []
        names: list[str] = []
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

    def _random_design(
        self, work: pd.DataFrame
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[RandomParameterSpec, ...]]:
        pieces: list[np.ndarray] = []
        names: list[str] = []
        specs: list[RandomParameterSpec] = []
        if self.random_continuous:
            pieces.append(_matrix(work, self.random_continuous))
            names.extend(self.random_continuous)
            specs.extend(self.random_continuous_specs)
        for spec in self.random_categorical_specs:
            dummy_table, dummy_names = _dummy_code_categorical(
                work[spec.name],
                spec.name,
                spec.reference,
            )
            if dummy_table.size:
                pieces.append(dummy_table)
                names.extend(dummy_names)
                specs.extend(
                    RandomParameterSpec(
                        name=dummy_name,
                        distribution=spec.distribution,
                        start_mean=spec.start_mean,
                        start_sd=spec.start_sd,
                    )
                    for dummy_name in dummy_names
                )
        if not pieces:
            return np.zeros((len(work), 0), dtype=float), (), ()
        return np.column_stack(pieces), tuple(names), tuple(specs)

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

    def _start_params(
        self, y_codes: np.ndarray, n_categories: int, n_fixed: int
    ) -> np.ndarray:
        fixed = np.zeros(n_fixed, dtype=float)
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

    def _unpack_params(
        self, theta: np.ndarray, n_fixed: int, n_thresholds: int
    ) -> ParameterState:
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
        self, theta: np.ndarray, fixed_names: Sequence[str], categories: Sequence[Any]
    ) -> tuple[list[str], list[str], list[str], np.ndarray]:
        state = self._unpack_params(theta, len(fixed_names), len(categories) - 1)
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

        for threshold_number, value in enumerate(state.thresholds, start=1):
            lower = categories[threshold_number - 1]
            upper = categories[threshold_number]
            names.append(f"threshold[{threshold_number}]")
            components.append("threshold")
            variables.append(f"{lower}|{upper}")
            values.append(float(value))

        return names, components, variables, np.asarray(values, dtype=float)

    def _natural_jacobian(
        self, theta: np.ndarray, fixed_names: Sequence[str], categories: Sequence[Any]
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        base = self._natural_parameters(theta, fixed_names, categories)[3]
        jacobian = np.empty((base.size, theta.size), dtype=float)
        steps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(theta), 1.0)
        for column in range(theta.size):
            step_vector = np.zeros(theta.size, dtype=float)
            step_vector[column] = steps[column]
            plus = self._natural_parameters(theta + step_vector, fixed_names, categories)[3]
            minus = self._natural_parameters(theta - step_vector, fixed_names, categories)[3]
            jacobian[:, column] = (plus - minus) / (2.0 * steps[column])
        return jacobian

    def _local_solutions_table(
        self,
        records: Sequence[dict[str, Any]],
        selected_start_id: int,
        fixed_names: Sequence[str],
        categories: Sequence[Any],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in records:
            start_id = int(record["start_id"])
            diagnostics = record["diagnostics"]
            names, components, variables, estimates = self._natural_parameters(
                diagnostics.params,
                fixed_names,
                categories,
            )
            for name, component, variable, estimate in zip(
                names,
                components,
                variables,
                estimates,
            ):
                rows.append(
                    {
                        "start_id": start_id,
                        "is_best": start_id == selected_start_id,
                        "final_log_likelihood": diagnostics.log_likelihood,
                        "optimizer": diagnostics.method,
                        "parameter": name,
                        "component": component,
                        "variable": variable,
                        "estimate": float(estimate),
                    }
                )
        return pd.DataFrame(rows)


RPOpitModel = RandomParametersOrderedProbit


def _multistart_vectors(
    base: np.ndarray,
    count: int,
    seed: int | None,
    scale_factor: float,
) -> list[np.ndarray]:
    base = np.asarray(base, dtype=float)
    starts = [base.copy()]
    if count <= 1:
        return starts
    rng = np.random.default_rng(seed)
    scale = float(scale_factor) * np.maximum(np.abs(base), 1.0)
    for _ in range(1, count):
        starts.append(base + rng.normal(loc=0.0, scale=scale, size=base.size))
    return starts


def _multistart_summary_table(
    records: Sequence[dict[str, Any]],
    selected_start_id: int,
    optimizer: str,
    n_parameters: int,
    n_observations: int,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        start_id = int(record["start_id"])
        diagnostics = record["diagnostics"]
        log_likelihood = float(diagnostics.log_likelihood)
        rows.append(
            {
                "start_id": start_id,
                "is_best": start_id == selected_start_id,
                "optimizer": optimizer,
                "starting_log_likelihood": float(record["starting_log_likelihood"]),
                "final_log_likelihood": log_likelihood,
                "AIC": 2.0 * n_parameters - 2.0 * log_likelihood,
                "BIC": np.log(n_observations) * n_parameters - 2.0 * log_likelihood,
                "converged": diagnostics.converged,
                "convergence_quality": convergence_quality(
                    diagnostics.converged,
                    diagnostics.gradient_norm,
                    tolerance,
                ),
                "status": diagnostics.status,
                "message": diagnostics.message,
                "iterations": diagnostics.iterations,
                "function_evaluations": diagnostics.function_evaluations,
                "gradient_norm": diagnostics.gradient_norm,
                "hessian_condition_number": diagnostics.hessian_condition_number,
                "starting_parameter_vector": _format_parameter_vector(
                    record["starting_params"]
                ),
                "final_parameter_vector": _format_parameter_vector(diagnostics.params),
            }
        )
    return pd.DataFrame(rows)


def _format_parameter_vector(params: np.ndarray) -> str:
    return json.dumps([float(value) for value in np.asarray(params, dtype=float)])


def _sum_optional_ints(values: Iterable[int | None]) -> int:
    return int(sum(0 if value is None else int(value) for value in values))


def _coerce_random_specs(
    random: Iterable[str | RandomParameterSpec | dict[str, Any]],
) -> list[RandomParameterSpec]:
    specs: list[RandomParameterSpec] = []
    for item in random:
        if isinstance(item, RandomParameterSpec):
            specs.append(item)
        elif isinstance(item, dict):
            if "name" in item:
                name = item["name"]
                config = {key: value for key, value in item.items() if key != "name"}
            elif len(item) == 1:
                name, config = next(iter(item.items()))
            else:
                raise ValueError(
                    "Random parameter specs must include name or one variable mapping."
                )
            if isinstance(config, str):
                config = {"distribution": config}
            elif config is None:
                config = {}
            elif not isinstance(config, dict):
                raise ValueError(f"Invalid random parameter specification for {name!r}.")
            specs.append(
                RandomParameterSpec(
                    name=str(name),
                    distribution=str(config.get("distribution", config.get("dist", "normal"))),
                    start_mean=float(config.get("start_mean", config.get("mean", 0.0))),
                    start_sd=float(config.get("start_sd", config.get("sd", 0.3))),
                )
            )
        else:
            specs.append(RandomParameterSpec(name=str(item)))
    return specs


def _coerce_random_categorical_specs(
    categorical: Iterable[RandomCategoricalVariableSpec | dict[str, Any]],
) -> list[RandomCategoricalVariableSpec]:
    specs: list[RandomCategoricalVariableSpec] = []
    for item in categorical:
        if isinstance(item, RandomCategoricalVariableSpec):
            specs.append(item)
        elif isinstance(item, dict):
            if "name" in item and "reference" in item:
                specs.append(
                    RandomCategoricalVariableSpec(
                        name=str(item["name"]),
                        reference=item["reference"],
                        distribution=str(item.get("distribution", item.get("dist", "normal"))),
                        start_mean=float(item.get("start_mean", item.get("mean", 0.0))),
                        start_sd=float(item.get("start_sd", item.get("sd", 0.3))),
                    )
                )
            elif len(item) == 1:
                name, config = next(iter(item.items()))
                if not isinstance(config, dict):
                    config = {"reference": config}
                if "reference" not in config:
                    raise ValueError(
                        f"Random categorical variable {name!r} requires a reference value."
                    )
                specs.append(
                    RandomCategoricalVariableSpec(
                        name=str(name),
                        reference=config["reference"],
                        distribution=str(config.get("distribution", config.get("dist", "normal"))),
                        start_mean=float(config.get("start_mean", config.get("mean", 0.0))),
                        start_sd=float(config.get("start_sd", config.get("sd", 0.3))),
                    )
                )
            else:
                raise ValueError(
                    "Random categorical specs must include name/reference or one variable mapping."
                )
        else:
            raise ValueError(
                "Random categorical specs must be RandomCategoricalVariableSpec or dict."
            )
    return specs


def _coerce_derived_categorical_specs(
    derived: Iterable[DerivedCategoricalSpec | dict[str, Any]],
) -> list[DerivedCategoricalSpec]:
    specs: list[DerivedCategoricalSpec] = []
    for item in derived:
        if isinstance(item, DerivedCategoricalSpec):
            specs.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("Derived categorical specs must be DerivedCategoricalSpec or dict.")
        if "name" in item and "source" in item:
            name = item["name"]
            config = {key: value for key, value in item.items() if key != "name"}
        elif len(item) == 1:
            name, config = next(iter(item.items()))
            if not isinstance(config, dict):
                raise ValueError(f"Derived categorical variable {name!r} must be a mapping.")
        else:
            raise ValueError("Derived categorical specs must include name/source or one mapping.")
        specs.append(_derived_spec_from_config(str(name), config))
    return specs


def _derived_spec_from_config(name: str, config: dict[str, Any]) -> DerivedCategoricalSpec:
    source = config.get("source")
    if source is None:
        raise ValueError(f"Derived categorical variable {name!r} requires a source.")
    method = str(config.get("method", "bins")).lower()
    if method == "quantile":
        bins = int(config.get("bins", 0))
        if bins < 2:
            raise ValueError(f"Derived quantile variable {name!r} requires bins >= 2.")
        return DerivedCategoricalSpec(
            name=name,
            source=str(source),
            method="quantile",
            quantile_bins=bins,
        )
    bins_raw = config.get("bins")
    if not isinstance(bins_raw, list) or not bins_raw:
        raise ValueError(f"Derived categorical variable {name!r} requires non-empty bins.")
    bins = []
    previous_upper: float | None = None
    open_ended_seen = False
    for index, raw_bin in enumerate(bins_raw):
        if not isinstance(raw_bin, dict) or "label" not in raw_bin:
            raise ValueError(f"Derived bin {index + 1} for {name!r} requires a label.")
        if open_ended_seen:
            raise ValueError(f"Open-ended derived bin for {name!r} must be last.")
        upper = raw_bin.get("upper")
        if upper is None:
            open_ended_seen = True
            bins.append(DerivedCategoricalBinSpec(label=str(raw_bin["label"]), upper=None))
            continue
        upper_value = float(upper)
        if previous_upper is not None and upper_value <= previous_upper:
            raise ValueError(f"Derived bins for {name!r} must have increasing upper values.")
        previous_upper = upper_value
        bins.append(DerivedCategoricalBinSpec(label=str(raw_bin["label"]), upper=upper_value))
    return DerivedCategoricalSpec(
        name=name,
        source=str(source),
        method="bins",
        bins=tuple(bins),
    )


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


def _matrix(work: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(work), 0), dtype=float)
    return work.loc[:, columns].astype(float).to_numpy()


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


def _derive_categorical_series(
    values: pd.Series,
    spec: DerivedCategoricalSpec,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if spec.method == "quantile":
        bins = int(spec.quantile_bins or 0)
        labels = [f"Q{index}" for index in range(1, bins + 1)]
        ranked = numeric.rank(method="first")
        return pd.qcut(ranked, q=bins, labels=labels).astype(str)

    labels: list[str] = []
    for value in numeric:
        matched = False
        for bin_spec in spec.bins:
            if bin_spec.upper is None or value <= bin_spec.upper:
                labels.append(bin_spec.label)
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Derived categorical variable {spec.name!r} does not cover value {value}."
            )
    return pd.Series(labels, index=values.index, name=spec.name)


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


def _random_spec_for_output(
    continuous: Sequence[RandomParameterSpec],
    categorical: Sequence[RandomCategoricalVariableSpec],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if continuous:
        output["continuous"] = {
            item.name: {
                "distribution": item.distribution,
                "start_mean": item.start_mean,
                "start_sd": item.start_sd,
            }
            for item in continuous
        }
    if categorical:
        output["categorical"] = {
            item.name: {
                "reference": item.reference,
                "distribution": item.distribution,
                "start_mean": item.start_mean,
                "start_sd": item.start_sd,
            }
            for item in categorical
        }
    return output


def _derived_spec_for_output(
    specs: Sequence[DerivedCategoricalSpec],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for spec in specs:
        if spec.method == "quantile":
            output[spec.name] = {
                "source": spec.source,
                "method": "quantile",
                "bins": spec.quantile_bins,
            }
        else:
            output[spec.name] = {
                "source": spec.source,
                "bins": [
                    (
                        {"label": bin_spec.label}
                        if bin_spec.upper is None
                        else {"upper": bin_spec.upper, "label": bin_spec.label}
                    )
                    for bin_spec in spec.bins
                ],
            }
    return output


def _validate_ordered_model_roles(
    dependent: str,
    fixed_names: tuple[str, ...],
    fixed_categorical_names: tuple[str, ...],
    random_names: tuple[str, ...],
    random_categorical_names: tuple[str, ...],
    group_id: str | None,
) -> None:
    for variable in fixed_names:
        if variable in random_names:
            raise ValueError(
                f"Variable {variable} is already modeled as a random parameter. "
                f"Its mean effect is estimated as beta_random_mean[{variable}]; "
                "do not also list it as fixed."
            )
    for variable in fixed_categorical_names:
        if variable in random_categorical_names:
            raise ValueError(
                f"Categorical variable {variable} is already modeled as a random parameter. "
                f"Its generated dummy mean effects are estimated as beta_random_mean[{variable}_value]; "
                "do not also list it as fixed."
            )
    duplicates = _duplicates(
        (
            dependent,
            *fixed_names,
            *fixed_categorical_names,
            *random_names,
            *random_categorical_names,
            *(() if group_id is None else (group_id,)),
        )
    )
    if duplicates:
        raise ValueError(f"Variables may not appear in multiple roles: {duplicates}")


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


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
