"""YAML model specification parsing for RPNB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RandomParameterSpec:
    """Configuration for one normally distributed random coefficient."""

    name: str
    distribution: str = "normal"
    start_mean: float = 0.0
    start_sd: float = 0.25

    def __post_init__(self) -> None:
        if self.distribution.lower() != "normal":
            raise ValueError(
                f"Unsupported random parameter distribution for {self.name!r}: "
                f"{self.distribution!r}. RPNB currently supports normal only."
            )
        if self.start_sd <= 0:
            raise ValueError(f"start_sd for {self.name!r} must be positive.")


@dataclass(frozen=True)
class CategoricalVariableSpec:
    """Configuration for one fixed categorical variable."""

    name: str
    reference: Any


@dataclass(frozen=True)
class RandomCategoricalVariableSpec:
    """Configuration for one categorical variable with random dummy coefficients."""

    name: str
    reference: Any
    distribution: str = "normal"
    start_mean: float = 0.0
    start_sd: float = 0.25

    def __post_init__(self) -> None:
        if self.distribution.lower() != "normal":
            raise ValueError(
                f"Unsupported random parameter distribution for {self.name!r}: "
                f"{self.distribution!r}. RPNB currently supports normal only."
            )
        if self.start_sd <= 0:
            raise ValueError(f"start_sd for {self.name!r} must be positive.")


@dataclass(frozen=True)
class DerivedCategoricalBinSpec:
    """One derived-category bin definition."""

    label: str
    upper: float | None = None


@dataclass(frozen=True)
class DerivedCategoricalSpec:
    """Configuration for a categorical variable derived from a numeric column."""

    name: str
    source: str
    method: str = "bins"
    bins: tuple[DerivedCategoricalBinSpec, ...] = ()
    quantile_bins: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Complete model, simulation, estimation, and output specification."""

    dependent: str
    offset: str | None = None
    fixed: tuple[str, ...] = ()
    fixed_categorical: tuple[CategoricalVariableSpec, ...] = ()
    random: tuple[RandomParameterSpec, ...] = ()
    random_categorical: tuple[RandomCategoricalVariableSpec, ...] = ()
    derived_categorical: tuple[DerivedCategoricalSpec, ...] = ()
    group_id: str | None = None
    intercept: bool = True
    draws: int = 200
    draw_type: str = "halton"
    correlated_random_parameters: bool = False
    seed: int | None = 12345
    maxiter: int = 1000
    tolerance: float = 1e-5
    optimizer: str = "bfgs"
    multistart: int = 1
    multistart_random_seed: int | None = 12345
    multistart_scale: float = 0.25
    auto_simplify_random_parameters: bool = False
    covariance: str = "bfgs"
    chunk_size: int | None = 10_000
    workers: int = 1
    checkpoint_interval: int = 10
    start_alpha: float = 0.5
    output_dir: str = "runs"
    missing: str = "drop"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fixed_categorical"] = [asdict(item) for item in self.fixed_categorical]
        data["random"] = [asdict(item) for item in self.random]
        data["random_categorical"] = [
            asdict(item) for item in self.random_categorical
        ]
        data["derived_categorical"] = [asdict(item) for item in self.derived_categorical]
        return data


def load_model_spec(path: str | Path) -> ModelSpec:
    """Load a model specification from a YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML model specification must contain a mapping.")
    return parse_model_spec(raw)


def parse_model_spec(raw: dict[str, Any]) -> ModelSpec:
    """Parse a permissive RPNB YAML model specification."""

    model = raw.get("model", raw)
    simulation = raw.get("simulation", {})
    estimation = raw.get("estimation", {})
    output = raw.get("output", {})

    dependent = _first_present(model, "dependent", "dependent_variable", "response", "y")
    if dependent is None:
        raise ValueError("Model specification requires a dependent count variable.")

    offset = _first_present(model, "offset", "offset_variable", "log_offset", "exposure_offset")

    fixed_raw = _first_present(
        model, "fixed", "fixed_variables", "fixed_covariates", default=()
    )
    fixed_continuous, fixed_categorical = _parse_fixed_terms(fixed_raw)
    random_raw = _first_present(
        model, "random", "random_variables", "random_parameters", default=()
    )
    distributions = model.get("random_parameter_distributions", {})
    random_continuous_raw, random_categorical_raw = _parse_random_terms(random_raw)
    random_specs = _parse_random_parameters(random_continuous_raw, distributions)
    random_categorical_specs = _parse_random_categorical_variables(
        random_categorical_raw,
        distributions,
    )
    derived_categorical_specs = _parse_derived_categorical_variables(
        model.get("derived_categorical", {})
    )

    fixed_names = tuple(str(item) for item in fixed_continuous)
    categorical_names = tuple(item.name for item in fixed_categorical)
    random_names = tuple(item.name for item in random_specs)
    random_categorical_names = tuple(item.name for item in random_categorical_specs)
    group_id = _first_present(model, "group_id", "panel_id", "panel", "group", default=None)
    _validate_variable_roles(
        dependent=str(dependent),
        offset=None if offset is None else str(offset),
        fixed_names=fixed_names,
        fixed_categorical_names=categorical_names,
        random_names=random_names,
        random_categorical_names=random_categorical_names,
        group_id=None if group_id is None else str(group_id),
    )

    intercept = bool(_first_present(model, "intercept", "add_intercept", default=True))
    correlated = bool(
        _first_present(
            model,
            "correlated_random_parameters",
            "correlated",
            "correlation",
            default=False,
        )
    )
    missing = str(_first_present(model, "missing", "missing_data", default="drop"))

    draws = int(_first_present(simulation, "draws", "n_draws", "num_draws", default=200))
    draw_type = str(_first_present(simulation, "draw_type", "draws_type", default="halton"))
    seed = _first_present(simulation, "seed", "random_seed", default=12345)
    seed = None if seed is None else int(seed)

    maxiter = int(_first_present(estimation, "maxiter", "max_iterations", default=1000))
    tolerance = float(_first_present(estimation, "tolerance", "tol", "gtol", default=1e-5))
    optimizer = str(
        _first_present(
            estimation,
            "optimizer",
            "optimization_method",
            "method",
            default="bfgs",
        )
    )
    multistart = int(_first_present(estimation, "multistart", "multi_start", default=1))
    multistart_random_seed = _first_present(
        estimation,
        "random_seed",
        "multistart_random_seed",
        "multistart_seed",
        default=12345,
    )
    multistart_random_seed = (
        None if multistart_random_seed is None else int(multistart_random_seed)
    )
    multistart_scale = float(
        _first_present(estimation, "multistart_scale", "multi_start_scale", default=0.25)
    )
    if multistart_scale < 0:
        raise ValueError("multistart_scale must be non-negative.")
    auto_simplify_random_parameters = bool(
        _first_present(
            estimation,
            "auto_simplify_random_parameters",
            "auto_simplify",
            default=False,
        )
    )
    covariance = str(_first_present(estimation, "covariance", "covariance_type", default="bfgs"))
    chunk_size_raw = _first_present(
        estimation, "chunk_size", "likelihood_chunk_size", default=None
    )
    if chunk_size_raw is None:
        chunk_size_raw = _first_present(
            simulation, "chunk_size", "likelihood_chunk_size", default=10_000
        )
    chunk_size = None if chunk_size_raw is None else int(chunk_size_raw)
    workers_raw = _first_present(estimation, "workers", "n_jobs", "processes", default=None)
    if workers_raw is None:
        workers_raw = _first_present(simulation, "workers", "n_jobs", "processes", default=1)
    checkpoint_interval = int(
        _first_present(
            estimation,
            "checkpoint_interval",
            "checkpoint_every",
            default=10,
        )
    )
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative.")
    start_alpha = float(_first_present(estimation, "start_alpha", "alpha_start", default=0.5))
    if start_alpha <= 0:
        raise ValueError("start_alpha must be positive.")

    output_dir = str(_first_present(output, "directory", "dir", "output_dir", default="runs"))

    return ModelSpec(
        dependent=str(dependent),
        offset=None if offset is None else str(offset),
        fixed=fixed_names,
        fixed_categorical=fixed_categorical,
        random=tuple(random_specs),
        random_categorical=tuple(random_categorical_specs),
        derived_categorical=tuple(derived_categorical_specs),
        group_id=None if group_id is None else str(group_id),
        intercept=intercept,
        draws=draws,
        draw_type=draw_type,
        correlated_random_parameters=correlated,
        seed=seed,
        maxiter=maxiter,
        tolerance=tolerance,
        optimizer=optimizer.lower(),
        multistart=multistart,
        multistart_random_seed=multistart_random_seed,
        multistart_scale=multistart_scale,
        auto_simplify_random_parameters=auto_simplify_random_parameters,
        covariance=covariance.lower(),
        chunk_size=chunk_size,
        workers=int(workers_raw),
        checkpoint_interval=checkpoint_interval,
        start_alpha=start_alpha,
        output_dir=output_dir,
        missing=missing.lower(),
    )


def _first_present(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _parse_fixed_terms(
    fixed_raw: Any,
) -> tuple[tuple[str, ...], tuple[CategoricalVariableSpec, ...]]:
    if fixed_raw is None:
        return (), ()
    if isinstance(fixed_raw, dict):
        continuous = _as_tuple(
            _first_present(
                fixed_raw,
                "continuous",
                "numeric",
                "variables",
                default=(),
            )
        )
        categorical_raw = _first_present(
            fixed_raw,
            "categorical",
            "factor",
            "factors",
            default={},
        )
        return (
            tuple(str(item) for item in continuous),
            tuple(_parse_categorical_variables(categorical_raw)),
        )
    return tuple(str(item) for item in _as_tuple(fixed_raw)), ()


def _parse_categorical_variables(categorical_raw: Any) -> list[CategoricalVariableSpec]:
    if categorical_raw is None:
        return []
    if not isinstance(categorical_raw, dict):
        raise ValueError("fixed.categorical must be a mapping of variable names to references.")

    specs: list[CategoricalVariableSpec] = []
    for name, config in categorical_raw.items():
        if isinstance(config, dict):
            if "reference" not in config:
                raise ValueError(f"Categorical variable {name!r} requires a reference value.")
            reference = config["reference"]
        else:
            reference = config
        specs.append(CategoricalVariableSpec(name=str(name), reference=reference))
    return specs


def _parse_random_terms(random_raw: Any) -> tuple[Any, Any]:
    if not isinstance(random_raw, dict):
        return random_raw, {}
    structured_keys = {"continuous", "numeric", "variables", "categorical", "factor", "factors"}
    if not (set(random_raw) & structured_keys):
        return random_raw, {}
    continuous_raw = _first_present(
        random_raw, "continuous", "numeric", "variables", default=()
    )
    categorical_raw = _first_present(
        random_raw, "categorical", "factor", "factors", default={}
    )
    return continuous_raw, categorical_raw


def _parse_random_parameters(
    random_raw: Any, distributions: dict[str, Any] | None
) -> list[RandomParameterSpec]:
    if random_raw is None:
        return []
    distributions = distributions or {}
    specs: list[RandomParameterSpec] = []

    if isinstance(random_raw, dict):
        items = list(random_raw.items())
    else:
        items = []
        for entry in _as_tuple(random_raw):
            if isinstance(entry, dict):
                if "name" in entry:
                    name = entry["name"]
                    config = {key: value for key, value in entry.items() if key != "name"}
                elif len(entry) == 1:
                    name, config = next(iter(entry.items()))
                else:
                    raise ValueError(
                        "Random parameter mappings must include a 'name' key or contain "
                        "a single variable-name key."
                    )
                items.append((name, config))
            else:
                items.append((entry, distributions.get(entry, {})))

    for name, config in items:
        if isinstance(config, str):
            config = {"distribution": config}
        elif config is None:
            config = {}
        elif not isinstance(config, dict):
            raise ValueError(f"Invalid random parameter specification for {name!r}.")

        dist_config = distributions.get(name, {})
        if isinstance(dist_config, str):
            dist_config = {"distribution": dist_config}
        merged = {**dist_config, **config}
        specs.append(
            RandomParameterSpec(
                name=str(name),
                distribution=str(merged.get("distribution", merged.get("dist", "normal"))),
                start_mean=float(merged.get("start_mean", merged.get("mean", 0.0))),
                start_sd=float(merged.get("start_sd", merged.get("sd", 0.25))),
            )
        )
    return specs


def _parse_random_categorical_variables(
    categorical_raw: Any, distributions: dict[str, Any] | None
) -> list[RandomCategoricalVariableSpec]:
    if categorical_raw is None:
        return []
    if not isinstance(categorical_raw, dict):
        raise ValueError("random.categorical must be a mapping of variable names to references.")
    distributions = distributions or {}
    specs: list[RandomCategoricalVariableSpec] = []
    for name, config in categorical_raw.items():
        if not isinstance(config, dict):
            config = {"reference": config}
        if "reference" not in config:
            raise ValueError(f"Random categorical variable {name!r} requires a reference value.")
        dist_config = distributions.get(name, {})
        if isinstance(dist_config, str):
            dist_config = {"distribution": dist_config}
        merged = {**dist_config, **config}
        specs.append(
            RandomCategoricalVariableSpec(
                name=str(name),
                reference=merged["reference"],
                distribution=str(merged.get("distribution", merged.get("dist", "normal"))),
                start_mean=float(merged.get("start_mean", merged.get("mean", 0.0))),
                start_sd=float(merged.get("start_sd", merged.get("sd", 0.25))),
            )
        )
    return specs


def _parse_derived_categorical_variables(derived_raw: Any) -> list[DerivedCategoricalSpec]:
    if derived_raw is None:
        return []
    if not isinstance(derived_raw, dict):
        raise ValueError("derived_categorical must be a mapping.")
    specs: list[DerivedCategoricalSpec] = []
    for name, config in derived_raw.items():
        if not isinstance(config, dict):
            raise ValueError(f"Derived categorical variable {name!r} must be a mapping.")
        source = config.get("source")
        if source is None:
            raise ValueError(f"Derived categorical variable {name!r} requires a source.")
        method = str(config.get("method", "bins")).lower()
        if method == "quantile":
            bins = int(config.get("bins", 0))
            if bins < 2:
                raise ValueError(f"Derived quantile variable {name!r} requires bins >= 2.")
            specs.append(
                DerivedCategoricalSpec(
                    name=str(name),
                    source=str(source),
                    method="quantile",
                    quantile_bins=bins,
                )
            )
            continue
        if method not in {"bins", "cut"}:
            raise ValueError(f"Unsupported derived categorical method for {name!r}: {method!r}.")
        bins_raw = config.get("bins")
        if not isinstance(bins_raw, list) or not bins_raw:
            raise ValueError(f"Derived categorical variable {name!r} requires non-empty bins.")
        bins = _parse_derived_bins(name, bins_raw)
        specs.append(
            DerivedCategoricalSpec(
                name=str(name),
                source=str(source),
                method="bins",
                bins=tuple(bins),
            )
        )
    return specs


def _parse_derived_bins(
    variable: str, bins_raw: list[Any]
) -> list[DerivedCategoricalBinSpec]:
    bins: list[DerivedCategoricalBinSpec] = []
    previous_upper: float | None = None
    open_ended_seen = False
    for index, item in enumerate(bins_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Derived bin {index + 1} for {variable!r} must be a mapping.")
        if "label" not in item:
            raise ValueError(f"Derived bin {index + 1} for {variable!r} requires a label.")
        if open_ended_seen:
            raise ValueError(f"Open-ended derived bin for {variable!r} must be last.")
        upper = item.get("upper")
        if upper is None:
            open_ended_seen = True
            bins.append(DerivedCategoricalBinSpec(label=str(item["label"]), upper=None))
            continue
        upper_value = float(upper)
        if previous_upper is not None and upper_value <= previous_upper:
            raise ValueError(f"Derived bins for {variable!r} must have increasing upper values.")
        previous_upper = upper_value
        bins.append(DerivedCategoricalBinSpec(label=str(item["label"]), upper=upper_value))
    return bins


def _validate_variable_roles(
    dependent: str,
    offset: str | None,
    fixed_names: tuple[str, ...],
    fixed_categorical_names: tuple[str, ...],
    random_names: tuple[str, ...],
    random_categorical_names: tuple[str, ...],
    group_id: str | None,
) -> None:
    if offset is not None:
        for variable in (
            *fixed_names,
            *fixed_categorical_names,
            *random_names,
            *random_categorical_names,
        ):
            if variable != offset:
                continue
            raise ValueError(
                f"Offset variable {offset} has coefficient fixed at 1 and cannot be listed "
                "as fixed or random."
            )
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
            *(() if offset is None else (offset,)),
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
