"""YAML model specification parsing."""

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
    start_sd: float = 0.3

    def __post_init__(self) -> None:
        if self.distribution.lower() != "normal":
            raise ValueError(
                f"Unsupported random parameter distribution for {self.name!r}: "
                f"{self.distribution!r}. rpopit currently supports normal only."
            )
        if self.start_sd <= 0:
            raise ValueError(f"start_sd for {self.name!r} must be positive.")


@dataclass(frozen=True)
class ModelSpec:
    """Complete model, simulation, estimation, and output specification."""

    dependent: str
    fixed: tuple[str, ...] = ()
    random: tuple[RandomParameterSpec, ...] = ()
    group_id: str | None = None
    categories: tuple[Any, ...] | None = None
    draws: int = 200
    draw_type: str = "halton"
    correlated_random_parameters: bool = False
    seed: int | None = 12345
    maxiter: int = 1000
    tolerance: float = 1e-4
    covariance: str = "bfgs"
    chunk_size: int | None = 10_000
    workers: int = 1
    output_dir: str = "runs"
    missing: str = "drop"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["random"] = [asdict(item) for item in self.random]
        return data


def load_model_spec(path: str | Path) -> ModelSpec:
    """Load a model specification from a YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML model specification must contain a mapping.")
    return parse_model_spec(raw)


def parse_model_spec(raw: dict[str, Any]) -> ModelSpec:
    """Parse a permissive NLOGIT-style YAML model specification."""

    model = raw.get("model", raw)
    simulation = raw.get("simulation", {})
    estimation = raw.get("estimation", {})
    output = raw.get("output", {})

    dependent = _first_present(model, "dependent", "dependent_variable", "response", "y")
    if dependent is None:
        raise ValueError("Model specification requires a dependent variable.")

    fixed = _as_tuple(
        _first_present(model, "fixed", "fixed_variables", "fixed_covariates", default=())
    )

    random_raw = _first_present(
        model, "random", "random_variables", "random_parameters", default=()
    )
    distributions = model.get("random_parameter_distributions", {})
    random_specs = _parse_random_parameters(random_raw, distributions)

    group_id = _first_present(model, "group_id", "panel_id", "panel", "group", default=None)
    categories = _first_present(model, "categories", "ordered_categories", default=None)

    draws = int(_first_present(simulation, "draws", "n_draws", "num_draws", default=200))
    draw_type = str(_first_present(simulation, "draw_type", "draws_type", default="halton"))
    seed = _first_present(simulation, "seed", "random_seed", default=12345)
    seed = None if seed is None else int(seed)

    correlated = bool(
        _first_present(
            model,
            "correlated_random_parameters",
            "correlated",
            "correlation",
            default=False,
        )
    )

    maxiter = int(_first_present(estimation, "maxiter", "max_iterations", default=1000))
    tolerance = float(_first_present(estimation, "tolerance", "tol", "gtol", default=1e-4))
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
    workers = int(workers_raw)
    missing = str(_first_present(model, "missing", "missing_data", default="drop"))
    output_dir = str(_first_present(output, "directory", "dir", "output_dir", default="runs"))

    return ModelSpec(
        dependent=str(dependent),
        fixed=tuple(str(item) for item in fixed),
        random=tuple(random_specs),
        group_id=None if group_id is None else str(group_id),
        categories=None if categories is None else tuple(categories),
        draws=draws,
        draw_type=draw_type,
        correlated_random_parameters=correlated,
        seed=seed,
        maxiter=maxiter,
        tolerance=tolerance,
        covariance=covariance.lower(),
        chunk_size=chunk_size,
        workers=workers,
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
                start_sd=float(merged.get("start_sd", merged.get("sd", 0.3))),
            )
        )
    return specs
