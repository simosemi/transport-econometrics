"""Compare fixed-only Ordered Probit with Random Parameters Ordered Probit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rpopit.config import ModelSpec
from rpopit.model import RandomParametersOrderedProbit
from rpopit.output import RPOpitResults


@dataclass
class ModelComparisonReport:
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    random_parameter_sds: pd.DataFrame
    ordered_probit: RPOpitResults
    random_parameters_ordered_probit: RPOpitResults
    null_model: RPOpitResults

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": directory / "model_comparison_metrics.csv",
            "coefficients": directory / "model_comparison_coefficients.csv",
            "random_sds": directory / "random_parameter_sds.csv",
            "excel": directory / "model_comparison_report.xlsx",
            "html": directory / "model_comparison_report.html",
        }
        self.metrics.to_csv(paths["metrics"], index=False)
        self.coefficients.to_csv(paths["coefficients"], index=False)
        self.random_parameter_sds.to_csv(paths["random_sds"], index=False)
        with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
            self.metrics.to_excel(writer, sheet_name="metrics", index=False)
            self.coefficients.to_excel(writer, sheet_name="coefficients", index=False)
            self.random_parameter_sds.to_excel(writer, sheet_name="random_sds", index=False)
        paths["html"].write_text(self._to_html(), encoding="utf-8")
        return paths

    def _to_html(self) -> str:
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>rpopit model comparison</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                "<h1>rpopit model comparison</h1>",
                "<h2>Fit metrics</h2>",
                self.metrics.to_html(index=False),
                "<h2>Coefficient tables</h2>",
                self.coefficients.to_html(index=False),
                "<h2>Random parameter standard deviations</h2>",
                self.random_parameter_sds.to_html(index=False),
                "</body></html>",
            ]
        )


def compare_ordered_models(data: pd.DataFrame, spec: ModelSpec) -> ModelComparisonReport:
    """Fit and compare Ordered Probit and Random Parameters Ordered Probit."""

    baseline_fixed = list(spec.fixed) + [item.name for item in spec.random]
    ordered_model = RandomParametersOrderedProbit(
        dependent=spec.dependent,
        fixed=baseline_fixed,
        random=[],
        group_id=None,
        categories=spec.categories,
        draws=1,
        draw_type=spec.draw_type,
        seed=spec.seed,
        maxiter=spec.maxiter,
        tolerance=spec.tolerance,
        covariance=spec.covariance,
        chunk_size=spec.chunk_size,
        workers=1,
        output_dir=spec.output_dir,
        missing=spec.missing,
    )
    rpopit_model = RandomParametersOrderedProbit.from_spec(spec)
    null_model = RandomParametersOrderedProbit(
        dependent=spec.dependent,
        fixed=[],
        random=[],
        group_id=None,
        categories=spec.categories,
        draws=1,
        draw_type=spec.draw_type,
        seed=spec.seed,
        maxiter=spec.maxiter,
        tolerance=spec.tolerance,
        covariance=spec.covariance,
        chunk_size=spec.chunk_size,
        workers=1,
        output_dir=spec.output_dir,
        missing=spec.missing,
    )

    ordered_results = ordered_model.fit(data, save_run=False, export=False)
    rpopit_results = rpopit_model.fit(data, save_run=False, export=False)
    null_results = null_model.fit(data, save_run=False, export=False)

    metrics = pd.DataFrame(
        [
            _metric_row("Ordered Probit", ordered_results, null_results.log_likelihood),
            _metric_row(
                "Random Parameters Ordered Probit",
                rpopit_results,
                null_results.log_likelihood,
            ),
        ]
    )
    coefficients = pd.concat(
        [
            _with_model_name("Ordered Probit", ordered_results.parameter_table),
            _with_model_name(
                "Random Parameters Ordered Probit", rpopit_results.parameter_table
            ),
        ],
        ignore_index=True,
    )
    random_sds = rpopit_results.parameter_table[
        rpopit_results.parameter_table["component"].isin(
            ["random_sd", "random_correlation"]
        )
    ].copy()
    random_sds.insert(0, "model", "Random Parameters Ordered Probit")
    return ModelComparisonReport(
        metrics=metrics,
        coefficients=coefficients,
        random_parameter_sds=random_sds,
        ordered_probit=ordered_results,
        random_parameters_ordered_probit=rpopit_results,
        null_model=null_results,
    )


def _metric_row(model_name: str, results: RPOpitResults, null_ll: float) -> dict[str, object]:
    return {
        "model": model_name,
        "LL": results.log_likelihood,
        "AIC": results.aic,
        "BIC": results.bic,
        "McFadden_pseudo_R2": 1.0 - (results.log_likelihood / null_ll),
        "n_observations": results.fit_statistics["n_observations"],
        "n_parameters": results.fit_statistics["n_parameters"],
        "converged": results.converged,
        "iterations": results.convergence["iterations"],
        "objective_calls": results.timing.get("objective_calls"),
        "average_objective_seconds": results.timing.get("average_objective_seconds"),
    }


def _with_model_name(model_name: str, table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out.insert(0, "model", model_name)
    return out
