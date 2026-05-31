"""Output tables, fit statistics, and export helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RPOpitResults:
    """Container for fitted model output."""

    parameter_table: pd.DataFrame
    fit_statistics: dict[str, Any]
    convergence: dict[str, Any]
    predicted_probabilities: pd.DataFrame | None = None
    marginal_effects: pd.DataFrame | None = None
    run_dir: Path | None = None
    model_spec: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    @property
    def log_likelihood(self) -> float:
        return float(self.fit_statistics["log_likelihood"])

    @property
    def aic(self) -> float:
        return float(self.fit_statistics["AIC"])

    @property
    def bic(self) -> float:
        return float(self.fit_statistics["BIC"])

    @property
    def converged(self) -> bool:
        return bool(self.convergence["converged"])

    def summary(self) -> str:
        status = "converged" if self.converged else "not converged"
        return (
            f"RPOpit fit: {status}; "
            f"LL={self.log_likelihood:.6f}; "
            f"AIC={self.aic:.6f}; "
            f"BIC={self.bic:.6f}"
        )

    def export(self, output_dir: str | Path | None = None) -> dict[str, Path]:
        """Export results to CSV, Excel, and HTML."""

        directory = Path(output_dir) if output_dir is not None else self.run_dir
        if directory is None:
            raise ValueError("output_dir is required when this result has no run_dir.")
        directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "coefficients_csv": directory / "coefficients.csv",
            "fit_statistics_csv": directory / "fit_statistics.csv",
            "convergence_csv": directory / "convergence.csv",
            "timing_csv": directory / "timing.csv",
            "excel": directory / "rpopit_results.xlsx",
            "html": directory / "rpopit_results.html",
        }
        self.parameter_table.to_csv(paths["coefficients_csv"], index=False)
        pd.DataFrame([self.fit_statistics]).to_csv(paths["fit_statistics_csv"], index=False)
        pd.DataFrame([self.convergence]).to_csv(paths["convergence_csv"], index=False)
        pd.DataFrame([self.timing]).to_csv(paths["timing_csv"], index=False)

        if self.predicted_probabilities is not None:
            paths["predicted_probabilities_csv"] = directory / "predicted_probabilities.csv"
            self.predicted_probabilities.to_csv(
                paths["predicted_probabilities_csv"], index=False
            )
        if self.marginal_effects is not None:
            paths["marginal_effects_csv"] = directory / "marginal_effects.csv"
            self.marginal_effects.to_csv(paths["marginal_effects_csv"], index=False)

        with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
            self.parameter_table.to_excel(writer, sheet_name="coefficients", index=False)
            pd.DataFrame([self.fit_statistics]).to_excel(
                writer, sheet_name="fit_statistics", index=False
            )
            pd.DataFrame([self.convergence]).to_excel(
                writer, sheet_name="convergence", index=False
            )
            pd.DataFrame([self.timing]).to_excel(writer, sheet_name="timing", index=False)
            if self.predicted_probabilities is not None:
                self.predicted_probabilities.to_excel(
                    writer, sheet_name="predicted_probabilities", index=False
                )
            if self.marginal_effects is not None:
                self.marginal_effects.to_excel(
                    writer, sheet_name="marginal_effects", index=False
                )

        paths["html"].write_text(self._to_html(), encoding="utf-8")
        return paths

    def _to_html(self) -> str:
        sections = [
            "<h1>rpopit results</h1>",
            "<h2>Fit statistics</h2>",
            pd.DataFrame([self.fit_statistics]).to_html(index=False),
            "<h2>Convergence</h2>",
            pd.DataFrame([self.convergence]).to_html(index=False),
            "<h2>Timing</h2>",
            pd.DataFrame([self.timing]).to_html(index=False),
            "<h2>Coefficient table</h2>",
            self.parameter_table.to_html(index=False),
        ]
        if self.marginal_effects is not None:
            sections.extend(
                ["<h2>Average marginal effects</h2>", self.marginal_effects.to_html(index=False)]
            )
        if self.predicted_probabilities is not None:
            sections.extend(
                [
                    "<h2>Predicted probabilities</h2>",
                    self.predicted_probabilities.head(100).to_html(index=False),
                ]
            )
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>rpopit results</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                *sections,
                "</body></html>",
            ]
        )


def build_parameter_table(
    names: list[str],
    components: list[str],
    variables: list[str],
    estimates: np.ndarray,
    covariance: np.ndarray | None,
) -> pd.DataFrame:
    """Build coefficient table with standard errors, z values, and p values."""

    estimates = np.asarray(estimates, dtype=float)
    if covariance is None:
        std_errors = np.full(estimates.size, np.nan)
    else:
        diagonal = np.diag(covariance)
        std_errors = np.sqrt(np.where(diagonal >= 0.0, diagonal, np.nan))

    z_values = estimates / std_errors
    z_values[~np.isfinite(z_values)] = np.nan
    p_values = 2.0 * norm.sf(np.abs(z_values))
    p_values[~np.isfinite(p_values)] = np.nan

    return pd.DataFrame(
        {
            "parameter": names,
            "component": components,
            "variable": variables,
            "estimate": estimates,
            "std_error": std_errors,
            "z_value": z_values,
            "p_value": p_values,
        }
    )
