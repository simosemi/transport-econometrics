"""Output tables, fit statistics, and export helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RPNBResults:
    """Container for fitted RPNB model output."""

    parameter_table: pd.DataFrame
    fit_statistics: dict[str, Any]
    convergence: dict[str, Any]
    preprocessing_summary: pd.DataFrame | None = None
    multistart_summary: pd.DataFrame | None = None
    local_solutions: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
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

    @property
    def alpha(self) -> float:
        alpha_rows = self.parameter_table.loc[self.parameter_table["parameter"] == "alpha"]
        if alpha_rows.empty:
            raise KeyError("alpha was not found in the parameter table.")
        return float(alpha_rows.iloc[0]["estimate"])

    def summary(self) -> str:
        status = "converged" if self.converged else "not converged"
        quality = self.convergence.get("convergence_quality", "not_reported")
        return (
            f"RPNB fit: {status}; "
            f"quality={quality}; "
            f"LL={self.log_likelihood:.6f}; "
            f"AIC={self.aic:.6f}; "
            f"BIC={self.bic:.6f}; "
            f"alpha={self.alpha:.6f}"
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
            "excel": directory / "rpnb_results.xlsx",
            "html": directory / "rpnb_results.html",
            "nlogit_text_report": directory / "nlogit_style_report.txt",
        }
        if self.preprocessing_summary is not None:
            paths["preprocessing_summary_csv"] = directory / "preprocessing_summary.csv"
            paths["preprocessing_summary_xlsx"] = directory / "preprocessing_summary.xlsx"
            paths["preprocessing_summary_html"] = directory / "preprocessing_summary.html"
        if self.multistart_summary is not None:
            paths["multistart_summary_csv"] = directory / "multistart_summary.csv"
            paths["multistart_summary_xlsx"] = directory / "multistart_summary.xlsx"
            paths["multistart_summary_html"] = directory / "multistart_summary.html"
        if self.local_solutions is not None:
            paths["local_solutions_csv"] = directory / "multistart_local_solutions.csv"

        self.parameter_table.to_csv(paths["coefficients_csv"], index=False)
        pd.DataFrame([self.fit_statistics]).to_csv(paths["fit_statistics_csv"], index=False)
        pd.DataFrame([self.convergence]).to_csv(paths["convergence_csv"], index=False)
        pd.DataFrame([self.timing]).to_csv(paths["timing_csv"], index=False)
        if self.preprocessing_summary is not None:
            self.preprocessing_summary.to_csv(
                paths["preprocessing_summary_csv"], index=False
            )
            with pd.ExcelWriter(
                paths["preprocessing_summary_xlsx"], engine="openpyxl"
            ) as writer:
                self.preprocessing_summary.to_excel(
                    writer, sheet_name="preprocessing_summary", index=False
                )
            paths["preprocessing_summary_html"].write_text(
                self._preprocessing_to_html(), encoding="utf-8"
            )
        if self.multistart_summary is not None:
            self.multistart_summary.to_csv(paths["multistart_summary_csv"], index=False)
            with pd.ExcelWriter(paths["multistart_summary_xlsx"], engine="openpyxl") as writer:
                self.multistart_summary.to_excel(
                    writer, sheet_name="multistart_summary", index=False
                )
            paths["multistart_summary_html"].write_text(
                self._multistart_to_html(), encoding="utf-8"
            )
        if self.local_solutions is not None:
            self.local_solutions.to_csv(paths["local_solutions_csv"], index=False)

        if self.predictions is not None:
            paths["predictions_csv"] = directory / "predictions.csv"
            self.predictions.to_csv(paths["predictions_csv"], index=False)
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
            if self.preprocessing_summary is not None:
                self.preprocessing_summary.to_excel(
                    writer, sheet_name="preprocessing_summary", index=False
                )
            if self.multistart_summary is not None:
                self.multistart_summary.to_excel(
                    writer, sheet_name="multistart_summary", index=False
                )
            if self.local_solutions is not None:
                self.local_solutions.to_excel(
                    writer, sheet_name="local_solutions", index=False
                )
            if self.predictions is not None:
                self.predictions.to_excel(writer, sheet_name="predictions", index=False)
            if self.marginal_effects is not None:
                self.marginal_effects.to_excel(
                    writer, sheet_name="marginal_effects", index=False
                )

        paths["html"].write_text(self._to_html(), encoding="utf-8")
        paths["nlogit_text_report"].write_text(
            self.nlogit_style_report(), encoding="utf-8"
        )
        return paths

    def _to_html(self) -> str:
        sections = [
            "<h1>RPNB results</h1>",
            "<h2>Fit statistics</h2>",
            pd.DataFrame([self.fit_statistics]).to_html(index=False),
            "<h2>Convergence</h2>",
            pd.DataFrame([self.convergence]).to_html(index=False),
            "<h2>Timing</h2>",
            pd.DataFrame([self.timing]).to_html(index=False),
            "<h2>Coefficient table</h2>",
            self.parameter_table.to_html(index=False),
        ]
        if self.preprocessing_summary is not None:
            sections.extend(
                [
                    "<h2>Preprocessing summary</h2>",
                    self.preprocessing_summary.to_html(index=False),
                ]
            )
        if self.multistart_summary is not None:
            sections.extend(
                [
                    "<h2>Multi-start summary</h2>",
                    self.multistart_summary.to_html(index=False),
                ]
            )
        if self.local_solutions is not None:
            sections.extend(
                [
                    "<h2>Local solutions</h2>",
                    self.local_solutions.to_html(index=False),
                ]
            )
        if self.marginal_effects is not None:
            sections.extend(
                ["<h2>Average marginal effects</h2>", self.marginal_effects.to_html(index=False)]
            )
        if self.predictions is not None:
            sections.extend(
                [
                    "<h2>Predictions</h2>",
                    self.predictions.head(100).to_html(index=False),
                ]
            )
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>RPNB results</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                *sections,
                "</body></html>",
            ]
        )

    def _preprocessing_to_html(self) -> str:
        table = (
            self.preprocessing_summary.to_html(index=False)
            if self.preprocessing_summary is not None
            else "<p>No preprocessing summary was generated.</p>"
        )
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>RPNB preprocessing summary</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                "<h1>RPNB preprocessing summary</h1>",
                table,
                "</body></html>",
            ]
        )

    def _multistart_to_html(self) -> str:
        table = (
            self.multistart_summary.to_html(index=False)
            if self.multistart_summary is not None
            else "<p>No multi-start summary was generated.</p>"
        )
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>RPNB multi-start summary</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                "<h1>RPNB multi-start summary</h1>",
                table,
                "</body></html>",
            ]
        )

    def nlogit_style_report(self) -> str:
        """Return a plain-text report arranged like an NLOGIT model output."""

        stats = self.fit_statistics
        restricted = stats.get("restricted_log_likelihood")
        restricted_text = (
            _format_float(restricted)
            if restricted is not None and pd.notna(restricted)
            else "Not available"
        )
        lines = [
            "RPNB NLOGIT-STYLE REPORT",
            "=" * 26,
            f"Model name: Random Parameters Negative Binomial",
            f"Dependent variable: {stats.get('dependent', '')}",
            f"N: {stats.get('n_observations', '')}",
            f"Groups: {stats.get('n_groups', '')}",
            f"Log likelihood: {_format_float(stats.get('log_likelihood'))}",
            f"Restricted log likelihood: {restricted_text}",
            f"AIC: {_format_float(stats.get('AIC'))}",
            f"BIC: {_format_float(stats.get('BIC'))}",
            f"Convergence quality: {self.convergence.get('convergence_quality', '')}",
            "",
            "FIXED PARAMETERS",
            "-" * 16,
            *_parameter_lines(self.parameter_table, "fixed_mean"),
            "",
            "RANDOM PARAMETER MEANS",
            "-" * 24,
            *_parameter_lines(self.parameter_table, "random_mean"),
            "",
            "RANDOM PARAMETER SCALE/SD",
            "-" * 25,
            *_parameter_lines(self.parameter_table, "random_sd"),
            "",
            "DISPERSION PARAMETER",
            "-" * 20,
            *_parameter_lines(self.parameter_table, "dispersion"),
            "",
            "Significance stars: *** p<0.01, ** p<0.05, * p<0.10",
        ]
        return "\n".join(lines)


def build_parameter_table(
    names: list[str],
    components: list[str],
    variables: list[str],
    estimates: np.ndarray,
    covariance: np.ndarray | None,
) -> pd.DataFrame:
    """Build coefficient table with standard errors, z values, p values, and IRRs."""

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
    stars = [_significance_stars(value) for value in p_values]
    irr = np.where(
        np.isin(components, ["fixed_mean", "random_mean"]),
        np.exp(estimates),
        np.nan,
    )

    return pd.DataFrame(
        {
            "parameter": names,
            "component": components,
            "variable": variables,
            "interpretation": [
                _parameter_interpretation(component, variable)
                for component, variable in zip(components, variables)
            ],
            "estimate": estimates,
            "std_error": std_errors,
            "z_value": z_values,
            "p_value": p_values,
            "significance": stars,
            "incidence_rate_ratio": irr,
        }
    )


def _parameter_interpretation(component: str, variable: str) -> str:
    if component == "random_mean":
        return (
            f"average/mean effect for {variable}; for generated categorical dummies, "
            "this is relative to the declared reference category"
        )
    if component == "random_sd":
        return (
            f"heterogeneity/standard deviation for {variable}; for generated categorical "
            "dummies, this is heterogeneity in the relative effect"
        )
    if component == "fixed_mean":
        return f"non-random fixed effect for {variable}"
    if component == "dispersion":
        return "negative binomial dispersion parameter"
    if component == "random_correlation":
        return f"correlation between random parameters {variable}"
    return component


def _parameter_lines(table: pd.DataFrame, component: str) -> list[str]:
    rows = table.loc[table["component"] == component]
    if rows.empty:
        return ["  None"]
    lines = []
    for row in rows.itertuples(index=False):
        lines.append(
            "  "
            f"{row.parameter:<36} "
            f"{_format_float(row.estimate):>12} "
            f"SE={_format_float(row.std_error):>12} "
            f"z={_format_float(row.z_value):>10} "
            f"p={_format_float(row.p_value):>10} "
            f"{row.significance}"
        )
    return lines


def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.6f}"
