"""Benchmark/report helpers for comparing RPNB with NLOGIT exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpnb.output import RPNBResults

CANONICAL_COMPONENTS = {
    "coefficient",
    "random_mean",
    "random_sd",
    "alpha",
    "offset_coefficient",
    "predicted_mean",
    "log_likelihood",
}


@dataclass
class NlogitComparisonReport:
    """Side-by-side RPNB versus NLOGIT benchmark output."""

    combined: pd.DataFrame
    coefficients: pd.DataFrame
    random_means: pd.DataFrame
    random_sds: pd.DataFrame
    alpha: pd.DataFrame
    offset_handling: pd.DataFrame
    predicted_means: pd.DataFrame
    log_likelihood: pd.DataFrame
    metadata: dict[str, Any]

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "combined": directory / "nlogit_rpnb_parameter_comparison.csv",
            "coefficients": directory / "coefficients_comparison.csv",
            "random_means": directory / "random_parameter_means_comparison.csv",
            "random_sds": directory / "random_parameter_sds_comparison.csv",
            "alpha": directory / "alpha_comparison.csv",
            "offset_handling": directory / "offset_handling_comparison.csv",
            "predicted_means": directory / "predicted_means_comparison.csv",
            "log_likelihood": directory / "log_likelihood_comparison.csv",
            "excel": directory / "nlogit_rpnb_comparison.xlsx",
            "html": directory / "nlogit_rpnb_comparison.html",
        }
        self.combined.to_csv(paths["combined"], index=False)
        self.coefficients.to_csv(paths["coefficients"], index=False)
        self.random_means.to_csv(paths["random_means"], index=False)
        self.random_sds.to_csv(paths["random_sds"], index=False)
        self.alpha.to_csv(paths["alpha"], index=False)
        self.offset_handling.to_csv(paths["offset_handling"], index=False)
        self.predicted_means.to_csv(paths["predicted_means"], index=False)
        self.log_likelihood.to_csv(paths["log_likelihood"], index=False)
        with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
            pd.DataFrame([self.metadata]).to_excel(writer, sheet_name="metadata", index=False)
            self.combined.to_excel(writer, sheet_name="all", index=False)
            self.coefficients.to_excel(writer, sheet_name="coefficients", index=False)
            self.random_means.to_excel(writer, sheet_name="random_means", index=False)
            self.random_sds.to_excel(writer, sheet_name="random_sds", index=False)
            self.alpha.to_excel(writer, sheet_name="alpha", index=False)
            self.offset_handling.to_excel(writer, sheet_name="offset", index=False)
            self.predicted_means.to_excel(writer, sheet_name="predicted_means", index=False)
            self.log_likelihood.to_excel(writer, sheet_name="log_likelihood", index=False)
        paths["html"].write_text(self._to_html(), encoding="utf-8")
        return paths

    def _to_html(self) -> str:
        highlighted = self.combined.sort_values("abs_difference", ascending=False).head(50)
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>NLOGIT vs RPNB</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                "<h1>NLOGIT vs RPNB comparison</h1>",
                "<h2>Metadata</h2>",
                pd.DataFrame([self.metadata]).to_html(index=False),
                "<h2>Largest absolute differences</h2>",
                highlighted.to_html(index=False),
                "<h2>Coefficients</h2>",
                self.coefficients.to_html(index=False),
                "<h2>Random parameter means</h2>",
                self.random_means.to_html(index=False),
                "<h2>Random parameter SDs</h2>",
                self.random_sds.to_html(index=False),
                "<h2>Alpha</h2>",
                self.alpha.to_html(index=False),
                "<h2>Offset handling</h2>",
                self.offset_handling.to_html(index=False),
                "<h2>Predicted means</h2>",
                self.predicted_means.head(100).to_html(index=False),
                "<h2>Log-likelihood</h2>",
                self.log_likelihood.to_html(index=False),
                "</body></html>",
            ]
        )


def rpnb_canonical_table(results: RPNBResults) -> pd.DataFrame:
    """Normalize RPNB estimates to the NLOGIT comparison interchange format."""

    rows: list[dict[str, Any]] = []
    offset_variable = str(results.fit_statistics.get("offset", "offset"))
    for row in results.parameter_table.to_dict(orient="records"):
        component = row["component"]
        variable = row["variable"]
        if component == "fixed_mean":
            out_component = "coefficient"
        elif component in {"random_mean", "random_sd"}:
            out_component = component
        elif component == "dispersion" and row["parameter"] == "alpha":
            out_component = "alpha"
        else:
            continue
        rows.append(
            {
                "component": out_component,
                "variable": variable,
                "estimate": float(row["estimate"]),
                "std_error": row.get("std_error", np.nan),
                "source_parameter": row["parameter"],
            }
        )
    rows.append(
        {
            "component": "offset_coefficient",
            "variable": offset_variable,
            "estimate": 1.0,
            "std_error": np.nan,
            "source_parameter": "fixed_offset_coefficient",
        }
    )
    rows.append(
        {
            "component": "log_likelihood",
            "variable": "LL",
            "estimate": float(results.log_likelihood),
            "std_error": np.nan,
            "source_parameter": "log_likelihood",
        }
    )
    if results.predictions is not None:
        for prediction in results.predictions.to_dict(orient="records"):
            rows.append(
                {
                    "component": "predicted_mean",
                    "variable": str(prediction["row_index"]),
                    "estimate": float(prediction["predicted_count"]),
                    "std_error": np.nan,
                    "source_parameter": "predicted_count",
                }
            )
    return pd.DataFrame(rows)


def load_nlogit_canonical_table(path: str | Path) -> pd.DataFrame:
    """Load NLOGIT estimates exported as component/variable/estimate rows."""

    table = pd.read_csv(path)
    required = {"component", "variable", "estimate"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            "NLOGIT CSV must contain columns component, variable, estimate. "
            f"Missing: {sorted(missing)}"
        )
    out = table.copy()
    out["component"] = out["component"].astype(str).map(_normalize_component)
    out["variable"] = out["variable"].astype(str).map(_normalize_variable)
    out["estimate"] = pd.to_numeric(out["estimate"], errors="raise")
    if "std_error" not in out.columns:
        out["std_error"] = np.nan
    unknown = sorted(set(out["component"]) - CANONICAL_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown NLOGIT component values: {unknown}")
    return out[["component", "variable", "estimate", "std_error"]]


def load_nlogit_predicted_means(path: str | Path) -> pd.DataFrame:
    """Load observation-level NLOGIT predicted means."""

    table = pd.read_csv(path)
    if {"component", "variable", "estimate"}.issubset(table.columns):
        out = load_nlogit_canonical_table(path)
        return out.loc[out["component"] == "predicted_mean"].reset_index(drop=True)

    row_column = _first_present_column(table, "row_index", "index", "id", "observation")
    mean_column = _first_present_column(
        table, "predicted_mean", "predicted_count", "mu", "estimate", "prediction"
    )
    if row_column is None or mean_column is None:
        raise ValueError(
            "Predicted means CSV must contain row_index plus predicted_mean "
            "(or predicted_count, mu, estimate, prediction)."
        )
    return pd.DataFrame(
        {
            "component": "predicted_mean",
            "variable": table[row_column].astype(str),
            "estimate": pd.to_numeric(table[mean_column], errors="raise"),
            "std_error": np.nan,
        }
    )


def compare_with_nlogit(
    rpnb_results: RPNBResults,
    nlogit_results_path: str | Path,
    nlogit_predictions_path: str | Path | None = None,
    tolerance: float = 1e-4,
    prediction_tolerance: float = 1e-5,
    metadata: dict[str, Any] | None = None,
) -> NlogitComparisonReport:
    """Create side-by-side comparison tables for RPNB and NLOGIT."""

    rpnb_table = rpnb_canonical_table(rpnb_results)
    nlogit_table = load_nlogit_canonical_table(nlogit_results_path)
    if nlogit_predictions_path is not None:
        nlogit_predictions = load_nlogit_predicted_means(nlogit_predictions_path)
        nlogit_table = pd.concat([nlogit_table, nlogit_predictions], ignore_index=True)

    combined = nlogit_table.merge(
        rpnb_table,
        on=["component", "variable"],
        how="outer",
        suffixes=("_nlogit", "_rpnb"),
        indicator=True,
    )
    combined["difference"] = combined["estimate_rpnb"] - combined["estimate_nlogit"]
    combined["abs_difference"] = combined["difference"].abs()
    combined["relative_difference"] = combined["difference"] / combined[
        "estimate_nlogit"
    ].replace(0.0, np.nan)
    combined["tolerance"] = np.where(
        combined["component"] == "predicted_mean", prediction_tolerance, tolerance
    )
    combined["within_tolerance"] = combined["abs_difference"] <= combined["tolerance"]
    combined = combined.sort_values(["component", "variable"]).reset_index(drop=True)

    meta = dict(metadata or {})
    compared = combined.loc[combined["_merge"] == "both"]
    predicted = _component(compared, "predicted_mean")
    meta.update(
        {
            "tolerance": tolerance,
            "prediction_tolerance": prediction_tolerance,
            "max_abs_difference": float(compared["abs_difference"].max(skipna=True)),
            "max_predicted_mean_abs_difference": float(
                predicted["abs_difference"].max(skipna=True)
            )
            if not predicted.empty
            else np.nan,
            "mean_predicted_mean_abs_difference": float(
                predicted["abs_difference"].mean(skipna=True)
            )
            if not predicted.empty
            else np.nan,
            "n_compared_rows": int((combined["_merge"] == "both").sum()),
            "n_predicted_mean_rows": int(len(predicted)),
            "n_missing_from_nlogit": int((combined["_merge"] == "right_only").sum()),
            "n_missing_from_rpnb": int((combined["_merge"] == "left_only").sum()),
        }
    )

    return NlogitComparisonReport(
        combined=combined,
        coefficients=_component(combined, "coefficient"),
        random_means=_component(combined, "random_mean"),
        random_sds=_component(combined, "random_sd"),
        alpha=_component(combined, "alpha"),
        offset_handling=_component(combined, "offset_coefficient"),
        predicted_means=_component(combined, "predicted_mean"),
        log_likelihood=_component(combined, "log_likelihood"),
        metadata=meta,
    )


def write_nlogit_template(
    path: str | Path,
    fixed_variables: list[str],
    random_variables: list[str],
    offset_variable: str,
    include_intercept: bool = True,
) -> Path:
    """Write a fill-in template for NLOGIT parameter estimates."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if include_intercept:
        rows.append(
            {
                "component": "coefficient",
                "variable": "Intercept",
                "estimate": "",
                "std_error": "",
                "notes": "NLOGIT constant/ONE coefficient.",
            }
        )
    for variable in fixed_variables:
        rows.append(
            {
                "component": "coefficient",
                "variable": variable,
                "estimate": "",
                "std_error": "",
                "notes": "",
            }
        )
    for variable in random_variables:
        rows.append(
            {
                "component": "random_mean",
                "variable": variable,
                "estimate": "",
                "std_error": "",
                "notes": "Mean of normally distributed random parameter.",
            }
        )
        rows.append(
            {
                "component": "random_sd",
                "variable": variable,
                "estimate": "",
                "std_error": "",
                "notes": "Standard deviation of normally distributed random parameter.",
            }
        )
    rows.extend(
        [
            {
                "component": "alpha",
                "variable": "alpha",
                "estimate": "",
                "std_error": "",
                "notes": "NB2 dispersion alpha where Var(y)=mu+alpha*mu^2.",
            },
            {
                "component": "offset_coefficient",
                "variable": offset_variable,
                "estimate": 1.0,
                "std_error": "",
                "notes": "Leave as 1 only if NLOGIT used this variable as a fixed offset.",
            },
            {
                "component": "log_likelihood",
                "variable": "LL",
                "estimate": "",
                "std_error": "",
                "notes": "Final simulated log-likelihood.",
            },
        ]
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def write_nlogit_prediction_template(path: str | Path, row_indices: pd.Series) -> Path:
    """Write an observation-level fill-in template for NLOGIT predicted means."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": row_indices,
            "predicted_mean": "",
            "notes": "Fill with NLOGIT fitted mean mu for this row.",
        }
    ).to_csv(output, index=False)
    return output


def _component(table: pd.DataFrame, component: str) -> pd.DataFrame:
    return table.loc[table["component"] == component].reset_index(drop=True)


def _normalize_component(value: str) -> str:
    token = value.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "coef": "coefficient",
        "coeff": "coefficient",
        "coefficients": "coefficient",
        "fixed": "coefficient",
        "fixed_mean": "coefficient",
        "constant": "coefficient",
        "intercept": "coefficient",
        "random_parameter_mean": "random_mean",
        "rp_mean": "random_mean",
        "mean": "random_mean",
        "random_parameter_sd": "random_sd",
        "random_standard_deviation": "random_sd",
        "sd": "random_sd",
        "standard_deviation": "random_sd",
        "dispersion": "alpha",
        "nb_alpha": "alpha",
        "overdispersion": "alpha",
        "offset": "offset_coefficient",
        "offset_coeff": "offset_coefficient",
        "offset_coefficient": "offset_coefficient",
        "prediction": "predicted_mean",
        "predicted": "predicted_mean",
        "predicted_count": "predicted_mean",
        "predicted_counts": "predicted_mean",
        "mu": "predicted_mean",
        "ll": "log_likelihood",
        "loglikelihood": "log_likelihood",
        "log_likelihood": "log_likelihood",
    }
    return aliases.get(token, token)


def _normalize_variable(value: str) -> str:
    token = value.strip()
    lowered = token.lower().replace(" ", "")
    if lowered in {"const", "constant", "one", "intercept"}:
        return "Intercept"
    if lowered in {"ll", "loglikelihood", "log_likelihood"}:
        return "LL"
    if lowered in {"dispersion", "nb_alpha", "overdispersion"}:
        return "alpha"
    return token


def _first_present_column(table: pd.DataFrame, *candidates: str) -> str | None:
    by_lower = {column.lower(): column for column in table.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None
