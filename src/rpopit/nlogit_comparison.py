"""Benchmark/report helpers for comparing rpopit with NLOGIT exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpopit.output import RPOpitResults

CANONICAL_COMPONENTS = {
    "coefficient",
    "threshold",
    "random_mean",
    "random_sd",
    "log_likelihood",
}


@dataclass
class NlogitComparisonReport:
    """Side-by-side rpopit versus NLOGIT benchmark output."""

    combined: pd.DataFrame
    coefficients: pd.DataFrame
    thresholds: pd.DataFrame
    random_means: pd.DataFrame
    random_sds: pd.DataFrame
    log_likelihood: pd.DataFrame
    metadata: dict[str, Any]

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "combined": directory / "nlogit_rpopit_parameter_comparison.csv",
            "coefficients": directory / "coefficients_comparison.csv",
            "thresholds": directory / "thresholds_comparison.csv",
            "random_means": directory / "random_parameter_means_comparison.csv",
            "random_sds": directory / "random_parameter_sds_comparison.csv",
            "log_likelihood": directory / "log_likelihood_comparison.csv",
            "excel": directory / "nlogit_rpopit_comparison.xlsx",
            "html": directory / "nlogit_rpopit_comparison.html",
        }
        self.combined.to_csv(paths["combined"], index=False)
        self.coefficients.to_csv(paths["coefficients"], index=False)
        self.thresholds.to_csv(paths["thresholds"], index=False)
        self.random_means.to_csv(paths["random_means"], index=False)
        self.random_sds.to_csv(paths["random_sds"], index=False)
        self.log_likelihood.to_csv(paths["log_likelihood"], index=False)
        with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
            pd.DataFrame([self.metadata]).to_excel(writer, sheet_name="metadata", index=False)
            self.combined.to_excel(writer, sheet_name="all_parameters", index=False)
            self.coefficients.to_excel(writer, sheet_name="coefficients", index=False)
            self.thresholds.to_excel(writer, sheet_name="thresholds", index=False)
            self.random_means.to_excel(writer, sheet_name="random_means", index=False)
            self.random_sds.to_excel(writer, sheet_name="random_sds", index=False)
            self.log_likelihood.to_excel(writer, sheet_name="log_likelihood", index=False)
        paths["html"].write_text(self._to_html(), encoding="utf-8")
        return paths

    def _to_html(self) -> str:
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>NLOGIT vs rpopit</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                "<h1>NLOGIT vs rpopit comparison</h1>",
                "<h2>Metadata</h2>",
                pd.DataFrame([self.metadata]).to_html(index=False),
                "<h2>Highlighted differences</h2>",
                self.combined.sort_values("abs_difference", ascending=False)
                .head(25)
                .to_html(index=False),
                "<h2>Coefficients</h2>",
                self.coefficients.to_html(index=False),
                "<h2>Thresholds</h2>",
                self.thresholds.to_html(index=False),
                "<h2>Random parameter means</h2>",
                self.random_means.to_html(index=False),
                "<h2>Random parameter SDs</h2>",
                self.random_sds.to_html(index=False),
                "<h2>Log-likelihood</h2>",
                self.log_likelihood.to_html(index=False),
                "</body></html>",
            ]
        )


def rpopit_canonical_table(results: RPOpitResults) -> pd.DataFrame:
    """Normalize rpopit estimates to the NLOGIT comparison interchange format."""

    rows: list[dict[str, Any]] = []
    for row in results.parameter_table.to_dict(orient="records"):
        component = row["component"]
        variable = row["variable"]
        if component == "fixed_mean":
            out_component = "coefficient"
        elif component == "threshold":
            out_component = "threshold"
            variable = row["parameter"]
        elif component in {"random_mean", "random_sd"}:
            out_component = component
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
            "component": "log_likelihood",
            "variable": "LL",
            "estimate": float(results.log_likelihood),
            "std_error": np.nan,
            "source_parameter": "log_likelihood",
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


def compare_with_nlogit(
    rpopit_results: RPOpitResults,
    nlogit_results_path: str | Path,
    tolerance: float = 1e-4,
    metadata: dict[str, Any] | None = None,
) -> NlogitComparisonReport:
    """Create side-by-side comparison tables for rpopit and NLOGIT."""

    rpopit_table = rpopit_canonical_table(rpopit_results)
    nlogit_table = load_nlogit_canonical_table(nlogit_results_path)
    combined = nlogit_table.merge(
        rpopit_table,
        on=["component", "variable"],
        how="outer",
        suffixes=("_nlogit", "_rpopit"),
        indicator=True,
    )
    combined["difference"] = combined["estimate_rpopit"] - combined["estimate_nlogit"]
    combined["abs_difference"] = combined["difference"].abs()
    combined["relative_difference"] = combined["difference"] / combined[
        "estimate_nlogit"
    ].replace(0.0, np.nan)
    combined["within_tolerance"] = combined["abs_difference"] <= tolerance
    combined = combined.sort_values(["component", "variable"]).reset_index(drop=True)

    meta = dict(metadata or {})
    meta.update(
        {
            "tolerance": tolerance,
            "max_abs_difference": float(combined["abs_difference"].max(skipna=True)),
            "n_compared_rows": int((combined["_merge"] == "both").sum()),
            "n_missing_from_nlogit": int((combined["_merge"] == "right_only").sum()),
            "n_missing_from_rpopit": int((combined["_merge"] == "left_only").sum()),
        }
    )

    return NlogitComparisonReport(
        combined=combined,
        coefficients=_component(combined, "coefficient"),
        thresholds=_component(combined, "threshold"),
        random_means=_component(combined, "random_mean"),
        random_sds=_component(combined, "random_sd"),
        log_likelihood=_component(combined, "log_likelihood"),
        metadata=meta,
    )


def write_nlogit_template(path: str | Path, variables: list[str]) -> Path:
    """Write a fill-in template for NLOGIT estimates."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"component": "coefficient", "variable": "x", "estimate": "", "std_error": ""},
        {"component": "random_mean", "variable": "z", "estimate": "", "std_error": ""},
        {"component": "random_sd", "variable": "z", "estimate": "", "std_error": ""},
        {
            "component": "threshold",
            "variable": "threshold[1]",
            "estimate": "",
            "std_error": "",
        },
        {
            "component": "threshold",
            "variable": "threshold[2]",
            "estimate": "",
            "std_error": "",
        },
        {
            "component": "log_likelihood",
            "variable": "LL",
            "estimate": "",
            "std_error": "",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False)
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
        "cutpoint": "threshold",
        "cut_point": "threshold",
        "thresholds": "threshold",
        "random_parameter_mean": "random_mean",
        "rp_mean": "random_mean",
        "mean": "random_mean",
        "random_parameter_sd": "random_sd",
        "random_standard_deviation": "random_sd",
        "sd": "random_sd",
        "standard_deviation": "random_sd",
        "ll": "log_likelihood",
        "loglikelihood": "log_likelihood",
        "log_likelihood": "log_likelihood",
    }
    return aliases.get(token, token)


def _normalize_variable(value: str) -> str:
    token = value.strip()
    lowered = token.lower().replace(" ", "")
    if lowered in {"ll", "loglikelihood", "log_likelihood"}:
        return "LL"
    if lowered.startswith("mu"):
        suffix = lowered.replace("mu", "")
        if suffix.isdigit():
            return f"threshold[{suffix}]"
    return token
