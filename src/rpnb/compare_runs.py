"""Compare completed RPNB run directories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


@dataclass
class RPNBRunComparison:
    """Tables comparing completed RPNB runs."""

    metrics: pd.DataFrame
    random_parameter_means: pd.DataFrame
    random_parameter_sds: pd.DataFrame

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics_csv": directory / "comparison_metrics.csv",
            "random_means_csv": directory / "random_parameter_means.csv",
            "random_sds_csv": directory / "random_parameter_sds.csv",
            "excel": directory / "rpnb_run_comparison.xlsx",
            "html": directory / "rpnb_run_comparison.html",
        }
        self.metrics.to_csv(paths["metrics_csv"], index=False)
        self.random_parameter_means.to_csv(paths["random_means_csv"], index=False)
        self.random_parameter_sds.to_csv(paths["random_sds_csv"], index=False)
        with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
            self.metrics.to_excel(writer, sheet_name="metrics", index=False)
            self.random_parameter_means.to_excel(
                writer, sheet_name="random_means", index=False
            )
            self.random_parameter_sds.to_excel(
                writer, sheet_name="random_sds", index=False
            )
        paths["html"].write_text(self._to_html(), encoding="utf-8")
        return paths

    def _to_html(self) -> str:
        sections = [
            "<h1>RPNB Run Comparison</h1>",
            "<h2>Fit Metrics</h2>",
            self.metrics.to_html(index=False),
            "<h2>Random Parameter Means</h2>",
            self.random_parameter_means.to_html(index=False),
            "<h2>Random Parameter SDs</h2>",
            self.random_parameter_sds.to_html(index=False),
        ]
        return "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>RPNB run comparison</title>",
                "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;}"
                "table{border-collapse:collapse;margin-bottom:2rem;}td,th{border:1px solid #ddd;"
                "padding:0.35rem 0.5rem;}th{background:#f5f5f5;}</style>",
                "</head><body>",
                *sections,
                "</body></html>",
            ]
        )


def compare_runs(run_dirs: Sequence[str | Path]) -> RPNBRunComparison:
    """Build comparison tables for completed RPNB run directories."""

    metric_rows: list[dict[str, Any]] = []
    random_mean_rows: list[dict[str, Any]] = []
    random_sd_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        fit_statistics = _read_single_row(run_path / "fit_statistics.csv")
        convergence = _read_optional_single_row(run_path / "convergence.csv")
        coefficients = _read_coefficients(run_path / "coefficients.csv")
        random_tests = _read_optional_frame(run_path / "random_parameter_tests.csv")
        random_screening = _read_optional_frame(
            run_path / "random_parameter_screening.csv"
        )
        run_label = run_path.name
        random_sd_summary = _random_sd_significance(
            coefficients,
            random_tests,
            random_screening,
        )

        metric_rows.append(
            {
                "run": run_label,
                "run_dir": str(run_path),
                "LL": _get(fit_statistics, "log_likelihood"),
                "AIC": _get(fit_statistics, "AIC"),
                "BIC": _get(fit_statistics, "BIC"),
                "alpha": _alpha(fit_statistics, coefficients),
                "convergence_quality": _get(convergence, "convergence_quality"),
                **random_sd_summary,
                "n_parameters": _get(fit_statistics, "n_parameters"),
            }
        )
        random_mean_rows.extend(
            _parameter_rows(run_label, run_path, coefficients, "random_mean")
        )
        random_sd_rows.extend(
            _parameter_rows(
                run_label,
                run_path,
                coefficients,
                "random_sd",
                random_screening=random_screening,
            )
        )

    return RPNBRunComparison(
        metrics=_rank_metrics(pd.DataFrame(metric_rows)),
        random_parameter_means=pd.DataFrame(random_mean_rows),
        random_parameter_sds=pd.DataFrame(random_sd_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rpnb.compare_runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Completed RPNB run directories to compare.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory where comparison CSV, Excel, and HTML files will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = compare_runs(args.runs)
    paths = report.export(args.out)
    print(f"Compared {len(args.runs)} RPNB runs.")
    print(f"Metrics: {paths['metrics_csv']}")
    print(f"Excel: {paths['excel']}")
    print(f"HTML: {paths['html']}")
    return 0


def _read_single_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required run output not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Run output is empty: {path}")
    return frame.iloc[0].to_dict()


def _read_optional_single_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {} if frame.empty else frame.iloc[0].to_dict()


def _read_coefficients(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required coefficient table not found: {path}")
    return pd.read_csv(path)


def _read_optional_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parameter_rows(
    run_label: str,
    run_path: Path,
    coefficients: pd.DataFrame,
    component: str,
    random_screening: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = coefficients.loc[coefficients["component"] == component]
    screening_by_variable = (
        random_screening.set_index("variable")
        if random_screening is not None
        and not random_screening.empty
        and "variable" in random_screening.columns
        else pd.DataFrame()
    )
    for row in subset.to_dict(orient="records"):
        variable = row.get("variable")
        screening_recommendation = None
        if not screening_by_variable.empty and variable in screening_by_variable.index:
            screening_recommendation = screening_by_variable.loc[
                variable,
                "recommendation",
            ]
        rows.append(
            {
                "run": run_label,
                "run_dir": str(run_path),
                "parameter": row.get("parameter"),
                "variable": variable,
                "estimate": row.get("estimate"),
                "std_error": row.get("std_error"),
                "z_value": row.get("z_value"),
                "p_value": row.get("p_value"),
                "significance": row.get("significance"),
                "screening_recommendation": screening_recommendation,
            }
        )
    return rows


def _alpha(fit_statistics: dict[str, Any], coefficients: pd.DataFrame) -> Any:
    alpha = _get(fit_statistics, "alpha")
    if alpha not in (None, "") and pd.notna(alpha):
        return alpha
    rows = coefficients.loc[coefficients["parameter"] == "alpha"]
    return None if rows.empty else rows.iloc[0].get("estimate")


def _get(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if pd.isna(value):
        return None
    return value


def _random_sd_significance(
    coefficients: pd.DataFrame,
    random_tests: pd.DataFrame,
    random_screening: pd.DataFrame,
) -> dict[str, Any]:
    sd_rows = coefficients.loc[coefficients["component"] == "random_sd"]
    p_values = (
        pd.to_numeric(sd_rows.get("p_value", pd.Series(dtype=float)), errors="coerce")
        if not sd_rows.empty
        else pd.Series(dtype=float)
    )
    significant = p_values < 0.05
    summary: dict[str, Any] = {
        "n_random_sds": int(len(sd_rows)),
        "n_significant_random_sds": int(significant.sum()),
        "min_random_sd_p_value": None
        if p_values.dropna().empty
        else float(p_values.min(skipna=True)),
        "random_sd_significance": (
            "At least one significant"
            if bool(significant.any())
            else ("None significant" if len(sd_rows) else "No random SDs")
        ),
    }
    if not random_tests.empty and "recommendation" in random_tests.columns:
        keep_random = random_tests["recommendation"].astype(str).eq("Keep Random")
        summary["n_lr_keep_random"] = int(keep_random.sum())
        summary["n_lr_treat_fixed"] = int(
            random_tests["recommendation"].astype(str).eq("Treat as Fixed").sum()
        )
    else:
        summary["n_lr_keep_random"] = 0
        summary["n_lr_treat_fixed"] = 0
    if not random_screening.empty and "recommendation" in random_screening.columns:
        recommendations = random_screening["recommendation"].astype(str)
        summary["n_screen_keep_random"] = int(recommendations.eq("Keep Random").sum())
        summary["n_screen_convert_to_fixed"] = int(
            recommendations.eq("Convert to Fixed").sum()
        )
        summary["n_sd_effectively_zero"] = int(
            _boolean_series(
                random_screening.get("sd_effectively_zero", pd.Series(dtype=bool))
            ).sum()
        )
        summary["n_sd_not_significant"] = int(
            _boolean_series(
                random_screening.get(
                    "sd_not_statistically_significant",
                    pd.Series(dtype=bool),
                )
            ).sum()
        )
    else:
        summary["n_screen_keep_random"] = 0
        summary["n_screen_convert_to_fixed"] = 0
        summary["n_sd_effectively_zero"] = 0
        summary["n_sd_not_significant"] = 0
    return summary


def _boolean_series(values: Any) -> pd.Series:
    series = pd.Series(values)
    if series.empty:
        return pd.Series(dtype=bool)
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if not isinstance(value, bool)
        else value
    ).fillna(False)


def _rank_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    ranked = metrics.copy()
    ranked["model_rank"] = pd.to_numeric(ranked["AIC"], errors="coerce").rank(
        method="min",
        ascending=True,
    )
    ranked["LL_rank"] = pd.to_numeric(ranked["LL"], errors="coerce").rank(
        method="min",
        ascending=False,
    )
    ranked["BIC_rank"] = pd.to_numeric(ranked["BIC"], errors="coerce").rank(
        method="min",
        ascending=True,
    )
    ranked["best_model_by_AIC"] = ranked["model_rank"].eq(1)
    return ranked.sort_values(["model_rank", "run"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
