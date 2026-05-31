"""CSV schema validation for real crash severity datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class DataValidationReport:
    """Validation outputs for a crash severity CSV."""

    cleaned_data: pd.DataFrame
    issues: pd.DataFrame
    missing_summary: pd.DataFrame
    summary: dict[str, Any]

    @property
    def valid(self) -> bool:
        return int(self.summary.get("n_errors", 0)) == 0


def load_schema(path: str | Path) -> dict[str, Any]:
    """Load a YAML or CSV schema file."""

    schema_path = Path(path)
    if schema_path.suffix.lower() in {".yaml", ".yml"}:
        with schema_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if "columns" not in raw:
            raise ValueError("Schema YAML must contain a 'columns' mapping.")
        return raw

    if schema_path.suffix.lower() == ".csv":
        rows = pd.read_csv(schema_path).fillna("")
        columns: dict[str, dict[str, Any]] = {}
        for row in rows.to_dict(orient="records"):
            name = str(row["column"])
            allowed = _parse_allowed_values(row.get("allowed_values", ""))
            columns[name] = {
                "role": _blank_to_none(row.get("role")),
                "type": str(row.get("type", "numeric") or "numeric"),
                "required": _to_bool(row.get("required", True)),
                "allow_missing": _to_bool(row.get("allow_missing", False)),
                "allowed_values": allowed,
                "description": _blank_to_none(row.get("description")),
            }
        return {"columns": columns}

    raise ValueError("Schema path must end with .yaml, .yml, or .csv.")


def validate_csv(
    data_path: str | Path,
    schema_path: str | Path,
    missing: str = "drop",
) -> DataValidationReport:
    """Read and validate a CSV against a schema."""

    data = pd.read_csv(data_path)
    schema = load_schema(schema_path)
    return validate_dataframe(data, schema, missing=missing)


def validate_dataframe(
    data: pd.DataFrame,
    schema: dict[str, Any],
    missing: str = "drop",
) -> DataValidationReport:
    """Validate a DataFrame and return issues plus a cleaned copy."""

    if missing not in {"drop", "error"}:
        raise ValueError("missing must be 'drop' or 'error'.")

    raw_columns = schema.get("columns", {})
    if not isinstance(raw_columns, dict):
        raise ValueError("schema['columns'] must be a mapping.")

    cleaned = data.copy()
    issues: list[dict[str, Any]] = []
    missing_rows = []
    required_for_missing: list[str] = []

    for name, spec in raw_columns.items():
        spec = spec or {}
        required = bool(spec.get("required", True))
        allow_missing = bool(spec.get("allow_missing", False))
        column_type = str(spec.get("type", "numeric")).lower()
        allowed_values = spec.get("allowed_values")

        if required and name not in cleaned.columns:
            issues.append(_issue("error", "required_column", name, "Missing required column."))
            continue
        if name not in cleaned.columns:
            continue

        missing_count = int(cleaned[name].isna().sum())
        missing_rows.append(
            {
                "column": name,
                "missing_count": missing_count,
                "missing_fraction": missing_count / len(cleaned) if len(cleaned) else 0.0,
                "allow_missing": allow_missing,
            }
        )
        if missing_count and required and not allow_missing:
            required_for_missing.append(name)
            severity = "warning" if missing == "drop" else "error"
            issues.append(
                _issue(
                    severity,
                    "missing_values",
                    name,
                    f"{missing_count} missing values found.",
                )
            )

        type_issues = _coerce_type(cleaned, name, column_type)
        issues.extend(type_issues)

        if allowed_values not in (None, ""):
            allowed = set(allowed_values)
            observed = set(cleaned.loc[cleaned[name].notna(), name].unique())
            invalid = sorted(observed - allowed)
            if invalid:
                issues.append(
                    _issue(
                        "error",
                        "allowed_values",
                        name,
                        f"Found values outside allowed set: {invalid}",
                    )
                )

    dropped_rows = 0
    if required_for_missing and missing == "drop":
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=required_for_missing)
        dropped_rows = before - len(cleaned)

    issues_frame = pd.DataFrame(
        issues,
        columns=["severity", "check", "column", "message"],
    )
    missing_frame = pd.DataFrame(
        missing_rows,
        columns=["column", "missing_count", "missing_fraction", "allow_missing"],
    )
    n_errors = int((issues_frame["severity"] == "error").sum()) if len(issues_frame) else 0
    n_warnings = (
        int((issues_frame["severity"] == "warning").sum()) if len(issues_frame) else 0
    )
    summary = {
        "n_rows_raw": int(len(data)),
        "n_rows_validated": int(len(cleaned)),
        "n_columns_raw": int(data.shape[1]),
        "n_schema_columns": int(len(raw_columns)),
        "dropped_rows_missing": int(dropped_rows),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "missing_policy": missing,
    }
    return DataValidationReport(cleaned, issues_frame, missing_frame, summary)


def export_validation_report(report: DataValidationReport, output_dir: str | Path) -> dict[str, Path]:
    """Export validation report tables. Raw input data are never modified."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": directory / "validation_summary.csv",
        "issues": directory / "validation_issues.csv",
        "missing": directory / "missing_summary.csv",
    }
    pd.DataFrame([report.summary]).to_csv(paths["summary"], index=False)
    report.issues.to_csv(paths["issues"], index=False)
    report.missing_summary.to_csv(paths["missing"], index=False)
    return paths


def _coerce_type(data: pd.DataFrame, name: str, column_type: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if column_type in {"numeric", "float", "integer", "int", "binary"}:
        original = data[name]
        coerced = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & coerced.isna()
        if invalid.any():
            issues.append(
                _issue(
                    "error",
                    "type",
                    name,
                    f"{int(invalid.sum())} non-numeric values found.",
                )
            )
        data[name] = coerced

    if column_type in {"integer", "int"}:
        non_integer = data[name].notna() & ~np.isclose(data[name] % 1, 0.0)
        if non_integer.any():
            issues.append(
                _issue(
                    "error",
                    "type",
                    name,
                    f"{int(non_integer.sum())} non-integer values found.",
                )
            )
    if column_type == "binary":
        observed = set(data.loc[data[name].notna(), name].unique())
        invalid = sorted(observed - {0, 1, 0.0, 1.0})
        if invalid:
            issues.append(
                _issue("error", "type", name, f"Binary column contains {invalid}.")
            )
    return issues


def _parse_allowed_values(value: Any) -> list[Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    parts = [part.strip() for part in str(value).split("|") if part.strip()]
    parsed: list[Any] = []
    for part in parts:
        try:
            number = float(part)
            parsed.append(int(number) if number.is_integer() else number)
        except ValueError:
            parsed.append(part)
    return parsed


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _blank_to_none(value: Any) -> Any:
    return None if value is None or value == "" else value


def _issue(severity: str, check: str, column: str, message: str) -> dict[str, Any]:
    return {
        "severity": severity,
        "check": check,
        "column": column,
        "message": message,
    }
