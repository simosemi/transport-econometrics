"""Generate an Ordered Probit vs RPOPIT comparison report for real data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rpopit.config import load_model_spec
from rpopit.data_validation import export_validation_report, load_schema, validate_dataframe
from rpopit.model_comparison import compare_ordered_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to crash severity CSV.")
    parser.add_argument("--spec", required=True, help="Path to rpopit YAML model spec.")
    parser.add_argument("--out", required=True, help="Output directory for comparison report.")
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional YAML or CSV schema to validate before estimation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_model_spec(args.spec)
    data = pd.read_csv(args.data)
    if args.schema:
        schema = load_schema(args.schema)
        validation = validate_dataframe(data, schema, missing=spec.missing)
        export_validation_report(validation, out_dir / "validation")
        if not validation.valid:
            print("Data validation failed. See validation reports in:", out_dir / "validation")
            return 1
        data = validation.cleaned_data

    report = compare_ordered_models(data, spec)
    paths = report.export(out_dir)
    print("Model comparison complete.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
