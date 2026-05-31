"""Validate an HSIS-style crash severity CSV against a schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from rpopit.data_validation import export_validation_report, validate_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to HSIS-style crash CSV.")
    parser.add_argument("--schema", required=True, help="Path to YAML or CSV schema.")
    parser.add_argument("--out", required=True, help="Output directory for validation tables.")
    parser.add_argument(
        "--missing",
        choices=["drop", "error"],
        default="drop",
        help="How to handle missing required values.",
    )
    parser.add_argument(
        "--write-cleaned",
        action="store_true",
        help="Write a cleaned copy to the output folder. Raw input is never modified.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_csv(args.data, args.schema, missing=args.missing)
    paths = export_validation_report(report, args.out)
    if args.write_cleaned:
        cleaned_path = Path(args.out) / "validated_cleaned_data.csv"
        report.cleaned_data.to_csv(cleaned_path, index=False)
        paths["cleaned"] = cleaned_path

    print("Validation summary:", report.summary)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
