"""Command line interface for rpopit."""

from __future__ import annotations

import argparse
from pathlib import Path

from rpopit.config import load_model_spec
from rpopit.model import RandomParametersOrderedProbit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpopit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Estimate a random parameters ordered probit.")
    fit.add_argument("--data", required=True, help="Path to CSV input data.")
    fit.add_argument("--spec", required=True, help="Path to YAML model specification.")
    fit.add_argument("--out", default=None, help="Output directory for timestamped run folder.")
    fit.add_argument("--no-export", action="store_true", help="Skip CSV/Excel/HTML export.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        spec = load_model_spec(args.spec)
        model = RandomParametersOrderedProbit.from_spec(spec)
        output_dir = Path(args.out) if args.out else None
        results = model.fit(args.data, save_run=True, output_dir=output_dir, export=not args.no_export)
        print(results.summary())
        if results.run_dir is not None:
            print(f"Run directory: {results.run_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
