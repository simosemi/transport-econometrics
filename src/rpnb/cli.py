"""Command line interface for RPNB."""

from __future__ import annotations

import argparse
from pathlib import Path

from rpnb.checkpoint import load_run_metadata
from rpnb.config import load_model_spec
from rpnb.model import RandomParametersNegativeBinomial


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpnb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit",
        help="Estimate a random parameters negative binomial model.",
    )
    fit.add_argument("--data", default=None, help="Path to CSV input data.")
    fit.add_argument("--spec", default=None, help="Path to YAML model specification.")
    fit.add_argument("--out", default=None, help="Output directory for timestamped run folder.")
    fit.add_argument("--no-export", action="store_true", help="Skip CSV/Excel/HTML export.")
    fit.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save optimizer checkpoint every N iterations. Use 0 to disable.",
    )
    fit.add_argument(
        "--resume",
        default=None,
        help="Resume from a previous RPNB run directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        resume_dir = Path(args.resume) if args.resume else None
        metadata = load_run_metadata(resume_dir) if resume_dir is not None else {}
        spec_path = Path(args.spec) if args.spec else None
        data_path = Path(args.data) if args.data else None

        if resume_dir is not None:
            if spec_path is None:
                spec_path = resume_dir / "model_spec.yaml"
            if data_path is None and metadata.get("data_path"):
                data_path = Path(metadata["data_path"])
        if spec_path is None:
            parser.error("--spec is required unless --resume points to a run with model_spec.yaml.")
        if data_path is None:
            parser.error(
                "--data is required unless --resume points to a run with run_metadata.yaml data_path."
            )

        spec = load_model_spec(spec_path)
        model = RandomParametersNegativeBinomial.from_spec(spec)
        if args.checkpoint_interval is not None:
            if args.checkpoint_interval < 0:
                parser.error("--checkpoint-interval must be non-negative.")
            model.checkpoint_interval = int(args.checkpoint_interval)
        output_dir = Path(args.out) if args.out else None
        results = model.fit(
            data_path,
            save_run=True,
            output_dir=output_dir,
            export=not args.no_export,
            resume_from=resume_dir,
            spec_path=spec_path,
        )
        print(results.summary())
        if results.run_dir is not None:
            print(f"Run directory: {results.run_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
