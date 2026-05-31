"""Compatibility wrapper for ``python -m rpopit.model_comparison_report``."""

from rpopit.model_comparison_report import main


if __name__ == "__main__":
    raise SystemExit(main())
