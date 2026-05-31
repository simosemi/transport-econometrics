"""Compatibility wrapper for ``python -m rpopit.validate_hsis_data``."""

from rpopit.validate_hsis_data import main


if __name__ == "__main__":
    raise SystemExit(main())
