"""
Compatibility wrapper for stimulus-location validation.

Use `validate_stimulus_location_routes_fixed.py` as the canonical implementation.
This file is kept only to avoid breaking older local commands.
"""

from __future__ import annotations

from tests.utils.validate_stimulus_location_routes_fixed import run_full_validation


def main() -> int:
    _, passed = run_full_validation()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
