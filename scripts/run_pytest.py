#!/usr/bin/env python3
"""Run pytest with early warning filters for noisy third-party imports."""

from __future__ import annotations

import sys
import warnings


def _install_warning_filters() -> None:
    try:
        from pyparsing.warnings import PyparsingDeprecationWarning
    except Exception:
        return

    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)


def main() -> int:
    _install_warning_filters()

    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
