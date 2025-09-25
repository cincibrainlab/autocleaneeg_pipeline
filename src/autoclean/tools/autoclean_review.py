"""Backward-compatible wrapper for the legacy Autoclean review GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .autoclean_exclude import run_autoclean_exclude

__all__ = ["run_autoclean_review"]


def run_autoclean_review(autoclean_dir: Union[str, Path] | None) -> None:
    """Launch the manual exclusion GUI (legacy entry point).

    Parameters
    ----------
    autoclean_dir : str | Path | None
        AutoClean workspace directory. When ``None`` a directory picker is
        presented to the user.

    Notes
    -----
    This wrapper preserves the historical ``autoclean_review`` entry point while
    the implementation now lives in :mod:`autoclean.tools.autoclean_exclude`.
    """

    run_autoclean_exclude(autoclean_dir)
