"""STAR spatial filtering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mne

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from meegkit import star


def _import_star():
    """Import the meegkit STAR implementation with guidance if missing."""

    try:
        from meegkit import star  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'meegkit' package is required for STAR cleaning. Install it with `pip install meegkit`."
        ) from exc
    return star


def run_star_cleaning(raw: mne.io.BaseRaw, *, lmbda: float = 2.0) -> mne.io.BaseRaw:
    """Apply the STAR spatial filter to attenuate artifacts."""

    star = _import_star()
    x_sc = raw.get_data().T
    y, _, _ = star.star(x_sc, lmbda)
    cleaned = raw.copy()
    cleaned._data = y.T
    return cleaned


__all__ = ["run_star_cleaning"]
