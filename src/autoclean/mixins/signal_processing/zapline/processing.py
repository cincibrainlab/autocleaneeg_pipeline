"""Zapline helpers for line-noise removal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mne

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from meegkit import dss


def _import_dss():
    """Import the meegkit DSS helpers with guidance if they are missing."""

    try:
        from meegkit import dss  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'meegkit' package is required for Zapline. Install it with `pip install meegkit`."
        ) from exc
    return dss


def run_zapline(
    raw: mne.io.BaseRaw,
    *,
    line_freq: float = 60.0,
    nkeep: int = 1,
) -> mne.io.BaseRaw:
    """Remove line noise using the DSS-based Zapline algorithm."""

    dss = _import_dss()
    data = raw.get_data().T
    sfreq = raw.info["sfreq"]
    out, _ = dss.dss_line(data, line_freq, sfreq, nkeep=nkeep)
    cleaned = raw.copy()
    cleaned._data = out.T
    return cleaned


__all__ = ["run_zapline"]
