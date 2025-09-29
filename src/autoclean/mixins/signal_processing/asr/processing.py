"""Artifact Subspace Reconstruction (ASR) helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mne
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from meegkit.asr import ASR
    from meegkit.utils.matrix import sliding_window


def _import_meegkit_for_asr():
    """Import the meegkit ASR dependencies with friendly guidance."""

    try:
        from meegkit.asr import ASR  # type: ignore
        from meegkit.utils.matrix import sliding_window  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'meegkit' package is required for ASR. Install it with `pip install meegkit`."
        ) from exc
    return ASR, sliding_window


def apply_asr(
    raw: mne.io.BaseRaw,
    *,
    method: str = "euclid",
    cutoff: float = 20.0,
    train_duration: int = 20,
) -> mne.io.BaseRaw:
    """Apply ASR to a Raw object and return a cleaned copy."""

    ASR, sliding_window = _import_meegkit_for_asr()

    sfreq = int(raw.info["sfreq"])
    nchan = raw.info["nchan"]
    raw_array = raw.get_data()

    asr = ASR(method=method, cutoff=cutoff)
    train_idx = np.arange(0, train_duration * sfreq, dtype=int)
    asr.fit(raw_array[:, train_idx])

    X = sliding_window(raw_array, window=int(sfreq), step=int(sfreq))
    Y = np.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i, :] = asr.transform(X[:, i, :])

    clean_array = Y.reshape(nchan, -1)
    cleaned = raw.copy()
    cleaned._data = clean_array
    return cleaned


__all__ = ["apply_asr"]
