"""Shared helpers for ICA visualization caches."""

from __future__ import annotations

import hashlib

import numpy as np
from mne.preprocessing import ICA


def get_ica_mixing_matrix(ica: ICA) -> np.ndarray:
    """Return the fitted ICA mixing matrix across supported MNE versions."""
    mixing = getattr(ica, "mixing_matrix_", None)
    if mixing is None:
        mixing = getattr(ica, "mixing_", None)
    if mixing is None:
        raise AttributeError("ICA has no mixing matrix; ensure ICA is fitted.")
    return np.asarray(mixing)


def get_ica_mixing_matrix_hash(ica: ICA) -> str:
    """Return a stable hash for the fitted ICA mixing matrix."""
    return hashlib.md5(get_ica_mixing_matrix(ica).tobytes()).hexdigest()
