"""Epoching Functions.

This module contains standalone functions for creating and processing epochs
from continuous EEG data. Includes regular epoching, event-based epoching,
and epoch quality assessment.

Functions
---------
create_regular_epochs : Create fixed-length epochs
create_eventid_epochs : Create epochs around specific events
create_statistical_learning_epochs : Create statistical learning epochs
detect_outlier_epochs : Identify outlier epochs
gfp_clean_epochs : Clean epochs using global field power
"""

from typing import Optional

import mne

from .eventid import create_eventid_epochs
from .quality import detect_outlier_epochs, gfp_clean_epochs

# Import implemented functions
from .regular import create_regular_epochs
from .statistical import create_statistical_learning_epochs


def create_sl_epochs(
    data: mne.io.Raw,
    tmin: float = 0,
    tmax: Optional[float] = None,
    num_syllables_per_epoch: Optional[int] = None,
    **kwargs,
):
    """Backward-compatible wrapper for statistical learning epoch creation.

    Older callers used ``tmax`` and ``num_syllables_per_epoch`` and expected only
    the primary epochs object rather than the ``(epochs, epochs_clean)`` tuple.
    """
    if num_syllables_per_epoch is not None:
        kwargs["num_syllables"] = num_syllables_per_epoch
    elif "num_syllables" not in kwargs:
        kwargs["num_syllables"] = 30

    if tmax is not None:
        approx_syllables = int(round(tmax / 0.3))
        if approx_syllables > 0 and "num_syllables_per_epoch" not in kwargs:
            kwargs["num_syllables"] = approx_syllables

    try:
        epochs, _epochs_clean = create_statistical_learning_epochs(
            data=data,
            tmin=tmin,
            **kwargs,
        )
        return epochs
    except ValueError as exc:
        if "Not enough events to skip initial 5 events" in str(exc):
            raise ValueError("No events found") from exc
        raise


__all__ = [
    "create_regular_epochs",
    "create_eventid_epochs",
    "create_statistical_learning_epochs",
    "create_sl_epochs",
    "detect_outlier_epochs",
    "gfp_clean_epochs",
]
