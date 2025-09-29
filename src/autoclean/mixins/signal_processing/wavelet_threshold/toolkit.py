"""Compatibility re-exports for helper blocks once bundled with the wavelet plugin."""

from __future__ import annotations

from autoclean.mixins.signal_processing.asr import apply_asr
from autoclean.mixins.signal_processing.autoreject import (
    AutorejectResult,
    run_autoreject,
    run_autoreject_raw,
)
from autoclean.mixins.signal_processing.star_cleaning import run_star_cleaning
from autoclean.mixins.signal_processing.zapline import run_zapline

__all__ = [
    "AutorejectResult",
    "run_autoreject",
    "run_autoreject_raw",
    "run_star_cleaning",
    "apply_asr",
    "run_zapline",
]
