"""Backward-compatible import shim for ICA PSD cache helpers."""

from autoclean.functions.visualization._ica_psd_cache import (  # noqa: F401
    ICAPSDCache,
    clear_psd_cache,
    get_cached_component_psds,
    get_psd_cache_stats,
)
