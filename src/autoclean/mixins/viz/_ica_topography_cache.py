"""Backward-compatible import shim for ICA topography cache helpers."""

from autoclean.functions.visualization._ica_topography_cache import (  # noqa: F401
    ICATopographyCache,
    apply_cached_topography,
    clear_topography_cache,
    get_cached_topographies,
    get_topography_cache_stats,
)
