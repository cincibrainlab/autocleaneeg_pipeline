"""Backward-compatible import shim for ICA source cache helpers."""

from autoclean.functions.visualization._ica_sources_cache import (  # noqa: F401
    ICASourcesCache,
    cache_aware_ica_method,
    clear_ica_cache,
    get_cached_ica_sources,
    get_ica_cache_stats,
    invalidate_ica_cache,
)
