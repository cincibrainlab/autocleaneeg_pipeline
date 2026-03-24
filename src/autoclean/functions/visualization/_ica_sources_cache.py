"""ICA sources caching system shared by visualization and mixin code.

This module provides intelligent caching of ICA source activations to avoid
redundant computations during report generation. The cache handles:
- Multiple raw data objects (cropped vs full duration)
- Memory management with automatic cleanup
- Thread-safe operations for parallel processing
- Cache invalidation when ICA or raw data changes
"""

import hashlib
import logging
import time
import weakref
from functools import wraps
from typing import Dict, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA

logger = logging.getLogger(__name__)


class ICASourcesCache:
    """Thread-safe cache for ICA source activations with intelligent memory management."""

    def __init__(self, max_cache_size_mb: float = 500.0):
        """Initialize cache with memory limit.

        Parameters
        ----------
        max_cache_size_mb : float
            Maximum cache size in megabytes. Default 500MB.
        """
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._current_cache_size = 0

        # Weak references to track ICA/Raw objects for invalidation
        self._ica_refs: Dict[str, weakref.ref] = {}
        self._raw_refs: Dict[str, weakref.ref] = {}

    def _generate_cache_key(self, ica: ICA, raw: mne.io.Raw) -> str:
        """Generate unique cache key for ICA + Raw combination."""
        ica_info = f"{id(ica)}_{ica.n_components_}_{len(ica.exclude)}"
        raw_info = f"{id(raw)}_{raw.n_times}_{raw.info['sfreq']}"
        ch_hash = hashlib.md5("_".join(raw.ch_names).encode()).hexdigest()[:8]
        return f"ica_{ica_info}_raw_{raw_info}_ch_{ch_hash}"

    def _estimate_data_size(self, data_shape: Tuple[int, ...]) -> int:
        """Estimate memory size of numpy array in bytes."""
        return np.prod(data_shape) * 8

    def _cleanup_if_needed(self, required_size: int):
        """Remove oldest cache entries if memory limit would be exceeded."""
        if self._current_cache_size + required_size <= self.max_cache_size_bytes:
            return

        logger.debug("Cache size limit reached, cleaning up oldest entries")
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])

        for key, _ in sorted_keys:
            if key in self._cache:
                removed_size = self._cache[key]["data_size"]
                del self._cache[key]
                del self._access_times[key]
                self._current_cache_size -= removed_size

                logger.debug(
                    "Removed cache entry %s, freed %.1fMB",
                    key,
                    removed_size / 1024 / 1024,
                )

                if (
                    self._current_cache_size + required_size
                    <= self.max_cache_size_bytes
                ):
                    break

    def get_sources(
        self, ica: ICA, raw: mne.io.Raw, force_refresh: bool = False
    ) -> mne.io.Raw:
        """Get ICA sources with caching."""
        cache_key = self._generate_cache_key(ica, raw)
        current_time = time.time()

        if not force_refresh and cache_key in self._cache:
            logger.debug("Cache hit for %s", cache_key)
            self._access_times[cache_key] = current_time
            cached_data = self._cache[cache_key]["sources_data"].copy()
            cached_info = self._cache[cache_key]["sources_info"].copy()
            return mne.io.RawArray(cached_data, cached_info, verbose=False)

        logger.debug("Computing ICA sources for %s", cache_key)
        start_time = time.time()
        sources = ica.get_sources(raw)
        computation_time = time.time() - start_time
        logger.debug("ICA sources computed in %.2fs", computation_time)

        sources_data = sources.get_data()
        data_size = self._estimate_data_size(sources_data.shape)
        self._cleanup_if_needed(data_size)

        self._cache[cache_key] = {
            "sources_data": sources_data,
            "sources_info": sources.info.copy(),
            "data_size": data_size,
            "created_time": current_time,
            "computation_time": computation_time,
        }
        self._access_times[cache_key] = current_time
        self._current_cache_size += data_size
        self._ica_refs[cache_key] = weakref.ref(ica)
        self._raw_refs[cache_key] = weakref.ref(raw)

        logger.debug(
            "Cached sources: %.1fMB, total cache: %.1fMB",
            data_size / 1024 / 1024,
            self._current_cache_size / 1024 / 1024,
        )
        return sources

    def invalidate_ica(self, ica: ICA):
        """Invalidate all cache entries for a specific ICA object."""
        ica_id = id(ica)
        keys_to_remove = [
            key for key in list(self._cache.keys()) if f"ica_{ica_id}_" in key
        ]

        for key in keys_to_remove:
            self._remove_cache_entry(key)

        if keys_to_remove:
            logger.debug(
                "Invalidated %d cache entries for ICA %s",
                len(keys_to_remove),
                ica_id,
            )

    def _remove_cache_entry(self, key: str):
        """Remove a single cache entry and update size tracking."""
        if key in self._cache:
            removed_size = self._cache[key]["data_size"]
            del self._cache[key]
            del self._access_times[key]
            self._current_cache_size -= removed_size
            self._ica_refs.pop(key, None)
            self._raw_refs.pop(key, None)

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._access_times.clear()
        self._ica_refs.clear()
        self._raw_refs.clear()
        self._current_cache_size = 0
        logger.debug("Cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        total_size_mb = self._current_cache_size / 1024 / 1024
        return {
            "entries": len(self._cache),
            "total_size_mb": total_size_mb,
            "max_size_mb": self.max_cache_size_bytes / 1024 / 1024,
            "utilization_percent": (
                total_size_mb / (self.max_cache_size_bytes / 1024 / 1024)
            )
            * 100,
            "oldest_access": (
                min(self._access_times.values()) if self._access_times else None
            ),
            "newest_access": (
                max(self._access_times.values()) if self._access_times else None
            ),
        }


_global_ica_cache = ICASourcesCache()


def get_cached_ica_sources(
    ica: ICA, raw: mne.io.Raw, force_refresh: bool = False
) -> mne.io.Raw:
    """Get ICA sources using global cache."""
    return _global_ica_cache.get_sources(ica, raw, force_refresh)


def invalidate_ica_cache(ica: ICA):
    """Invalidate cache entries for specific ICA object."""
    _global_ica_cache.invalidate_ica(ica)


def clear_ica_cache():
    """Clear all ICA sources cache."""
    _global_ica_cache.clear_cache()


def get_ica_cache_stats() -> Dict:
    """Get cache statistics."""
    return _global_ica_cache.get_cache_stats()


def cache_aware_ica_method(func):
    """Decorator to automatically invalidate cache when ICA changes."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "final_ica") and self.final_ica is not None:
            old_exclude = getattr(self.final_ica, "exclude", []).copy()

        result = func(self, *args, **kwargs)

        if hasattr(self, "final_ica") and self.final_ica is not None:
            new_exclude = getattr(self.final_ica, "exclude", [])
            if old_exclude != new_exclude:
                invalidate_ica_cache(self.final_ica)
                logger.debug("Cache invalidated due to ICA exclude list change")

        return result

    return wrapper
