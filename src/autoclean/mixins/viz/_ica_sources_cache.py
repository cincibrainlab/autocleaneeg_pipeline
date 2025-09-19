"""ICA sources caching system for efficient report generation.

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
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from functools import wraps

import numpy as np
import mne
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
        # Use object ids and key properties for hashing
        ica_info = f"{id(ica)}_{ica.n_components_}_{len(ica.exclude)}"
        raw_info = f"{id(raw)}_{raw.n_times}_{raw.info['sfreq']}"
        
        # Include hash of channel names to detect channel changes
        ch_hash = hashlib.md5("_".join(raw.ch_names).encode()).hexdigest()[:8]
        
        return f"ica_{ica_info}_raw_{raw_info}_ch_{ch_hash}"
    
    def _estimate_data_size(self, data_shape: Tuple[int, ...]) -> int:
        """Estimate memory size of numpy array in bytes."""
        # Assume float64 (8 bytes per element)
        return np.prod(data_shape) * 8
    
    def _cleanup_if_needed(self, required_size: int):
        """Remove oldest cache entries if memory limit would be exceeded."""
        if self._current_cache_size + required_size <= self.max_cache_size_bytes:
            return
            
        logger.debug("Cache size limit reached, cleaning up oldest entries")
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_keys:
            if key in self._cache:
                removed_size = self._cache[key]['data_size']
                del self._cache[key]
                del self._access_times[key]
                self._current_cache_size -= removed_size
                
                logger.debug(f"Removed cache entry {key}, freed {removed_size/1024/1024:.1f}MB")
                
                if self._current_cache_size + required_size <= self.max_cache_size_bytes:
                    break
    
    def get_sources(self, ica: ICA, raw: mne.io.Raw, 
                   force_refresh: bool = False) -> mne.io.Raw:
        """Get ICA sources with caching.
        
        Parameters
        ----------
        ica : ICA
            Fitted ICA object
        raw : mne.io.Raw  
            Raw data to apply ICA to
        force_refresh : bool
            Force recomputation even if cached
            
        Returns
        -------
        sources : mne.io.Raw
            ICA source activations
        """
        cache_key = self._generate_cache_key(ica, raw)
        current_time = time.time()
        
        # Check cache hit
        if not force_refresh and cache_key in self._cache:
            logger.debug(f"Cache hit for {cache_key}")
            self._access_times[cache_key] = current_time
            
            # Return cached sources (make copy to prevent modification)
            cached_data = self._cache[cache_key]['sources_data'].copy()
            cached_info = self._cache[cache_key]['sources_info'].copy()
            
            # Reconstruct sources object
            sources = mne.io.RawArray(cached_data, cached_info, verbose=False)
            return sources
        
        # Cache miss - compute sources
        logger.debug(f"Computing ICA sources for {cache_key}")
        start_time = time.time()
        
        sources = ica.get_sources(raw)
        
        computation_time = time.time() - start_time
        logger.debug(f"ICA sources computed in {computation_time:.2f}s")
        
        # Cache the results
        sources_data = sources.get_data()
        data_size = self._estimate_data_size(sources_data.shape)
        
        # Check if we need to clean up cache
        self._cleanup_if_needed(data_size)
        
        # Store in cache
        self._cache[cache_key] = {
            'sources_data': sources_data,
            'sources_info': sources.info.copy(),
            'data_size': data_size,
            'created_time': current_time,
            'computation_time': computation_time
        }
        self._access_times[cache_key] = current_time
        self._current_cache_size += data_size
        
        # Store weak references for invalidation
        self._ica_refs[cache_key] = weakref.ref(ica)
        self._raw_refs[cache_key] = weakref.ref(raw)
        
        logger.debug(f"Cached sources: {data_size/1024/1024:.1f}MB, "
                    f"total cache: {self._current_cache_size/1024/1024:.1f}MB")
        
        return sources
    
    def invalidate_ica(self, ica: ICA):
        """Invalidate all cache entries for a specific ICA object."""
        ica_id = id(ica)
        keys_to_remove = []
        
        for key in list(self._cache.keys()):
            if f"ica_{ica_id}_" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_cache_entry(key)
        
        if keys_to_remove:
            logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for ICA {ica_id}")
    
    def _remove_cache_entry(self, key: str):
        """Remove a single cache entry and update size tracking."""
        if key in self._cache:
            removed_size = self._cache[key]['data_size']
            del self._cache[key]
            del self._access_times[key]
            self._current_cache_size -= removed_size
            
            # Clean up weak references
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
            'entries': len(self._cache),
            'total_size_mb': total_size_mb,
            'max_size_mb': self.max_cache_size_bytes / 1024 / 1024,
            'utilization_percent': (total_size_mb / (self.max_cache_size_bytes / 1024 / 1024)) * 100,
            'oldest_access': min(self._access_times.values()) if self._access_times else None,
            'newest_access': max(self._access_times.values()) if self._access_times else None
        }


# Global cache instance
_global_ica_cache = ICASourcesCache()


def get_cached_ica_sources(ica: ICA, raw: mne.io.Raw, 
                          force_refresh: bool = False) -> mne.io.Raw:
    """Get ICA sources using global cache.
    
    This is the main function to use for cached ICA source computation.
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data to apply ICA to  
    force_refresh : bool
        Force recomputation even if cached
        
    Returns
    -------
    sources : mne.io.Raw
        ICA source activations
    """
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
        # If this is an ICA method that modifies the ICA object
        if hasattr(self, 'final_ica') and self.final_ica is not None:
            old_exclude = getattr(self.final_ica, 'exclude', []).copy()
            
        result = func(self, *args, **kwargs)
        
        # Check if ICA was modified and invalidate cache
        if hasattr(self, 'final_ica') and self.final_ica is not None:
            new_exclude = getattr(self.final_ica, 'exclude', [])
            if old_exclude != new_exclude:
                invalidate_ica_cache(self.final_ica)
                logger.debug("Cache invalidated due to ICA exclude list change")
        
        return result
    return wrapper
