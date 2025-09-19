"""ICA component PSD batch computation and caching system.

This module provides efficient batch computation and caching of Power Spectral 
Density (PSD) for ICA components. Instead of computing PSDs one-by-one using
Welch's method, this system:

- Computes PSDs for all components in vectorized operations
- Caches PSD arrays and frequency vectors for reuse
- Provides thread-safe access with memory management
- Handles different raw data objects (cropped vs full duration)
"""

import hashlib
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from mne.time_frequency import psd_array_welch
from mne.preprocessing import ICA
import mne

logger = logging.getLogger(__name__)


class ICAPSDCache:
    """Thread-safe cache for ICA component PSDs with batch computation."""
    
    def __init__(self, max_cache_size_mb: float = 100.0):
        """Initialize PSD cache.
        
        Parameters
        ----------
        max_cache_size_mb : float
            Maximum cache size in megabytes (default: 100MB)
        """
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        logger.debug(f"ICA PSD cache initialized with {max_cache_size_mb}MB limit")
    
    def _generate_cache_key(self, ica: ICA, raw: mne.io.Raw, 
                          psd_params: Dict[str, Any]) -> str:
        """Generate unique cache key for ICA + Raw + PSD parameters."""
        # ICA hash
        ica_hash = hashlib.md5(ica.mixing_.tobytes()).hexdigest()[:8]
        
        # Raw data hash (using shape and first/last samples for speed)
        raw_data = raw.get_data()
        raw_hash = hashlib.md5(
            f"{raw_data.shape}_{raw_data[0, 0]:.6f}_{raw_data[-1, -1]:.6f}".encode()
        ).hexdigest()[:8]
        
        # PSD parameters hash
        param_str = "_".join(f"{k}={v}" for k, v in sorted(psd_params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"ica_psd_{ica_hash}_{raw_hash}_{param_hash}"
    
    def _estimate_data_size(self, n_components: int, n_freqs: int) -> int:
        """Estimate memory usage for PSD data."""
        # Each component: n_freqs float64 values
        bytes_per_component = n_freqs * 8  # float64
        # Frequency array (shared)
        freq_bytes = n_freqs * 8
        # Metadata overhead
        overhead = 1024
        
        return n_components * bytes_per_component + freq_bytes + overhead
    
    def _cleanup_if_needed(self, required_size: int):
        """Clean up old cache entries if needed."""
        current_size = sum(
            self._estimate_data_size(
                entry['psd_data'].shape[0],
                entry['psd_data'].shape[1]
            )
            for entry in self._cache.values()
        )
        
        if current_size + required_size <= self.max_cache_size_bytes:
            return
        
        # Sort by access time (LRU eviction)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._access_times.get(k, 0)
        )
        
        for key in sorted_keys:
            del self._cache[key]
            del self._access_times[key]
            
            current_size = sum(
                self._estimate_data_size(
                    entry['psd_data'].shape[0],
                    entry['psd_data'].shape[1]
                )
                for entry in self._cache.values()
            )
            
            if current_size + required_size <= self.max_cache_size_bytes:
                break
    
    def get_component_psds(self, ica: ICA, raw: mne.io.Raw,
                          component_indices: Optional[List[int]] = None,
                          fmin: float = 1.0, fmax: Optional[float] = None,
                          n_fft: Optional[int] = None,
                          force_refresh: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached PSDs for ICA components.
        
        Parameters
        ----------
        ica : ICA
            Fitted ICA object
        raw : mne.io.Raw
            Raw data to compute PSDs from
        component_indices : list of int or None
            Component indices to get. If None, gets all components
        fmin : float
            Minimum frequency for PSD computation
        fmax : float or None
            Maximum frequency for PSD computation (default: Nyquist)
        n_fft : int or None
            FFT length for Welch's method
        force_refresh : bool
            Force recomputation even if cached
            
        Returns
        -------
        psd_data : np.ndarray, shape (n_components, n_freqs)
            PSD values for requested components
        freqs : np.ndarray, shape (n_freqs,)
            Frequency values
        """
        if component_indices is None:
            component_indices = list(range(ica.n_components_))
        
        # Set default parameters
        sfreq = raw.info['sfreq']
        if fmax is None:
            fmax = sfreq / 2.0
        if n_fft is None:
            n_fft = min(2048, raw.n_times)
        
        psd_params = {
            'fmin': fmin,
            'fmax': fmax,
            'n_fft': n_fft,
            'sfreq': sfreq
        }
        
        cache_key = self._generate_cache_key(ica, raw, psd_params)
        current_time = time.time()
        
        with self._lock:
            # Check cache hit
            if not force_refresh and cache_key in self._cache:
                logger.debug(f"PSD cache hit for {cache_key}")
                self._access_times[cache_key] = current_time
                
                cached_data = self._cache[cache_key]
                all_psd_data = cached_data['psd_data']
                freqs = cached_data['freqs']
                
                # Return requested components
                requested_psd = all_psd_data[component_indices]
                return requested_psd, freqs
            
            # Cache miss - compute PSDs
            logger.debug(f"Computing batch PSDs for {cache_key}")
            start_time = time.time()
            
            psd_data, freqs = self._compute_batch_psds(
                ica, raw, psd_params
            )
            
            computation_time = time.time() - start_time
            logger.debug(f"Batch PSDs computed in {computation_time:.2f}s for {ica.n_components_} components")
            
            # Estimate memory usage and cleanup if needed
            data_size = self._estimate_data_size(psd_data.shape[0], psd_data.shape[1])
            self._cleanup_if_needed(data_size)
            
            # Cache the results
            self._cache[cache_key] = {
                'psd_data': psd_data,
                'freqs': freqs,
                'creation_time': current_time,
                'n_components': psd_data.shape[0],
                'params': psd_params.copy(),
            }
            self._access_times[cache_key] = current_time
            
            # Return requested components
            requested_psd = psd_data[component_indices]
            return requested_psd, freqs
    
    def _compute_batch_psds(self, ica: ICA, raw: mne.io.Raw, 
                           psd_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PSDs for all components efficiently using vectorized operations."""
        try:
            # Get ICA sources efficiently (should use our existing cache)
            from autoclean.mixins.viz._ica_sources_cache import get_cached_ica_sources
            
            try:
                sources = get_cached_ica_sources(ica, raw)
            except ImportError:
                # Fallback if cache not available
                sources = ica.get_sources(raw)
            
            # Get all component data at once
            component_data = sources.get_data()  # Shape: (n_components, n_times)
            
            # Vectorized PSD computation using Welch's method
            # This is much faster than computing each component individually
            psd_data, freqs = psd_array_welch(
                component_data,
                sfreq=psd_params['sfreq'],
                fmin=psd_params['fmin'],
                fmax=psd_params['fmax'],
                n_fft=psd_params['n_fft'],
                n_overlap=psd_params['n_fft'] // 2,
                n_jobs=1,  # Keep single-threaded for cache consistency
                verbose=False
            )
            
            return psd_data, freqs
            
        except Exception as exc:
            logger.error(f"Batch PSD computation failed: {exc}")
            # Return empty arrays as fallback
            n_components = ica.n_components_
            n_freqs = max(1, psd_params['n_fft'] // 2 + 1)
            
            return (
                np.zeros((n_components, n_freqs)),
                np.linspace(psd_params['fmin'], psd_params['fmax'], n_freqs)
            )
    
    def clear_cache(self, ica: Optional[ICA] = None, raw: Optional[mne.io.Raw] = None):
        """Clear cached PSDs."""
        with self._lock:
            if ica is None and raw is None:
                # Clear all cache
                self._cache.clear()
                self._access_times.clear()
                logger.debug("Cleared all PSD cache")
            else:
                # Clear specific entries (partial clearing not implemented for simplicity)
                # In practice, cache keys are specific enough that this is rarely needed
                self._cache.clear()
                self._access_times.clear()
                logger.debug("Cleared PSD cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_components = sum(
                entry['n_components'] for entry in self._cache.values()
            )
            
            estimated_size = sum(
                self._estimate_data_size(
                    entry['psd_data'].shape[0],
                    entry['psd_data'].shape[1]
                )
                for entry in self._cache.values()
            )
            
            return {
                'entries': len(self._cache),
                'total_components': total_components,
                'size_mb': estimated_size / (1024 * 1024),
                'max_size_mb': self.max_cache_size_bytes / (1024 * 1024),
                'utilization_percent': (estimated_size / self.max_cache_size_bytes) * 100,
            }


# Global cache instance
_psd_cache = ICAPSDCache()


def get_cached_component_psds(ica: ICA, raw: mne.io.Raw,
                             component_indices: Optional[List[int]] = None,
                             fmin: float = 1.0, fmax: Optional[float] = None,
                             n_fft: Optional[int] = None,
                             force_refresh: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get cached PSDs for ICA components.
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data to compute PSDs from
    component_indices : list of int or None
        Component indices to get. If None, gets all components
    fmin : float
        Minimum frequency for PSD computation
    fmax : float or None
        Maximum frequency for PSD computation
    n_fft : int or None
        FFT length for Welch's method
    force_refresh : bool
        Force recomputation even if cached
        
    Returns
    -------
    psd_data : np.ndarray
        PSD values for requested components
    freqs : np.ndarray
        Frequency values
    """
    return _psd_cache.get_component_psds(
        ica, raw, component_indices, fmin, fmax, n_fft, force_refresh
    )


def clear_psd_cache(ica: Optional[ICA] = None, raw: Optional[mne.io.Raw] = None):
    """Clear cached PSDs."""
    _psd_cache.clear_cache(ica, raw)


def get_psd_cache_stats() -> Dict[str, Any]:
    """Get PSD cache statistics."""
    return _psd_cache.get_cache_stats()
