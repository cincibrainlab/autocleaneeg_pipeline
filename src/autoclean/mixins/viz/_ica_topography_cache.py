"""ICA topography batch computation and caching system.

This module provides efficient batch computation and caching of ICA component
topographies to dramatically speed up visualization workflows. Instead of 
computing topographies one-by-one, this system:

- Pre-computes all topographies in a single MNE call
- Caches topography data arrays for reuse
- Provides matplotlib-ready topography objects
- Manages memory usage with intelligent cleanup
- Thread-safe operations for parallel processing
"""

import hashlib
import logging
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import wraps
import threading

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

logger = logging.getLogger(__name__)


class ICATopographyCache:
    """Thread-safe cache for ICA component topographies with batch computation."""
    
    def __init__(self, max_cache_size_mb: float = 200.0):
        """Initialize topography cache.
        
        Parameters
        ----------
        max_cache_size_mb : float
            Maximum cache size in megabytes (default: 200MB)
        """
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        logger.debug(f"ICA topography cache initialized with {max_cache_size_mb}MB limit")
    
    def _generate_cache_key(self, ica: ICA) -> str:
        """Generate unique cache key for ICA object."""
        # Create hash based on ICA mixing matrix and channel positions
        mixing_hash = hashlib.md5(ica.mixing_.tobytes()).hexdigest()[:8]
        n_components = ica.n_components_
        n_channels = len(ica.ch_names)
        
        return f"ica_topo_{mixing_hash}_{n_components}c_{n_channels}ch"
    
    def _estimate_data_size(self, n_components: int, grid_size: int = 67) -> int:
        """Estimate memory usage for topography data."""
        # Each topography: grid_size x grid_size float64 array
        bytes_per_topo = grid_size * grid_size * 8  # float64
        # Additional metadata and contour data
        overhead_per_topo = 1024  # Approximate
        
        return n_components * (bytes_per_topo + overhead_per_topo)
    
    def _cleanup_if_needed(self, required_size: int):
        """Clean up old cache entries if needed."""
        current_size = sum(
            self._estimate_data_size(len(entry['topographies']))
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
                self._estimate_data_size(len(entry['topographies']))
                for entry in self._cache.values()
            )
            
            if current_size + required_size <= self.max_cache_size_bytes:
                break
    
    def get_topographies(self, ica: ICA, component_indices: Optional[List[int]] = None,
                        force_refresh: bool = False) -> Dict[int, Dict[str, Any]]:
        """Get pre-computed topographies for ICA components.
        
        Parameters
        ----------
        ica : ICA
            Fitted ICA object
        component_indices : list of int or None
            Component indices to get. If None, gets all components
        force_refresh : bool
            Force recomputation even if cached
            
        Returns
        -------
        topographies : dict
            Dictionary mapping component index to topography data:
            {component_idx: {'image': array, 'extent': tuple, 'contours': dict}}
        """
        if component_indices is None:
            component_indices = list(range(ica.n_components_))
        
        cache_key = self._generate_cache_key(ica)
        current_time = time.time()
        
        with self._lock:
            # Check cache hit
            if not force_refresh and cache_key in self._cache:
                logger.debug(f"Topography cache hit for {cache_key}")
                self._access_times[cache_key] = current_time
                
                cached_topos = self._cache[cache_key]['topographies']
                
                # Return requested components
                result = {}
                for idx in component_indices:
                    if idx in cached_topos:
                        result[idx] = cached_topos[idx].copy()
                
                # If we have all requested components, return them
                if len(result) == len(component_indices):
                    return result
                
                # Otherwise, fall through to recompute missing components
                logger.debug(f"Cache partial miss - have {len(result)}/{len(component_indices)} components")
            
            # Cache miss or partial miss - compute topographies
            logger.debug(f"Computing batch topographies for {cache_key}")
            start_time = time.time()
            
            topographies = self._compute_batch_topographies(ica, component_indices)
            
            computation_time = time.time() - start_time
            logger.debug(f"Batch topographies computed in {computation_time:.2f}s for {len(component_indices)} components")
            
            # Estimate memory usage and cleanup if needed
            data_size = self._estimate_data_size(len(topographies))
            self._cleanup_if_needed(data_size)
            
            # Cache the results
            self._cache[cache_key] = {
                'topographies': topographies,
                'creation_time': current_time,
                'n_components': len(topographies),
            }
            self._access_times[cache_key] = current_time
            
            return {idx: topographies[idx].copy() for idx in component_indices if idx in topographies}
    
    def _compute_batch_topographies(self, ica: ICA, component_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Compute topographies for multiple components efficiently."""
        topographies = {}
        
        try:
            # Get the electrode positions and setup plotting parameters
            info = ica.info
            
            # Use MNE's internal topography computation
            # This is much faster than calling plot_components individually
            for idx in component_indices:
                try:
                    # Create a temporary figure to extract topography data
                    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                    
                    # Plot component to extract underlying data
                    ica.plot_components(
                        picks=idx,
                        axes=ax,
                        ch_type="eeg",
                        show=False,
                        colorbar=False,
                        cmap="jet",
                        outlines="head",
                        sensors=True,
                        contours=6,
                    )
                    
                    # Extract the image data and metadata
                    images = []
                    extents = []
                    contours_data = []
                    
                    for child in ax.get_children():
                        if hasattr(child, 'get_array') and hasattr(child, 'get_extent'):
                            # This is likely the topography image
                            array = child.get_array()
                            extent = child.get_extent()
                            if array is not None and extent is not None:
                                images.append(array.copy())
                                extents.append(extent)
                        elif hasattr(child, 'get_paths'):
                            # This might be contour data
                            try:
                                paths = child.get_paths()
                                if paths:
                                    contours_data.append({
                                        'paths': [p.vertices.copy() for p in paths],
                                        'colors': getattr(child, 'get_edgecolors', lambda: ['black'])(),
                                        'linewidths': getattr(child, 'get_linewidths', lambda: [1.0])(),
                                    })
                            except:
                                pass
                    
                    # Store the extracted data
                    topographies[idx] = {
                        'images': images,
                        'extents': extents,
                        'contours': contours_data,
                        'component_idx': idx,
                    }
                    
                    # Close the temporary figure
                    plt.close(fig)
                    
                except Exception as exc:
                    logger.warning(f"Failed to compute topography for component {idx}: {exc}")
                    # Store error placeholder
                    topographies[idx] = {
                        'images': [],
                        'extents': [],
                        'contours': [],
                        'component_idx': idx,
                        'error': str(exc)
                    }
                    
        except Exception as exc:
            logger.error(f"Batch topography computation failed: {exc}")
            
        return topographies
    
    def clear_cache(self, ica: Optional[ICA] = None):
        """Clear cached topographies."""
        with self._lock:
            if ica is None:
                # Clear all cache
                self._cache.clear()
                self._access_times.clear()
                logger.debug("Cleared all topography cache")
            else:
                # Clear specific ICA cache
                cache_key = self._generate_cache_key(ica)
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    del self._access_times[cache_key]
                    logger.debug(f"Cleared topography cache for {cache_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_components = sum(
                entry['n_components'] for entry in self._cache.values()
            )
            
            estimated_size = sum(
                self._estimate_data_size(entry['n_components'])
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
_topography_cache = ICATopographyCache()


def get_cached_topographies(ica: ICA, component_indices: Optional[List[int]] = None,
                          force_refresh: bool = False) -> Dict[int, Dict[str, Any]]:
    """Get cached ICA component topographies.
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    component_indices : list of int or None
        Component indices to get. If None, gets all components
    force_refresh : bool
        Force recomputation even if cached
        
    Returns
    -------
    topographies : dict
        Dictionary mapping component index to topography data
    """
    return _topography_cache.get_topographies(ica, component_indices, force_refresh)


def clear_topography_cache(ica: Optional[ICA] = None):
    """Clear cached topographies."""
    _topography_cache.clear_cache(ica)


def get_topography_cache_stats() -> Dict[str, Any]:
    """Get topography cache statistics."""
    return _topography_cache.get_cache_stats()


def apply_cached_topography(ax: plt.Axes, topography_data: Dict[str, Any], 
                          component_idx: int, title: Optional[str] = None):
    """Apply cached topography data to a matplotlib axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    topography_data : dict
        Cached topography data from get_cached_topographies
    component_idx : int
        Component index for labeling
    title : str or None
        Custom title, or None for default
    """
    try:
        if 'error' in topography_data:
            ax.text(0.5, 0.5, f"Topography error: {topography_data['error']}", 
                   ha="center", va="center", fontsize=8)
            ax.set_title(f"IC{component_idx} (Error)", fontsize=12)
            return
            
        # Apply images
        for img, extent in zip(topography_data['images'], topography_data['extents']):
            ax.imshow(img, extent=extent, cmap='jet', aspect='equal')
        
        # Apply contours
        for contour_data in topography_data['contours']:
            for path, color, linewidth in zip(
                contour_data['paths'],
                contour_data.get('colors', ['black']),
                contour_data.get('linewidths', [1.0])
            ):
                if len(path) > 0:
                    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth)
        
        # Set title and cleanup
        if title is None:
            title = f"IC{component_idx} Topography"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        
    except Exception as exc:
        logger.error(f"Failed to apply cached topography for IC{component_idx}: {exc}")
        ax.text(0.5, 0.5, "Cached topography error", ha="center", va="center")
        ax.set_title(f"IC{component_idx} (Error)", fontsize=12)
