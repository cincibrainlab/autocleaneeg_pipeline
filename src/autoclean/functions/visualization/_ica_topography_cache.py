"""ICA topography batch computation and caching system."""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from mne.preprocessing import ICA

from autoclean.functions.visualization._ica_cache_utils import (
    get_ica_mixing_matrix_hash,
)

logger = logging.getLogger(__name__)


class ICATopographyCache:
    """Thread-safe cache for ICA component topographies with batch computation."""

    def __init__(self, max_cache_size_mb: float = 200.0):
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        logger.debug(
            "ICA topography cache initialized with %sMB limit",
            max_cache_size_mb,
        )

    def _generate_cache_key(self, ica: ICA) -> str:
        mixing_hash = get_ica_mixing_matrix_hash(ica)[:8]
        return f"ica_topo_{mixing_hash}_{ica.n_components_}c_{len(ica.ch_names)}ch"

    def _estimate_data_size(self, n_components: int, grid_size: int = 67) -> int:
        bytes_per_topo = grid_size * grid_size * 8
        overhead_per_topo = 1024
        return n_components * (bytes_per_topo + overhead_per_topo)

    def _cleanup_if_needed(self, required_size: int):
        current_size = sum(
            self._estimate_data_size(len(entry["topographies"]))
            for entry in self._cache.values()
        )
        if current_size + required_size <= self.max_cache_size_bytes:
            return

        sorted_keys = sorted(
            self._cache.keys(), key=lambda key: self._access_times.get(key, 0)
        )
        for key in sorted_keys:
            del self._cache[key]
            del self._access_times[key]
            current_size = sum(
                self._estimate_data_size(len(entry["topographies"]))
                for entry in self._cache.values()
            )
            if current_size + required_size <= self.max_cache_size_bytes:
                break

    def get_topographies(
        self,
        ica: ICA,
        component_indices: Optional[List[int]] = None,
        force_refresh: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        if component_indices is None:
            component_indices = list(range(ica.n_components_))

        cache_key = self._generate_cache_key(ica)
        current_time = time.time()

        with self._lock:
            if not force_refresh and cache_key in self._cache:
                logger.debug("Topography cache hit for %s", cache_key)
                self._access_times[cache_key] = current_time
                cached_topos = self._cache[cache_key]["topographies"]
                result = {
                    idx: cached_topos[idx].copy()
                    for idx in component_indices
                    if idx in cached_topos
                }
                if len(result) == len(component_indices):
                    return result
                logger.debug(
                    "Cache partial miss - have %d/%d components",
                    len(result),
                    len(component_indices),
                )

            logger.debug("Computing batch topographies for %s", cache_key)
            start_time = time.time()
            topographies = self._compute_batch_topographies(ica, component_indices)
            computation_time = time.time() - start_time
            logger.debug(
                "Batch topographies computed in %.2fs for %d components",
                computation_time,
                len(component_indices),
            )

            data_size = self._estimate_data_size(len(topographies))
            self._cleanup_if_needed(data_size)
            self._cache[cache_key] = {
                "topographies": topographies,
                "creation_time": current_time,
                "n_components": len(topographies),
            }
            self._access_times[cache_key] = current_time
            return {
                idx: topographies[idx].copy()
                for idx in component_indices
                if idx in topographies
            }

    def _compute_batch_topographies(
        self, ica: ICA, component_indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        topographies = {}
        try:
            for idx in component_indices:
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
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

                    images = []
                    extents = []
                    contours_data = []
                    for child in ax.get_children():
                        if hasattr(child, "get_array") and hasattr(child, "get_extent"):
                            array = child.get_array()
                            extent = child.get_extent()
                            if array is not None and extent is not None:
                                images.append(array.copy())
                                extents.append(extent)
                        elif hasattr(child, "get_paths"):
                            try:
                                paths = child.get_paths()
                                if paths:
                                    contours_data.append(
                                        {
                                            "paths": [p.vertices.copy() for p in paths],
                                            "colors": getattr(
                                                child,
                                                "get_edgecolors",
                                                lambda: ["black"],
                                            )(),
                                            "linewidths": getattr(
                                                child, "get_linewidths", lambda: [1.0]
                                            )(),
                                        }
                                    )
                            except Exception:
                                pass

                    topographies[idx] = {
                        "images": images,
                        "extents": extents,
                        "contours": contours_data,
                        "xlim": tuple(float(v) for v in ax.get_xlim()),
                        "ylim": tuple(float(v) for v in ax.get_ylim()),
                        "component_idx": idx,
                    }
                    plt.close(fig)
                except Exception as exc:
                    logger.warning(
                        "Failed to compute topography for component %s: %s",
                        idx,
                        exc,
                    )
                    topographies[idx] = {
                        "images": [],
                        "extents": [],
                        "contours": [],
                        "xlim": None,
                        "ylim": None,
                        "component_idx": idx,
                        "error": str(exc),
                    }
        except Exception as exc:
            logger.error("Batch topography computation failed: %s", exc)

        return topographies

    def clear_cache(self, ica: Optional[ICA] = None):
        """Clear cached topographies."""
        with self._lock:
            if ica is None:
                self._cache.clear()
                self._access_times.clear()
                logger.debug("Cleared all topography cache")
            else:
                cache_key = self._generate_cache_key(ica)
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    del self._access_times[cache_key]
                    logger.debug("Cleared topography cache for %s", cache_key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_components = sum(
                entry["n_components"] for entry in self._cache.values()
            )
            estimated_size = sum(
                self._estimate_data_size(entry["n_components"])
                for entry in self._cache.values()
            )
            return {
                "entries": len(self._cache),
                "total_components": total_components,
                "size_mb": estimated_size / (1024 * 1024),
                "max_size_mb": self.max_cache_size_bytes / (1024 * 1024),
                "utilization_percent": (estimated_size / self.max_cache_size_bytes)
                * 100,
            }


_topography_cache = ICATopographyCache()


def get_cached_topographies(
    ica: ICA, component_indices: Optional[List[int]] = None, force_refresh: bool = False
) -> Dict[int, Dict[str, Any]]:
    """Get cached ICA component topographies."""
    return _topography_cache.get_topographies(ica, component_indices, force_refresh)


def clear_topography_cache(ica: Optional[ICA] = None):
    """Clear cached topographies."""
    _topography_cache.clear_cache(ica)


def get_topography_cache_stats() -> Dict[str, Any]:
    """Get topography cache statistics."""
    return _topography_cache.get_cache_stats()


def apply_cached_topography(
    ax: plt.Axes,
    topography_data: Dict[str, Any],
    component_idx: int,
    title: Optional[str] = None,
):
    """Apply cached topography data to a matplotlib axes."""
    try:
        if "error" in topography_data:
            ax.text(
                0.5,
                0.5,
                f"Topography error: {topography_data['error']}",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_title(f"IC{component_idx} (Error)", fontsize=12)
            return

        for img, extent in zip(topography_data["images"], topography_data["extents"]):
            ax.imshow(img, extent=extent, cmap="jet", aspect="equal")

        for contour_data in topography_data["contours"]:
            for path, color, linewidth in zip(
                contour_data["paths"],
                contour_data.get("colors", ["black"]),
                contour_data.get("linewidths", [1.0]),
            ):
                if len(path) > 0:
                    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth)

        xlim = topography_data.get("xlim")
        ylim = topography_data.get("ylim")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_aspect("equal")

        if title is None:
            title = f"IC{component_idx} Topography"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    except Exception as exc:
        logger.error(
            "Failed to apply cached topography for IC%s: %s",
            component_idx,
            exc,
        )
        ax.text(0.5, 0.5, "Cached topography error", ha="center", va="center")
        ax.set_title(f"IC{component_idx} (Error)", fontsize=12)
