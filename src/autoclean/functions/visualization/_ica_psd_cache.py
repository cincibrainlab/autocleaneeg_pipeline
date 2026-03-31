"""ICA component PSD batch computation and caching system."""

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch

from autoclean.functions.visualization._ica_cache_utils import (
    get_ica_mixing_matrix_hash,
)
from autoclean.functions.visualization._ica_sources_cache import get_cached_ica_sources

logger = logging.getLogger(__name__)


class ICAPSDCache:
    """Thread-safe cache for ICA component PSDs with batch computation."""

    def __init__(self, max_cache_size_mb: float = 100.0):
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        logger.debug("ICA PSD cache initialized with %sMB limit", max_cache_size_mb)

    def _generate_cache_key(
        self, ica: ICA, raw: mne.io.Raw, psd_params: Dict[str, Any]
    ) -> str:
        ica_hash = get_ica_mixing_matrix_hash(ica)[:8]
        raw_data = raw.get_data()
        raw_hash = hashlib.md5(
            f"{raw_data.shape}_{raw_data[0, 0]:.6f}_{raw_data[-1, -1]:.6f}".encode()
        ).hexdigest()[:8]
        param_str = "_".join(f"{k}={v}" for k, v in sorted(psd_params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"ica_psd_{ica_hash}_{raw_hash}_{param_hash}"

    def _estimate_data_size(self, n_components: int, n_freqs: int) -> int:
        bytes_per_component = n_freqs * 8
        freq_bytes = n_freqs * 8
        overhead = 1024
        return n_components * bytes_per_component + freq_bytes + overhead

    def _cleanup_if_needed(self, required_size: int):
        current_size = sum(
            self._estimate_data_size(
                entry["psd_data"].shape[0], entry["psd_data"].shape[1]
            )
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
                self._estimate_data_size(
                    entry["psd_data"].shape[0], entry["psd_data"].shape[1]
                )
                for entry in self._cache.values()
            )
            if current_size + required_size <= self.max_cache_size_bytes:
                break

    def get_component_psds(
        self,
        ica: ICA,
        raw: mne.io.Raw,
        component_indices: Optional[List[int]] = None,
        fmin: float = 1.0,
        fmax: Optional[float] = None,
        n_fft: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if component_indices is None:
            component_indices = list(range(ica.n_components_))

        sfreq = raw.info["sfreq"]
        if fmax is None:
            fmax = sfreq / 2.0
        if n_fft is None:
            n_fft = min(2048, raw.n_times)

        psd_params = {"fmin": fmin, "fmax": fmax, "n_fft": n_fft, "sfreq": sfreq}
        cache_key = self._generate_cache_key(ica, raw, psd_params)
        current_time = time.time()

        with self._lock:
            if not force_refresh and cache_key in self._cache:
                logger.debug("PSD cache hit for %s", cache_key)
                self._access_times[cache_key] = current_time
                cached_data = self._cache[cache_key]
                return cached_data["psd_data"][component_indices], cached_data["freqs"]

            logger.debug("Computing batch PSDs for %s", cache_key)
            start_time = time.time()
            psd_data, freqs = self._compute_batch_psds(ica, raw, psd_params)
            computation_time = time.time() - start_time
            logger.debug(
                "Batch PSDs computed in %.2fs for %d components",
                computation_time,
                ica.n_components_,
            )

            data_size = self._estimate_data_size(psd_data.shape[0], psd_data.shape[1])
            self._cleanup_if_needed(data_size)
            self._cache[cache_key] = {
                "psd_data": psd_data,
                "freqs": freqs,
                "creation_time": current_time,
                "n_components": psd_data.shape[0],
                "params": psd_params.copy(),
            }
            self._access_times[cache_key] = current_time
            return psd_data[component_indices], freqs

    def _compute_batch_psds(
        self, ica: ICA, raw: mne.io.Raw, psd_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            sources = get_cached_ica_sources(ica, raw)
            component_data = sources.get_data()
            psd_data, freqs = psd_array_welch(
                component_data,
                sfreq=psd_params["sfreq"],
                fmin=psd_params["fmin"],
                fmax=psd_params["fmax"],
                n_fft=psd_params["n_fft"],
                n_overlap=psd_params["n_fft"] // 2,
                n_jobs=1,
                verbose=False,
            )
            return psd_data, freqs
        except Exception as exc:
            logger.error("Batch PSD computation failed: %s", exc)
            n_components = ica.n_components_
            n_freqs = max(1, psd_params["n_fft"] // 2 + 1)
            return (
                np.zeros((n_components, n_freqs)),
                np.linspace(psd_params["fmin"], psd_params["fmax"], n_freqs),
            )

    def clear_cache(self, ica: Optional[ICA] = None, raw: Optional[mne.io.Raw] = None):
        """Clear cached PSDs."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.debug("Cleared PSD cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_components = sum(
                entry["n_components"] for entry in self._cache.values()
            )
            estimated_size = sum(
                self._estimate_data_size(
                    entry["psd_data"].shape[0], entry["psd_data"].shape[1]
                )
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


_psd_cache = ICAPSDCache()


def get_cached_component_psds(
    ica: ICA,
    raw: mne.io.Raw,
    component_indices: Optional[List[int]] = None,
    fmin: float = 1.0,
    fmax: Optional[float] = None,
    n_fft: Optional[int] = None,
    force_refresh: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get cached PSDs for ICA components."""
    return _psd_cache.get_component_psds(
        ica, raw, component_indices, fmin, fmax, n_fft, force_refresh
    )


def clear_psd_cache(ica: Optional[ICA] = None, raw: Optional[mne.io.Raw] = None):
    """Clear cached PSDs."""
    _psd_cache.clear_cache(ica, raw)


def get_psd_cache_stats() -> Dict[str, Any]:
    """Get PSD cache statistics."""
    return _psd_cache.get_cache_stats()
