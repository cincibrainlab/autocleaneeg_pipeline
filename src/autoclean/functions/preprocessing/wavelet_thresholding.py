"""Compatibility shim that forwards to the relocated wavelet plugin module."""

from __future__ import annotations

from autoclean.mixins.signal_processing.wavelet_threshold import processing as _processing

__all__ = getattr(_processing, "__all__", [])


def __getattr__(name: str):
    return getattr(_processing, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_processing)))
