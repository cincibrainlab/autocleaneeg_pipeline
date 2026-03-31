"""Lazy API exports for serve workspace administration."""

from __future__ import annotations

from importlib import import_module

__all__ = ["create_app"]


def __getattr__(name: str):
    if name != "create_app":
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module("autoclean.api.server")
    value = module.create_app
    globals()[name] = value
    return value
