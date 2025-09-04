"""Utility functions for status icons with ASCII fallbacks.

This module provides small helper functions that return either emoji
icons or ASCII fallbacks.  Users can force ASCII mode by setting the
``AUTOCLEAN_ASCII`` or ``APP_ASCII`` environment variable to ``1``.
If neither variable is set, an automatic check is performed to see if
Unicode symbols can be encoded on the current stdout stream.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

# Map icon keys to (emoji, ascii) pairs
ICONS: Dict[str, Tuple[str, str]] = {
    "ok": ("✅", "OK"),
    "warn": ("⚠️", "!"),
    "err": ("❌", "X"),
    "info": ("ℹ️", "i"),
    "work": ("🔧", "+"),
    "arrow": ("→", "->"),
}


def _supports_unicode() -> bool:
    """Return True if the current stdout encoding supports Unicode."""
    try:
        test_icon = ICONS["ok"][0]
        test_icon.encode(sys.stdout.encoding or "utf-8")
        return True
    except Exception:
        return False


ASCII_MODE: bool = (
    os.getenv("AUTOCLEAN_ASCII") == "1"
    or os.getenv("APP_ASCII") == "1"
    or not _supports_unicode()
)


def pick_icon(key: str) -> str:
    """Return the icon for ``key`` using ASCII fallback when needed."""
    emoji, ascii = ICONS.get(key, ("", ""))
    return ascii if ASCII_MODE else emoji
