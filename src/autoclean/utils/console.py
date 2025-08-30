from __future__ import annotations

"""
Themed Console utilities for AutoClean EEG.

Provides a single get_console() factory and theme selection that respects:
- NO_COLOR / FORCE_COLOR
- AUTOCLEAN_THEME (auto|dark|light|hc|mono)
- AUTOCLEAN_COLOR_DEPTH (auto|8|256|truecolor)

Use semantic styles in markup: brand, title, subtitle, header, accent,
info, success, warning, error, muted, dim, border
"""

import os
from typing import Optional

from rich.console import Console
from rich.style import Style
from rich.theme import Theme


AUTOCLEAN_THEME = os.getenv("AUTOCLEAN_THEME", "auto")
AUTOCLEAN_COLOR_DEPTH = os.getenv("AUTOCLEAN_COLOR_DEPTH", "auto")
NO_COLOR = os.getenv("NO_COLOR") is not None
FORCE_COLOR = os.getenv("FORCE_COLOR") is not None


def _choose_color_system():
    if NO_COLOR:
        return None
    if AUTOCLEAN_COLOR_DEPTH == "8":
        return "standard"
    if AUTOCLEAN_COLOR_DEPTH == "256":
        return "256"
    if AUTOCLEAN_COLOR_DEPTH == "truecolor":
        return "truecolor"
    return "auto"


def _is_probably_dark_terminal() -> bool:
    if os.getenv("TERM_PROGRAM") in {"iTerm.app", "Apple_Terminal"}:
        return True
    return True


def _resolve_mode(explicit: Optional[str] = None) -> str:
    mode = (explicit or AUTOCLEAN_THEME or "auto").lower()
    if mode == "auto":
        return "dark" if _is_probably_dark_terminal() else "light"
    if mode in ("hc", "high-contrast"):
        return "hc"
    if mode in ("mono", "monochrome"):
        return "mono"
    if mode in ("dark", "light"):
        return mode
    return "dark"


def _theme_for(mode: str) -> Theme:
    if mode == "mono":
        return Theme(
            {
                "brand": "bold",
                "title": "bold underline",
                "subtitle": "italic",
                "header": "bold",
                "accent": "bold",
                "info": "bold",
                "success": "bold",
                "warning": "bold",
                "error": "bold",
                "muted": "dim",
                "dim": "dim",
                "border": "bold",
            }
        )

    if mode == "hc":
        return Theme(
            {
                "brand": Style(color="#000000", bgcolor="#FFD54F", bold=True),
                "title": Style(color="#000000", bgcolor="#FFFFFF", bold=True),
                "subtitle": Style(color="#000000", bgcolor="#EEEEEE"),
                "header": Style(color="#000000", bgcolor="#DDDDDD", bold=True),
                "accent": "bold",
                "info": "bright_white",
                "success": "bright_green",
                "warning": "bright_yellow bold",
                "error": "bright_red bold",
                "muted": "grey93 on #333333",
                "dim": "dim",
                "border": "bright_white",
            }
        )

    if mode == "light":
        return Theme(
            {
                "brand": "#005f99 bold",
                "title": "#003554 bold",
                "subtitle": "#334155 italic",
                "header": "#0f172a bold",
                "accent": "#124559",
                "info": "#1e40af",
                "success": "#166534",
                "warning": "#b45309",
                "error": "#b91c1c",
                "muted": "#475569",
                "dim": "dim",
                "border": "#0ea5e9",
            }
        )

    return Theme(
        {
            "brand": "#00b8d9 bold",
            "title": "bold",
            "subtitle": "italic dim",
            "header": "bold #e2e8f0",
            "accent": "#80cbc4",
            "info": "#7dd3fc",
            "success": "bold #22c55e",
            "warning": "bold #f59e0b",
            "error": "bold #ef4444",
            "muted": "#94a3b8",
            "dim": "dim",
            "border": "#38bdf8",
        }
    )


GLOBAL_CONSOLE: Optional[Console] = None


def make_console(theme_mode: Optional[str] = None) -> Console:
    if NO_COLOR and not FORCE_COLOR:
        return Console(
            color_system=None, no_color=True, highlight=False, soft_wrap=True
        )
    mode = _resolve_mode(theme_mode)
    return Console(
        theme=_theme_for(mode),
        color_system=_choose_color_system(),
        highlight=False,
        soft_wrap=True,
        force_terminal=True if FORCE_COLOR else None,
    )


def get_console(args: Optional[object] = None) -> Console:
    global GLOBAL_CONSOLE
    if GLOBAL_CONSOLE is None:
        explicit_mode = getattr(args, "theme", None) if args else None
        GLOBAL_CONSOLE = make_console(explicit_mode)
    return GLOBAL_CONSOLE
