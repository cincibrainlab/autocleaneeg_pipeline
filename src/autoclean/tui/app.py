"""Compatibility wrapper for the AutoClean Serve v2 TUI."""

from autoclean.tui.v2_app import (
    ActivityEvent,
    AppState,
    AutoCleanTUI,
    ServiceSettings,
    StatusBar,
    main,
    run_tui,
)

__all__ = [
    "ActivityEvent",
    "AppState",
    "AutoCleanTUI",
    "ServiceSettings",
    "StatusBar",
    "main",
    "run_tui",
]
