"""Log view widget for displaying scrolling activity logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static


class LogEntry(Static):
    """A single log entry."""

    def __init__(
        self,
        timestamp: datetime,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.timestamp = timestamp
        self.event_type = event_type
        self.message = message
        self.details = details or {}

    def render(self) -> str:
        # Time formatting
        time_str = self.timestamp.strftime("%H:%M:%S")

        # Event type styling
        type_styles = {
            "ready": ("green", "READY"),
            "dispatch": ("cyan", "DISPATCH"),
            "complete": ("bold green", "DONE"),
            "error": ("red", "ERROR"),
            "warning": ("yellow", "WARN"),
            "info": ("blue", "INFO"),
            "service_start": ("green", "START"),
            "service_stop": ("yellow", "STOP"),
        }

        style, label = type_styles.get(self.event_type, ("white", "LOG"))

        return f"[dim]{time_str}[/] [{style}][{label:^8}][/] {self.message}"


class LogView(Widget):
    """A scrolling log view widget."""

    DEFAULT_CSS = """
    LogView {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }

    LogView VerticalScroll {
        height: 100%;
    }

    LogView .log-entry {
        height: auto;
        padding: 0;
    }

    LogView .log-entry.error {
        color: $error;
    }

    LogView .log-entry.warning {
        color: $warning;
    }

    LogView .log-entry.ready, LogView .log-entry.complete {
        color: $success;
    }

    LogView .log-entry.dispatch {
        color: $primary;
    }
    """

    def __init__(self, max_entries: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_entries = max_entries
        self._entries: list[LogEntry] = []

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log-scroll")

    def add_entry(
        self,
        event_type: str,
        message: str,
        timestamp: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a new log entry."""
        if timestamp is None:
            timestamp = datetime.now()

        entry = LogEntry(
            timestamp=timestamp,
            event_type=event_type,
            message=message,
            details=details,
            classes="log-entry",
        )
        entry.add_class(event_type)

        self._entries.insert(0, entry)

        # Trim if exceeding max
        if len(self._entries) > self.max_entries:
            removed = self._entries.pop()
            try:
                removed.remove()
            except Exception:
                pass

        # Mount at top of scroll
        try:
            scroll = self.query_one("#log-scroll", VerticalScroll)
            scroll.mount(entry, before=0)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all log entries."""
        try:
            scroll = self.query_one("#log-scroll", VerticalScroll)
            for child in list(scroll.children):
                child.remove()
            self._entries.clear()
        except Exception:
            pass

    def set_entries(self, entries: list[dict[str, Any]]) -> None:
        """Set entries from a list of dictionaries."""
        self.clear()

        for entry_data in entries:
            self.add_entry(
                event_type=entry_data.get("event_type", "info"),
                message=entry_data.get("message", ""),
                timestamp=entry_data.get("timestamp"),
                details=entry_data.get("details"),
            )

    @property
    def entry_count(self) -> int:
        """Return the number of entries."""
        return len(self._entries)
