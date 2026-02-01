"""Activity log screen for AutoClean TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Label, Select, Static

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class LogEntry(Static):
    """A single log entry widget."""

    def __init__(
        self,
        timestamp: str,
        event_type: str,
        message: str,
        file_path: str = "",
        route_id: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.timestamp = timestamp
        self.event_type = event_type
        self.message = message
        self.file_path = file_path
        self.route_id = route_id

    def render(self) -> str:
        # Color coding by event type
        type_styles = {
            "ready": ("green", "READY"),
            "dispatch": ("cyan", "DISPATCH"),
            "complete": ("bold green", "COMPLETE"),
            "error": ("red", "ERROR"),
            "service_start": ("blue", "START"),
            "service_stop": ("yellow", "STOP"),
            "info": ("white", "INFO"),
            "warning": ("yellow", "WARN"),
        }

        style, label = type_styles.get(self.event_type, ("white", "INFO"))

        parts = [
            f"[dim]{self.timestamp}[/]",
            f"[{style}][{label:^8}][/]",
            self.message,
        ]

        if self.route_id:
            parts.append(f"[dim]({self.route_id})[/]")

        if self.file_path:
            parts.append(f"[dim]{self.file_path}[/]")

        return " ".join(parts)


class ActivityScreen(Screen):
    """Activity log screen showing event history."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "clear_log", "Clear"),
        ("f", "filter_type", "Filter"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Activity Log", classes="section-header")

            # Filter row
            with Horizontal(classes="filter-row"):
                yield Label("Filter by type:")
                yield Select(
                    [
                        ("All Events", "all"),
                        ("Ready", "ready"),
                        ("Dispatch", "dispatch"),
                        ("Complete", "complete"),
                        ("Errors", "error"),
                        ("Service", "service"),
                    ],
                    value="all",
                    id="type-filter",
                )
                yield Button("Clear Log", id="btn-clear", variant="default")
                yield Button("Refresh", id="btn-refresh", variant="primary")

            # Log entries
            with VerticalScroll(id="log-scroll", classes="log-container"):
                yield Static("No activity logged yet", classes="empty-state", id="empty-log")

            # Status
            yield Static("", id="log-status", classes="help-text")

    def on_mount(self) -> None:
        """Initialize activity log."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh activity log from app state."""
        app: AutoCleanTUI = self.app  # type: ignore

        # Get filter value
        try:
            type_filter = self.query_one("#type-filter", Select).value
        except Exception:
            type_filter = "all"

        log_scroll = self.query_one("#log-scroll", VerticalScroll)

        # Clear existing entries
        for child in list(log_scroll.children):
            child.remove()

        events = app.state.activity_log

        if not events:
            log_scroll.mount(
                Static("No activity logged yet", classes="empty-state", id="empty-log")
            )
            self._update_status(0, 0)
            return

        # Filter events
        if type_filter == "all":
            filtered = events
        elif type_filter == "service":
            filtered = [e for e in events if e.event_type.startswith("service")]
        else:
            filtered = [e for e in events if e.event_type == type_filter]

        # Mount entries
        for event in filtered:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            file_str = str(event.file_path.name) if event.file_path else ""

            entry = LogEntry(
                timestamp=timestamp,
                event_type=event.event_type,
                message=event.message,
                file_path=file_str,
                route_id=event.route_id or "",
                classes="log-entry",
            )
            entry.add_class(event.event_type)
            log_scroll.mount(entry)

        self._update_status(len(filtered), len(events))

    def _update_status(self, shown: int, total: int) -> None:
        """Update the status line."""
        status = self.query_one("#log-status", Static)
        if shown == total:
            status.update(f"Showing {total} event(s)")
        else:
            status.update(f"Showing {shown} of {total} event(s)")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        self.refresh_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-clear":
            self.action_clear_log()
        elif button_id == "btn-refresh":
            self.action_refresh()

    def action_refresh(self) -> None:
        """Refresh the activity log."""
        self.refresh_data()
        self.notify("Activity log refreshed")

    def action_clear_log(self) -> None:
        """Clear the activity log."""
        app: AutoCleanTUI = self.app  # type: ignore
        app.state.activity_log.clear()
        self.refresh_data()
        self.notify("Activity log cleared")

    def action_filter_type(self) -> None:
        """Cycle through filter types."""
        select = self.query_one("#type-filter", Select)
        options = ["all", "ready", "dispatch", "complete", "error", "service"]
        current = select.value
        try:
            idx = options.index(current)
            next_idx = (idx + 1) % len(options)
            select.value = options[next_idx]
        except ValueError:
            select.value = "all"
