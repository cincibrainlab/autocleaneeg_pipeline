"""Dashboard screen for AutoClean TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class StatBox(Static):
    """A single statistic display box."""

    value = reactive(0)
    label_text = reactive("")
    box_class = reactive("")

    def __init__(
        self,
        value: int = 0,
        label: str = "",
        box_class: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.label_text = label
        self.box_class = box_class

    def compose(self) -> ComposeResult:
        yield Static(str(self.value), classes="stat-value")
        yield Static(self.label_text, classes="stat-label")

    def watch_value(self, value: int) -> None:
        try:
            stat_value = self.query_one(".stat-value", Static)
            stat_value.update(str(value))
        except Exception:
            pass

    def on_mount(self) -> None:
        if self.box_class:
            self.add_class(self.box_class)


class ActivityItem(Static):
    """A single activity log entry."""

    def __init__(self, timestamp: str, event_type: str, message: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.timestamp = timestamp
        self.event_type = event_type
        self.message = message

    def render(self) -> str:
        type_colors = {
            "ready": "[green]",
            "dispatch": "[cyan]",
            "complete": "[bold green]",
            "error": "[red]",
            "service_start": "[blue]",
            "service_stop": "[yellow]",
            "info": "",
        }
        color = type_colors.get(self.event_type, "")
        end_color = "[/]" if color else ""
        return f"[dim]{self.timestamp}[/] {color}{self.message}{end_color}"


class DashboardScreen(Screen):
    """Dashboard home screen showing overview statistics and recent activity."""

    BINDINGS = [
        ("s", "start_service", "Start"),
        ("p", "stop_service", "Stop"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Lane Overview", classes="section-header")

            with Horizontal(classes="stats-container"):
                yield StatBox(
                    0, "Waiting", "pending", id="stat-pending", classes="stat-box"
                )
                yield StatBox(
                    0, "Running", "running", id="stat-running", classes="stat-box"
                )
                yield StatBox(
                    0, "Completed", "ready", id="stat-completed", classes="stat-box"
                )
                yield StatBox(
                    0, "Needs attention", "failed", id="stat-failed", classes="stat-box"
                )

            with Horizontal(classes="service-controls"):
                yield Button("Start Service", id="btn-start", variant="success")
                yield Button("Stop Service", id="btn-stop", variant="error")
                yield Button("Validate Config", id="btn-validate", variant="primary")

            yield Static("Recent Activity", classes="section-header")

            with Vertical(id="activity-feed", classes="activity-container"):
                yield Static(
                    "No recent activity", classes="empty-state", id="empty-activity"
                )

    def on_mount(self) -> None:
        self.set_interval(2.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore

        try:
            pending_box = self.query_one("#stat-pending", StatBox)
            pending_box.value = app.state.pending_count

            running_box = self.query_one("#stat-running", StatBox)
            running_box.value = app.state.running_count

            completed_box = self.query_one("#stat-completed", StatBox)
            completed_box.value = app.state.completed_count

            failed_box = self.query_one("#stat-failed", StatBox)
            failed_box.value = app.state.failed_count
        except Exception:
            pass

        self._update_activity_feed()

    def _update_activity_feed(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        feed = self.query_one("#activity-feed", Vertical)

        for child in list(feed.children):
            child.remove()

        if not app.state.activity_log:
            feed.mount(Static("No recent activity", classes="empty-state"))
            return

        for event in app.state.activity_log[:10]:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            item = ActivityItem(timestamp, event.event_type, event.message)
            item.add_class(event.event_type)
            feed.mount(item)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "btn-start":
            self.action_start_service()
        elif button_id == "btn-stop":
            self.action_stop_service()
        elif button_id == "btn-validate":
            self.action_validate_config()

    def action_start_service(self) -> None:
        self.app.action_start_service()

    def action_stop_service(self) -> None:
        self.app.action_stop_service()

    def action_validate_config(self) -> None:
        self.app.action_validate_config()
        self.refresh_data()

    def action_refresh(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_workspace_data()
        self.refresh_data()
        self.notify("Dashboard refreshed")
