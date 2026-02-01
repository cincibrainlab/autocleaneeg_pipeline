"""Service control screen for AutoClean TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static, Switch

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class ServiceScreen(Screen):
    """Service control panel for starting/stopping the ingestion service."""

    BINDINGS = [
        ("s", "start_service", "Start"),
        ("p", "stop_service", "Stop"),
        ("t", "toggle_mode", "Toggle Mode"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="service-container"):
            yield Static("Service Control", classes="section-header")

            # Service status
            yield Static("", id="service-status", classes="service-status")

            # Control buttons
            with Horizontal(classes="service-controls"):
                yield Button("Start Service", id="btn-start", variant="success")
                yield Button("Stop Service", id="btn-stop", variant="error")
                yield Button("Toggle Mode", id="btn-toggle-mode", variant="default")

            # Parameters section
            yield Static("Service Parameters", classes="section-header")

            with Vertical(classes="service-params"):
                # Max cycles
                with Horizontal(classes="param-row"):
                    yield Label("Max Cycles:", classes="param-label")
                    yield Input(
                        value="1000",
                        placeholder="Maximum cycles to run",
                        id="input-max-cycles",
                        classes="param-input",
                    )

                # Idle limit
                with Horizontal(classes="param-row"):
                    yield Label("Idle Limit:", classes="param-label")
                    yield Input(
                        value="10",
                        placeholder="Idle cycles before exit",
                        id="input-idle-limit",
                        classes="param-input",
                    )

                # Sleep seconds
                with Horizontal(classes="param-row"):
                    yield Label("Sleep (sec):", classes="param-label")
                    yield Input(
                        value="1.0",
                        placeholder="Seconds between cycles",
                        id="input-sleep",
                        classes="param-input",
                    )

                # Max events
                with Horizontal(classes="param-row"):
                    yield Label("Max Events:", classes="param-label")
                    yield Input(
                        value="1",
                        placeholder="Max watch events per cycle",
                        id="input-max-events",
                        classes="param-input",
                    )

            # Options section
            yield Static("Options", classes="section-header")

            with Vertical(classes="service-params"):
                # Dry run toggle
                with Horizontal(classes="param-row"):
                    yield Label("Dry Run:", classes="param-label")
                    yield Switch(value=False, id="switch-dry-run")
                    yield Static(
                        "Print commands without executing",
                        classes="help-text",
                    )

                # Watch mode toggle
                with Horizontal(classes="param-row"):
                    yield Label("Watch Mode:", classes="param-label")
                    yield Switch(value=True, id="switch-watch")
                    yield Static(
                        "Use watchfiles for file monitoring",
                        classes="help-text",
                    )

                # Require sentinel toggle
                with Horizontal(classes="param-row"):
                    yield Label("Require Sentinel:", classes="param-label")
                    yield Switch(value=True, id="switch-sentinel")
                    yield Static(
                        "Require .ready sentinel files",
                        classes="help-text",
                    )

            # Status/log section
            yield Static("Service Log", classes="section-header")
            yield Static("", id="service-log", classes="help-text")

    def on_mount(self) -> None:
        """Initialize service screen."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh service status."""
        app: AutoCleanTUI = self.app  # type: ignore

        # Update status
        status_widget = self.query_one("#service-status", Static)
        if app.state.service_running:
            status_widget.update(
                f"[bold green]Service Running[/]\n"
                f"Mode: {app.state.mode}  |  "
                f"Workspace: {app.state.workspace_dir or 'Not configured'}"
            )
            status_widget.remove_class("stopped")
            status_widget.add_class("running")
        else:
            status_widget.update(
                f"[bold]Service Stopped[/]\n"
                f"Mode: {app.state.mode}  |  "
                f"Workspace: {app.state.workspace_dir or 'Not configured'}"
            )
            status_widget.remove_class("running")
            status_widget.add_class("stopped")

        # Update button states
        try:
            btn_start = self.query_one("#btn-start", Button)
            btn_stop = self.query_one("#btn-stop", Button)

            btn_start.disabled = app.state.service_running
            btn_stop.disabled = not app.state.service_running
        except Exception:
            pass

        # Update service log
        self._update_service_log()

    def _update_service_log(self) -> None:
        """Update the service log section."""
        app: AutoCleanTUI = self.app  # type: ignore

        # Show recent service-related events
        log_widget = self.query_one("#service-log", Static)

        service_events = [
            e for e in app.state.activity_log
            if e.event_type.startswith("service") or e.event_type in ("error", "complete")
        ][:5]

        if not service_events:
            log_widget.update("[dim]No recent service activity[/]")
            return

        lines = []
        for event in service_events:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            if event.event_type == "service_start":
                lines.append(f"[green]{timestamp}[/] Service started")
            elif event.event_type == "service_stop":
                lines.append(f"[yellow]{timestamp}[/] Service stopped")
            elif event.event_type == "error":
                lines.append(f"[red]{timestamp}[/] {event.message}")
            elif event.event_type == "complete":
                lines.append(f"[green]{timestamp}[/] {event.message}")

        log_widget.update("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-start":
            self.action_start_service()
        elif button_id == "btn-stop":
            self.action_stop_service()
        elif button_id == "btn-toggle-mode":
            self.action_toggle_mode()

    def _get_service_params(self) -> dict:
        """Get current service parameters from inputs."""
        params = {}

        try:
            max_cycles = self.query_one("#input-max-cycles", Input).value
            params["max_cycles"] = int(max_cycles) if max_cycles else 1000
        except (ValueError, Exception):
            params["max_cycles"] = 1000

        try:
            idle_limit = self.query_one("#input-idle-limit", Input).value
            params["idle_limit"] = int(idle_limit) if idle_limit else 10
        except (ValueError, Exception):
            params["idle_limit"] = 10

        try:
            sleep_sec = self.query_one("#input-sleep", Input).value
            params["sleep_seconds"] = float(sleep_sec) if sleep_sec else 1.0
        except (ValueError, Exception):
            params["sleep_seconds"] = 1.0

        try:
            max_events = self.query_one("#input-max-events", Input).value
            params["max_events"] = int(max_events) if max_events else 1
        except (ValueError, Exception):
            params["max_events"] = 1

        try:
            params["dry_run"] = self.query_one("#switch-dry-run", Switch).value
        except Exception:
            params["dry_run"] = False

        try:
            params["use_watchfiles"] = self.query_one("#switch-watch", Switch).value
        except Exception:
            params["use_watchfiles"] = True

        try:
            params["require_sentinel"] = self.query_one("#switch-sentinel", Switch).value
        except Exception:
            params["require_sentinel"] = True

        return params

    def action_start_service(self) -> None:
        """Start the ingestion service."""
        app: AutoCleanTUI = self.app  # type: ignore

        if app.state.service_running:
            self.notify("Service is already running", severity="warning")
            return

        if not app.state.workspace_dir:
            self.notify("No workspace configured", severity="error")
            return

        if not app.state.config_valid:
            self.notify("Configuration is invalid", severity="error")
            return

        params = self._get_service_params()

        if params["dry_run"]:
            self.notify("Dry run mode: commands will be logged but not executed")

        app._start_service()
        self.refresh_data()

    def action_stop_service(self) -> None:
        """Stop the ingestion service."""
        app: AutoCleanTUI = self.app  # type: ignore

        if not app.state.service_running:
            self.notify("Service is not running", severity="warning")
            return

        app._stop_service()
        self.refresh_data()

    def action_toggle_mode(self) -> None:
        """Toggle between test and live mode."""
        app: AutoCleanTUI = self.app  # type: ignore

        if app.state.service_running:
            self.notify(
                "Cannot change mode while service is running",
                severity="warning",
            )
            return

        app.action_toggle_mode()
        self.refresh_data()
