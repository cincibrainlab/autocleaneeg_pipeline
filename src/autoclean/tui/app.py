"""Main Textual App for AutoClean Automation Console."""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
)

from autoclean.tui.screens.activity import ActivityScreen
from autoclean.tui.screens.config import ConfigScreen
from autoclean.tui.screens.dashboard import DashboardScreen
from autoclean.utils.ingestion import ServeConfigError
from autoclean.tui.screens.queue import QueueScreen
from autoclean.tui.screens.routes import RoutesScreen
from autoclean.tui.screens.service import ServiceScreen


@dataclass
class ActivityEvent:
    """Represents an activity log event."""

    timestamp: datetime
    event_type: str
    message: str
    file_path: Optional[Path] = None
    route_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    """Application state container."""

    workspace_dir: Optional[Path] = None
    mode: str = "test"
    service_running: bool = False
    service_process: Optional[subprocess.Popen] = None

    # Queue statistics
    pending_count: int = 0
    ready_count: int = 0
    running_count: int = 0
    failed_count: int = 0

    # Activity log
    activity_log: list[ActivityEvent] = field(default_factory=list)

    # Config state
    config_valid: bool = False
    config_errors: list[str] = field(default_factory=list)
    config_warnings: list[str] = field(default_factory=list)


class NavSidebar(Container):
    """Navigation sidebar widget."""

    def compose(self) -> ComposeResult:
        yield ListView(
            ListItem(Label("Dashboard"), id="nav-dashboard"),
            ListItem(Label("Routes"), id="nav-routes"),
            ListItem(Label("Queue"), id="nav-queue"),
            ListItem(Label("Activity"), id="nav-activity"),
            ListItem(Label("Config"), id="nav-config"),
            ListItem(Label("Service"), id="nav-service"),
            id="nav-list",
        )


class StatusBar(Static):
    """Status bar showing mode and service status."""

    mode = reactive("test")
    service_running = reactive(False)

    def render(self) -> str:
        mode_display = f"[bold cyan]{self.mode}[/]"
        if self.service_running:
            status = "[bold green]Running[/]"
        else:
            status = "[dim]Stopped[/]"
        return f"Mode: {mode_display}  |  Service: {status}"


class MainContent(Container):
    """Container for the main content area."""

    pass


class AutoCleanTUI(App):
    """AutoClean Automation Console TUI Application."""

    TITLE = "AutoClean Automation Console"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("d", "show_dashboard", "Dashboard", show=True),
        Binding("r", "show_routes", "Routes", show=True),
        Binding("u", "show_queue", "Queue", show=True),
        Binding("a", "show_activity", "Activity", show=True),
        Binding("c", "show_config", "Config", show=True),
        Binding("e", "show_service", "Service", show=True),
        Binding("s", "start_service", "Start", show=True),
        Binding("p", "stop_service", "Stop", show=True),
        Binding("v", "validate_config", "Validate", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("t", "toggle_mode", "Toggle Mode", show=False),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "routes": RoutesScreen,
        "queue": QueueScreen,
        "activity": ActivityScreen,
        "config": ConfigScreen,
        "service": ServiceScreen,
    }

    # Reactive attributes
    current_view = reactive("dashboard")

    def __init__(
        self,
        workspace_path: Optional[Path] = None,
        mode: str = "test",
        watch_files: bool = True,
    ) -> None:
        super().__init__()
        self.state = AppState(
            workspace_dir=workspace_path,
            mode=mode,
        )
        self._file_watcher_task = None
        self._watcher_stop_event = threading.Event()
        self._watch_files = watch_files

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="sidebar"):
                yield NavSidebar()
                yield StatusBar(id="status-bar")
            with Vertical(id="content-area"):
                yield MainContent(id="main-content")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on app mount."""
        self._update_status_bar()
        self.push_screen("dashboard")
        if self.state.workspace_dir:
            self._load_workspace_data()
            if self._watch_files:
                self._start_file_watcher()

    def _update_status_bar(self) -> None:
        """Update the status bar with current state."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.mode = self.state.mode
        status_bar.service_running = self.state.service_running

    def _load_workspace_data(self) -> None:
        """Load workspace configuration and queue data."""
        if not self.state.workspace_dir:
            return

        self._load_config()
        self._load_queue()

    def _load_config(self) -> None:
        """Load and validate serve configuration."""
        if not self.state.workspace_dir:
            return

        config_file = (
            self.state.workspace_dir / f"serve-{self.state.mode}.yaml"
        )
        if not config_file.exists():
            self.state.config_valid = False
            self.state.config_errors = [f"Config file not found: {config_file}"]
            return

        try:
            from autoclean.utils.ingestion import (
                load_serve_config,
                parse_serve_config,
            )

            raw_config = load_serve_config(config_file)
            _, warnings = parse_serve_config(
                raw_config, self.state.workspace_dir, strict=False
            )
            self.state.config_valid = True
            self.state.config_errors = []
            self.state.config_warnings = list(warnings)
        except ServeConfigError as exc:
            self.state.config_valid = False
            self.state.config_errors = list(exc.errors)
            self.state.config_warnings = list(exc.warnings)
        except Exception as exc:
            self.state.config_valid = False
            self.state.config_errors = [str(exc)]

    def _load_queue(self) -> None:
        """Load queue data and update statistics."""
        if not self.state.workspace_dir:
            return

        queue_path = self.state.workspace_dir / "queue.json"
        if not queue_path.exists():
            self.state.pending_count = 0
            self.state.ready_count = 0
            self.state.running_count = 0
            self.state.failed_count = 0
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            entries = queue.entries()

            pending = 0
            failed = 0
            processed = 0

            for entry_data in entries.values():
                status = entry_data.get("status", "pending")
                if status == "pending":
                    pending += 1
                elif status == "failed":
                    failed += 1
                elif status == "processed":
                    processed += 1

            self.state.pending_count = pending
            self.state.failed_count = failed
            # ready_count and running_count would come from live monitoring
            self.state.ready_count = 0
            self.state.running_count = 1 if self.state.service_running else 0
        except Exception:
            pass

    @work(exclusive=True, thread=True)
    def _start_file_watcher(self) -> None:
        """Start watching workspace files for changes."""
        if not self.state.workspace_dir:
            return

        try:
            from watchfiles import watch

            queue_path = self.state.workspace_dir / "queue.json"
            paths_to_watch = [self.state.workspace_dir]

            # Use stop_event to allow clean shutdown
            for changes in watch(
                *paths_to_watch,
                recursive=False,
                stop_event=self._watcher_stop_event,
            ):
                if self._watcher_stop_event.is_set():
                    break
                for change_type, path in changes:
                    if Path(path) == queue_path:
                        self.call_from_thread(self._load_queue)
                        self.call_from_thread(self._refresh_current_screen)
        except ImportError:
            pass
        except Exception:
            pass

    def _refresh_current_screen(self) -> None:
        """Refresh the current screen with updated data."""
        if self.screen and hasattr(self.screen, "refresh_data"):
            self.screen.refresh_data()

    def action_quit(self) -> None:
        """Quit the application, stopping file watcher first."""
        self._watcher_stop_event.set()
        self.exit()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle navigation selection."""
        item_id = event.item.id
        if item_id:
            view_name = item_id.replace("nav-", "")
            self._navigate_to(view_name)

    def _navigate_to(self, view_name: str) -> None:
        """Navigate to a specific view."""
        if view_name in self.SCREENS:
            self.current_view = view_name
            # Pop all screens back to base, then push new one
            while len(self.screen_stack) > 1:
                self.pop_screen()
            self.push_screen(view_name)

    def action_show_dashboard(self) -> None:
        self._navigate_to("dashboard")

    def action_show_routes(self) -> None:
        self._navigate_to("routes")

    def action_show_queue(self) -> None:
        self._navigate_to("queue")

    def action_show_activity(self) -> None:
        self._navigate_to("activity")

    def action_show_config(self) -> None:
        self._navigate_to("config")

    def action_show_service(self) -> None:
        self._navigate_to("service")

    def action_toggle_mode(self) -> None:
        """Toggle between test and live mode."""
        self.state.mode = "live" if self.state.mode == "test" else "test"
        self._update_status_bar()
        self._load_workspace_data()
        self._refresh_current_screen()
        self.notify(f"Switched to {self.state.mode} mode")

    def action_start_service(self) -> None:
        """Start the ingestion service."""
        if self.state.service_running:
            self.notify("Service is already running", severity="warning")
            return
        self._start_service()

    def action_stop_service(self) -> None:
        """Stop the ingestion service."""
        if not self.state.service_running:
            self.notify("Service is not running", severity="warning")
            return
        self._stop_service()

    def action_validate_config(self) -> None:
        """Validate the current configuration."""
        self._load_config()
        if self.state.config_valid:
            if self.state.config_warnings:
                self.notify(
                    f"Config valid with {len(self.state.config_warnings)} warnings",
                    severity="warning",
                )
            else:
                self.notify("Configuration is valid", severity="information")
        else:
            self.notify(
                f"Configuration has {len(self.state.config_errors)} errors",
                severity="error",
            )

    def action_show_help(self) -> None:
        """Show help screen."""
        self.notify(
            "F1: Help | D: Dashboard | R: Routes | U: Queue | "
            "A: Activity | C: Config | E: Service | Q: Quit"
        )

    @work(exclusive=True, thread=True)
    def _start_service(self) -> None:
        """Start the ingestion service in background."""
        if not self.state.workspace_dir:
            self.call_from_thread(
                self.notify, "No workspace configured", severity="error"
            )
            return

        try:
            from autoclean.utils.ingestion import resolve_runtime_cli

            config_path = (
                self.state.workspace_dir / "deploy" / f"serve-{self.state.mode}.yaml"
            )
            if not config_path.exists():
                config_path = (
                    self.state.workspace_dir / f"serve-{self.state.mode}.yaml"
                )

            # Try to find the runtime CLI
            runtime_dir = self.state.workspace_dir / "runtimes" / self.state.mode
            try:
                cli_path = resolve_runtime_cli(runtime_dir)
            except FileNotFoundError:
                cli_path = Path(sys.executable).parent / "autocleaneeg-pipeline"

            cmd = [
                str(cli_path),
                "serve",
                "run",
                "--mode",
                self.state.mode,
                "--path",
                str(self.state.workspace_dir),
                "--max-cycles",
                "1000",
                "--idle-limit",
                "10",
            ]

            self.state.service_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.state.service_running = True
            self.call_from_thread(self._update_status_bar)
            self.call_from_thread(
                self.notify, "Service started", severity="information"
            )
            self.call_from_thread(self._add_activity_event, "service_start", "Service started")

            # Monitor process
            self.state.service_process.wait()
            self.state.service_running = False
            self.state.service_process = None
            self.call_from_thread(self._update_status_bar)
            self.call_from_thread(
                self.notify, "Service stopped", severity="information"
            )
            self.call_from_thread(self._add_activity_event, "service_stop", "Service stopped")
        except Exception as exc:
            self.state.service_running = False
            self.call_from_thread(
                self.notify, f"Failed to start service: {exc}", severity="error"
            )

    def _stop_service(self) -> None:
        """Stop the running service."""
        if self.state.service_process:
            self.state.service_process.terminate()
            self.state.service_running = False
            self._update_status_bar()
            self._add_activity_event("service_stop", "Service stopped by user")
            self.notify("Service stopped", severity="information")

    def _add_activity_event(
        self,
        event_type: str,
        message: str,
        file_path: Optional[Path] = None,
        route_id: Optional[str] = None,
    ) -> None:
        """Add an event to the activity log."""
        event = ActivityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            message=message,
            file_path=file_path,
            route_id=route_id,
        )
        self.state.activity_log.insert(0, event)
        # Keep only last 100 events
        self.state.activity_log = self.state.activity_log[:100]

    def get_routes(self) -> list[Any]:
        """Get parsed routes from configuration."""
        if not self.state.workspace_dir:
            return []

        config_file = (
            self.state.workspace_dir / f"serve-{self.state.mode}.yaml"
        )
        if not config_file.exists():
            return []

        try:
            from autoclean.utils.ingestion import (
                load_serve_config,
                parse_serve_config,
            )

            raw_config = load_serve_config(config_file)
            config, _ = parse_serve_config(
                raw_config, self.state.workspace_dir, strict=False
            )
            return config.routes
        except Exception:
            return []

    def get_queue_entries(self) -> dict[str, Any]:
        """Get all queue entries."""
        if not self.state.workspace_dir:
            return {}

        queue_path = self.state.workspace_dir / "queue.json"
        if not queue_path.exists():
            return {}

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            return queue.entries()
        except Exception:
            return {}

    def get_config_yaml(self) -> str:
        """Get the raw YAML configuration content."""
        if not self.state.workspace_dir:
            return ""

        config_file = (
            self.state.workspace_dir / f"serve-{self.state.mode}.yaml"
        )
        if not config_file.exists():
            return f"# Config file not found: {config_file}"

        try:
            return config_file.read_text(encoding="utf-8")
        except Exception as exc:
            return f"# Error reading config: {exc}"


def run_tui(workspace_path: Optional[Path] = None, mode: str = "test") -> None:
    """Run the TUI application.

    Args:
        workspace_path: Path to the serve workspace directory.
        mode: Configuration mode ("test" or "live").
    """
    app = AutoCleanTUI(workspace_path=workspace_path, mode=mode)
    app.run()


def main() -> int:
    """Entry point for standalone TUI launcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoClean Automation Console TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autocleaneeg-tui --path /path/to/workspace
  autocleaneeg-tui --path /path/to/workspace --mode live
        """,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to serve workspace directory",
    )
    parser.add_argument(
        "--mode",
        choices=["test", "live"],
        default="test",
        help="Configuration mode (default: test)",
    )

    args = parser.parse_args()

    # Try to get workspace from stored config if not provided
    workspace_path = args.path
    if workspace_path is None:
        try:
            from autoclean.utils.user_config import user_config

            stored = user_config.get_serve_workspace()
            if stored:
                workspace_path = stored
        except Exception:
            pass

    if workspace_path is None:
        print("Error: No workspace specified and no stored workspace found.")
        print("Use --path to specify a workspace directory.")
        return 1

    workspace_path = workspace_path.expanduser().resolve()
    if not workspace_path.exists():
        print(f"Error: Workspace not found: {workspace_path}")
        return 1

    run_tui(workspace_path=workspace_path, mode=args.mode)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
