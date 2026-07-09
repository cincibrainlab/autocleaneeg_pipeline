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
from textual.widgets import (
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
from autoclean.tui.screens.queue import QueueScreen
from autoclean.tui.screens.routes import RoutesScreen
from autoclean.tui.screens.service import ServiceScreen
from autoclean.utils.ingestion import ServeConfigError


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
class ServiceSettings:
    """User-configurable service execution settings."""

    max_cycles: int = 1000
    idle_limit: int = 10
    sleep_seconds: float = 1.0
    max_events: int = 1
    dry_run: bool = False
    use_watchfiles: bool = True
    require_sentinel: bool = True


@dataclass
class AppState:
    """Application state container."""

    workspace_dir: Optional[Path] = None
    mode: str = "test"
    service_running: bool = False
    service_process: Optional[subprocess.Popen] = None
    service_stop_requested: bool = False
    service_settings: ServiceSettings = field(default_factory=ServiceSettings)
    service_log_path: Optional[Path] = None

    # Queue statistics
    pending_count: int = 0
    ready_count: int = 0
    completed_count: int = 0
    running_count: int = 0
    failed_count: int = 0

    # Activity log
    activity_log: list[ActivityEvent] = field(default_factory=list)

    # Config state
    config_valid: bool = False
    config_errors: list[str] = field(default_factory=list)
    config_warnings: list[str] = field(default_factory=list)
    service_started_at: Optional[datetime] = None
    service_last_command: list[str] = field(default_factory=list)
    service_last_config_source: str = ""
    service_last_returncode: Optional[int] = None
    last_completed_file: Optional[str] = None
    last_failed_file: Optional[str] = None
    last_failed_error: Optional[str] = None


class NavSidebar(Container):
    """Navigation sidebar widget."""

    def compose(self) -> ComposeResult:
        yield ListView(
            ListItem(Label("Dashboard"), id="nav-dashboard"),
            ListItem(Label("Routes"), id="nav-routes"),
            ListItem(Label("Work Board"), id="nav-queue"),
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
        mode_display = (
            "[bold cyan]Draft[/]" if self.mode == "test" else "[bold cyan]Production[/]"
        )
        if self.service_running:
            status = "[bold green]Running[/]"
        else:
            status = "[dim]Stopped[/]"
        return f"Lane: {mode_display}  |  Service: {status}"


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

    def get_queue_path(self) -> Optional[Path]:
        """Get the active mode-specific queue path."""
        if not self.state.workspace_dir:
            return None
        return self.state.workspace_dir / f"queue-{self.state.mode}.json"

    def get_config_path(self, deployed: bool = False) -> Optional[Path]:
        """Get the active mode-specific config path."""
        if not self.state.workspace_dir:
            return None
        if deployed:
            return self.state.workspace_dir / "deploy" / f"serve-{self.state.mode}.yaml"
        return self.state.workspace_dir / f"serve-{self.state.mode}.yaml"

    def get_mode_label(self, mode: Optional[str] = None) -> str:
        """Return the operator-facing lane label."""
        selected_mode = mode or self.state.mode
        return "Draft" if selected_mode == "test" else "Production"

    def get_service_config_source(self) -> tuple[str, Optional[Path]]:
        """Describe which config source the current lane will use."""
        deployed_config = self.get_config_path(deployed=True)
        operator_config = self.get_config_path(deployed=False)
        if deployed_config is not None and deployed_config.exists():
            return ("deployed", deployed_config)
        if operator_config is not None and operator_config.exists():
            return ("operator", operator_config)
        return ("missing", operator_config or deployed_config)

    def configure_service(self, params: dict[str, Any]) -> None:
        """Store service settings for the next launch."""
        self.state.service_settings = ServiceSettings(**params)

    def build_service_command(self, cli_path: Path) -> list[str]:
        """Build the serve run command using current app state."""
        if not self.state.workspace_dir:
            raise ValueError("No workspace configured")

        settings = self.state.service_settings
        config_source, config_path = self.get_service_config_source()
        use_operator_config = config_source != "deployed"
        if config_source == "missing" or config_path is None:
            raise FileNotFoundError("No serve configuration available")

        cmd = [
            str(cli_path),
            "serve",
            "run",
            "--mode",
            self.state.mode,
            "--path",
            str(self.state.workspace_dir),
            "--max-cycles",
            str(settings.max_cycles),
            "--idle-limit",
            str(settings.idle_limit),
            "--sleep-seconds",
            str(settings.sleep_seconds),
            "--max-events",
            str(settings.max_events),
        ]

        queue_path = self.get_queue_path()
        if queue_path is not None:
            cmd.extend(["--queue-path", str(queue_path)])
        if settings.dry_run:
            cmd.append("--dry-run")
        if not settings.use_watchfiles:
            cmd.append("--no-watch")
        if not settings.require_sentinel:
            cmd.append("--no-sentinel")
        if use_operator_config:
            cmd.append("--use-operator")

        return cmd

    def _load_config(self) -> None:
        """Load and validate serve configuration."""
        if not self.state.workspace_dir:
            return

        config_file = self.get_config_path(deployed=False)
        if config_file is None:
            return
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
        queue_path = self.get_queue_path()
        if queue_path is None:
            return

        if not queue_path.exists():
            self.state.pending_count = 0
            self.state.ready_count = 0
            self.state.completed_count = 0
            self.state.running_count = 0
            self.state.failed_count = 0
            self.state.last_completed_file = None
            self.state.last_failed_file = None
            self.state.last_failed_error = None
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            entries = queue.entries()

            pending = 0
            processing = 0
            failed = 0
            processed = 0
            latest_processed_at = ""
            latest_failed_at = ""

            self.state.last_completed_file = None
            self.state.last_failed_file = None
            self.state.last_failed_error = None
            for path_str, entry_data in entries.items():
                status = entry_data.get("status", "pending")
                if status == "pending":
                    pending += 1
                elif status == "processing":
                    processing += 1
                elif status == "failed":
                    failed += 1
                    failed_at = str(
                        entry_data.get("failed_at") or entry_data.get("added_at") or ""
                    )
                    if failed_at >= latest_failed_at:
                        latest_failed_at = failed_at
                        self.state.last_failed_file = Path(path_str).name
                        self.state.last_failed_error = entry_data.get("last_error")
                elif status == "processed":
                    processed += 1
                    processed_at = str(
                        entry_data.get("processed_at")
                        or entry_data.get("added_at")
                        or ""
                    )
                    if processed_at >= latest_processed_at:
                        latest_processed_at = processed_at
                        self.state.last_completed_file = Path(path_str).name

            self.state.pending_count = pending
            self.state.completed_count = processed
            self.state.failed_count = failed
            self.state.ready_count = 0
            self.state.running_count = processing
        except Exception:
            pass

    @work(exclusive=True, thread=True)
    def _start_file_watcher(self) -> None:
        """Start watching workspace files for changes."""
        if not self.state.workspace_dir:
            return

        try:
            from watchfiles import watch

            queue_paths = {
                self.state.workspace_dir / "queue-test.json",
                self.state.workspace_dir / "queue-live.json",
            }
            config_paths = {
                self.state.workspace_dir / "serve-test.yaml",
                self.state.workspace_dir / "serve-live.yaml",
            }
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
                    changed_path = Path(path)
                    if changed_path in queue_paths:
                        self.call_from_thread(self._load_queue)
                        self.call_from_thread(self._refresh_current_screen)
                    elif changed_path in config_paths:
                        self.call_from_thread(self._load_config)
                        self.call_from_thread(self._refresh_current_screen)
        except ImportError:
            pass
        except Exception:
            pass

    def _refresh_current_screen(self) -> None:
        """Refresh the current screen with updated data."""
        try:
            screen = self.screen
        except Exception:
            return
        if screen and hasattr(screen, "refresh_data"):
            screen.refresh_data()

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
        self.notify(f"Switched to {self.get_mode_label()} lane")

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
            "F1: Help | D: Dashboard | R: Routes | U: Work Board | "
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

            # Try to find the runtime CLI
            runtime_dir = self.state.workspace_dir / "runtimes" / self.state.mode
            try:
                cli_path = resolve_runtime_cli(runtime_dir)
            except FileNotFoundError:
                cli_path = Path(sys.executable).parent / "autocleaneeg-pipeline"

            cmd = self.build_service_command(cli_path)
            config_source, _ = self.get_service_config_source()
            log_path = self.state.workspace_dir / f"serve-{self.state.mode}.log"
            self.state.service_log_path = log_path
            self.state.service_stop_requested = False
            self.state.service_started_at = datetime.now()
            self.state.service_last_command = list(cmd)
            self.state.service_last_config_source = config_source
            self.state.service_last_returncode = None

            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n[{datetime.now().isoformat()}] Starting service: {' '.join(cmd)}\n"
                )
                log_handle.flush()
                self.state.service_process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.state.service_running = True
                self.call_from_thread(self._update_status_bar)
                self.call_from_thread(
                    self.notify, "Service started", severity="information"
                )
                self.call_from_thread(
                    self._add_activity_event,
                    "service_start",
                    f"Service started ({log_path.name})",
                )

                # Monitor process
                returncode = self.state.service_process.wait()

            stop_requested = self.state.service_stop_requested
            self.state.service_stop_requested = False
            self.state.service_running = False
            self.state.service_started_at = None
            self.state.service_process = None
            self.state.service_last_returncode = returncode
            self.call_from_thread(self._update_status_bar)
            self.call_from_thread(self._refresh_current_screen)
            if stop_requested or returncode == 0:
                self.call_from_thread(
                    self.notify, "Service stopped", severity="information"
                )
                self.call_from_thread(
                    self._add_activity_event, "service_stop", "Service stopped"
                )
            else:
                self.call_from_thread(
                    self.notify,
                    f"Service exited with code {returncode}",
                    severity="error",
                )
                self.call_from_thread(
                    self._add_activity_event,
                    "error",
                    f"Service exited with code {returncode}",
                )
        except Exception as exc:
            self.state.service_running = False
            self.state.service_stop_requested = False
            self.state.service_started_at = None
            self.state.service_process = None
            self.call_from_thread(self._update_status_bar)
            self.call_from_thread(
                self.notify, f"Failed to start service: {exc}", severity="error"
            )

    def _stop_service(self) -> None:
        """Stop the running service."""
        if self.state.service_process:
            self.state.service_stop_requested = True
            self.state.service_process.terminate()
            self._add_activity_event("service_stop", "Service stop requested by user")
            self.notify("Stopping service...", severity="information")

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
        self._refresh_current_screen()

    def get_routes(self) -> list[Any]:
        """Get parsed routes from configuration."""
        config_file = self.get_config_path(deployed=False)
        if config_file is None:
            return []
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

    def get_route_specs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """Get route registry specs for operator-friendly management."""
        workspace_dir = self.state.workspace_dir
        if workspace_dir is None:
            return []

        try:
            from autoclean.utils.serve_routes import load_route_specs

            routes = load_route_specs(workspace_dir)
            if include_archived:
                return routes
            return [route for route in routes if not route.get("archived", False)]
        except Exception:
            return []

    def get_route_spec(self, route_id: str) -> Optional[dict[str, Any]]:
        """Get one route spec from the route registry."""
        for route in self.get_route_specs(include_archived=True):
            if route.get("id") == route_id:
                return route
        return None

    def sync_route_registry(self) -> bool:
        """Recompile route registry into serve YAML files."""
        workspace_dir = self.state.workspace_dir
        if workspace_dir is None:
            return False

        try:
            from autoclean.utils.serve_routes import sync_route_registry

            sync_route_registry(workspace_dir)
            self._load_config()
            return True
        except Exception:
            return False

    def upsert_route_spec(
        self,
        *,
        route_id: str,
        existing_route_id: Optional[str] = None,
        taskfile: str,
        montage: str,
        ingestion_folders: list[str],
        file_globs: list[str],
        mode_scope: str,
        enabled: bool,
        recursive: bool,
    ) -> tuple[bool, str]:
        """Create or update one route spec from TUI form data."""
        workspace_dir = self.state.workspace_dir
        if workspace_dir is None:
            return False, "No serve workspace configured"

        route_id = route_id.strip()
        taskfile = taskfile.strip()
        montage = montage.strip()
        folders = [item.strip() for item in ingestion_folders if item.strip()]
        globs = [item.strip() for item in file_globs if item.strip()]

        if existing_route_id is not None and route_id != existing_route_id:
            return False, "Route ID is locked during edit. Create a new route instead."
        if not route_id:
            return False, "Route ID is required"
        if not taskfile:
            return False, "Task file is required"
        if not montage:
            return False, "Montage is required"
        if not folders:
            return False, "At least one ingestion folder is required"

        try:
            from autoclean.utils.serve_routes import (
                sync_route_registry,
                upsert_route_spec,
            )

            modes = ["test", "live"] if mode_scope == "both" else ["test"]
            updates: dict[str, Any] = {
                "modes": modes,
                "taskfile": str(Path(taskfile).expanduser().resolve()),
                "montage": montage,
                "ingestion_folders": [
                    str(Path(item).expanduser().resolve()) for item in folders
                ],
                "enabled": enabled,
                "recursive": recursive,
            }
            if globs:
                updates["file_globs"] = globs

            upsert_route_spec(workspace_dir, route_id, updates)
            sync_route_registry(workspace_dir)
            self._load_config()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def preview_route_spec(
        self,
        *,
        taskfile: str,
        montage: str,
        ingestion_folders: list[str],
        file_globs: list[str],
        mode_scope: str,
        recursive: bool,
    ) -> dict[str, Any]:
        """Resolve and preview route form data without saving."""
        folders = [item.strip() for item in ingestion_folders if item.strip()]
        globs = [item.strip() for item in file_globs if item.strip()]
        preview: dict[str, Any] = {
            "taskfile": taskfile.strip(),
            "montage": montage.strip(),
            "folders": [],
            "mode_scope": (
                "Draft + Production" if mode_scope == "both" else "Draft only"
            ),
            "matches": [],
            "warnings": [],
        }
        if taskfile.strip():
            preview["taskfile"] = str(Path(taskfile.strip()).expanduser().resolve())
            if not Path(preview["taskfile"]).exists():
                preview["warnings"].append("Task file does not exist yet.")
        else:
            preview["warnings"].append("Task file is required.")
        for folder in folders:
            resolved = Path(folder).expanduser().resolve()
            preview["folders"].append(str(resolved))
            if not resolved.exists():
                preview["warnings"].append(f"Folder missing: {resolved}")
                continue
            patterns = globs or ["*"]
            for pattern in patterns:
                iterator = (
                    resolved.rglob(pattern) if recursive else resolved.glob(pattern)
                )
                for match in iterator:
                    if match.is_file():
                        preview["matches"].append(str(match))
                    if len(preview["matches"]) >= 5:
                        break
                if len(preview["matches"]) >= 5:
                    break
            if len(preview["matches"]) >= 5:
                break
        if not folders:
            preview["warnings"].append("At least one ingestion folder is required.")
        if not montage.strip():
            preview["warnings"].append("Montage is required.")
        if not preview["matches"]:
            preview["warnings"].append(
                "No matching files found in the selected folders yet."
            )
        return preview

    def set_route_enabled(self, route_id: str, enabled: bool) -> bool:
        """Toggle a route in the route registry and recompile configs."""
        workspace_dir = self.state.workspace_dir
        if workspace_dir is None:
            return False

        try:
            from autoclean.utils.serve_routes import (
                sync_route_registry,
                upsert_route_spec,
            )

            upsert_route_spec(workspace_dir, route_id, {"enabled": enabled})
            sync_route_registry(workspace_dir)
            self._load_config()
            return True
        except Exception:
            return False

    def set_route_archived(self, route_id: str, archived: bool) -> bool:
        """Archive or restore a route and recompile configs."""
        workspace_dir = self.state.workspace_dir
        if workspace_dir is None:
            return False

        try:
            from autoclean.utils.serve_routes import (
                archive_route_spec,
                sync_route_registry,
                unarchive_route_spec,
            )

            if archived:
                archive_route_spec(workspace_dir, route_id)
            else:
                unarchive_route_spec(workspace_dir, route_id)
            sync_route_registry(workspace_dir)
            self._load_config()
            return True
        except Exception:
            return False

    def promote_route(self, route_id: str) -> bool:
        """Promote a draft route into production and recompile configs."""
        workspace_dir = self.state.workspace_dir
        if workspace_dir is None:
            return False

        try:
            from autoclean.utils.serve_routes import (
                promote_route_spec,
                sync_route_registry,
            )

            promote_route_spec(workspace_dir, route_id)
            sync_route_registry(workspace_dir)
            self._load_config()
            return True
        except Exception:
            return False

    def get_queue_entries(self) -> dict[str, Any]:
        """Get all queue entries."""
        queue_path = self.get_queue_path()
        if queue_path is None:
            return {}

        if not queue_path.exists():
            return {}

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            return queue.entries()
        except Exception:
            return {}

    def get_service_runtime_snapshot(self) -> dict[str, Any]:
        """Return operator-facing runtime details for the service screen."""
        config_source, config_path = self.get_service_config_source()
        queue_path = self.get_queue_path()
        uptime = None
        if self.state.service_started_at is not None and self.state.service_running:
            uptime_seconds = int(
                (datetime.now() - self.state.service_started_at).total_seconds()
            )
            minutes, seconds = divmod(uptime_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return {
            "lane": self.get_mode_label(),
            "workspace": (
                str(self.state.workspace_dir)
                if self.state.workspace_dir
                else "Not configured"
            ),
            "queue_path": str(queue_path) if queue_path else "Unavailable",
            "config_source": self.state.service_last_config_source or config_source,
            "config_path": str(config_path) if config_path else "Unavailable",
            "log_path": (
                str(self.state.service_log_path)
                if self.state.service_log_path
                else "Unavailable"
            ),
            "pid": (
                self.state.service_process.pid if self.state.service_process else None
            ),
            "uptime": uptime,
            "command": (
                " ".join(self.state.service_last_command)
                if self.state.service_last_command
                else "Not started yet"
            ),
            "completed": self.state.last_completed_file,
            "failed": self.state.last_failed_file,
            "failed_error": self.state.last_failed_error,
        }

    def read_service_log_tail(self, line_count: int = 12) -> str:
        """Read the tail of the current service log file."""
        log_path = self.state.service_log_path
        if log_path is None or not log_path.exists():
            return ""
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-line_count:])
        except Exception:
            return ""

    def get_config_yaml(self) -> str:
        """Get the raw YAML configuration content."""
        config_file = self.get_config_path(deployed=False)
        if config_file is None:
            return ""
        if not config_file.exists():
            return f"# Config file not found: {config_file}"

        try:
            return config_file.read_text(encoding="utf-8")
        except Exception as exc:
            return f"# Error reading config: {exc}"

    def deploy_current_config(self) -> tuple[bool, str]:
        """Validate and deploy the active config using the CLI contract."""
        if not self.state.workspace_dir:
            return False, "No workspace configured"

        source = self.get_config_path(deployed=False)
        target = self.get_config_path(deployed=True)
        if source is None or target is None:
            return False, "No workspace configured"
        if not source.exists():
            return False, f"Config file not found: {source}"

        try:
            from autoclean.utils.ingestion import (
                ServeConfigError,
                load_serve_config,
                parse_serve_config,
            )

            raw_config = load_serve_config(source)
            parse_serve_config(raw_config, self.state.workspace_dir, strict=True)
        except ServeConfigError as exc:
            return False, (
                "Cannot deploy invalid configuration: " + "; ".join(exc.errors)
            )
        except Exception as exc:
            return False, f"Deploy failed: {exc}"

        try:
            import shutil

            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.chmod(0o644)
            shutil.copy2(source, target)
            target.chmod(0o444)
        except Exception as exc:
            return False, f"Deploy failed: {exc}"

        self._load_config()
        return True, f"Configuration deployed to {target.name}"


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
