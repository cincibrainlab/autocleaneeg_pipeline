"""Operator-first Textual TUI for AutoClean Serve."""

from __future__ import annotations

import argparse
import os
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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
    TabPane,
    TabbedContent,
)

from autoclean.utils.ingestion import ServeConfigError
from autoclean.tui.v2_state import ServeWorkspaceSnapshot, build_workspace_snapshot


@dataclass
class ActivityEvent:
    timestamp: datetime
    event_type: str
    message: str
    file_path: Optional[Path] = None
    route_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceSettings:
    max_cycles: int = 1000
    idle_limit: int = 10
    sleep_seconds: float = 1.0
    max_events: int = 1
    dry_run: bool = False
    use_watchfiles: bool = True
    require_sentinel: bool = True


@dataclass
class AppState:
    workspace_dir: Optional[Path] = None
    mode: str = "test"
    service_running: bool = False
    service_process: Optional[subprocess.Popen] = None
    service_stop_requested: bool = False
    service_settings: ServiceSettings = field(default_factory=ServiceSettings)
    service_log_path: Optional[Path] = None
    pending_count: int = 0
    ready_count: int = 0
    completed_count: int = 0
    running_count: int = 0
    failed_count: int = 0
    activity_log: list[ActivityEvent] = field(default_factory=list)
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


class StatusBar(Static):
    mode = reactive("test")
    service_running = reactive(False)
    config_source = reactive("missing")
    queue_summary = reactive("Queue: 0")
    last_refresh = reactive("Never")
    last_action = reactive("")

    def render(self) -> str:
        lane = "[bold cyan]Draft[/]" if self.mode == "test" else "[bold magenta]Production[/]"
        service = "[bold green]Running[/]" if self.service_running else "[bold yellow]Stopped[/]"
        source = f"Config: [bold]{self.config_source}[/]"
        action = f" | [green]{self.last_action}[/]" if self.last_action else ""
        return (
            f"Lane: {lane} | Service: {service} | {source} | {self.queue_summary} | "
            f"Updated: {self.last_refresh}{action}"
        )


class AutoCleanTUI(App):
    TITLE = "AutoClean Serve Console"
    CSS_PATH = "styles_v2.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("1", "show_home", "Home", show=True),
        Binding("2", "show_routes", "Routes", show=True),
        Binding("3", "show_queue", "Queue", show=True),
        Binding("4", "show_publish", "Publish", show=True),
        Binding("5", "show_service", "Service", show=True),
        Binding("l", "toggle_mode", "Toggle Lane", show=True),
        Binding("r", "refresh_snapshot", "Refresh", show=True),
        Binding("s", "toggle_service", "Start/Stop", show=True),
    ]

    def __init__(
        self,
        workspace_path: Optional[Path] = None,
        mode: str = "test",
        watch_files: bool = True,
    ) -> None:
        super().__init__()
        self.state = AppState(workspace_dir=workspace_path, mode=mode)
        self._watch_files = watch_files
        self._watcher_stop_event = threading.Event()
        self._snapshot: Optional[ServeWorkspaceSnapshot] = None
        self._route_row_ids: list[str] = []
        self._queue_row_paths: list[str] = []
        self._selected_route_id: Optional[str] = None
        self._selected_queue_path: Optional[str] = None
        self._last_action_message = ""
        self._route_editor_existing_id: Optional[str] = None
        self._route_editor_mode = "create"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar(id="status-bar")
        with TabbedContent(id="main-tabs", initial="tab-home"):
            with TabPane("Home", id="tab-home"):
                with Horizontal(classes="top-cards"):
                    yield Static("", id="home-next", classes="card emphasis")
                    yield Static("", id="home-lane", classes="card")
                    yield Static("", id="home-queue", classes="card")
                with Horizontal(classes="top-cards"):
                    yield Static("", id="home-routes", classes="card")
                    yield Static("", id="home-publish", classes="card")
                    yield Static("", id="home-service", classes="card")
                with Horizontal(classes="action-row"):
                    yield Button("Follow next action", id="btn-home-next", variant="primary")
                    yield Button("Refresh snapshot", id="btn-home-refresh", variant="default")
                with VerticalScroll(classes="detail-scroll"):
                    yield Static("", id="home-events", classes="detail-block")
            with TabPane("Routes", id="tab-routes"):
                with Horizontal(classes="split"):
                    with Vertical(classes="pane list-pane"):
                        with Horizontal(classes="filter-row"):
                            yield Label("View")
                            yield Select(
                                [("Active", "active"), ("Archived", "archived"), ("All", "all")],
                                value="active",
                                id="routes-view",
                            )
                        yield DataTable(id="routes-table")
                        with Horizontal(classes="action-row"):
                            yield Button("New", id="btn-route-new", variant="primary")
                            yield Button("Edit", id="btn-route-edit")
                            yield Button("Enable/Disable", id="btn-route-toggle")
                            yield Button("Promote", id="btn-route-promote", variant="success")
                            yield Button("Archive", id="btn-route-archive", variant="warning")
                            yield Button("Sync", id="btn-route-sync")
                    with VerticalScroll(classes="pane detail-pane"):
                        yield Static("", id="route-detail", classes="detail-block")
                        yield Static("Route editor", classes="section-title")
                        with Vertical(id="route-editor", classes="editor-block"):
                            with Horizontal(classes="editor-row"):
                                yield Label("Route ID", classes="editor-label")
                                yield Input(placeholder="resting-biosemi64", id="route-id")
                            with Horizontal(classes="editor-row"):
                                yield Label("Task File", classes="editor-label")
                                yield Input(placeholder="/path/to/Task.py", id="route-taskfile")
                            with Horizontal(classes="editor-row"):
                                yield Label("Montage", classes="editor-label")
                                yield Input(placeholder="biosemi64", id="route-montage")
                            with Horizontal(classes="editor-row"):
                                yield Label("Folders", classes="editor-label")
                                yield Input(placeholder="/data/incoming, /data/incoming2", id="route-folders")
                            with Horizontal(classes="editor-row"):
                                yield Label("File globs", classes="editor-label")
                                yield Input(placeholder="*.set, *_rest.set", id="route-globs")
                            with Horizontal(classes="editor-row"):
                                yield Label("Scope", classes="editor-label")
                                yield Select(
                                    [("Draft only", "test"), ("Draft + Production", "both")],
                                    value="test",
                                    id="route-scope",
                                )
                            with Horizontal(classes="editor-row"):
                                yield Label("Enabled", classes="editor-label")
                                yield Switch(value=True, id="route-enabled")
                                yield Label("Recursive", classes="editor-label")
                                yield Switch(value=True, id="route-recursive")
                            with Horizontal(classes="action-row"):
                                yield Button("Preview", id="btn-route-preview")
                                yield Button("Save", id="btn-route-save", variant="success")
                                yield Button("Reset", id="btn-route-reset")
                            yield Static("", id="route-preview", classes="detail-block")
            with TabPane("Queue", id="tab-queue"):
                with Horizontal(classes="split"):
                    with Vertical(classes="pane list-pane"):
                        with Horizontal(classes="filter-row"):
                            yield Label("Status")
                            yield Select(
                                [
                                    ("All", "all"),
                                    ("Needs attention", "failed"),
                                    ("Running", "processing"),
                                    ("Waiting", "pending"),
                                    ("Completed", "processed"),
                                ],
                                value="all",
                                id="queue-status-filter",
                            )
                            yield Label("Route")
                            yield Select([("All Routes", "all")], value="all", id="queue-route-filter")
                        yield DataTable(id="queue-table")
                        with Horizontal(classes="action-row"):
                            yield Button("Retry failed", id="btn-queue-retry", variant="warning")
                            yield Button("Remove", id="btn-queue-remove", variant="error")
                            yield Button("Clear completed", id="btn-queue-clear")
                            yield Button("Refresh", id="btn-queue-refresh")
                    with VerticalScroll(classes="pane detail-pane"):
                        yield Static("", id="queue-detail", classes="detail-block")
            with TabPane("Publish", id="tab-publish"):
                with Horizontal(classes="split"):
                    with VerticalScroll(classes="pane detail-pane"):
                        yield Static("", id="publish-summary", classes="detail-block")
                        with Horizontal(classes="action-row"):
                            yield Button("Validate", id="btn-publish-validate", variant="primary")
                            yield Button("Deploy", id="btn-publish-deploy", variant="success")
                            yield Button("Refresh", id="btn-publish-refresh")
                    with VerticalScroll(classes="pane detail-pane"):
                        yield Static("", id="publish-yaml", classes="detail-block code-block")
            with TabPane("Service", id="tab-service"):
                with Horizontal(classes="split"):
                    with Vertical(classes="pane detail-pane"):
                        yield Static("", id="service-summary", classes="detail-block")
                        yield Static("Service parameters", classes="section-title")
                        with Vertical(classes="editor-block"):
                            with Horizontal(classes="editor-row"):
                                yield Label("Max cycles", classes="editor-label")
                                yield Input(value="1000", id="service-max-cycles")
                            with Horizontal(classes="editor-row"):
                                yield Label("Idle limit", classes="editor-label")
                                yield Input(value="10", id="service-idle-limit")
                            with Horizontal(classes="editor-row"):
                                yield Label("Sleep sec", classes="editor-label")
                                yield Input(value="1.0", id="service-sleep-seconds")
                            with Horizontal(classes="editor-row"):
                                yield Label("Max events", classes="editor-label")
                                yield Input(value="1", id="service-max-events")
                            with Horizontal(classes="editor-row"):
                                yield Label("Dry run", classes="editor-label")
                                yield Switch(value=False, id="service-dry-run")
                                yield Label("Watchfiles", classes="editor-label")
                                yield Switch(value=True, id="service-watchfiles")
                                yield Label("Sentinel", classes="editor-label")
                                yield Switch(value=True, id="service-sentinel")
                            with Horizontal(classes="action-row"):
                                yield Button("Start", id="btn-service-start", variant="success")
                                yield Button("Stop", id="btn-service-stop", variant="error")
                                yield Button("Refresh", id="btn-service-refresh")
                    with VerticalScroll(classes="pane detail-pane"):
                        yield Static("", id="service-log", classes="detail-block code-block")
        yield Footer()

    def on_mount(self) -> None:
        self._initialize_tables()
        self.refresh_snapshot()
        if self.state.workspace_dir and self._watch_files:
            self._start_file_watcher()

    def _initialize_tables(self) -> None:
        routes_table = self.query_one("#routes-table", DataTable)
        routes_table.add_columns("Route", "State", "Lane", "Montage", "Task")
        routes_table.cursor_type = "row"
        queue_table = self.query_one("#queue-table", DataTable)
        queue_table.add_columns("File", "Status", "Route", "When", "Error")
        queue_table.cursor_type = "row"

    def _set_last_action(self, message: str) -> None:
        self._last_action_message = message
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        bar.mode = self.state.mode
        bar.service_running = self.state.service_running
        bar.config_source = self.state.service_last_config_source or self.get_service_config_source()[0]
        bar.queue_summary = (
            f"Queue: {self.state.failed_count} failed / {self.state.running_count} running / "
            f"{self.state.pending_count} waiting"
        )
        if self._snapshot is not None:
            bar.last_refresh = self._snapshot.last_refresh.strftime("%H:%M:%S")
        bar.last_action = self._last_action_message

    def get_queue_path(self) -> Optional[Path]:
        if not self.state.workspace_dir:
            return None
        return self.state.workspace_dir / f"queue-{self.state.mode}.json"

    def get_config_path(self, deployed: bool = False) -> Optional[Path]:
        if not self.state.workspace_dir:
            return None
        if deployed:
            return self.state.workspace_dir / "deploy" / f"serve-{self.state.mode}.yaml"
        return self.state.workspace_dir / f"serve-{self.state.mode}.yaml"

    def get_mode_label(self, mode: Optional[str] = None) -> str:
        return "Draft" if (mode or self.state.mode) == "test" else "Production"

    def get_service_config_source(self) -> tuple[str, Optional[Path]]:
        deployed_config = self.get_config_path(deployed=True)
        operator_config = self.get_config_path(deployed=False)
        if deployed_config is not None and deployed_config.exists():
            return ("deployed", deployed_config)
        if operator_config is not None and operator_config.exists():
            return ("operator", operator_config)
        return ("missing", operator_config or deployed_config)

    def configure_service(self, params: dict[str, Any]) -> None:
        self.state.service_settings = ServiceSettings(**params)

    def build_service_command(self, cli_path: Path) -> list[str]:
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
        if not self.state.workspace_dir:
            return
        config_file = self.get_config_path(deployed=False)
        if config_file is None:
            return
        if not config_file.exists():
            self.state.config_valid = False
            self.state.config_errors = [f"Config file not found: {config_file}"]
            self.state.config_warnings = []
            return
        try:
            from autoclean.utils.ingestion import load_serve_config, parse_serve_config

            raw_config = load_serve_config(config_file)
            parse_serve_config(raw_config, self.state.workspace_dir, strict=True)
            _, warnings = parse_serve_config(raw_config, self.state.workspace_dir, strict=False)
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
            self.state.config_warnings = []

    def _load_queue(self) -> None:
        queue_path = self.get_queue_path()
        if queue_path is None or not queue_path.exists():
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
            self.state.pending_count = 0
            self.state.completed_count = 0
            self.state.running_count = 0
            self.state.failed_count = 0
            latest_processed_at = ""
            latest_failed_at = ""
            self.state.last_completed_file = None
            self.state.last_failed_file = None
            self.state.last_failed_error = None
            for path_str, entry_data in entries.items():
                status = str(entry_data.get("status", "pending"))
                if status == "pending":
                    self.state.pending_count += 1
                elif status == "processing":
                    self.state.running_count += 1
                elif status == "processed":
                    self.state.completed_count += 1
                    processed_at = str(entry_data.get("processed_at") or entry_data.get("added_at") or "")
                    if processed_at >= latest_processed_at:
                        latest_processed_at = processed_at
                        self.state.last_completed_file = Path(path_str).name
                elif status == "failed":
                    self.state.failed_count += 1
                    failed_at = str(entry_data.get("failed_at") or entry_data.get("added_at") or "")
                    if failed_at >= latest_failed_at:
                        latest_failed_at = failed_at
                        self.state.last_failed_file = Path(path_str).name
                        self.state.last_failed_error = str(entry_data.get("last_error") or "")
        except Exception:
            pass

    @work(exclusive=True, thread=True)
    def _start_file_watcher(self) -> None:
        if not self.state.workspace_dir:
            return
        try:
            from watchfiles import watch

            paths_to_watch = [self.state.workspace_dir]
            for changes in watch(*paths_to_watch, recursive=False, stop_event=self._watcher_stop_event):
                if self._watcher_stop_event.is_set():
                    break
                for _, path in changes:
                    changed_path = Path(path)
                    if changed_path.name.startswith("queue-") or changed_path.name.startswith("serve-"):
                        self.call_from_thread(self.refresh_snapshot)
                        break
        except Exception:
            return

    def _add_activity_event(
        self,
        event_type: str,
        message: str,
        file_path: Optional[Path] = None,
        route_id: Optional[str] = None,
    ) -> None:
        self.state.activity_log.insert(
            0,
            ActivityEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                message=message,
                file_path=file_path,
                route_id=route_id,
            ),
        )
        self.state.activity_log = self.state.activity_log[:100]

    def get_routes(self) -> list[Any]:
        config_file = self.get_config_path(deployed=False)
        if config_file is None or not config_file.exists():
            return []
        try:
            from autoclean.utils.ingestion import load_serve_config, parse_serve_config

            raw_config = load_serve_config(config_file)
            config, _ = parse_serve_config(raw_config, self.state.workspace_dir, strict=False)
            return config.routes
        except Exception:
            return []

    def get_route_specs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        if self.state.workspace_dir is None:
            return []
        try:
            from autoclean.utils.serve_routes import load_route_specs

            routes = load_route_specs(self.state.workspace_dir)
            if include_archived:
                return routes
            return [route for route in routes if not route.get("archived", False)]
        except Exception:
            return []

    def get_route_spec(self, route_id: str) -> Optional[dict[str, Any]]:
        for route in self.get_route_specs(include_archived=True):
            if str(route.get("id")) == route_id:
                return route
        return None

    def sync_route_registry(self) -> bool:
        if self.state.workspace_dir is None:
            return False
        try:
            from autoclean.utils.serve_routes import sync_route_registry

            sync_route_registry(self.state.workspace_dir)
            self._load_config()
            self._add_activity_event("route_sync", "Route registry synced")
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
            from autoclean.utils.serve_routes import sync_route_registry, upsert_route_spec

            modes = ["test", "live"] if mode_scope == "both" else ["test"]
            updates: dict[str, Any] = {
                "modes": modes,
                "taskfile": str(Path(taskfile).expanduser().resolve()),
                "montage": montage,
                "ingestion_folders": [str(Path(item).expanduser().resolve()) for item in folders],
                "enabled": enabled,
                "recursive": recursive,
            }
            if globs:
                updates["file_globs"] = globs
            upsert_route_spec(workspace_dir, route_id, updates)
            sync_route_registry(workspace_dir)
            self._load_config()
            self._add_activity_event("route_saved", f"Saved route {route_id}", route_id=route_id)
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
        folders = [item.strip() for item in ingestion_folders if item.strip()]
        globs = [item.strip() for item in file_globs if item.strip()]
        preview: dict[str, Any] = {
            "taskfile": taskfile.strip(),
            "montage": montage.strip(),
            "folders": [],
            "mode_scope": "Draft + Production" if mode_scope == "both" else "Draft only",
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
                iterator = resolved.rglob(pattern) if recursive else resolved.glob(pattern)
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
            preview["warnings"].append("No matching files found in the selected folders yet.")
        return preview

    def set_route_enabled(self, route_id: str, enabled: bool) -> bool:
        if self.state.workspace_dir is None:
            return False
        try:
            from autoclean.utils.serve_routes import sync_route_registry, upsert_route_spec

            upsert_route_spec(self.state.workspace_dir, route_id, {"enabled": enabled})
            sync_route_registry(self.state.workspace_dir)
            self._load_config()
            self._add_activity_event(
                "route_toggle",
                f"{'Enabled' if enabled else 'Disabled'} route {route_id}",
                route_id=route_id,
            )
            return True
        except Exception:
            return False

    def set_route_archived(self, route_id: str, archived: bool) -> bool:
        if self.state.workspace_dir is None:
            return False
        try:
            from autoclean.utils.serve_routes import archive_route_spec, sync_route_registry, unarchive_route_spec

            if archived:
                archive_route_spec(self.state.workspace_dir, route_id)
            else:
                unarchive_route_spec(self.state.workspace_dir, route_id)
            sync_route_registry(self.state.workspace_dir)
            self._load_config()
            self._add_activity_event(
                "route_archive",
                f"{'Archived' if archived else 'Restored'} route {route_id}",
                route_id=route_id,
            )
            return True
        except Exception:
            return False

    def promote_route(self, route_id: str) -> bool:
        if self.state.workspace_dir is None:
            return False
        try:
            from autoclean.utils.serve_routes import promote_route_spec, sync_route_registry

            promote_route_spec(self.state.workspace_dir, route_id)
            sync_route_registry(self.state.workspace_dir)
            self._load_config()
            self._add_activity_event("route_promote", f"Promoted route {route_id}", route_id=route_id)
            return True
        except Exception:
            return False

    def get_queue_entries(self) -> dict[str, Any]:
        queue_path = self.get_queue_path()
        if queue_path is None or not queue_path.exists():
            return {}
        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            return queue.entries()
        except Exception:
            return {}

    def get_service_runtime_snapshot(self) -> dict[str, Any]:
        config_source, config_path = self.get_service_config_source()
        queue_path = self.get_queue_path()
        uptime = None
        if self.state.service_started_at is not None and self.state.service_running:
            uptime_seconds = int((datetime.now() - self.state.service_started_at).total_seconds())
            minutes, seconds = divmod(uptime_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return {
            "lane": self.get_mode_label(),
            "workspace": str(self.state.workspace_dir) if self.state.workspace_dir else "Not configured",
            "queue_path": str(queue_path) if queue_path else "Unavailable",
            "config_source": self.state.service_last_config_source or config_source,
            "config_path": str(config_path) if config_path else "Unavailable",
            "log_path": str(self.state.service_log_path) if self.state.service_log_path else "Unavailable",
            "pid": self.state.service_process.pid if self.state.service_process else None,
            "uptime": uptime,
            "command": " ".join(self.state.service_last_command) if self.state.service_last_command else "Not started yet",
            "completed": self.state.last_completed_file,
            "failed": self.state.last_failed_file,
            "failed_error": self.state.last_failed_error,
        }

    def read_service_log_tail(self, line_count: int = 20) -> str:
        log_path = self.state.service_log_path
        if log_path is None or not log_path.exists():
            return ""
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-line_count:])
        except Exception:
            return ""

    def get_config_yaml(self) -> str:
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
        if not self.state.workspace_dir:
            return False, "No workspace configured"
        source = self.get_config_path(deployed=False)
        target = self.get_config_path(deployed=True)
        if source is None or target is None:
            return False, "No workspace configured"
        if not source.exists():
            return False, f"Config file not found: {source}"
        try:
            from autoclean.utils.ingestion import load_serve_config, parse_serve_config

            raw_config = load_serve_config(source)
            parse_serve_config(raw_config, self.state.workspace_dir, strict=True)
        except ServeConfigError as exc:
            return False, "Cannot deploy invalid configuration: " + "; ".join(exc.errors)
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
        self._add_activity_event("deploy", f"Deployed {target.name}")
        return True, f"Configuration deployed to {target.name}"

    def refresh_snapshot(self) -> None:
        self._load_config()
        self._load_queue()
        config_source, _ = self.get_service_config_source()
        operator_path = self.get_config_path(deployed=False) or Path("serve-missing.yaml")
        deployed_path = self.get_config_path(deployed=True) or Path("deploy/serve-missing.yaml")
        self._snapshot = build_workspace_snapshot(
            workspace_dir=self.state.workspace_dir,
            mode=self.state.mode,
            config_valid=self.state.config_valid,
            config_errors=self.state.config_errors,
            config_warnings=self.state.config_warnings,
            route_specs=self.get_route_specs(include_archived=True),
            queue_entries=self.get_queue_entries(),
            service_snapshot=self.get_service_runtime_snapshot(),
            activity_log=self.state.activity_log,
            operator_config_path=operator_path,
            deployed_config_path=deployed_path,
            config_source=config_source,
        )
        self.state.service_running = bool(self.state.service_process and self.state.service_process.poll() is None)
        self._update_status_bar()
        self._refresh_home_tab()
        self._refresh_routes_tab()
        self._refresh_queue_tab()
        self._refresh_publish_tab()
        self._refresh_service_tab()

    def _refresh_home_tab(self) -> None:
        if self._snapshot is None:
            return
        snapshot = self._snapshot
        self.query_one("#home-next", Static).update(
            f"[b]{snapshot.recommended_action.title}[/b]\n\n{snapshot.recommended_action.description}\n\n"
            f"Action: {snapshot.recommended_action.button_label}"
        )
        self.query_one("#home-lane", Static).update(
            f"[b]{snapshot.lane_label} lane[/b]\nWorkspace: {snapshot.workspace_dir or 'Not configured'}\n"
            f"Config source: {snapshot.publish.config_source}"
        )
        self.query_one("#home-queue", Static).update(
            f"[b]Queue[/b]\nNeeds attention: {snapshot.queue.failed}\nRunning: {snapshot.queue.processing}\n"
            f"Waiting: {snapshot.queue.pending}\nCompleted: {snapshot.queue.processed}"
        )
        ready = len([route for route in snapshot.routes if route.state == "Ready"])
        draft_only = len([route for route in snapshot.routes if route.state == "Draft only"])
        attention = len([route for route in snapshot.routes if route.state == "Attention"])
        self.query_one("#home-routes", Static).update(
            f"[b]Routes[/b]\nReady: {ready}\nDraft only: {draft_only}\nNeeds attention: {attention}\n"
            f"Total tracked: {len(snapshot.routes)}"
        )
        self.query_one("#home-publish", Static).update(
            f"[b]Publish[/b]\nConfig valid: {'yes' if snapshot.publish.config_valid else 'no'}\n"
            f"Warnings: {len(snapshot.publish.config_warnings)}\nNeeds deploy: {'yes' if snapshot.publish.needs_deploy else 'no'}"
        )
        self.query_one("#home-service", Static).update(
            f"[b]Service[/b]\nRunning: {'yes' if snapshot.service.running else 'no'}\n"
            f"Lane: {snapshot.service.lane}\nCommand: {snapshot.service.command}"
        )
        events = ["[b]Recent activity[/b]"]
        if snapshot.recent_events:
            for event in snapshot.recent_events:
                events.append(f"{event.timestamp} [{event.event_type}] {event.message}")
        else:
            events.append("No operator activity recorded yet.")
        self.query_one("#home-events", Static).update("\n".join(events))

    def _refresh_routes_tab(self) -> None:
        if self._snapshot is None:
            return
        view = str(self.query_one("#routes-view", Select).value)
        routes = self._snapshot.routes
        if view == "active":
            routes = [route for route in routes if not route.archived]
        elif view == "archived":
            routes = [route for route in routes if route.archived]
        table = self.query_one("#routes-table", DataTable)
        table.clear()
        self._route_row_ids = []
        for route in routes:
            lane = "Draft + Production" if "live" in route.modes else "Draft only"
            task_name = Path(route.taskfile).name if route.taskfile else "-"
            table.add_row(route.label, route.state, lane, route.montage or "-", task_name, key=route.route_id)
            self._route_row_ids.append(route.route_id)
        if self._selected_route_id not in self._route_row_ids:
            self._selected_route_id = self._route_row_ids[0] if self._route_row_ids else None
        self._refresh_route_detail()
        if not self._route_editor_existing_id and self._selected_route_id:
            self._load_route_editor(self.get_route_spec(self._selected_route_id), edit_mode=False)

    def _refresh_route_detail(self) -> None:
        if self._snapshot is None:
            return
        route = next((item for item in self._snapshot.routes if item.route_id == self._selected_route_id), None)
        if route is None:
            self.query_one("#route-detail", Static).update("Select a route to inspect it.")
            return
        issues = route.issues or [route.summary]
        lanes = ", ".join(route.modes) if route.modes else "test"
        folders = "\n".join(f"- {path}" for path in route.ingestion_folders) or "- None"
        globs = ", ".join(route.file_globs) or "*"
        detail = [
            f"[b]{route.label}[/b]",
            f"State: {route.state}",
            f"Route ID: {route.route_id}",
            f"Task file: {route.taskfile or '-'}",
            f"Montage: {route.montage or '-'}",
            f"Lanes: {lanes}",
            f"Enabled: {'yes' if route.enabled else 'no'}",
            f"Archived: {'yes' if route.archived else 'no'}",
            f"Priority: {route.priority}",
            f"File globs: {globs}",
            "Folders:",
            folders,
            "",
            "What needs attention:",
        ]
        detail.extend(f"- {issue}" for issue in issues)
        detail.append("")
        detail.append("Operator actions: edit, promote, archive, enable/disable, sync.")
        self.query_one("#route-detail", Static).update("\n".join(detail))

    def _refresh_queue_route_filter(self) -> None:
        routes = [item.route_id for item in self._snapshot.queue_items if item.route_id != "-"] if self._snapshot else []
        options = [("All Routes", "all")]
        for route_id in sorted(set(routes)):
            options.append((route_id, route_id))
        self.query_one("#queue-route-filter", Select).set_options(options)

    def _refresh_queue_tab(self) -> None:
        if self._snapshot is None:
            return
        self._refresh_queue_route_filter()
        status_filter = str(self.query_one("#queue-status-filter", Select).value)
        route_filter = str(self.query_one("#queue-route-filter", Select).value)
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        self._queue_row_paths = []
        items = self._snapshot.queue_items
        for item in items:
            if status_filter != "all" and item.status != status_filter:
                continue
            if route_filter != "all" and item.route_id != route_filter:
                continue
            error = item.last_error[:72] + ("..." if len(item.last_error) > 72 else "") if item.last_error else "-"
            when = item.updated_at or item.added_at
            table.add_row(item.file_name, item.status, item.route_id, when[:19] or "-", error, key=item.path)
            self._queue_row_paths.append(item.path)
        if self._selected_queue_path not in self._queue_row_paths:
            self._selected_queue_path = self._queue_row_paths[0] if self._queue_row_paths else None
        self._refresh_queue_detail()

    def _refresh_queue_detail(self) -> None:
        if self._snapshot is None:
            return
        item = next((entry for entry in self._snapshot.queue_items if entry.path == self._selected_queue_path), None)
        if item is None:
            self.query_one("#queue-detail", Static).update("Select a queue item to inspect it.")
            return
        lines = [
            f"[b]{item.file_name}[/b]",
            f"Status: {item.status}",
            f"Route: {item.route_id}",
            f"Path: {item.path}",
            f"Added: {item.added_at or '-'}",
            f"Latest event: {item.updated_at or '-'}",
        ]
        if item.last_error:
            lines.extend(["", "Last error:", item.last_error])
        else:
            lines.extend(["", "No error text recorded for this item."])
        lines.extend(["", "Safe actions: retry failed items, remove one item, clear completed items."])
        self.query_one("#queue-detail", Static).update("\n".join(lines))

    def _refresh_publish_tab(self) -> None:
        if self._snapshot is None:
            return
        publish = self._snapshot.publish
        summary = [
            f"[b]{self._snapshot.lane_label} publish status[/b]",
            f"Operator config: {publish.operator_config_path}",
            f"Deployed config: {publish.deployed_config_path}",
            f"Config source in use: {publish.config_source}",
            f"Strict validation: {'passing' if publish.config_valid else 'failing'}",
            f"Needs deploy: {'yes' if publish.needs_deploy else 'no'}",
            "",
            f"Errors ({len(publish.config_errors)}):",
        ]
        summary.extend(f"- {error}" for error in publish.config_errors) or summary.append("- None")
        summary.append("")
        summary.append(f"Warnings ({len(publish.config_warnings)}):")
        summary.extend(f"- {warning}" for warning in publish.config_warnings) or summary.append("- None")
        self.query_one("#publish-summary", Static).update("\n".join(summary))
        self.query_one("#publish-yaml", Static).update(self.get_config_yaml() or "# No operator config present")

    def _refresh_service_tab(self) -> None:
        if self._snapshot is None:
            return
        service = self._snapshot.service
        summary = [
            f"[b]{self._snapshot.lane_label} service[/b]",
            f"Running: {'yes' if service.running else 'no'}",
            f"Workspace: {service.workspace}",
            f"Queue path: {service.queue_path}",
            f"Config source: {service.config_source}",
            f"Config path: {service.config_path}",
            f"Log path: {service.log_path}",
            f"PID: {service.pid or '-'}",
            f"Uptime: {service.uptime or '-'}",
            f"Last completed: {service.completed or '-'}",
            f"Last failed: {service.failed or '-'}",
        ]
        if service.failed_error:
            summary.extend(["", "Last failure detail:", service.failed_error])
        summary.extend(["", "Command:", service.command])
        self.query_one("#service-summary", Static).update("\n".join(summary))
        self.query_one("#service-log", Static).update(self.read_service_log_tail() or "No service log yet.")
        self._sync_service_form_from_state()

    def _sync_service_form_from_state(self) -> None:
        settings = self.state.service_settings
        self.query_one("#service-max-cycles", Input).value = str(settings.max_cycles)
        self.query_one("#service-idle-limit", Input).value = str(settings.idle_limit)
        self.query_one("#service-sleep-seconds", Input).value = str(settings.sleep_seconds)
        self.query_one("#service-max-events", Input).value = str(settings.max_events)
        self.query_one("#service-dry-run", Switch).value = settings.dry_run
        self.query_one("#service-watchfiles", Switch).value = settings.use_watchfiles
        self.query_one("#service-sentinel", Switch).value = settings.require_sentinel

    def _read_service_form(self) -> Optional[dict[str, Any]]:
        try:
            return {
                "max_cycles": int(self.query_one("#service-max-cycles", Input).value),
                "idle_limit": int(self.query_one("#service-idle-limit", Input).value),
                "sleep_seconds": float(self.query_one("#service-sleep-seconds", Input).value),
                "max_events": int(self.query_one("#service-max-events", Input).value),
                "dry_run": self.query_one("#service-dry-run", Switch).value,
                "use_watchfiles": self.query_one("#service-watchfiles", Switch).value,
                "require_sentinel": self.query_one("#service-sentinel", Switch).value,
            }
        except ValueError:
            self.notify("Service settings must be numeric where expected", severity="error")
            return None

    def _set_active_tab(self, tab_id: str) -> None:
        self.query_one("#main-tabs", TabbedContent).active = tab_id

    def action_show_home(self) -> None:
        self._set_active_tab("tab-home")

    def action_show_routes(self) -> None:
        self._set_active_tab("tab-routes")

    def action_show_queue(self) -> None:
        self._set_active_tab("tab-queue")

    def action_show_publish(self) -> None:
        self._set_active_tab("tab-publish")

    def action_show_service(self) -> None:
        self._set_active_tab("tab-service")

    def action_refresh_snapshot(self) -> None:
        self.refresh_snapshot()
        self._set_last_action("Snapshot refreshed")
        self.notify("Snapshot refreshed")

    def action_toggle_mode(self) -> None:
        self.state.mode = "live" if self.state.mode == "test" else "test"
        self._add_activity_event("lane_toggle", f"Switched to {self.get_mode_label()} lane")
        self.refresh_snapshot()
        self._set_last_action(f"Switched to {self.get_mode_label()} lane")
        self.notify(f"Switched to {self.get_mode_label()} lane")

    def action_toggle_service(self) -> None:
        if self.state.service_running:
            self._stop_service()
        else:
            params = self._read_service_form()
            if params is None:
                return
            self.configure_service(params)
            self._start_service()

    @work(exclusive=True, thread=True)
    def _start_service(self) -> None:
        if not self.state.workspace_dir:
            self.call_from_thread(self.notify, "No workspace configured", severity="error")
            return
        try:
            from autoclean.utils.ingestion import resolve_runtime_cli

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
                log_handle.write(f"\n[{datetime.now().isoformat()}] Starting service: {' '.join(cmd)}\n")
                log_handle.flush()
                self.state.service_process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.state.service_running = True
                self.call_from_thread(self.refresh_snapshot)
                self.call_from_thread(self._set_last_action, "Service started")
                self.call_from_thread(self.notify, "Service started", severity="information")
                self.call_from_thread(self._add_activity_event, "service_start", f"Service started ({log_path.name})")
                returncode = self.state.service_process.wait()
            stop_requested = self.state.service_stop_requested
            self.state.service_stop_requested = False
            self.state.service_running = False
            self.state.service_started_at = None
            self.state.service_process = None
            self.state.service_last_returncode = returncode
            self.call_from_thread(self.refresh_snapshot)
            if stop_requested or returncode == 0:
                self.call_from_thread(self._set_last_action, "Service stopped")
                self.call_from_thread(self.notify, "Service stopped", severity="information")
                self.call_from_thread(self._add_activity_event, "service_stop", "Service stopped")
            else:
                self.call_from_thread(self._set_last_action, f"Service exited with code {returncode}")
                self.call_from_thread(self.notify, f"Service exited with code {returncode}", severity="error")
                self.call_from_thread(
                    self._add_activity_event,
                    "service_error",
                    f"Service exited with code {returncode}",
                )
        except Exception as exc:
            self.state.service_running = False
            self.state.service_stop_requested = False
            self.state.service_started_at = None
            self.state.service_process = None
            self.call_from_thread(self.refresh_snapshot)
            self.call_from_thread(self.notify, f"Failed to start service: {exc}", severity="error")

    def _stop_service(self) -> None:
        if self.state.service_process:
            self.state.service_stop_requested = True
            self.state.service_process.terminate()
            self._add_activity_event("service_stop", "Service stop requested by user")
            self._set_last_action("Stopping service")
            self.notify("Stopping service...", severity="information")

    def _load_route_editor(self, route_spec: Optional[dict[str, Any]], *, edit_mode: bool) -> None:
        route_spec = route_spec or {}
        self._route_editor_mode = "edit" if edit_mode and route_spec.get("id") else "create"
        self._route_editor_existing_id = str(route_spec.get("id")) if edit_mode and route_spec.get("id") else None
        route_id_input = self.query_one("#route-id", Input)
        route_id_input.disabled = bool(self._route_editor_existing_id)
        route_id_input.value = str(route_spec.get("id") or "")
        self.query_one("#route-taskfile", Input).value = str(route_spec.get("taskfile") or "")
        self.query_one("#route-montage", Input).value = str(route_spec.get("montage") or "")
        self.query_one("#route-folders", Input).value = ", ".join(route_spec.get("ingestion_folders", []))
        self.query_one("#route-globs", Input).value = ", ".join(route_spec.get("file_globs", []))
        modes = route_spec.get("modes", ["test"])
        self.query_one("#route-scope", Select).value = "both" if "live" in modes else "test"
        self.query_one("#route-enabled", Switch).value = bool(route_spec.get("enabled", True))
        self.query_one("#route-recursive", Switch).value = bool(route_spec.get("recursive", True))
        if route_spec:
            self._refresh_route_preview()
        else:
            self.query_one("#route-preview", Static).update(
                "Create a route by selecting a task, a montage, and one or more ingestion folders."
            )

    def _refresh_route_preview(self) -> None:
        preview = self.preview_route_spec(
            taskfile=self.query_one("#route-taskfile", Input).value,
            montage=self.query_one("#route-montage", Input).value,
            ingestion_folders=self.query_one("#route-folders", Input).value.split(","),
            file_globs=self.query_one("#route-globs", Input).value.split(","),
            mode_scope=str(self.query_one("#route-scope", Select).value),
            recursive=self.query_one("#route-recursive", Switch).value,
        )
        lines = [
            f"[b]{'Editing' if self._route_editor_existing_id else 'New'} route preview[/b]",
            f"Task file: {preview['taskfile'] or 'Missing'}",
            f"Montage: {preview['montage'] or 'Missing'}",
            f"Scope: {preview['mode_scope']}",
            "Folders:",
        ]
        if preview["folders"]:
            lines.extend(f"- {folder}" for folder in preview["folders"])
        else:
            lines.append("- None")
        lines.append("Sample matches:")
        if preview["matches"]:
            lines.extend(f"- {match}" for match in preview["matches"])
        else:
            lines.append("- None yet")
        if preview["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in preview["warnings"])
        self.query_one("#route-preview", Static).update("\n".join(lines))

    def _selected_route_spec(self) -> Optional[dict[str, Any]]:
        if not self._selected_route_id:
            return None
        return self.get_route_spec(self._selected_route_id)

    def _mutate_queue(self, mutator: Any) -> tuple[bool, str]:
        queue_path = self.get_queue_path()
        if queue_path is None or not queue_path.exists():
            return False, "Queue file not found"
        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            message = mutator(queue.entries())
            queue.save()
            self._load_queue()
            return True, message
        except Exception as exc:
            return False, str(exc)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "btn-home-next":
            self._follow_recommendation()
        elif button_id == "btn-home-refresh":
            self.action_refresh_snapshot()
        elif button_id == "btn-route-new":
            self._set_active_tab("tab-routes")
            self._load_route_editor(None, edit_mode=False)
        elif button_id == "btn-route-edit":
            self._load_route_editor(self._selected_route_spec(), edit_mode=True)
        elif button_id == "btn-route-preview":
            self._refresh_route_preview()
        elif button_id == "btn-route-save":
            ok, error = self.upsert_route_spec(
                route_id=self.query_one("#route-id", Input).value,
                existing_route_id=self._route_editor_existing_id,
                taskfile=self.query_one("#route-taskfile", Input).value,
                montage=self.query_one("#route-montage", Input).value,
                ingestion_folders=self.query_one("#route-folders", Input).value.split(","),
                file_globs=self.query_one("#route-globs", Input).value.split(","),
                mode_scope=str(self.query_one("#route-scope", Select).value),
                enabled=self.query_one("#route-enabled", Switch).value,
                recursive=self.query_one("#route-recursive", Switch).value,
            )
            if ok:
                self.refresh_snapshot()
                self._set_last_action("Route saved")
                self.notify("Route saved")
            else:
                self.notify(error or "Failed to save route", severity="error")
                self._refresh_route_preview()
        elif button_id == "btn-route-reset":
            self._load_route_editor(self._selected_route_spec() if self._route_editor_existing_id else None, edit_mode=bool(self._route_editor_existing_id))
        elif button_id == "btn-route-toggle":
            route = self._selected_route_spec()
            if route and self.set_route_enabled(str(route.get("id")), not bool(route.get("enabled", True))):
                self.refresh_snapshot()
                self._set_last_action("Route toggled")
                self.notify("Route updated")
        elif button_id == "btn-route-promote":
            route = self._selected_route_spec()
            if route and self.promote_route(str(route.get("id"))):
                self.refresh_snapshot()
                self._set_last_action("Route promoted")
                self.notify("Route promoted to Production")
        elif button_id == "btn-route-archive":
            route = self._selected_route_spec()
            if route and self.set_route_archived(str(route.get("id")), not bool(route.get("archived", False))):
                self.refresh_snapshot()
                self._set_last_action("Route archive state updated")
                self.notify("Route archive state updated")
        elif button_id == "btn-route-sync":
            if self.sync_route_registry():
                self.refresh_snapshot()
                self._set_last_action("Routes synced")
                self.notify("Route registry synced")
            else:
                self.notify("Failed to sync route registry", severity="error")
        elif button_id == "btn-queue-refresh":
            self.refresh_snapshot()
            self._set_last_action("Queue refreshed")
        elif button_id == "btn-queue-retry":
            ok, message = self._mutate_queue(self._retry_failed_entries)
            if ok:
                self.refresh_snapshot()
                self._set_last_action(message)
                self.notify(message)
            else:
                self.notify(message, severity="error")
        elif button_id == "btn-queue-remove":
            ok, message = self._mutate_queue(self._remove_selected_entry)
            if ok:
                self.refresh_snapshot()
                self._set_last_action(message)
                self.notify(message)
            else:
                self.notify(message, severity="error")
        elif button_id == "btn-queue-clear":
            ok, message = self._mutate_queue(self._clear_processed_entries)
            if ok:
                self.refresh_snapshot()
                self._set_last_action(message)
                self.notify(message)
            else:
                self.notify(message, severity="error")
        elif button_id == "btn-publish-validate":
            self._load_config()
            self.refresh_snapshot()
            self._set_last_action("Validation refreshed")
            self.notify("Configuration revalidated")
        elif button_id == "btn-publish-deploy":
            ok, message = self.deploy_current_config()
            if ok:
                self.refresh_snapshot()
                self._set_last_action("Configuration deployed")
                self.notify(message)
            else:
                self.notify(message, severity="error")
        elif button_id == "btn-publish-refresh":
            self.refresh_snapshot()
            self._set_last_action("Publish state refreshed")
        elif button_id == "btn-service-start":
            params = self._read_service_form()
            if params is None:
                return
            self.configure_service(params)
            self._start_service()
        elif button_id == "btn-service-stop":
            self._stop_service()
        elif button_id == "btn-service-refresh":
            self.refresh_snapshot()
            self._set_last_action("Service state refreshed")

    def _retry_failed_entries(self, entries: dict[str, Any]) -> str:
        retried = 0
        for data in entries.values():
            if data.get("status") == "failed":
                data["status"] = "pending"
                data.pop("last_error", None)
                data.pop("failed_at", None)
                retried += 1
        if retried == 0:
            return "No failed items to retry"
        self._add_activity_event("queue_retry", f"Retried {retried} failed item(s)")
        return f"Retried {retried} failed item(s)"

    def _remove_selected_entry(self, entries: dict[str, Any]) -> str:
        if not self._selected_queue_path:
            raise ValueError("No queue item selected")
        removed = entries.pop(self._selected_queue_path, None)
        if removed is None:
            raise ValueError("Selected queue item was not found")
        file_name = Path(self._selected_queue_path).name
        self._add_activity_event("queue_remove", f"Removed {file_name}", file_path=Path(self._selected_queue_path))
        return f"Removed {file_name}"

    def _clear_processed_entries(self, entries: dict[str, Any]) -> str:
        to_remove = [path for path, data in entries.items() if data.get("status") == "processed"]
        for path in to_remove:
            del entries[path]
        if not to_remove:
            return "No completed items to clear"
        self._add_activity_event("queue_clear", f"Cleared {len(to_remove)} completed item(s)")
        return f"Cleared {len(to_remove)} completed item(s)"

    def _follow_recommendation(self) -> None:
        if self._snapshot is None:
            return
        recommendation = self._snapshot.recommended_action
        if recommendation.direct_action == "start_service":
            params = self._read_service_form()
            if params is None:
                return
            self.configure_service(params)
            self._set_active_tab("tab-service")
            self._start_service()
            return
        if recommendation.direct_action == "refresh":
            self.action_refresh_snapshot()
            return
        self._set_active_tab(recommendation.target_tab)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_index = getattr(event, "cursor_row", None)
        if row_index is None and getattr(event, "coordinate", None) is not None:
            row_index = event.coordinate.row
        if row_index is None:
            return
        table_id = event.data_table.id
        if table_id == "routes-table" and 0 <= row_index < len(self._route_row_ids):
            self._selected_route_id = self._route_row_ids[row_index]
            self._refresh_route_detail()
        elif table_id == "queue-table" and 0 <= row_index < len(self._queue_row_paths):
            self._selected_queue_path = self._queue_row_paths[row_index]
            self._refresh_queue_detail()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "routes-view":
            self._refresh_routes_tab()
        elif event.select.id in {"queue-status-filter", "queue-route-filter"}:
            self._refresh_queue_tab()

    def action_quit(self) -> None:
        self._watcher_stop_event.set()
        self.exit()


def run_tui(workspace_path: Optional[Path] = None, mode: str = "test") -> None:
    if os.environ.get("AUTOCLEAN_TUI_LEGACY") == "1":
        from autoclean.tui.legacy_app import AutoCleanTUI as LegacyAutoCleanTUI

        LegacyAutoCleanTUI(workspace_path=workspace_path, mode=mode).run()
        return
    AutoCleanTUI(workspace_path=workspace_path, mode=mode).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoClean Serve TUI")
    parser.add_argument("--path", type=Path, default=None, help="Serve workspace path")
    parser.add_argument("--mode", choices=["test", "live"], default="test")
    args = parser.parse_args()
    run_tui(workspace_path=args.path, mode=args.mode)


__all__ = [
    "ActivityEvent",
    "AppState",
    "AutoCleanTUI",
    "ServiceSettings",
    "StatusBar",
    "main",
    "run_tui",
]
