"""Routes view screen for AutoClean TUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label, Select, Static, Switch

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class RouteEditorScreen(Screen):
    """Create or edit a route spec."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, route_spec: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.route_spec = route_spec or {}
        self.is_edit_mode = bool(self.route_spec.get("id"))

    def compose(self) -> ComposeResult:
        route_id = str(self.route_spec.get("id", ""))
        taskfile = str(self.route_spec.get("taskfile", ""))
        montage = str(self.route_spec.get("montage", ""))
        folders = ", ".join(self.route_spec.get("ingestion_folders", []))
        globs = ", ".join(self.route_spec.get("file_globs", []))
        modes = self.route_spec.get("modes", ["test"])
        mode_scope = "both" if "live" in modes else "test"
        enabled = bool(self.route_spec.get("enabled", True))
        recursive = bool(self.route_spec.get("recursive", True))
        heading = "Edit Route" if self.is_edit_mode else "Create Route"

        with Vertical():
            yield Static(heading, classes="section-header")
            yield Static(
                "Create routes in Draft first. Editing keeps the current route ID locked "
                "so you do not accidentally fork a second route.",
                classes="help-text",
            )

            with Vertical(classes="service-params"):
                with Horizontal(classes="param-row"):
                    yield Label("Route ID:", classes="param-label")
                    yield Input(
                        value=route_id,
                        placeholder="resting-biosemi64",
                        id="input-route-id",
                        classes="param-input",
                        disabled=self.is_edit_mode,
                    )

                with Horizontal(classes="param-row"):
                    yield Label("Task File:", classes="param-label")
                    yield Input(
                        value=taskfile,
                        placeholder="/path/to/RestingEyesOpen.py",
                        id="input-taskfile",
                        classes="param-input",
                    )

                with Horizontal(classes="param-row"):
                    yield Label("Montage:", classes="param-label")
                    yield Input(
                        value=montage,
                        placeholder="biosemi64",
                        id="input-montage",
                        classes="param-input",
                    )

                with Horizontal(classes="param-row"):
                    yield Label("Folders:", classes="param-label")
                    yield Input(
                        value=folders,
                        placeholder="/data/incoming/resting, /data/incoming/resting2",
                        id="input-folders",
                        classes="param-input",
                    )

                with Horizontal(classes="param-row"):
                    yield Label("File Globs:", classes="param-label")
                    yield Input(
                        value=globs,
                        placeholder="*.set, *_resting.set",
                        id="input-globs",
                        classes="param-input",
                    )

                with Horizontal(classes="param-row"):
                    yield Label("Scope:", classes="param-label")
                    yield Select(
                        [("Draft only", "test"), ("Draft + Production", "both")],
                        value=mode_scope,
                        id="select-scope",
                    )

                with Horizontal(classes="param-row"):
                    yield Label("Enabled:", classes="param-label")
                    yield Switch(value=enabled, id="switch-enabled")

                with Horizontal(classes="param-row"):
                    yield Label("Recursive:", classes="param-label")
                    yield Switch(value=recursive, id="switch-recursive")

            with Horizontal(classes="service-controls"):
                yield Button("Preview", id="btn-preview-route", variant="primary")
                yield Button("Save Route", id="btn-save-route", variant="success")
                yield Button("Cancel", id="btn-cancel-route", variant="default")

            yield Static(
                "Preview resolves paths and shows sample matching files before you save. "
                "New routes default to Draft only; promote them later when they are trustworthy.",
                classes="help-text",
            )
            yield Static("", id="route-preview", classes="help-text")

    def on_mount(self) -> None:
        self._refresh_preview()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel-route":
            self.action_cancel()
            return
        if event.button.id == "btn-preview-route":
            self._refresh_preview()
            return
        if event.button.id == "btn-save-route":
            self._save_route()

    def _refresh_preview(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        preview = app.preview_route_spec(
            taskfile=self.query_one("#input-taskfile", Input).value,
            montage=self.query_one("#input-montage", Input).value,
            ingestion_folders=self.query_one("#input-folders", Input).value.split(","),
            file_globs=self.query_one("#input-globs", Input).value.split(","),
            mode_scope=str(self.query_one("#select-scope", Select).value),
            recursive=self.query_one("#switch-recursive", Switch).value,
        )

        lines = [
            "Preview",
            f"Task file: {preview['taskfile'] or 'Missing'}",
            f"Montage: {preview['montage'] or 'Missing'}",
            f"Scope: {preview['mode_scope']}",
            "Resolved folders:",
        ]
        for folder in preview["folders"][:3]:
            lines.append(f"  - {folder}")
        if not preview["folders"]:
            lines.append("  - None yet")
        lines.append("Sample matching files:")
        for match in preview["matches"][:5]:
            lines.append(f"  - {match}")
        if not preview["matches"]:
            lines.append("  - No matching files yet")
        if preview["warnings"]:
            lines.append("Warnings:")
            for warning in preview["warnings"]:
                lines.append(f"  - {warning}")
        self.query_one("#route-preview", Static).update("\n".join(lines))

    def _save_route(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        ok, error = app.upsert_route_spec(
            route_id=self.query_one("#input-route-id", Input).value,
            existing_route_id=str(self.route_spec.get("id")) if self.is_edit_mode else None,
            taskfile=self.query_one("#input-taskfile", Input).value,
            montage=self.query_one("#input-montage", Input).value,
            ingestion_folders=self.query_one("#input-folders", Input).value.split(","),
            file_globs=self.query_one("#input-globs", Input).value.split(","),
            mode_scope=str(self.query_one("#select-scope", Select).value),
            enabled=self.query_one("#switch-enabled", Switch).value,
            recursive=self.query_one("#switch-recursive", Switch).value,
        )
        if not ok:
            self.notify(error or "Failed to save route", severity="error")
            self._refresh_preview()
            return

        self.notify("Route saved")
        self.app.pop_screen()


class RoutesScreen(Screen):
    """Routes view showing configured automation routes."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("n", "new_route", "New"),
        ("e", "edit_route", "Edit"),
        ("t", "toggle_route", "Enable/Disable"),
        ("x", "archive_route", "Archive"),
        ("p", "promote_route", "Promote"),
        ("s", "sync_routes", "Sync"),
        ("enter", "show_details", "Details"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Configured Routes", classes="section-header")
            with Horizontal(classes="filter-row"):
                yield Label("View:")
                yield Select(
                    [
                        ("Active routes", "active"),
                        ("Archived routes", "archived"),
                        ("All routes", "all"),
                    ],
                    value="active",
                    id="route-view-filter",
                )
            yield DataTable(id="routes-table", classes="routes-table")
            yield Static("", id="route-details", classes="help-text")

    def on_mount(self) -> None:
        self.set_interval(2.0, self.refresh_data)
        self.call_after_refresh(self._initialize_table)

    def _initialize_table(self) -> None:
        table = self.query_one("#routes-table", DataTable)
        if not table.columns:
            table.add_columns(
                "ID",
                "Draft",
                "Production",
                "State",
                "Enabled",
                "Priority",
                "Task",
                "Montage",
                "Folders",
            )
        table.cursor_type = "row"
        self.refresh_data()

    def on_show(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        routes = app.get_route_specs(include_archived=True)
        table = self.query_one("#routes-table", DataTable)
        if not table.columns:
            return
        table.clear()

        try:
            route_filter = self.query_one("#route-view-filter", Select).value
        except Exception:
            route_filter = "active"

        filtered = []
        for route in routes:
            archived = bool(route.get("archived", False))
            if route_filter == "active" and archived:
                continue
            if route_filter == "archived" and not archived:
                continue
            filtered.append(route)

        if not filtered:
            details = self.query_one("#route-details", Static)
            if route_filter == "archived":
                details.update(
                    "No archived routes. Use X on an active route when a workflow should "
                    "be retired without deleting its history."
                )
            else:
                details.update(
                    "No active routes yet.\n"
                    "Press N to create your first route in Draft. This screen is now the "
                    "operator path; you do not need to start with a CLI command."
                )
            return

        for route in filtered:
            modes = route.get("modes", [])
            enabled = bool(route.get("enabled", True))
            archived = bool(route.get("archived", False))
            folders = route.get("ingestion_folders", [])
            draft_str = "[green]Yes[/]" if "test" in modes else "[dim]-[/]"
            live_str = "[green]Yes[/]" if "live" in modes else "[dim]-[/]"
            state_str = "[yellow]Archived[/]" if archived else "[green]Active[/]"
            enabled_str = "[green]Yes[/]" if enabled else "[red]No[/]"
            task_label = Path(str(route.get("taskfile", ""))).name or str(route.get("taskfile", ""))
            table.add_row(
                str(route["id"]),
                draft_str,
                live_str,
                state_str,
                enabled_str,
                str(route.get("priority", 0)),
                task_label,
                str(route.get("montage", "")),
                str(len(folders)),
                key=str(route["id"]),
            )

        details = self.query_one("#route-details", Static)
        details.update(
            f"{len(filtered)} route(s) shown. Use N to create, E to edit, X to archive or restore, "
            "T to enable/disable, P to promote, and S to rebuild compiled configs."
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        self.refresh_data()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._show_route_details(event.row_key.value if event.row_key else None)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_route_details(event.row_key.value if event.row_key else None)

    def _show_route_details(self, route_id: str | None) -> None:
        if not route_id:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = app.get_route_spec(route_id)
        if not route:
            return

        details = self.query_one("#route-details", Static)
        modes = route.get("modes", [])
        folders = route.get("ingestion_folders", [])
        file_globs = route.get("file_globs", [])
        archived = bool(route.get("archived", False))
        lines = [
            f"[bold]{route['id']}[/]",
            "Draft: "
            f"{'Yes' if 'test' in modes else 'No'}  |  "
            f"Production: {'Yes' if 'live' in modes else 'No'}  |  "
            f"State: {'Archived' if archived else 'Active'}",
            f"Enabled: {'Yes' if route.get('enabled', True) else 'No'}  |  Task: {route.get('taskfile', '')}",
            f"Montage: {route.get('montage', '')}  |  Priority: {route.get('priority', 0)}",
            "Folders:",
        ]
        for folder in folders[:3]:
            lines.append(f"  - {folder}")
        if len(folders) > 3:
            lines.append(f"  ... and {len(folders) - 3} more")
        lines.append(f"File patterns: {', '.join(file_globs) if file_globs else '*'}")
        if route.get("ingestion_excludes"):
            lines.append("Excludes:")
            for entry in route["ingestion_excludes"][:3]:
                lines.append(f"  - {entry}")
        lines.append("")
        lines.append("Safe actions:")
        lines.append("  N = create a new route")
        lines.append("  E = edit this route")
        lines.append("  T = enable/disable this route")
        lines.append("  X = archive or restore this route")
        lines.append("  P = add this route to Production")
        lines.append("  S = rebuild serve-test.yaml and serve-live.yaml")
        details.update("\n".join(lines))

    def _selected_route_id(self) -> str | None:
        table = self.query_one("#routes-table", DataTable)
        if table.cursor_row is None:
            return None
        row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        if row_key is None:
            return None
        return str(row_key.value)

    def action_refresh(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_config()
        self.refresh_data()
        self.notify("Routes refreshed")

    def action_new_route(self) -> None:
        self.app.push_screen(RouteEditorScreen())

    def action_edit_route(self) -> None:
        route_id = self._selected_route_id()
        if route_id is None:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = app.get_route_spec(route_id)
        if route is None:
            self.notify("Route not found", severity="error")
            return

        self.app.push_screen(RouteEditorScreen(route))

    def action_toggle_route(self) -> None:
        route_id = self._selected_route_id()
        if route_id is None:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = app.get_route_spec(route_id)
        if route is None:
            self.notify("Route not found", severity="error")
            return
        if route.get("archived", False):
            self.notify("Restore the route before changing enabled state", severity="warning")
            return

        enabled = not bool(route.get("enabled", True))
        if app.set_route_enabled(route_id, enabled):
            self.refresh_data()
            state = "enabled" if enabled else "disabled"
            self.notify(f"Route {state}: {route_id}")
            return

        self.notify("Failed to update route", severity="error")

    def action_archive_route(self) -> None:
        route_id = self._selected_route_id()
        if route_id is None:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = app.get_route_spec(route_id)
        if route is None:
            self.notify("Route not found", severity="error")
            return

        archived = not bool(route.get("archived", False))
        if app.set_route_archived(route_id, archived):
            self.refresh_data()
            verb = "archived" if archived else "restored"
            self.notify(f"Route {verb}: {route_id}")
            return
        self.notify("Failed to change route state", severity="error")

    def action_promote_route(self) -> None:
        route_id = self._selected_route_id()
        if route_id is None:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = app.get_route_spec(route_id)
        if route is None:
            self.notify("Route not found", severity="error")
            return
        if route.get("archived", False):
            self.notify("Restore the route before promoting it", severity="warning")
            return
        if "live" in route.get("modes", []):
            self.notify("Route is already in production")
            return

        if app.promote_route(route_id):
            self.refresh_data()
            self.notify(f"Route promoted to production: {route_id}")
            return

        self.notify("Failed to promote route", severity="error")

    def action_sync_routes(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        if app.sync_route_registry():
            self.refresh_data()
            self.notify("Route registry synced")
            return
        self.notify("Failed to sync route registry", severity="error")

    def action_show_details(self) -> None:
        route_id = self._selected_route_id()
        if route_id is None:
            return

        self._show_route_details(route_id)
        self.notify("Route details shown below table")
