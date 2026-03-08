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
    """Simple create/edit screen for route specs."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, route_spec: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.route_spec = route_spec or {}

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
        heading = "Edit Route" if route_id else "Create Route"

        with Vertical():
            yield Static(heading, classes="section-header")
            yield Static(
                "This form saves back into the route registry and recompiles the Draft "
                "and Production configs. Use one route for one workflow.",
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
                yield Button("Save Route", id="btn-save-route", variant="success")
                yield Button("Cancel", id="btn-cancel-route", variant="default")

            yield Static(
                "Folders and globs use comma-separated values. Create in Draft first, "
                "then switch to Draft + Production when the route is trustworthy.",
                classes="help-text",
            )

    def action_cancel(self) -> None:
        """Close the editor without saving."""
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle save/cancel actions."""
        if event.button.id == "btn-cancel-route":
            self.action_cancel()
            return
        if event.button.id == "btn-save-route":
            self._save_route()

    def _save_route(self) -> None:
        """Persist the route via the shared route registry backend."""
        app: AutoCleanTUI = self.app  # type: ignore
        ok, error = app.upsert_route_spec(
            route_id=self.query_one("#input-route-id", Input).value,
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
        ("p", "promote_route", "Promote"),
        ("s", "sync_routes", "Sync"),
        ("enter", "show_details", "Details"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Configured Routes", classes="section-header")
            yield DataTable(id="routes-table", classes="routes-table")
            yield Static("", id="route-details", classes="help-text")

    def on_mount(self) -> None:
        """Initialize the routes table."""
        table = self.query_one("#routes-table", DataTable)
        table.add_columns(
            "ID",
            "Draft",
            "Production",
            "Enabled",
            "Priority",
            "Task",
            "Montage",
            "Folders",
            "Globs",
        )
        table.cursor_type = "row"
        self.refresh_data()

    def on_show(self) -> None:
        """Refresh whenever the route screen becomes active again."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh routes data from app state."""
        app: AutoCleanTUI = self.app  # type: ignore
        routes = app.get_route_specs()

        table = self.query_one("#routes-table", DataTable)
        table.clear()

        if not routes:
            # Show empty state message
            details = self.query_one("#route-details", Static)
            details.update(
                "No routes configured yet.\n"
                "Create one with `serve route upsert ...` and this screen will let you"
                " toggle and promote it without hand-editing YAML."
            )
            return

        for route in routes:
            modes = route.get("modes", [])
            enabled = bool(route.get("enabled", True))
            folders = route.get("ingestion_folders", [])
            file_globs = route.get("file_globs", [])
            enabled_str = "[green]Yes[/]" if enabled else "[red]No[/]"
            draft_str = "[green]Yes[/]" if "test" in modes else "[dim]-[/]"
            live_str = "[green]Yes[/]" if "live" in modes else "[dim]-[/]"
            folders_str = str(len(folders))
            globs_str = ", ".join(file_globs[:2])
            if len(file_globs) > 2:
                globs_str += f" (+{len(file_globs) - 2})"
            task_label = Path(str(route.get("taskfile", ""))).name or str(
                route.get("taskfile", "")
            )

            table.add_row(
                str(route["id"]),
                draft_str,
                live_str,
                enabled_str,
                str(route.get("priority", 0)),
                task_label,
                str(route.get("montage", "")),
                folders_str,
                globs_str,
                key=str(route["id"]),
            )

        # Update details
        details = self.query_one("#route-details", Static)
        details.update(
            f"{len(routes)} route(s) configured. "
            "Use N to create, E to edit, T to enable/disable, P to promote, "
            "and S to rebuild compiled configs."
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show details when a row is selected."""
        self._show_route_details(event.row_key.value if event.row_key else None)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show details when a row is highlighted."""
        self._show_route_details(event.row_key.value if event.row_key else None)

    def _show_route_details(self, route_id: str | None) -> None:
        """Display detailed information about a route."""
        if not route_id:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        routes = app.get_route_specs()

        route = None
        for r in routes:
            if r["id"] == route_id:
                route = r
                break

        if not route:
            return

        details = self.query_one("#route-details", Static)

        # Build details text
        modes = route.get("modes", [])
        folders = route.get("ingestion_folders", [])
        file_globs = route.get("file_globs", [])
        lines = [
            f"[bold]{route['id']}[/]",
            "Draft: "
            f"{'Yes' if 'test' in modes else 'No'}  |  "
            f"Production: {'Yes' if 'live' in modes else 'No'}  |  "
            f"Enabled: {'Yes' if route.get('enabled', True) else 'No'}",
            f"Task: {route.get('taskfile', '')}  |  Montage: {route.get('montage', '')}",
            "Priority: "
            f"{route.get('priority', 0)}  |  "
            f"Recursive: {'Yes' if route.get('recursive', True) else 'No'}  |  "
            f"Sentinel: {route.get('sentinel_ext', '.ready')}",
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

        if route.get("automation_root"):
            lines.append(f"Automation root: {route['automation_root']}")
        if route.get("workspace_name"):
            lines.append(f"Workspace name: {route['workspace_name']}")
        if route.get("version"):
            lines.append(f"Version: {route['version']}")
        lines.append("")
        lines.append("Safe actions:")
        lines.append("  N = create a new route")
        lines.append("  E = edit this route")
        lines.append("  T = enable/disable this route")
        lines.append("  P = add this route to production")
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
        """Refresh routes data."""
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_config()
        self.refresh_data()
        self.notify("Routes refreshed")

    def action_new_route(self) -> None:
        """Open the create-route flow."""
        self.app.push_screen(RouteEditorScreen())

    def action_edit_route(self) -> None:
        """Open the editor for the selected route."""
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
        """Toggle route enabled status using the route registry backend."""
        route_id = self._selected_route_id()
        if route_id is None:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = next((spec for spec in app.get_route_specs() if spec["id"] == route_id), None)
        if route is None:
            self.notify("Route not found", severity="error")
            return

        enabled = not bool(route.get("enabled", True))
        if app.set_route_enabled(route_id, enabled):
            self.refresh_data()
            state = "enabled" if enabled else "disabled"
            self.notify(f"Route {state}: {route_id}")
            return

        self.notify("Failed to update route", severity="error")

    def action_promote_route(self) -> None:
        """Promote selected route into production."""
        route_id = self._selected_route_id()
        if route_id is None:
            return

        app: AutoCleanTUI = self.app  # type: ignore
        route = next((spec for spec in app.get_route_specs() if spec["id"] == route_id), None)
        if route is None:
            self.notify("Route not found", severity="error")
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
        """Rebuild compiled serve configs from route specs."""
        app: AutoCleanTUI = self.app  # type: ignore
        if app.sync_route_registry():
            self.refresh_data()
            self.notify("Route registry synced")
            return
        self.notify("Failed to sync route registry", severity="error")

    def action_show_details(self) -> None:
        """Show expanded details for selected route."""
        route_id = self._selected_route_id()
        if route_id is None:
            return

        self._show_route_details(route_id)
        self.notify("Route details shown below table")
