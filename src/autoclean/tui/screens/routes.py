"""Routes view screen for AutoClean TUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class RoutesScreen(Screen):
    """Routes view showing configured automation routes."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
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
            "Use T to enable/disable, P to promote a draft route to production, "
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
