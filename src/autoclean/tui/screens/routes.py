"""Routes view screen for AutoClean TUI."""

from __future__ import annotations

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
        ("t", "toggle_route", "Toggle"),
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
        routes = app.get_routes()

        table = self.query_one("#routes-table", DataTable)
        table.clear()

        if not routes:
            # Show empty state message
            details = self.query_one("#route-details", Static)
            details.update("No routes configured. Check your serve configuration.")
            return

        for route in routes:
            enabled_str = "[green]Yes[/]" if route.enabled else "[red]No[/]"
            folders_str = str(len(route.ingestion_folders))
            globs_str = ", ".join(route.file_globs[:2])
            if len(route.file_globs) > 2:
                globs_str += f" (+{len(route.file_globs) - 2})"

            table.add_row(
                route.id,
                enabled_str,
                str(route.priority),
                route.taskfile,
                route.montage,
                folders_str,
                globs_str,
                key=route.id,
            )

        # Update details
        details = self.query_one("#route-details", Static)
        details.update(f"{len(routes)} route(s) configured. Select a row for details.")

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
        routes = app.get_routes()

        route = None
        for r in routes:
            if r.id == route_id:
                route = r
                break

        if not route:
            return

        details = self.query_one("#route-details", Static)

        # Build details text
        lines = [
            f"[bold]{route.id}[/]",
            f"Task: {route.taskfile}  |  Montage: {route.montage}",
            f"Priority: {route.priority}  |  Recursive: {'Yes' if route.recursive else 'No'}  |  Sentinel: {route.sentinel_ext}",
            "Folders:",
        ]

        for folder in route.ingestion_folders[:3]:
            lines.append(f"  - {folder}")
        if len(route.ingestion_folders) > 3:
            lines.append(f"  ... and {len(route.ingestion_folders) - 3} more")

        lines.append(f"File patterns: {', '.join(route.file_globs)}")

        if route.version:
            lines.append(f"Version: {route.version}")

        details.update("\n".join(lines))

    def action_refresh(self) -> None:
        """Refresh routes data."""
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_config()
        self.refresh_data()
        self.notify("Routes refreshed")

    def action_toggle_route(self) -> None:
        """Toggle route enabled status (display only - actual toggle requires config edit)."""
        table = self.query_one("#routes-table", DataTable)
        if table.cursor_row is None:
            return

        self.notify(
            "To toggle routes, edit the serve YAML configuration file",
            severity="warning",
        )

    def action_show_details(self) -> None:
        """Show expanded details for selected route."""
        table = self.query_one("#routes-table", DataTable)
        if table.cursor_row is None:
            return

        row_key = table.get_row_at(table.cursor_row)
        if row_key:
            # Already showing in details pane
            self.notify("Route details shown below table")
