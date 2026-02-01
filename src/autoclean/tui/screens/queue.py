"""Queue monitor screen for AutoClean TUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Select, Static

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class QueueScreen(Screen):
    """Queue monitor showing queued files and their status."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("y", "retry_failed", "Retry"),
        ("d", "remove_entry", "Remove"),
        ("c", "clear_processed", "Clear Done"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Queue Monitor", classes="section-header")

            # Filter row
            with Horizontal(classes="filter-row"):
                yield Label("Filter by status:")
                yield Select(
                    [
                        ("All", "all"),
                        ("Pending", "pending"),
                        ("Processed", "processed"),
                        ("Failed", "failed"),
                    ],
                    value="all",
                    id="status-filter",
                )
                yield Label("Route:")
                yield Select(
                    [("All Routes", "all")],
                    value="all",
                    id="route-filter",
                )

            # Queue table
            yield DataTable(id="queue-table", classes="queue-container")

            # Actions row
            with Horizontal(classes="queue-actions"):
                yield Button("Retry Failed", id="btn-retry", variant="warning")
                yield Button("Remove Selected", id="btn-remove", variant="error")
                yield Button("Clear Processed", id="btn-clear", variant="default")
                yield Button("Refresh", id="btn-refresh", variant="primary")

            # Details/status
            yield Static("", id="queue-status", classes="help-text")

    def on_mount(self) -> None:
        """Initialize the queue table."""
        table = self.query_one("#queue-table", DataTable)
        table.add_columns(
            "File",
            "Route",
            "Status",
            "Added",
            "Error",
        )
        table.cursor_type = "row"
        self._populate_route_filter()
        self.refresh_data()

    def _populate_route_filter(self) -> None:
        """Populate the route filter with available routes."""
        app: AutoCleanTUI = self.app  # type: ignore
        routes = app.get_routes()

        options = [("All Routes", "all")]
        for route in routes:
            options.append((route.id, route.id))

        try:
            route_select = self.query_one("#route-filter", Select)
            route_select.set_options(options)
        except Exception:
            pass

    def refresh_data(self) -> None:
        """Refresh queue data from app state."""
        app: AutoCleanTUI = self.app  # type: ignore
        entries = app.get_queue_entries()

        # Get filter values
        try:
            status_filter = self.query_one("#status-filter", Select).value
            route_filter = self.query_one("#route-filter", Select).value
        except Exception:
            status_filter = "all"
            route_filter = "all"

        table = self.query_one("#queue-table", DataTable)
        table.clear()

        if not entries:
            status = self.query_one("#queue-status", Static)
            status.update("Queue is empty")
            return

        # Stats
        stats = {"pending": 0, "processed": 0, "failed": 0}

        for path_str, data in entries.items():
            entry_status = data.get("status", "pending")
            entry_route = data.get("route_id", "")

            # Update stats
            if entry_status in stats:
                stats[entry_status] += 1

            # Apply filters
            if status_filter != "all" and entry_status != status_filter:
                continue
            if route_filter != "all" and entry_route != route_filter:
                continue

            # Format data
            file_name = Path(path_str).name
            added_at = data.get("added_at", "")
            if added_at:
                # Truncate timestamp to just time
                if "T" in added_at:
                    added_at = added_at.split("T")[1][:8]

            error = data.get("last_error", "")
            if error and len(error) > 30:
                error = error[:27] + "..."

            # Status styling
            status_display = entry_status
            if entry_status == "pending":
                status_display = "[yellow]pending[/]"
            elif entry_status == "processed":
                status_display = "[green]processed[/]"
            elif entry_status == "failed":
                status_display = "[red]failed[/]"

            table.add_row(
                file_name,
                entry_route or "-",
                status_display,
                added_at,
                error or "-",
                key=path_str,
            )

        # Update status
        status_widget = self.query_one("#queue-status", Static)
        status_widget.update(
            f"Pending: {stats['pending']}  |  "
            f"Processed: {stats['processed']}  |  "
            f"Failed: {stats['failed']}  |  "
            f"Total: {len(entries)}"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        self.refresh_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-retry":
            self.action_retry_failed()
        elif button_id == "btn-remove":
            self.action_remove_entry()
        elif button_id == "btn-clear":
            self.action_clear_processed()
        elif button_id == "btn-refresh":
            self.action_refresh()

    def action_refresh(self) -> None:
        """Refresh queue data."""
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_queue()
        self.refresh_data()
        self.notify("Queue refreshed")

    def action_retry_failed(self) -> None:
        """Retry all failed entries."""
        app: AutoCleanTUI = self.app  # type: ignore

        if not app.state.workspace_dir:
            self.notify("No workspace configured", severity="error")
            return

        queue_path = app.state.workspace_dir / "queue.json"
        if not queue_path.exists():
            self.notify("Queue file not found", severity="error")
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            entries = queue.entries()

            retried = 0
            for path_str, data in entries.items():
                if data.get("status") == "failed":
                    data["status"] = "pending"
                    data.pop("last_error", None)
                    data.pop("failed_at", None)
                    retried += 1

            if retried > 0:
                queue.save()
                app._load_queue()
                self.refresh_data()
                self.notify(f"Retried {retried} failed entries")
            else:
                self.notify("No failed entries to retry")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")

    def action_remove_entry(self) -> None:
        """Remove the selected entry from the queue."""
        table = self.query_one("#queue-table", DataTable)
        if table.cursor_row is None:
            self.notify("No entry selected", severity="warning")
            return

        # Get the row key from the cursor coordinate
        try:
            row_key = table.coordinate_to_cell_key(
                (table.cursor_row, 0)
            ).row_key
            path_str = str(row_key.value) if row_key else None
        except Exception:
            self.notify("Could not get selected entry", severity="error")
            return

        if not path_str:
            self.notify("No entry selected", severity="warning")
            return

        app: AutoCleanTUI = self.app  # type: ignore

        if not app.state.workspace_dir:
            self.notify("No workspace configured", severity="error")
            return

        queue_path = app.state.workspace_dir / "queue.json"
        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            entries = queue.entries()

            if path_str in entries:
                del entries[path_str]
                queue.save()
                app._load_queue()
                self.refresh_data()
                self.notify("Entry removed")
            else:
                self.notify("Entry not found", severity="warning")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")

    def action_clear_processed(self) -> None:
        """Clear all processed entries from the queue."""
        app: AutoCleanTUI = self.app  # type: ignore

        if not app.state.workspace_dir:
            self.notify("No workspace configured", severity="error")
            return

        queue_path = app.state.workspace_dir / "queue.json"
        if not queue_path.exists():
            self.notify("Queue file not found", severity="error")
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            entries = queue.entries()

            to_remove = [
                path for path, data in entries.items()
                if data.get("status") == "processed"
            ]

            for path in to_remove:
                del entries[path]

            if to_remove:
                queue.save()
                app._load_queue()
                self.refresh_data()
                self.notify(f"Cleared {len(to_remove)} processed entries")
            else:
                self.notify("No processed entries to clear")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")
