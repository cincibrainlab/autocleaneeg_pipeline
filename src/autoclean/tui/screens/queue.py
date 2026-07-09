"""Operator-facing work board screen for AutoClean TUI."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Select, Static

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


STATUS_STYLES = {
    "pending": "[yellow]Waiting[/]",
    "processing": "[cyan]Running[/]",
    "processed": "[green]Completed[/]",
    "failed": "[red]Needs attention[/]",
}


class QueueScreen(Screen):
    """Work board showing queued files and their status."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("y", "retry_failed", "Retry"),
        ("d", "remove_entry", "Remove"),
        ("c", "clear_processed", "Clear Done"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._confirm_key: tuple[str, str] | None = None
        self._confirm_started_at: float = 0.0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Work Board", classes="section-header")

            with Horizontal(classes="filter-row"):
                yield Label("Filter by status:")
                yield Select(
                    [
                        ("All", "all"),
                        ("Waiting", "pending"),
                        ("Running", "processing"),
                        ("Completed", "processed"),
                        ("Needs attention", "failed"),
                    ],
                    value="all",
                    id="status-filter",
                )
                yield Label("Route:")
                yield Select([("All Routes", "all")], value="all", id="route-filter")

            yield DataTable(id="queue-table", classes="queue-container")

            with Horizontal(classes="queue-actions"):
                yield Button("Retry Needs Attention", id="btn-retry", variant="warning")
                yield Button("Remove Selected", id="btn-remove", variant="error")
                yield Button("Clear Completed", id="btn-clear", variant="default")
                yield Button("Refresh", id="btn-refresh", variant="primary")

            yield Static("", id="queue-status", classes="help-text")

    def on_mount(self) -> None:
        table = self.query_one("#queue-table", DataTable)
        table.add_columns("File", "Route", "Status", "Added", "Error")
        table.cursor_type = "row"
        self.set_interval(2.0, self.refresh_data)
        self.refresh_data()

    def _populate_route_filter(self, route_ids: set[str]) -> None:
        options = [("All Routes", "all")]
        for route_id in sorted(route_ids):
            options.append((route_id, route_id))
        try:
            route_select = self.query_one("#route-filter", Select)
            route_select.set_options(options)
        except Exception:
            pass

    def _get_selected_path(self) -> str | None:
        table = self.query_one("#queue-table", DataTable)
        if table.cursor_row is None:
            return None
        try:
            row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        except Exception:
            return None
        if row_key is None:
            return None
        return str(row_key.value)

    def _confirm(self, action: str, target: str) -> bool:
        now = monotonic()
        token = (action, target)
        if self._confirm_key == token and (now - self._confirm_started_at) <= 5.0:
            self._confirm_key = None
            self._confirm_started_at = 0.0
            return True
        self._confirm_key = token
        self._confirm_started_at = now
        self.notify(
            f"Press again within 5 seconds to confirm {action}: {target}",
            severity="warning",
        )
        return False

    def refresh_data(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        entries = app.get_queue_entries()

        try:
            status_filter = self.query_one("#status-filter", Select).value
            route_filter = self.query_one("#route-filter", Select).value
        except Exception:
            status_filter = "all"
            route_filter = "all"

        route_ids = {
            str(data.get("route_id"))
            for data in entries.values()
            if data.get("route_id")
        }
        self._populate_route_filter(route_ids)

        table = self.query_one("#queue-table", DataTable)
        table.clear()

        if not entries:
            status = self.query_one("#queue-status", Static)
            status.update(f"No work items yet in the {app.get_mode_label()} lane.")
            return

        stats = {"pending": 0, "processing": 0, "processed": 0, "failed": 0}

        for path_str, data in sorted(
            entries.items(),
            key=lambda item: str(item[1].get("added_at") or ""),
            reverse=True,
        ):
            entry_status = str(data.get("status", "pending"))
            entry_route = str(data.get("route_id", "") or "")
            if entry_status in stats:
                stats[entry_status] += 1

            if status_filter != "all" and entry_status != status_filter:
                continue
            if route_filter != "all" and entry_route != route_filter:
                continue

            file_name = Path(path_str).name
            added_at = str(data.get("added_at", ""))
            if "T" in added_at:
                added_at = added_at.split("T")[1][:8]
            error = str(data.get("last_error") or "")
            if error and len(error) > 50:
                error = error[:47] + "..."

            table.add_row(
                file_name,
                entry_route or "-",
                STATUS_STYLES.get(entry_status, entry_status),
                added_at or "-",
                error or "-",
                key=path_str,
            )

        status_widget = self.query_one("#queue-status", Static)
        status_widget.update(
            f"Waiting: {stats['pending']}  |  "
            f"Running: {stats['processing']}  |  "
            f"Completed: {stats['processed']}  |  "
            f"Needs attention: {stats['failed']}  |  "
            f"Total: {len(entries)}"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        self.refresh_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
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
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_queue()
        self.refresh_data()
        self.notify("Work board refreshed")

    def action_retry_failed(self) -> None:
        app: AutoCleanTUI = self.app  # type: ignore
        queue_path = app.get_queue_path()
        if queue_path is None or not queue_path.exists():
            self.notify("Queue file not found", severity="error")
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            retried = 0
            for data in queue.entries().values():
                if data.get("status") == "failed":
                    data["status"] = "pending"
                    data.pop("last_error", None)
                    data.pop("failed_at", None)
                    retried += 1
            if retried:
                queue.save()
                app._load_queue()
                self.refresh_data()
                self.notify(f"Moved {retried} item(s) back to Waiting")
                return
            self.notify("No items need attention right now")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")

    def action_remove_entry(self) -> None:
        path_str = self._get_selected_path()
        if not path_str:
            self.notify("No work item selected", severity="warning")
            return
        if not self._confirm("remove", Path(path_str).name):
            return

        app: AutoCleanTUI = self.app  # type: ignore
        queue_path = app.get_queue_path()
        if queue_path is None:
            self.notify("Queue file not found", severity="error")
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            removed = queue.entries().pop(path_str, None)
            if removed is None:
                self.notify("Selected work item was not found", severity="warning")
                return
            queue.save()
            app._load_queue()
            self.refresh_data()
            self.notify(f"Removed {Path(path_str).name}")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")

    def action_clear_processed(self) -> None:
        if not self._confirm("clear completed items", "Completed"):
            return

        app: AutoCleanTUI = self.app  # type: ignore
        queue_path = app.get_queue_path()
        if queue_path is None:
            self.notify("Queue file not found", severity="error")
            return

        try:
            from autoclean.utils.ingestion import IngestionQueue

            queue = IngestionQueue(queue_path)
            entries = queue.entries()
            to_remove = [
                path
                for path, data in entries.items()
                if data.get("status") == "processed"
            ]
            for path in to_remove:
                del entries[path]
            if to_remove:
                queue.save()
                app._load_queue()
                self.refresh_data()
                self.notify(f"Cleared {len(to_remove)} completed item(s)")
                return
            self.notify("No completed items to clear")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")
