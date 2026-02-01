"""Route tree widget for displaying automation routes."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Tree


class RouteTree(Widget):
    """A tree widget for displaying automation routes and their details."""

    def __init__(self, routes: list[Any] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._routes = routes or []

    def compose(self) -> ComposeResult:
        yield Tree("Routes", id="route-tree")

    def on_mount(self) -> None:
        """Build the tree on mount."""
        self.refresh_routes(self._routes)

    def refresh_routes(self, routes: list[Any]) -> None:
        """Refresh the tree with new routes data."""
        self._routes = routes
        tree = self.query_one("#route-tree", Tree)
        tree.clear()

        if not routes:
            tree.root.add_leaf("No routes configured")
            return

        for route in routes:
            # Route node
            enabled_icon = "[green]+[/]" if route.enabled else "[red]-[/]"
            route_label = f"{enabled_icon} {route.id}"
            route_node = tree.root.add(route_label, expand=False)

            # Route details
            route_node.add_leaf(f"Task: {route.taskfile}")
            route_node.add_leaf(f"Montage: {route.montage}")
            route_node.add_leaf(f"Priority: {route.priority}")

            if route.version:
                route_node.add_leaf(f"Version: {route.version}")

            # Folders subtree
            if route.ingestion_folders:
                folders_node = route_node.add("Ingestion Folders", expand=False)
                for folder in route.ingestion_folders:
                    folders_node.add_leaf(str(folder))

            # File patterns
            if route.file_globs:
                patterns_node = route_node.add("File Patterns", expand=False)
                for pattern in route.file_globs:
                    patterns_node.add_leaf(pattern)

            # Settings
            settings_node = route_node.add("Settings", expand=False)
            settings_node.add_leaf(f"Recursive: {route.recursive}")
            settings_node.add_leaf(f"Sentinel: {route.sentinel_ext}")
            settings_node.add_leaf(f"Workspace: {route.workspace_name}")

    def expand_all(self) -> None:
        """Expand all nodes in the tree."""
        tree = self.query_one("#route-tree", Tree)
        for node in tree.root.children:
            node.expand()
            for child in node.children:
                child.expand()

    def collapse_all(self) -> None:
        """Collapse all nodes in the tree."""
        tree = self.query_one("#route-tree", Tree)
        for node in tree.root.children:
            node.collapse()
