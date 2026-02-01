"""Stats bar widget for displaying queue statistics."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class StatItem(Static):
    """A single statistic item."""

    value = reactive(0)
    label = reactive("")
    style_class = reactive("")

    def __init__(
        self,
        value: int = 0,
        label: str = "",
        style_class: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.label = label
        self.style_class = style_class

    def render(self) -> str:
        return f"[bold]{self.value}[/]\n{self.label}"

    def on_mount(self) -> None:
        self.add_class("stat-item")
        if self.style_class:
            self.add_class(self.style_class)


class StatsBar(Widget):
    """A horizontal bar displaying queue statistics."""

    pending = reactive(0)
    ready = reactive(0)
    running = reactive(0)
    failed = reactive(0)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield StatItem(
                self.pending,
                "Pending",
                "pending",
                id="stat-pending",
            )
            yield StatItem(
                self.ready,
                "Ready",
                "ready",
                id="stat-ready",
            )
            yield StatItem(
                self.running,
                "Running",
                "running",
                id="stat-running",
            )
            yield StatItem(
                self.failed,
                "Failed",
                "failed",
                id="stat-failed",
            )

    def watch_pending(self, value: int) -> None:
        try:
            item = self.query_one("#stat-pending", StatItem)
            item.value = value
        except Exception:
            pass

    def watch_ready(self, value: int) -> None:
        try:
            item = self.query_one("#stat-ready", StatItem)
            item.value = value
        except Exception:
            pass

    def watch_running(self, value: int) -> None:
        try:
            item = self.query_one("#stat-running", StatItem)
            item.value = value
        except Exception:
            pass

    def watch_failed(self, value: int) -> None:
        try:
            item = self.query_one("#stat-failed", StatItem)
            item.value = value
        except Exception:
            pass

    def update_stats(
        self,
        pending: int = 0,
        ready: int = 0,
        running: int = 0,
        failed: int = 0,
    ) -> None:
        """Update all statistics at once."""
        self.pending = pending
        self.ready = ready
        self.running = running
        self.failed = failed
