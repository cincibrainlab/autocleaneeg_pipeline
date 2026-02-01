"""Configuration viewer screen for AutoClean TUI."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from autoclean.tui.app import AutoCleanTUI


class ConfigScreen(Screen):
    """Configuration panel for viewing and validating YAML config."""

    BINDINGS = [
        ("v", "validate", "Validate"),
        ("d", "deploy", "Deploy"),
        ("e", "open_editor", "Edit"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Configuration", classes="section-header")

            # Config status
            yield Static("", id="config-status", classes="config-status")

            # YAML content viewer
            with VerticalScroll(id="yaml-scroll", classes="config-yaml"):
                yield Static("", id="yaml-content")

            # Actions row
            with Horizontal(classes="config-actions"):
                yield Button("Validate", id="btn-validate", variant="primary")
                yield Button("Deploy", id="btn-deploy", variant="success")
                yield Button("Open in Editor", id="btn-editor", variant="default")
                yield Button("Refresh", id="btn-refresh", variant="default")

            # Errors/warnings section
            yield Static("", id="config-messages", classes="help-text")

    def on_mount(self) -> None:
        """Initialize config view."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh config data from app state."""
        app: AutoCleanTUI = self.app  # type: ignore

        # Update status
        status_widget = self.query_one("#config-status", Static)
        if app.state.config_valid:
            status_widget.update(
                f"[bold green]Configuration Valid[/] - Mode: {app.state.mode}"
            )
            status_widget.remove_class("invalid")
            status_widget.add_class("valid")
        else:
            status_widget.update(
                f"[bold red]Configuration Invalid[/] - Mode: {app.state.mode}"
            )
            status_widget.remove_class("valid")
            status_widget.add_class("invalid")

        # Load and display YAML
        yaml_content = app.get_config_yaml()
        yaml_widget = self.query_one("#yaml-content", Static)

        # Basic syntax highlighting for YAML
        highlighted = self._highlight_yaml(yaml_content)
        yaml_widget.update(highlighted)

        # Show errors/warnings
        messages = self.query_one("#config-messages", Static)
        lines = []

        if app.state.config_errors:
            lines.append("[bold red]Errors:[/]")
            for error in app.state.config_errors:
                lines.append(f"  [red]- {error}[/]")

        if app.state.config_warnings:
            lines.append("[bold yellow]Warnings:[/]")
            for warning in app.state.config_warnings:
                lines.append(f"  [yellow]- {warning}[/]")

        if not lines:
            if app.state.config_valid:
                lines.append("[green]No errors or warnings[/]")

        messages.update("\n".join(lines))

    def _highlight_yaml(self, content: str) -> str:
        """Apply basic syntax highlighting to YAML content."""
        lines = []
        for line in content.split("\n"):
            # Comments
            if line.strip().startswith("#"):
                lines.append(f"[dim italic]{line}[/]")
                continue

            # Key-value pairs
            if ":" in line and not line.strip().startswith("-"):
                parts = line.split(":", 1)
                key = parts[0]
                value = parts[1] if len(parts) > 1 else ""

                # Highlight key
                highlighted_key = f"[cyan]{key}[/]:"

                # Highlight value based on type
                value_stripped = value.strip()
                if value_stripped.lower() in ("true", "false"):
                    highlighted_value = f"[magenta]{value}[/]"
                elif value_stripped.isdigit() or (
                    value_stripped.replace(".", "").replace("-", "").isdigit()
                ):
                    highlighted_value = f"[yellow]{value}[/]"
                elif value_stripped.startswith('"') or value_stripped.startswith("'"):
                    highlighted_value = f"[green]{value}[/]"
                elif value_stripped.startswith("[") or value_stripped.startswith("{"):
                    highlighted_value = f"[white]{value}[/]"
                else:
                    highlighted_value = value

                lines.append(f"{highlighted_key}{highlighted_value}")
            # List items
            elif line.strip().startswith("-"):
                indent = len(line) - len(line.lstrip())
                item = line.strip()
                lines.append(f"{' ' * indent}[blue]{item}[/]")
            else:
                lines.append(line)

        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-validate":
            self.action_validate()
        elif button_id == "btn-deploy":
            self.action_deploy()
        elif button_id == "btn-editor":
            self.action_open_editor()
        elif button_id == "btn-refresh":
            self.action_refresh()

    def action_validate(self) -> None:
        """Validate the configuration."""
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_config()
        self.refresh_data()

        if app.state.config_valid:
            if app.state.config_warnings:
                self.notify(
                    f"Config valid with {len(app.state.config_warnings)} warning(s)",
                    severity="warning",
                )
            else:
                self.notify("Configuration is valid", severity="information")
        else:
            self.notify(
                f"Configuration has {len(app.state.config_errors)} error(s)",
                severity="error",
            )

    def action_deploy(self) -> None:
        """Deploy configuration from operator to deployed."""
        app: AutoCleanTUI = self.app  # type: ignore

        if not app.state.workspace_dir:
            self.notify("No workspace configured", severity="error")
            return

        if not app.state.config_valid:
            self.notify("Cannot deploy invalid configuration", severity="error")
            return

        source = app.state.workspace_dir / f"serve-{app.state.mode}.yaml"
        deploy_dir = app.state.workspace_dir / "deploy"
        target = deploy_dir / f"serve-{app.state.mode}.yaml"

        try:
            deploy_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source, target)
            self.notify(f"Configuration deployed to {target.name}", severity="information")
            app._add_activity_event(
                "info",
                f"Configuration deployed: {app.state.mode}",
            )
        except Exception as exc:
            self.notify(f"Deploy failed: {exc}", severity="error")

    def action_open_editor(self) -> None:
        """Open configuration in external editor."""
        app: AutoCleanTUI = self.app  # type: ignore

        if not app.state.workspace_dir:
            self.notify("No workspace configured", severity="error")
            return

        config_file = app.state.workspace_dir / f"serve-{app.state.mode}.yaml"

        if not config_file.exists():
            self.notify("Config file not found", severity="error")
            return

        # Get editor from environment or use defaults
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            if sys.platform == "darwin":
                editor = "open"
            elif sys.platform.startswith("win"):
                editor = "notepad"
            else:
                editor = "xdg-open"

        try:
            subprocess.Popen([editor, str(config_file)])
            self.notify(f"Opening {config_file.name} in {editor}")
        except Exception as exc:
            self.notify(f"Failed to open editor: {exc}", severity="error")

    def action_refresh(self) -> None:
        """Refresh configuration view."""
        app: AutoCleanTUI = self.app  # type: ignore
        app._load_config()
        self.refresh_data()
        self.notify("Configuration refreshed")
