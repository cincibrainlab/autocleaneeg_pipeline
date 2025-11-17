#!/usr/bin/env python3
"""
AutoClean EEG Pipeline - Command Line Interface

This module provides a flexible CLI for AutoClean that works both as a
standalone tool (via uv tool) and within development environments.
"""

import argparse
import csv
import json
import keyword
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from rich.panel import Panel
from rich.table import Table

# Block management commands
from autoclean import __version__, cli_blocks
from autoclean.utils.audit import verify_access_log_integrity
from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled
from autoclean.utils.builtins import BuiltinRegistry
from autoclean.utils.cli_display import CLIDisplay
from autoclean.utils.config import (
    disable_compliance_mode,
    enable_compliance_mode,
    get_compliance_status,
    load_user_config,
    save_user_config,
)
from autoclean.utils.console import get_console, make_console
from autoclean.utils.database import DB_PATH
from autoclean.utils.file_system import update_status_marker
from autoclean.utils.logging import has_logged_errors, message
from autoclean.utils.montage import load_valid_montages
from autoclean.utils.montage_validation import (
    analyze_channels,
    create_3d_plot,
    create_stats_plot,
    extract_channel_info,
    extract_file_metadata,
    generate_html_report,
    suggest_montages,
)
from autoclean.utils.task_discovery import (
    extract_config_from_task,
    get_task_by_name,
    get_task_overrides,
    safe_discover_tasks,
)
from autoclean.utils.template_renderer import (
    render_template,
    validate_python_identifier,
)
from autoclean.utils.user_config import user_config

# Optional Neo support for extended formats (XDAT)
try:
    import neo
    import neo.io

    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False

# ------------------------------------------------------------
# CLI Process Logging
# ------------------------------------------------------------

# Maximum log file size (5MB)
MAX_LOG_SIZE = 5 * 1024 * 1024


def _strip_wrapping_quotes(text: Optional[str]) -> Optional[str]:
    """Remove a single or nested pair of matching wrapping quotes from text.

    Handles common copy/paste cases like "'/Users/me/My Folder'" or '"/path"'.
    Returns None unchanged and leaves interior quotes untouched.
    """
    if text is None:
        return None
    s = text.strip()
    # Remove up to two layers of matching quotes (single or double)
    for _ in range(2):
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        else:
            break
    return s


def _sanitize_arguments(args: List[str]) -> List[str]:
    """Sanitize command-line arguments to remove sensitive information."""
    sanitized = []

    # Patterns for sensitive information
    sensitive_patterns = [
        # File paths with potentially sensitive directory names
        r"(/[Uu]sers?/[^/]+/[Dd]esktop|/[Uu]sers?/[^/]+/[Dd]ocuments)",
        r"(/home/[^/]+/[Dd]esktop|/home/[^/]+/[Dd]ocuments)",
        # API tokens and keys
        r"(--?(?:token|key|password|pass|secret)(?:=|\s+)\S+)",
        # Email addresses
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    ]

    for arg in args:
        sanitized_arg = arg
        for pattern in sensitive_patterns:
            if re.search(pattern, arg, re.IGNORECASE):
                # Replace file paths with just the filename
                if "/" in arg or "\\" in arg:
                    path_obj = Path(arg)
                    sanitized_arg = (
                        f"[REDACTED]/{path_obj.name}" if path_obj.name else "[REDACTED]"
                    )
                else:
                    # For tokens/keys, show only the parameter name
                    if "=" in arg:
                        param_name = arg.split("=")[0]
                        sanitized_arg = f"{param_name}=[REDACTED]"
                    else:
                        sanitized_arg = "[REDACTED]"
                break

        sanitized.append(sanitized_arg)

    return sanitized


def _pretty_timestamp(value: Optional[str], *, default: str = "not yet checked") -> str:
    """Render ISO timestamps in a user-friendly format."""

    if not value or value in {"never", "not yet checked"}:
        return default
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    local_dt = dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M %Z")


def _rotate_log(log_path: Path) -> None:
    """Rotate log file when it gets too large."""
    try:
        # Keep last 5 rotated logs
        for i in range(4, 0, -1):
            old_path = log_path.with_suffix(f".{i}.txt")
            new_path = log_path.with_suffix(f".{i + 1}.txt")
            if old_path.exists():
                old_path.rename(new_path)

        # Move current log to .1
        if log_path.exists():
            rotated_path = log_path.with_suffix(".1.txt")
            log_path.rename(rotated_path)
    except Exception:
        # If rotation fails, truncate the log
        try:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log rotated due to size limit\n"
                )
        except Exception:
            pass


def _log_cli_execution(args: argparse.Namespace) -> None:
    """Log CLI execution to workspace process log with security and error handling."""
    try:
        # Only log if workspace exists to avoid setup errors
        workspace_dir = user_config.config_dir
        if not workspace_dir.exists():
            return

        log_path = workspace_dir / "process_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get original command arguments, excluding the script name
        original_args = (
            sys.argv[1:] if hasattr(sys, "argv") and len(sys.argv) > 1 else []
        )

        # Skip logging if no meaningful command (just bare invocation)
        if not original_args or not args.command:
            return

        # Sanitize arguments for security
        safe_args = _sanitize_arguments(original_args)
        command_str = f"autocleaneeg-pipeline {' '.join(safe_args)}"

        # Check file size and rotate if necessary
        if log_path.exists() and log_path.stat().st_size > MAX_LOG_SIZE:
            _rotate_log(log_path)

        # Atomic write to prevent corruption
        log_entry = f"[{timestamp}] {command_str}\n"

        # Write to temporary file first, then move
        temp_path = log_path.with_suffix(".tmp")
        try:
            # Read existing content if file exists
            existing_content = ""
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as f:
                    existing_content = f.read()

            # Write to temp file
            with temp_path.open("w", encoding="utf-8") as f:
                f.write(existing_content + log_entry)

            # Atomic move
            temp_path.replace(log_path)

        except Exception:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    except Exception as e:
        # Log to stderr but don't break CLI functionality
        print(f"Warning: Failed to log command execution: {e}", file=sys.stderr)


# ------------------------------------------------------------
# Rich help integration
# ------------------------------------------------------------
def _get_current_montage() -> Optional[str]:
    """
    Retrieve the current montage from the active task file with guard logic.
    Returns None if no task is active or montage cannot be determined.
    """
    try:
        active_task = user_config.get_active_task()
        if not active_task:
            return None

        active_task_path = user_config.get_custom_task_path(active_task)
        if not active_task_path or not active_task_path.exists():
            return None

        source = active_task_path.read_text(encoding="utf-8")
        block, _, _ = _locate_montage_block(source)
        if block:
            return _extract_montage_value(block)
        return None
    except Exception:
        return None


def _print_context(
    console,
    style: str = "centered",
    include_system_info: bool = True,
    include_disk: bool = True,
) -> None:
    """
    Unified context display function with configurable formatting.

    Args:
        console: Rich console instance
        style: Display style - 'centered' (with emojis) or 'compact' (left-aligned, info tags)
        include_system_info: Show Python/OS/Time info (centered style only)
        include_disk: Show disk space (centered style only)
    """
    from rich.align import Align
    from rich.text import Text

    # System info (centered style only)
    if include_system_info and style == "centered":
        try:
            import platform as _platform

            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            os_name = _platform.system() or "UnknownOS"
            os_rel = _platform.release() or ""
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            info = Text()
            info.append("🐍 Python ", style="muted")
            info.append(py_ver, style="accent")
            info.append("  •  ", style="muted")
            info.append("🖥 ", style="muted")
            info.append(f"{os_name} {os_rel}".strip(), style="accent")
            info.append("  •  ", style="muted")
            info.append("🕒 ", style="muted")
            info.append(now_str, style="accent")
            console.print(Align.center(info))
        except Exception:
            pass

    # Gather core configuration data
    workspace_dir = user_config.config_dir
    workspace_valid = False
    workspace_display = str(workspace_dir)
    active_task_val = None
    montage_val = None
    input_val = None
    input_type = None  # 'file', 'dir', 'missing', or None

    try:
        workspace_valid = user_config._is_workspace_valid()  # type: ignore[attr-defined]
    except Exception:
        workspace_valid = workspace_dir.exists() and (workspace_dir / "tasks").exists()

    home = str(Path.home())
    if workspace_display.startswith(home):
        workspace_display = workspace_display.replace(home, "~", 1)

    try:
        active_task_val = user_config.get_active_task()
    except Exception:
        pass

    try:
        montage_val = _get_current_montage()
    except Exception:
        pass

    try:
        input_val = user_config.get_active_source()
        if input_val:
            p = Path(input_val)
            input_display = str(p)
            if input_display.startswith(home):
                input_display = input_display.replace(home, "~", 1)
            if p.exists():
                if p.is_file():
                    input_type = "file"
                elif p.is_dir():
                    input_type = "dir"
                else:
                    input_type = "other"
            else:
                input_type = "missing"
            input_val = input_display
    except Exception:
        pass

    # Create configuration table
    from rich.table import Table

    if style == "centered":
        from rich import box

        config_table = Table(
            show_header=False, box=box.ROUNDED, padding=(0, 1), border_style="dim"
        )
        config_table.add_column("Property", style="muted", no_wrap=True)
        config_table.add_column("Value", style="accent")
    else:  # compact
        from rich import box

        config_table = Table(
            show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False
        )
        config_table.add_column("Property", style="muted", no_wrap=True)
        config_table.add_column("Value", style="accent")

    # Add workspace row
    ws_icon = "✓" if workspace_valid else "⚠"
    ws_style = "success" if workspace_valid else "warning"
    ws_label = Text()
    if style == "centered":
        ws_label.append(f"{ws_icon} ", style=ws_style)
    ws_label.append("Workspace", style="muted")
    config_table.add_row(
        ws_label,
        Text(workspace_display, style=ws_style if not workspace_valid else "accent"),
    )

    # Add active task row
    task_label = Text()
    if style == "centered":
        task_label.append("🎯 ", style="muted")
    task_label.append("Task", style="muted")
    if active_task_val:
        config_table.add_row(task_label, Text(active_task_val, style="accent"))
    else:
        config_table.add_row(task_label, Text("not set", style="warning"))

    # Add montage row
    montage_label = Text()
    if style == "centered":
        montage_label.append("📊 ", style="muted")
    montage_label.append("Montage", style="muted")
    if montage_val:
        config_table.add_row(montage_label, Text(montage_val, style="accent"))
    else:
        montage_status = "not configured" if active_task_val else "not set"
        config_table.add_row(montage_label, Text(montage_status, style="warning"))

    # Add input row
    input_label = Text()
    if style == "centered":
        if input_type == "file":
            input_label.append("📄 ", style="muted")
        elif input_type == "dir":
            input_label.append("📂 ", style="muted")
        elif input_type == "missing":
            input_label.append("⚠ ", style="warning")
        else:
            input_label.append("📁 ", style="muted")
    input_label.append("Input", style="muted")
    if input_val:
        input_style = "warning" if input_type == "missing" else "accent"
        config_table.add_row(input_label, Text(input_val, style=input_style))
    else:
        config_table.add_row(input_label, Text("not set", style="warning"))

    # Print the table
    if style == "centered":
        console.print(Align.center(config_table))
        # Task library status below table
        try:
            registry = BuiltinRegistry()
            registry_info = registry.registry_status()
            commit_raw = registry_info.get("commit")
            commit = commit_raw or "not yet synced"
            if commit in {"unknown", ""}:
                commit = "not yet synced"
            elif commit == "local-snapshot":
                commit = "local snapshot"
            synced_at = _pretty_timestamp(registry_info.get("synced_at"))
            status_line = Text()
            status_line.append("Task Library: ", style="muted")
            status_line.append(commit, style="accent")
            status_line.append("  ", style="muted")
            status_line.append("last checked ", style="muted")
            status_line.append(synced_at, style="accent")
            console.print(Align.center(status_line))

            last_error = registry_info.get("last_error")
            if isinstance(last_error, dict):
                err_text = Text()
                err_text.append("Working offline: ", style="warning")
                err_text.append(
                    last_error.get("message", "connection problem"), style="warning"
                )
                timestamp = _pretty_timestamp(
                    last_error.get("timestamp"), default="time not recorded"
                )
                if timestamp:
                    err_text.append("  ", style="muted")
                    err_text.append(timestamp, style="muted")
                console.print(Align.center(err_text))
        except Exception:
            pass
    else:  # compact
        console.print(config_table)

    # Disk space (centered style only)
    if include_disk and style == "centered":
        try:
            workspace_dir = user_config.config_dir
            usage_path = (
                workspace_dir
                if workspace_dir.exists()
                else (
                    workspace_dir.parent
                    if workspace_dir.parent.exists()
                    else Path.home()
                )
            )
            du = shutil.disk_usage(str(usage_path))
            free_gb = du.free / (1024**3)
            free_line = Text()
            free_line.append("💾 ", style="muted")
            free_line.append("Free space ", style="muted")
            free_line.append(f"{free_gb:.1f} GB", style="accent")
            console.print(Align.center(free_line))
            console.print()
        except Exception:
            pass


def _print_startup_context(console) -> None:
    """Print system info, workspace path, and free disk space (shared for header/help)."""
    _print_context(
        console, style="centered", include_system_info=True, include_disk=True
    )


def _render_task_install_summary(
    console,
    *,
    task_name: str,
    installed_path: Path,
    source_label: str,
    activated: bool,
    install_status: str,
) -> None:
    """Render a friendly summary after a task install completes."""

    status_text = "Installed" if install_status == "installed" else "Already installed"
    summary_lines = [
        f"[success]✓[/success] {status_text} [accent]{task_name}[/accent]",
        f"[muted]Source:[/muted] {source_label}",
        f"[muted]Location:[/muted] [info]{installed_path}[/info]",
    ]

    next_steps: list[str] = []
    if activated:
        next_steps.append(
            "Active now – run [accent]process <file>[/accent] to start processing"
        )
    else:
        next_steps.append(
            f"Set active with [accent]task set {task_name}[/accent] (or rerun with --activate)"
        )

    next_steps.append("Edit or review the task file if adjustments are needed")
    next_steps.append(
        "Process data via [accent]autocleaneeg-pipeline process <file>[/accent]"
    )

    steps_text = "\n".join(f"  • {step}" for step in next_steps)
    body = "\n".join(summary_lines) + "\n\n[header]Next steps[/header]\n" + steps_text

    console.print()
    console.print(
        Panel(
            body,
            title="[title]Task Install[/title]",
            border_style="border",
            padding=(1, 2),
        )
    )
    console.print()


def _render_input_summary(console, *, input_path: Path) -> None:
    """Show a concise panel after setting an active input path."""

    exists = input_path.exists()
    if exists and input_path.is_file():
        path_kind = "File"
    elif exists and input_path.is_dir():
        path_kind = "Folder"
    else:
        path_kind = "Path"

    details = [
        f"[success]✓[/success] Active input saved",
        f"[muted]Type:[/muted] {path_kind}",
        f"[muted]Location:[/muted] [info]{input_path}[/info]",
    ]

    next_steps = [
        "Run [accent]autocleaneeg-pipeline process[/accent] to use this input by default",
        "Use [accent]input show[/accent] to verify or [accent]input unset[/accent] to clear",
    ]

    body = "\n".join(details) + "\n\n[header]Next steps[/header]\n" + "\n".join(
        f"  • {step}" for step in next_steps
    )

    console.print()
    console.print(
        Panel(
            body,
            title="[title]Active Input[/title]",
            border_style="border",
            padding=(1, 2),
        )
    )
    console.print()


class RichHelpAction(argparse.Action):
    """Subparser -h/--help: show styled header + context, then default help."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        console = get_console(
            namespace if isinstance(namespace, argparse.Namespace) else None
        )
        _simple_header(console)
        _print_startup_context(console)
        console.print(parser.format_help())
        sys.exit(0)


class RootRichHelpAction(argparse.Action):
    """Root -h/--help: show styled header + context; supports optional topic like '-h auth'."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        console = get_console(
            namespace if isinstance(namespace, argparse.Namespace) else None
        )
        _simple_header(console)
        _print_startup_context(console)

        topic = (values or "").strip().lower() if isinstance(values, str) else None
        _print_root_help(console, topic)
        sys.exit(0)


def _print_root_help(console, topic: Optional[str] = None) -> None:
    """Print the root help menu with optional topic sections, in a clean minimalist layout."""
    from rich.table import Table as _Table

    # Compact usage line for quick orientation
    console.print(
        "[muted]Usage:[/muted] [accent]autocleaneeg-pipeline <command> [options][/accent]"
    )
    console.print()

    if topic in {"auth", "authentication"}:
        console.print("[header]Auth Commands[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")

        rows = [
            ("🔐 auth login", "Login to Auth0 (compliance mode)"),
            ("🔓 auth logout", "Logout and clear tokens"),
            ("👤 auth whoami", "Show authenticated user"),
            ("🩺 auth diagnostics", "Diagnose Auth0 configuration/connectivity"),
            ("⚙️ auth setup", "Enable Part-11 compliance (permanent)"),
            ("🟢 auth enable", "Enable compliance mode (non-permanent)"),
            ("🔴 auth disable", "Disable compliance mode (if permitted)"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print(
            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
        )
        console.print()
        return

    if topic in {"task", "tasks"}:
        console.print("[header]Task Commands[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        rows = [
            ("📜 task list", "List available tasks (same as 'list-tasks')"),
            ("📂 task explore", "Open the workspace tasks folder"),
            ("✏️  task edit [name|path]", "Edit task (omit uses active)"),
            ("📥 task import <path>", "Copy a task file into workspace"),
            ("📄 task copy [name|path]", "Copy task (omit uses active)"),
            ("🗑  task delete [name|path]", "Delete task (omit uses active)"),
            ("🎯 task set [name]", "Set active task (interactive if omitted)"),
            ("🧹 task unset", "Clear the active task"),
            ("👁️  task show", "Show the current active task"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
        )
        console.print()
        return

    if topic in {"block", "blocks"}:
        console.print("[header]Block Commands[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        rows = [
            ("🧱 blocks list", "List available processing blocks"),
            ("🔎 blocks info <name>", "Show metadata for a block"),
            ("📦 blocks deps <name>", "Show dependencies for a block"),
            ("🔄 blocks update", "Refresh blocks from the task registry"),
            ("💾 blocks install <name>", "Install block to cache"),
            ("🔒 blocks lock", "Generate lock file for reproducibility"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
        )
        console.print()
        return

    if topic in {"montage", "montages"}:
        console.print("[header]Montage Commands[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        tbl.add_row(
            "🧭 montage list",
            "List EEG montages defined in configs/montages.yaml",
        )
        tbl.add_row(
            "🛠 montage set [name]",
            "Select or apply a montage to the active task",
        )
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Edit configs/montages.yaml to add or rename montages.[/muted]"
        )
        console.print()
        return

    if topic in {"input", "inputs"}:
        console.print("[header]Input Commands[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        rows = [
            (
                "📁 input set [path]",
                "Set active input path (file or directory; interactive if omitted)",
            ),
            ("🧹 input unset", "Clear the active input path"),
            ("👁️  input show", "Show the current active input path"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
        )
        console.print()
        return

    if topic in {"source", "sources"}:
        console.print(
            "[header]Source Commands[/header] [warning](deprecated — use 'input')[/warning]"
        )
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        rows = [
            ("📁 source set [path]", "Alias of 'input set' (interactive if omitted)"),
            ("🧹 source unset", "Alias of 'input unset'"),
            ("👁️  source show", "Alias of 'input show'"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
        )
        console.print()
        return

    if topic in {"workspace", "setup"}:
        console.print("[header]Workspace[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        rows = [
            ("🗂  workspace", "Configure workspace folder (wizard)"),
            ("👀 workspace show", "Show current workspace path/status"),
            ("📂 workspace explore", "Open the workspace folder"),
            ("📏 workspace size", "Show total workspace size"),
            ("📌 workspace set <path>", "Change the workspace folder"),
            ("❎ workspace unset", "Unassign current workspace (clear config)"),
            ("📁 workspace cd [--spawn]", "Print path for cd, or spawn subshell"),
            ("🏠 workspace default", "Set recommended default location"),
            ("—", "—"),
            ("🔐 auth setup|enable|disable", "Compliance controls (Auth0)"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
        )
        console.print()
        return

    if topic in {"settings", "setting"}:
        console.print("[header]Settings Commands[/header]")
        tbl = _Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Command", style="accent", no_wrap=True)
        tbl.add_column("Description", style="muted")
        rows = [
            (
                "🎨 settings theme [name]",
                "Configure CLI color theme (interactive if omitted)",
            ),
            ("🧹 settings theme --clear", "Reset theme to auto-detect"),
        ]
        for c, d in rows:
            tbl.add_row(c, d)
        console.print(tbl)
        console.print()
        console.print(
            "[muted]Available themes:[/muted] [accent]auto, dark, light, hc (high contrast), mono[/accent]"
        )
        console.print(
            "[muted]Theme is saved persistently and applies to all CLI commands.[/muted]"
        )
        console.print()
        return

    console.print(
        "[muted]Tip: use 'help <topic>' to see focused command groups.[/muted]"
    )
    console.print()
    console.print("[header]Main Commands[/header]")

    grid = _Table.grid(expand=True, padding=(0, 3))
    columns = 6
    for _ in range(columns):
        grid.add_column(ratio=1)

    def _format_entry(command: str, description: str) -> str:
        return f"[accent]{command}[/accent]\n[muted]{description}[/muted]"

    entries = [
        ("✨ wizard", "Guided first-run setup"),
        ("🗂\u00a0 workspace", "Configure workspace folder"),
        ("🗂\u00a0 task", "Manage tasks (list, explore)"),
        ("🧱 blocks", "Manage bundled processing blocks"),
        ("📁\u00a0 input", "Manage active input path"),
        ("▶\u00a0 process", "Process EEG data"),
        ("👁\u00a0 view", "View EEG file (MNE-QT)"),
        ("🧭\u00a0 montage", "List montages and update active task"),
        ("📝 review", "Start review GUI"),
        ("🚫 exclude", "Start inclusion/exclusion GUI"),
        ("🎨 settings", "Customize CLI defaults and theme"),
        ("🔐 auth", "Authentication & Part-11 commands"),
    ]

    for idx in range(0, len(entries), columns):
        row_entries = [
            _format_entry(command, description)
            for command, description in entries[idx : idx + columns]
        ]
        while len(row_entries) < columns:
            row_entries.append("")
        grid.add_row(*row_entries)

    console.print(grid)
    console.print()
    console.print("[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]")
    console.print()


def attach_rich_help(p: argparse.ArgumentParser, *, root: bool = False) -> None:
    # Replace default help with our rich-aware action
    if any(a.option_strings == ["-h"] for a in p._actions):  # remove default
        for a in list(p._actions):
            if a.option_strings == ["-h"] or a.option_strings == ["-h", "--help"]:
                p._actions.remove(a)
                break
    action = RootRichHelpAction if root else RichHelpAction
    nargs = "?" if root else 0
    p.add_argument(
        "-h",
        "--help",
        action=action,
        nargs=nargs,
        help='Show help (use "-h auth" for authentication help)',
    )


# Simple branding constants
PRODUCT_NAME = "AutoClean EEG"
TAGLINE = "Automated EEG Processing Software"
LOGO_ICON = "🧠"
DIVIDER = "═════════════════════════════════════════════════"

# Try to import database functions (used conditionally in login)
try:
    from autoclean.utils.database import (
        manage_database_conditionally,
        set_database_path,
    )

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Try to import inquirer (used for interactive setup)
try:
    import inquirer

    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False

# Tame noisy third-party INFO logs by default (user can override)
if os.getenv("AUTOCLEAN_VERBOSE_LIBS") not in {"1", "true", "True", "YES", "yes"}:
    # Ensure MNE reduces verbose backend messages (like "Using qt as 2D backend.")
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
    # Reduce VisPy noise sometimes emitted by visualization deps
    os.environ.setdefault("VISPY_LOG_LEVEL", "ERROR")
    import logging as _logging

    for _name in ("OpenGL", "OpenGL.acceleratesupport"):
        try:
            _logging.getLogger(_name).setLevel(_logging.ERROR)
        except Exception:
            pass

# Try to import autoclean core components (may fail in some environments)
try:
    from autoclean.core.pipeline import Pipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for AutoClean CLI."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Basic Usage:
  autocleaneeg-pipeline workspace                      # First time setup
  autocleaneeg-pipeline process RestingEyesOpen data.raw   # Process single file
  autocleaneeg-pipeline process ica                     # Apply ICA control sheet edits
  autocleaneeg-pipeline task list                      # Show available tasks
  autocleaneeg-pipeline review                         # Start review GUI
  autocleaneeg-pipeline exclude [path]                 # Start inclusion/exclusion GUI

Active Task (Simplified Workflow):
  autocleaneeg-pipeline task set                       # Select active task interactively
  autocleaneeg-pipeline task show                      # Show current active task
  autocleaneeg-pipeline process data.raw               # Process file with active task

Custom Tasks:
  autocleaneeg-pipeline task add my_task.py            # Add custom task file
  autocleaneeg-pipeline task list                      # List all tasks

Montages:
  autocleaneeg-pipeline montage list                  # Show available montages
  autocleaneeg-pipeline montage set                   # Update active task montage


For detailed help on any command: autocleaneeg-pipeline <command> --help
        """,
    )

    # Global UI options
    parser.add_argument(
        "--theme",
        choices=["auto", "dark", "light", "hc", "mono"],
        default="auto",
        help="CLI color theme (default: auto). Use 'mono' for no hues, 'hc' for high contrast.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # Attach rich help to root
    attach_rich_help(parser, root=True)

    # Process command
    wizard_parser = subparsers.add_parser(
        "wizard",
        help="Guided setup for workspace, tasks, montage, and input",
        add_help=False,
    )
    attach_rich_help(wizard_parser)

    process_parser = subparsers.add_parser(
        "process", help="Process EEG data", add_help=False
    )
    attach_rich_help(process_parser)

    # Positional arguments for simple usage: autocleaneeg-pipeline process TaskName FilePath
    # Both are optional - if omitted, uses active task and active input from persistent state
    # Zero-argument usage: autocleaneeg-pipeline process (when task and input are configured)
    process_parser.add_argument(
        "task_name",
        nargs="?",
        type=str,
        help="Task name (optional if active task is set with 'task set')",
    )
    process_parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="EEG file or directory (optional if active input is set with 'input set')",
    )

    # Optional named arguments (for advanced usage)
    process_parser.add_argument(
        "--task", type=str, help="Task name (alternative to positional)"
    )
    process_parser.add_argument(
        "--task-file", type=Path, help="Python task file to use"
    )

    # Input options (for advanced usage)
    process_parser.add_argument(
        "--file",
        type=Path,
        help="Single EEG file to process (alternative to positional)",
    )
    process_parser.add_argument(
        "--dir",
        "--directory",
        type=Path,
        dest="directory",
        help="Directory containing EEG files to process (alternative to positional)",
    )

    process_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: workspace/output)",
    )
    process_parser.add_argument(
        "--format",
        type=str,
        default="*.{raw,set}",
        help="File format glob pattern for directory processing (default: *.{raw,set}). Examples: '*.raw', '*.edf', '*.set'. Note: '.raw' will be auto-corrected to '*.raw'",
    )
    process_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    process_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose/debug output (default: on)",
    )
    process_parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        metavar="N",
        help="Process files in parallel (default: 3 concurrent files, max: 8)",
    )
    process_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts (non-interactive mode for automation)",
    )

    process_subparsers = process_parser.add_subparsers(
        dest="process_action", help="Process subcommands"
    )
    ica_parser = process_subparsers.add_parser(
        "ica", help="Apply ICA control sheet edits", add_help=False
    )
    attach_rich_help(ica_parser)
    ica_parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Directory containing metadata (default: workspace/metadata)",
    )
    ica_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without modifying files",
    )
    ica_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    # List tasks command (alias for 'task list')
    list_tasks_parser = subparsers.add_parser(
        "list-tasks", help="List all available tasks", add_help=False
    )
    attach_rich_help(list_tasks_parser)
    list_tasks_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Show detailed information (default: on)",
    )
    list_tasks_parser.add_argument(
        "--overrides",
        action="store_true",
        help="Show workspace tasks that override built-in tasks",
    )

    # Review command
    review_parser = subparsers.add_parser(
        "review", help="Start review GUI", add_help=False
    )
    attach_rich_help(review_parser)
    review_parser.add_argument(
        "--output",
        type=Path,
        required=False,  # Changed from required=True to required=False
        help="AutoClean output directory to review (default: workspace/output)",
    )

    # Exclude command (GUI inclusion/exclusion helper)
    exclude_parser = subparsers.add_parser(
        "exclude", help="Start inclusion/exclusion GUI", add_help=False
    )
    attach_rich_help(exclude_parser)
    # Keep arguments aligned with the standalone helper for familiarity
    exclude_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=(
            "Optional path to a run folder or exports directory. Defaults to CWD, "
            "or ./exports if present."
        ),
    )
    exclude_parser.add_argument(
        "--exports",
        type=Path,
        help="Directly specify the exports directory to inspect.",
    )
    exclude_parser.add_argument(
        "--task-root",
        type=Path,
        dest="task_root",
        help="Optional task/run root directory so related reports can be listed.",
    )

    # Task management commands
    task_parser = subparsers.add_parser(
        "task", help="Manage custom tasks", add_help=False
    )
    # Replace error method with custom handler after subparsers are created
    original_error = task_parser.error
    
    def custom_task_error(message: str) -> None:
        """Custom error handler for task subcommands."""
        # Check if this is an invalid choice error for task_action
        if "invalid choice" in message.lower() and "task_action" in message:
            # Extract the invalid action from the error message
            import re
            match = re.search(r"invalid choice: '([^']+)'", message)
            invalid_action = match.group(1) if match else "unknown"
            
            # Get valid actions from task_parser's subparsers action
            valid_actions = []
            for action in task_parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    valid_actions = sorted(action.choices.keys())
                    break
            
            # Show custom error message
            # Use a throwaway console so we don't mutate the global theme cache
            console = make_console()
            console.print()
            
            # Build command descriptions dynamically
            command_descriptions = {
                "list": "List available tasks",
                "explore": "Open workspace tasks folder",
                "edit": "Edit a task",
                "copy": "Copy a task",
                "delete": "Delete a task",
                "set": "Set active task",
                "unset": "Clear active task",
                "show": "Show current active task",
                "use": "Install and activate a task",
                "install": "Install a task from library/file",
                "sync": "Check for task updates",
                "update": "Update task library index",
                "search": "Search for tasks",
                "diagnose": "Run workspace health check",
                "diff": "Compare task versions",
                "schema": "View task schema",
            }
            
            commands_text = "\n".join(
                f"  [accent]task {action}[/accent]"
                f"{' ' * (20 - len(action) - 5)}{command_descriptions.get(action, '')}"
                for action in valid_actions
            )
            
            console.print(
                Panel(
                    f"[error]Unknown task command:[/error] [accent]'{invalid_action}'[/accent]\n\n"
                    f"[header]Valid task commands:[/header]\n{commands_text}\n\n"
                    "[muted]Run 'task --help' for detailed help[/muted]",
                    title="[title]Task Command Help[/title]",
                    border_style="border",
                    padding=(1, 2),
                )
            )
            console.print()
            sys.exit(2)
        
        # For other errors, use default argparse behavior
        original_error(message)
    
    attach_rich_help(task_parser)
    task_subparsers = task_parser.add_subparsers(
        dest="task_action", help="Task actions"
    )
    # Replace error method after subparsers are created so we can access valid actions
    task_parser.error = custom_task_error

    # Delete task (alias with path/name support)
    delete_task_parser = task_subparsers.add_parser(
        "delete",
        help="Delete a workspace task file (omit target to use active task)",
        add_help=False,
    )
    attach_rich_help(delete_task_parser)
    delete_task_parser.add_argument(
        "target",
        type=str,
        nargs="?",
        help="Task name (workspace) or path to task file (omit to use active task)",
    )
    delete_task_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation"
    )

    # List all tasks (replaces old list-tasks command)
    list_all_parser = task_subparsers.add_parser(
        "list", help="List all available tasks", add_help=False
    )
    attach_rich_help(list_all_parser)
    list_all_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Show detailed information (default: on)",
    )
    list_all_parser.add_argument(
        "--overrides",
        action="store_true",
        help="Show workspace tasks that override built-in tasks",
    )
    list_all_parser.add_argument(
        "--source",
        choices=["all", "library", "builtin", "workspace", "active"],
        default="all",
        help="Filter by task source (default: all)",
    )
    list_all_parser.add_argument(
        "--status",
        choices=["all", "installed", "available", "outdated", "customized"],
        default="all",
        help="Filter by task status (default: all)",
    )
    list_all_parser.add_argument(
        "--category",
        choices=["all", "resting", "auditory", "visual", "rodent"],
        default="all",
        help="Filter by task category (default: all)",
    )
    list_all_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # Explore tasks folder (open in OS file browser)
    explore_parser = task_subparsers.add_parser(
        "explore", help="Open the workspace tasks folder in your OS", add_help=False
    )
    attach_rich_help(explore_parser)

    # Edit task (open in shell editor)
    edit_parser = task_subparsers.add_parser(
        "edit",
        help="Edit a task in your editor (omit target to use active task)",
        add_help=False,
    )
    attach_rich_help(edit_parser)
    edit_parser.add_argument(
        "target",
        type=str,
        nargs="?",
        help="Task name (workspace) or path to a task file (omit to use active task)",
    )
    edit_parser.add_argument(
        "--name",
        type=str,
        help="When copying a built-in task, save as this name (without .py)",
    )

    schema_parser = task_subparsers.add_parser(
        "schema",
        help="Inspect or export the task schema",
        add_help=False,
    )
    attach_rich_help(schema_parser)
    schema_subparsers = schema_parser.add_subparsers(
        dest="task_schema_action",
        help="Task schema commands",
    )

    schema_export_parser = schema_subparsers.add_parser(
        "export",
        help="Export the canonical task schema to JSON",
        add_help=False,
    )
    attach_rich_help(schema_export_parser)
    schema_export_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write schema JSON to this path (defaults to stdout)",
    )
    schema_export_parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for JSON output (set to 0 for compact)",
    )
    edit_parser.add_argument(
        "--force",
        action="store_true",
        help="When copying a built-in task, overwrite existing without prompting",
    )

    # Copy task (name or path)
    copy_parser = task_subparsers.add_parser(
        "copy",
        help="Copy a task into workspace (omit source to pick from list)",
        add_help=False,
    )
    attach_rich_help(copy_parser)
    copy_parser.add_argument(
        "source",
        type=str,
        nargs="?",
        help="Task name (workspace/built-in) or path to copy from (omit to use active task)",
    )
    copy_parser.add_argument(
        "--name",
        type=str,
        help="Destination filename (without .py)",
    )
    copy_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file without prompting",
    )

    # Set active task
    set_task_parser = task_subparsers.add_parser(
        "set",
        help="Set the active task (used when no task specified in process)",
        add_help=False,
    )
    attach_rich_help(set_task_parser)
    set_task_parser.add_argument(
        "task_name",
        type=str,
        nargs="?",
        help="Task name to set as active (omit to choose interactively)",
    )

    # Unset active task
    unset_task_parser = task_subparsers.add_parser(
        "unset", help="Clear the active task", add_help=False
    )
    attach_rich_help(unset_task_parser)

    # Show active task
    show_task_parser = task_subparsers.add_parser(
        "show", help="Show the current active task", add_help=False
    )
    attach_rich_help(show_task_parser)

    # Use task (install + activate in one step)
    use_task_parser = task_subparsers.add_parser(
        "use",
        help="Install and activate a task in one step (from library or built-in)",
        add_help=False,
    )
    attach_rich_help(use_task_parser)
    use_task_parser.add_argument(
        "task_name",
        nargs="?",
        type=str,
        help="Task to install and set active (omit for interactive selection)",
    )
    use_task_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing workspace copy without prompting",
    )
    use_task_parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Install only, don't set as active task",
    )

    # Install task (unified command for any source)
    install_task_parser = task_subparsers.add_parser(
        "install",
        help="Install a task from library, file, or built-in source",
        add_help=False,
    )
    attach_rich_help(install_task_parser)
    install_task_parser.add_argument(
        "task_source",
        type=str,
        help="Task name (from library) or file path to install",
    )
    install_task_parser.add_argument(
        "--source",
        choices=["auto", "library", "file", "builtin"],
        default="auto",
        help="Explicitly specify source type (default: auto-detect)",
    )
    install_task_parser.add_argument(
        "--name",
        type=str,
        help="Custom name for installed task (default: use original name)",
    )
    install_task_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing workspace copy without prompting",
    )
    install_task_parser.add_argument(
        "--activate",
        action="store_true",
        help="Set as active task after installing",
    )

    # Sync tasks (check for updates)
    sync_task_parser = task_subparsers.add_parser(
        "sync",
        help="Check workspace tasks for updates from library/built-in sources",
        add_help=False,
    )
    attach_rich_help(sync_task_parser)
    sync_task_parser.add_argument(
        "--update",
        action="store_true",
        help="Apply updates automatically (creates backups)",
    )
    sync_task_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )

    # Update library (promoted from task library update)
    update_task_parser = task_subparsers.add_parser(
        "update",
        help="Update task library index from GitHub registry",
        add_help=False,
    )
    attach_rich_help(update_task_parser)
    update_task_parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip network check (use cached data only)",
    )

    # Diagnose workspace health
    diagnose_task_parser = task_subparsers.add_parser(
        "diagnose",
        help="Check workspace health and task integrity",
        add_help=False,
    )
    attach_rich_help(diagnose_task_parser)

    # Diff task (compare workspace vs source)
    diff_task_parser = task_subparsers.add_parser(
        "diff",
        help="Show differences between workspace task and its source",
        add_help=False,
    )
    attach_rich_help(diff_task_parser)
    diff_task_parser.add_argument(
        "task_name",
        type=str,
        help="Name of the task to compare",
    )
    diff_task_parser.add_argument(
        "--color",
        action="store_true",
        default=True,
        help="Use colorized output (default: on)",
    )
    diff_task_parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Lines of context to show (default: 3)",
    )

    # Search tasks
    search_task_parser = task_subparsers.add_parser(
        "search",
        help="Search for tasks across all sources",
        add_help=False,
    )
    attach_rich_help(search_task_parser)
    search_task_parser.add_argument(
        "query",
        type=str,
        help="Search term (matches name, description, tags)",
    )
    search_task_parser.add_argument(
        "--source",
        choices=["all", "library", "builtin", "workspace"],
        default="all",
        help="Limit search to specific source (default: all)",
    )

    # Montage commands
    montage_parser = subparsers.add_parser(
        "montage", help="Inspect EEG montage definitions", add_help=False
    )
    attach_rich_help(montage_parser)
    montage_subparsers = montage_parser.add_subparsers(
        dest="montage_action", help="Montage actions"
    )

    montage_list_parser = montage_subparsers.add_parser(
        "list", help="List available EEG montages", add_help=False
    )
    attach_rich_help(montage_list_parser)

    montage_set_parser = montage_subparsers.add_parser(
        "set", help="Update the active task's montage", add_help=False
    )
    attach_rich_help(montage_set_parser)
    montage_set_parser.add_argument(
        "montage_name",
        nargs="?",
        help="Montage identifier to apply (omit to pick interactively)",
    )
    montage_set_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt and apply immediately",
    )

    montage_test_parser = montage_subparsers.add_parser(
        "test",
        help="Generate montage validation report for active input",
        add_help=False,
    )
    attach_rich_help(montage_test_parser)

    # Blocks management commands
    blocks_parser = subparsers.add_parser(
        "blocks", help="Manage bundled processing blocks", add_help=False
    )
    attach_rich_help(blocks_parser)
    blocks_subparsers = blocks_parser.add_subparsers(
        dest="blocks_action", help="Block actions"
    )

    blocks_list_parser = blocks_subparsers.add_parser(
        "list", help="List all bundled blocks", add_help=False
    )
    attach_rich_help(blocks_list_parser)

    blocks_info_parser = blocks_subparsers.add_parser(
        "info", help="Show detailed information about a block", add_help=False
    )
    attach_rich_help(blocks_info_parser)
    blocks_info_parser.add_argument(
        "block_name",
        type=str,
        help="Name of the block to inspect",
    )

    blocks_deps_parser = blocks_subparsers.add_parser(
        "deps", help="Show dependencies for a block", add_help=False
    )
    attach_rich_help(blocks_deps_parser)
    blocks_deps_parser.add_argument(
        "block_name",
        type=str,
        help="Name of the block to check dependencies for",
    )

    blocks_update_parser = blocks_subparsers.add_parser(
        "update", help="Update blocks from task-registry", add_help=False
    )
    attach_rich_help(blocks_update_parser)
    blocks_update_parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip online check (offline mode)",
    )

    blocks_install_parser = blocks_subparsers.add_parser(
        "install", help="Install a block to cache", add_help=False
    )
    attach_rich_help(blocks_install_parser)
    blocks_install_parser.add_argument(
        "block_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the block to install (not needed with --locked)",
    )
    blocks_install_parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Git commit hash to install (for reproducibility). Browse GitHub registry to find commit hashes.",
    )
    blocks_install_parser.add_argument(
        "--locked",
        action="store_true",
        help="Install all blocks from blocks.lock file",
    )
    blocks_install_parser.add_argument(
        "--lock-file",
        type=str,
        default="blocks.lock",
        help="Path to lock file (default: blocks.lock)",
    )

    blocks_lock_parser = blocks_subparsers.add_parser(
        "lock", help="Generate lock file from current block state", add_help=False
    )
    attach_rich_help(blocks_lock_parser)
    blocks_lock_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="blocks.lock",
        help="Output file path (default: blocks.lock)",
    )

    # Source management commands (deprecated alias)
    source_parser = subparsers.add_parser(
        "source",
        help="[deprecated] Manage active input path (use 'input')",
        add_help=False,
    )
    attach_rich_help(source_parser)
    source_subparsers = source_parser.add_subparsers(
        dest="source_action", help="Source actions"
    )

    # Set active source
    set_source_parser = source_subparsers.add_parser(
        "set",
        help="[deprecated] Set the active input path (use 'input set')",
        add_help=False,
    )
    attach_rich_help(set_source_parser)
    set_source_parser.add_argument(
        "source_path",
        nargs="?",
        help="Input path to set as active (file or directory, omit to choose interactively)",
    )

    # Unset active source
    unset_source_parser = source_subparsers.add_parser(
        "unset",
        help="[deprecated] Clear the active input path (use 'input unset')",
        add_help=False,
    )
    attach_rich_help(unset_source_parser)

    # Show active source
    show_source_parser = source_subparsers.add_parser(
        "show",
        help="[deprecated] Show the current active input path (use 'input show')",
        add_help=False,
    )
    attach_rich_help(show_source_parser)

    # Input management commands (preferred)
    input_parser = subparsers.add_parser(
        "input", help="Manage active input path", add_help=False
    )
    attach_rich_help(input_parser)
    input_subparsers = input_parser.add_subparsers(
        dest="input_action", help="Input actions"
    )

    # Set active input
    set_input_parser = input_subparsers.add_parser(
        "set",
        help="Set the active input path (used when no input specified in process)",
        add_help=False,
    )
    attach_rich_help(set_input_parser)
    set_input_parser.add_argument(
        "source_path",
        nargs="?",
        help="Input path to set as active (file or directory; quotes OK; omit to choose interactively)",
    )

    # Unset active input
    unset_input_parser = input_subparsers.add_parser(
        "unset", help="Clear the active input path", add_help=False
    )
    attach_rich_help(unset_input_parser)

    # Show active input
    show_input_parser = input_subparsers.add_parser(
        "show", help="Show the current active input path", add_help=False
    )
    attach_rich_help(show_input_parser)

    # Show config location
    config_parser = subparsers.add_parser(
        "config", help="Manage user configuration", add_help=False
    )
    attach_rich_help(config_parser)
    config_subparsers = config_parser.add_subparsers(
        dest="config_action", help="Config actions"
    )

    # Show config location
    _cfg_show = config_subparsers.add_parser(
        "show", help="Show configuration directory location", add_help=False
    )
    attach_rich_help(_cfg_show)

    # Setup/reconfigure workspace
    _cfg_setup = config_subparsers.add_parser(
        "setup", help="Reconfigure workspace location", add_help=False
    )
    attach_rich_help(_cfg_setup)

    # Reset config
    reset_parser = config_subparsers.add_parser(
        "reset", help="Reset configuration to defaults", add_help=False
    )
    attach_rich_help(reset_parser)
    reset_parser.add_argument(
        "--confirm", action="store_true", help="Confirm the reset action"
    )

    # Export/import config
    export_parser = config_subparsers.add_parser(
        "export", help="Export configuration", add_help=False
    )
    attach_rich_help(export_parser)
    export_parser.add_argument(
        "export_path", type=Path, help="Directory to export configuration to"
    )

    import_parser = config_subparsers.add_parser(
        "import", help="Import configuration", add_help=False
    )
    attach_rich_help(import_parser)
    import_parser.add_argument(
        "import_path", type=Path, help="Directory to import configuration from"
    )

    # Workspace command (replaces old 'setup' for workspace configuration)
    workspace_parser = subparsers.add_parser(
        "workspace", help="Configure workspace folder", add_help=False
    )
    attach_rich_help(workspace_parser)
    workspace_subparsers = workspace_parser.add_subparsers(
        dest="workspace_action", help="Workspace actions"
    )

    ws_explore = workspace_subparsers.add_parser(
        "explore", help="Open the workspace folder in Finder/Explorer", add_help=False
    )
    attach_rich_help(ws_explore)

    # Show current workspace path
    ws_show = workspace_subparsers.add_parser(
        "show", help="Show current workspace path/status", add_help=False
    )
    attach_rich_help(ws_show)

    ws_size = workspace_subparsers.add_parser(
        "size", help="Show total workspace size", add_help=False
    )
    attach_rich_help(ws_size)

    ws_set = workspace_subparsers.add_parser(
        "set", help="Change the workspace folder", add_help=False
    )
    attach_rich_help(ws_set)
    ws_set.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="New workspace directory path (omit to choose interactively)",
    )

    ws_unset = workspace_subparsers.add_parser(
        "unset", help="Unassign current workspace (clear config)", add_help=False
    )
    attach_rich_help(ws_unset)

    ws_default = workspace_subparsers.add_parser(
        "default",
        help="Set workspace to the recommended default location",
        add_help=False,
    )
    attach_rich_help(ws_default)

    ws_cd = workspace_subparsers.add_parser(
        "cd",
        help="Change directory to workspace (prints path or spawns subshell)",
        add_help=False,
    )
    attach_rich_help(ws_cd)
    ws_cd.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn an interactive shell in the workspace directory",
    )
    ws_cd.add_argument(
        "--print",
        choices=["auto", "bash", "zsh", "fish", "powershell", "cmd"],
        help="Print a shell-specific cd command you can eval",
    )

    # Export access log command
    export_log_parser = subparsers.add_parser(
        "export-access-log",
        help="Export database access log with integrity verification",
        add_help=False,
    )
    attach_rich_help(export_log_parser)
    export_log_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: access-log-{timestamp}.json)",
    )
    export_log_parser.add_argument(
        "--format",
        choices=["json", "csv", "human"],
        default="json",
        help="Output format (default: json)",
    )
    export_log_parser.add_argument(
        "--start-date", type=str, help="Start date filter (YYYY-MM-DD format)"
    )
    export_log_parser.add_argument(
        "--end-date", type=str, help="End date filter (YYYY-MM-DD format)"
    )
    export_log_parser.add_argument(
        "--operation", type=str, help="Filter by operation type"
    )
    export_log_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify integrity, don't export data",
    )
    export_log_parser.add_argument(
        "--database",
        type=Path,
        help="Path to database file (default: auto-detect from workspace)",
    )

    # Authentication commands (for compliance mode)
    _login = subparsers.add_parser(
        "login", help="Login to Auth0 for compliance mode", add_help=False
    )
    attach_rich_help(_login)
    _logout = subparsers.add_parser(
        "logout", help="Logout and clear authentication tokens", add_help=False
    )
    attach_rich_help(_logout)
    _whoami = subparsers.add_parser(
        "whoami", help="Show current authenticated user", add_help=False
    )
    attach_rich_help(_whoami)
    auth_diag_parser = subparsers.add_parser(
        "auth0-diagnostics",
        help="Diagnose Auth0 configuration and connectivity issues",
        add_help=False,
    )
    attach_rich_help(auth_diag_parser)
    auth_diag_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Show detailed diagnostic information (default: on)",
    )

    # Clean task command
    clean_task_parser = subparsers.add_parser(
        "clean-task",
        help="Remove task output directory and database entries",
        add_help=False,
    )
    attach_rich_help(clean_task_parser)
    clean_task_parser.add_argument("task", help="Task name to clean")
    clean_task_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to configured workspace)",
    )
    clean_task_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    clean_task_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    # Report command group
    report_parser = subparsers.add_parser(
        "report", help="Generate LLM-backed textual reports", add_help=False
    )
    attach_rich_help(report_parser)
    report_subparsers = report_parser.add_subparsers(
        dest="report_action", help="Report actions"
    )

    report_create_parser = report_subparsers.add_parser(
        "create", help="Create textual reports from a run context", add_help=False
    )
    attach_rich_help(report_create_parser)
    report_create_parser.add_argument("--run-id", required=True)
    report_create_parser.add_argument(
        "--context-json",
        type=Path,
        required=True,
        help="Path to a saved run context JSON.",
    )
    report_create_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for reports.",
    )

    report_chat_parser = report_subparsers.add_parser(
        "chat", help="Interactive Q&A about a run", add_help=False
    )
    attach_rich_help(report_chat_parser)
    report_chat_parser.add_argument(
        "--context-json",
        type=Path,
        required=False,
        help="Path to a saved run context JSON. If omitted, uses the latest run's LLM context if available, else reconstructs from processing_log + PDF.",
    )

    # View command
    view_parser = subparsers.add_parser(
        "view", help="View EEG files using MNE-QT Browser", add_help=False
    )
    attach_rich_help(view_parser)
    view_parser.add_argument("file", nargs="?", type=Path, help="Path to EEG file")
    view_parser.add_argument(
        "--no-view", action="store_true", help="Validate without viewing"
    )

    # Version command
    _version = subparsers.add_parser(
        "version", help="Show version information", add_help=False
    )  # Help command (for consistency)
    attach_rich_help(_version)
    _help = subparsers.add_parser(
        "help", help="Show detailed help information", add_help=False
    )
    _help.add_argument("topic", nargs="?", help="Optional help topic (e.g., 'auth')")
    attach_rich_help(_help)

    # Tutorial command
    _tutorial = subparsers.add_parser(
        "tutorial", help="Show a helpful tutorial for first-time users", add_help=False
    )
    attach_rich_help(_tutorial)

    # Auth command group (aliases for authentication/Part-11 tasks)
    auth_parser = subparsers.add_parser(
        "auth", help="Authentication & Part-11 commands", add_help=False
    )
    attach_rich_help(auth_parser)
    auth_subparsers = auth_parser.add_subparsers(
        dest="auth_action", help="Auth actions"
    )

    auth_login = auth_subparsers.add_parser(
        "login", help="Login to Auth0", add_help=False
    )
    attach_rich_help(auth_login)

    auth_logout = auth_subparsers.add_parser(
        "logout", help="Logout and clear tokens", add_help=False
    )
    attach_rich_help(auth_logout)

    auth_whoami = auth_subparsers.add_parser(
        "whoami", help="Show authenticated user", add_help=False
    )
    attach_rich_help(auth_whoami)

    auth_diag = auth_subparsers.add_parser(
        "diagnostics", help="Diagnose Auth0 configuration/connectivity", add_help=False
    )
    attach_rich_help(auth_diag)

    auth_setup = auth_subparsers.add_parser(
        "setup", help="Enable Part-11 compliance (permanent)", add_help=False
    )
    attach_rich_help(auth_setup)

    auth_enable = auth_subparsers.add_parser(
        "enable", help="Enable compliance mode (non-permanent)", add_help=False
    )
    attach_rich_help(auth_enable)

    auth_disable = auth_subparsers.add_parser(
        "disable", help="Disable compliance mode (if permitted)", add_help=False
    )
    attach_rich_help(auth_disable)

    # Settings commands
    settings_parser = subparsers.add_parser(
        "settings", help="Manage persistent CLI settings", add_help=False
    )
    attach_rich_help(settings_parser)
    settings_subparsers = settings_parser.add_subparsers(
        dest="settings_action", help="Settings actions"
    )

    # Theme settings
    theme_parser = settings_subparsers.add_parser(
        "theme", help="Configure CLI color theme", add_help=False
    )
    attach_rich_help(theme_parser)
    theme_parser.add_argument(
        "theme_name",
        nargs="?",
        choices=["auto", "dark", "light", "hc", "mono"],
        help="Theme to set (omit to choose interactively, or use 'clear' to reset)",
    )
    theme_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the saved theme preference (revert to auto)",
    )

    return parser


def _show_process_guard(args) -> bool:
    """Show interactive guard with key information before processing.

    Returns True if user confirms to proceed, False to cancel.
    """
    console = get_console(args)

    # Determine target output directory for status tagging
    output_root = (
        Path(args.output)
        if getattr(args, "output", None)
        else user_config.get_default_output_dir()
    )

    # Get current values
    task_name = args.task_name or args.task
    input_path = args.input_path or args.file or args.directory

    # If no task, try to get active task
    if not task_name and not args.task_file:
        active_task = user_config.get_active_task()
        if active_task:
            task_name = active_task

    # If no input, try to get active source
    if not input_path:
        active_source = user_config.get_active_source()
        if active_source and active_source != "NONE":
            input_path = Path(active_source)

    dataset_name = None
    try:
        if task_name:
            dataset_name = extract_config_from_task(task_name, "dataset_name")
    except Exception:  # pylint: disable=broad-except
        dataset_name = None

    task_dir_name = None
    if dataset_name:
        task_dir_name = str(dataset_name)
    elif task_name:
        task_dir_name = str(task_name)
    task_root = (output_root / task_dir_name) if task_dir_name else None

    def _mark_cancelled(reason: str) -> None:
        """Write a cancellation status marker if directories exist."""
        if task_root is None:
            return
        update_status_marker(
            task_root,
            "cancelled-before-start",
            details={
                "reason": reason,
                "task": task_name,
                "input": str(input_path) if input_path else None,
            },
            create_task_root=False,
        )

    # Header
    console.print()
    console.print("📋 [bold cyan]Process Command Guard[/bold cyan]")
    console.print("═" * 50)

    # Workspace Information
    console.print()
    console.print("🏠 [bold]Workspace Information[/bold]")
    workspace_dir = user_config.config_dir
    console.print(f"   Directory: [accent]{workspace_dir}[/accent]")

    # Free space calculation
    try:
        usage_path = workspace_dir if workspace_dir.exists() else workspace_dir.parent
        du = shutil.disk_usage(str(usage_path))
        free_gb = du.free / (1024**3)
        console.print(f"   Free Space: [accent]{free_gb:.1f} GB[/accent]")
    except Exception:
        console.print("   Free Space: [muted]Unable to determine[/muted]")

    # Task Information
    console.print()
    console.print("🎯 [bold]Task Information[/bold]")
    if task_name:
        console.print(f"   Task: [accent]{task_name}[/accent]")

        # Try to extract montage information
        try:
            montage_info = extract_config_from_task(task_name, "montage")
            if montage_info:
                if isinstance(montage_info, dict) and "value" in montage_info:
                    montage_value = montage_info["value"]
                else:
                    montage_value = montage_info
                console.print(f"   Montage: [accent]{montage_value}[/accent]")
            else:
                console.print("   Montage: [muted]Not specified in task[/muted]")
        except Exception:
            console.print("   Montage: [muted]Unable to extract[/muted]")

    elif args.task_file:
        console.print(f"   Task File: [accent]{args.task_file}[/accent]")
    else:
        console.print("   Task: [red]Not specified[/red]")

    # Input Information
    console.print()
    console.print("📁 [bold]Input Information[/bold]")
    if input_path:
        console.print(f"   Path: [accent]{input_path}[/accent]")

        if input_path.exists():
            if input_path.is_file():
                console.print("   Type: [accent]Single File[/accent]")
                # Show file size
                try:
                    file_size = input_path.stat().st_size / (1024**2)  # MB
                    console.print(f"   Size: [accent]{file_size:.1f} MB[/accent]")
                except Exception:
                    console.print("   Size: [muted]Unable to determine[/muted]")
            elif input_path.is_dir():
                console.print("   Type: [accent]Directory[/accent]")

                # Count files based on format pattern
                format_pattern = getattr(args, "format", "*.{raw,set}")
                try:

                    def _expand_brace_glob(pat: str) -> list[str]:
                        if "{" not in pat or "}" not in pat:
                            return [pat]
                        start = pat.find("{")
                        end = pat.find("}", start + 1)
                        if start == -1 or end == -1 or end < start:
                            return [pat]
                        prefix = pat[:start]
                        suffix = pat[end + 1 :]
                        body = pat[start + 1 : end]
                        options = [o.strip() for o in body.split(",") if o.strip()]
                        return [f"{prefix}{o}{suffix}" for o in options] or [pat]

                    patterns = _expand_brace_glob(format_pattern)
                    files_set: set[Path] = set()
                    files: list[Path] = []
                    if getattr(args, "recursive", False):
                        for p in patterns:
                            for f in input_path.rglob(p):
                                if f not in files_set:
                                    files_set.add(f)
                                    files.append(f)
                    else:
                        for p in patterns:
                            for f in input_path.glob(p):
                                if f not in files_set:
                                    files_set.add(f)
                                    files.append(f)

                    file_count = len(files)
                    console.print(
                        f"   Files to process: [accent]{file_count}[/accent] (pattern: {format_pattern})"
                    )

                    if getattr(args, "recursive", False):
                        console.print("   Recursive search: [accent]Enabled[/accent]")

                except Exception:
                    console.print("   Files: [muted]Unable to count[/muted]")
        else:
            console.print("   Status: [red]Path does not exist[/red]")
    else:
        console.print("   Path: [red]Not specified[/red]")

    # Additional Processing Options
    if hasattr(args, "parallel") and args.parallel:
        console.print()
        console.print("⚙️  [bold]Processing Options[/bold]")
        console.print(
            f"   Parallel processing: [accent]{args.parallel} concurrent files[/accent]"
        )

    # Confirmation prompt
    console.print()
    console.print("═" * 50)

    try:
        from rich.prompt import Confirm

        try:
            proceed = Confirm.ask(
                "🚀 [bold]Proceed with processing?[/bold]", default=False
            )
        except KeyboardInterrupt:
            message("warning", "Processing cancelled by user (Ctrl+C).")
            _mark_cancelled("user-interrupted-before-start")
            return False
        if not proceed:
            _mark_cancelled("user-declined-guard")
        return proceed
    except ImportError:
        # Fallback for systems without rich Confirm
        try:
            response = input("🚀 Proceed with processing? (y/N): ").lower().strip()
            proceed = response in ["y", "yes"]
            if not proceed:
                _mark_cancelled("user-declined-guard")
            return proceed
        except (EOFError, KeyboardInterrupt):
            message("warning", "Processing cancelled by user (Ctrl+C).")
            _mark_cancelled("user-interrupted-before-start")
            return False


def validate_args(args) -> bool:
    """Validate command line arguments."""
    if args.command == "process" and getattr(args, "process_action", None) == "ica":
        return True
    if args.command == "process":
        # Normalize positional vs named arguments
        task_name = args.task_name or args.task
        input_path = args.input_path or args.file or args.directory

        # If no task specified, check for active task first
        if not task_name and not args.task_file:
            # Try to use active task
            active_task = user_config.get_active_task(ensure_exists=False)
            if active_task:
                # Validate that the active task still exists
                custom_tasks = user_config.list_custom_tasks()
                if active_task in custom_tasks:
                    task_name = active_task
                    message("info", f"Using active task: {active_task}")
                else:
                    message(
                        "warning",
                        f"Active task '{active_task}' no longer exists in workspace",
                    )
                    message(
                        "info",
                        "Please set a new active task with: autocleaneeg-pipeline task set",
                    )
                    user_config.set_active_task(None)
                    active_task = None

            # If still no task, show help
        if not task_name and not args.task_file:
            console = get_console(args)
            _simple_header(console)
            _print_startup_context(console)
            try:
                from rich.table import Table as _Table

                console.print("[header]Process EEG[/header]")
                console.print(
                    "[muted]Usage:[/muted] [accent]autocleaneeg-pipeline process [TASK] [FILE] [OPTIONS][/accent]"
                )
                console.print()
                console.print(
                    "[success]💡 Quick Start:[/success] Run with no arguments when task and input are configured:"
                )
                console.print(
                    "   [accent]autocleaneeg-pipeline task set MyTask[/accent]"
                )
                console.print(
                    "   [accent]autocleaneeg-pipeline input set[/accent]  [dim]# choose your data folder[/dim]"
                )
                console.print("   [accent]autocleaneeg-pipeline process[/accent]")
                console.print()

                tbl = _Table(show_header=False, box=None, padding=(0, 1))
                tbl.add_column("Item", style="accent", no_wrap=True)
                tbl.add_column("Details", style="muted")
                tbl.add_row("task|--task", "Task name (e.g., RestingEyesOpen)")
                tbl.add_row("--task-file", "Path to Python task file")
                tbl.add_row("file|--file", "Single EEG file (.raw, .edf, .set, .fif)")
                tbl.add_row(
                    "dir|--dir", "Directory of EEG files (use --format, --recursive)"
                )
                tbl.add_row(
                    "--format",
                    "Glob pattern (default: *.{raw,set}; '*.raw', '*.edf', ...)",
                )
                tbl.add_row("--recursive", "Search subdirectories for matching files")
                tbl.add_row("-p N", "Process N files in parallel (default 3, max 8)")
                tbl.add_row("--dry-run", "Show what would run without processing")
                console.print(tbl)
                console.print(
                    "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
                )
                console.print()
            except Exception:
                console.print(
                    "Usage: autocleaneeg-pipeline process [TASK] [FILE] [OPTIONS]"
                )
                console.print(
                    "Quick Start: Run with no arguments when task and input are configured"
                )
                console.print(
                    "  autocleaneeg-pipeline task set MyTask && autocleaneeg-pipeline input set"
                )
                console.print("  autocleaneeg-pipeline process")
            return False

        if task_name and args.task_file:
            message("error", "Cannot specify both task name and --task-file")
            return False

        # Check input exists - with fallback to task config
        if input_path and not input_path.exists():
            message("error", f"Input path does not exist: {input_path}")
            return False
        elif not input_path:
            # Try to get input_path from task config as fallback
            task_input_path = None
            if task_name:
                task_input_path = extract_config_from_task(task_name, "input_path")

            if task_input_path:
                input_path = Path(task_input_path)
                if not input_path.exists():
                    message(
                        "error",
                        f"Input path from task config does not exist: {input_path}",
                    )
                    return False
                message("info", f"Using input path from task config: {input_path}")
            else:
                # Try to get input_path from active input as fallback
                active_source = user_config.get_active_source()
                if active_source:
                    input_path = Path(active_source)
                    if not input_path.exists():
                        message(
                            "error",
                            f"Active input path no longer exists: {input_path}",
                        )
                        message(
                            "info",
                            "Please set a new active input with: autocleaneeg-pipeline input set",
                        )
                        return False
                    message("info", f"Using active input: {input_path}")
                else:
                    message(
                        "error",
                        "No input file or directory provided. Use --file/--dir or set an active input with 'autocleaneeg-pipeline input set'.",
                    )
                    # No fallback available, show help
                    console = get_console(args)
                    _simple_header(console)
                    _print_startup_context(console)
                    try:
                        from rich.table import Table as _Table

                        console.print("[header]Process EEG[/header]")
                        console.print(
                            "[muted]Usage:[/muted] [accent]autocleaneeg-pipeline process [TaskName|--task-file FILE] <file|--dir DIR> [options][/accent]"
                        )
                        console.print(
                            "[dim]Note: Task is optional if you have set an active task with 'task set'[/dim]"
                        )
                        console.print()

                        tbl = _Table(show_header=False, box=None, padding=(0, 1))
                        tbl.add_column("Item", style="accent", no_wrap=True)
                        tbl.add_column("Details", style="muted")
                        tbl.add_row("task|--task", "Task name (e.g., RestingEyesOpen)")
                        tbl.add_row("--task-file", "Path to Python task file")
                        tbl.add_row(
                            "file|--file", "Single EEG file (.raw, .edf, .set, .fif)"
                        )
                        tbl.add_row(
                            "dir|--dir",
                            "Directory of EEG files (use --format, --recursive)",
                        )
                        tbl.add_row(
                            "--format",
                            "Glob pattern (default: *.{raw,set}; '*.raw', '*.edf', ...)",
                        )
                        tbl.add_row(
                            "--recursive", "Search subdirectories for matching files"
                        )
                        tbl.add_row(
                            "-p N", "Process N files in parallel (default 3, max 8)"
                        )
                        tbl.add_row(
                            "--dry-run", "Show what would run without processing"
                        )
                        console.print(tbl)
                        console.print(
                            "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
                        )
                        console.print()
                    except Exception:
                        console.print(
                            "Usage: autocleaneeg-pipeline process [TaskName|--task-file FILE] <file|--dir DIR> [options]"
                        )
                        console.print(
                            "Note: Task is optional if you have set an active task with 'task set'"
                        )
                    return False

        # Store normalized values back to args
        args.final_task = task_name
        args.final_input = input_path

        # Check task file exists if provided
        if args.task_file and not args.task_file.exists():
            message("error", f"Task file does not exist: {args.task_file}")
            return False

        # Show process guard when no explicit arguments were provided
        # This triggers when the user ran "autocleaneeg-pipeline process" without arguments
        # and we resolved task/input from active settings, or when only partial arguments were given
        show_guard = False

        # Check if this was a minimal command invocation
        if not (args.task_name or args.task or args.task_file):
            # No task specified on command line - using active task
            show_guard = True
        elif not (args.input_path or args.file or args.directory):
            # No input specified on command line - using active input
            show_guard = True

        # Don't show guard for dry run (already shows what would be processed)
        if hasattr(args, "dry_run") and args.dry_run:
            show_guard = False

        # Don't show guard if --yes/-y flag is set (non-interactive mode)
        if hasattr(args, "yes") and args.yes:
            show_guard = False

        # Show the guard if needed
        if show_guard:
            if not _show_process_guard(args):
                message("info", "Processing cancelled by user")
                return False

    elif args.command == "view":
        # Friendly brief help when file is missing
        if not getattr(args, "file", None):
            console = get_console(args)
            _simple_header(console)
            _print_startup_context(console)
            try:
                from rich.table import Table as _Table

                console.print("[header]View EEG[/header]")
                console.print(
                    "[muted]Usage:[/muted] [accent]autocleaneeg-pipeline view <file> [--no-view][/accent]"
                )
                console.print()

                tbl = _Table(show_header=False, box=None, padding=(0, 1))
                tbl.add_column("Item", style="accent", no_wrap=True)
                tbl.add_column("Details", style="muted")
                tbl.add_row("file", "Path to EEG file (.set, .edf, .fif, .raw)")
                tbl.add_row("--no-view", "Validate without opening the viewer")
                console.print(tbl)
                console.print(
                    "[muted]Docs:[/muted] [accent]https://docs.autocleaneeg.org[/accent]"
                )
                console.print()
            except Exception:
                console.print("Usage: autocleaneeg-pipeline view <file> [--no-view]")
            return False

    elif args.command == "review":
        # Set default output directory if not provided
        if not args.output:
            args.output = user_config.get_default_output_dir()
            message("info", f"Using default workspace output directory: {args.output}")

        if not args.output.exists():
            message("error", f"Output directory does not exist: {args.output}")
            return False

    elif args.command == "exclude":
        # Pre-resolve paths and validate if user provided an explicit target
        try:
            exports_dir, _task_root = _resolve_exclude_paths(args)
            # Only hard-fail early if the user specified a target that doesn't exist
            if getattr(args, "exports", None) or getattr(args, "path", None):
                if not exports_dir.exists():
                    message("error", f"Exports directory does not exist: {exports_dir}")
                    return False
        except Exception:
            # Fallback to runtime validation
            pass

    return True


def cmd_process(args) -> int:
    """Execute the process command."""
    try:
        # Check if Pipeline is available
        if not PIPELINE_AVAILABLE:
            message(
                "error",
                "Pipeline not available. Please ensure autoclean is properly installed.",
            )
            return 1

        # Initialize pipeline with verbose logging if requested
        pipeline_kwargs = {"output_dir": args.output}
        if args.verbose:
            pipeline_kwargs["verbose"] = "debug"

        pipeline = Pipeline(**pipeline_kwargs)

        # Add Python task file if provided
        if args.task_file:
            task_name = pipeline.add_task(args.task_file)
            message("info", f"Loaded Python task: {task_name}")
        else:
            task_name = args.final_task

            # Check if this is a custom task using the new discovery system
            task_class = get_task_by_name(task_name)
            if task_class:
                # Task found via discovery system
                message("info", f"Loaded task: {task_name}")
            else:
                # Fall back to old method for compatibility
                custom_task_path = user_config.get_custom_task_path(task_name)
                if custom_task_path:
                    task_name = pipeline.add_task(custom_task_path)
                    message(
                        "info",
                        f"Loaded custom task '{args.final_task}' from user configuration",
                    )

        if args.dry_run:
            message("info", "DRY RUN - No processing will be performed")
            message("info", f"Would process: {args.final_input}")
            message("info", f"Task: {task_name}")
            message("info", f"Output: {args.output}")
            if args.final_input.is_dir():
                message("info", f"File format: {args.format}")
                if args.recursive:
                    message("info", "Recursive search: enabled")
            return 0

        # Process files
        if args.final_input.is_file():
            message("info", f"Processing single file: {args.final_input}")
            pipeline.process_file(file_path=args.final_input, task=task_name)
        else:
            message("info", f"Processing directory: {args.final_input}")
            message("info", f"Using file format: {args.format}")
            if args.recursive:
                message("info", "Recursive search: enabled")

            # Use parallel processing if requested
            if hasattr(args, "parallel") and args.parallel:
                import asyncio

                max_concurrent = min(max(1, args.parallel), 8)  # Clamp between 1-8
                message(
                    "info", f"Parallel processing: {max_concurrent} concurrent files"
                )
                asyncio.run(
                    pipeline.process_directory_async(
                        directory_path=args.final_input,
                        task=task_name,
                        pattern=args.format,
                        sub_directories=args.recursive,
                        max_concurrent=max_concurrent,
                    )
                )
            else:
                pipeline.process_directory(
                    directory=args.final_input,
                    task=task_name,
                    pattern=args.format,
                    recursive=args.recursive,
                )
        if has_logged_errors():
            message("info", "Processing failed with errors. See log for details.")
            return 1

        message("info", "Processing completed successfully!")
        return 0

    except Exception as e:
        # Check if this is a BlockDependencyError (already displayed with details)
        from autoclean.utils.block_errors import BlockDependencyError

        if isinstance(e, BlockDependencyError):
            # Error already displayed with user-friendly help
            # Exit cleanly without additional error message
            return 1

        # For other exceptions, show error message
        message("error", f"Processing failed: {str(e)}")
        return 1


def cmd_process_ica(args) -> int:
    """Process ICA control sheet updates."""
    try:
        from autoclean.tools.ica import process_ica_control_sheet

        # Prefer ICA control sheet in the ICA directory under task root
        if args.metadata_dir:
            base_path = Path(args.metadata_dir)
            ica_dir = base_path.parent / "ica"
        else:
            ica_dir = user_config.config_dir / "ica"
        control_sheet = ica_dir / "ica_control_sheet.csv"
        # Backwards compatibility: fall back to metadata_dir if provided and file not found
        if args.metadata_dir and not control_sheet.exists():
            legacy_control_sheet = Path(args.metadata_dir) / "ica_control_sheet.csv"
            if legacy_control_sheet.exists():
                control_sheet = legacy_control_sheet
        if not control_sheet.exists():
            message("error", f"ICA control sheet not found: {control_sheet}")
            return 1

        updated = []
        with control_sheet.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manual_add = row.get("manual_add", "").strip()
                manual_drop = row.get("manual_drop", "").strip()
                status = row.get("status", "").strip().lower()
                if manual_add or manual_drop or status == "pending":
                    raw_path = Path(row.get("raw_path") or row.get("raw_file") or "")
                    ica_fif = Path(row.get("ica_file") or row.get("ica_fif") or "")
                    derivatives_dir = Path(row.get("derivatives_dir") or "")
                    unprocessed_file = Path(row.get("unprocessed_file") or "")
                    autoclean_dict = {
                        "ica_dir": ica_dir,
                        "derivatives_dir": derivatives_dir,
                        "unprocessed_file": unprocessed_file,
                    }
                    process_ica_control_sheet(
                        raw_path=raw_path,
                        ica_fif=ica_fif,
                        autoclean_dict=autoclean_dict,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                    )
                    updated.append(str(raw_path))

        if updated:
            message("info", f"Updated {len(updated)} file(s) with ICA edits:")
            for item in updated:
                message("info", f" - {item}")
        else:
            message("info", "No ICA updates needed.")
        return 0

    except ImportError as e:
        message("error", f"ICA tooling not available: {e}")
        return 1
    except Exception as e:
        message("error", f"ICA processing failed: {e}")
        return 1


def cmd_list_tasks(args) -> int:
    """Execute the list-tasks command with unified view and filters."""
    try:
        console = get_console(args)

        # If --overrides flag is specified, show override information
        if hasattr(args, "overrides") and args.overrides:
            overrides = get_task_overrides()

            if not overrides:
                console.print("\n[success]✓[/success] [title]No Task Overrides[/title]")
                console.print(
                    "[muted]All tasks are using their built-in package versions.[/muted]"
                )
                return 0

            console.print(
                f"\n[title]Task Overrides[/title] [muted]({len(overrides)} found)[/muted]\n"
            )

            override_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            override_table.add_column("Task Name", style="accent", no_wrap=True)
            override_table.add_column("Workspace Source", style="info")
            override_table.add_column("Built-in Source", style="muted")
            override_table.add_column("Description", style="muted", max_width=40)

            for override in sorted(overrides, key=lambda x: x.task_name):
                workspace_file = Path(override.workspace_source).name
                builtin_file = Path(override.builtin_source).name
                override_table.add_row(
                    override.task_name,
                    workspace_file,
                    builtin_file,
                    override.description or "No description",
                )

            override_panel = Panel(
                override_table,
                title="[title]Workspace Tasks Overriding Built-in Tasks[/title]",
                border_style="border",
                padding=(1, 1),
            )
            console.print(override_panel)

            console.print(
                "\n[muted]💡 Tip: Move workspace tasks to a different name to use built-in versions.[/muted]"
            )
            return 0

        # Get filters
        source_filter = getattr(args, "source", "all")
        status_filter = getattr(args, "status", "all")
        category_filter = getattr(args, "category", "all")
        output_format = getattr(args, "format", "table")

        # Minimal header
        from rich.align import Align as _Align
        from rich.text import Text as _Text

        if output_format == "table":
            console.print()
            head = _Text()
            head.append("Tasks", style="title")
            console.print(_Align.center(head))
            console.print(
                _Align.center(_Text("Available processing tasks", style="subtitle"))
            )
            console.print()

        registry = BuiltinRegistry()
        registry_info = registry.registry_status()

        if output_format == "table":
            commit_raw = registry_info.get("commit")
            commit = commit_raw or "not yet synced"
            if commit in {"unknown", ""}:
                commit = "not yet synced"
            elif commit == "local-snapshot":
                commit = "local snapshot"
            synced_at = _pretty_timestamp(registry_info.get("synced_at"))
            last_error = registry_info.get("last_error")

            summary_line = f"[info]Task Library version:[/info] {commit} [muted](last checked {synced_at})[/muted]"
            console.print(_Align.center(_Text.from_markup(summary_line)))
            if isinstance(last_error, dict):
                err_msg = last_error.get("message", "connection problem")
                err_time = _pretty_timestamp(
                    last_error.get("timestamp"), default="time not recorded"
                )
                console.print(
                    _Align.center(
                        _Text.from_markup(
                            f"[warning]Last online check failed:[/warning] {err_msg} [muted]{err_time}[/muted]"
                        )
                    )
                )

        # Discover all tasks
        valid_tasks, invalid_files, skipped_files = safe_discover_tasks()
        workspace_dir = user_config.tasks_dir

        def _montage_label(task_name: str) -> str:
            try:
                montage_info = extract_config_from_task(task_name, "montage")
            except Exception:  # pylint: disable=broad-except
                return "—"

            if isinstance(montage_info, dict):
                enabled = montage_info.get("enabled", True)
                value = montage_info.get("value")
                if value is None:
                    return "None"
                label = str(value)
                if not enabled:
                    label = f"{label} [muted](disabled)[/muted]"
                return label

            if montage_info is None:
                return "None"

            return str(montage_info)

        def _get_category(task_name: str) -> str:
            """Determine task category from name."""
            name_lower = task_name.lower()
            if any(x in name_lower for x in ["rest", "resting"]):
                return "resting"
            if any(x in name_lower for x in ["assr", "chirp", "mmn", "auditory"]):
                return "auditory"
            if any(x in name_lower for x in ["vep", "visual"]):
                return "visual"
            if any(x in name_lower for x in ["mouse", "rodent"]):
                return "rodent"
            return "other"

        # Build unified task list
        unified_tasks = []

        # Get library tasks
        library_tasks = registry.list_tasks()
        library_task_names = {t.name for t in library_tasks}

        # Load library index for descriptions
        library_index = {}
        try:
            import json

            if registry._cache_index_path().exists():
                index_data = json.loads(registry._cache_index_path().read_text())
            else:
                index_data = json.loads(registry._pkg_index_text())

            for entry in index_data.get("tasks", []):
                library_index[entry["name"]] = entry
        except Exception:
            pass

        # Process workspace tasks
        workspace_tasks = [t for t in valid_tasks if "autoclean/tasks" not in t.source]
        workspace_names = {t.name for t in workspace_tasks}

        # Process built-in tasks
        builtin_tasks = [t for t in valid_tasks if "autoclean/tasks" in t.source]
        _builtin_names = {t.name for t in builtin_tasks}

        # Add workspace tasks (installed, possibly customized)
        for task in workspace_tasks:
            # Determine source (library or user)
            if task.name in library_task_names:
                source = "library"
                # Only check sync status for library tasks
                sync = registry.task_sync_status(task.name, workspace_dir)
                sync_status = sync.get("status") or "unknown"
            else:
                source = "workspace"
                # User-created tasks don't have sync status
                sync_status = "user_created"

            unified_tasks.append(
                {
                    "name": task.name,
                    "source": source,
                    "sync_status": sync_status,
                    "install_status": "installed",
                    "montage": _montage_label(task.name),
                    "description": task.description or "No description",
                    "category": _get_category(task.name),
                }
            )

        # Add library tasks not in workspace (available to install)
        for ltask in library_tasks:
            if ltask.name not in workspace_names:
                lib_entry = library_index.get(ltask.name, {})
                unified_tasks.append(
                    {
                        "name": ltask.name,
                        "source": "library",
                        "sync_status": "not_installed",
                        "install_status": "available",
                        "montage": "—",
                        "description": lib_entry.get("description", "No description"),
                        "category": _get_category(ltask.name),
                    }
                )

        # Add built-in tasks not overridden by workspace
        for task in builtin_tasks:
            if task.name not in workspace_names:
                unified_tasks.append(
                    {
                        "name": task.name,
                        "source": "builtin",
                        "sync_status": "builtin",
                        "install_status": "available",
                        "montage": _montage_label(task.name),
                        "description": task.description or "No description",
                        "category": _get_category(task.name),
                    }
                )

        # Apply filters
        filtered_tasks = unified_tasks

        if source_filter != "all":
            if source_filter == "active":
                active_task = user_config.get_active_task()
                filtered_tasks = [t for t in filtered_tasks if t["name"] == active_task]
            else:
                filtered_tasks = [
                    t for t in filtered_tasks if t["source"] == source_filter
                ]

        if status_filter != "all":
            status_map = {
                "installed": lambda t: t["install_status"] == "installed",
                "available": lambda t: t["install_status"] == "available",
                "outdated": lambda t: t["sync_status"] in ["outdated", "missing"],
                "customized": lambda t: t["sync_status"] == "modified",
            }
            if status_filter in status_map:
                filtered_tasks = [
                    t for t in filtered_tasks if status_map[status_filter](t)
                ]

        if category_filter != "all":
            filtered_tasks = [
                t for t in filtered_tasks if t["category"] == category_filter
            ]

        # Sort by name
        filtered_tasks.sort(key=lambda t: t["name"])

        # Output in requested format
        if output_format == "json":
            import json

            output_data = {
                "registry_info": registry_info,
                "tasks": filtered_tasks,
                "invalid_files": [
                    {"file": f.source, "error": f.error} for f in invalid_files
                ],
                "skipped_files": [
                    {"file": f.source, "reason": f.reason} for f in skipped_files
                ],
            }
            console.print(json.dumps(output_data, indent=2))
            return 0

        # Table output with unified view
        if filtered_tasks:
            # Separate installed and available
            installed = [
                t for t in filtered_tasks if t["install_status"] == "installed"
            ]
            available = [
                t for t in filtered_tasks if t["install_status"] == "available"
            ]

            # Show installed tasks
            if installed:
                installed_table = Table(
                    show_header=True, header_style="header", box=None, padding=(0, 1)
                )
                installed_table.add_column("Task Name", style="accent", no_wrap=True)
                installed_table.add_column("Source", style="info", no_wrap=True)
                installed_table.add_column("Status", style="info", no_wrap=True)
                installed_table.add_column("Montage", style="info", no_wrap=True)
                installed_table.add_column("Description", style="muted", max_width=50)

                for task in installed:
                    # Source indicator
                    if task["source"] == "library":
                        source_text = "[info]library[/info]"
                    elif task["source"] == "workspace":
                        source_text = "[accent]user[/accent]"
                    else:
                        source_text = "[muted]builtin[/muted]"

                    # Sync status
                    sync_state = task["sync_status"]
                    if sync_state == "synced":
                        status_text = "[success]✓ synced[/success]"
                    elif sync_state == "modified":
                        status_text = "[warning]⚠ customized[/warning]"
                    elif sync_state == "outdated":
                        status_text = "[warning]↻ outdated[/warning]"
                    elif sync_state == "missing":
                        status_text = "[error]✗ orphaned[/error]"
                    elif sync_state == "user_created":
                        status_text = "[success]✓ ready[/success]"
                    else:
                        status_text = "[muted]—[/muted]"

                    installed_table.add_row(
                        task["name"],
                        source_text,
                        status_text,
                        task["montage"],
                        task["description"],
                    )

                installed_panel = Panel(
                    installed_table,
                    title="[title]Installed Tasks (in workspace)[/title]",
                    border_style="border",
                    padding=(1, 1),
                )
                console.print(installed_panel)

            # Show available tasks
            if available:
                available_table = Table(
                    show_header=True, header_style="header", box=None, padding=(0, 1)
                )
                available_table.add_column("Task Name", style="accent", no_wrap=True)
                available_table.add_column("Source", style="info", no_wrap=True)
                available_table.add_column("Montage", style="info", no_wrap=True)
                available_table.add_column("Description", style="muted", max_width=50)

                for task in available:
                    # Source indicator
                    if task["source"] == "library":
                        source_text = "[info]📦 library[/info]"
                    else:
                        source_text = "[muted]builtin[/muted]"

                    available_table.add_row(
                        task["name"],
                        source_text,
                        task["montage"],
                        task["description"],
                    )

                available_panel = Panel(
                    available_table,
                    title="[title]Available to Install[/title]",
                    border_style="border",
                    padding=(1, 1),
                )
                console.print(available_panel)

            # Summary
            summary = _Text()
            summary.append("Found ", style="muted")
            summary.append(str(len(filtered_tasks)), style="accent")
            summary.append(" tasks ", style="muted")
            summary.append(
                f"({len(installed)} installed, {len(available)} available)",
                style="muted",
            )
            if invalid_files or skipped_files:
                summary.append("  •  ", style="muted")
                summary.append(
                    f"{len(skipped_files)} skipped, {len(invalid_files)} invalid",
                    style="muted",
                )
            console.print(_Align.center(summary))
            console.print()

        else:
            console.print(
                Panel(
                    "[muted]No tasks found matching filters.[/muted]",
                    title="[title]Tasks[/title]",
                    border_style="border",
                    padding=(1, 1),
                )
            )

        # Show skipped and invalid files if present
        if skipped_files and output_format == "table":
            skipped_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            skipped_table.add_column("File", style="warning", no_wrap=True)
            skipped_table.add_column("Reason", style="muted", max_width=70)

            for file in skipped_files:
                file_name = Path(file.source).name
                skipped_table.add_row(file_name, file.reason)

            skipped_panel = Panel(
                skipped_table,
                title="[title]Skipped Task Files[/title]",
                border_style="border",
                padding=(1, 1),
            )
            console.print(skipped_panel)

        if invalid_files and output_format == "table":
            invalid_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            invalid_table.add_column("File", style="error", no_wrap=True)
            invalid_table.add_column("Error", style="warning", max_width=70)

            for file in invalid_files:
                file_path = Path(file.source)
                if file_path.is_absolute():
                    display_name = file_path.name
                else:
                    display_name = file.source

                invalid_table.add_row(display_name, file.error)

            invalid_panel = Panel(
                invalid_table,
                title="[title]Invalid Task Files[/title]",
                border_style="border",
                padding=(1, 1),
            )
            console.print(invalid_panel)

        return 0

    except Exception as e:
        message("error", f"Failed to list tasks: {str(e)}")
        return 1


def cmd_blocks(args) -> int:
    """Execute block-related commands."""
    action = getattr(args, "blocks_action", None)
    if not action:
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "blocks")
        return 0

    # Map sub-actions to their handlers to reduce branching and improve maintainability
    _actions = {
        "list": cli_blocks.cmd_blocks_list,
        "info": cli_blocks.cmd_blocks_info,
        "deps": cli_blocks.cmd_blocks_deps,
        "update": cli_blocks.cmd_blocks_update,
        "install": cli_blocks.cmd_blocks_install,
        "lock": cli_blocks.cmd_blocks_lock,
    }
    handler = _actions.get(action)
    if handler is not None:
        return handler(args)

    message("error", f"Unknown blocks action: {action}")
    return 1


def cmd_montage(args) -> int:
    """Execute montage-related commands."""
    action = getattr(args, "montage_action", None)
    if not action:
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "montage")
        return 0

    if action == "list":
        return cmd_montage_list(args)
    if action == "set":
        return cmd_montage_set(args)
    if action == "test":
        return cmd_montage_test(args)

    message("error", f"Unknown montage action: {action}")
    return 1


def cmd_montage_list(args) -> int:
    """List EEG montages defined in the configuration file."""
    console = get_console(args)

    try:
        montages = load_valid_montages()
    except Exception as exc:  # pylint: disable=broad-except
        message("error", f"Failed to load montage definitions: {exc}")
        return 1

    if not montages:
        console.print(
            "[warning]No montage definitions found in configs/montages.yaml[/warning]"
        )
        return 0

    from rich.align import Align
    from rich.text import Text

    console.print()
    console.print(Align.center(Text("EEG Montage Catalog", style="title")))
    console.print(
        Align.center(Text("Defined in configs/montages.yaml", style="subtitle"))
    )
    console.print()

    table = Table(show_header=True, header_style="header", box=None, padding=(0, 1))
    table.add_column("Montage", style="accent", no_wrap=True)
    table.add_column("Description", style="muted")

    for montage_id, description in sorted(montages.items()):
        display_description = description or "[muted]No description provided[/muted]"
        table.add_row(montage_id, display_description)

    console.print(table)
    console.print()
    console.print(
        f"[muted]{len(montages)} montage(s) available · Edit configs/montages.yaml to customize.[/muted]"
    )

    return 0


def cmd_montage_set(args) -> int:
    """Interactively set the montage for the active task."""

    try:
        montages = load_valid_montages()
    except Exception as exc:  # pylint: disable=broad-except
        message("error", f"Failed to load montage definitions: {exc}")
        return 1

    if not montages:
        message(
            "error",
            "No montage definitions found in configs/montages.yaml; nothing to set.",
        )
        return 1

    active_task = user_config.get_active_task()
    if not active_task:
        message("error", "No active task is currently set.")
        message("info", "Use: autocleaneeg-pipeline task set")
        return 1

    task_path = user_config.get_custom_task_path(active_task)
    if task_path is None or not task_path.exists():
        message(
            "error",
            (
                f"Active task '{active_task}' is not a workspace task and cannot be edited."
            ),
        )
        message(
            "info",
            "Copy the task into your workspace with: autocleaneeg-pipeline task copy",
        )
        return 1

    try:
        with task_path.open("r", encoding="utf-8", newline="") as f:
            task_source = f.read()
    except Exception as exc:  # pylint: disable=broad-except
        message("error", f"Failed to read task file '{task_path.name}': {exc}")
        return 1

    block_text, block_start, block_end = _locate_montage_block(task_source)
    if block_text is None:
        message(
            "error",
            f"Task '{active_task}' does not define a montage block that can be edited.",
        )
        return 1

    current_value = _extract_montage_value(block_text)

    selected_montage = getattr(args, "montage_name", None)
    if selected_montage:
        if selected_montage not in montages:
            message(
                "error",
                f"Montage '{selected_montage}' not found. Run 'montage list' to view options.",
            )
            return 1
    else:
        selected_montage = _prompt_for_montage(args, montages, current_value)
        if not selected_montage:
            message("info", "No montage selected.")
            return 0

    if current_value == selected_montage:
        message(
            "info",
            f"Montage is already set to '{selected_montage}'. No changes needed.",
        )
        return 0

    confirmed = args.force
    if not confirmed:
        try:
            from rich.prompt import Confirm

            prompt = (
                f"Update montage in {task_path.name} from "
                f"'{current_value or 'None'}' to '{selected_montage}'?"
            )
            confirmed = Confirm.ask(prompt, default=True)
        except Exception:
            response = input(
                f"Apply montage '{selected_montage}' to {task_path.name}? [y/N] "
            ).strip()
            confirmed = response.lower() in {"y", "yes"}

    if not confirmed:
        message("info", "Aborted. No changes made.")
        return 0

    updated_block = _replace_montage_value(block_text, selected_montage)
    if updated_block is None:
        message(
            "error",
            "Failed to update montage value; the task file may have unexpected formatting.",
        )
        return 1

    updated_source = task_source[:block_start] + updated_block + task_source[block_end:]

    try:
        with task_path.open("w", encoding="utf-8", newline="") as f:
            f.write(updated_source)
    except Exception as exc:  # pylint: disable=broad-except
        message("error", f"Failed to write task file '{task_path.name}': {exc}")
        return 1

    message(
        "success",
        f"✓ Montage for '{active_task}' updated to '{selected_montage}' in {task_path.name}",
    )
    return 0


def _load_xdat_via_neo(file_path: Path, montage_name: str = "MouseEEGv2_H32"):
    """Load XDAT file using plugin system.

    Args:
        file_path: Path to the .xdat file
        montage_name: Montage to apply (default: MouseEEGv2_H32)

    Returns:
        MNE Raw object with probe mapping applied via plugin

    Raises:
        ImportError: If Neo is not available
        Exception: If file loading fails or plugin not found
    """
    from autoclean.io.import_ import get_plugin_for_combination

    # Try to get the XDAT + montage plugin
    try:
        plugin = get_plugin_for_combination("NEURONEXUS_XDAT", montage_name)
        raw = plugin.import_and_configure(file_path, autoclean_dict={}, preload=True)
        return raw
    except ValueError as e:
        # Plugin not found - fall back to basic Neo loading
        message("warning", f"No plugin found for XDAT + {montage_name}: {e}")
        message("info", "Falling back to basic XDAT loading without montage mapping")

        if not NEO_AVAILABLE:
            raise ImportError(
                "Neo package is required for XDAT files. "
                "Install with: pip install neo"
            )

        import mne
        import numpy as np

        # Basic Neo loading without mapping
        stem = file_path.stem
        if stem.endswith("_data"):
            base_stem = stem[:-5]
            json_file = file_path.parent / f"{base_stem}.xdat.json"
        elif stem.endswith("_timestamp"):
            base_stem = stem[:-10]
            json_file = file_path.parent / f"{base_stem}.xdat.json"
        else:
            json_file = file_path.with_suffix(".xdat.json")

        reader_file = str(json_file) if json_file.exists() else str(file_path)

        reader = neo.io.NeuroNexusIO(filename=reader_file)
        block = reader.read_block()
        segment = block.segments[0]

        all_data = []
        ch_names = []
        ch_types_list = []
        sfreq = None

        for analog_signal in segment.analogsignals:
            sig_data = analog_signal.magnitude.T
            all_data.append(sig_data)

            if sfreq is None:
                sfreq = float(analog_signal.sampling_rate.magnitude)

            for ch_name in analog_signal.array_annotations.get("channel_names", []):
                ch_names.append(str(ch_name))
                ch_name_lower = ch_name.lower()
                if "din" in ch_name_lower or "dout" in ch_name_lower:
                    ch_types_list.append("stim")
                elif "aux" in ch_name_lower:
                    ch_types_list.append("misc")
                else:
                    ch_types_list.append("eeg")

        data = np.vstack(all_data)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types_list)
        raw = mne.io.RawArray(data, info)

        return raw


def cmd_montage_test(args) -> int:
    """Generate montage validation report for the active input file."""
    from pathlib import Path

    import mne

    console = get_console(args)

    # Header
    console.print()
    console.print("[bold cyan]🧭 Montage Validation Test[/bold cyan]")
    console.print()

    # Get active input file
    active_source = user_config.get_active_source()
    if not active_source:
        message("error", "No active input file is currently set.")
        message("info", "Use: autocleaneeg-pipeline input select")
        return 1

    input_file = Path(active_source)
    if not input_file.exists():
        message("error", f"Active input file does not exist: {input_file}")
        return 1

    # Get active task
    active_task = user_config.get_active_task()
    if not active_task:
        message("error", "No active task is currently set.")
        message("info", "Use: autocleaneeg-pipeline task set")
        return 1

    # Get montage from active task
    task_path = user_config.get_custom_task_path(active_task)
    if task_path is None or not task_path.exists():
        message(
            "error",
            f"Active task '{active_task}' is not a workspace task.",
        )
        return 1

    try:
        with task_path.open("r", encoding="utf-8", newline="") as f:
            task_source = f.read()
    except Exception as exc:
        message("error", f"Failed to read task file: {exc}")
        return 1

    block_text, _, _ = _locate_montage_block(task_source)
    if block_text is None:
        message(
            "error",
            f"Task '{active_task}' does not define a montage block.",
        )
        return 1

    montage_name = _extract_montage_value(block_text)
    if not montage_name:
        message("error", "No montage is configured in the active task.")
        message("info", "Use: autocleaneeg-pipeline montage set")
        return 1

    # Print configuration
    console.print(f"[info]Input file:[/info] {input_file.name}")
    console.print(f"[info]Active task:[/info] {active_task}")
    console.print(f"[info]Montage:[/info] {montage_name}")
    console.print()

    # Load EEG file
    console.print("[info]Loading EEG file...[/info]")
    try:
        # Determine file type and load accordingly
        suffix = input_file.suffix.lower()
        if suffix == ".bdf":
            raw = mne.io.read_raw_bdf(
                str(input_file),
                preload=True,
                stim_channel="auto",
                exclude=[],
                verbose=False,
            )
        elif suffix == ".edf":
            raw = mne.io.read_raw_edf(str(input_file), preload=True, verbose=False)
        elif suffix == ".set":
            raw = mne.io.read_raw_eeglab(str(input_file), preload=True, verbose=False)
        elif suffix == ".fif":
            raw = mne.io.read_raw_fif(str(input_file), preload=True, verbose=False)
        elif suffix == ".xdat":
            raw = _load_xdat_via_neo(input_file)
        else:
            message("error", f"Unsupported file format: {suffix}")
            message("info", "Supported formats: .bdf, .edf, .set, .fif, .xdat")
            return 1
    except Exception as exc:
        message("error", f"Failed to load EEG file: {exc}")
        return 1

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    n_eeg_channels = len(eeg_picks)
    console.print(f"[success]✓[/success] Loaded {n_eeg_channels} EEG channels")

    # Check if a plugin exists for this format/montage combination
    # Plugins can perform complex transformations (remapping, dropping channels, etc.)
    from autoclean.io.import_ import (
        discover_plugins,
        find_plugin_for_combination,
        get_format_from_extension,
    )

    # Ensure plugins are discovered so format registry is populated
    discover_plugins()

    # Map file suffix to format_id (use get_format_from_extension for consistency)
    format_id = get_format_from_extension(suffix)
    if not format_id:
        # Fallback for unknown formats
        format_id = suffix.upper().lstrip(".")

    plugin = find_plugin_for_combination(format_id, montage_name)
    plugin_report_data = {}

    if plugin:
        console.print(f"[info]Using plugin: {plugin.__class__.__name__}[/info]")

        # Save copy of raw data before plugin processing
        raw_before = raw.copy()

        # Process through plugin
        try:
            raw_after = plugin.import_and_configure(input_file, {}, preload=True)
            raw = raw_after  # Use plugin-processed data for the rest of the analysis

            eeg_picks = mne.pick_types(raw.info, eeg=True)
            n_eeg_after = len(eeg_picks)
            console.print(
                f"[success]✓[/success] Plugin processed: {n_eeg_after} EEG channels"
            )

            # Generate custom report if plugin supports it
            if hasattr(plugin, "generate_montage_report"):
                console.print("[info]Generating plugin-specific report...[/info]")
                # Note: montage will be loaded below, so we'll call this after montage loading
                # Store plugin and raw_before for later use
                plugin_report_data = {
                    "plugin": plugin,
                    "raw_before": raw_before,
                    "has_custom_report": True,
                }
        except Exception as exc:
            console.print(f"[yellow]⚠[/yellow] Plugin processing failed: {exc}")
            console.print("[info]Falling back to standard loading...[/info]")
            # Continue with raw data from standard loading

    # Determine rename map for BDF files
    rename_map = {}
    if suffix == ".bdf":
        for ch in raw.ch_names:
            if ch != "Status" and "_" in ch:
                _, std_name = ch.split("_", 1)
                if std_name == "Afz":
                    std_name = "AFz"
                rename_map[ch] = std_name

    # Load montage (try standard first, then custom)
    try:
        # Try loading as standard MNE montage
        montage = mne.channels.make_standard_montage(montage_name)
        montage_channels = set(montage.get_positions()["ch_pos"].keys())
    except ValueError:
        # Not a standard montage, try loading custom montage
        try:
            from pathlib import Path as PathLib

            # Check in package data/montages directory
            # cli.py is in src/autoclean/cli.py, so parent is src/autoclean
            package_dir = PathLib(__file__).parent
            custom_montage_file = (
                package_dir / "data" / "montages" / f"{montage_name}.sfp"
            )

            if custom_montage_file.exists():
                message(
                    "info", f"Loading custom montage from {custom_montage_file.name}"
                )
                montage = mne.channels.read_custom_montage(str(custom_montage_file))
                montage_channels = set(montage.get_positions()["ch_pos"].keys())
            else:
                message(
                    "error",
                    f"Montage '{montage_name}' not found (checked standard and custom)",
                )
                return 1
        except Exception as exc:
            message("error", f"Failed to load custom montage '{montage_name}': {exc}")
            return 1
    except Exception as exc:
        message("error", f"Failed to load montage '{montage_name}': {exc}")
        return 1

    # Extract raw channel info BEFORE renaming
    console.print("[info]Analyzing channel positions...[/info]")
    raw_channel_info = extract_channel_info(raw, rename_map, montage_channels)

    # Extract metadata
    metadata = extract_file_metadata(raw, str(input_file))

    # Apply renaming and montage
    if rename_map:
        raw.rename_channels(rename_map)

    raw.set_montage(montage, match_case=False, on_missing="warn")

    # Analyze channels (pass montage_name for mouse-scale detection)
    analysis = analyze_channels(raw, montage, montage_name=montage_name)

    # Get alternative montage suggestions
    suggestions = suggest_montages(analysis["file_channels"])

    # Generate visualizations
    console.print("[info]Generating visualizations...[/info]")
    plot3d_b64 = create_3d_plot(analysis, f"3D Electrode Positions - {montage_name}")
    stats_b64 = create_stats_plot(analysis)

    # Determine output directory and file path
    default_output = user_config.get_default_output_dir()
    task_root = default_output / active_task

    qa_dir = task_root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    report_file = qa_dir / "montage_validation_report.html"

    # Generate plugin-specific report if available
    custom_report_sections = []
    if plugin_report_data.get("has_custom_report"):
        try:
            plugin = plugin_report_data["plugin"]
            raw_before = plugin_report_data["raw_before"]
            custom_report = plugin.generate_montage_report(
                raw_before, raw, montage, analysis
            )
            custom_report_sections = custom_report.get("html_sections", [])

            # Add custom stats to analysis
            if "summary_stats" in custom_report:
                analysis["plugin_stats"] = custom_report["summary_stats"]

            # Display info messages
            for info_msg in custom_report.get("info_messages", []):
                console.print(f"[info]{info_msg}[/info]")

            # Display warnings
            for warning in custom_report.get("warnings", []):
                console.print(f"[yellow]⚠ {warning}[/yellow]")

        except Exception as exc:
            console.print(f"[yellow]⚠[/yellow] Plugin report generation failed: {exc}")

    # Generate HTML report
    console.print("[info]Generating HTML report...[/info]")
    generate_html_report(
        str(input_file),
        montage_name,
        analysis,
        suggestions,
        report_file,
        None,  # topomap_b64 (not used)
        plot3d_b64,
        stats_b64,
        metadata,
        raw_channel_info,
        rename_map,
        custom_report_sections,
    )

    console.print()
    console.print(f"[success]✓ Report saved:[/success] {report_file}")
    console.print()
    console.print(
        f"Match: [{'green' if analysis['match_pct'] >= 95 else 'yellow' if analysis['match_pct'] >= 90 else 'red'}]{analysis['match_pct']:.1f}%[/] ({len(analysis['matched'])}/{len(analysis['file_channels'])} channels)"
    )
    console.print(
        f"Positioned: {analysis['n_positioned']}/{len(analysis['file_channels'])}"
    )
    console.print()

    return 0


def _prompt_for_montage(args, montages, current_value):
    """Prompt the user to choose a montage interactively."""

    items = sorted(montages.items())

    try:
        from rich.prompt import Prompt

        console = get_console(args)
        console.print()

        table = Table(show_header=True, header_style="header", box=None, padding=(0, 1))
        table.add_column("#", style="muted", width=3)
        table.add_column("Montage", style="accent", no_wrap=True)
        table.add_column("Description", style="muted")

        for idx, (name, desc) in enumerate(items, 1):
            descriptor = desc or "No description"
            if current_value and name == current_value:
                descriptor = f"{descriptor} [muted](current)[/muted]"
            table.add_row(str(idx), name, descriptor)

        console.print(table)

        current_msg = f" (current: {current_value})" if current_value else ""
        console.print(
            f"Select a montage by number or name{current_msg}. Press Enter to cancel."
        )

        choice = Prompt.ask("Choice", default="", show_default=False)
        if not choice.strip():
            return None

        choice = choice.strip()
        # Allow direct montage name entry
        if choice in montages:
            return choice

        try:
            idx = int(choice)
        except ValueError:
            console.print("[error]Invalid selection.[/error]")
            return None

        if 1 <= idx <= len(items):
            return items[idx - 1][0]

        console.print("[error]Selection out of range.[/error]")
        return None

    except Exception:
        print()
        for idx, (name, desc) in enumerate(items, 1):
            descriptor = desc or "No description"
            marker = " (current)" if current_value and name == current_value else ""
            print(f"  {idx}. {name} - {descriptor}{marker}")
        choice = input("Choice (blank to cancel): ").strip()
        if not choice:
            return None
        if choice in montages:
            return choice
        try:
            idx = int(choice)
        except ValueError:
            print("Invalid selection.")
            return None
        if 1 <= idx <= len(items):
            return items[idx - 1][0]
        print("Selection out of range.")
        return None


def _locate_montage_block(text: str):
    """Find the montage configuration block in a task file."""

    match = re.search(r"[\"']montage[\"']\s*:\s*\{", text)
    if not match:
        return None, None, None

    brace_start = text.find("{", match.start())
    if brace_start == -1:
        return None, None, None

    depth = 0
    in_string: Optional[str] = None
    escape = False

    for idx in range(brace_start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
        else:
            if char in {'"', "'"}:
                in_string = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    block_end = idx + 1
                    block_start = match.start()
                    return text[block_start:block_end], block_start, block_end

    return None, None, None


def _extract_montage_value(block_text: str) -> Optional[str]:
    """Extract the montage value from the montage block."""

    value_match = re.search(
        r"([\"\']value[\"\']\s*:\s*)(?P<quote>[\"\'])(?P<val>.*?)(?P=quote)",
        block_text,
        re.DOTALL,
    )
    if value_match:
        return value_match.group("val")

    none_match = re.search(r"[\"\']value[\"\']\s*:\s*None", block_text)
    if none_match:
        return None

    return None


def _replace_montage_value(block_text: str, new_value: str) -> Optional[str]:
    """Return a montage block with the value replaced."""

    string_match = re.search(
        r"([\"\']value[\"\']\s*:\s*)(?P<quote>[\"\'])(?P<val>.*?)(?P=quote)",
        block_text,
        re.DOTALL,
    )
    if string_match:
        prefix = block_text[: string_match.start()]
        suffix = block_text[string_match.end() :]
        quote = string_match.group("quote")
        replacement = f"{string_match.group(1)}{quote}{new_value}{quote}"
        return prefix + replacement + suffix

    none_match = re.search(r"([\"\']value[\"\']\s*:\s*)None", block_text)
    if none_match:
        prefix = block_text[: none_match.start()]
        suffix = block_text[none_match.end() :]
        replacement = f'{none_match.group(1)}"{new_value}"'
        return prefix + replacement + suffix

    return None


def cmd_review(args) -> int:
    """Execute the review command."""
    try:
        # Check if Pipeline is available
        if not PIPELINE_AVAILABLE:
            message(
                "error",
                "Pipeline not available. Please ensure autoclean is properly installed.",
            )
            return 1

        pipeline = Pipeline(output_dir=args.output)

        message("info", f"Starting review GUI for: {args.output}")
        pipeline.start_autoclean_review()

        return 0

    except Exception as e:
        message("error", f"Failed to start review GUI: {str(e)}")
        return 1


def _resolve_exclude_paths(args) -> tuple[Path, Optional[Path]]:
    """Resolve exports and task root paths for the exclude GUI.

    Mirrors the logic in autoclean.tools.autoclean_exclude.determine_paths but
    kept inline to avoid importing GUI deps at module import time.
    """
    # --exports takes precedence
    if getattr(args, "exports", None):
        exports_dir = Path(args.exports).expanduser().resolve()
        task_root = (
            Path(args.task_root).expanduser().resolve()
            if getattr(args, "task_root", None)
            else exports_dir.parent
        )
        return exports_dir, task_root

    # Positional path can be a run folder or an exports folder
    if getattr(args, "path", None):
        candidate = Path(args.path).expanduser().resolve()
        exports_candidate = candidate / "exports"
        if exports_candidate.exists():
            return exports_candidate, candidate
        return candidate, candidate.parent if candidate.parent.exists() else None

    # Default: prefer most recent task exports in workspace, then ./exports, then current directory
    workspace_output = user_config.get_default_output_dir()
    if workspace_output.exists():
        # Find the most recent task directory in workspace output
        task_dirs = [d for d in workspace_output.iterdir() if d.is_dir()]
        if task_dirs:
            # Sort by modification time, most recent first
            task_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            most_recent_task = task_dirs[0]
            task_exports = most_recent_task / "exports"
            if task_exports.exists():
                return task_exports, most_recent_task

    cwd = Path.cwd()
    default_exports = cwd / "exports"
    if default_exports.exists():
        return default_exports, cwd
    return cwd, cwd


def cmd_exclude(args) -> int:
    """Launch the inclusion/exclusion GUI via CLI command."""
    try:
        exports_dir, task_root = _resolve_exclude_paths(args)
        if not exports_dir.exists():
            message("error", f"Exports directory not found: {exports_dir}")
            return 1

        # Import lazily to avoid requiring GUI deps for non-GUI commands
        from autoclean.tools.autoclean_exclude import (
            run_autoclean_exclusion_tool,
        )

        message("info", f"Starting exclusion GUI at: {exports_dir}")
        run_autoclean_exclusion_tool(exports_dir=exports_dir, task_root=task_root)
        return 0
    except SystemExit as e:
        # Propagate exit codes from the underlying tool cleanly
        return int(getattr(e, "code", 1) or 0)
    except Exception as e:
        message("error", f"Failed to start exclusion GUI: {str(e)}")
        return 1


def cmd_workspace(args) -> int:
    """Workspace command dispatcher and helpers."""
    # No subcommand → show elegant workspace help
    if not getattr(args, "workspace_action", None):
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "workspace")
        return 0

    action = args.workspace_action
    if action == "explore":
        return cmd_workspace_explore(args)
    if action == "show":
        return cmd_workspace_show(args)
    if action == "size":
        return cmd_workspace_size(args)
    if action == "set":
        return cmd_workspace_set(args)
    if action == "unset":
        return cmd_workspace_unset(args)
    if action == "default":
        return cmd_workspace_default(args)
    if action == "cd":
        return cmd_workspace_cd(args)
    message("error", f"Unknown workspace action: {action}")
    return 1


def cmd_workspace_explore(_args) -> int:
    """Open the workspace directory in the system file browser."""
    try:
        ws = user_config.config_dir
        ws.mkdir(parents=True, exist_ok=True)
        message("info", f"Opening workspace folder: {ws}")
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", str(ws)], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(str(ws))  # type: ignore[attr-defined]
            else:
                if shutil.which("xdg-open"):
                    subprocess.run(["xdg-open", str(ws)], check=False)
                else:
                    print(str(ws))
        except Exception:
            print(str(ws))
        return 0
    except Exception as e:
        message("error", f"Failed to open workspace folder: {e}")
        return 1


def cmd_workspace_show(_args) -> int:
    """Show the current workspace path and whether it's configured/valid."""
    try:
        from rich.align import Align as _Align
        from rich.text import Text as _Text

        console = get_console()

        # Title (minimal, no product banner)
        title = _Text()
        console.print()
        title.append("AutocleanEEG Pipeline Workspace", style="title")
        console.print(_Align.center(title))
        console.print(_Align.center(_Text("Current configuration", style="subtitle")))
        console.print()

        # Determine validity
        ws = user_config.config_dir
        try:
            is_valid = user_config._is_workspace_valid()  # type: ignore[attr-defined]
        except Exception:
            is_valid = ws.exists() and (ws / "tasks").exists()

        # Workspace line
        home = str(Path.home())
        display_ws = str(ws)
        if display_ws.startswith(home):
            display_ws = display_ws.replace(home, "~", 1)
        ws_line = _Text()
        if is_valid:
            ws_line.append("✓ ", style="success")
            ws_line.append("Workspace ", style="muted")
            ws_line.append(display_ws, style="accent")
        else:
            ws_line.append("⚠ ", style="warning")
            ws_line.append("Workspace not configured — ", style="muted")
            ws_line.append(display_ws, style="accent")
        console.print(_Align.center(ws_line))

        # Active task (only if set)
        try:
            active_task = user_config.get_active_task()
            if active_task:
                at = _Text()
                at.append("🎯 ", style="muted")
                at.append("Active task: ", style="muted")
                at.append(str(active_task), style="accent")
                console.print(_Align.center(at))
        except Exception:
            pass

        # Active input (only if set)
        try:
            active_source = user_config.get_active_source()
            if active_source and active_source != "NONE":
                sp = Path(active_source)
                display_src = str(sp)
                if display_src.startswith(home):
                    display_src = display_src.replace(home, "~", 1)
                src = _Text()
                if sp.exists():
                    if sp.is_file():
                        src.append("📄 ", style="muted")
                        src.append("Input file: ", style="muted")
                    elif sp.is_dir():
                        src.append("📂 ", style="muted")
                        src.append("Input folder: ", style="muted")
                    else:
                        src.append("📁 ", style="muted")
                        src.append("Input: ", style="muted")
                else:
                    src.append("⚠ ", style="warning")
                    src.append("Input missing — ", style="muted")
                src.append(display_src, style="accent")
                console.print(_Align.center(src))
        except Exception:
            pass

        # Free disk (helpful context, keep minimal)
        try:
            usage_path = (
                ws
                if ws.exists()
                else (ws.parent if ws.parent.exists() else Path.home())
            )
            du = shutil.disk_usage(str(usage_path))
            free_gb = du.free / (1024**3)
            free_line = _Text()
            free_line.append("💾 ", style="muted")
            free_line.append("Free space ", style="muted")
            free_line.append(f"{free_gb:.1f} GB", style="accent")
            console.print(_Align.center(free_line))
        except Exception:
            pass

        console.print()
        # If invalid, provide a concise tip
        if not is_valid:
            tip = _Text()
            tip.append("Run ", style="muted")
            tip.append("autocleaneeg-pipeline workspace", style="accent")
            tip.append(" to configure.", style="muted")
            console.print(_Align.center(tip))
            console.print()

        return 0
    except Exception as e:
        message("error", f"Failed to show workspace: {e}")
        return 1


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                continue
    except Exception:
        pass
    return total


def _fmt_bytes(n: int) -> str:
    gb = n / (1024**3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = n / (1024**2)
    if mb >= 1:
        return f"{mb:.2f} MB"
    kb = n / 1024
    if kb >= 1:
        return f"{kb:.2f} KB"
    return f"{n} B"


def cmd_workspace_size(_args) -> int:
    """Show total workspace size."""
    try:
        ws = user_config.config_dir
        size_b = _dir_size_bytes(ws) if ws.exists() else 0
        console = get_console()
        from rich.align import Align as _Align
        from rich.text import Text as _Text

        line = _Text()
        line.append("📂 ", style="muted")
        line.append("Workspace: ", style="muted")
        line.append(str(ws), style="accent")
        console.print(_Align.center(line))

        size_line = _Text()
        size_line.append("Total size: ", style="muted")
        size_line.append(_fmt_bytes(size_b), style="accent")
        console.print(_Align.center(size_line))
        console.print()
        return 0
    except Exception as e:
        message("error", f"Failed to compute workspace size: {e}")
        return 1


def cmd_workspace_set(args) -> int:
    """Change the workspace folder to the given path and initialize structure."""
    try:
        # If no path provided, enter interactive mode to choose a folder
        if not getattr(args, "path", None):
            # Directly invoke the reconfiguration wizard to avoid duplicate prompts
            try:
                chosen = user_config._run_setup_wizard(
                    is_first_time=False, show_branding=False
                )
            except Exception:
                # Fallback to standard setup if private API unavailable
                chosen = user_config.setup_workspace(show_branding=False)
            message("success", "✓ Workspace updated")
            message("info", str(chosen))
            return 0

        new_path = args.path.expanduser().resolve()
        new_path.mkdir(parents=True, exist_ok=True)
        # Initialize structure and save config
        user_config._save_global_config(new_path)
        user_config._create_workspace_structure(new_path)
        # Update current instance
        user_config.config_dir = new_path
        user_config.tasks_dir = new_path / "tasks"
        message("success", "✓ Workspace updated")
        message("info", str(new_path))
        return 0
    except Exception as e:
        message("error", f"Failed to set workspace: {e}")
        return 1


def cmd_workspace_unset(_args) -> int:
    """Unassign current workspace by clearing saved config."""
    try:
        import platformdirs  # local import to avoid global dep in CLI

        cfg = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        if cfg.exists():
            cfg.unlink()
            message("success", "✓ Workspace unassigned (config cleared)")
        else:
            message("info", "No saved workspace configuration found")
        # Reset in-memory paths to default suggestion
        user_config.config_dir = user_config._get_workspace_path()
        user_config.tasks_dir = user_config.config_dir / "tasks"
        return 0
    except Exception as e:
        message("error", f"Failed to unset workspace: {e}")
        return 1


def cmd_workspace_default(_args) -> int:
    """Set workspace to the cross-platform default documents path."""
    try:
        import platformdirs  # local import to avoid global dep at module import

        default_path = Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"
        default_path.mkdir(parents=True, exist_ok=True)

        # Initialize structure and save config
        user_config._save_global_config(default_path)
        user_config._create_workspace_structure(default_path)

        # Update current instance
        user_config.config_dir = default_path
        user_config.tasks_dir = default_path / "tasks"

        message("success", "✓ Workspace set to default location")
        message("info", str(default_path))
        return 0
    except Exception as e:
        message("error", f"Failed to set default workspace: {e}")
        return 1


def _detect_shell() -> list:
    """Return command list for user's interactive shell."""

    if sys.platform.startswith("win"):
        # Prefer PowerShell if available
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if pwsh:
            return [pwsh]
        return [os.environ.get("COMSPEC", "cmd")]

    # Unix-like
    shell = os.environ.get("SHELL")
    if shell:
        return [shell]
    return ["/bin/sh"]


def _detect_shell_kind() -> str:
    """Best-effort detection of user's shell kind for printing snippets."""

    if sys.platform.startswith("win"):
        # Heuristic: if running inside PowerShell, PSModulePath is usually set
        if os.environ.get("PSModulePath"):
            return "powershell"
        return "cmd"

    sh = os.environ.get("SHELL", "")
    if "fish" in sh:
        return "fish"
    if "zsh" in sh:
        return "zsh"
    if "bash" in sh:
        return "bash"
    return "bash"


def _esc_for_bash_zsh(path: str) -> str:
    return path.replace("'", "'\"'\"'")


def _esc_for_fish(path: str) -> str:
    return path.replace('"', '\\"')


def _esc_for_powershell(path: str) -> str:
    return path.replace("'", "''")


def _esc_for_cmd(path: str) -> str:
    return path.replace('"', '""')


def cmd_workspace_cd(args) -> int:
    """Change directory to the workspace.

    Default behavior prints the absolute path to stdout so users can:
      cd "$(autocleaneeg-pipeline workspace cd)"

    With --spawn, launches a new interactive shell in that directory.
    """
    try:
        ws = user_config.config_dir
        ws.mkdir(parents=True, exist_ok=True)

        if getattr(args, "spawn", False):
            shell_cmd = _detect_shell()
            message("info", f"Spawning shell in: {ws}")
            try:
                subprocess.call(shell_cmd, cwd=str(ws))
            except OSError as exc:
                message("error", f"Failed to spawn shell: {exc}")
                return 1
            return 0

        # Optional: print a shell-specific snippet for eval
        if getattr(args, "print", None):
            kind = args.print if args.print != "auto" else _detect_shell_kind()
            p = str(ws)
            if kind in ("bash", "zsh"):
                print(f"cd '{_esc_for_bash_zsh(p)}'")
            elif kind == "fish":
                print(f'cd "{_esc_for_fish(p)}"')
            elif kind == "powershell":
                print(f"Set-Location -Path '{_esc_for_powershell(p)}'")
            else:  # cmd
                print(f'cd /D "{_esc_for_cmd(p)}"')
            return 0

        # Default: print path only (no styling) for command substitution
        print(str(ws))
        return 0
    except (OSError, RuntimeError) as exc:
        message("error", f"Failed to resolve workspace directory: {exc}")
        return 1


def _simple_header(
    console, title: Optional[str] = None, subtitle: Optional[str] = None
):
    """Simple, consistent header for setup."""
    from rich.align import Align
    from rich.text import Text

    console.print()

    # Create branding content (no borders; app name with version)
    branding_text = Text()
    branding_text.append(
        f"{LOGO_ICON} AutocleanEEG Pipeline ({__version__})", style="brand"
    )
    branding_text.append(f"\n{TAGLINE}", style="accent")

    # Print centered branding (no borders)
    console.print(Align.center(branding_text))
    console.print()
    if title:
        console.print(f"[title]{title}[/title]")
    if subtitle:
        console.print(f"[subtitle]{subtitle}[/subtitle]")
    console.print()


def _wizard_collect_state() -> (
    Tuple[Path, bool, Optional[str], Optional[Path], Optional[str], Optional[str]]
):
    """Gather current workspace, task, montage, and input state for summaries."""

    workspace_dir = user_config.config_dir
    try:
        workspace_valid = user_config._is_workspace_valid()  # type: ignore[attr-defined]
    except Exception:
        workspace_valid = workspace_dir.exists() and (workspace_dir / "tasks").exists()

    active_task = None
    active_task_path: Optional[Path] = None
    try:
        active_task = user_config.get_active_task()
        if active_task:
            candidate = user_config.get_custom_task_path(active_task)
            if candidate and candidate.exists():
                active_task_path = candidate
    except Exception:
        active_task = None

    active_montage: Optional[str] = None
    if active_task_path:
        try:
            source = active_task_path.read_text(encoding="utf-8")
            block, _, _ = _locate_montage_block(source)
            if block:
                active_montage = _extract_montage_value(block)
        except Exception:
            active_montage = None

    active_input: Optional[str] = None
    try:
        active_input = user_config.get_active_source()
    except Exception:
        active_input = None

    return (
        workspace_dir,
        workspace_valid,
        active_task,
        active_task_path,
        active_montage,
        active_input,
    )


def _wizard_render_state(display: CLIDisplay, title: str) -> None:
    """Render a compact table showing the current configuration state."""

    (
        workspace_dir,
        workspace_valid,
        active_task,
        active_task_path,
        active_montage,
        active_input,
    ) = _wizard_collect_state()

    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Setting", style="accent", no_wrap=True)
    table.add_column("Status", style="muted")

    home = str(Path.home())
    workspace_display = str(workspace_dir)
    if workspace_display.startswith(home):
        workspace_display = workspace_display.replace(home, "~", 1)

    status_icon = "✓" if workspace_valid else "⚠"
    table.add_row(
        "Workspace",
        (
            f"{status_icon} {workspace_display}"
            if workspace_valid
            else f"⚠ {workspace_display} [warning](needs setup)[/warning]"
        ),
    )

    if active_task:
        location = (
            f"[muted]({active_task_path.name})[/muted]"
            if active_task_path is not None
            else "[warning](built-in)[/warning]"
        )
        table.add_row("Task", f"🎯 {active_task} {location}")
    else:
        table.add_row("Task", "[warning]not selected[/warning]")

    if active_montage:
        table.add_row("Montage", f"🧭 {active_montage}")
    elif active_task_path is None and active_task:
        table.add_row(
            "Montage",
            "[warning]cannot edit built-in tasks[/warning]",
        )
    else:
        table.add_row("Montage", "[warning]not configured[/warning]")

    if active_input:
        display_path = active_input
        if display_path.startswith(home):
            display_path = display_path.replace(home, "~", 1)
        table.add_row("Input", f"📁 {display_path}")
    else:
        table.add_row("Input", "[muted]prompt each run[/muted]")

    display.blank_line()
    display.header(title, style="title")
    display.console.print(table)
    display.blank_line()


def _wizard_create_task_from_template(
    display: CLIDisplay, class_name: str, file_name: str, *, overwrite: bool = False
) -> Path:
    """Create a custom task file from the bundled template."""

    templates_dir = Path(__file__).resolve().parent / "templates"
    template_path = templates_dir / "custom_task_template.jinja"
    target_path = user_config.tasks_dir / f"{file_name}.py"

    try:
        validate_python_identifier(class_name, label="Task class name")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    if not template_path.is_file():
        raise RuntimeError(f"Task template not found at {template_path}")

    try:
        customized = render_template(template_path, {"class_name": class_name})
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to materialize task template: {exc}") from exc

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to prepare task directory: {exc}") from exc

    if overwrite:
        try:
            target_path.write_text(customized, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to write task file: {exc}") from exc
    else:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(target_path, flags)
        except FileExistsError as exc:
            raise RuntimeError(f"Task file already exists at {target_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to create task file: {exc}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(customized)

    display.success(
        "Custom task created",
        f"Saved template to {target_path}",
    )

    return target_path


def _wizard_normalize_class_name(raw: str) -> str:
    """Normalize arbitrary user input into a valid Task class name."""

    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", raw).strip()
    parts = [part.capitalize() for part in cleaned.split() if part]
    if not parts:
        parts = ["Custom", "Task"]
    candidate = "".join(parts)
    if candidate and candidate[0].isdigit():
        candidate = f"Task{candidate}"
    if not candidate:
        candidate = "CustomTask"
    if keyword.iskeyword(candidate.lower()) or keyword.iskeyword(candidate):
        candidate = f"{candidate.capitalize()}Task"
    if not candidate.isidentifier():
        candidate = f"Task_{candidate}"
    if not candidate.isidentifier():
        candidate = "TaskCustom"
    return candidate


def _wizard_slugify(raw: str) -> str:
    """Create a filesystem-friendly slug for the task filename."""

    slug = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_").lower()
    if not slug:
        slug = "custom_task"
    return slug


def cmd_wizard(args) -> int:
    """Run the guided onboarding wizard for new users."""

    try:
        display = CLIDisplay(console=get_console(args))
    except Exception as exc:  # pylint: disable=broad-except
        message("error", f"Wizard requires Rich console support: {exc}")
        return 1

    from rich.table import Table
    from rich.text import Text

    try:
        console = display.console
        _simple_header(
            console,
            "Guided Setup Wizard",
            "Configure workspace, task, montage, and input in one flow",
        )

        intro = Text()
        intro.append("Welcome to the AutoCleanEEG Pipeline!\n", style="brand")
        intro.append(
            "This guided setup prepares a workspace, creates or selects a task, "
            "updates the montage, and stores a default input path."
        )
        intro.append(
            "\n\nPress Ctrl+C any time to exit and resume later; "
            "changes already applied will be preserved.",
            style="muted",
        )
        intro.append(
            "\nSee https://autocleaneeg.org for full documentation.", style="muted"
        )

        display.panel(intro, title="Getting started", style="border")

        _wizard_render_state(display, "Snapshot before changes")

        # Step 1 — Workspace configuration
        workspace_dir, workspace_valid, *_ = _wizard_collect_state()
        display.header(
            "Step 1 • Workspace",
            "Choose where AutoClean stores configuration and outputs",
        )

        if workspace_valid:
            keep_workspace = display.prompt_yes_no(
                "Keep using this workspace?",
                default=True,
            )
        else:
            display.warning(
                "Workspace not fully initialized",
                "We'll create the recommended folders now.",
            )
            keep_workspace = False

        if not keep_workspace:
            while True:
                entry = display.prompt_text(
                    "Workspace folder",
                    default=str(workspace_dir),
                    show_default=True,
                )
                entry = (entry or "").strip()
                if not entry:
                    entry = str(workspace_dir)
                candidate = Path(entry).expanduser().resolve()

                if not display.prompt_yes_no(
                    f"Use workspace at {candidate}?",
                    default=True,
                ):
                    continue

                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    user_config._save_global_config(candidate)  # type: ignore[attr-defined]
                    user_config._create_workspace_structure(candidate)  # type: ignore[attr-defined]
                    user_config.config_dir = candidate
                    user_config.tasks_dir = candidate / "tasks"
                    display.success("Workspace ready", str(candidate))
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    display.error("Workspace setup failed", str(exc))
                    if not display.prompt_yes_no(
                        "Try a different location?", default=True
                    ):
                        return 1
        else:
            display.success("Workspace confirmed", str(workspace_dir))

        # Step 2 — Task selection / creation
        (
            _ws,
            _valid,
            active_task,
            active_task_path,
            _active_montage,
            _active_input,
        ) = _wizard_collect_state()

        display.header(
            "Step 2 • Task file",
            "Select or create the processing recipe for your data",
        )

        custom_tasks = user_config.list_custom_tasks()
        if custom_tasks:
            task_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            task_table.add_column("#", style="muted", width=3)
            task_table.add_column("Task", style="accent")
            task_table.add_column("Source", style="muted")
            for idx, (name, meta) in enumerate(sorted(custom_tasks.items()), start=1):
                marker = " [success](active)[/success]" if name == active_task else ""
                task_table.add_row(
                    str(idx),
                    f"{name}{marker}",
                    str(Path(meta["file_path"]).name),
                )
            display.console.print(task_table)
        else:
            display.info(
                "No custom tasks yet",
                "We'll generate one from the bundled template.",
            )

        options: List[Tuple[str, str]] = []
        if active_task:
            options.append(("keep", "Keep current active task"))
        if custom_tasks:
            options.append(("choose", "Select an existing custom task"))
        options.append(("create", "Create a new task file from the template"))

        option_table = Table(show_header=False, box=None, padding=(0, 1))
        option_table.add_column("Option", style="accent", no_wrap=True)
        option_table.add_column("Description", style="muted")
        for key, desc in options:
            option_table.add_row(key, desc)
        display.console.print(option_table)

        choice_keys = [key for key, _ in options]
        selection = display.prompt_choice(
            "Task setup option",
            choice_keys,
            default=choice_keys[0],
        )

        if selection == "keep" and active_task:
            display.success("Active task unchanged", active_task)
        elif selection == "choose" and custom_tasks:
            items = list(enumerate(sorted(custom_tasks.items()), start=1))
            while True:
                default_choice = active_task or ""
                answer = display.prompt_text(
                    "Task to activate (number or class name)",
                    default=default_choice,
                    show_default=bool(default_choice),
                )
                answer = (answer or "").strip()
                if not answer and default_choice:
                    answer = default_choice

                chosen_name: Optional[str] = None
                if answer.isdigit():
                    idx = int(answer)
                    for number, (name, _meta) in items:
                        if number == idx:
                            chosen_name = name
                            break
                else:
                    for _number, (name, _meta) in items:
                        if name.lower() == answer.lower():
                            chosen_name = name
                            break

                if not chosen_name:
                    display.warning(
                        "Task not found",
                        "Enter the listed number or exact class name.",
                    )
                    continue

                if user_config.set_active_task(chosen_name):
                    display.success("Active task set", chosen_name)
                    active_task = chosen_name
                    active_task_path = Path(custom_tasks[chosen_name]["file_path"])
                else:
                    display.error("Failed to save task selection")
                    return 1
                break
        else:  # create new task
            while True:
                raw_class = display.prompt_text(
                    "Task class name (CamelCase)",
                    default="CustomTask",
                    show_default=True,
                )
                class_name = _wizard_normalize_class_name(raw_class or "")
                slug_default = _wizard_slugify(class_name)
                raw_slug = display.prompt_text(
                    "Filename (without .py)",
                    default=slug_default,
                    show_default=True,
                )
                slug = _wizard_slugify(raw_slug or slug_default)
                target_path = user_config.tasks_dir / f"{slug}.py"
                allow_overwrite = False

                if target_path.exists():
                    if not display.prompt_yes_no(
                        f"{target_path} exists. Overwrite?",
                        default=False,
                    ):
                        continue
                    allow_overwrite = True

                try:
                    active_task_path = _wizard_create_task_from_template(
                        display, class_name, slug, overwrite=allow_overwrite
                    )
                except RuntimeError as exc:  # pylint: disable=broad-except
                    display.error("Could not create task", str(exc))
                    if not display.prompt_yes_no("Try again?", default=True):
                        return 1
                    continue

                if user_config.set_active_task(class_name):
                    active_task = class_name
                    display.success("Active task set", class_name)
                else:
                    display.error("Failed to save active task")
                    return 1
                break

        # Step 3 — Montage selection
        (_, _, active_task, active_task_path, current_montage, _) = (
            _wizard_collect_state()
        )

        display.header(
            "Step 3 • Montage",
            "Align the task with the correct electrode layout",
        )

        montages = {}
        try:
            montages = load_valid_montages()
        except Exception as exc:  # pylint: disable=broad-except
            display.error("Failed to load montage catalog", str(exc))

        if not active_task:
            display.warning(
                "No active task",
                "Set a task before editing the montage.",
            )
        elif not active_task_path or not active_task_path.exists():
            display.warning(
                "Built-in task",
                "Copy the task into your workspace to edit its montage.",
            )
        elif not montages:
            display.warning(
                "No montage definitions found",
                "Update configs/montages.yaml and rerun the wizard.",
            )
        else:
            montage_items = list(enumerate(sorted(montages.items()), start=1))
            montage_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            montage_table.add_column("#", style="muted", width=3)
            montage_table.add_column("Montage", style="accent")
            montage_table.add_column("Description", style="muted")
            for idx, (name, description) in montage_items:
                descriptor = description or "No description"
                if current_montage and name == current_montage:
                    descriptor = f"{descriptor} [success](current)[/success]"
                montage_table.add_row(str(idx), name, descriptor)
            display.console.print(montage_table)

            if display.prompt_yes_no(
                "Change the montage now?",
                default=current_montage is None,
            ):
                while True:
                    default_choice = current_montage or ""
                    answer = display.prompt_text(
                        "Montage (number or name)",
                        default=default_choice,
                        show_default=bool(default_choice),
                    )
                    answer = (answer or "").strip()
                    if not answer and default_choice:
                        answer = default_choice

                    chosen_montage: Optional[str] = None
                    if answer in montages:
                        chosen_montage = answer
                    elif answer.isdigit():
                        idx = int(answer)
                        for number, (name, _desc) in montage_items:
                            if number == idx:
                                chosen_montage = name
                                break

                    if not chosen_montage:
                        display.warning(
                            "Invalid selection",
                            "Pick a number from the table or type a montage name.",
                        )
                        continue

                    if chosen_montage == current_montage:
                        display.info("Montage unchanged", chosen_montage)
                        break

                    try:
                        source = active_task_path.read_text(encoding="utf-8")
                    except Exception as exc:  # pylint: disable=broad-except
                        display.error("Failed to read task file", str(exc))
                        return 1

                    block, start, end = _locate_montage_block(source)
                    if block is None:
                        display.warning(
                            "Montage block not found",
                            "Edit the task manually to update its montage.",
                        )
                        break

                    updated_block = _replace_montage_value(block, chosen_montage)
                    if updated_block is None:
                        display.error(
                            "Montage update failed",
                            "The task file has unexpected formatting.",
                        )
                        return 1

                    new_source = source[:start] + updated_block + source[end:]
                    try:
                        active_task_path.write_text(new_source, encoding="utf-8")
                    except Exception as exc:  # pylint: disable=broad-except
                        display.error("Failed to save task file", str(exc))
                        return 1

                    display.success(
                        "Montage updated",
                        f"{chosen_montage} saved to {active_task_path.name}",
                    )
                    current_montage = chosen_montage
                    break
            else:
                display.info(
                    "Montage unchanged",
                    current_montage or "No montage specified",
                )

        # Step 4 — Input source
        (*_, _, current_input) = _wizard_collect_state()

        display.header(
            "Step 4 • Input",
            "Store a default EEG file or directory to speed up processing",
        )

        if current_input:
            display.info("Current input", current_input)

        if display.prompt_yes_no(
            "Set or update the default input path now?",
            default=current_input is None,
        ):
            while True:
                answer = display.prompt_text(
                    "Input path (file or directory)",
                    default=current_input or "",
                    show_default=bool(current_input),
                )
                answer = _strip_wrapping_quotes(answer)
                if not answer:
                    display.info("Input unchanged", "Will prompt during processing")
                    break

                candidate = Path(answer).expanduser().resolve()
                if not candidate.exists():
                    display.warning("Path not found", str(candidate))
                    if not display.prompt_yes_no("Try another path?", default=True):
                        break
                    continue

                # Guard against obvious path traversal outside user-controlled roots
                allowed_roots = []
                try:
                    workspace_root = (
                        user_config.get_default_output_dir().resolve().parent
                    )
                    allowed_roots.append(workspace_root)
                except Exception:
                    pass
                try:
                    allowed_roots.append(Path.home().resolve())
                except Exception:
                    pass

                def _is_within_allowed(path: Path, roots: list[Path]) -> bool:
                    for root in roots:
                        try:
                            path.relative_to(root)
                            return True
                        except ValueError:
                            continue
                    return False

                if allowed_roots and not _is_within_allowed(candidate, allowed_roots):
                    display.warning(
                        "Path outside workspace/home",
                        (
                            "The selected path is outside your workspace or home directory and "
                            "may expose sensitive locations."
                        ),
                    )
                    if not display.prompt_yes_no(
                        "Use this path anyway?", default=False
                    ):
                        continue

                if user_config.set_active_source(str(candidate)):
                    display.success("Input saved", str(candidate))
                else:
                    display.error("Failed to store input path")
                    return 1
                break
        else:
            display.info("Input unchanged", current_input or "Will prompt each run")

        _wizard_render_state(display, "All set!")

        closing = Text()
        closing.append("Next steps", style="header")
        closing.append(
            "\n• Run `autocleaneeg-pipeline` to review your selections"
            "\n• Use `autocleaneeg-pipeline process to start processing"
            "\n• Rerun this wizard anytime to adjust defaults",
            style="muted",
        )
        display.panel(closing, style="border")

        return 0

    except KeyboardInterrupt:
        display.warning("Wizard interrupted", "Resume anytime with 'wizard'")
        return 1


def _run_interactive_setup() -> int:
    """Run interactive setup wizard with arrow key navigation."""

    try:
        console = get_console()
        _simple_header(console, "Setup", "Configure your workspace or compliance")

        # Show current workspace path directly beneath the banner (centered)
        try:
            from rich.align import Align as _SAlign
            from rich.text import Text as _SText

            workspace_dir = user_config.config_dir
            home = str(Path.home())
            display_path = str(workspace_dir)
            if display_path.startswith(home):
                display_path = display_path.replace(home, "~", 1)

            ws = _SText()
            ws.append("📂 ", style="muted")
            ws.append("Workspace: ", style="muted")
            ws.append(display_path, style="accent")
            console.print(_SAlign.center(ws))

            # Active task line (or guard if not set)
            try:
                active = _SText()
                active.append("🎯 ", style="muted")
                active.append("Active task: ", style="muted")
                current = user_config.get_active_task()
                if current:
                    active.append(str(current), style="accent")
                else:
                    active.append("not set", style="warning")
                console.print(_SAlign.center(active))
            except Exception:
                pass

            # Centered setup hint line
            hint = _SText()
            hint.append("Use arrow keys to navigate  •  Enter to select", style="muted")
            console.print(_SAlign.center(hint))

            # Centered compliance status
            from rich.text import Text as _CText

            status = _CText()
            compliance = get_compliance_status()
            if compliance["permanent"]:
                status.append("Compliance: permanently enabled", style="warning")
            elif compliance["enabled"]:
                status.append("Compliance: enabled", style="info")
            else:
                status.append("Compliance: disabled", style="muted")
            console.print(_SAlign.center(status))
            console.print()
        except Exception:
            pass

        if not INQUIRER_AVAILABLE:
            console.print(
                "[warning]⚠ Interactive prompts not available. Running basic setup...[/warning]"
            )
            user_config.setup_workspace()
            return 0

        # Get current compliance status
        compliance_status = get_compliance_status()
        is_enabled = compliance_status["enabled"]
        is_permanent = compliance_status["permanent"]

        # (status already shown centered under the banner above)

        # Check if compliance mode is permanently enabled
        if is_permanent:
            # Only allow workspace configuration
            questions = [
                inquirer.List(
                    "setup_type",
                    message="Select an option:",
                    choices=[
                        ("Configure workspace folder", "workspace_only"),
                        ("Exit", "exit"),
                    ],
                    default="workspace_only",
                )
            ]
        else:
            # Build setup options
            choices = [("Configure workspace folder", "workspace")]

            if is_enabled:
                choices.append(("Disable compliance mode", "disable_compliance"))
            else:
                choices.append(("Enable compliance mode", "enable_compliance"))

            questions = [
                inquirer.List(
                    "setup_type",
                    message="Select an option:",
                    choices=choices,
                    default="workspace",
                )
            ]

        answers = inquirer.prompt(questions)
        if not answers:  # User canceled
            return 0

        setup_type = answers["setup_type"]

        if setup_type == "exit":
            return 0
        elif setup_type in {"workspace", "workspace_only"}:
            return _setup_basic_mode()
        elif setup_type == "enable_compliance":
            # Enable compliance mode
            return _enable_compliance_mode()
        elif setup_type == "disable_compliance":
            # Disable compliance mode
            return _disable_compliance_mode()
        elif setup_type == "compliance":
            # Legacy compliance setup (permanent)
            return _setup_compliance_mode()

    except KeyboardInterrupt:
        return 0
    except Exception as e:
        console.print(f"[error]❌ Interactive setup failed: {str(e)}[/error]")
        return 1


def _setup_basic_mode() -> int:
    """Setup basic (non-compliance) mode."""

    try:
        console = get_console()

        if not INQUIRER_AVAILABLE:
            user_config.setup_workspace()
            return 0

        console.print()
        console.print("[title]Workspace Configuration[/title]")
        console.print(
            "[muted]Choose or confirm the folder where AutoClean stores config and tasks.[/muted]"
        )
        console.print()

        # Always run full workspace setup (includes prompting to change location if exists)
        # Don't show branding since we already showed it at the start of setup
        _ = user_config.setup_workspace(show_branding=False)

        # Update user configuration - auto-backup enabled by default
        user_config_data = load_user_config()

        # Ensure compliance and workspace are dictionaries
        if not isinstance(user_config_data.get("compliance"), dict):
            user_config_data["compliance"] = {}
        if not isinstance(user_config_data.get("workspace"), dict):
            user_config_data["workspace"] = {}

        user_config_data["compliance"]["enabled"] = False
        user_config_data["workspace"]["auto_backup"] = True  # Always enabled for safety

        save_user_config(user_config_data)

        console.print("[success]✓ Workspace configured[/success]")
        console.print(
            "[muted]Next: run 'autocleaneeg-pipeline task list' or 'process'.[/muted]"
        )

        return 0

    except Exception as e:
        console.print(f"[error]❌ Basic setup failed: {str(e)}[/error]")
        return 1


def _setup_compliance_mode() -> int:
    """Setup FDA 21 CFR Part 11 compliance mode with developer-managed Auth0."""
    from autoclean.utils.cli_display import setup_display

    try:
        if not INQUIRER_AVAILABLE:
            setup_display.error("Interactive setup requires 'inquirer' package")
            setup_display.info("Install with: pip install inquirer")
            return 1

        setup_display.blank_line()
        setup_display.header(
            "FDA 21 CFR Part 11 Compliance Setup", "Regulatory compliance mode"
        )
        # Show workspace location beneath header (centered, minimalist)
        try:
            from rich.align import Align as _XAlign
            from rich.text import Text as _XText

            ws_line = _XText()
            home = str(Path.home())
            display_path = str(user_config.config_dir)
            if display_path.startswith(home):
                display_path = display_path.replace(home, "~", 1)
            ws_line.append("📂 ", style="muted")
            ws_line.append("Workspace: ", style="muted")
            ws_line.append(display_path, style="accent")
            setup_display.console.print(_XAlign.center(ws_line))
            # Active task line (or guard)
            try:
                at_line = _XText()
                at_line.append("🎯 ", style="muted")
                at_line.append("Active task: ", style="muted")
                current = user_config.get_active_task()
                if current:
                    at_line.append(str(current), style="accent")
                else:
                    at_line.append("not set", style="warning")
                setup_display.console.print(_XAlign.center(at_line))
            except Exception:
                pass
            setup_display.blank_line()
        except Exception:
            pass
        setup_display.warning("Once enabled, compliance mode cannot be disabled")
        setup_display.blank_line()
        setup_display.console.print("[bold]This mode provides:[/bold]")
        setup_display.list_item("Mandatory user authentication")
        setup_display.list_item("Tamper-proof audit trails")
        setup_display.list_item("Encrypted data storage")
        setup_display.list_item("Electronic signature support")
        setup_display.blank_line()

        # Confirm user understands permanent nature
        confirm_question = [
            inquirer.Confirm(
                "confirm_permanent",
                message="Do you understand that compliance mode cannot be disabled once enabled?",
                default=False,
            )
        ]

        confirm_answer = inquirer.prompt(confirm_question)
        if not confirm_answer or not confirm_answer["confirm_permanent"]:
            setup_display.info("Compliance mode setup canceled")
            return 0

        # Setup Part-11 workspace with suffix
        user_config.setup_part11_workspace()

        # Ask about electronic signatures
        signature_question = [
            inquirer.Confirm(
                "require_signatures",
                message="Require electronic signatures for processing runs?",
                default=True,
            )
        ]

        signature_answer = inquirer.prompt(signature_question)
        if not signature_answer:
            return 0

        # Configure Auth0 manager with developer credentials
        auth_manager = get_auth0_manager()
        auth_manager.configure_developer_auth0()

        # Update user configuration
        user_config_data = load_user_config()

        # Ensure compliance and workspace are dictionaries
        if not isinstance(user_config_data.get("compliance"), dict):
            user_config_data["compliance"] = {}
        if not isinstance(user_config_data.get("workspace"), dict):
            user_config_data["workspace"] = {}

        user_config_data["compliance"]["enabled"] = True
        user_config_data["compliance"]["permanent"] = True  # Cannot be disabled
        user_config_data["compliance"]["auth_provider"] = "auth0"
        user_config_data["compliance"]["require_electronic_signatures"] = (
            signature_answer["require_signatures"]
        )
        user_config_data["workspace"][
            "auto_backup"
        ] = True  # Always enabled for compliance

        save_user_config(user_config_data)

        setup_display.success("Compliance mode setup complete!")
        setup_display.blank_line()
        setup_display.console.print("[bold]Next steps:[/bold]")
        setup_display.list_item(
            "Run 'autocleaneeg-pipeline login' to authenticate", indent=0
        )
        setup_display.list_item(
            "Use 'autocleaneeg-pipeline whoami' to check authentication status",
            indent=0,
        )
        setup_display.list_item(
            "All processing will now include audit trails and user authentication",
            indent=0,
        )

        return 0

    except Exception as e:
        setup_display.error("Compliance setup failed", str(e))
        return 1


def _enable_compliance_mode() -> int:
    """Enable FDA 21 CFR Part 11 compliance mode (non-permanent)."""
    try:
        if not INQUIRER_AVAILABLE:
            message("error", "Interactive setup requires 'inquirer' package.")
            return 1

        message("info", "\n🔐 Enable FDA 21 CFR Part 11 Compliance Mode")
        message("info", "This mode provides:")
        message("info", "• User authentication (when processing)")
        message("info", "• Audit trails")
        message("info", "• Electronic signature support")
        message("info", "• Can be disabled later")

        # Confirm enabling
        confirm_question = [
            inquirer.Confirm(
                "confirm_enable", message="Enable compliance mode?", default=True
            )
        ]

        confirm_answer = inquirer.prompt(confirm_question)
        if not confirm_answer or not confirm_answer["confirm_enable"]:
            message("info", "Compliance mode not enabled.")
            return 0

        # Configure Auth0 manager with developer credentials
        try:
            auth_manager = get_auth0_manager()
            auth_manager.configure_developer_auth0()
            message("info", "✓ Auth0 configured")
        except Exception as e:
            message("warning", f"Auth0 configuration failed: {e}")
            message("info", "You can configure Auth0 later for authentication")

        # Enable compliance mode (non-permanent)
        if enable_compliance_mode(permanent=False):
            message("success", "✓ Compliance mode enabled!")
            message("info", "\nNext steps:")
            message(
                "info",
                "1. Run 'autocleaneeg-pipeline login' to authenticate (when needed)",
            )
            message(
                "info", "2. Use 'autocleaneeg-pipeline auth disable' to turn it off"
            )
            return 0
        else:
            message("error", "Failed to enable compliance mode")
            return 1

    except Exception as e:
        message("error", f"Failed to enable compliance mode: {e}")
        return 1


def _disable_compliance_mode() -> int:
    """Disable FDA 21 CFR Part 11 compliance mode."""
    try:
        if not INQUIRER_AVAILABLE:
            message("error", "Interactive setup requires 'inquirer' package.")
            return 1

        message("info", "\n🔓 Disable FDA 21 CFR Part 11 Compliance Mode")

        # Check if permanent
        compliance_status = get_compliance_status()
        if compliance_status["permanent"]:
            message("error", "Cannot disable permanently enabled compliance mode")
            return 1

        message("warning", "This will disable:")
        message("warning", "• Required authentication")
        message("warning", "• Audit trail logging")
        message("warning", "• Electronic signatures")

        # Confirm disabling
        confirm_question = [
            inquirer.Confirm(
                "confirm_disable",
                message="Are you sure you want to disable compliance mode?",
                default=False,
            )
        ]

        confirm_answer = inquirer.prompt(confirm_question)
        if not confirm_answer or not confirm_answer["confirm_disable"]:
            message("info", "Compliance mode remains enabled.")
            return 0

        # Disable compliance mode
        if disable_compliance_mode():
            message("success", "✓ Compliance mode disabled!")
            message("info", "AutoClean will now operate in standard mode")
            return 0
        else:
            message("error", "Failed to disable compliance mode")
            return 1

    except Exception as e:
        message("error", f"Failed to disable compliance mode: {e}")
        return 1


# FUTURE FEATURE: User-managed Auth0 setup (commented out for now)
# def _setup_compliance_mode_user_managed() -> int:
#     """Setup FDA 21 CFR Part 11 compliance mode with user-managed Auth0."""
#     try:
#         import inquirer
#         from autoclean.utils.config import load_user_config, save_user_config
#         from autoclean.utils.auth import get_auth0_manager, validate_auth0_config
#
#         message("info", "\n🔐 FDA 21 CFR Part 11 Compliance Setup")
#         message("warning", "This mode requires Auth0 account and application setup.")
#
#         # Setup workspace first
#         user_config.setup_workspace()
#
#         # Explain Auth0 requirements
#         message("info", "\nAuth0 Application Setup Instructions:")
#         message("info", "1. Create an Auth0 account at https://auth0.com")
#         message("info", "2. Go to Applications > Create Application")
#         message("info", "3. Choose 'Native' as the application type (for CLI apps)")
#         message("info", "4. In your application settings, configure:")
#         message("info", "   - Allowed Callback URLs: http://localhost:8080/callback")
#         message("info", "   - Allowed Logout URLs: http://localhost:8080/logout")
#         message("info", "   - Grant Types: Authorization Code, Refresh Token (default for Native)")
#         message("info", "5. Copy your Domain, Client ID, and Client Secret")
#         message("info", "6. Your domain will be something like: your-tenant.us.auth0.com\n")
#
#         # Confirm user is ready
#         ready_question = [
#             inquirer.Confirm(
#                 'auth0_ready',
#                 message="Do you have your Auth0 application configured and credentials ready?",
#                 default=False
#             )
#         ]
#
#         ready_answer = inquirer.prompt(ready_question)
#         if not ready_answer or not ready_answer['auth0_ready']:
#             message("info", "Please set up your Auth0 application first, then run:")
#             message("info", "autoclean setup --compliance-mode")
#             return 0
#
#         # Get Auth0 configuration
#         auth_questions = [
#             inquirer.Text(
#                 'domain',
#                 message="Auth0 Domain (e.g., your-tenant.auth0.com)",
#                 validate=lambda _, x: len(x) > 0 and '.auth0.com' in x
#             ),
#             inquirer.Text(
#                 'client_id',
#                 message="Auth0 Client ID",
#                 validate=lambda _, x: len(x) > 0
#             ),
#             inquirer.Password(
#                 'client_secret',
#                 message="Auth0 Client Secret",
#                 validate=lambda _, x: len(x) > 0
#             ),
#             inquirer.Confirm(
#                 'require_signatures',
#                 message="Require electronic signatures for processing runs?",
#                 default=True
#             )
#         ]
#
#         auth_answers = inquirer.prompt(auth_questions)
#         if not auth_answers:
#             return 0
#
#         # Validate Auth0 configuration
#         message("info", "Validating Auth0 configuration...")
#
#         is_valid, error_msg = validate_auth0_config(
#             auth_answers['domain'],
#             auth_answers['client_id'],
#             auth_answers['client_secret']
#         )
#
#         if not is_valid:
#             message("error", f"Auth0 configuration invalid: {error_msg}")
#             return 1
#
#         message("success", "✓ Auth0 configuration validated!")
#
#         # Configure Auth0 manager
#         auth_manager = get_auth0_manager()
#         auth_manager.configure_auth0(
#             auth_answers['domain'],
#             auth_answers['client_id'],
#             auth_answers['client_secret']
#         )
#
#         # Update user configuration
#         user_config_data = load_user_config()
#
#         # Ensure compliance and workspace are dictionaries
#         if not isinstance(user_config_data.get('compliance'), dict):
#             user_config_data['compliance'] = {}
#         if not isinstance(user_config_data.get('workspace'), dict):
#             user_config_data['workspace'] = {}
#
#         user_config_data['compliance']['enabled'] = True
#         user_config_data['compliance']['auth_provider'] = 'auth0'
#         user_config_data['compliance']['require_electronic_signatures'] = auth_answers['require_signatures']
#         user_config_data['workspace']['auto_backup'] = True  # Always enabled for compliance
#
#         save_user_config(user_config_data)
#
#         message("success", "✓ Compliance mode setup complete!")
#         message("info", "\nNext steps:")
#         message("info", "1. Run 'autoclean login' to authenticate")
#         message("info", "2. Use 'autoclean whoami' to check authentication status")
#         message("info", "3. All processing will now include audit trails and user authentication")
#
#         return 0
#
#     except ImportError:
#         message("error", "Interactive setup requires 'inquirer' package.")
#         message("info", "Install with: pip install inquirer")
#         return 1
#     except Exception as e:
#         message("error", f"Compliance setup failed: {e}")
#         return 1


def cmd_version(args) -> int:
    """Show version information."""
    try:
        console = get_console(args)

        # Professional header consistent with setup
        console.print(f"[title]{LOGO_ICON} Autoclean-EEG Version Information:[/title]")
        console.print(f"  🏷️  [brand]{__version__}[/brand]")

        # GitHub and support info
        console.print("\n[header]GitHub Repository:[/header]")
        console.print(
            "  [info]https://github.com/cincibrainlab/autoclean_pipeline[/info]"
        )
        console.print("  [muted]Report issues, contribute, or get help[/muted]")

        return 0
    except ImportError:
        print("AutoClean EEG (version unknown)")
        return 0


def cmd_task(args) -> int:
    """Execute task management commands."""
    if not getattr(args, "task_action", None):
        # Show elegant task help (like '-h task') when no subcommand provided
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "task")
        return 0
    if args.task_action == "list":
        return cmd_list_tasks(args)
    elif args.task_action == "explore":
        return cmd_task_explore(args)
    elif args.task_action == "edit":
        return cmd_task_edit(args)
    elif args.task_action == "delete":
        return cmd_task_delete(args)
    elif args.task_action == "copy":
        return cmd_task_copy(args)
    elif args.task_action == "set":
        return cmd_task_set(args)
    elif args.task_action == "unset":
        return cmd_task_unset(args)
    elif args.task_action == "show":
        return cmd_task_show(args)
    elif args.task_action == "use":
        return cmd_task_use(args)
    elif args.task_action == "install":
        return cmd_task_install(args)
    elif args.task_action == "sync":
        return cmd_task_sync(args)
    elif args.task_action == "update":
        return cmd_task_update(args)
    elif args.task_action == "diagnose":
        return cmd_task_diagnose(args)
    elif args.task_action == "diff":
        return cmd_task_diff(args)
    elif args.task_action == "search":
        return cmd_task_search(args)
    elif args.task_action == "schema":
        return cmd_task_schema(args)
    # This point should be unreachable because argparse enforces valid subcommands.
    message("error", "Task command dispatch failed")
    return 1


def cmd_task_schema(args) -> int:
    """Handle task schema subcommands."""

    if not getattr(args, "task_schema_action", None):
        console = get_console(args)
        _simple_header(console)
        console.print(
            "[header]Task Schema[/header]\n\n"
            "Use `autocleaneeg-pipeline task schema export` to emit the canonical "
            "task configuration schema as JSON.",
        )
        return 0

    if args.task_schema_action == "export":
        return cmd_task_schema_export(args)

    message("error", "Unknown task schema action")
    return 1


def cmd_task_schema_export(args) -> int:
    """Export the canonical task schema as JSON."""

    from autoclean.configkit.schema import export_task_schema_layout

    payload = export_task_schema_layout()
    indent = None if args.indent is None or args.indent < 0 else args.indent
    text = json.dumps(payload, indent=indent)

    if args.output:
        output_path: Path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
            if indent is not None:
                fh.write("\n")
        message("success", f"Task schema exported to {output_path}")
    else:
        console = get_console(args)
        console.print(text)

    return 0


def cmd_task_explore(_args) -> int:
    """Open the workspace tasks directory in the system file browser."""
    try:
        tasks_dir = user_config.tasks_dir
        tasks_dir.mkdir(parents=True, exist_ok=True)

        # Detect platform and open folder
        platform = sys.platform
        path_str = str(tasks_dir)
        message("info", f"Opening tasks folder: {tasks_dir}")

        try:
            if platform.startswith("darwin"):
                subprocess.run(["open", path_str], check=False)
            elif platform.startswith("win"):
                os.startfile(path_str)  # type: ignore[attr-defined]
            else:
                # Linux and others
                if shutil.which("xdg-open"):
                    subprocess.run(["xdg-open", path_str], check=False)
                else:
                    # Fallback: print path if no opener available
                    print(path_str)
        except Exception:
            print(path_str)

        return 0
    except Exception as e:
        message("error", f"Failed to open tasks folder: {e}")
        return 1


def _resolve_task_file(target: str) -> Optional[Path]:
    """Resolve a task target to a file path.

    Accepts:
      - Absolute/relative path to a file
      - Task name present in workspace tasks directory (with or without .py)
    """
    try:
        p = Path(target).expanduser()
        if p.exists() and p.is_file():
            return p
        # Try within workspace tasks dir
        candidates = []
        if target.endswith(".py"):
            candidates.append(user_config.tasks_dir / target)
        else:
            candidates.append(user_config.tasks_dir / f"{target}.py")
            candidates.append(user_config.tasks_dir / target)
        for c in candidates:
            if c.exists() and c.is_file():
                return c
    except Exception:
        pass
    return None


def _detect_editor() -> Optional[list]:
    """Detect a suitable editor command as a list.

    Order:
      $VISUAL, $EDITOR, nano, vim/vi, notepad (Windows)
    """
    env = os.environ
    for var in ("VISUAL", "EDITOR"):
        ed = env.get(var)
        if ed:
            return ed.split()
    if shutil.which("nano"):
        return ["nano"]
    if shutil.which("vim"):
        return ["vim"]
    if shutil.which("vi"):
        return ["vi"]
    if sys.platform.startswith("win"):
        return ["notepad"]
    return None


def cmd_task_edit(args) -> int:
    """Open specified task in the user's editor."""
    try:
        target = getattr(args, "target", None)
        if not target:
            try:
                from rich.prompt import Confirm as _Confirm

                active = user_config.get_active_task()
                if active:
                    if not _Confirm.ask(f"Open active task '{active}'?", default=True):
                        message("info", "Canceled")
                        return 0
                    target = active
                else:
                    message(
                        "error",
                        "No target provided and no active task is set.",
                    )
                    message(
                        "info",
                        "Set one with: autocleaneeg-pipeline task set or pass a task name/path",
                    )
                    return 1
            except Exception:
                message(
                    "error",
                    "Interactive prompt unavailable and no target provided. Re-run with a task name/path.",
                )
                return 1

        # Resolve task target
        f = _resolve_task_file(target)
        if not f:
            # Try to resolve by discovered task name (built-in or workspace)
            try:
                from rich.prompt import Confirm as _Confirm
                from rich.prompt import Prompt as _Prompt

                from autoclean.utils.task_discovery import safe_discover_tasks

                tasks, _, _ = safe_discover_tasks()
                match = None
                for t in tasks:
                    if t.name.lower() == target.lower():
                        match = t
                        break
                if match:
                    src = Path(match.source)
                    # If built-in (not in workspace), offer to copy into workspace first
                    if user_config.tasks_dir not in src.parents:
                        # Determine destination filename
                        suggested_name = args.name or src.stem
                        if not args.force:
                            # Ask to confirm copy and allow rename
                            console = get_console()
                            console.print(
                                f"[muted]Built-in task detected:[/muted] [accent]{match.name}[/accent]"
                            )
                            if not _Confirm.ask(
                                "Copy to your workspace to edit?", default=True
                            ):
                                message("info", "Canceled")
                                return 0
                            new_name = _Prompt.ask(
                                "Save as (filename)", default=f"{suggested_name}.py"
                            )
                        else:
                            new_name = (
                                f"{suggested_name}.py"
                                if not str(suggested_name).endswith(".py")
                                else suggested_name
                            )

                        dest = user_config.tasks_dir / new_name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if dest.exists():
                            if args.force:
                                try:
                                    shutil.copy2(src, dest)
                                except Exception as e:
                                    message("error", f"Failed to overwrite: {e}")
                                    return 1
                            else:
                                # Offer choices: open existing, overwrite, rename, cancel
                                choice = _Prompt.ask(
                                    "File exists. Choose",
                                    choices=["open", "overwrite", "rename", "cancel"],
                                    default="open",
                                )
                                if choice == "open":
                                    f = dest
                                elif choice == "overwrite":
                                    try:
                                        shutil.copy2(src, dest)
                                        f = dest
                                    except Exception as e:
                                        message(
                                            "error", f"Failed to overwrite file: {e}"
                                        )
                                        return 1
                                elif choice == "rename":
                                    renamed = _Prompt.ask(
                                        "New filename", default=f"copy_of_{new_name}"
                                    )
                                    dest = user_config.tasks_dir / (
                                        renamed
                                        if renamed.endswith(".py")
                                        else f"{renamed}.py"
                                    )
                                    try:
                                        shutil.copy2(src, dest)
                                        f = dest
                                    except Exception as e:
                                        message(
                                            "error",
                                            f"Failed to copy as '{dest.name}': {e}",
                                        )
                                        return 1
                                else:  # cancel
                                    message("info", "Canceled")
                                    return 0
                        else:
                            try:
                                shutil.copy2(src, dest)
                                f = dest
                                message(
                                    "info",
                                    f"Copied built-in task to workspace: {dest.name}",
                                )
                            except Exception as e:
                                message("error", f"Failed to copy task: {e}")
                                return 1
                    else:
                        f = src
                else:
                    message(
                        "error",
                        f"Task not found: {target}. Use 'autocleaneeg-pipeline task list' or provide a path.",
                    )
                    return 1
            except ImportError:
                message(
                    "error",
                    "Discovery tools unavailable. Provide a path to the task file.",
                )
                return 1
            except Exception as e:
                message("error", f"Task resolution failed: {e}")
                return 1

        editor = _detect_editor()
        if not editor:
            message(
                "error",
                "No editor found. Set $EDITOR or $VISUAL, or install nano/vim.",
            )
            return 1

        message("info", f"Opening: {f}")
        try:
            subprocess.call(editor + [str(f)])
            return 0
        except FileNotFoundError:
            message("error", f"Editor not found: {' '.join(editor)}")
            return 1
    except Exception as e:
        message("error", f"Failed to edit task: {e}")
        return 1


def cmd_task_delete(args) -> int:
    """Delete a task file from the workspace tasks directory."""
    try:
        # Determine target: use provided or confirm using active task
        target = getattr(args, "target", None)
        if not target:
            try:
                from rich.prompt import Confirm as _Confirm

                active = user_config.get_active_task()
                if active:
                    if not _Confirm.ask(f"Use active task '{active}'?", default=True):
                        message("info", "Canceled")
                        return 0
                    target = active
                else:
                    message(
                        "error",
                        "No target provided and no active task is set.",
                    )
                    message(
                        "info",
                        "Set one with: autocleaneeg-pipeline task set or pass a task name/path",
                    )
                    return 1
            except Exception:
                message(
                    "error",
                    "Interactive prompt unavailable and no target provided. Re-run with a task name/path.",
                )
                return 1

        # Resolve to path; don't allow deleting built-ins
        f = _resolve_task_file(target)
        if not f:
            message("error", f"Task not found: {target}")
            message(
                "info",
                "Provide a workspace task name or a path under your workspace tasks/",
            )
            return 1

        # Ensure within workspace tasks dir
        try:
            ws_tasks = user_config.tasks_dir.resolve()
            if ws_tasks not in f.resolve().parents:
                message("error", "Refusing to delete files outside workspace tasks/")
                return 1
        except Exception:
            message("error", "Could not verify task location")
            return 1

        if not args.force:
            try:
                from rich.prompt import Confirm as _Confirm

                if not _Confirm.ask(f"Delete '{f.name}'?", default=False):
                    message("info", "Canceled")
                    return 0
            except Exception:
                message(
                    "warning",
                    "Interactive prompt unavailable; re-run with --force to skip",
                )
                return 1

        try:
            f.unlink()
            message("success", f"✓ Deleted: {f.name}")
            return 0
        except Exception as e:
            message("error", f"Failed to delete file: {e}")
            return 1
    except Exception as e:
        message("error", f"Delete failed: {e}")
        return 1


def cmd_task_copy(args) -> int:
    """Copy a task to a new file in the workspace tasks directory."""
    try:
        source = (
            args.source if hasattr(args, "source") else getattr(args, "target", None)
        )
        if not source:
            # Interactive selection across built-in and custom tasks
            try:
                from autoclean.utils.task_discovery import safe_discover_tasks

                tasks, _, _ = safe_discover_tasks()
                if not tasks:
                    message("error", "No tasks available to copy")
                    return 1

                # Build choices with type label
                choices = []
                for t in sorted(tasks, key=lambda x: x.name.lower()):
                    is_builtin = "autoclean/tasks" in t.source
                    kind = "built-in" if is_builtin else "custom"
                    choices.append((t.name, t.source, kind))

                # Prefer inquirer if available
                if INQUIRER_AVAILABLE:
                    import inquirer as _inq

                    prompt_q = [
                        _inq.List(
                            "task",
                            message="Select a task to copy",
                            choices=[
                                (f"{name}  [{kind}]", i)
                                for i, (name, _src, kind) in enumerate(choices)
                            ],
                        )
                    ]
                    ans = _inq.prompt(prompt_q)
                    if not ans:
                        message("info", "Canceled")
                        return 0
                    idx = ans["task"]
                else:
                    # Rich-based listing with numeric input
                    try:
                        from pathlib import Path as _P

                        from rich.console import Console as _Console
                        from rich.prompt import Prompt as _Prompt
                        from rich.table import Table as _Table

                        _c = _Console()
                        _c.print()
                        tbl = _Table(
                            show_header=True,
                            header_style="header",
                            box=None,
                            padding=(0, 1),
                        )
                        tbl.add_column("#", style="muted", width=4)
                        tbl.add_column("Task", style="accent", no_wrap=True)
                        tbl.add_column("Type", style="muted", width=9)
                        tbl.add_column("Source", style="muted")
                        for i, (name, src, kind) in enumerate(choices, 1):
                            tbl.add_row(str(i), name, kind, _P(src).name)
                        _c.print(tbl)
                        choice = _Prompt.ask(
                            "Select a task number (or Enter to cancel)",
                            default="",
                            show_default=False,
                        )
                        if not choice.strip():
                            message("info", "Canceled")
                            return 0
                        idx = int(choice) - 1
                        if idx < 0 or idx >= len(choices):
                            message("error", "Invalid selection")
                            return 1
                    except Exception:
                        # Basic fallback
                        print("Available tasks:")
                        for i, (name, _src, kind) in enumerate(choices, 1):
                            print(f"  {i}. {name} [{kind}]")
                        choice = input(
                            "Select a task number (or Enter to cancel): "
                        ).strip()
                        if not choice:
                            print("Canceled")
                            return 0
                        idx = int(choice) - 1
                        if idx < 0 or idx >= len(choices):
                            message("error", "Invalid selection")
                            return 1

                # Resolve chosen source (use name for resolution first)
                chosen_name, chosen_src, _k = choices[idx]
                source = chosen_name
                src_path = Path(chosen_src)
            except Exception as e:
                message("error", f"Failed to list tasks for selection: {e}")
                return 1

        # Resolve source: path or discovered name
        src_path = _resolve_task_file(source) if isinstance(source, str) else src_path
        if not src_path:
            try:
                from autoclean.utils.task_discovery import safe_discover_tasks

                tasks, _, _ = safe_discover_tasks()
                for t in tasks:
                    if t.name.lower() == str(source).lower():
                        src_path = Path(t.source)
                        break
            except Exception:
                src_path = None

        if not src_path or not src_path.exists():
            message("error", f"Task not found: {source}")
            return 1

        # Prepare destination with unified name entry (derive filename + class)
        dest_dir = user_config.tasks_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Extract original class name for reference
        orig_class = None
        try:
            orig_class, _desc = user_config._extract_task_info(src_path)
        except Exception:
            pass

        # Utilities
        def _ts():
            from datetime import datetime as _dt

            return _dt.now().strftime("%Y%m%d_%H%M%S")

        def _to_camel(s: str) -> str:
            import re as _re

            parts = _re.split(r"[^0-9A-Za-z]+", s)
            camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
            if camel and camel[0].isdigit():
                camel = "T" + camel
            return camel or "TaskCopy"

        def _to_snake(s: str) -> str:
            import re as _re

            # Lowercase, replace non-alnum with underscores, collapse repeats
            s = s.strip().lower()
            s = _re.sub(r"[^0-9a-z]+", "_", s)
            s = _re.sub(r"_+", "_", s).strip("_")
            return s or "task_copy"

        def _is_valid_class_name(name: str) -> bool:
            try:
                return bool(name) and name.isidentifier() and not name[0].isdigit()
            except Exception:
                return False

        def _unique_dest(base: str) -> Path:
            base = base.removesuffix(".py")
            p = dest_dir / f"{base}.py"
            return p if not p.exists() else dest_dir / f"{base}_{_ts()}.py"

        # Unified name source: --name or prompt
        raw_name = getattr(args, "name", None)
        if not raw_name:
            # Provide elegant, minimal instructions
            try:
                from rich.align import Align as _Align
                from rich.console import Console as _Console
                from rich.prompt import Prompt as _Prompt
                from rich.text import Text as _Text

                _c = _Console()
                _c.print()
                line = _Text()
                line.append("Name your new task", style="title")
                _c.print(_Align.center(line))
                hint = _Text()
                hint.append(
                    "Type a descriptive name; we'll create a valid class and filename.",
                    style="muted",
                )
                _c.print(_Align.center(hint))
                ex = _Text()
                ex.append("Examples: ", style="muted")
                ex.append("My Resting Task", style="accent")
                ex.append("  •  ", style="muted")
                ex.append("resting_basic_v2", style="accent")
                _c.print(_Align.center(ex))
                _c.print()
                raw_name = _Prompt.ask(
                    "Enter task name (or Enter to cancel)",
                    default="",
                    show_default=False,
                ).strip()
                if not raw_name:
                    message("info", "Canceled")
                    return 0
            except Exception:
                raw_name = input("Enter task name (or press Enter to cancel): ").strip()
                if not raw_name:
                    print("Canceled")
                    return 0

        # Derive names from unified input
        snake = _to_snake(raw_name)
        camel = _to_camel(raw_name)
        if not _is_valid_class_name(camel):
            camel = f"T{camel}" if camel and camel[0].isdigit() else "TaskCopy"

        # Avoid class name collisions with existing workspace tasks
        existing = set(user_config.list_custom_tasks().keys())
        new_class = camel if camel not in existing else f"{camel}_{_ts()}"
        dest = _unique_dest(snake)

        # Conflict handling
        if dest.exists() and not getattr(args, "force", False):
            try:
                from rich.prompt import Prompt as _Prompt

                choice = _Prompt.ask(
                    f"{dest.name} exists. Choose",
                    choices=["overwrite", "rename", "cancel"],
                    default="rename",
                )
                if choice == "cancel":
                    message("info", "Canceled")
                    return 0
                if choice == "rename":
                    new_name = _Prompt.ask(
                        "New filename", default=f"copy_of_{dest.name}"
                    )
                    dest = dest_dir / (
                        new_name if new_name.endswith(".py") else f"{new_name}.py"
                    )
            except Exception:
                message(
                    "warning",
                    "Interactive prompt unavailable; re-run with --force or --name",
                )
                return 1

        # Execute copy with class rename to the derived class name
        try:
            if orig_class and new_class and new_class != orig_class:
                import re as _re

                text = src_path.read_text(encoding="utf-8")
                # Replace class definition line safely (single occurrence)
                pattern = rf"(^\s*class\s+){orig_class}(\s*\()"
                new_text, n = _re.subn(pattern, rf"\1{new_class}\2", text, flags=_re.M)
                if n == 0:
                    message(
                        "warning",
                        "Could not locate class definition for rename; copying as-is.",
                    )
                    shutil.copy2(src_path, dest)
                else:
                    dest.write_text(new_text, encoding="utf-8")
            else:
                shutil.copy2(src_path, dest)
            message("success", f"✓ Copied to: {dest.name}")
        except Exception as e:
            message("error", f"Failed to copy file: {e}")
            return 1

        # Provide usage
        try:
            class_name, _ = user_config._extract_task_info(dest)
            # Summary
            from rich.align import Align as _Align
            from rich.console import Console as _Console
            from rich.text import Text as _Text

            _c = _Console()
            _c.print()
            t = _Text()
            t.append("Task copied", style="success")
            _c.print(_Align.center(t))
            fline = _Text()
            fline.append("File: ", style="muted")
            fline.append(dest.name, style="accent")
            cline = _Text()
            cline.append("Class: ", style="muted")
            cline.append(class_name, style="accent")
            _c.print(_Align.center(fline))
            _c.print(_Align.center(cline))
            if class_name != new_class:
                note = _Text()
                note.append("Note: ", style="muted")
                note.append(
                    "class detected differs from requested rename", style="warning"
                )
                _c.print(_Align.center(note))
            _c.print()
            print("Use with:")
            print(f"  autocleaneeg-pipeline process {class_name} <data_file>")
            try:
                from rich.prompt import Confirm as _Confirm

                if _Confirm.ask("Set this task as the active task?", default=True):
                    if user_config.set_active_task(class_name):
                        message("success", f"✓ Active task set to: {class_name}")
                    else:
                        message("error", "Failed to save active task configuration.")
            except Exception:
                try:
                    resp = (
                        input("Set this task as the active task? [Y/n]: ")
                        .strip()
                        .lower()
                    )
                    if resp in ("", "y", "yes"):
                        if user_config.set_active_task(class_name):
                            message("success", f"✓ Active task set to: {class_name}")
                        else:
                            message(
                                "error", "Failed to save active task configuration."
                            )
                except Exception:
                    pass
        except Exception as e:
            message("warning", f"Could not verify Task class after copy: {e}")

        return 0
    except Exception as e:
        message("error", f"Copy failed: {e}")
        return 1


def cmd_task_set(args) -> int:
    """Set the active task."""
    try:
        # If task name provided, use it directly
        if hasattr(args, "task_name") and args.task_name:
            task_name = args.task_name

            # Validate task exists in custom tasks
            custom_tasks = user_config.list_custom_tasks()
            if task_name not in custom_tasks:
                message("error", f"Task '{task_name}' not found in workspace.")
                message("info", "Available tasks:")
                for name in custom_tasks:
                    print(f"  • {name}")
                return 1
        else:
            # Interactive selection
            task_name = user_config.select_active_task_interactive()
            if task_name is None:
                message("info", "No task selected.")
                return 0

        # Set the active task
        if user_config.set_active_task(task_name):
            message("success", f"✓ Active task set to: {task_name}")
            message("info", "Now you can use: autocleaneeg-pipeline process <file>")
            return 0
        else:
            message("error", "Failed to save active task configuration.")
            return 1

    except Exception as e:
        message("error", f"Failed to set active task: {e}")
        return 1


def cmd_task_unset(_args) -> int:
    """Clear the active task."""
    try:
        current_task = user_config.get_active_task()
        if current_task is None:
            message("info", "No active task is currently set.")
            return 0

        if user_config.set_active_task(None):
            message("success", f"✓ Active task cleared (was: {current_task})")
            return 0
        else:
            message("error", "Failed to clear active task configuration.")
            return 1

    except Exception as e:
        message("error", f"Failed to unset active task: {e}")
        return 1


def cmd_task_show(_args) -> int:
    """Show the current active task."""
    try:
        active_task = user_config.get_active_task(ensure_exists=False)

        if active_task is None:
            message("info", "No active task is currently set.")
            message("info", "Set one with: autocleaneeg-pipeline task set")
        else:
            message("info", f"Active task: {active_task}")

            # Verify the task still exists
            custom_tasks = user_config.list_custom_tasks()
            if active_task in custom_tasks:
                task_info = custom_tasks[active_task]
                message(
                    "info",
                    f"Description: {task_info.get('description', 'No description')}",
                )
                message("info", f"File: {task_info['file_path']}")
            else:
                message(
                    "warning",
                    f"Active task '{active_task}' no longer exists in workspace",
                )
                message(
                    "info",
                    "Consider setting a different active task or adding the missing task file.",
                )
                user_config.set_active_task(None)
                message(
                    "info",
                    "Cleared stale active task reference from workspace settings.",
                )

        return 0

    except Exception as e:
        message("error", f"Failed to show active task: {e}")
        return 1


def cmd_task_use(args) -> int:
    """Install and activate a task in one step (one-step workflow)."""
    console = get_console(args)
    registry = BuiltinRegistry()

    # Interactive selection if no task name provided
    if not args.task_name:
        # Show available tasks for selection
        tasks = registry.list_tasks()
        if not tasks:
            console.print("[warning]No library tasks available.[/warning]")
            return 1

        console.print("\n[header]Select a task to install and activate:[/header]\n")

        from rich.table import Table

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Task", style="accent")
        table.add_column("Description", style="muted")

        for i, task in enumerate(tasks, 1):
            # Try to get description from registry
            desc = "No description"
            try:
                index = json.loads(
                    registry._cache_index_path().read_text()
                    if registry._cache_index_path().exists()
                    else registry._pkg_index_text()
                )
                for entry in index.get("tasks", []):
                    if entry.get("name") == task.name:
                        desc = entry.get("description", "No description")
                        break
            except Exception:
                pass
            table.add_row(str(i), task.name, desc)

        console.print(table)

        try:
            from rich.prompt import Prompt

            choice = Prompt.ask("\nSelect task by number", default="")
            if not choice.strip():
                console.print("[muted]Selection cancelled.[/muted]")
                return 0

            choice_num = int(choice)
            if 1 <= choice_num <= len(tasks):
                task_name = tasks[choice_num - 1].name
            else:
                console.print("[error]Invalid selection.[/error]")
                return 1
        except (ValueError, KeyboardInterrupt):
            console.print("\n[muted]Selection cancelled.[/muted]")
            return 0
    else:
        task_name = args.task_name

    # Check if task exists in library
    task = registry.get_task(task_name)
    if not task:
        console.print(f"[error]✗[/error] Task '{task_name}' not found in library")
        console.print("[muted]Run 'task list' to see available tasks.[/muted]")
        return 1

    # Check if already exists in workspace
    dest_path = user_config.tasks_dir / f"{task_name}.py"
    skip_install = False

    if dest_path.exists() and not args.force:
        sync_status = registry.task_sync_status(task_name, user_config.tasks_dir)
        status = sync_status.get("status")

        if status == "synced":
            console.print(
                f"[success]✓[/success] {task_name} already installed and up to date"
            )
            skip_install = True
        elif status == "modified":
            console.print(
                f"[warning]⚠[/warning] {task_name} exists with customizations"
            )
            console.print(
                "[muted]Use --force to overwrite (this will delete your changes)[/muted]"
            )
            return 1
        else:
            skip_install = False

    # Install task if needed
    if not skip_install:
        try:
            installed_path = registry.materialize_task_to(
                task_name, user_config.tasks_dir
            )
            console.print(
                f"[success]✓[/success] {task_name} installed to [info]{installed_path}[/info]"
            )
        except Exception as exc:
            console.print(f"[error]✗[/error] Installation failed: {exc}")
            return 1

    # Set as active task (unless --no-activate)
    if not args.no_activate:
        if user_config.set_active_task(task_name):
            console.print(
                f"[success]✓[/success] Active task set to: [accent]{task_name}[/accent]"
            )
            console.print(
                "[muted]Ready to process: autocleaneeg-pipeline process <file>[/muted]"
            )
        else:
            console.print("[error]✗[/error] Failed to set active task")
            return 1

    return 0


def cmd_task_install(args) -> int:
    """Install a task from any source (library, file, or built-in) - unified command."""
    console = get_console(args)
    registry = BuiltinRegistry()

    task_source = args.task_source
    source_type = args.source  # auto, library, file, builtin
    custom_name = args.name
    force = args.force
    activate = args.activate
    installed_path: Optional[Path] = None
    installed_from = ""
    install_status = "installed"

    # Auto-detect source type if not specified
    if source_type == "auto":
        # Check if it's a file path
        source_path = Path(task_source)
        if source_path.exists() and source_path.is_file():
            source_type = "file"
        else:
            # Try library first, then builtin
            library_tasks = [t.name for t in registry.list_tasks()]
            if task_source in library_tasks:
                source_type = "library"
            else:
                # Check if it's a builtin task
                valid_tasks, _, _ = safe_discover_tasks()
                builtin_names = [
                    t.name for t in valid_tasks if "autoclean/tasks" in t.source
                ]
                if task_source in builtin_names:
                    source_type = "builtin"
                else:
                    console.print(
                        f"[error]✗[/error] Cannot find '{task_source}' as file, library task, or built-in task"
                    )
                    console.print(
                        "[muted]Tip: Use 'task search' to find tasks or 'task list' to see all available tasks.[/muted]"
                    )
                    return 1

    # Install based on source type
    try:
        if source_type == "file":
            # Install from file
            source_path = Path(task_source)
            if not source_path.exists():
                console.print(f"[error]✗[/error] File not found: {task_source}")
                return 1

            # Determine destination name
            if custom_name:
                dest_name = f"{custom_name}.py"
            else:
                dest_name = source_path.name

            dest_path = user_config.tasks_dir / dest_name

            # Check if already exists
            if dest_path.exists() and not force:
                console.print(
                    f"[error]✗[/error] Task '{dest_name}' already exists. Use --force to overwrite."
                )
                return 1

            # Copy file
            import shutil

            shutil.copy2(source_path, dest_path)

            # Extract task name
            try:
                class_name, _ = user_config._extract_task_info(dest_path)
                task_name = class_name
            except Exception:
                task_name = dest_path.stem
            installed_path = dest_path
            installed_from = "local file"

        elif source_type == "library":
            # Install from library
            task_name = custom_name if custom_name else task_source

            # Check if task exists in library
            task = registry.get_task(task_source)
            if not task:
                console.print(
                    f"[error]✗[/error] Task '{task_source}' not found in library"
                )
                console.print(
                    "[muted]Run 'task list --source=library' to see available library tasks.[/muted]"
                )
                return 1

            dest_path = user_config.tasks_dir / f"{task_name}.py"

            # Check if already exists
            if dest_path.exists() and not force:
                sync_status = registry.task_sync_status(
                    task_source, user_config.tasks_dir
                )
                status = sync_status.get("status")

                if status == "synced":
                    install_status = "already_installed"
                    installed_path = dest_path
                    installed_from = "library registry"
                elif status == "modified":
                    console.print(
                        f"[warning]⚠[/warning] {task_name} exists with customizations"
                    )
                    console.print(
                        "[muted]Use --force to overwrite (this will delete your changes)[/muted]"
                    )
                    return 1
                else:
                    console.print(
                        f"[error]✗[/error] Task '{task_name}' already exists. Use --force to overwrite."
                )
                return 1
            else:
                # Install task
                materialized_path = registry.materialize_task_to(
                    task_source, user_config.tasks_dir
                )

                # Rename if custom name specified
                if custom_name and custom_name != task_source:
                    new_path = user_config.tasks_dir / f"{custom_name}.py"
                    materialized_path.rename(new_path)
                    materialized_path = new_path
                    task_name = custom_name
                installed_path = materialized_path
                installed_from = "library registry"

        elif source_type == "builtin":
            # Install from built-in
            task_name = custom_name if custom_name else task_source

            # Verify task exists in built-in
            valid_tasks, _, _ = safe_discover_tasks()
            builtin_task = None
            for t in valid_tasks:
                if "autoclean/tasks" in t.source and t.name == task_source:
                    builtin_task = t
                    break

            if not builtin_task:
                console.print(
                    f"[error]✗[/error] Built-in task '{task_source}' not found"
                )
                console.print(
                    "[muted]Run 'task list --source=builtin' to see available built-in tasks.[/muted]"
                )
                return 1

            dest_path = user_config.tasks_dir / f"{task_name}.py"

            # Check if already exists
            if dest_path.exists() and not force:
                console.print(
                    f"[error]✗[/error] Task '{task_name}' already exists. Use --force to overwrite."
                )
                return 1

            # Copy from built-in source
            import shutil

            shutil.copy2(builtin_task.source, dest_path)
            installed_path = dest_path
            installed_from = "built-in catalog"
        else:
            console.print(
                f"[error]✗[/error] Unknown source type '{source_type}'. Use file, library, or builtin."
            )
            return 1

        if installed_path is None:
            console.print(
                "[error]✗[/error] Task installation completed without determining destination path"
            )
            return 1

        # Optionally activate
        activated = False
        if activate:
            if user_config.set_active_task(task_name):
                console.print(
                    f"[success]✓[/success] Active task set to: [accent]{task_name}[/accent]"
                )
                activated = True
            else:
                console.print("[error]✗[/error] Failed to set active task")
                return 1
        _render_task_install_summary(
            console,
            task_name=task_name,
            installed_path=installed_path,
            source_label=installed_from or source_type,
            activated=activated,
            install_status=install_status,
        )
        return 0

    except Exception as e:
        console.print(f"[error]✗[/error] Installation failed: {e}")
        return 1


def cmd_task_sync(args) -> int:
    """Check workspace tasks for updates and optionally apply them."""
    console = get_console(args)
    registry = BuiltinRegistry()

    console.print("→ Checking workspace tasks against sources...\n")

    # Get all workspace tasks
    workspace_tasks = user_config.list_custom_tasks()
    if not workspace_tasks:
        console.print("[muted]No tasks installed in workspace.[/muted]")
        return 0

    # Check sync status for each task
    from rich.table import Table

    synced_tasks = []
    customized_tasks = []
    outdated_tasks = []
    orphaned_tasks = []

    for task_name in workspace_tasks.keys():
        sync_status = registry.task_sync_status(task_name, user_config.tasks_dir)
        status = sync_status.get("status")
        source = sync_status.get("source", "unknown")

        if status == "synced":
            synced_tasks.append((task_name, source))
        elif status == "modified":
            customized_tasks.append((task_name, source))
        elif status == "not_installed":
            # This shouldn't happen for workspace tasks, but handle it
            continue
        elif status == "missing":
            orphaned_tasks.append(task_name)
        else:
            # Check if outdated by comparing remote hash
            record = registry.manifest.task_record(task_name)
            if record:
                remote_hash = record.get("remote_hash")
                workspace_hash = sync_status.get("workspace_hash")
                if remote_hash and workspace_hash and remote_hash != workspace_hash:
                    outdated_tasks.append((task_name, source))
                else:
                    customized_tasks.append((task_name, source))
            else:
                customized_tasks.append((task_name, source))

    # Display results
    console.print("[header]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/header]")
    console.print("[header] SYNC STATUS[/header]")
    console.print("[header]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/header]\n")

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Status", width=2)
    table.add_column("Task", style="accent")
    table.add_column("Source", style="muted")
    table.add_column("Info", style="info")

    for task_name, source in synced_tasks:
        table.add_row("✓", task_name, f"[{source}]", "up to date")

    for task_name, source in customized_tasks:
        table.add_row("⚠", task_name, f"[{source}]", "customized")

    for task_name, source in outdated_tasks:
        table.add_row("↻", task_name, f"[{source}]", "[warning]outdated[/warning]")

    for task_name in orphaned_tasks:
        table.add_row("✗", task_name, "[orphan]", "[error]source removed[/error]")

    console.print(table)
    console.print()

    # Summary
    console.print("[header]Summary:[/header]")
    if outdated_tasks:
        console.print(f"  • {len(outdated_tasks)} task(s) have updates available")
    if orphaned_tasks:
        console.print(
            f"  • {len(orphaned_tasks)} orphaned task(s) (source no longer exists)"
        )
    if customized_tasks:
        console.print(f"  • {len(customized_tasks)} customized task(s)")
    if not outdated_tasks and not orphaned_tasks:
        console.print("  • All tasks are up to date")

    # Next steps
    if outdated_tasks or orphaned_tasks:
        console.print("\n[header]Next steps:[/header]")
        if outdated_tasks:
            console.print(
                "  • Run [accent]'task sync --update'[/accent] to apply updates (will backup customized tasks)"
            )
            console.print(
                f"  • Run [accent]'task diff {outdated_tasks[0][0]}'[/accent] to preview changes"
            )
        if orphaned_tasks:
            console.print(
                "  • Run [accent]'task delete <name>'[/accent] to remove orphaned tasks"
            )

    # Apply updates if requested
    if args.update and outdated_tasks:
        if args.dry_run:
            console.print("\n[muted]Dry run - no changes will be made[/muted]")
            return 0

        console.print(f"\n→ Updating {len(outdated_tasks)} task(s)...\n")

        from datetime import datetime

        backup_suffix = datetime.now().strftime("%Y-%m-%d-%H%M%S")

        for task_name, source in outdated_tasks:
            try:
                # Create backup
                task_path = user_config.tasks_dir / f"{task_name}.py"
                backup_path = (
                    user_config.tasks_dir / f"{task_name}.backup.{backup_suffix}.py"
                )

                import shutil

                shutil.copy2(task_path, backup_path)
                console.print(f"  ⚠ Backed up {task_name} to {backup_path.name}")

                # Reinstall from library
                registry.materialize_task_to(task_name, user_config.tasks_dir)
                console.print(f"  ✓ Updated {task_name} from {source}")
            except Exception as exc:
                console.print(f"  ✗ Failed to update {task_name}: {exc}")

        console.print("\n[success]✓[/success] Update complete")

    return 0


def cmd_task_update(args) -> int:
    """Update task library index from GitHub registry (promoted from task library update)."""
    console = get_console(args)
    registry = BuiltinRegistry()

    console.print("→ Updating task library from GitHub registry...\n")

    allow_network = not getattr(args, "no_network", False)
    msg = registry.update_cache(allow_network=allow_network)
    console.print(msg)

    summary = registry.last_update_summary()
    new = summary.get("new", [])
    updated = summary.get("updated", [])
    removed = summary.get("removed", [])

    total_changes = len(new) + len(updated) + len(removed)
    if total_changes == 0:
        console.print("[success]✓[/success] Library is up to date")
        console.print("[muted]No new tasks or updates available.[/muted]")
        return 0

    from rich.table import Table

    console.print(f"[success]✓[/success] Found {total_changes} change(s) in library\n")

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Task", style="accent")
    table.add_column("Change", style="info")

    for name in sorted(new):
        table.add_row(name, "[success]new[/success]")
    for name in sorted(updated):
        table.add_row(name, "[warning]updated[/warning]")
    for name in sorted(removed):
        table.add_row(name, "[error]removed[/error]")

    console.print(table)
    console.print()

    # Helpful next steps
    if new:
        console.print("[header]New tasks available:[/header]")
        console.print(
            "  • Run [accent]'task list --source=library'[/accent] to see all library tasks"
        )
        console.print(
            "  • Run [accent]'task use <name>'[/accent] to install and activate a new task"
        )
        console.print()

    if updated:
        console.print("[header]Tasks with updates:[/header]")
        console.print(
            "  • Run [accent]'task sync'[/accent] to check if your workspace tasks are outdated"
        )
        console.print(
            "  • Run [accent]'task sync --update'[/accent] to automatically update outdated tasks"
        )
        console.print()

    if removed:
        console.print("[header]Removed tasks:[/header]")
        console.print("  • These tasks are no longer in the library")
        console.print("  • Your workspace copies will continue to work")
        console.print()

    return 0


def cmd_task_diagnose(args) -> int:
    """Run workspace health check."""
    console = get_console(args)
    registry = BuiltinRegistry()

    console.print("→ Running workspace health check...\n")

    # Check tasks
    workspace_tasks = user_config.list_custom_tasks()
    num_tasks = len(workspace_tasks)
    num_customized = 0
    num_outdated = 0
    num_orphaned = 0
    import_errors = []

    for task_name in workspace_tasks.keys():
        sync_status = registry.task_sync_status(task_name, user_config.tasks_dir)
        status = sync_status.get("status")

        if status == "modified":
            num_customized += 1
        elif status == "missing":
            num_orphaned += 1

        # Check for outdated
        record = registry.manifest.task_record(task_name)
        if record:
            remote_hash = record.get("remote_hash")
            workspace_hash = sync_status.get("workspace_hash")
            if remote_hash and workspace_hash and remote_hash != workspace_hash:
                num_outdated += 1

    # Check cache status
    reg_status = registry.registry_status()
    commit = reg_status.get("commit", "unknown")
    synced_at = reg_status.get("synced_at")

    # Calculate cache age
    cache_age = "unknown"
    if synced_at:
        try:
            from datetime import datetime, timezone

            synced_time = datetime.fromisoformat(synced_at)
            now = datetime.now(timezone.utc)
            delta = now - synced_time
            hours = delta.total_seconds() / 3600
            if hours < 1:
                cache_age = "< 1 hour ago"
            elif hours < 24:
                cache_age = f"{int(hours)} hours ago"
            else:
                days = int(hours / 24)
                cache_age = f"{days} days ago"
        except Exception:
            pass

    # Check network
    network_ok = True
    try:
        import socket

        socket.create_connection(("github.com", 443), timeout=2).close()
    except Exception:
        network_ok = False

    # Calculate cache size
    cache_size = "unknown"
    try:
        import os

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(registry.cache_root):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        cache_size = f"{total_size / (1024*1024):.1f} MB"
    except Exception:
        pass

    # Overall health
    issues = num_outdated + num_orphaned + len(import_errors)
    health = "GOOD ✓" if issues == 0 else "NEEDS ATTENTION ⚠"
    health_style = "success" if issues == 0 else "warning"

    # Display results
    console.print("[header]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/header]")
    console.print(
        f"[header] WORKSPACE HEALTH: [{health_style}]{health}[/{health_style}][/header]"
    )
    console.print("[header]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/header]\n")

    console.print("[info]Tasks:[/info]")
    console.print(f"  • {num_tasks} tasks installed in workspace")
    if num_customized > 0:
        console.print(f"  • {num_customized} customized")
    if num_outdated > 0:
        console.print(f"  • {num_outdated} outdated")
    if len(import_errors) > 0:
        console.print(f"  • {len(import_errors)} with import errors")
    if num_orphaned > 0:
        console.print(f"  • {num_orphaned} orphan(s)")

    console.print("\n[info]Cache:[/info]")
    console.print(f"  • Library cache: {commit} ({cache_age})")
    console.print(f"  • Network: {'OK' if network_ok else 'Offline'}")
    console.print(f"  • Cache size: {cache_size}")

    console.print("\n[info]Configuration:[/info]")
    active_task = user_config.get_active_task()
    workspace_dir = user_config.config_dir
    tasks_dir = user_config.tasks_dir

    console.print(
        f"  • Active task: {active_task or 'None'} {'✓' if active_task else '✗'}"
    )
    console.print(
        f"  • Workspace: {workspace_dir} {'✓' if workspace_dir.exists() else '✗'}"
    )
    console.print(
        f"  • Tasks directory: {tasks_dir} {'✓' if tasks_dir.exists() else '✗'}"
    )

    if issues > 0:
        console.print("\n[warning]Issues:[/warning]")
        if num_outdated > 0:
            console.print(f"  ⚠ {num_outdated} outdated task(s)")
        if num_orphaned > 0:
            console.print(f"  ⚠ {num_orphaned} orphaned task file(s)")

        console.print("\n[info]Recommendations:[/info]")
        if num_outdated > 0:
            console.print(
                "  • Run [accent]'task sync --update'[/accent] to update outdated tasks"
            )
        if num_orphaned > 0:
            console.print(
                "  • Run [accent]'task delete <name>'[/accent] to remove orphaned tasks"
            )
        if num_customized > 0:
            console.print("  • Review customizations before updating")

    console.print(
        f"\n[info]Overall:[/info] Your workspace is {health.lower()}{' with minor cleanup needed' if issues > 0 else ''}"
    )

    return 0


def cmd_task_diff(args) -> int:
    """Show differences between workspace task and its source."""
    console = get_console(args)
    registry = BuiltinRegistry()
    task_name = args.task_name

    # Check if task exists in workspace
    workspace_path = user_config.tasks_dir / f"{task_name}.py"
    if not workspace_path.exists():
        console.print(f"[error]✗[/error] Task '{task_name}' not found in workspace")
        console.print("[muted]Install it first with: task install {task_name}[/muted]")
        return 1

    # Get task from library
    task = registry.get_task(task_name)
    if not task:
        console.print(f"[error]✗[/error] Task '{task_name}' not found in library")
        console.print(
            "[muted]This task may be user-created (no source to compare).[/muted]"
        )
        return 1

    # Get source path (try cache first, then package)
    cache_path = registry._cache_path_for(task.path)
    if cache_path.exists():
        source_path = cache_path
        source_label = "library"
    else:
        # Try package
        try:
            resource = registry._pkg_task_resource(task.path)
            if resource.is_file():
                # Read from package resource
                with resource.open("r", encoding="utf-8") as f:
                    source_content = f.read()
                source_path = None
                source_label = "bundled package"
            else:
                console.print(f"[error]✗[/error] Source not found for '{task_name}'")
                return 1
        except Exception:
            console.print(f"[error]✗[/error] Source not found for '{task_name}'")
            return 1

    # Read both files
    try:
        workspace_content = workspace_path.read_text(encoding="utf-8")
        if source_path:
            source_content = source_path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[error]✗[/error] Failed to read files: {exc}")
        return 1

    # Check if identical
    if workspace_content == source_content:
        console.print(
            f"[success]✓[/success] No differences - workspace copy matches {source_label}"
        )
        return 0

    # Show diff
    console.print(f"→ Comparing workspace copy vs {source_label} source\n")
    console.print("[header]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/header]")
    console.print("[header] DIFFERENCES[/header]")
    console.print("[header]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/header]\n")

    # Generate unified diff
    import difflib

    workspace_lines = workspace_content.splitlines(keepends=True)
    source_lines = source_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        source_lines,
        workspace_lines,
        fromfile=f"{source_label}/{task.path}",
        tofile=f"workspace/{task_name}.py",
        lineterm="",
        n=args.context,
    )

    diff_lines = list(diff)
    if not diff_lines:
        console.print("[muted]No differences detected (whitespace may differ)[/muted]")
        return 0

    # Display diff with color if enabled
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            console.print(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            console.print(f"[info]{line}[/info]")
        elif line.startswith("-"):
            console.print(f"[error]{line}[/error]")
        elif line.startswith("+"):
            console.print(f"[success]{line}[/success]")
        else:
            console.print(f"[muted]{line}[/muted]")

    console.print(
        "\n[muted]Use 'task sync --update' to apply source changes (backup will be created)[/muted]"
    )

    return 0


def cmd_task_search(args) -> int:
    """Search for tasks across all sources."""
    console = get_console(args)
    registry = BuiltinRegistry()
    query = args.query.lower()

    console.print(f"→ Searching for [accent]'{args.query}'[/accent]...\n")

    results = []

    # Search library tasks
    if args.source in ["all", "library"]:
        try:
            # Get registry data
            if registry._cache_index_path().exists():
                index_text = registry._cache_index_path().read_text(encoding="utf-8")
            else:
                index_text = registry._pkg_index_text()

            index = json.loads(index_text)
            for entry in index.get("tasks", []):
                name = entry.get("name", "")
                description = entry.get("description", "")
                category = entry.get("category", "")

                if (
                    query in name.lower()
                    or query in description.lower()
                    or query in category.lower()
                ):

                    # Check if installed
                    workspace_path = user_config.tasks_dir / f"{name}.py"
                    status = "installed" if workspace_path.exists() else "available"

                    results.append(
                        {
                            "name": name,
                            "description": description,
                            "source": "library",
                            "status": status,
                            "category": category,
                        }
                    )
        except Exception:
            pass

    # Search workspace tasks
    if args.source in ["all", "workspace"]:
        workspace_tasks = user_config.list_custom_tasks()
        for task_name, task_info in workspace_tasks.items():
            description = task_info.get("description", "")
            if query in task_name.lower() or query in description.lower():
                results.append(
                    {
                        "name": task_name,
                        "description": description,
                        "source": "workspace",
                        "status": "installed",
                        "category": "",
                    }
                )

    # Display results
    if not results:
        console.print(
            f"[warning]No tasks found matching [accent]'{args.query}'[/accent][/warning]"
        )
        console.print(
            "[muted]Try a different search term or run 'task list' to see all tasks.[/muted]"
        )
        return 0

    console.print(f"[success]Found {len(results)} task(s):[/success]\n")

    from rich.table import Table

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Task", style="accent")
    table.add_column("Source", style="muted")
    table.add_column("Status", style="info")
    table.add_column("Description", style="dim")

    for result in sorted(results, key=lambda x: x["name"]):
        status_text = result["status"]
        if status_text == "installed":
            status_text = "[success]installed[/success]"
        else:
            status_text = "[muted]available[/muted]"

        table.add_row(
            result["name"],
            f"[{result['source']}]",
            status_text,
            (
                result["description"][:50] + "..."
                if len(result["description"]) > 50
                else result["description"]
            ),
        )

    console.print(table)
    console.print()

    # Show helpful next step
    available_count = sum(1 for r in results if r["status"] == "available")
    if available_count > 0:
        console.print(
            "[muted]Use 'task use <name>' to install and activate a task[/muted]"
        )

    return 0


def cmd_source(args) -> int:
    """Deprecated: execute source management commands (alias of 'input')."""
    # Friendly deprecation hint
    message("warning", "'source' is deprecated. Use 'input' (e.g., 'input set').")
    # No subcommand → show elegant source help
    if not getattr(args, "source_action", None):
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "source")
        return 0

    if args.source_action == "set":
        return cmd_source_set(args)
    elif args.source_action == "unset":
        return cmd_source_unset(args)
    elif args.source_action == "show":
        return cmd_source_show(args)
    else:
        message("error", "No source action specified")
        return 1


def cmd_input(args) -> int:
    """Execute input management commands (preferred)."""
    # No subcommand → show elegant input help
    if not getattr(args, "input_action", None):
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "input")
        return 0

    # Reuse the existing source handlers to avoid duplication
    # Note: we pass through args; it carries 'source_path' if provided
    if args.input_action == "set":
        return cmd_source_set(args)
    elif args.input_action == "unset":
        return cmd_source_unset(args)
    elif args.input_action == "show":
        return cmd_source_show(args)
    else:
        message("error", "No input action specified")
        return 1


def cmd_source_set(args) -> int:
    """Set the active input path (stored internally as 'source')."""
    try:
        console = get_console(
            args if isinstance(args, argparse.Namespace) else None
        )
        source_path = getattr(args, "source_path", None)
        source_path = _strip_wrapping_quotes(source_path)

        # If path provided directly, use it
        if source_path:
            path = Path(source_path).expanduser().resolve()
            if not path.exists():
                message("error", f"Input path does not exist: {path}")
                return 1

            if user_config.set_active_source(str(path)):
                message("success", f"Active input set to: {path}")
                _render_input_summary(console, input_path=path)
                return 0
            else:
                message("error", "Failed to save active input configuration")
                return 1

        # Otherwise, use interactive selection
        selected_source = user_config.select_active_source_interactive()
        selected_source = _strip_wrapping_quotes(selected_source)
        if selected_source is None:
            message("info", "Canceled")
            return 0
        elif selected_source == "NONE":
            # User chose to have no default
            if user_config.set_active_source(None):
                message(
                    "success", "Active input cleared - will prompt for input each time"
                )
                return 0
            else:
                message("error", "Failed to clear active input")
                return 1
        else:
            # User selected a path
            if user_config.set_active_source(selected_source):
                message("success", f"Active input set to: {selected_source}")
                _render_input_summary(console, input_path=Path(selected_source))
                return 0
            else:
                message("error", "Failed to save active input configuration")
                return 1

    except Exception as e:
        message("error", f"Failed to set active input: {e}")
        return 1


def cmd_source_unset(_args) -> int:
    """Clear the active input path."""
    try:
        if user_config.set_active_source(None):
            message("success", "Active input cleared")
            return 0
        else:
            message("error", "Failed to clear active input")
            return 1

    except Exception as e:
        message("error", f"Failed to unset active input: {e}")
        return 1


def cmd_source_show(_args) -> int:
    """Show the current active input path."""
    try:
        active_source = user_config.get_active_source()

        if active_source is None:
            message("info", "No active input path is currently set.")
            message("info", "Set one with: autocleaneeg-pipeline input set")
        else:
            path = Path(active_source)
            if path.exists():
                if path.is_file():
                    message("info", f"Active input (file): {active_source}")
                elif path.is_dir():
                    message("info", f"Active input (directory): {active_source}")
                else:
                    message("info", f"Active input: {active_source}")
            else:
                message(
                    "warning", f"Active input path no longer exists: {active_source}"
                )
                message("info", "Consider setting a new active input.")

        return 0

    except Exception as e:
        message("error", f"Failed to show active input: {e}")
        return 1


def cmd_config(args) -> int:
    """Execute configuration management commands."""
    if args.config_action == "show":
        return cmd_config_show(args)
    elif args.config_action == "setup":
        return cmd_config_setup(args)
    elif args.config_action == "reset":
        return cmd_config_reset(args)
    elif args.config_action == "export":
        return cmd_config_export(args)
    elif args.config_action == "import":
        return cmd_config_import(args)
    else:
        message("error", "No config action specified")
        return 1


def cmd_config_show(_args) -> int:
    """Show user configuration directory."""
    config_dir = user_config.config_dir
    message("info", f"User configuration directory: {config_dir}")

    custom_tasks = user_config.list_custom_tasks()
    print(f"  • Custom tasks: {len(custom_tasks)}")
    print(f"  • Tasks directory: {config_dir / 'tasks'}")
    print(f"  • Config file: {config_dir / 'user_config.json'}")

    return 0


def cmd_config_setup(_args) -> int:
    """Reconfigure workspace location."""
    try:
        user_config.setup_workspace()
        return 0
    except Exception as e:
        message("error", f"Failed to reconfigure workspace: {str(e)}")
        return 1


def cmd_config_reset(args) -> int:
    """Reset user configuration to defaults."""
    if not args.confirm:
        message("error", "This will delete all custom tasks and reset configuration.")
        print("Use --confirm to proceed with reset.")
        return 1

    try:
        user_config.reset_config()
        message("info", "User configuration reset to defaults")
        return 0
    except Exception as e:
        message("error", f"Failed to reset configuration: {str(e)}")
        return 1


def cmd_config_export(args) -> int:
    """Export user configuration."""
    try:
        if user_config.export_config(args.export_path):
            return 0
        else:
            return 1
    except Exception as e:
        message("error", f"Failed to export configuration: {str(e)}")
        return 1


def cmd_config_import(args) -> int:
    """Import user configuration."""
    try:
        if user_config.import_config(args.import_path):
            return 0
        else:
            return 1
    except Exception as e:
        message("error", f"Failed to import configuration: {str(e)}")
        return 1


def cmd_settings(args) -> int:
    """Execute settings management commands."""
    if not getattr(args, "settings_action", None):
        # Show settings help when no subcommand provided
        console = get_console(args)
        _simple_header(console)
        _print_startup_context(console)
        _print_root_help(console, "settings")
        return 0

    if args.settings_action == "theme":
        return cmd_settings_theme(args)
    else:
        message("error", "Unknown settings action")
        return 1


def cmd_settings_theme(args) -> int:
    """Configure CLI color theme."""
    console = get_console(args)

    # Handle --clear flag
    if args.clear:
        if user_config.set_theme(None):
            console.print(
                "[success]✓[/success] Theme preference cleared (reverted to auto)"
            )
            console.print(
                "[muted]The CLI will now auto-detect your terminal background.[/muted]"
            )
            return 0
        else:
            console.print("[error]✗[/error] Failed to clear theme preference")
            return 1

    # If theme_name provided, set it directly
    if args.theme_name:
        theme = args.theme_name
        if user_config.set_theme(theme):
            console.print(
                f"[success]✓[/success] Theme set to: [accent]{theme}[/accent]"
            )
            console.print(
                "[muted]Restart your CLI session to see the new theme.[/muted]"
            )
            return 0
        else:
            console.print(f"[error]✗[/error] Failed to set theme to: {theme}")
            return 1

    # Interactive theme selection
    selected = user_config.select_theme_interactive()
    if selected:
        if user_config.set_theme(selected):
            console.print(
                f"\n[success]✓[/success] Theme set to: [accent]{selected}[/accent]"
            )
            console.print(
                "[muted]Restart your CLI session to see the new theme.[/muted]"
            )
            return 0
        else:
            console.print("[error]✗[/error] Failed to save theme preference")
            return 1
    else:
        console.print("[muted]Theme selection cancelled.[/muted]")
        return 0


def cmd_clean_task(args) -> int:
    """Remove task output directory and database entries."""
    console = get_console(args)

    # Determine output directory
    output_dir = args.output_dir or user_config._get_workspace_path()

    # Find matching task directories (could be task name or dataset name)
    potential_dirs = []

    # First try exact match
    exact_match = output_dir / args.task
    if exact_match.exists() and (exact_match / "bids").exists():
        potential_dirs.append(exact_match)

    # If no exact match, search for directories containing the task name
    if not potential_dirs:
        for item in output_dir.iterdir():
            if item.is_dir() and args.task.lower() in item.name.lower():
                if (item / "bids").exists():
                    potential_dirs.append(item)

    if not potential_dirs:
        message("warning", f"No task directories found matching: {args.task}")
        message("info", f"Searched in: {output_dir}")
        return 1

    if len(potential_dirs) > 1:
        console.print(
            f"\n[warning]Multiple directories found matching '{args.task}':[/warning]"
        )
        for i, dir_path in enumerate(potential_dirs, 1):
            console.print(f"  {i}. {dir_path.name}")
        console.print("\nPlease be more specific or use the full directory name.")
        return 1

    # Use the single matching directory
    task_root_dir = potential_dirs[0]

    # Count files and calculate size
    total_files = 0
    total_size = 0
    for item in task_root_dir.rglob("*"):
        if item.is_file():
            total_files += 1
            total_size += item.stat().st_size

    # Format size for display
    size_mb = total_size / (1024 * 1024)
    size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb / 1024:.1f} GB"

    # Database entries (if database exists) - search by both task name and directory name
    db_entries = 0
    if DB_PATH and Path(DB_PATH).exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            # Search for entries matching either the provided task name or the directory name
            cursor.execute(
                "SELECT COUNT(*) FROM runs WHERE task = ? OR task = ?",
                (args.task, task_root_dir.name),
            )
            db_entries = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass

    # Display what will be deleted
    console.print("\n[title]Task Cleanup Summary:[/title]")
    console.print(f"Task: [accent]{args.task}[/accent]")
    console.print(f"Directory: [accent]{task_root_dir}[/accent]")
    console.print(f"Files: [warning]{total_files:,}[/warning]")
    console.print(f"Size: [warning]{size_str}[/warning]")
    if db_entries > 0:
        console.print(f"Database entries: [warning]{db_entries}[/warning]")

    if args.dry_run:
        console.print("\n[warning]DRY RUN - No files will be deleted[/warning]")
        return 0

    # Simple Y/N confirmation
    if not args.force:
        confirm = (
            console.input("\n[error]Delete this task? (Y/N):[/error] ").strip().upper()
        )
        if confirm != "Y":
            console.print("[warning]Cancelled[/warning]")
            return 1

    # Perform deletion
    try:
        # Delete filesystem
        console.print("\n[header]Cleaning task files...[/header]")
        shutil.rmtree(task_root_dir)
        console.print(f"[success]✓ Removed directory: {task_root_dir}[/success]")

        # Delete database entries for both task name and directory name
        if db_entries > 0 and DB_PATH:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM runs WHERE task = ? OR task = ?",
                (args.task, task_root_dir.name),
            )
            conn.commit()
            conn.close()
            console.print(f"[success]✓ Removed {db_entries} database entries[/success]")

        console.print("\n[success]Task cleaned successfully![/success]")
        return 0

    except Exception as e:
        console.print(f"\n[error]Error during cleanup: {e}[/error]")
        return 1


def cmd_report(args) -> int:
    """Dispatch for 'report' subcommands."""
    action = getattr(args, "report_action", None)
    if action == "create":
        return cmd_report_create(args)
    if action == "chat":
        return cmd_report_chat(args)
    message("error", "No report action specified")
    return 1


def cmd_report_create(args) -> int:
    """Generate textual reports from a run context."""
    import json
    from pathlib import Path

    from autoclean.reporting.llm_reporting import create_reports, run_context_from_dict

    data = json.loads(Path(args.context_json).read_text())
    ctx = run_context_from_dict(data)
    # If a run_id is provided on CLI and differs, override to maintain traceability
    if getattr(args, "run_id", None):
        try:
            object.__setattr__(ctx, "run_id", args.run_id)
        except Exception:
            # Dataclass is mutable by default; fallback assignment
            ctx.run_id = args.run_id
    out_dir = Path(args.out_dir)
    create_reports(ctx, out_dir)
    message("success", f"Wrote reports under {out_dir}")
    return 0


def cmd_report_chat(args) -> int:
    """Interactive chat about a run context."""
    import json
    from pathlib import Path

    from autoclean.reporting.llm_reporting import LLMClient

    # Ensure database path is set so we can read the latest run
    try:
        from autoclean.utils.database import set_database_path
        from autoclean.utils.user_config import user_config as _ucfg

        out_dir = _ucfg.get_default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        set_database_path(out_dir)
    except Exception:
        pass

    def _select_run_interactively(records: list[dict]) -> dict | None:
        try:
            from rich.prompt import Prompt as _Prompt
            from rich.table import Table as _Table

            from autoclean.utils.console import get_console as _get_console
        except Exception:
            return None

        if not records:
            return None

        def _key(r):
            return (r.get("created_at") or "", r.get("id") or 0)

        recs = sorted(records, key=_key, reverse=True)

        # Build rows
        rows = []
        for r in recs:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    import json as _json

                    meta = _json.loads(meta)
                except Exception:
                    meta = {}
            backup = bool(meta.get("directory_backup"))
            json_sum = meta.get("json_summary") or {}
            outputs = json_sum.get("outputs") or {}
            outputs_count = (
                len(outputs)
                if isinstance(outputs, dict)
                else (len(outputs) if hasattr(outputs, "__len__") else 0)
            )
            basename = Path(r.get("unprocessed_file") or "").name
            rows.append(
                {
                    "rec": r,
                    "rid": r.get("run_id") or "",
                    "created": r.get("created_at") or "",
                    "task": r.get("task") or "",
                    "file": basename,
                    "status": r.get("status") or "",
                    "success": "Yes" if r.get("success") else "No",
                    "backup": "Yes" if backup else "No",
                    "artifacts": str(outputs_count or 0),
                }
            )

        c = _get_console()
        c.print()
        tbl = _Table(show_header=True, header_style="header", box=None, padding=(0, 1))
        tbl.add_column("#", style="muted", width=4)
        tbl.add_column("Run ID", style="accent")
        tbl.add_column("Created", style="muted")
        tbl.add_column("Task")
        tbl.add_column("File")
        tbl.add_column("Status")
        tbl.add_column("Success")
        tbl.add_column("Backup")
        tbl.add_column("Artifacts")
        max_rows = min(20, len(rows))
        for i, r in enumerate(rows[:max_rows], 1):
            rid_short = r["rid"][:8] + ("…" if len(r["rid"]) > 8 else "")
            tbl.add_row(
                str(i),
                rid_short,
                r["created"],
                r["task"],
                r["file"],
                r["status"],
                r["success"],
                r["backup"],
                r["artifacts"],
            )
        c.print(tbl)
        c.print()
        ans = _Prompt.ask("Select a run number (Enter to cancel)", default="")
        ans = (ans or "").strip()
        if not ans:
            return None
        try:
            idx = int(ans)
        except Exception:
            return None
        if idx < 1 or idx > max_rows:
            return None
        return recs[idx - 1]

    def _load_latest_context_json() -> str | None:
        try:
            from rich.prompt import Confirm as _Confirm
            from rich.text import Text as _Text

            from autoclean import __version__ as ac_version
            from autoclean.utils.console import get_console as _get_console
            from autoclean.utils.database import manage_database_conditionally
        except Exception:
            return None

        try:
            records = manage_database_conditionally("get_collection") or []
        except Exception:
            records = []
        if not records:
            return None

        # Pick most recent by created_at (ISO) or id as fallback
        def _key(r):
            return (r.get("created_at") or "", r.get("id") or 0)

        latest = sorted(records, key=_key)[-1]
        # Ask consent to use latest; else show interactive selector
        use_latest = True
        try:
            _c = _get_console()
            t = _Text()
            t.append("Use latest run? ", style="title")
            _c.print(t)
            use_latest = _Confirm.ask(
                f"Run {latest.get('run_id', '')} on {latest.get('created_at', '')} (task: {latest.get('task', '')})?",
                default=True,
            )
        except Exception:
            use_latest = True
        rec = latest if use_latest else _select_run_interactively(records)
        if not rec:
            # User canceled selection
            return "__CANCELLED__"
        meta = rec.get("metadata") or {}
        # Parse metadata JSON if returned as a string from get_collection
        if isinstance(meta, str):
            try:
                import json as _json

                meta = _json.loads(meta)
            except Exception:
                meta = {}
        spd = meta.get("step_prepare_directories") or {}
        metadata_dir = Path(spd.get("metadata", ""))
        bids_dir = Path(spd.get("bids", "")) if spd.get("bids") else None
        reports_dir = Path(spd.get("reports", "")) if spd.get("reports") else None
        if not metadata_dir.exists():
            return None

        # Prefer previously generated LLM context
        try:
            from autoclean.utils.path_resolution import resolve_moved_path

            metadata_dir_resolved = resolve_moved_path(metadata_dir)
        except Exception:
            metadata_dir_resolved = metadata_dir
        try:
            reports_dir_resolved = (
                resolve_moved_path(reports_dir) if reports_dir else None
            )
        except Exception:
            reports_dir_resolved = reports_dir

        llm_candidates = []
        basename = Path(rec.get("unprocessed_file") or "").stem
        if reports_dir_resolved:
            llm_candidates.append(
                reports_dir_resolved / "llm" / basename / "context.json"
            )
            llm_candidates.append(reports_dir_resolved / "llm" / "context.json")
        llm_candidates.append(metadata_dir_resolved / "llm_reports" / "context.json")

        for llm_ctx in llm_candidates:
            if llm_ctx and llm_ctx.exists():
                try:
                    return llm_ctx.read_text(encoding="utf-8")
                except Exception:
                    continue

        # Reconstruct context from per-file processing log + PDF
        try:
            import ast
            import csv

            input_file = rec.get("unprocessed_file") or ""
            if not basename:
                basename = Path(input_file).stem
            derivatives_root = Path(spd.get("clean", metadata_dir_resolved.parent))
            logs_dir = Path(spd.get("logs")) if spd.get("logs") else None
            per_file_candidates = []
            if reports_dir_resolved:
                per_file_candidates.append(
                    reports_dir_resolved
                    / "run_reports"
                    / f"{basename}_processing_log.csv"
                )
            per_file_candidates.append(
                derivatives_root / f"{basename}_processing_log.csv"
            )
            if logs_dir:
                per_file_candidates.append(logs_dir / f"{basename}_processing_log.csv")

            # Use exports path from step_prepare_directories
            if spd.get("exports"):
                final_files_dir_resolved = Path(spd["exports"])  # direct path
                per_file_candidates.append(
                    final_files_dir_resolved / f"{basename}_processing_log.csv"
                )

            per_file_csv = next((p for p in per_file_candidates if p.exists()), None)
            if per_file_csv is None:
                return None

            row = None
            with per_file_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
            if not row:
                return None

            pdf_name = rec.get("report_file") or f"{basename}_autoclean_report.pdf"
            pdf_candidates = []
            if reports_dir_resolved:
                pdf_candidates.append(reports_dir_resolved / "run_reports" / pdf_name)
                pdf_candidates.append(reports_dir_resolved / pdf_name)
            pdf_candidates.append(metadata_dir_resolved / pdf_name)

            pdf_path = next(
                (p for p in pdf_candidates if p.exists()), pdf_candidates[-1]
            )

            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            def _to_int(x):
                try:
                    return int(float(x))
                except Exception:
                    return None

            def _to_list_of_floats(x):
                if x is None or x == "":
                    return []
                try:
                    v = ast.literal_eval(x)
                    if isinstance(v, (list, tuple)):
                        return [float(y) for y in v]
                    return [float(v)]
                except Exception:
                    parts = [
                        p
                        for p in str(x).replace("[", "").replace("]", "").split(",")
                        if p.strip()
                    ]
                    out = []
                    for p in parts:
                        try:
                            out.append(float(p))
                        except Exception:
                            pass
                    return out

            def _to_list_of_ints(x):
                try:
                    v = ast.literal_eval(x)
                    if isinstance(v, (list, tuple)):
                        return [int(float(y)) for y in v]
                    return []
                except Exception:
                    return []

            # Minimal context dict compatible with RunContext shape
            data = {
                "run_id": rec.get("run_id") or "",
                "dataset_name": None,
                "input_file": input_file,
                "montage": None,
                "resample_hz": _to_float(row.get("proc_sRate1")),
                "reference": None,
                "filter_params": {
                    "l_freq": _to_float(row.get("proc_filt_lowcutoff")),
                    "h_freq": _to_float(row.get("proc_filt_highcutoff")),
                    "notch_freqs": _to_list_of_floats(row.get("proc_filt_notch")),
                    "notch_widths": _to_float(row.get("proc_filt_notch_width")),
                },
                "ica": None,
                "epochs": None,
                "durations_s": _to_float(row.get("proc_xmax_post")),
                "n_channels": _to_int(row.get("net_nbchan_post")),
                "bids_root": str(bids_dir) if bids_dir else None,
                "bids_subject_id": None,
                "pipeline_version": ac_version,
                "mne_version": None,
                "compliance_user": None,
                "notes": ([f"flags: {row['flags']}"] if row.get("flags") else []),
                "figures": (
                    {"autoclean_report_pdf": str(pdf_path)} if pdf_path.exists() else {}
                ),
            }

            # Epochs
            try:
                v = ast.literal_eval(row.get("epoch_limits", ""))
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    data["epochs"] = {
                        "tmin": float(v[0]) if v[0] is not None else None,
                        "tmax": float(v[1]) if v[1] is not None else None,
                        "baseline": None,
                        "total_epochs": None,
                        "kept_epochs": _to_int(row.get("epoch_trials")),
                        "rejected_epochs": _to_int(row.get("epoch_badtrials")),
                        "rejection_rules": {},
                    }
                    k = data["epochs"]["kept_epochs"]
                    rj = data["epochs"]["rejected_epochs"]
                    if k is not None and rj is not None:
                        data["epochs"]["total_epochs"] = k + rj
            except Exception:
                pass

            # ICA
            ncomp = _to_int(row.get("proc_nComps"))
            removed = _to_list_of_ints(row.get("proc_removeComps"))
            if ncomp is not None or removed:
                data["ica"] = {
                    "method": row.get("ica_method") or "unspecified",
                    "n_components": ncomp,
                    "removed_indices": removed,
                    "labels_histogram": {},
                    "classifier": row.get("classification_method") or None,
                }

            return json.dumps(data, indent=2)
        except Exception:
            return None

    if getattr(args, "context_json", None):
        ctx = Path(args.context_json).read_text()
    else:
        ctx = _load_latest_context_json()
        if ctx == "__CANCELLED__":
            message("info", "Canceled")
            return 0
        if not ctx:
            message(
                "error",
                "Could not locate latest run context or reconstruct from outputs. Provide --context-json explicitly.",
            )
            return 1
    # Graceful guard: require API key for chat
    try:
        import os as _os

        api_key = _os.getenv("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        message(
            "warning", "OPENAI_API_KEY not set. Chat requires an API key for the LLM."
        )
        message(
            "info",
            "Set the key in your environment, then rerun: export OPENAI_API_KEY=sk-...",
        )
        return 0

    llm = LLMClient()
    print("Type a question about this run (Ctrl-C to exit).")
    try:
        while True:
            q = input("> ")
            if not q.strip():
                continue
            system = "You answer questions strictly using the JSON context."
            user = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer briefly."
            schema = json.dumps(
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            )
            ans = llm.generate_json(system, user, schema)
            print(ans.get("answer", ""))
    except KeyboardInterrupt:
        print()
    return 0


def cmd_view(args) -> int:
    """View EEG files using the autocleaneeg-view module."""
    file_path = getattr(args, "file", None)
    if file_path is None:
        try:
            active_source = user_config.get_active_source()
        except Exception:
            active_source = None
        if active_source:
            file_path = active_source
        else:
            message(
                "error",
                "No EEG file specified. Provide a path or set an active input before running 'view'.",
            )
            return 1

    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        message("error", f"File not found: {file_path}")
        return 1
    if file_path.is_dir():
        message("error", f"Path points to a directory, not an EEG file: {file_path}")
        return 1

    try:
        from autocleaneeg_view.viewer import load_eeg_file, view_eeg
    except ImportError as exc:
        message(
            "error",
            "autocleaneeg-view module not available. Install it with 'pip install autocleaneeg-view'.",
        )
        message("debug", f"Import error: {exc}")
        return 1

    message("info", f"Preparing {file_path.name} in autocleaneeg-view...")

    try:
        if args.no_view:
            load_eeg_file(str(file_path))
            message("success", f"Loaded {file_path.name} (viewer not opened)")
        else:
            raw = load_eeg_file(str(file_path))
            view_eeg(raw)
            message("success", "Viewer closed")
        return 0
    except FileNotFoundError as exc:
        message("error", f"Failed to open {file_path.name}: {exc}")
        return 1
    except Exception as exc:  # Broad guard: renderer/backend issues, etc.
        message("error", f"Failed to launch viewer: {exc}")
        return 1


def cmd_auth(args) -> int:
    """Dispatch for 'auth' subcommands."""
    action = getattr(args, "auth_action", None)
    if action == "login":
        return cmd_login(args)
    if action == "logout":
        return cmd_logout(args)
    if action == "whoami":
        return cmd_whoami(args)
    if action == "diagnostics":
        return cmd_auth0_diagnostics(args)
    if action == "setup":
        return _setup_compliance_mode()
    if action == "enable":
        return _enable_compliance_mode()
    if action == "disable":
        return _disable_compliance_mode()
    message("error", "No auth action specified")
    return 1


def cmd_help(args) -> int:
    """Help alias: shows the same styled root help as '-h/--help'."""
    console = get_console(args)
    _simple_header(console)
    _print_startup_context(console)
    topic = getattr(args, "topic", None)
    _print_root_help(console, topic.strip().lower() if isinstance(topic, str) else None)
    return 0


def cmd_tutorial(_args) -> int:
    """Show a helpful tutorial for first-time users."""
    console = get_console()

    # Use the tutorial header for consistent branding
    _simple_header(console, "Tutorial", "Interactive guide to AutoClean EEG")

    console.print("\n[title]🚀 Welcome to the AutoClean EEG Tutorial![/title]")
    console.print(
        "This tutorial will walk you through the basics of using AutoClean EEG."
    )
    console.print("\n[header]Step 1: Configure your workspace[/header]")
    console.print(
        "The first step is to set up your workspace. This is where AutoClean EEG will store its configuration and any custom tasks you create."
    )
    console.print("To do this, run the following command:")
    console.print("\n[accent]autocleaneeg-pipeline workspace[/accent]\n")

    console.print("\n[header]Step 2: List available tasks[/header]")
    console.print(
        "Once your workspace is set up, you can see the built-in processing tasks that are available."
    )
    console.print("To do this, run the following command:")
    console.print("\n[accent]autocleaneeg-pipeline task list[/accent]\n")

    console.print("\n[header]Step 3: Process a file[/header]")
    console.print(
        "Now you are ready to process a file. You will need to specify the task you want to use and the path to the file you want to process."
    )
    console.print(
        "For example, to process a file called 'data.raw' with the 'RestingEyesOpen' task, you would run the following command:"
    )
    console.print(
        "\n[accent]autocleaneeg-pipeline process RestingEyesOpen data.raw[/accent]\n"
    )

    return 0


def cmd_export_access_log(args) -> int:
    """Export database access log with integrity verification."""
    try:
        # Get workspace directory for database discovery and default output location
        workspace_dir = user_config._get_workspace_path()

        # Determine database path
        if args.database:
            db_path = Path(args.database)
        elif DB_PATH:
            db_path = DB_PATH / "pipeline.db"
        else:
            # Try to find database in workspace
            if workspace_dir:
                # Check for database directly in workspace directory
                workspace_db = workspace_dir / "pipeline.db"
                if workspace_db.exists():
                    db_path = workspace_db
                else:
                    # Fall back to checking workspace/output/ directory (most common location)
                    output_db = workspace_dir / "output" / "pipeline.db"
                    if output_db.exists():
                        db_path = output_db
                    else:
                        # Finally, look in output subdirectories (for multiple runs)
                        potential_outputs = workspace_dir / "output"
                        if potential_outputs.exists():
                            # Look for most recent output directory with database
                            for output_dir in sorted(
                                potential_outputs.iterdir(), reverse=True
                            ):
                                if output_dir.is_dir():
                                    potential_db = output_dir / "pipeline.db"
                                    if potential_db.exists():
                                        db_path = potential_db
                                        break
                            else:
                                message(
                                    "error",
                                    "No database found in workspace directory, output directory, or output subdirectories",
                                )
                                return 1
                        else:
                            message(
                                "error",
                                "No database found in workspace directory and no output directory exists",
                            )
                            return 1
            else:
                message(
                    "error", "No workspace configured and no database path provided"
                )
                return 1

        if not db_path.exists():
            message("error", f"Database file not found: {db_path}")
            return 1

        message("info", f"Using database: {db_path}")

        # Verify integrity first (need to temporarily set DB_PATH for verification)
        from autoclean.utils import database

        original_db_path = database.DB_PATH
        database.DB_PATH = db_path.parent

        try:
            integrity_result = verify_access_log_integrity()
        finally:
            database.DB_PATH = original_db_path

        if args.verify_only:
            if integrity_result["status"] == "valid":
                message("success", f"✓ {integrity_result['message']}")
                return 0
            elif integrity_result["status"] == "compromised":
                message("error", f"✗ {integrity_result['message']}")
                if "issues" in integrity_result:
                    for issue in integrity_result["issues"]:
                        message("error", f"  - {issue}")
                return 1
            else:
                message("error", f"✗ {integrity_result['message']}")
                return 1

        # Report integrity status
        if integrity_result["status"] == "valid":
            message("success", f"✓ {integrity_result['message']}")
        elif integrity_result["status"] == "compromised":
            message("warning", f"⚠ {integrity_result['message']}")
            if "issues" in integrity_result:
                for issue in integrity_result["issues"]:
                    message("warning", f"  - {issue}")
        else:
            message("warning", f"⚠ {integrity_result['message']}")

        # Query access log with filters
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM database_access_log WHERE 1=1"
        params = []

        # Apply date filters
        if args.start_date:
            query += " AND date(timestamp) >= ?"
            params.append(args.start_date)

        if args.end_date:
            query += " AND date(timestamp) <= ?"
            params.append(args.end_date)

        # Apply operation filter
        if args.operation:
            query += " AND operation LIKE ?"
            params.append(f"%{args.operation}%")

        query += " ORDER BY log_id ASC"

        cursor.execute(query, params)
        entries = cursor.fetchall()
        conn.close()

        if not entries:
            message("warning", "No access log entries found matching filters")
            return 0

        # Determine output file
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use .jsonl extension for JSON Lines format
            if args.format == "json":
                extension = "jsonl"
            else:
                extension = args.format
            filename = f"access-log-{timestamp}.{extension}"
            # Default to workspace directory, not current working directory
            output_file = workspace_dir / filename if workspace_dir else Path(filename)

        # Export data
        export_data = []
        for entry in entries:
            entry_dict = dict(entry)
            # Parse JSON fields for export
            if entry_dict.get("user_context"):
                try:
                    entry_dict["user_context"] = json.loads(entry_dict["user_context"])
                except json.JSONDecodeError:
                    pass
            if entry_dict.get("details"):
                try:
                    entry_dict["details"] = json.loads(entry_dict["details"])
                except json.JSONDecodeError:
                    pass
            export_data.append(entry_dict)

        # Write export file
        if args.format == "json":
            # Write as JSONL (JSON Lines) format - more compact and easier to process
            with open(output_file, "w") as f:
                # First line: metadata
                metadata = {
                    "type": "metadata",
                    "export_timestamp": datetime.now().isoformat(),
                    "database_path": str(db_path),
                    "total_entries": len(export_data),
                    "integrity_status": integrity_result["status"],
                    "integrity_message": integrity_result["message"],
                    "filters_applied": {
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "operation": args.operation,
                    },
                }
                f.write(json.dumps(metadata) + "\n")

                # Subsequent lines: one JSON object per access log entry
                for entry in export_data:
                    entry["type"] = "access_log"
                    f.write(json.dumps(entry) + "\n")

        elif args.format == "csv":
            with open(output_file, "w", newline="") as f:
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in export_data:
                        # Flatten JSON fields for CSV
                        csv_entry = {}
                        for key, value in entry.items():
                            if isinstance(value, (dict, list)):
                                csv_entry[key] = json.dumps(value)
                            else:
                                csv_entry[key] = value
                        writer.writerow(csv_entry)

        elif args.format == "human":
            with open(output_file, "w") as f:
                f.write("AutoClean Database Access Log Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Export Date: {datetime.now().isoformat()}\n")
                f.write(f"Database: {db_path}\n")
                f.write(f"Total Entries: {len(export_data)}\n")
                f.write(f"Integrity Status: {integrity_result['status']}\n")
                f.write(f"Integrity Message: {integrity_result['message']}\n\n")

                if args.start_date or args.end_date or args.operation:
                    f.write("Filters Applied:\n")
                    if args.start_date:
                        f.write(f"  Start Date: {args.start_date}\n")
                    if args.end_date:
                        f.write(f"  End Date: {args.end_date}\n")
                    if args.operation:
                        f.write(f"  Operation: {args.operation}\n")
                    f.write("\n")

                f.write("Access Log Entries:\n")
                f.write("-" * 30 + "\n\n")

                for i, entry in enumerate(export_data, 1):
                    f.write(f"Entry {i} (ID: {entry['log_id']})\n")
                    f.write(f"  Timestamp: {entry['timestamp']}\n")
                    f.write(f"  Operation: {entry['operation']}\n")

                    if entry.get("user_context"):
                        user_ctx = entry["user_context"]
                        if isinstance(user_ctx, dict):
                            # Handle both old and new format
                            user = user_ctx.get(
                                "user", user_ctx.get("username", "unknown")
                            )
                            host = user_ctx.get(
                                "host", user_ctx.get("hostname", "unknown")
                            )
                            f.write(f"  User: {user}\n")
                            f.write(f"  Host: {host}\n")

                    if entry.get("details") and entry["details"]:
                        f.write(
                            f"  Details: {json.dumps(entry['details'], indent=4)}\n"
                        )

                    f.write(f"  Hash: {entry['log_hash'][:16]}...\n")
                    f.write("\n")

        message("success", f"✓ Access log exported to: {output_file}")
        message("info", f"Format: {args.format}, Entries: {len(export_data)}")

        return 0

    except Exception as e:
        message("error", f"Failed to export access log: {e}")
        return 1


def cmd_login(args) -> int:
    """Execute the login command."""
    try:
        if not is_compliance_mode_enabled():
            message("error", "Compliance mode is not enabled.")
            message(
                "info",
                "Run 'autocleaneeg-pipeline auth setup' to enable compliance mode and configure Auth0.",
            )
            return 1

        auth_manager = get_auth0_manager()

        # Always refresh configuration with latest credentials from environment/.env
        try:
            message("debug", "Loading Auth0 credentials from environment/.env files...")
            auth_manager.configure_developer_auth0()
        except Exception as e:
            message("error", f"Failed to configure Auth0: {e}")
            message(
                "info",
                "Check your .env file or environment variables for Auth0 credentials.",
            )
            return 1

        # Double-check configuration after loading
        if not auth_manager.is_configured():
            message("error", "Auth0 not configured.")
            message(
                "info",
                "Check your .env file or environment variables for Auth0 credentials.",
            )
            return 1

        if auth_manager.is_authenticated():
            user_info = auth_manager.get_current_user()
            user_email = user_info.get("email", "Unknown") if user_info else "Unknown"
            message("info", f"Already logged in as: {user_email}")
            return 0

        message("info", "Starting Auth0 login process...")

        if auth_manager.login():
            user_info = auth_manager.get_current_user()
            user_email = user_info.get("email", "Unknown") if user_info else "Unknown"
            message("success", f"✓ Login successful! Welcome, {user_email}")

            # Store user in database
            if user_info and DATABASE_AVAILABLE:
                # Set database path for the operation
                output_dir = user_config.get_default_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)
                set_database_path(output_dir)

                # Initialize database with all tables (including new auth tables)
                manage_database_conditionally("create_collection")

                user_record = {
                    "auth0_user_id": user_info.get("sub"),
                    "email": user_info.get("email"),
                    "name": user_info.get("name"),
                    "user_metadata": user_info,
                }
                manage_database_conditionally("store_authenticated_user", user_record)

            return 0
        else:
            message("error", "Login failed. Please try again.")
            return 1

    except Exception as e:
        message("error", f"Login error: {e}")
        return 1


def cmd_logout(args) -> int:
    """Execute the logout command."""
    try:
        if not is_compliance_mode_enabled():
            message(
                "info", "Compliance mode is not enabled. No authentication to clear."
            )
            return 0

        auth_manager = get_auth0_manager()

        if not auth_manager.is_authenticated():
            message("info", "Not currently logged in.")
            return 0

        user_info = auth_manager.get_current_user()
        user_email = user_info.get("email", "Unknown") if user_info else "Unknown"

        auth_manager.logout()
        message("success", f"✓ Logged out successfully. Goodbye, {user_email}!")

        return 0

    except Exception as e:
        message("error", f"Logout error: {e}")
        return 1


def cmd_whoami(args) -> int:
    """Execute the whoami command."""
    try:
        if not is_compliance_mode_enabled():
            message("info", "Compliance mode: Disabled")
            message("info", "Authentication: Not required")
            return 0

        auth_manager = get_auth0_manager()

        if not auth_manager.is_configured():
            message("info", "Compliance mode: Enabled")
            message("info", "Authentication: Not configured")
            message(
                "info",
                "Run 'autocleaneeg-pipeline auth setup' to configure Auth0.",
            )
            return 0

        if not auth_manager.is_authenticated():
            message("info", "Compliance mode: Enabled")
            message("info", "Authentication: Not logged in")
            message("info", "Run 'autocleaneeg-pipeline login' to authenticate.")
            return 0

        user_info = auth_manager.get_current_user()
        if user_info:
            message("info", "Compliance mode: Enabled")
            message("info", "Authentication: Logged in")
            message("info", f"Email: {user_info.get('email', 'Unknown')}")
            message("info", f"Name: {user_info.get('name', 'Unknown')}")
            message("info", f"User ID: {user_info.get('sub', 'Unknown')}")

            # Check token expiration
            if (
                hasattr(auth_manager, "token_expires_at")
                and auth_manager.token_expires_at
            ):
                expires_str = auth_manager.token_expires_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                message("info", f"Token expires: {expires_str}")
        else:
            message("warning", "User information unavailable")

        return 0

    except Exception as e:
        message("error", f"Error checking authentication status: {e}")
        return 1


def cmd_auth0_diagnostics(args) -> int:
    """Execute the auth0-diagnostics command."""
    try:
        console = get_console(args)

        # Header
        console.print("\n🔍 [title]Auth0 Configuration Diagnostics[/title]")
        console.print("[muted]Checking Auth0 setup and connectivity...[/muted]\n")

        # 1. Check compliance mode
        compliance_enabled = is_compliance_mode_enabled()
        console.print(
            f"✓ Compliance mode: {'[success]Enabled[/success]' if compliance_enabled else '[warning]Disabled[/warning]'}"
        )

        if not compliance_enabled:
            console.print(
                "[info]ℹ Auth0 is only used in compliance mode. Run 'autocleaneeg-pipeline auth enable' to enable.[/info]"
            )
            return 0

        # 2. Check environment variables
        console.print("\n📋 [header]Environment Variables[/header]")
        env_table = Table(show_header=True, header_style="header")
        env_table.add_column("Variable", style="accent", no_wrap=True)
        env_table.add_column("Status", style="success")
        env_table.add_column("Value Preview", style="muted")

        env_vars = [
            ("AUTOCLEAN_AUTH0_DOMAIN", os.getenv("AUTOCLEAN_AUTH0_DOMAIN")),
            ("AUTOCLEAN_AUTH0_CLIENT_ID", os.getenv("AUTOCLEAN_AUTH0_CLIENT_ID")),
            (
                "AUTOCLEAN_AUTH0_CLIENT_SECRET",
                os.getenv("AUTOCLEAN_AUTH0_CLIENT_SECRET"),
            ),
            ("AUTOCLEAN_AUTH0_AUDIENCE", os.getenv("AUTOCLEAN_AUTH0_AUDIENCE")),
            ("AUTOCLEAN_DEVELOPMENT_MODE", os.getenv("AUTOCLEAN_DEVELOPMENT_MODE")),
        ]

        for var_name, var_value in env_vars:
            if var_value:
                if "SECRET" in var_name:
                    preview = f"{var_value[:8]}..." if len(var_value) > 8 else "***"
                elif len(var_value) > 30:
                    preview = f"{var_value[:30]}..."
                else:
                    preview = var_value
                env_table.add_row(var_name, "✓ Set", preview)
            else:
                env_table.add_row(var_name, "[error]✗ Not Set[/error]", "")

        console.print(env_table)

        # 3. Check .env file
        console.print("\n📄 [header].env File Detection[/header]")
        env_paths = [
            Path(".env"),
            Path(".env.local"),
            Path("../.env"),
            Path("../../.env"),
        ]
        env_found = False
        for env_path in env_paths:
            if env_path.exists():
                console.print(
                    f"✓ Found .env file: [accent]{env_path.absolute()}[/accent]"
                )
                env_found = True
                if args.verbose:
                    try:
                        with open(env_path, "r") as f:
                            content = f.read()
                            auth_lines = [
                                line
                                for line in content.split("\n")
                                if "AUTOCLEAN_AUTH0" in line
                                and not line.strip().startswith("#")
                            ]
                            if auth_lines:
                                console.print(
                                    "[muted]  Auth0 variables in file:[/muted]"
                                )
                                for line in auth_lines:
                                    # Mask secrets
                                    if "SECRET" in line and "=" in line:
                                        key, value = line.split("=", 1)
                                        masked_value = (
                                            f"{value[:8]}..."
                                            if len(value) > 8
                                            else "***"
                                        )
                                        console.print(
                                            f"[muted]    {key}={masked_value}[/muted]"
                                        )
                                    else:
                                        console.print(f"[muted]    {line}[/muted]")
                    except Exception as e:
                        console.print(f"[error]  Error reading file: {e}[/error]")
                break

        if not env_found:
            console.print(
                "[warning]⚠ No .env file found in current or parent directories[/warning]"
            )

        # 4. Test credential loading
        console.print("\n🔧 [header]Credential Loading Test[/header]")
        try:
            auth_manager = get_auth0_manager()
            credentials = auth_manager._load_developer_credentials()

            if credentials:
                console.print("✓ Credentials loaded successfully")
                console.print(
                    f"  Source: [accent]{credentials.get('source', 'unknown')}[/accent]"
                )
                console.print(
                    f"  Domain: [accent]{credentials.get('domain', 'NOT FOUND')}[/accent]"
                )
                client_id = credentials.get("client_id", "NOT FOUND")
                if client_id != "NOT FOUND":
                    console.print(f"  Client ID: [accent]{client_id[:8]}...[/accent]")
                else:
                    console.print(f"  Client ID: [error]{client_id}[/error]")
            else:
                console.print("[error]✗ Failed to load credentials[/error]")
                console.print(
                    "[warning]  Try setting environment variables or checking .env file[/warning]"
                )
        except Exception as e:
            console.print(f"[error]✗ Error loading credentials: {e}[/error]")

        # 5. Test Auth0 domain connectivity
        console.print("\n🌐 [bold]Domain Connectivity Test[/bold]")
        openid_accessible = False
        connectivity_error = None

        try:
            # Get fresh credentials to ensure we test the correct domain
            auth_manager = get_auth0_manager()
            credentials = auth_manager._load_developer_credentials()

            if credentials and credentials.get("domain"):
                domain = credentials["domain"]
                console.print(f"Testing connection to: [accent]{domain}[/accent]")

                # Test basic connectivity
                try:
                    response = requests.get(f"https://{domain}", timeout=10)
                    if response.status_code in [
                        200,
                        404,
                        403,
                    ]:  # Any of these indicates domain exists
                        console.print("✓ Domain is reachable")
                        if args.verbose:
                            console.print(f"  HTTP Status: {response.status_code}")
                            console.print(
                                f"  Response time: {response.elapsed.total_seconds():.2f}s"
                            )
                    else:
                        connectivity_error = (
                            f"Unexpected status code: {response.status_code}"
                        )
                        console.print(f"[warning]⚠ {connectivity_error}[/warning]")
                except requests.Timeout:
                    connectivity_error = "Connection timeout"
                    console.print(f"[error]✗ {connectivity_error}[/error]")
                except requests.ConnectionError:
                    connectivity_error = "Connection failed - check domain name"
                    console.print(f"[error]✗ {connectivity_error}[/error]")
                except Exception as e:
                    connectivity_error = f"Connection error: {e}"
                    console.print(f"[error]✗ {connectivity_error}[/error]")

                # Test Auth0 authorization endpoint (what login actually uses)
                try:
                    auth_url = f"https://{domain}/authorize"
                    response = requests.get(auth_url, timeout=10, allow_redirects=False)
                    # Auth0 authorize endpoint should return 400 (missing parameters) or redirect, not 404
                    if response.status_code in [400, 302, 301]:
                        openid_accessible = True
                        console.print("✓ Auth0 authorization endpoint accessible")
                        if args.verbose:
                            console.print(f"  Authorization URL: {auth_url}")
                            console.print(f"  Response status: {response.status_code}")
                    else:
                        console.print(
                            f"[warning]⚠ Auth0 authorization endpoint unexpected status: {response.status_code}[/warning]"
                        )

                    # Also test the well-known endpoint (optional)
                    well_known_url = (
                        f"https://{domain}/.well-known/openid_configuration"
                    )
                    response = requests.get(well_known_url, timeout=5)
                    if response.status_code == 200:
                        console.print("✓ OpenID configuration also accessible")
                        if args.verbose:
                            config = response.json()
                            console.print(
                                f"  Issuer: {config.get('issuer', 'unknown')}"
                            )
                    else:
                        console.print(
                            f"[muted]ℹ OpenID config not available (status: {response.status_code}) - this is optional[/muted]"
                        )

                except Exception as e:
                    console.print(
                        f"[warning]⚠ Could not test Auth0 endpoints: {e}[/warning]"
                    )
            else:
                connectivity_error = "No domain configured"
                console.print("[error]✗ No domain configured[/error]")
        except Exception as e:
            connectivity_error = f"Error testing connectivity: {e}"
            console.print(f"[error]✗ {connectivity_error}[/error]")

        # 6. Configuration summary
        console.print("\n📊 [header]Configuration Summary[/header]")
        try:
            auth_manager = get_auth0_manager()
            # Ensure configuration is loaded from environment/credentials
            credentials = auth_manager._load_developer_credentials()
            if credentials and not auth_manager.is_configured():
                auth_manager.configure_developer_auth0()

            summary_table = Table(show_header=True, header_style="header")
            summary_table.add_column("Component", style="accent")
            summary_table.add_column("Status", style="success")
            summary_table.add_column("Details", style="muted")

            # Check if configured
            is_configured = auth_manager.is_configured()
            summary_table.add_row(
                "Auth0 Configuration",
                "✓ Valid" if is_configured else "[error]✗ Invalid[/error]",
                "Ready for login" if is_configured else "Missing required credentials",
            )

            # Check authentication status
            is_authenticated = auth_manager.is_authenticated()
            summary_table.add_row(
                "Authentication",
                (
                    "✓ Logged in"
                    if is_authenticated
                    else "[warning]Not logged in[/warning]"
                ),
                (
                    "Valid session"
                    if is_authenticated
                    else "Run 'autocleaneeg-pipeline login'"
                ),
            )

            # Check config file
            config_file = auth_manager.config_file
            config_exists = config_file.exists()
            summary_table.add_row(
                "Config File",
                "✓ Exists" if config_exists else "[warning]Not found[/warning]",
                str(config_file) if config_exists else "Will be created on first setup",
            )

            console.print(summary_table)

        except Exception as e:
            console.print(f"[error]Error generating summary: {e}[/error]")

        # 7. Recommendations
        console.print("\n💡 [header]Recommendations[/header]")
        try:
            auth_manager = get_auth0_manager()
            credentials = auth_manager._load_developer_credentials()

            if not credentials:
                console.print("1. Set Auth0 environment variables:")
                console.print(
                    '   [accent]export AUTOCLEAN_AUTH0_DOMAIN="your-tenant.us.auth0.com"[/accent]'
                )
                console.print(
                    '   [accent]export AUTOCLEAN_AUTH0_CLIENT_ID="your_client_id"[/accent]'
                )
                console.print(
                    '   [accent]export AUTOCLEAN_AUTH0_CLIENT_SECRET="your_client_secret"[/accent]'
                )
                console.print("2. Or create a .env file with these variables")
                console.print(
                    "3. Install python-dotenv if using .env: [accent]pip install python-dotenv[/accent]"
                )
            elif connectivity_error:
                # Don't recommend login if there are connectivity issues
                console.print(
                    f"[error]⚠ Auth0 connectivity issue detected:[/error] {connectivity_error}"
                )
                console.print("")
                console.print("Please check:")
                console.print("1. Verify your Auth0 domain is correct in .env file")
                console.print("2. Ensure your Auth0 application is properly configured")
                console.print(
                    "3. Check that your Auth0 application type is 'Native' (for CLI apps)"
                )
                console.print("4. Verify your Auth0 tenant is active and accessible")
                console.print("")
                console.print(
                    "Once connectivity is fixed, run [accent]autocleaneeg-pipeline login[/accent] to authenticate"
                )
            elif not openid_accessible:
                console.print(
                    "[warning]⚠ Auth0 OpenID configuration not accessible[/warning]"
                )
                console.print("")
                console.print("This may indicate:")
                console.print("1. Auth0 domain is incorrect")
                console.print("2. Auth0 application is not properly configured")
                console.print("3. Network or firewall issues")
                console.print("")
                console.print(
                    "You can try [accent]autocleaneeg-pipeline login[/accent] but it may fail"
                )
            elif not auth_manager.is_authenticated():
                console.print("✓ Configuration looks good!")
                console.print(
                    "1. Run [accent]autocleaneeg-pipeline login[/accent] to authenticate"
                )
            else:
                console.print(
                    "✓ Configuration looks good! You're ready to use Auth0 authentication."
                )

        except Exception as e:
            console.print(f"[error]Error generating recommendations: {e}[/error]")

        return 0

    except Exception as e:
        message("error", f"Diagnostics error: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the AutoClean CLI."""
    parser = create_parser()

    # --------------------------------------------------------------
    # Intercept unknown top-level commands to show a rich error page
    # with banner, workspace/task/input context, and available commands.
    # --------------------------------------------------------------
    try:
        raw_argv = list(argv) if isinstance(argv, list) else sys.argv[1:]

        # If help flags are present, defer to argparse's help handling
        if any(t in ("-h", "--help") for t in raw_argv):
            raise Exception("defer-parse")

        # Extract first non-option token, accounting for options that expect values
        first_non_option: Optional[str] = None
        i = 0
        while i < len(raw_argv):
            tok = raw_argv[i]
            if tok == "--":  # end of options
                if i + 1 < len(raw_argv):
                    first_non_option = raw_argv[i + 1]
                break
            if tok.startswith("-"):
                # Skip option value for known global options
                if tok == "--theme" and i + 1 < len(raw_argv):
                    i += 2
                    continue
                # Other flags have no values at root level currently
                i += 1
                continue
            first_non_option = tok
            break

        if first_non_option:
            # Discover known top-level commands from the parser
            sub_actions = [
                a
                for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)  # type: ignore[attr-defined]
            ]
            if sub_actions:
                top_level = sub_actions[0].choices.keys()
                if first_non_option not in top_level:
                    # Build suggestions from second-level subparsers (e.g., task edit)
                    suggestions: list[str] = []
                    for parent_name, child_parser in sub_actions[0].choices.items():
                        try:
                            child_subs = [
                                a
                                for a in child_parser._actions
                                if isinstance(a, argparse._SubParsersAction)  # type: ignore[attr-defined]
                            ]
                            if child_subs:
                                child_choices = child_subs[0].choices.keys()
                                if first_non_option in child_choices:
                                    suggestions.append(
                                        f"{parent_name} {first_non_option}"
                                    )
                        except Exception:
                            pass

                    # Render elegant unknown command screen
                    console = get_console(None)
                    _simple_header(console)
                    _print_startup_context(console)
                    try:
                        from rich.table import Table as _Table
                        from rich.text import Text

                        err = Text()
                        err.append("Unknown command: ", style="error")
                        err.append(first_non_option, style="accent")
                        console.print(err)

                        if suggestions:
                            console.print()
                            console.print("Did you mean:")
                            tbl = _Table(show_header=False, box=None, padding=(0, 1))
                            tbl.add_column("Suggestion", style="accent")
                            for s in suggestions[:3]:
                                tbl.add_row(s)
                            console.print(tbl)

                        console.print()
                        _print_root_help(console)
                    except Exception:
                        # Minimal fallback without rich constructs
                        print(f"Unknown command: {first_non_option}")
                        print(
                            "Try one of: process, task, input, view, review, workspace, settings, help"
                        )
                    return 2
    except Exception:
        # Defer to argparse for normal parsing and help behavior
        pass

    # Parse arguments - custom error handling is done via TaskSubcommandParser
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Log CLI execution for process tracking
    # ------------------------------------------------------------------
    _log_cli_execution(args)

    # ------------------------------------------------------------------
    # Always inform the user where the AutoClean workspace is (or will be)
    # so they can easily locate their configuration and results.  This runs
    # for *every* CLI invocation, including the bare `autocleaneeg-pipeline` call.
    # ------------------------------------------------------------------
    def _render_task_reminder(console_obj, text: str) -> None:
        try:
            from rich.align import Align as _WarnAlign
            from rich.text import Text as _WarnText

            warn_line = _WarnText()
            warn_line.append("⚠ ", style="warning")
            warn_line.append(text, style="warning")
            console_obj.print(_WarnAlign.center(warn_line))
        except Exception:
            console_obj.print(f"[warning]{text}[/warning]")

    workspace_dir = user_config.config_dir
    tasks_dir = workspace_dir / "tasks"

    skip_task_reminder = (
        args.command == "workspace" and getattr(args, "workspace_action", None) == "set"
    )
    reminder_text: Optional[str] = None
    if not skip_task_reminder:
        try:
            has_task_files = tasks_dir.exists() and any(
                entry.is_file() and entry.suffix.lower() == ".py"
                for entry in tasks_dir.iterdir()
            )
            if not has_task_files:
                reminder_text = (
                    f"Workspace tasks folder has no task files at {tasks_dir}. "
                    "Run 'autocleaneeg-pipeline workspace set' to initialize default tasks."
                )
        except OSError:
            reminder_text = (
                f"Workspace tasks folder could not be inspected at {tasks_dir}. "
                "Run 'autocleaneeg-pipeline workspace set' to initialize default tasks."
            )

    post_command_reminder = reminder_text if args.command else None

    # For real sub-commands, log the workspace path via the existing logger.
    if args.command and args.command != "workspace":
        # Compact branding header for consistency across all commands (except workspace which has its own branding)
        console = get_console(args)
        _print_context(
            console, style="compact", include_system_info=False, include_disk=False
        )

    if not args.command:
        # Show our custom 80s-style main interface instead of default help
        console = get_console(args)
        _simple_header(console)

        # Display centered context info (system, workspace, task, montage, input, disk)
        _print_context(
            console, style="centered", include_system_info=True, include_disk=True
        )

        # Minimal centered key commands belt (for quick discovery)
        try:
            from rich.align import Align as _KAlign
            from rich.text import Text as _KText

            key_cmds = [
                "help",
                "workspace",
                "view",
                "montage",
                "task",
                "input",
                "process",
                "review",
            ]
            belt = _KText()
            for i, cmd in enumerate(key_cmds):
                if i > 0:
                    belt.append("  •  ", style="muted")
                belt.append(cmd, style="accent")

            console.print()
            console.print(_KAlign.center(belt))
            console.print()
        except Exception:
            pass

        # Centered docs and GitHub links (minimalist, wrapped to avoid wide lines)
        try:
            from rich.align import Align as _LAlign
            from rich.text import Text as _LText

            # Docs line
            docs_line = _LText()
            docs_line.append("📘 Quick Start & Docs ", style="muted")
            docs_line.append("https://docs.autocleaneeg.org", style="accent")
            console.print(_LAlign.center(docs_line))

            # GitHub link line
            gh_line = _LText()
            gh_line.append("GitHub ", style="muted")
            gh_line.append(
                "https://github.com/cincibrainlab/autoclean_pipeline", style="accent"
            )
            console.print(_LAlign.center(gh_line))

            # GitHub meta line (short descriptors)
            gh_meta = _LText()
            gh_meta.append("code", style="muted")
            gh_meta.append("  •  ", style="muted")
            gh_meta.append("issues", style="muted")
            gh_meta.append("  •  ", style="muted")
            gh_meta.append("discussions", style="muted")
            console.print(_LAlign.center(gh_meta))
            console.print()
        except Exception:
            pass

        # Centered attribution
        try:
            from rich.align import Align as _AAlign
            from rich.text import Text as _AText

            lab = _AText()
            lab.append(
                "Pedapati Lab @ Cincinnati Children's Hospital Medical Center",
                style="muted",
            )
            console.print(_AAlign.center(lab))
            console.print()
        except Exception:
            pass

        if reminder_text:
            _render_task_reminder(console, reminder_text)

        # (Quick Start section intentionally removed for a cleaner minimalist banner)

        return 0

    # Validate arguments
    if not validate_args(args):
        return 1

    def _finish(result: int) -> int:
        if post_command_reminder:
            console = get_console(args)
            _render_task_reminder(console, post_command_reminder)
        return result

    # Execute command
    if args.command == "wizard":
        return _finish(cmd_wizard(args))
    try:
        if args.command == "process":
            if getattr(args, "process_action", None) == "ica":
                return _finish(cmd_process_ica(args))
            return _finish(cmd_process(args))
        if args.command == "list-tasks":
            return _finish(cmd_list_tasks(args))
        if args.command == "review":
            return _finish(cmd_review(args))
        if args.command == "exclude":
            return _finish(cmd_exclude(args))
        if args.command == "task":
            return _finish(cmd_task(args))
        if args.command == "montage":
            return _finish(cmd_montage(args))
        if args.command == "blocks":
            return _finish(cmd_blocks(args))
        if args.command == "input":
            return _finish(cmd_input(args))
        if args.command == "source":
            return _finish(cmd_source(args))
        if args.command == "config":
            return _finish(cmd_config(args))
        if args.command == "settings":
            return _finish(cmd_settings(args))
        if args.command == "workspace":
            return _finish(cmd_workspace(args))
        if args.command == "export-access-log":
            return _finish(cmd_export_access_log(args))
        if args.command == "login":
            return _finish(cmd_login(args))
        if args.command == "logout":
            return _finish(cmd_logout(args))
        if args.command == "whoami":
            return _finish(cmd_whoami(args))
        if args.command == "auth0-diagnostics":
            return _finish(cmd_auth0_diagnostics(args))
        if args.command == "auth":
            return _finish(cmd_auth(args))
        if args.command == "clean-task":
            return _finish(cmd_clean_task(args))
        if args.command == "view":
            return _finish(cmd_view(args))
        if args.command == "report":
            return _finish(cmd_report(args))
        if args.command == "version":
            return _finish(cmd_version(args))
        if args.command == "help":
            return _finish(cmd_help(args))
        if args.command == "tutorial":
            return _finish(cmd_tutorial(args))

        parser.print_help()
        return _finish(1)
    except KeyboardInterrupt:
        message("warning", "Interrupted by user (Ctrl+C). Exiting gracefully.")
        return _finish(130)


if __name__ == "__main__":
    sys.exit(main())
