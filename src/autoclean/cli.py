#!/usr/bin/env python3
"""
AutoClean EEG Pipeline - Command Line Interface

This module provides a flexible CLI for AutoClean that works both as a
standalone tool (via uv tool) and within development environments.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# import requests - moved to function level (0.347s import overhead)
from rich.panel import Panel
from rich.table import Table

from autoclean import __version__
from autoclean.utils.audit import verify_access_log_integrity
from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled
from autoclean.utils.config import (
    disable_compliance_mode,
    enable_compliance_mode,
    get_compliance_status,
    load_user_config,
    save_user_config,
)
from autoclean.utils.console import get_console
from autoclean.utils.logging import message

# Import lazy task discovery functions for better performance
from autoclean.utils.lazy_task_discovery import (
    extract_config_from_task_lazy as extract_config_from_task,
    get_task_by_name_lazy as get_task_by_name,
    safe_discover_tasks_lazy as safe_discover_tasks,
)
from autoclean.utils.task_discovery import (
    get_task_overrides,
)

# Heavy imports moved to function-level for better startup performance
# This avoids loading heavy modules until actually needed
# from autoclean.utils.user_config import user_config - moved to conditional import in main()

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
        from autoclean.utils.user_config import user_config
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
def _print_startup_context(console, show_system_info: bool = True) -> None:
    """Print system info, workspace path, and free disk space (shared for header/help).
    
    Args:
        console: Rich console instance
        show_system_info: If False, skips expensive system info gathering for faster startup
    """
    if show_system_info:
        try:
            import platform as _platform

            from rich.align import Align
            from rich.text import Text

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

    # Workspace + disk
    try:
        from rich.align import Align as _Align
        from rich.text import Text as _Text

        workspace_dir = user_config.config_dir
        try:
            # Prefer strict validity check (requires saved setup + structure)
            valid_ws = user_config._is_workspace_valid()  # type: ignore[attr-defined]
        except Exception:
            # Fallback to basic existence check
            valid_ws = workspace_dir.exists() and (workspace_dir / "tasks").exists()
        home = str(Path.home())
        display_path = str(workspace_dir)
        if display_path.startswith(home):
            display_path = display_path.replace(home, "~", 1)

        ws = _Text()
        if valid_ws:
            ws.append("✓ ", style="success")
            ws.append("Workspace ", style="muted")
            ws.append(display_path, style="accent")
            console.print(_Align.center(ws))
        else:
            ws.append("⚠ ", style="warning")
            ws.append("Workspace not configured — ", style="muted")
            ws.append(display_path, style="accent")
            console.print(_Align.center(ws))
            tip = _Text()
            tip.append("Run ", style="muted")
            tip.append("autocleaneeg-pipeline workspace", style="accent")
            tip.append(" to configure.", style="muted")
            console.print(_Align.center(tip))

        # Always show active task line beneath Workspace (or guard if not set)
        try:
            active_task = user_config.get_active_task()
            at = _Text()
            at.append("🎯 ", style="muted")
            at.append("Active task: ", style="muted")
            if active_task:
                at.append(str(active_task), style="accent")
            else:
                at.append("not set", style="warning")
            console.print(_Align.center(at))
        except Exception:
            pass

        # Show active input (or guard if not set/missing)
        try:
            active_source = user_config.get_active_source()
            src = _Text()
            if active_source:
                sp = Path(active_source)
                display_src = str(sp)
                home = str(Path.home())
                if display_src.startswith(home):
                    display_src = display_src.replace(home, "~", 1)
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
                    src.append(display_src, style="accent")
                else:
                    src.append("⚠ ", style="warning")
                    src.append("Input missing — ", style="muted")
                    src.append(display_src, style="accent")
            else:
                src.append("📁 ", style="muted")
                src.append("Active input: ", style="muted")
                src.append("not set", style="warning")
            console.print(_Align.center(src))
        except Exception:
            pass

        # Disk free (only if system info is requested)
        if show_system_info:
            usage_path = (
                workspace_dir
                if workspace_dir.exists()
                else (
                    workspace_dir.parent if workspace_dir.parent.exists() else Path.home()
                )
            )
            du = shutil.disk_usage(str(usage_path))
            free_gb = du.free / (1024**3)
            free_line = _Text()
            free_line.append("💾 ", style="muted")
            free_line.append("Free space ", style="muted")
            free_line.append(f"{free_gb:.1f} GB", style="accent")
            console.print(_Align.center(free_line))
        
        console.print()
    except Exception:
        pass


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

    console.print("[header]Commands[/header]")
    tbl = _Table(show_header=False, box=None, padding=(0, 1))
    tbl.add_column("Command", style="accent", no_wrap=True)
    tbl.add_column("Description", style="muted")

    rows = [
        ("❓ help", "Show help and topics (alias for -h/--help)"),
        ("🗂\u00a0 workspace", "Configure workspace folder"),
        ("👁\u00a0 view", "View EEG file (MNE-QT)"),
        ("🗂\u00a0 task", "Manage tasks (list, explore)"),
        ("📁\u00a0 input", "Manage active input path"),
        ("▶\u00a0 process", "Process EEG data"),
        ("📝 review", "Start review GUI"),
        ("🔐 auth", "Authentication & Part-11 commands"),
    ]
    for c, d in rows:
        tbl.add_row(c, d)
    console.print(tbl)
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

Active Task (Simplified Workflow):
  autocleaneeg-pipeline task set                       # Select active task interactively
  autocleaneeg-pipeline task show                      # Show current active task
  autocleaneeg-pipeline process data.raw               # Process file with active task

Custom Tasks:
  autocleaneeg-pipeline task add my_task.py            # Add custom task file
  autocleaneeg-pipeline task list                      # List all tasks


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
    process_parser = subparsers.add_parser(
        "process", help="Process EEG data", add_help=False
    )
    attach_rich_help(process_parser)

    # Positional arguments for simple usage: autocleaneeg-pipeline process TaskName FilePath
    process_parser.add_argument(
        "task_name", nargs="?", type=str, help="Task name (e.g., RestingEyesOpen)"
    )
    process_parser.add_argument(
        "input_path", nargs="?", type=Path, help="EEG file or directory to process"
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

    # Task management commands
    task_parser = subparsers.add_parser(
        "task", help="Manage custom tasks", add_help=False
    )
    attach_rich_help(task_parser)
    task_subparsers = task_parser.add_subparsers(
        dest="task_action", help="Task actions"
    )

    # Add task
    add_task_parser = task_subparsers.add_parser(
        "add", help="Add a custom task", add_help=False
    )
    attach_rich_help(add_task_parser)
    add_task_parser.add_argument("task_file", type=Path, help="Python task file to add")
    add_task_parser.add_argument(
        "--name", type=str, help="Custom name for the task (default: filename)"
    )
    add_task_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing task with same name"
    )

    # Remove task
    remove_task_parser = task_subparsers.add_parser(
        "remove", help="Remove a custom task", add_help=False
    )
    attach_rich_help(remove_task_parser)
    remove_task_parser.add_argument(
        "task_name", type=str, help="Name of the task to remove"
    )

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
    edit_parser.add_argument(
        "--force",
        action="store_true",
        help="When copying a built-in task, overwrite existing without prompting",
    )

    # Import task (copy file into workspace tasks folder)
    import_parser = task_subparsers.add_parser(
        "import", help="Import a task file into your workspace", add_help=False
    )
    attach_rich_help(import_parser)
    import_parser.add_argument(
        "source",
        type=Path,
        help="Path to a Python task file to import (.py)",
    )
    import_parser.add_argument(
        "--name",
        type=str,
        help="Save as this filename (without .py)",
    )
    import_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file without prompting",
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

    return parser


def _show_process_guard(args) -> bool:
    """Show interactive guard with key information before processing.

    Returns True if user confirms to proceed, False to cancel.
    """
    console = get_console(args)

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
            from autoclean.utils.lazy_task_discovery import extract_config_from_task_lazy
            montage_info = extract_config_from_task_lazy(task_name, "montage")
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

        return Confirm.ask("🚀 [bold]Proceed with processing?[/bold]", default=False)
    except ImportError:
        # Fallback for systems without rich Confirm
        try:
            response = input("🚀 Proceed with processing? (y/N): ").lower().strip()
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
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
            active_task = user_config.get_active_task()
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

            # If still no task, show help
        if not task_name and not args.task_file:
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
                    "Usage: autocleaneeg-pipeline process [TaskName|--task-file FILE] <file|--dir DIR> [options]"
                )
                console.print(
                    "Note: Task is optional if you have set an active task with 'task set'"
                )
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
                from autoclean.utils.lazy_task_discovery import extract_config_from_task_lazy
                task_input_path = extract_config_from_task_lazy(task_name, "input_path")

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
            from autoclean.utils.lazy_task_discovery import get_task_by_name_lazy
            task_class = get_task_by_name_lazy(task_name)
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
        message("info", "Processing completed successfully!")
        return 0

    except Exception as e:
        message("error", f"Processing failed: {str(e)}")
        return 1


def cmd_process_ica(args) -> int:
    """Process ICA control sheet updates."""
    try:
        from autoclean.tools.ica import process_ica_control_sheet

        metadata_dir = (
            Path(args.metadata_dir)
            if args.metadata_dir
            else user_config.config_dir / "metadata"
        )
        control_sheet = metadata_dir / "ica_control_sheet.csv"
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
                    ica_fif = Path(row.get("ica_fif") or row.get("ica_file") or "")
                    derivatives_dir = Path(row.get("derivatives_dir") or "")
                    unprocessed_file = Path(row.get("unprocessed_file") or "")
                    autoclean_dict = {
                        "metadata_dir": metadata_dir,
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
    """Execute the list-tasks command."""
    try:
        console = get_console(args)

        # If --overrides flag is specified, show override information
        if hasattr(args, "overrides") and args.overrides:
            from autoclean.utils.task_discovery import get_task_overrides
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

        # Minimal header
        from rich.align import Align as _Align
        from rich.text import Text as _Text

        console.print()
        head = _Text()
        head.append("Tasks", style="title")
        console.print(_Align.center(head))
        console.print(
            _Align.center(_Text("Available processing tasks", style="subtitle"))
        )
        console.print()

        from autoclean.utils.lazy_task_discovery import safe_discover_tasks_lazy
        valid_tasks, invalid_files, skipped_files = safe_discover_tasks_lazy()

        # --- Built-in Tasks ---
        built_in_tasks = [
            task for task in valid_tasks if "autoclean/tasks" in task.source
        ]
        if built_in_tasks:
            built_in_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            built_in_table.add_column("Task Name", style="accent", no_wrap=True)
            built_in_table.add_column("Module", style="muted")
            built_in_table.add_column("Description", style="muted", max_width=50)

            for task in sorted(built_in_tasks, key=lambda x: x.name):
                # Extract just the module name from the full path
                module_name = Path(task.source).stem
                built_in_table.add_row(
                    task.name, module_name + ".py", task.description or "No description"
                )

            built_in_panel = Panel(
                built_in_table,
                title="[title]Built-in Tasks[/title]",
                border_style="border",
                padding=(1, 1),
            )
            console.print(built_in_panel)
        else:
            console.print(
                Panel(
                    "[muted]No built-in tasks found[/muted]",
                    title="[title]Built-in Tasks[/title]",
                    border_style="border",
                    padding=(1, 1),
                )
            )

        # --- Custom Tasks ---
        custom_tasks = [
            task for task in valid_tasks if "autoclean/tasks" not in task.source
        ]
        if custom_tasks:
            custom_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            custom_table.add_column("Task Name", style="accent", no_wrap=True)
            custom_table.add_column("File", style="muted")
            custom_table.add_column("Description", style="muted", max_width=50)

            for task in sorted(custom_tasks, key=lambda x: x.name):
                # Show just the filename for custom tasks
                file_name = Path(task.source).name
                custom_table.add_row(
                    task.name, file_name, task.description or "No description"
                )

            custom_panel = Panel(
                custom_table,
                title="[title]Custom Tasks[/title]",
                border_style="border",
                padding=(1, 1),
            )
            console.print(custom_panel)
        else:
            console.print(
                Panel(
                    "[muted]No custom tasks found.\n"
                    "Use [accent]autocleaneeg-pipeline task add <file.py>[/accent] to add one.[/muted]",
                    title="[title]Custom Tasks[/title]",
                    border_style="border",
                    padding=(1, 1),
                )
            )

        # --- Skipped Task Files ---
        if skipped_files:
            skipped_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            skipped_table.add_column("File", style="warning", no_wrap=True)
            skipped_table.add_column("Reason", style="muted", max_width=70)

            for file in skipped_files:
                # Show just the filename for skipped files
                file_name = Path(file.source).name
                skipped_table.add_row(file_name, file.reason)

            skipped_panel = Panel(
                skipped_table,
                title="[title]Skipped Task Files[/title]",
                border_style="border",
                padding=(1, 1),
            )
            console.print(skipped_panel)

        # --- Invalid Task Files ---
        if invalid_files:
            invalid_table = Table(
                show_header=True, header_style="header", box=None, padding=(0, 1)
            )
            invalid_table.add_column("File", style="error", no_wrap=True)
            invalid_table.add_column("Error", style="warning", max_width=70)

            for file in invalid_files:
                # Show relative path if in workspace, otherwise just filename
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

        # Summary line (centered, minimal)
        summary = _Text()
        summary.append("Found ", style="muted")
        summary.append(str(len(valid_tasks)), style="accent")
        summary.append(" tasks ", style="muted")
        summary.append(
            f"({len(built_in_tasks)} built-in, {len(custom_tasks)} custom)",
            style="muted",
        )
        if skipped_files or invalid_files:
            summary.append("  •  ", style="muted")
            summary.append(
                f"{len(skipped_files)} skipped, {len(invalid_files)} invalid",
                style="muted",
            )
        console.print(_Align.center(summary))
        console.print()

        return 0

    except Exception as e:
        message("error", f"Failed to list tasks: {str(e)}")
        return 1


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
    try:
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
    except Exception:
        return ["/bin/sh"]


def _detect_shell_kind() -> str:
    """Best-effort detection of user's shell kind for printing snippets."""
    try:
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
    except Exception:
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
            except Exception as e:
                message("error", f"Failed to spawn shell: {e}")
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
    except Exception as e:
        message("error", f"Failed to resolve workspace directory: {e}")
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
        user_config_data["compliance"][
            "require_electronic_signatures"
        ] = signature_answer["require_signatures"]
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
    if args.task_action == "add":
        return cmd_task_add(args)
    elif args.task_action == "remove":
        return cmd_task_remove(args)
    elif args.task_action == "list":
        return cmd_list_tasks(args)
    elif args.task_action == "explore":
        return cmd_task_explore(args)
    elif args.task_action == "edit":
        return cmd_task_edit(args)
    elif args.task_action == "import":
        return cmd_task_import(args)
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
    else:
        message("error", "No task action specified")
        return 1


def cmd_task_add(args) -> int:
    """Add a custom task by copying to workspace tasks folder."""
    try:
        if not args.task_file.exists():
            message("error", f"Task file not found: {args.task_file}")
            return 1

        # Ensure workspace exists
        if not user_config.tasks_dir.exists():
            user_config.tasks_dir.mkdir(parents=True, exist_ok=True)

        # Determine destination name
        if args.name:
            dest_name = f"{args.name}.py"
        else:
            dest_name = args.task_file.name

        dest_file = user_config.tasks_dir / dest_name

        # Check if task already exists
        if dest_file.exists() and not args.force:
            message(
                "error", f"Task '{dest_name}' already exists. Use --force to overwrite."
            )
            return 1

        # Copy the task file
        shutil.copy2(args.task_file, dest_file)

        # Extract class name for usage message
        try:
            class_name, _ = user_config._extract_task_info(dest_file)
            task_name = class_name
        except Exception:
            task_name = dest_file.stem

        message("info", f"Task '{task_name}' added to workspace!")
        print(f"📁 Copied to: {dest_file}")
        print("\nUse your custom task with:")
        print(f"  autocleaneeg-pipeline process {task_name} <data_file>")

        return 0

    except Exception as e:
        message("error", f"Failed to add custom task: {str(e)}")
        return 1


def cmd_task_remove(args) -> int:
    """Remove a custom task by deleting from workspace tasks folder."""
    try:
        # Find task file by class name or filename
        custom_tasks = user_config.list_custom_tasks()

        task_file = None
        if args.task_name in custom_tasks:
            # Found by class name
            task_file = Path(custom_tasks[args.task_name]["file_path"])
        else:
            # Try by filename
            potential_file = user_config.tasks_dir / f"{args.task_name}.py"
            if potential_file.exists():
                task_file = potential_file

        if not task_file or not task_file.exists():
            message("error", f"Task '{args.task_name}' not found")
            return 1

        # Remove the file
        task_file.unlink()
        message("info", f"Task '{args.task_name}' removed from workspace!")
        return 0

    except Exception as e:
        message("error", f"Failed to remove custom task: {str(e)}")
        return 1


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

                from autoclean.utils.lazy_task_discovery import safe_discover_tasks_lazy

                tasks, _, _ = safe_discover_tasks_lazy()
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


def cmd_task_import(args) -> int:
    """Import a task Python file into the workspace tasks folder.

    Guards and guidance:
    - Verifies source exists and is a .py file
    - Warns on private names (leading underscore) and common test/fixture names
    - Prompts before overwriting unless --force
    - Lets user rename via --name or interactive prompt
    - After copy, shows how to use the task
    """
    try:
        src: Path = args.source.expanduser().resolve()
        if not src.exists() or not src.is_file():
            message("error", f"Source file not found: {src}")
            message("info", "Provide a valid Python file path (e.g., /path/to/task.py)")
            return 1
        if src.suffix.lower() != ".py":
            message("error", "Only Python files (.py) can be imported as tasks")
            return 1

        # Heuristic warnings
        if src.name.startswith("_"):
            message(
                "warning",
                "File starts with '_' and may be ignored by loader. Consider renaming.",
            )
        lower = src.name.lower()
        if any(x in lower for x in ("test", "fixture", "template")):
            message(
                "warning",
                "File name looks like a test/fixture/template. Ensure it contains a Task subclass.",
            )

        # Destination
        dest_dir = user_config.tasks_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = args.name or src.stem
        dest_name = dest_name if dest_name.endswith(".py") else f"{dest_name}.py"
        dest = dest_dir / dest_name

        # Confirm overwrite if exists
        if dest.exists() and not args.force:
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
                    "Interactive prompt unavailable; re-run with --force or --name to control destination.",
                )
                return 1

        # Perform copy
        try:
            shutil.copy2(src, dest)
            message("success", f"✓ Imported task to workspace: {dest.name}")
        except Exception as e:
            message("error", f"Failed to copy file: {e}")
            return 1

        # Extract class name and show usage
        try:
            class_name, _ = user_config._extract_task_info(dest)
            message("info", f"Detected task class: {class_name}")
            print("\nUse your task with:")
            print(f"  autocleaneeg-pipeline process {class_name} <data_file>")
        except Exception:
            message(
                "warning",
                "Could not detect Task class. Ensure the file defines a class extending Task.",
            )

        return 0
    except Exception as e:
        message("error", f"Failed to import task: {e}")
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
                from autoclean.utils.lazy_task_discovery import safe_discover_tasks_lazy

                tasks, _, _ = safe_discover_tasks_lazy()
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
                from autoclean.utils.lazy_task_discovery import safe_discover_tasks_lazy

                tasks, _, _ = safe_discover_tasks_lazy()
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
        active_task = user_config.get_active_task()

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

        return 0

    except Exception as e:
        message("error", f"Failed to show active task: {e}")
        return 1


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
                f"Run {latest.get('run_id','')} on {latest.get('created_at','')} (task: {latest.get('task','')})?",
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
        if not metadata_dir.exists():
            return None

        # Prefer previously generated LLM context
        try:
            from autoclean.utils.path_resolution import resolve_moved_path

            metadata_dir_resolved = resolve_moved_path(metadata_dir)
        except Exception:
            metadata_dir_resolved = metadata_dir

        llm_ctx = metadata_dir_resolved / "llm_reports" / "context.json"
        if llm_ctx.exists():
            try:
                return llm_ctx.read_text(encoding="utf-8")
            except Exception:
                pass

        # Reconstruct context from per-file processing log + PDF
        try:
            import ast
            import csv

            input_file = rec.get("unprocessed_file") or ""
            basename = Path(input_file).stem
            derivatives_root = metadata_dir_resolved.parent
            per_file_csv = derivatives_root / f"{basename}_processing_log.csv"
            if not per_file_csv.exists() and bids_dir:
                try:
                    bids_dir_resolved = resolve_moved_path(bids_dir)
                except Exception:
                    bids_dir_resolved = Path(bids_dir)
                alt = (
                    Path(bids_dir_resolved)
                    / "final_files"
                    / f"{basename}_processing_log.csv"
                )
                if alt.exists():
                    per_file_csv = alt
            if not per_file_csv.exists():
                return None

            row = None
            with per_file_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
            if not row:
                return None

            pdf_path = metadata_dir_resolved / (
                rec.get("report_file") or f"{basename}_autoclean_report.pdf"
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
                "figures": {"autoclean_report_pdf": str(pdf_path)}
                if pdf_path.exists()
                else {},
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
    """View EEG files using autoclean-view."""
    # Check if file exists
    if not args.file.exists():
        message("error", f"File not found: {args.file}")
        return 1

    # Build command
    cmd = ["autoclean-view", str(args.file)]
    if args.no_view:
        cmd.append("--no-view")

    # Launch viewer
    message("info", f"Opening {args.file.name} in MNE-QT Browser...")

    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            message(
                "success", "Viewer closed" if not args.no_view else "File validated"
            )
            return 0
        else:
            message("error", f"Error: {process.stderr}")
            return 1
    except FileNotFoundError:
        message(
            "error", "autoclean-view not installed. Run: pip install autoclean-view"
        )
        return 1
    except Exception as e:
        message("error", f"Failed to launch viewer: {str(e)}")
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
                            "Try one of: process, task, input, view, review, workspace, help"
                        )
                    return 2
    except Exception:
        # Defer to argparse for normal parsing and help behavior
        pass

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Log CLI execution for process tracking (skip for lightweight commands)
    # ------------------------------------------------------------------
    # Only log commands that actually need workspace tracking
    if not hasattr(args, 'command') or args.command not in {"help", "version"}:
        _log_cli_execution(args)

    # ------------------------------------------------------------------
    # Only import user_config for commands that actually need workspace info
    # This avoids heavy imports for simple commands like help, version, etc.
    # ------------------------------------------------------------------
    
    # Commands categorized by their import requirements
    
    # Lightweight commands: No workspace, minimal imports
    lightweight_commands = {
        "help", "version", "auth", "login", "logout", "whoami", "auth0-diagnostics"
    }
    
    # Workspace commands: Need config but no task discovery  
    workspace_commands = {
        "workspace", "config", "input", "source"
    }
    
    # Processing commands: Need full imports and task discovery
    processing_commands = {
        "process", "task", "list-tasks", "review", "view", "clean-task"
    }
    
    if args.command in lightweight_commands:
        # Skip workspace initialization for lightweight commands
        workspace_dir = None
        user_config = None
    elif args.command in workspace_commands or args.command in processing_commands:
        # Import user_config for workspace and processing commands
        from autoclean.utils.user_config import user_config
        workspace_dir = user_config.config_dir
    else:
        # Default case: import user_config for unknown commands
        from autoclean.utils.user_config import user_config
        workspace_dir = user_config.config_dir

    # For real sub-commands, log the workspace path via the existing logger.
    if args.command and args.command != "workspace" and workspace_dir is not None:
        # Compact branding header for consistency across all commands (except workspace which has its own branding)
        console = get_console(args)

        if workspace_dir.exists() and (workspace_dir / "tasks").exists():
            console.print(
                f"[success]Autoclean Workspace Directory:[/success] {workspace_dir}"
            )
        else:
            message(
                "warning",
                f"Workspace directory not configured yet: {workspace_dir} (run 'autocleaneeg-pipeline workspace' to configure)",
            )

        # Always show active task status for real sub-commands
        if user_config is not None:
            try:
                current = user_config.get_active_task()
                if current:
                    console.print(f"[info]Active task:[/info] [accent]{current}[/accent]")
                else:
                    console.print(
                        "[warning]Active task not set[/warning] [muted](run 'autocleaneeg-pipeline task set')[/muted]"
                    )
            except Exception:
                pass

            # Show active input (file vs folder) for real sub-commands
            try:
                active_src = user_config.get_active_source()
                if active_src:
                    p = Path(active_src)
                    if p.exists():
                        if p.is_file():
                            console.print(
                                f"[info]Input file:[/info] [accent]{active_src}[/accent]"
                            )
                        elif p.is_dir():
                            console.print(
                                f"[info]Input folder:[/info] [accent]{active_src}[/accent]"
                            )
                        else:
                            console.print(
                                f"[info]Input:[/info] [accent]{active_src}[/accent]"
                            )
                    else:
                        console.print(
                            f"[warning]Input missing[/warning] [muted]—[/muted] [accent]{active_src}[/accent]"
                        )
                else:
                    console.print(
                        "[warning]Active input not set[/warning] [muted](run 'autocleaneeg-pipeline input set')[/muted]"
                    )
            except Exception:
                pass

    if not args.command:
        # Show our custom 80s-style main interface instead of default help
        console = get_console(args)
        _simple_header(console)

        # Centered system info: Python, OS, Date/Time
        try:
            import platform as _platform

            from rich.align import Align
            from rich.text import Text

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
            console.print()
        except Exception:
            pass

        # Show workspace info elegantly beneath the banner (centered)
        try:
            from rich.align import Align
            from rich.text import Text

            valid_ws = workspace_dir.exists() and (workspace_dir / "tasks").exists()
            home = str(Path.home())
            display_path = str(workspace_dir)
            if display_path.startswith(home):
                display_path = display_path.replace(home, "~", 1)

            ws = Text()
            if valid_ws:
                ws.append("✓ ", style="success")
                ws.append("Workspace ", style="muted")
                ws.append(display_path, style="accent")
                console.print(Align.center(ws))
            else:
                ws.append("⚠ ", style="warning")
                ws.append("Workspace not configured — ", style="muted")
                ws.append(display_path, style="accent")
                console.print(Align.center(ws))

                tip = Text()
                tip.append("Run ", style="muted")
                tip.append("autocleaneeg-pipeline workspace", style="accent")
                tip.append(" to configure.", style="muted")
                console.print(Align.center(tip))
            # Always show active task line beneath Workspace (or guard if not set)
            try:
                active_task = user_config.get_active_task()
                at = Text()
                at.append("🎯 ", style="muted")
                at.append("Active task: ", style="muted")
                if active_task:
                    at.append(str(active_task), style="accent")
                else:
                    at.append("not set", style="warning")
                console.print(Align.center(at))
            except Exception:
                pass

            # Show active input (file vs folder) beneath task
            try:
                from rich.align import Align as _SrcAlign
                from rich.text import Text as _SrcText

                active_src = user_config.get_active_source()
                src_line = _SrcText()
                if active_src:
                    p = Path(active_src)
                    display_src = str(p)
                    if display_src.startswith(home):
                        display_src = display_src.replace(home, "~", 1)
                    if p.exists():
                        if p.is_file():
                            src_line.append("📄 ", style="muted")
                            src_line.append("Input file: ", style="muted")
                        elif p.is_dir():
                            src_line.append("📂 ", style="muted")
                            src_line.append("Input folder: ", style="muted")
                        else:
                            src_line.append("📁 ", style="muted")
                            src_line.append("Input: ", style="muted")
                        src_line.append(display_src, style="accent")
                    else:
                        src_line.append("⚠ ", style="warning")
                        src_line.append("Input missing — ", style="muted")
                        src_line.append(display_src, style="accent")
                else:
                    src_line.append("📁 ", style="muted")
                    src_line.append("Active input: ", style="muted")
                    src_line.append("not set", style="warning")
                console.print(_SrcAlign.center(src_line))
            except Exception:
                pass
        except Exception:
            # Suppress fallback to avoid left-justified output in banner
            pass

        # Disk free space for workspace volume (guarded)
        try:
            from rich.align import Align as _Align
            from rich.text import Text as _Text

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
            free_line = _Text()
            free_line.append("💾 ", style="muted")
            free_line.append("Free space ", style="muted")
            free_line.append(f"{free_gb:.1f} GB", style="accent")
            console.print(_Align.center(free_line))
        except Exception:
            pass

        # Minimal centered key commands belt (for quick discovery)
        try:
            from rich.align import Align as _KAlign
            from rich.text import Text as _KText

            key_cmds = [
                "help",
                "workspace",
                "view",
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

        # (Quick Start section intentionally removed for a cleaner minimalist banner)

        return 0

    # Validate arguments
    if not validate_args(args):
        return 1

    # Execute command
    if args.command == "process":
        if getattr(args, "process_action", None) == "ica":
            return cmd_process_ica(args)
        return cmd_process(args)
    elif args.command == "list-tasks":
        return cmd_list_tasks(args)
    elif args.command == "review":
        return cmd_review(args)
    elif args.command == "task":
        return cmd_task(args)
    elif args.command == "input":
        return cmd_input(args)
    elif args.command == "source":
        return cmd_source(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "workspace":
        return cmd_workspace(args)
    elif args.command == "export-access-log":
        return cmd_export_access_log(args)
    elif args.command == "login":
        return cmd_login(args)
    elif args.command == "logout":
        return cmd_logout(args)
    elif args.command == "whoami":
        return cmd_whoami(args)
    elif args.command == "auth0-diagnostics":
        return cmd_auth0_diagnostics(args)
    elif args.command == "auth":
        return cmd_auth(args)
    elif args.command == "clean-task":
        return cmd_clean_task(args)
    elif args.command == "view":
        return cmd_view(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "version":
        return cmd_version(args)
    elif args.command == "help":
        return cmd_help(args)
    elif args.command == "tutorial":
        return cmd_tutorial(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
