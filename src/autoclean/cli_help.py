"""Helpers for rich command-line output and help text."""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from autoclean import __version__
from autoclean.utils.user_config import user_config
from autoclean.utils.console import get_console

PRODUCT_NAME = "AutoClean EEG"
TAGLINE = "Automated EEG Processing Software"
LOGO_ICON = "🧠"
DIVIDER = "═════════════════════════════════════════════════"

def _simple_header(
    console, title: Optional[str] = None, subtitle: Optional[str] = None
):
    """Simple, consistent header for setup."""
    from rich.align import Align
    from rich.panel import Panel
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



# Rich help integration
# ------------------------------------------------------------
def _print_startup_context(console) -> None:
    """Print system info, workspace path, and free disk space (shared for header/help)."""
    try:
        from rich.text import Text
        from rich.align import Align
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

    # Workspace + disk
    try:
        from rich.text import Text as _Text
        from rich.align import Align as _Align

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

        # Disk free
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


