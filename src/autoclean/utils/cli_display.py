"""
Unified CLI display system for AutoClean EEG.

Provides consistent, professional console output throughout the CLI interface
without relying on the logging system. Uses Rich exclusively for clean,
visually appealing user interactions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, TaskID
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Panel = None
    Text = None
    Table = None
    Progress = None
    TaskID = None
    Confirm = None
    Prompt = None
    Align = None


class CLIDisplay:
    """Unified CLI display system using Rich for professional console output."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize CLI display system."""
        if not RICH_AVAILABLE:
            raise ImportError("Rich library is required for CLI display system")

        self.console = console or Console()

        # Status indicators
        self.SUCCESS = "[green]✓[/green]"
        self.WARNING = "[yellow]⚠[/yellow]"
        self.ERROR = "[red]❌[/red]"
        self.INFO = "[blue]ℹ[/blue]"
        self.WORKING = "[blue]🔧[/blue]"
        self.ARROW = "[dim]→[/dim]"

        # Spacing constants
        self.SECTION_SPACING = "\n"
        self.ITEM_SPACING = ""

    def header(
        self, title: str, subtitle: Optional[str] = None, style: str = "bold blue"
    ) -> None:
        """Display a section header."""
        text = Text()
        text.append(title, style=style)
        if subtitle:
            text.append(f"\n{subtitle}", style="dim")

        self.console.print(text)

    def success(self, message: str, details: Optional[str] = None) -> None:
        """Display a success message."""
        self.console.print(f"{self.SUCCESS} [bold green]{message}[/bold green]")
        if details:
            self.console.print(f"  [dim cyan]{details}[/dim cyan]")

    def warning(self, message: str, details: Optional[str] = None) -> None:
        """Display a warning message."""
        self.console.print(f"{self.WARNING} [bold yellow]{message}[/bold yellow]")
        if details:
            self.console.print(f"  [dim yellow]{details}[/dim yellow]")

    def error(self, message: str, details: Optional[str] = None) -> None:
        """Display an error message."""
        self.console.print(f"{self.ERROR} [bold red]{message}[/bold red]")
        if details:
            self.console.print(f"  [dim red]{details}[/dim red]")

    def info(self, message: str, details: Optional[str] = None) -> None:
        """Display an info message."""
        self.console.print(f"{self.INFO} [bold blue]{message}[/bold blue]")
        if details:
            self.console.print(f"  [dim blue]{details}[/dim blue]")

    def working(self, message: str) -> None:
        """Display a working/in-progress message."""
        self.console.print(f"{self.WORKING} [bold cyan]{message}[/bold cyan]")

    def step(self, message: str, status: str = "pending") -> None:
        """Display a step in a process."""
        if status == "completed":
            icon = self.SUCCESS
        elif status == "error":
            icon = self.ERROR
        elif status == "working":
            icon = self.WORKING
        else:
            icon = self.ARROW

        self.console.print(f"{icon} {message}")

    def list_item(
        self, message: str, value: Optional[str] = None, indent: int = 2
    ) -> None:
        """Display a list item with optional value."""
        spaces = " " * indent
        if value:
            self.console.print(f"{spaces}[bold]{message}:[/bold] {value}")
        else:
            self.console.print(f"{spaces}• {message}")

    def key_value(
        self, key: str, value: str, key_style: str = "bold", value_style: str = "dim"
    ) -> None:
        """Display a key-value pair."""
        self.console.print(
            f"[{key_style}]{key}:[/{key_style}] [{value_style}]{value}[/{value_style}]"
        )

    def separator(self, char: str = "─", length: int = 50, style: str = "dim") -> None:
        """Display a visual separator."""
        self.console.print(f"[{style}]{char * length}[/{style}]")

    def blank_line(self, count: int = 1) -> None:
        """Add blank lines for spacing."""
        for _ in range(count):
            self.console.print()

    def panel(
        self,
        content: Union[str, Text],
        title: Optional[str] = None,
        style: str = "blue",
        padding: tuple = (0, 2),
    ) -> None:
        """Display content in a panel."""
        panel = Panel(
            content,
            title=title,
            style=style,
            padding=padding,
            title_align="left" if title else "center",
        )
        self.console.print(panel)

    def table(
        self, headers: List[str], rows: List[List[str]], title: Optional[str] = None
    ) -> None:
        """Display a table."""
        table = Table(title=title, show_header=True, header_style="bold blue")

        for header in headers:
            table.add_column(header)

        for row in rows:
            table.add_row(*row)

        self.console.print(table)

    def prompt_yes_no(self, question: str, default: bool = False) -> bool:
        """Prompt for yes/no confirmation."""
        return Confirm.ask(question, default=default, console=self.console)

    def prompt_text(
        self, question: str, default: Optional[str] = None, show_default: bool = True
    ) -> str:
        """Prompt for text input."""
        return Prompt.ask(
            question, default=default, show_default=show_default, console=self.console
        )

    def prompt_choice(
        self, question: str, choices: List[str], default: Optional[str] = None
    ) -> str:
        """Prompt for choice from a list."""
        return Prompt.ask(
            question, choices=choices, default=default, console=self.console
        )

    def workspace_info(self, workspace_path: Path, is_valid: bool = True) -> None:
        """Display workspace information in a clean format."""
        status_icon = self.SUCCESS if is_valid else self.WARNING
        status_text = (
            "[bold green]properly configured[/bold green]"
            if is_valid
            else "[bold yellow]needs setup[/bold yellow]"
        )

        self.console.print(f"{status_icon} Workspace is {status_text}")
        self.console.print()
        self.console.print(
            f"[bold cyan]Location:[/bold cyan] [bright_blue]{workspace_path}[/bright_blue]"
        )
        self.console.print()

    def system_info_table(self, info: Dict[str, str]) -> None:
        """Display system information in a clean table format."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bright_white")

        for key, value in info.items():
            # Add colors for specific values
            if "GPU" in key and (
                "✓" in value or "Apple" in value or "NVIDIA" in value or "CUDA" in value
            ):
                formatted_value = f"[bold green]{value}[/bold green]"
            elif "None detected" in value:
                formatted_value = f"[dim]{value}[/dim]"
            else:
                formatted_value = f"[bright_white]{value}[/bright_white]"

            table.add_row(f"{key}:", formatted_value)

        self.console.print(table)

    def setup_complete(
        self, workspace_path: Path, additional_info: Optional[List[str]] = None
    ) -> None:
        """Display setup completion message."""
        self.blank_line()
        self.success("Setup complete!", f"[bright_blue]{workspace_path}[/bright_blue]")

        if additional_info:
            self.blank_line()
            for info in additional_info:
                self.console.print(
                    f"  [green]•[/green] [bright_white]{info}[/bright_white]"
                )
        self.blank_line()

    def migration_prompt(self, old_path: Path, new_path: Path) -> bool:
        """Prompt for workspace migration."""
        self.console.print("\n[bold cyan]Workspace Migration[/bold cyan]")
        self.console.print(
            f"[yellow]From:[/yellow] [dim bright_white]{old_path}[/dim bright_white]"
        )
        self.console.print(
            f"[green]To:[/green]   [bright_blue]{new_path}[/bright_blue]"
        )
        self.console.print()

        return self.prompt_yes_no(
            "[bold]Migrate existing tasks and configuration?[/bold]", default=False
        )

    def centered(self, content: Union[str, Text], style: Optional[str] = None) -> None:
        """Display centered content."""
        if isinstance(content, str) and style:
            content = Text(content, style=style)
        self.console.print(Align.center(content))

    def boxed_header(
        self,
        main_text: str,
        subtitle: Optional[str] = None,
        title: Optional[str] = None,
        main_style: str = "bold bright_cyan",
        subtitle_style: str = "bright_blue",
        box_style: str = "cyan",
    ) -> None:
        """Create a professional boxed header."""
        content = Text()
        content.append(main_text, style=main_style)
        if subtitle:
            content.append(f"\n{subtitle}", style=subtitle_style)

        panel = Panel(
            Align.center(content),
            style=box_style,
            padding=(0, 1),
            title=title,
            title_align="center" if title else None,
        )

        self.console.print(panel)


class SetupDisplay(CLIDisplay):
    """Specialized display methods for setup wizard."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize setup display system."""
        super().__init__(console)

    def welcome_header(self, is_first_time: bool = True) -> None:
        """Display welcome header for setup."""
        # Simple branding constants
        PRODUCT_NAME = "AutoClean EEG"
        TAGLINE = "Professional EEG Processing & Analysis Platform"
        LOGO_ICON = "🧠"

        self.blank_line()

        if is_first_time:
            # Create branding content
            branding_text = Text()
            branding_text.append(
                f"{LOGO_ICON} Welcome to AutoClean", style="bold bright_cyan"
            )
            branding_text.append(f"\n{TAGLINE}", style="bright_blue")

            # Create panel with branding
            branding_panel = Panel(
                Align.center(branding_text),
                style="cyan",
                padding=(0, 1),
                title_align="center",
            )

            self.console.print(branding_panel)
            self.blank_line()
            self.console.print(
                "[bright_white]Let's set up your workspace for EEG processing.[/bright_white]"
            )
            self.console.print(
                "[dim cyan]This workspace will contain your custom tasks, configuration, and results.[/dim cyan]"
            )
        else:
            # Reconfiguration - simple section title, no boxed header
            self.console.print("[bold yellow]⚙️ Workspace Reconfiguration[/bold yellow]")
            self.console.print("[dim]Setting up new workspace location[/dim]")

        self.blank_line()

    def setup_progress(self, step: str, details: Optional[str] = None) -> None:
        """Display setup progress."""
        self.working(step)
        if details:
            self.console.print(f"  {details}", style="dim")

    def workspace_location_prompt(self, default_dir: Path) -> Path:
        """Prompt for workspace location with clean formatting."""
        self.console.print("[bold cyan]Workspace Location[/bold cyan]")
        self.console.print(
            f"[dim bright_white]Default: [/dim bright_white][bright_blue]{default_dir}[/bright_blue]"
        )
        self.console.print(
            "[dim green]• Custom tasks  • Configuration  • Results  • Easy backup[/dim green]"
        )
        self.blank_line()

        try:
            response = self.prompt_text(
                "[bold]Press Enter for default, or enter custom path[/bold]",
                default="",
                show_default=False,
            ).strip()

            if response:
                chosen_dir = Path(response).expanduser()
                self.success("Using custom location", str(chosen_dir))
            else:
                chosen_dir = default_dir
                self.success("Using default location", str(chosen_dir))

            self.blank_line()
            return chosen_dir

        except (EOFError, KeyboardInterrupt):
            self.warning("Using default location due to interrupt")
            self.blank_line()
            return default_dir

    def compliance_status_display(self, is_enabled: bool, is_permanent: bool) -> None:
        """Display compliance status information."""
        if is_permanent:
            self.warning("FDA 21 CFR Part 11 compliance mode is permanently enabled")
            self.info("You can only configure workspace location in compliance mode")
        elif is_enabled:
            self.info("FDA 21 CFR Part 11 compliance mode is currently enabled")
        else:
            self.console.print(
                "[dim cyan]Current compliance mode: [/dim cyan][dim yellow]disabled[/dim yellow]"
            )
        self.blank_line()

    def setup_complete_summary(self, workspace_path: Path) -> None:
        """Display setup completion summary."""
        self.blank_line()
        self.separator("═", 50, "green")
        self.success("Setup complete!", str(workspace_path))

        # Additional files created
        created_files = [
            "📋 Template task created",
            "📄 Example script added",
            "🔧 Built-in task examples copied",
        ]

        self.blank_line()
        self.console.print("[bold bright_green]Files Created:[/bold bright_green]")
        for file_info in created_files:
            self.console.print(
                f"  [green]•[/green] [bright_white]{file_info}[/bright_white]"
            )

        self.blank_line()
        self.separator("═", 50, "green")


# Global instances for easy import
cli_display = CLIDisplay()
setup_display = SetupDisplay()
