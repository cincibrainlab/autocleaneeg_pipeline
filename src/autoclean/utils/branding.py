"""
Branding utilities for AutoClean EEG.

Provides consistent visual identity across all CLI elements including
logos, taglines, and styling.
"""

import platform

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Optional dependencies - may not be available in all contexts
try:
    from autoclean import __version__

    VERSION_AVAILABLE = True
except ImportError:
    VERSION_AVAILABLE = False
    __version__ = "unknown"


class AutoCleanBranding:
    """Centralized branding for AutoClean EEG."""

    # Product information
    PRODUCT_NAME = "AutoClean EEG"
    TAGLINE = "A high-throughput, MNE-based EEG automation platform with integrated vision AI and 21 CFR Part 11 compliance."

    # Logo components
    LOGO_ICON = "🧠"
    WAVE_PATTERN = "∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿"

    @classmethod
    def get_ascii_logo(cls) -> str:
        """Get minimalist ASCII logo."""
        return f"""
╭─────────────────────────────────────────╮
│  {cls.LOGO_ICON} {cls.PRODUCT_NAME}                        │
│     {cls.WAVE_PATTERN}                │
╰─────────────────────────────────────────╯"""

    @classmethod
    def get_80s_ascii_logo(cls) -> str:
        """Get radical 80s-style ASCII logo."""
        return """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ╔═╗╦ ╦╔╦╗╔═╗╔═╗╦  ╔═╗╔═╗╔╗╔  ╔═╗╔═╗╔═╗                ║
║   ╠═╣║ ║ ║ ║ ║║  ║  ║╣ ╠═╣║║║  ║╣ ║╣ ║ ╦                ║
║   ╩ ╩╚═╝ ╩ ╚═╝╚═╝╩═╝╚═╝╩ ╩╝╚╝  ╚═╝╚═╝╚═╝                ║
║                                                           ║
║         ⚡ Professional EEG Processing & Analysis ⚡      ║
║               ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝"""

    @classmethod
    def get_retro_welcome(cls) -> str:
        """Get a retro-style welcome for main interface."""
        return """
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ╔═╗╦ ╦╔╦╗╔═╗╔═╗╦  ╔═╗╔═╗╔╗╔  ╔═╗╔═╗╔═╗                 │
│   ╠═╣║ ║ ║ ║ ║║  ║  ║╣ ╠═╣║║║  ║╣ ║╣ ║ ╦                 │
│   ╩ ╩╚═╝ ╩ ╚═╝╚═╝╩═╝╚═╝╩ ╩╝╚╝  ╚═╝╚═╝╚═╝                 │
│                                                             │
│         🧠 Professional EEG Processing & Analysis 🧠        │
│                   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘"""

    @classmethod
    def get_simple_welcome(cls) -> str:
        """Get a simple crystal-clear welcome."""
        return """

            ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
                  A U T O C L E A N   E E G
              ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
            ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

         🧠 Professional EEG Processing & Analysis 🧠
"""

    @classmethod
    def print_main_interface(cls, console: Console = None) -> None:
        """Print the main CLI interface with 80s style."""
        if console is None:
            console = Console()

        # Print the crystal clear logo
        console.print(cls.get_simple_welcome(), style="bright_cyan")

        # Version and platform info (concise)
        if VERSION_AVAILABLE:
            version = __version__
            platform_name = platform.system()
            arch = platform.machine()
            console.print(f"[dim]v{version} • {platform_name} {arch}[/dim]")
        else:
            console.print("[dim]AutoClean EEG[/dim]")

        # Quick start info
        console.print("\n[bold bright_green]🚀 Quick Start:[/bold bright_green]")
        console.print(
            "  [bright_yellow]autoclean-eeg setup[/bright_yellow]    - Set up your workspace"
        )
        console.print(
            "  [bright_yellow]autoclean-eeg help[/bright_yellow]     - Show all commands"
        )
        console.print(
            "  [bright_yellow]autoclean-eeg version[/bright_yellow]  - System information"
        )

        console.print(
            "\n[dim]New to AutoClean? Start with [bright_yellow]autoclean-eeg setup[/bright_yellow] to get configured![/dim]"
        )

        # Available commands (concise)
        console.print("\n[bold]Available Commands:[/bold]")
        console.print("  [cyan]process[/cyan]     Process EEG data")
        console.print("  [cyan]setup[/cyan]       Configure workspace")
        console.print("  [cyan]list-tasks[/cyan]  Show available tasks")
        console.print("  [cyan]review[/cyan]      Launch results GUI")
        console.print("  [cyan]version[/cyan]     System information")

        console.print(
            "\n[dim]Run [bright_white]autoclean-eeg <command> --help[/bright_white] for detailed usage.[/dim]"
        )

        # GitHub and support info
        console.print("\n[bold]Support & Contributing:[/bold]")
        console.print(
            "  [blue]https://github.com/cincibrainlab/autoclean_pipeline[/blue]"
        )
        console.print("  [dim]Report issues, contribute, or get help[/dim]")

    @classmethod
    def get_compact_logo(cls) -> str:
        """Get compact single-line logo."""
        return f"{cls.LOGO_ICON} {cls.PRODUCT_NAME} ∿∿∿∿∿∿∿∿∿∿∿"

    @classmethod
    def get_header_text(cls, include_tagline: bool = True) -> Text:
        """Get formatted header text for rich panels."""
        text = Text()
        text.append(f"{cls.LOGO_ICON} {cls.PRODUCT_NAME}", style="bold green")
        if include_tagline:
            text.append(f"\n   {cls.TAGLINE}", style="dim green")
        return text

    @classmethod
    def get_welcome_panel(cls, console: Console = None) -> Panel:
        """Get welcome panel with full branding."""
        if console is None:
            console = Console()

        header_text = cls.get_header_text(include_tagline=True)
        return Panel(
            header_text,
            style="green",
            padding=(1, 2),
            title="Welcome",
            title_align="left",
        )

    @classmethod
    def get_status_panel(
        cls, title: str, style: str = "blue", show_icon: bool = False
    ) -> Panel:
        """Get status panel with consistent branding."""
        if show_icon:
            header_text = f"{cls.LOGO_ICON} [bold]{title}[/bold]"
        else:
            header_text = f"[bold]{title}[/bold]"
        return Panel(header_text, style=style, padding=(0, 2))

    @classmethod
    def get_professional_header(cls, console: Console = None) -> None:
        """Get professional header for existing workspace scenarios."""
        if console is None:
            console = Console()

        # Main product header - more prominent
        console.print(f"\n[bold green]{cls.LOGO_ICON} {cls.PRODUCT_NAME}[/bold green]")
        console.print(f"[dim green]{cls.WAVE_PATTERN}[/dim green]")
        console.print("[dim]Professional EEG Processing & Analysis Platform[/dim]")

    @classmethod
    def get_simple_divider(cls) -> str:
        """Get a simple divider for clean separation."""
        return f"[dim]{cls.WAVE_PATTERN * 2}[/dim]"

    @classmethod
    def print_logo(cls, console: Console = None, style: str = "ascii") -> None:
        """Print logo to console."""
        if console is None:
            console = Console()

        if style == "ascii":
            console.print(cls.get_ascii_logo(), style="green")
        elif style == "compact":
            console.print(cls.get_compact_logo(), style="bold green")
        elif style == "panel":
            console.print(cls.get_welcome_panel(console))

    @classmethod
    def print_tagline(cls, console: Console = None) -> None:
        """Print tagline separately."""
        if console is None:
            console = Console()
        console.print(f"[dim]{cls.TAGLINE}[/dim]")

    @classmethod
    def print_tutorial_header(cls, console: Console = None) -> None:
        """Print a header for the tutorial."""
        if console is None:
            console = Console()

        # Print the crystal clear logo
        console.print(cls.get_simple_welcome(), style="bright_cyan")

        # Version and platform info (concise)
        if VERSION_AVAILABLE:
            version = __version__
            platform_name = platform.system()
            arch = platform.machine()
            console.print(f"[dim]v{version} • {platform_name} {arch}[/dim]")
        else:
            console.print("[dim]AutoClean EEG[/dim]")


# Convenience functions for easy import
def get_logo(style: str = "compact") -> str:
    """Get logo in specified style."""
    if style == "ascii":
        return AutoCleanBranding.get_ascii_logo()
    elif style == "compact":
        return AutoCleanBranding.get_compact_logo()
    else:
        return AutoCleanBranding.get_compact_logo()


def get_welcome_panel() -> Panel:
    """Get welcome panel with branding."""
    return AutoCleanBranding.get_welcome_panel()


def get_product_info() -> tuple[str, str]:
    """Get product name and tagline."""
    return AutoCleanBranding.PRODUCT_NAME, AutoCleanBranding.TAGLINE
