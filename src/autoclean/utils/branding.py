"""
Branding utilities for AutoClean EEG.

Provides consistent visual identity across all CLI elements including
logos, taglines, and styling.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


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
            title_align="left"
        )
    
    @classmethod
    def get_status_panel(cls, title: str, style: str = "blue", show_icon: bool = False) -> Panel:
        """Get status panel with consistent branding."""
        if show_icon:
            header_text = f"{cls.LOGO_ICON} [bold]{title}[/bold]"
        else:
            header_text = f"[bold]{title}[/bold]"
        return Panel(
            header_text,
            style=style,
            padding=(0, 2)
        )
    
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