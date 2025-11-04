"""User-friendly error handling for processing block dependency issues.

This module provides graceful error messages for non-technical users when
processing blocks have missing dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from autoclean.utils.block_dependencies import (
    generate_install_command,
    is_uv_tool_install,
)


class BlockDependencyError(Exception):
    """Raised when a processing block has missing dependencies.

    This exception is caught at the pipeline level and formatted into
    a user-friendly message with clear installation instructions.
    """

    def __init__(
        self,
        block_name: str,
        missing_packages: List[Tuple[str, str]],
        what_it_does: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        block_name : str
            Name of the block with missing dependencies
        missing_packages : list of tuple
            List of (package_name, version_spec) tuples
        what_it_does : str, optional
            Plain-language explanation of what the block does
        """
        self.block_name = block_name
        self.missing_packages = missing_packages
        self.what_it_does = what_it_does
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format brief message for exception string."""
        pkg_names = ", ".join(pkg for pkg, _ in self.missing_packages)
        return f"Block '{self.block_name}' requires: {pkg_names}"

    def print_detailed_help(self, console: Optional[Console] = None) -> None:
        """Print a detailed, user-friendly help message.

        Parameters
        ----------
        console : Console, optional
            Rich console to print to. If None, creates a new one.
        """
        if console is None:
            console = Console()

        # Header
        console.print()
        header = Text("🚫 Missing Dependencies", style="bold red")
        console.print(Panel(header, border_style="red"))
        console.print()

        # What happened? (Simple explanation)
        console.print("[bold]What happened?[/bold]")
        console.print(
            f"  The '[accent]{self.block_name}[/accent]' processing block needs "
            "additional software that"
        )
        console.print("  isn't currently installed on your system.")
        console.print()

        # What does this block do? (If provided)
        if self.what_it_does:
            console.print(f"[bold]What does {self.block_name} do?[/bold]")
            console.print(f"  {self.what_it_does}")
            console.print()

        # What you need (List packages)
        console.print("[bold]What you need:[/bold]")
        for pkg, version in self.missing_packages:
            console.print(f"  • [accent]{pkg}[/accent] (version {version})")
            # Add friendly description for known packages
            desc = _get_package_description(pkg)
            if desc:
                console.print(f"    [dim]{desc}[/dim]")
        console.print()

        # How to fix this
        console.print("[bold yellow]How to fix this:[/bold yellow]")
        console.print()

        if is_uv_tool_install():
            console.print(
                "  [dim]Since you installed with 'uv tool install', run:[/dim]"
            )
            console.print()

            # Option 1: Individual packages
            cmd = generate_install_command(self.missing_packages)
            console.print(f"    [accent]{cmd}[/accent]")
            console.print()
            console.print(
                "  [dim]This will take about 30 seconds and add the required software.[/dim]"
            )
            console.print()

            # Option 2: All blocks
            console.print("  [dim]Or install all optional blocks at once:[/dim]")
            console.print()
            console.print(
                '    [accent]uv tool install "autocleaneeg-pipeline[blocks-all]"[/accent]'
            )
        else:
            console.print("  [dim]Install with pip:[/dim]")
            console.print()
            for pkg, version in self.missing_packages:
                console.print(f"    [accent]pip install '{pkg}{version}'[/accent]")

        console.print()

        # Need help?
        console.print("[bold]Need help?[/bold]")
        console.print(
            "  📘 Docs: [link]https://docs.autocleaneeg.org/blocks/dependencies[/link]"
        )
        console.print(
            "  💬 Get support: [link]https://github.com/cincibrainlab/autoclean_pipeline/discussions[/link]"
        )
        console.print()


def _get_package_description(package_name: str) -> Optional[str]:
    """Get friendly description for known packages."""
    descriptions = {
        "meegkit": "A specialized toolbox for cleaning EEG/MEG data",
        "fooof": "Tools for analyzing brain oscillations",
        "specparam": "Spectral parameterization for brain signals",
        "autoreject": "Automated artifact rejection for EEG",
    }
    return descriptions.get(package_name)


def raise_dependency_error(
    block_name: str,
    missing_packages: List[Tuple[str, str]],
    what_it_does: Optional[str] = None,
) -> None:
    """Raise a BlockDependencyError with user-friendly formatting.

    This function should be called from within blocks when imports fail.

    Parameters
    ----------
    block_name : str
        Name of the block
    missing_packages : list of tuple
        List of (package_name, version_spec) tuples
    what_it_does : str, optional
        Plain-language explanation of what the block does

    Raises
    ------
    BlockDependencyError
        Always raised with formatted message

    Examples
    --------
    >>> try:
    ...     from meegkit import dss
    ... except ImportError:
    ...     raise_dependency_error(
    ...         "zapline",
    ...         [("meegkit", ">=0.1.9")],
    ...         "remove power line noise using advanced signal processing"
    ...     )
    """
    raise BlockDependencyError(block_name, missing_packages, what_it_does)


def format_dependency_missing_message(
    block_name: str,
    missing_packages: List[Tuple[str, str]],
    what_it_does: Optional[str] = None,
) -> str:
    """Format a dependency error as a plain text message.

    For cases where Rich formatting isn't available.

    Parameters
    ----------
    block_name : str
        Name of the block
    missing_packages : list of tuple
        List of (package_name, version_spec) tuples
    what_it_does : str, optional
        Plain-language explanation

    Returns
    -------
    str
        Formatted plain text error message
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Missing Dependencies")
    lines.append("=" * 60)
    lines.append("")

    lines.append("What happened?")
    lines.append(f"  The '{block_name}' processing block needs additional software")
    lines.append("  that isn't currently installed.")
    lines.append("")

    if what_it_does:
        lines.append(f"What does {block_name} do?")
        lines.append(f"  {what_it_does}")
        lines.append("")

    lines.append("What you need:")
    for pkg, version in missing_packages:
        lines.append(f"  • {pkg} (version {version})")
    lines.append("")

    lines.append("How to fix this:")
    if is_uv_tool_install():
        cmd = generate_install_command(missing_packages)
        lines.append(f"  {cmd}")
    else:
        for pkg, version in missing_packages:
            lines.append(f"  pip install '{pkg}{version}'")
    lines.append("")

    lines.append("Need help?")
    lines.append("  Docs: https://docs.autocleaneeg.org/blocks/dependencies")
    lines.append(
        "  Support: https://github.com/cincibrainlab/autoclean_pipeline/discussions"
    )
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
