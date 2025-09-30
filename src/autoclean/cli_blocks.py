#!/usr/bin/env python3
"""Block management CLI commands."""

import json
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from autoclean.utils.console import get_console


def get_blocks_dir() -> Path:
    """Get the bundled blocks directory."""
    return Path(__file__).parent / "blocks"


def load_registry() -> dict:
    """Load the block registry."""
    registry_path = get_blocks_dir() / "registry.json"
    if not registry_path.exists():
        return {"blocks": [], "total_blocks": 0}

    with open(registry_path) as f:
        return json.load(f)


def load_block_manifest(category: str, block_name: str) -> Optional[dict]:
    """Load manifest for a specific block."""
    manifest_path = get_blocks_dir() / category / block_name / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        return json.load(f)


def cmd_blocks_list(args) -> int:
    """List all bundled blocks."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    registry = load_registry()
    blocks = registry.get("blocks", [])

    if not blocks:
        console.print("[warning]No blocks found in bundle[/warning]")
        return 0

    # Group by category
    categories = {}
    for block in blocks:
        cat = block.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(block)

    console.print(f"\n[success]Bundled Blocks ({len(blocks)} total)[/success]\n")

    for category in sorted(categories.keys()):
        # Create table for this category
        table = Table(
            title=f"[accent]{category.replace('_', ' ').title()}[/accent]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )

        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Version", style="green")
        table.add_column("Description", style="white")

        for block in sorted(categories[category], key=lambda b: b.get("name", "")):
            table.add_row(
                block.get("name", "unknown"),
                block.get("version", "?"),
                block.get("description", "")[:80],
            )

        console.print(table)
        console.print()

    # Show registry metadata
    console.print(
        f"[dim]Registry generated: {registry.get('generated_at', 'unknown')}[/dim]"
    )
    console.print(
        f"[dim]Source: {registry.get('source', 'unknown')}[/dim]"
    )
    console.print()

    return 0


def cmd_blocks_info(args) -> int:
    """Show detailed info for a specific block."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    block_name = args.block_name

    # Find block in registry
    registry = load_registry()
    blocks = registry.get("blocks", [])
    block_entry = next((b for b in blocks if b.get("name") == block_name), None)

    if not block_entry:
        console.print(f"[error]Block not found: {block_name}[/error]")
        console.print("[dim]Run 'blocks list' to see available blocks[/dim]")
        return 1

    # Load full manifest
    category = block_entry.get("category", "unknown")
    manifest = load_block_manifest(category, block_name)

    if not manifest:
        console.print(f"[error]Manifest not found for: {block_name}[/error]")
        return 1

    # Display block info
    console.print()
    console.print(
        Panel(
            f"[accent bold]{manifest.get('name', 'Unknown')}[/accent bold] [dim]v{manifest.get('version', '?')}[/dim]",
            title="Block Information",
            border_style="cyan",
        )
    )

    console.print(f"\n[bold]Description:[/bold] {manifest.get('description', 'N/A')}")
    console.print(f"[bold]Category:[/bold] {manifest.get('category', 'N/A')}")
    console.print(f"[bold]Author:[/bold] {manifest.get('author', 'N/A')}")
    console.print(f"[bold]License:[/bold] {manifest.get('license', 'N/A')}")

    # API info
    if "api" in manifest and manifest["api"]:
        api = manifest["api"]
        console.print("\n[bold]API:[/bold]")
        console.print(f"  Mixin Class: [cyan]{api.get('mixin_class', 'N/A')}[/cyan]")
        console.print(f"  Method: [cyan]{api.get('mixin_method', 'N/A')}[/cyan]")
        console.print(f"  Config Key: [yellow]{api.get('config_key', 'N/A')}[/yellow]")

    # Dependencies
    if "dependencies" in manifest:
        deps = manifest["dependencies"]
        if "packages" in deps and deps["packages"]:
            console.print("\n[bold]Required Packages:[/bold]")
            for pkg, version in deps["packages"].items():
                console.print(f"  {pkg} {version}")

    # Tags
    if "tags" in manifest and manifest["tags"]:
        console.print(f"\n[bold]Tags:[/bold] {', '.join(manifest['tags'])}")

    # References
    if "references" in manifest and manifest["references"]:
        console.print("\n[bold]References:[/bold]")
        for ref in manifest["references"]:
            console.print(f"  - {ref.get('name', 'Unknown')}")
            if "citation" in ref:
                console.print(f"    [dim]{ref['citation']}[/dim]")
            if "doi" in ref:
                console.print(f"    [link]https://doi.org/{ref['doi']}[/link]")

    # Registry links
    if "registry" in manifest:
        reg = manifest["registry"]
        console.print("\n[bold]Links:[/bold]")
        if "documentation_url" in reg:
            console.print(f"  Docs: [link]{reg['documentation_url']}[/link]")
        if "source_url" in reg:
            console.print(f"  Source: [link]{reg['source_url']}[/link]")

    console.print()
    return 0


def cmd_blocks_update(args) -> int:
    """Update blocks from task-registry (placeholder for now)."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    console.print("[warning]Block update functionality coming in v2.5.0[/warning]")
    console.print()
    console.print("[dim]Planned functionality:[/dim]")
    console.print("  - Fetch latest blocks from task-registry")
    console.print("  - Compare versions and show updates available")
    console.print("  - Allow selective block updates")
    console.print()
    console.print(
        "[info]For now, blocks are bundled with pipeline releases.[/info]"
    )
    console.print()

    return 0