#!/usr/bin/env python3
"""Block management CLI commands."""

import json
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autoclean.utils.block_registry import BlockRegistry
from autoclean.utils.console import get_console


def get_blocks_dir() -> Path:
    """Get the bundled blocks directory."""
    return Path(__file__).parent / "blocks"


def load_registry() -> dict:
    """Load the block registry (legacy function for backwards compatibility)."""
    registry_path = get_blocks_dir() / "registry.json"
    if not registry_path.exists():
        return {"blocks": [], "total_blocks": 0}

    with open(registry_path) as f:
        return json.load(f)


def load_block_manifest(category: str, block_name: str) -> Optional[dict]:
    """Load manifest for a specific block (from bundled or cache)."""
    # Try cache first
    cache_path = Path.home() / ".config" / "autocleaneeg" / ".block_cache" / category / block_name / "manifest.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback to bundled
    manifest_path = get_blocks_dir() / category / block_name / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        return json.load(f)


def cmd_blocks_list(args) -> int:
    """List all available blocks (from cache/bundled)."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    registry = BlockRegistry()
    blocks = registry.list_blocks()

    if not blocks:
        console.print("[warning]No blocks found[/warning]")
        return 0

    # Group by category
    categories = {}
    for block in blocks:
        cat = block.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(block)

    # Get registry status
    registry_info = registry.registry_status()
    commit_raw = registry_info.get("commit")
    commit = commit_raw or "not yet synced"
    if commit in {"unknown", ""}:
        commit = "not yet synced"

    console.print(f"\n[success]Processing Blocks ({len(blocks)} total)[/success]")
    console.print(f"[dim]Registry version: {commit}[/dim]\n")

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
        table.add_column("Source", style="yellow", no_wrap=True)
        table.add_column("Description", style="white")

        for block in sorted(categories[category], key=lambda b: b.name):
            source = registry.block_source(block.name)
            # Get sync status
            status_info = registry.block_sync_status(block.name)
            status = status_info.get("status", "unknown")

            # Format source display
            if status == "synced":
                source_display = f"{source}"
            elif status == "outdated":
                source_display = f"{source} (outdated)"
            elif status == "cache_only":
                source_display = "cache"
            else:
                source_display = source

            table.add_row(
                block.name,
                block.version,
                source_display,
                block.description[:80],
            )

        console.print(table)
        console.print()

    # Show last update summary if available
    update_summary = registry.last_update_summary()
    if any(update_summary.values()):
        console.print("[dim]Last update:[/dim]")
        if update_summary["new"]:
            console.print(f"  [success]New:[/success] {', '.join(update_summary['new'])}")
        if update_summary["updated"]:
            console.print(f"  [accent]Updated:[/accent] {', '.join(update_summary['updated'])}")
        if update_summary["removed"]:
            console.print(f"  [warning]Removed:[/warning] {', '.join(update_summary['removed'])}")
        console.print()

    return 0


def cmd_blocks_info(args) -> int:
    """Show detailed info for a specific block."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    block_name = args.block_name

    # Find block in registry
    registry = BlockRegistry()
    block = registry.get_block(block_name)

    if not block:
        console.print(f"[error]Block not found: {block_name}[/error]")
        console.print("[dim]Run 'blocks list' to see available blocks[/dim]")
        return 1

    # Get sync status
    status_info = registry.block_sync_status(block.name)
    status = status_info.get("status", "unknown")
    source = status_info.get("source", "unknown")

    # Load full manifest
    manifest = load_block_manifest(block.category, block_name)

    if not manifest:
        console.print(f"[error]Manifest not found for: {block_name}[/error]")
        return 1

    # Display block info with sync status
    console.print()
    title_text = Text()
    title_text.append("Block Information", style="bold")

    console.print(
        Panel(
            f"[accent bold]{manifest.get('name', 'Unknown')}[/accent bold] [dim]v{manifest.get('version', '?')}[/dim]",
            title=title_text,
            border_style="cyan",
        )
    )

    console.print(f"\n[bold]Description:[/bold] {manifest.get('description', 'N/A')}")
    console.print(f"[bold]Category:[/bold] {manifest.get('category', 'N/A')}")
    console.print(f"[bold]Source:[/bold] {source}")
    console.print(f"[bold]Status:[/bold] {status}")
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
    """Update blocks from task-registry GitHub."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    console.print("→ Updating blocks from GitHub registry...\n")

    registry = BlockRegistry()
    allow_network = not getattr(args, "no_network", False)
    msg = registry.update_cache(allow_network=allow_network)

    console.print(msg)

    # Show update summary if available
    update_summary = registry.last_update_summary()
    if any(update_summary.values()):
        console.print()
        if update_summary["new"]:
            console.print("[success]New blocks:[/success]")
            for name in update_summary["new"]:
                console.print(f"  + {name}")

        if update_summary["updated"]:
            console.print("[accent]Updated blocks:[/accent]")
            for name in update_summary["updated"]:
                console.print(f"  ↻ {name}")

        if update_summary["removed"]:
            console.print("[warning]Removed blocks:[/warning]")
            for name in update_summary["removed"]:
                console.print(f"  - {name}")

    console.print()
    return 0


def cmd_blocks_install(args) -> int:
    """Install/download a specific block to cache."""
    console = get_console(args.theme if hasattr(args, "theme") else None)

    block_name = args.block_name
    commit_hash = getattr(args, "commit", None)
    registry = BlockRegistry()

    # Check if block exists
    block = registry.get_block(block_name)
    if not block:
        console.print(f"[error]Block not found: {block_name}[/error]")
        console.print("[dim]Run 'blocks list' to see available blocks[/dim]")
        return 1

    if commit_hash:
        console.print(f"→ Installing block [accent]{block_name}[/accent] from commit [yellow]{commit_hash[:8]}[/yellow]...")
    else:
        console.print(f"→ Installing block [accent]{block_name}[/accent]...")

    try:
        dest_path = registry.materialize_block_to(block_name, registry.cache_root, commit=commit_hash)
        console.print(f"[success]✓[/success] Block installed to cache: {dest_path}")
        if commit_hash:
            console.print(f"[dim]Installed from commit: {commit_hash}[/dim]")
        console.print()
        console.print("[dim]The block will be automatically discovered on next pipeline run.[/dim]")
        console.print()
        return 0
    except Exception as exc:
        console.print(f"[error]Failed to install block: {exc}[/error]")
        return 1